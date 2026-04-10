import base64
import binascii
import json
import math
import os
import re
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .docling import DOCLING_EXTENSIONS, convert_with_docling, normalize_whitespace
except ImportError:
    from docling import DOCLING_EXTENSIONS, convert_with_docling, normalize_whitespace

try:
    from markdowncleaner import CleanerOptions, MarkdownCleaner
except ImportError:
    CleanerOptions = None
    MarkdownCleaner = None

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "microsoft/harrier-oss-v1-270m").strip() or "microsoft/harrier-oss-v1-270m"
EMBEDDING_TASK_PREFIX = os.getenv("EMBEDDING_TASK_PREFIX", "").strip()
EMBEDDING_PROMPT_NAME = os.getenv("EMBEDDING_PROMPT_NAME", "sts_query").strip()
EMBEDDING_PROMPT = os.getenv("EMBEDDING_PROMPT", "").strip()
CHUNK_TARGET_CHARS = int(os.getenv("SUMMARY_CHUNK_TARGET_CHARS", "1400"))
SUBMODULAR_TOP_K = int(os.getenv("SUMMARY_SUBMODULAR_TOP_K", os.getenv("SUMMARY_MMR_TOP_K", "5")))
SUBMODULAR_COVERAGE_WEIGHT = float(os.getenv("SUMMARY_SUBMODULAR_COVERAGE_WEIGHT", "1.0"))
SUBMODULAR_RELEVANCE_WEIGHT = float(os.getenv("SUMMARY_SUBMODULAR_RELEVANCE_WEIGHT", "0.35"))
PIPELINE_OUTPUT_DIR = Path(os.getenv("SUMMARY_OUTPUT_DIR", str(Path(__file__).resolve().parent / "outputs"))).resolve()

_embedding_model = None
_embedding_lock = threading.Lock()
_markdown_cleaner = None
_markdown_cleaner_lock = threading.Lock()


class SummaryPipelineError(RuntimeError):
    pass


@dataclass
class SourceDocument:
    name: str
    path: str
    text: str


@dataclass
class ChunkRecord:
    index: int
    source_name: str
    headers: dict[str, str]
    header_path: str
    text: str
    preview: str


@dataclass
class ClusterRecord:
    cluster_id: str
    label: int
    size: int
    title: str
    chunk_indices: list[int]
    representative_indices: list[int]
    representative_previews: list[str]
    summary: str = ""


def normalize_latex(text: str) -> str:
    """Fix non-standard LaTeX macros produced by Docling conversion.

    Common issues:
    - \\text{ \\texttimes } â†’ \\times  (multiplication sign)
    - \\overbar â†’ \\overline          (overline accent)
    - R^{...} â†’ \\mathbb{R}^{...}    (real number set notation)
    - &amp; â†’ &                        (HTML entity leftover)
    """
    if not text:
        return text

    # Fix \text{ \texttimes } â†’ \times (with varying whitespace)
    text = re.sub(r"\\text\{\s*\\texttimes\s*\}", r"\\times", text)

    # Fix \overbar â†’ \overline (non-standard macro)
    text = text.replace("\\overbar", "\\overline")

    # Fix HTML entities that survive conversion
    text = text.replace("&amp;", "&")

    return text


def sanitize_pipeline_text(text: str) -> str:
    """Remove extraction artifacts that skew embeddings and summaries."""
    if not text:
        return ""

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        if line.startswith("<!--") and line.endswith("-->"):
            continue
        if line in {"()", "( )"}:
            continue
        cleaned_lines.append(raw_line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = normalize_latex(cleaned)
    return normalize_whitespace(cleaned)


def get_markdown_cleaner():
    global _markdown_cleaner
    if MarkdownCleaner is None or CleanerOptions is None:
        return None
    if _markdown_cleaner is not None:
        return _markdown_cleaner

    with _markdown_cleaner_lock:
        if _markdown_cleaner is not None:
            return _markdown_cleaner
        _markdown_cleaner = MarkdownCleaner(
            options=CleanerOptions(
                fix_encoding_mojibake=True,
                normalize_quotation_symbols=True,
                remove_short_lines=False,
                remove_whole_lines=True,
                remove_sections=True,
                remove_duplicate_headlines=True,
                remove_footnotes_in_text=True,
                replace_within_lines=True,
                remove_within_lines=True,
                contract_empty_lines=True,
                crimp_linebreaks=True,
                remove_references_heuristically=True,
            )
        )
        return _markdown_cleaner


def clean_extracted_markdown(text: str) -> str:
    cleaner = get_markdown_cleaner()
    if cleaner is None or not text:
        return text
    try:
        return cleaner.clean_markdown_string(text)
    except Exception:
        return text


def slugify_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "output"


def build_run_output_dir(documents: list[SourceDocument]) -> Path:
    PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = slugify_filename(documents[0].name if documents else "summary")
    run_dir = PIPELINE_OUTPUT_DIR / f"{stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def json_dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def write_text(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8-sig")
    return str(path)


def get_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    with _embedding_lock:
        if _embedding_model is not None:
            return _embedding_model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SummaryPipelineError(
                "Khong tim thay sentence-transformers. Hay cai them dependency backend de tao embeddings."
            ) from exc

        model_kwargs = {"dtype": "auto"}

        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cpu",
            model_kwargs=model_kwargs,
        )
        return _embedding_model


def preview_text(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def decode_attachment_to_text(attachment) -> str:
    suffix = Path(attachment.path or attachment.name or "").suffix.lower()

    if attachment.kind == "text" and attachment.content:
        return sanitize_pipeline_text(normalize_whitespace(attachment.content))

    if attachment.encoding == "base64" and attachment.data:
        try:
            raw_bytes = base64.b64decode(attachment.data)
        except (ValueError, binascii.Error) as exc:
            raise SummaryPipelineError(f"File {attachment.name} co du lieu base64 khong hop le.") from exc

        if suffix in DOCLING_EXTENSIONS:
            text, _note = convert_with_docling(raw_bytes, suffix)
            text = clean_extracted_markdown(text)
            text = sanitize_pipeline_text(normalize_whitespace(text))
            if text:
                return text

        for encoding in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
            try:
                text = raw_bytes.decode(encoding)
                text = sanitize_pipeline_text(normalize_whitespace(text))
                if text:
                    return text
            except UnicodeDecodeError:
                continue

    if attachment.content:
        content = sanitize_pipeline_text(normalize_whitespace(attachment.content))
        if content.startswith("FILE: ") and "\nCONTENT:" in content:
            return sanitize_pipeline_text(normalize_whitespace(content.split("\nCONTENT:", 1)[1]))
        return content

    raise SummaryPipelineError(f"Khong doc duoc noi dung tu file {attachment.name}.")


def attachments_to_documents(attachments: list) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for attachment in attachments:
        if attachment.kind == "image":
            continue
        text = decode_attachment_to_text(attachment)
        if text:
            documents.append(
                SourceDocument(
                    name=attachment.name,
                    path=attachment.path,
                    text=text,
                )
            )
    return documents


def dump_documents(documents: list[SourceDocument], run_dir: Path) -> str:
    markdown_parts = []
    for document in documents:
        markdown_parts.append(
            "\n".join(
                [
                    f"# {document.name}",
                    "",
                    f"Path: {document.path}",
                    "",
                    document.text,
                ]
            ).strip()
        )

    return write_text(
        run_dir / "01_documents.md",
        "\n\n---\n\n".join(markdown_parts).strip() + "\n",
    )


def heading_level(line: str) -> tuple[int, str] | None:
    match = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
    if not match:
        return None
    return len(match.group(1)), match.group(2).strip()


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def header_path(headers: dict[str, str]) -> str:
    ordered = []
    for key in ("h1", "h2", "h3", "h4", "h5", "h6"):
        value = headers.get(key)
        if value:
            ordered.append(value)
    return " > ".join(ordered)


def chunk_document(document: SourceDocument) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    current_headers: dict[str, str] = {}
    section_lines: list[str] = []
    next_index = 1

    def flush_section():
        nonlocal next_index, section_lines
        section_text = sanitize_pipeline_text(normalize_whitespace("\n".join(section_lines)))
        section_lines = []
        if not section_text:
            return

        paragraphs = split_paragraphs(section_text)
        buffer = ""
        for paragraph in paragraphs:
            candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
            if buffer and len(candidate) > CHUNK_TARGET_CHARS:
                hdr_path = header_path(current_headers)
                chunks.append(
                    ChunkRecord(
                        index=next_index,
                        source_name=document.name,
                        headers=dict(current_headers),
                        header_path=hdr_path,
                        text=buffer,
                        preview=preview_text(buffer),
                    )
                )
                next_index += 1
                buffer = paragraph
            else:
                buffer = candidate

        if buffer:
            hdr_path = header_path(current_headers)
            chunks.append(
                ChunkRecord(
                    index=next_index,
                    source_name=document.name,
                    headers=dict(current_headers),
                    header_path=hdr_path,
                    text=buffer,
                    preview=preview_text(buffer),
                )
            )
            next_index += 1

    for raw_line in document.text.splitlines():
        line = raw_line.rstrip()
        heading = heading_level(line.strip())
        if heading:
            flush_section()
            level, title = heading
            current_headers = {key: value for key, value in current_headers.items() if int(key[1:]) < level}
            current_headers[f"h{level}"] = title
            continue
        section_lines.append(line)

    flush_section()

    if not chunks:
        text = sanitize_pipeline_text(normalize_whitespace(document.text))
        if text:
            chunks.append(
                ChunkRecord(
                    index=1,
                    source_name=document.name,
                    headers={},
                    header_path="",
                    text=text,
                    preview=preview_text(text),
                )
            )

    return chunks


def chunk_documents(documents: list[SourceDocument]) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    global_index = 1
    for document in documents:
        for chunk in chunk_document(document):
            chunk.index = global_index
            chunks.append(chunk)
            global_index += 1
    return chunks


def dump_chunks(chunks: list[ChunkRecord], run_dir: Path) -> str:
    markdown_parts = []
    for chunk in chunks:
        title = chunk.header_path or chunk.source_name or f"Chunk {chunk.index}"
        markdown_parts.append(
            "\n".join(
                [
                    f"## Chunk {chunk.index}",
                    "",
                    f"- Source: {chunk.source_name}",
                    f"- Header: {title}",
                    "",
                    chunk.text,
                ]
            ).strip()
        )

    return write_text(
        run_dir / "02_chunks.md",
        "\n\n---\n\n".join(markdown_parts).strip() + "\n",
    )


def embed_chunks(chunks: list[ChunkRecord]) -> np.ndarray:
    model = get_embedding_model()
    texts = []
    for chunk in chunks:
        chunk_text = sanitize_pipeline_text(chunk.text)
        content = f"{chunk.header_path}\n\n{chunk_text}".strip() if chunk.header_path else chunk_text
        if EMBEDDING_TASK_PREFIX:
            texts.append(f"{EMBEDDING_TASK_PREFIX}{content}".strip())
        else:
            texts.append(content)

    encode_kwargs = {
        "batch_size": 8,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    if EMBEDDING_PROMPT:
        encode_kwargs["prompt"] = EMBEDDING_PROMPT
    elif EMBEDDING_PROMPT_NAME:
        encode_kwargs["prompt_name"] = EMBEDDING_PROMPT_NAME

    embeddings = model.encode(
        texts,
        **encode_kwargs,
    )
    return np.asarray(embeddings, dtype=np.float32)


def dump_embeddings(chunks: list[ChunkRecord], embeddings: np.ndarray, run_dir: Path) -> str:
    npy_path = run_dir / "03_embeddings.npy"
    np.save(npy_path, embeddings)
    return str(npy_path)


def cluster_embeddings(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    item_count = len(embeddings)
    if item_count == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    if item_count <= 4:
        return np.zeros((item_count,), dtype=np.int32), np.ones((item_count,), dtype=np.float32)

    try:
        from umap import UMAP
        from fast_plscan import PLSCAN
    except ImportError as exc:
        raise SummaryPipelineError(
            "Khong tim thay umap-learn/fast-plscan. Hay cai them dependency backend de cluster chunks."
        ) from exc

    n_neighbors = min(8, max(2, item_count - 1))
    use_umap = item_count >= 6
    if use_umap:
        try:
            reducer = UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
                init="random",
            )
            reduced = reducer.fit_transform(embeddings)
        except Exception:
            reduced = embeddings
    else:
        reduced = embeddings

    min_samples = 2 if item_count < 24 else 3
    try:
        clusterer = PLSCAN(
            min_samples=min_samples,
            metric="euclidean",
        )
        clusterer.fit(reduced)
        labels = clusterer.labels_
        probabilities = getattr(clusterer, "probabilities_", np.ones((item_count,), dtype=np.float32))
    except Exception:
        return np.zeros((item_count,), dtype=np.int32), np.ones((item_count,), dtype=np.float32)

    if all(int(label) == -1 for label in labels):
        return np.zeros((item_count,), dtype=np.int32), np.ones((item_count,), dtype=np.float32)

    return np.asarray(labels, dtype=np.int32), np.asarray(probabilities, dtype=np.float32)


def dump_clusters(
    chunks: list[ChunkRecord],
    labels: np.ndarray,
    probabilities: np.ndarray,
    run_dir: Path,
) -> str:
    records = []
    cluster_groups: dict[int, list[dict]] = defaultdict(list)
    for chunk, label, probability in zip(chunks, labels, probabilities):
        record = {
            "index": chunk.index,
            "source_name": chunk.source_name,
            "header_path": chunk.header_path,
            "preview": chunk.preview,
            "cluster_label": int(label),
            "cluster_probability": float(probability),
        }
        records.append(record)
        cluster_groups[int(label)].append(record)

    markdown_parts = []
    for cluster_label in sorted(cluster_groups.keys()):
        members = cluster_groups[cluster_label]
        markdown_parts.append(
            "\n".join(
                [
                    f"## Cluster {cluster_label}",
                    "",
                    *[
                        f"- Chunk {item['index']} | p={item['cluster_probability']:.3f} | {item['header_path'] or item['source_name']} | {item['preview']}"
                        for item in members
                    ],
                ]
            ).strip()
        )

    return write_text(
        run_dir / "04_clusters.md",
        "\n\n---\n\n".join(markdown_parts).strip() + "\n",
    )


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.divide(vectors, np.clip(norms, 1e-12, None))
    return safe @ safe.T


def select_submodular_indices(
    cluster_embeddings: np.ndarray,
    top_k: int | None = None,
    coverage_weight: float = SUBMODULAR_COVERAGE_WEIGHT,
    relevance_weight: float = SUBMODULAR_RELEVANCE_WEIGHT,
) -> list[int]:
    """Greedy submodular selector for cluster summarization.

    Objective:
        F(S) = alpha * sum_i max_{j in S} sim(i, j)
             + beta  * sum_{j in S} rel(j)

    Where:
        - sim(i, j) is cosine similarity on normalized chunk embeddings.
        - rel(j) is similarity between chunk j and the cluster centroid.

    The facility-location term favors representative coverage of the whole
    cluster; the modular relevance term keeps the selected subset anchored to
    the dominant topic of that cluster. Both terms are monotone submodular (or
    modular), so a simple greedy algorithm works well under a fixed budget K.
    """
    item_count = len(cluster_embeddings)
    top_k = SUBMODULAR_TOP_K if top_k is None else max(1, min(top_k, item_count))
    if item_count == 0:
        return []
    if item_count <= top_k:
        return list(range(item_count))

    norms = np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
    embeddings_norm = cluster_embeddings / np.clip(norms, 1e-12, None)

    similarity = np.clip(embeddings_norm @ embeddings_norm.T, 0.0, 1.0)
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=0.0)

    cluster_centroid = np.mean(embeddings_norm, axis=0, keepdims=True)
    cluster_centroid = cluster_centroid / np.clip(
        np.linalg.norm(cluster_centroid, axis=1, keepdims=True), 1e-12, None
    )
    relevance = np.clip((embeddings_norm @ cluster_centroid.T).reshape(-1), 0.0, 1.0)
    relevance = np.nan_to_num(relevance, nan=0.0, posinf=1.0, neginf=0.0)

    covered = np.zeros((item_count,), dtype=np.float32)
    selected: list[int] = []
    remaining = set(range(item_count))

    while remaining and len(selected) < top_k:
        best_index = None
        best_gain = -math.inf

        for candidate in remaining:
            coverage_gain = float(np.maximum(covered, similarity[:, candidate]).sum() - covered.sum())
            relevance_gain = float(relevance[candidate])
            total_gain = coverage_weight * coverage_gain + relevance_weight * relevance_gain
            if total_gain > best_gain:
                best_gain = total_gain
                best_index = candidate

        if best_index is None:
            break

        selected.append(best_index)
        covered = np.maximum(covered, similarity[:, best_index])
        remaining.remove(best_index)

    return selected


def select_mmr_indices(cluster_embeddings: np.ndarray, top_k: int | None = None) -> list[int]:
    """Backward-compatible wrapper kept for existing call sites/artifacts."""
    return select_submodular_indices(cluster_embeddings, top_k=top_k)


def dominant_cluster_title(cluster_chunks: list[ChunkRecord]) -> str:
    header_counts = Counter(chunk.header_path for chunk in cluster_chunks if chunk.header_path)
    if header_counts:
        return header_counts.most_common(1)[0][0]

    source_counts = Counter(chunk.source_name for chunk in cluster_chunks)
    if source_counts:
        return source_counts.most_common(1)[0][0]

    return "Cluster"


def build_clusters(chunks: list[ChunkRecord], embeddings: np.ndarray, labels: np.ndarray) -> list[ClusterRecord]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[int(label)].append(index)

    clusters: list[ClusterRecord] = []
    next_noise_id = 1
    for raw_label in sorted(grouped.keys(), key=lambda value: (value == -1, value)):
        member_indices = grouped[raw_label]
        if raw_label == -1:
            for member_index in member_indices:
                cluster_chunks = [chunks[member_index]]
                clusters.append(
                    ClusterRecord(
                        cluster_id=f"noise_{next_noise_id}",
                        label=-1,
                        size=1,
                        title=dominant_cluster_title(cluster_chunks),
                        chunk_indices=[chunks[member_index].index],
                        representative_indices=[chunks[member_index].index],
                        representative_previews=[chunks[member_index].preview],
                    )
                )
                next_noise_id += 1
            continue

        cluster_chunks = [chunks[index] for index in member_indices]
        local_selected = select_submodular_indices(embeddings[member_indices])
        selected_indices = [member_indices[position] for position in local_selected]
        selected_chunks = [chunks[index] for index in selected_indices]
        clusters.append(
            ClusterRecord(
                cluster_id=f"cluster_{raw_label}",
                label=raw_label,
                size=len(member_indices),
                title=dominant_cluster_title(cluster_chunks),
                chunk_indices=[chunk.index for chunk in cluster_chunks],
                representative_indices=[chunk.index for chunk in selected_chunks],
                representative_previews=[chunk.preview for chunk in selected_chunks],
            )
        )

    return clusters


def dump_mmr_selection(clusters: list[ClusterRecord], run_dir: Path) -> str:
    markdown_parts = []
    for cluster in clusters:
        markdown_parts.append(
            "\n".join(
                [
                    f"## {cluster.cluster_id}",
                    "",
                    f"- Title: {cluster.title}",
                    f"- Size: {cluster.size}",
                    f"- Selected indices: {', '.join(map(str, cluster.representative_indices))}",
                    "",
                    *[f"- {preview}" for preview in cluster.representative_previews],
                ]
            ).strip()
        )

    return write_text(
        run_dir / "05_mmr_selection.md",
        "\n\n---\n\n".join(markdown_parts).strip() + "\n",
    )


def build_cluster_summary_prompt(cluster: ClusterRecord, selected_chunks: list[ChunkRecord]) -> list[dict]:
    evidence_blocks = []
    for chunk in selected_chunks:
        heading = chunk.header_path or chunk.source_name
        evidence_blocks.append(
            "\n".join(
                [
                    f"[Chunk {chunk.index}] {heading}",
                    chunk.text,
                ]
            )
        )

    return [
        {
            "role": "system",
            "content": (
                "Context:\n"
                "Ban la he thong tom tat tai lieu tieng Viet. Ban nhan duoc mot nhom doan van ban dai dien cho cung mot cum noi dung. "
                "Cac doan nay co the den tu nhieu loai tai lieu khac nhau nhu bao cao, tai lieu ky thuat, bai viet hoc thuat, ghi chu hoac van ban tong hop.\n\n"
                "Instruction:\n"
                "Hay viet mot ban tom tat ngan gon, mach lac va de hieu cho cum noi dung nay. Tom tat can neu duoc chu de chinh va cac y quan trong nhat. "
                "Neu trong dau vao co thong tin cu the nhu quy trinh, thanh phan, quyet dinh, ket qua, so lieu hoac ket luan thi giu lai o muc can thiet de nguoi doc hieu dung noi dung.\n\n"
                "Constraints:\n"
                "- Chi su dung thong tin co trong phan bang chung.\n"
                "- Khong suy dien, khong them thong tin ngoai dau vao.\n"
                "- Viet bang tieng Viet.\n"
                "- Uu tien van phong tu nhien, mach lac, noi y ro rang.\n"
                "- Tom tat du y nhung tranh lan man, tranh lap lai cung mot noi dung.\n"
                "- Uu tien giu lai cac thong tin quan trong va cu the neu chung giup lam ro noi dung.\n"
                "- Do dai nen tuong xung voi luong thong tin trong dau vao; khong rut qua muc nhung cung khong mo rong khong can thiet.\n"
                "- Khong viet meta-commentary nhu 'Duoi day la tom tat', 'Tom tat noi dung', 'Bang chung cho thay'.\n"
                "- Khong dung bullet, khong dung tieu de, khong dung tag nhu <think>."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "Relevance:",
                    f"- Tieu de nhom: {cluster.title}",
                    f"- So chunk trong nhom: {cluster.size}",
                    "",
                    "Bang chung:",
                    "\n\n".join(evidence_blocks),
                ]
            ),
        },
    ]

def build_global_summary_prompt(clusters: list[ClusterRecord]) -> list[dict]:
    cluster_blocks = []
    for cluster in clusters:
        if cluster.label == -1 or cluster.cluster_id.startswith("noise_"):
            continue
        if not (cluster.summary or "").strip():
            continue
        cluster_blocks.append(
            "\n".join(
                [
                    f"{cluster.cluster_id}: {cluster.title}",
                    cluster.summary.strip(),
                ]
            ).strip()
        )

    return [
        {
            "role": "system",
            "content": (
                "Context:\n"
                "Ban la he thong tong hop va tom tat tai lieu tieng Viet. Ban nhan duoc cac ban tom tat cum da duoc sap xep theo trinh tu noi dung. "
                "Cac cum nay phan anh nhung phan quan trong cua mot tai lieu hoac mot tap tai lieu.\n\n"
                "Instruction:\n"
                "Hay tong hop cac tom tat cum thanh mot ban tom tat tong the ro rang, mach lac va de theo doi. "
                "Ban tom tat can phan anh dung noi dung chinh cua toan bo tai lieu, ket noi cac y theo logic tu nhien, va giu lai nhung diem quan trong nhat. "
                "Neu dau vao co muc tieu, quy trinh, phuong phap, phat hien, so lieu, ket qua hoac ket luan thi hay dua vao khi chung thuc su quan trong cho viec hieu tai lieu.\n\n"
                "Constraints:\n"
                "- Chi su dung thong tin co trong cac tom tat cum.\n"
                "- Khong suy dien, khong them thong tin moi.\n"
                "- Viet bang tieng Viet.\n"
                "- Uu tien van phong mach lac, tu nhien, ket noi y tot.\n"
                "- Tom tat du y nhung tranh dai dong, tranh lap y va tranh liet ke vun.\n"
                "- Uu tien giu lai cac chi tiet quan trong neu chung giup ban tom tat ro hon.\n"
                "- Do dai nen tuong xung voi luong thong tin trong dau vao; khong rut qua muc nhung cung khong mo rong khong can thiet.\n"
                "- Khong viet meta-commentary nhu 'Duoi day la ban tong hop', 'Toi da tong hop'.\n"
                "- Khong dung bullet, khong dung tieu de, khong dung tag nhu <think>."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "Relevance:",
                    "Cac tom tat theo cum chu de:",
                    "\n\n".join(cluster_blocks) if cluster_blocks else "Khong co cum hop le nao sau khi loc noise.",
                ]
            ),
        },
    ]


def _document_cluster_sequence(clusters: list[ClusterRecord]) -> list[int]:
    chunk_order = []
    for cluster_index, cluster in enumerate(clusters):
        for chunk_index in cluster.chunk_indices:
            chunk_order.append((chunk_index, cluster_index))

    if not chunk_order:
        return []

    chunk_order.sort(key=lambda item: item[0])
    sequence = []
    for _, cluster_index in chunk_order:
        if not sequence or sequence[-1] != cluster_index:
            sequence.append(cluster_index)
    return sequence


def _markov_transition_probs(sequence: list[int], cluster_count: int) -> tuple[np.ndarray, np.ndarray]:
    smoothing = 1e-3
    start_counts = np.full((cluster_count,), smoothing, dtype=np.float64)
    transition_counts = np.full((cluster_count, cluster_count), smoothing, dtype=np.float64)

    if sequence:
        start_counts[sequence[0]] += 1.0

    for current_index, next_index in zip(sequence, sequence[1:]):
        transition_counts[current_index, next_index] += 1.0

    start_probs = start_counts / np.clip(start_counts.sum(), 1e-12, None)
    transition_probs = transition_counts / np.clip(transition_counts.sum(axis=1, keepdims=True), 1e-12, None)
    return start_probs, transition_probs


def _solve_markov_hamiltonian(start_probs: np.ndarray, transition_probs: np.ndarray) -> list[int]:
    cluster_count = len(start_probs)
    if cluster_count <= 1:
        return list(range(cluster_count))

    if cluster_count > 18:
        remaining = set(range(cluster_count))
        first = int(np.argmax(start_probs))
        path = [first]
        remaining.remove(first)
        while remaining:
            current = path[-1]
            next_index = max(remaining, key=lambda candidate: float(transition_probs[current, candidate]))
            path.append(int(next_index))
            remaining.remove(next_index)
        return path

    log_start = np.log(np.clip(start_probs, 1e-12, None))
    log_transition = np.log(np.clip(transition_probs, 1e-12, None))

    best_scores: dict[tuple[int, int], float] = {}
    parents: dict[tuple[int, int], int] = {}

    for cluster_index in range(cluster_count):
        mask = 1 << cluster_index
        best_scores[(mask, cluster_index)] = float(log_start[cluster_index])

    for subset_size in range(2, cluster_count + 1):
        next_scores: dict[tuple[int, int], float] = {}
        for mask, last_index in list(best_scores.keys()):
            if mask.bit_count() != subset_size - 1:
                continue
            base_score = best_scores[(mask, last_index)]
            for candidate in range(cluster_count):
                if mask & (1 << candidate):
                    continue
                new_mask = mask | (1 << candidate)
                score = base_score + float(log_transition[last_index, candidate])
                key = (new_mask, candidate)
                if score > next_scores.get(key, -math.inf):
                    next_scores[key] = score
                    parents[key] = last_index
        best_scores.update(next_scores)

    full_mask = (1 << cluster_count) - 1
    last_index = max(
        range(cluster_count),
        key=lambda candidate: best_scores.get((full_mask, candidate), -math.inf),
    )

    path = [last_index]
    mask = full_mask
    while len(path) < cluster_count:
        parent = parents[(mask, path[-1])]
        mask ^= 1 << path[-1]
        path.append(parent)
    path.reverse()
    return path


def order_clusters_markov(clusters: list[ClusterRecord]) -> list[ClusterRecord]:
    if len(clusters) <= 2:
        return list(clusters)

    sequence = _document_cluster_sequence(clusters)
    if len(sequence) <= 1:
        return list(clusters)

    start_probs, transition_probs = _markov_transition_probs(sequence, len(clusters))
    ordered_indices = _solve_markov_hamiltonian(start_probs, transition_probs)
    return [clusters[index] for index in ordered_indices]


def dump_markov_ordering(clusters: list[ClusterRecord], ordered_clusters: list[ClusterRecord], run_dir: Path) -> str:
    cluster_lookup = {cluster.cluster_id: cluster for cluster in clusters}
    cluster_to_index = {cluster.cluster_id: index for index, cluster in enumerate(clusters)}

    sequence_indices = _document_cluster_sequence(clusters)
    sequence_cluster_ids = [clusters[index].cluster_id for index in sequence_indices]
    start_probs, transition_probs = _markov_transition_probs(sequence_indices, len(clusters))

    markdown_parts = [
        "# Markov Ordering",
        "",
        "## Original Cluster Order",
        "",
        *[
            f"- {index + 1}. {cluster.cluster_id} | {cluster.title}"
            for index, cluster in enumerate(clusters)
        ],
        "",
        "## Document Cluster Sequence",
        "",
        (" -> ".join(sequence_cluster_ids) if sequence_cluster_ids else "(empty)"),
        "",
        "## Ordered Cluster Sequence",
        "",
        *[
            f"- {index + 1}. {cluster.cluster_id} | {cluster.title}"
            for index, cluster in enumerate(ordered_clusters)
        ],
        "",
        "## Start Probabilities",
        "",
        *[
            f"- {cluster.cluster_id}: {float(start_probs[cluster_to_index[cluster.cluster_id]]):.6f}"
            for cluster in clusters
        ],
        "",
        "## Transition Matrix",
        "",
    ]

    for source_cluster in clusters:
        source_index = cluster_to_index[source_cluster.cluster_id]
        probs = []
        for target_cluster in clusters:
            target_index = cluster_to_index[target_cluster.cluster_id]
            probs.append(f"{target_cluster.cluster_id}={float(transition_probs[source_index, target_index]):.6f}")
        markdown_parts.append(f"- {source_cluster.cluster_id}: " + ", ".join(probs))

    markdown_path = run_dir / "06b_markov_order.md"
    return write_text(markdown_path, "\n".join(markdown_parts).strip() + "\n")


def pipeline_debug_payload(documents: list[SourceDocument], chunks: list[ChunkRecord], clusters: list[ClusterRecord]) -> dict:
    return {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "cluster_count": len(clusters),
        "documents": [asdict(document) | {"text": preview_text(document.text, 240)} for document in documents],
        "clusters": [asdict(cluster) for cluster in clusters],
    }


def dump_cluster_summaries(clusters: list[ClusterRecord], run_dir: Path) -> str:
    markdown_parts = []
    for cluster in clusters:
        markdown_parts.append(
            "\n".join(
                [
                    f"## {cluster.cluster_id}: {cluster.title}",
                    "",
                    f"- Size: {cluster.size}",
                    f"- Representative indices: {', '.join(map(str, cluster.representative_indices))}",
                    "",
                    cluster.summary.strip() or "(empty)",
                ]
            ).strip()
        )

    return write_text(
        run_dir / "06_cluster_summaries.md",
        "\n\n---\n\n".join(markdown_parts).strip() + "\n",
    )


def dump_final_summary(overall_summary: str, clusters: list[ClusterRecord], run_dir: Path) -> str:
    return write_text(
        run_dir / "07_final_summary.md",
        "\n".join(
            [
                "# Overall Summary",
                "",
                overall_summary.strip() or "(empty)",
            ]
        ).strip()
        + "\n",
    )


