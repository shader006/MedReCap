"""
main.py - FastAPI backend dung llama-cpp-python + Unsloth GGUF
Khong can Ollama. Load model truc tiep tu file .gguf.
"""

import json
import os
import re
import subprocess
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

try:
    from .docling import MAX_ATTACHMENT_BYTES, MAX_ATTACHMENT_TEXT_CHARS, extract_attachment_text, format_size
    from .summary_pipeline import (
        SummaryPipelineError,
        attachments_to_documents,
        build_run_output_dir,
        build_cluster_summary_prompt,
        build_clusters,
        build_global_summary_prompt,
        chunk_documents,
        order_clusters_markov,
        dump_chunks,
        dump_cluster_summaries,
        dump_clusters,
        dump_documents,
        dump_embeddings,
        dump_final_summary,
        dump_markov_ordering,
        dump_mmr_selection,
        embed_chunks,
        cluster_embeddings,
        pipeline_debug_payload,
    )
except ImportError:
    from docling import MAX_ATTACHMENT_BYTES, MAX_ATTACHMENT_TEXT_CHARS, extract_attachment_text, format_size
    from summary_pipeline import (
        SummaryPipelineError,
        attachments_to_documents,
        build_run_output_dir,
        build_cluster_summary_prompt,
        build_clusters,
        build_global_summary_prompt,
        chunk_documents,
        order_clusters_markov,
        dump_chunks,
        dump_cluster_summaries,
        dump_clusters,
        dump_documents,
        dump_embeddings,
        dump_final_summary,
        dump_markov_ordering,
        dump_mmr_selection,
        embed_chunks,
        cluster_embeddings,
        pipeline_debug_payload,
    )

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent
PORT = int(os.getenv("PORT", 8888))
MODEL_DIR = (BACKEND_DIR / os.getenv("MODEL_DIR", "../models")).resolve()
# Dat ten file .gguf muon dung tai day. De rong "" de dung MODEL_PATH / MODEL_FILE / auto-detect.
PREFERRED_MODEL_FILE = "Qwen3.5-2B-Q4_K_M.gguf"
MODEL_PATH_ENV = os.getenv("MODEL_PATH", "").strip()
MODEL_FILE = os.getenv("MODEL_FILE", "").strip()
N_CTX = int(os.getenv("N_CTX", 32768))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", -1))
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", 48000))
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", 24000))
PROMPT_RESERVED_TOKENS = int(os.getenv("PROMPT_RESERVED_TOKENS", 2048))
SUMMARY_N_CTX = int(os.getenv("SUMMARY_N_CTX", str(N_CTX)))
SUMMARY_PROMPT_REPEAT = max(1, int(os.getenv("SUMMARY_PROMPT_REPEAT", "1")))


def find_gguf() -> Path:
    if PREFERRED_MODEL_FILE:
        preferred_file = (MODEL_DIR / PREFERRED_MODEL_FILE).resolve()
        if not preferred_file.exists():
            raise RuntimeError(
                f"PREFERRED_MODEL_FILE '{PREFERRED_MODEL_FILE}' khong ton tai trong '{MODEL_DIR.resolve()}'."
            )
        return preferred_file

    if MODEL_PATH_ENV:
        configured_path = Path(MODEL_PATH_ENV)
        if not configured_path.is_absolute():
            configured_path = (BACKEND_DIR / configured_path).resolve()
        if not configured_path.exists():
            raise RuntimeError(
                f"MODEL_PATH dang tro toi file khong ton tai: '{configured_path}'."
            )
        if configured_path.suffix.lower() != ".gguf":
            raise RuntimeError(
                f"MODEL_PATH phai tro toi file .gguf, nhung dang la: '{configured_path.name}'."
            )
        return configured_path

    if MODEL_FILE:
        configured_file = (MODEL_DIR / MODEL_FILE).resolve()
        if not configured_file.exists():
            raise RuntimeError(
                f"MODEL_FILE '{MODEL_FILE}' khong ton tai trong '{MODEL_DIR.resolve()}'."
            )
        return configured_file

    files = sorted(MODEL_DIR.glob("*.gguf"))
    if not files:
        raise RuntimeError(
            f"Khong tim thay file .gguf trong '{MODEL_DIR.resolve()}'. "
            "Hay chay setup.bat de tai model truoc."
        )
    return files[0]

from llama_cpp import Llama

print("Loading model...")
model_path = find_gguf()
print(f"Model: {model_path.name}")

llm = Llama(
    model_path=str(model_path),
    n_gpu_layers=N_GPU_LAYERS,
    n_ctx=N_CTX,
    n_batch=1024,
    n_threads=8,
    verbose=False,
    chat_format="chatml",
)
generation_lock = threading.Lock()
print("Model ready.")


def get_local_summary_llm():
    return llm

app = FastAPI(title="Qwen3.5-2B API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class AttachmentPayload(BaseModel):
    kind: str
    name: str
    path: str
    size: int = 0
    mime_type: str = ""
    truncated: bool = False
    content: str = ""
    encoding: str = ""
    data: str = ""


class ChatRequest(BaseModel):
    messages: list[Message]
    attachments: list[AttachmentPayload] = []
    stream: bool = False
    temperature: float = 1.0
    max_tokens: int = 12288
    thinking: bool = False


class SummaryRequest(BaseModel):
    attachments: list[AttachmentPayload]
    temperature: float = 0.2
    cluster_max_tokens: int = 1280
    final_max_tokens: int = 2048


@dataclass
class CompletionResult:
    content: str
    raw_content: str
    response_payload: dict


def count_text_tokens(text: str) -> int:
    if not text:
        return 0
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))


def count_message_tokens(messages: list[dict]) -> dict[str, int]:
    total = 0
    for message in messages:
        total += count_text_tokens(message.get("role", ""))
        total += count_text_tokens(message.get("content", ""))
    return {
        "message_count": len(messages),
        "total_tokens": total,
    }


def build_token_report(
    documents: list,
    chunks: list,
    clusters: list,
    raw_llm_outputs: dict,
    ordered_clusters: list,
) -> dict:
    document_records = []
    for document in documents:
        document_records.append(
            {
                "name": document.name,
                "chars": len(document.text),
                "tokens": count_text_tokens(document.text),
            }
        )

    chunk_records = []
    for chunk in chunks:
        chunk_text = f"{chunk.header_path}\n\n{chunk.text}".strip() if chunk.header_path else chunk.text
        chunk_records.append(
            {
                "index": chunk.index,
                "source_name": chunk.source_name,
                "chars": len(chunk_text),
                "tokens": count_text_tokens(chunk_text),
            }
        )

    cluster_usage_lookup = {
        item["cluster_id"]: (item.get("response_payload") or {}).get("usage", {})
        for item in raw_llm_outputs.get("cluster_summaries", [])
    }
    cluster_prompt_lookup = {
        item["cluster_id"]: count_message_tokens(item.get("prompt") or [])
        for item in raw_llm_outputs.get("cluster_summaries", [])
    }

    cluster_records = []
    for cluster in clusters:
        prompt_stats = cluster_prompt_lookup.get(cluster.cluster_id, {"message_count": 0, "total_tokens": 0})
        usage = cluster_usage_lookup.get(cluster.cluster_id, {})
        cluster_records.append(
            {
                "cluster_id": cluster.cluster_id,
                "title": cluster.title,
                "size": cluster.size,
                "prompt_tokens_estimated": prompt_stats["total_tokens"],
                "prompt_message_count": prompt_stats["message_count"],
                "summary_tokens_estimated": count_text_tokens(cluster.summary or ""),
                "prompt_tokens_reported": usage.get("prompt_tokens", 0),
                "completion_tokens_reported": usage.get("completion_tokens", 0),
                "total_tokens_reported": usage.get("total_tokens", 0),
            }
        )

    final_prompt_messages = build_global_summary_prompt(ordered_clusters)
    final_prompt_stats = count_message_tokens(final_prompt_messages)
    final_payload = raw_llm_outputs.get("final_summary") or {}
    final_usage = (final_payload.get("response_payload") or {}).get("usage", {})

    return {
        "documents": {
            "count": len(document_records),
            "total_tokens": sum(item["tokens"] for item in document_records),
            "records": document_records,
        },
        "chunks": {
            "count": len(chunk_records),
            "total_tokens": sum(item["tokens"] for item in chunk_records),
            "max_tokens": max((item["tokens"] for item in chunk_records), default=0),
            "records": chunk_records,
        },
        "cluster_summaries": {
            "count": len(cluster_records),
            "records": cluster_records,
        },
        "final_summary": {
            "cluster_order": [cluster.cluster_id for cluster in ordered_clusters],
            "prompt_tokens_estimated": final_prompt_stats["total_tokens"],
            "prompt_message_count": final_prompt_stats["message_count"],
            "summary_tokens_estimated": count_text_tokens(final_payload.get("content", "")),
            "prompt_tokens_reported": final_usage.get("prompt_tokens", 0),
            "completion_tokens_reported": final_usage.get("completion_tokens", 0),
            "total_tokens_reported": final_usage.get("total_tokens", 0),
        },
    }


def dump_token_report(token_report: dict, run_dir: Path) -> str:
    lines = [
        "# Token Counts",
        "",
        f"- Document total: {token_report['documents']['total_tokens']}",
        f"- Chunk total: {token_report['chunks']['total_tokens']}",
        f"- Max chunk: {token_report['chunks']['max_tokens']}",
        "",
        "## Documents",
        "",
    ]
    for record in token_report["documents"]["records"]:
        lines.append(f"- {record['name']}: {record['tokens']} tokens ({record['chars']} chars)")

    lines.extend(
        [
            "",
            "## Cluster Prompts",
            "",
        ]
    )
    for record in token_report["cluster_summaries"]["records"]:
        lines.append(
            "- "
            + f"{record['cluster_id']} | prompt_est={record['prompt_tokens_estimated']} | "
            + f"summary_est={record['summary_tokens_estimated']} | "
            + f"usage_prompt={record['prompt_tokens_reported']} | "
            + f"usage_completion={record['completion_tokens_reported']}"
        )

    final_record = token_report["final_summary"]
    lines.extend(
        [
            "",
            "## Final Summary",
            "",
            f"- Prompt est: {final_record['prompt_tokens_estimated']}",
            f"- Summary est: {final_record['summary_tokens_estimated']}",
            f"- Usage prompt: {final_record['prompt_tokens_reported']}",
            f"- Usage completion: {final_record['completion_tokens_reported']}",
            f"- Usage total: {final_record['total_tokens_reported']}",
        ]
    )

    path = run_dir / "10_token_counts.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return str(path)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": model_path.name,
        "model_ready": True,
        "ctx": N_CTX,
        "summary_ctx": SUMMARY_N_CTX,
        "mode": "local",
        "remote": None,
    }


@app.post("/attachments/process")
async def process_attachment(attachment: AttachmentPayload):
    block = await run_in_threadpool(
        extract_attachment_text,
        attachment,
        MAX_ATTACHMENT_BYTES,
        MAX_ATTACHMENT_TEXT_CHARS,
    )
    return {
        "path": attachment.path,
        "kind": attachment.kind,
        "content": block,
        "truncated": "Noi dung da bi cat bot de vua context." in block,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    requested_max_tokens = max(128, min(req.max_tokens, N_CTX // 2))
    available_prompt_tokens = max(1024, N_CTX - requested_max_tokens - PROMPT_RESERVED_TOKENS)
    raw_messages = [m.dict() for m in req.messages]
    messages = await run_in_threadpool(prepare_messages, raw_messages, req.attachments, available_prompt_tokens)
    max_tokens = min(requested_max_tokens, max(256, N_CTX // 2))

    params = dict(
        messages=messages,
        max_tokens=max_tokens,
        temperature=req.temperature,
        top_k=20,
        top_p=0.95 if req.thinking else 1.0,
        repeat_penalty=1.1,
        stream=req.stream,
    )

    if req.stream:
        return StreamingResponse(
            iterate_in_threadpool(_stream(params)),
            media_type="text/event-stream",
        )

    response = await run_in_threadpool(_create_chat_completion, params)
    content = response["choices"][0]["message"]["content"]
    content = strip_thinking(content)
    usage = response.get("usage", {})
    return {
        "content": content,
        "model": model_path.name,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


@app.post("/summaries/generate")
async def generate_summary(req: SummaryRequest):
    if not req.attachments:
        return {"error": "Khong co attachment nao de tom tat."}

    run_dir = None
    stage_timings: dict[str, float] = {}

    def record_stage(stage_name: str, started_at: float) -> None:
        stage_timings[stage_name] = round(time.perf_counter() - started_at, 4)

    try:
        pipeline_started_at = time.perf_counter()

        started_at = time.perf_counter()
        documents = await run_in_threadpool(attachments_to_documents, req.attachments)
        record_stage("attachments_to_documents", started_at)
        if not documents:
            raise SummaryPipelineError("Khong co noi dung van ban hop le de tom tat.")

        started_at = time.perf_counter()
        run_dir = await run_in_threadpool(build_run_output_dir, documents)
        record_stage("build_run_output_dir", started_at)

        started_at = time.perf_counter()
        artifact_paths = {
            "run_dir": str(run_dir),
            "documents": await run_in_threadpool(dump_documents, documents, run_dir),
        }
        record_stage("dump_documents", started_at)

        started_at = time.perf_counter()
        chunks = await run_in_threadpool(chunk_documents, documents)
        record_stage("chunk_documents", started_at)
        if not chunks:
            raise SummaryPipelineError("Khong tao duoc chunk nao tu tai lieu.")

        started_at = time.perf_counter()
        artifact_paths["chunks"] = await run_in_threadpool(dump_chunks, chunks, run_dir)
        record_stage("dump_chunks", started_at)

        started_at = time.perf_counter()
        embeddings = await run_in_threadpool(embed_chunks, chunks)
        record_stage("embed_chunks", started_at)

        started_at = time.perf_counter()
        artifact_paths["embeddings"] = await run_in_threadpool(dump_embeddings, chunks, embeddings, run_dir)
        record_stage("dump_embeddings", started_at)

        started_at = time.perf_counter()
        labels, _probabilities = await run_in_threadpool(cluster_embeddings, embeddings)
        record_stage("cluster_embeddings", started_at)

        started_at = time.perf_counter()
        artifact_paths["clusters"] = await run_in_threadpool(dump_clusters, chunks, labels, _probabilities, run_dir)
        record_stage("dump_clusters", started_at)

        started_at = time.perf_counter()
        clusters = await run_in_threadpool(build_clusters, chunks, embeddings, labels)
        record_stage("build_clusters", started_at)

        started_at = time.perf_counter()
        artifact_paths["mmr_selection"] = await run_in_threadpool(dump_mmr_selection, clusters, run_dir)
        record_stage("dump_mmr_selection", started_at)
        raw_llm_outputs = {"cluster_summaries": [], "final_summary": None}

        chunk_by_index = {chunk.index: chunk for chunk in chunks}
        cluster_summaries_started_at = time.perf_counter()
        for cluster in clusters:
            selected_chunks = [chunk_by_index[index] for index in cluster.representative_indices if index in chunk_by_index]
            prompt = build_cluster_summary_prompt(cluster, selected_chunks)
            try:
                cluster_completion = await generate_completion(
                    prompt,
                    max_tokens=req.cluster_max_tokens,
                    temperature=req.temperature,
                    thinking=False,
                    prompt_repeat=SUMMARY_PROMPT_REPEAT,
                )
                cluster.summary = cluster_completion.content
                raw_llm_outputs["cluster_summaries"].append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "title": cluster.title,
                        "prompt": prompt,
                        "raw_content": cluster_completion.raw_content,
                        "content": cluster_completion.content,
                        "response_payload": cluster_completion.response_payload,
                        "fallback_used": False,
                    }
                )
            except SummaryPipelineError as exc:
                fallback_summary = build_extractive_cluster_summary(selected_chunks)
                cluster.summary = fallback_summary
                raw_llm_outputs["cluster_summaries"].append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "title": cluster.title,
                        "prompt": prompt,
                        "error": str(exc),
                        "raw_content": getattr(exc, "raw_content", ""),
                        "content": fallback_summary,
                        "response_payload": getattr(exc, "response_payload", {}),
                        "fallback_used": True,
                    }
                )
        record_stage("generate_cluster_summaries", cluster_summaries_started_at)

        started_at = time.perf_counter()
        artifact_paths["cluster_summaries"] = await run_in_threadpool(dump_cluster_summaries, clusters, run_dir)
        record_stage("dump_cluster_summaries", started_at)

        started_at = time.perf_counter()
        ordered_clusters = await run_in_threadpool(order_clusters_markov, clusters)
        record_stage("order_clusters_markov", started_at)

        started_at = time.perf_counter()
        artifact_paths["markov_order"] = await run_in_threadpool(dump_markov_ordering, clusters, ordered_clusters, run_dir)
        record_stage("dump_markov_ordering", started_at)

        final_prompt_messages = build_global_summary_prompt(ordered_clusters)
        final_summary_started_at = time.perf_counter()
        try:
            final_completion = await generate_completion(
                final_prompt_messages,
                max_tokens=req.final_max_tokens,
                temperature=req.temperature,
                thinking=False,
                prompt_repeat=SUMMARY_PROMPT_REPEAT,
            )
            overall_summary = final_completion.content
            raw_llm_outputs["final_summary"] = {
                "raw_content": final_completion.raw_content,
                "content": final_completion.content,
                "response_payload": final_completion.response_payload,
                "fallback_used": False,
            }
        except SummaryPipelineError as exc:
            overall_summary = build_extractive_overall_summary(ordered_clusters)
            raw_llm_outputs["final_summary"] = {
                "error": str(exc),
                "raw_content": getattr(exc, "raw_content", ""),
                "content": overall_summary,
                "response_payload": getattr(exc, "response_payload", {}),
                "fallback_used": True,
            }
        record_stage("generate_final_summary", final_summary_started_at)

        raw_llm_outputs["final_summary"]["cluster_order"] = [cluster.cluster_id for cluster in ordered_clusters]

        started_at = time.perf_counter()
        artifact_paths["final_summary"] = await run_in_threadpool(dump_final_summary, overall_summary, ordered_clusters, run_dir)
        record_stage("dump_final_summary", started_at)

        started_at = time.perf_counter()
        raw_llm_path = run_dir / "08_raw_llm_outputs.json"
        raw_llm_path.write_text(json.dumps(raw_llm_outputs, ensure_ascii=False, indent=2), encoding="utf-8-sig")
        artifact_paths["raw_llm_outputs"] = str(raw_llm_path)
        record_stage("dump_raw_llm_outputs", started_at)

        started_at = time.perf_counter()
        token_report = await run_in_threadpool(
            build_token_report,
            documents,
            chunks,
            clusters,
            raw_llm_outputs,
            ordered_clusters,
        )
        artifact_paths["token_counts"] = await run_in_threadpool(dump_token_report, token_report, run_dir)
        record_stage("dump_token_counts", started_at)

        stage_timings["total"] = round(time.perf_counter() - pipeline_started_at, 4)
        timings_payload = {
            "stages_seconds": stage_timings,
            "ordered_stage_names": list(stage_timings.keys()),
        }
        timings_md_path = run_dir / "09_pipeline_timings.md"
        timings_md_path.write_text(
            "\n".join(
                [
                    "# Pipeline Timings",
                    "",
                    *[f"- {stage}: {seconds:.4f}s" for stage, seconds in stage_timings.items()],
                ]
            )
            + "\n",
            encoding="utf-8-sig",
        )
        artifact_paths["pipeline_timings"] = str(timings_md_path)

        return {
            "model": model_path.name,
            "overall_summary": overall_summary,
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "label": cluster.label,
                    "title": cluster.title,
                    "size": cluster.size,
                    "summary": cluster.summary,
                    "representative_indices": cluster.representative_indices,
                    "representative_previews": cluster.representative_previews,
                }
                for cluster in clusters
            ],
            "artifacts": artifact_paths,
            "token_counts": token_report,
            "debug": pipeline_debug_payload(documents, chunks, clusters),
        }
    except SummaryPipelineError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        error_payload = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        if run_dir is not None:
            crash_path = run_dir / "99_pipeline_crash.json"
            crash_path.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")
            error_payload["crash_log"] = str(crash_path)
        return error_payload


def _create_chat_completion(params: dict):
    with generation_lock:
        return llm.create_chat_completion(**params)


def _create_local_summary_chat_completion(params: dict):
    local_llm = get_local_summary_llm()
    with generation_lock:
        return local_llm.create_chat_completion(**params)


def _stream(params: dict):
    in_think = False
    think_buf = ""

    with generation_lock:
        for chunk in llm.create_chat_completion(**params):
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            done = chunk["choices"][0].get("finish_reason") is not None

            if not token:
                if done:
                    yield sse({"token": "", "done": True})
                continue

            if not in_think and "<think>" in token:
                in_think = True
                think_buf = token
                continue

            if in_think:
                think_buf += token
                if "</think>" in think_buf:
                    after = think_buf.split("</think>", 1)[1]
                    in_think = False
                    think_buf = ""
                    if after.strip():
                        yield sse({"token": after, "done": done})
                continue

            yield sse({"token": token, "done": done})


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def generate_completion_text(
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.2,
    thinking: bool = False,
) -> str:
    result = await generate_completion(messages, max_tokens, temperature, thinking)
    return result.content


def extract_completion_content(response: dict) -> str:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return (message.get("content") or "").strip()


def apply_prompt_repetition(messages: list[dict], repeat_count: int) -> list[dict]:
    repeated_messages = [dict(message) for message in messages]
    if repeat_count <= 1:
        return repeated_messages

    for index in range(len(repeated_messages) - 1, -1, -1):
        if repeated_messages[index].get("role") != "user":
            continue
        content = (repeated_messages[index].get("content") or "").strip()
        if content:
            repeated_messages[index]["content"] = "\n\n".join([content] * repeat_count)
        return repeated_messages
    return repeated_messages


async def generate_completion(
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.2,
    thinking: bool = False,
    prompt_repeat: int = 1,
) -> CompletionResult:
    repeated_messages = apply_prompt_repetition(messages, prompt_repeat if not thinking else 1)

    # Qwen3/3.5: append /no_think to suppress <think> reasoning block
    if not thinking:
        for msg in reversed(repeated_messages):
            if msg.get("role") == "user":
                msg["content"] = msg["content"].rstrip() + "\n/no_think"
                break

    params = dict(
        messages=repeated_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=20,
        top_p=0.95 if thinking else 1.0,
        repeat_penalty=1.1,
        stream=False,
    )
    response = await run_in_threadpool(_create_chat_completion, params)

    raw_content = extract_completion_content(response)
    if not raw_content:
        rescued = extract_rescue_content("", response)
        if rescued:
            return CompletionResult(
                content=rescued,
                raw_content="",
                response_payload=response,
            )

        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or "unknown"
        reasoning_len = len(message.get("reasoning_content") or "")
        exc = SummaryPipelineError(
            "Model khong tra ve message.content cho summary "
            f"(finish_reason={finish_reason}, reasoning_len={reasoning_len}). "
            "Hay giam prompt, tang max_tokens, hoac tat che do reasoning tren model/server."
        )
        setattr(exc, "response_payload", response)
        setattr(exc, "raw_content", "")
        raise exc

    raw_content_stripped = raw_content.strip()
    stripped_content = strip_thinking(raw_content)
    stripped_content = strip_cjk(stripped_content)
    final_content = stripped_content if stripped_content != raw_content_stripped else raw_content_stripped
    if not final_content:
        rescued = extract_rescue_content(raw_content, response)
        if rescued:
            final_content = rescued

    if not final_content:
        choice = (response.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason") or "unknown"
        exc = SummaryPipelineError(
            "Model chi tra ve reasoning hoac noi dung bi cat sau khi loai bo <think> "
            f"(finish_reason={finish_reason}). Hay giam prompt hoac tang max_tokens."
        )
        setattr(exc, "response_payload", response)
        setattr(exc, "raw_content", raw_content)
        raise exc
    return CompletionResult(
        content=final_content,
        raw_content=raw_content,
        response_payload=response,
    )


def strip_thinking(content: str) -> str:
    pattern = re.compile(r"<think>.*?(</think>|$)", flags=re.DOTALL | re.IGNORECASE)
    content = pattern.sub("", content)
    return content.strip()


def extract_rescue_content(raw_content: str, response: dict) -> str:
    """Best-effort salvage path when strip_thinking removes all visible content."""
    candidates: list[str] = []

    if raw_content:
        candidates.append(raw_content)
        no_tag = re.sub(r"</?think>", "", raw_content, flags=re.IGNORECASE).strip()
        if no_tag and no_tag != raw_content:
            candidates.append(no_tag)

    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    reasoning = (message.get("reasoning_content") or "").strip()
    if reasoning:
        candidates.append(reasoning)

    for candidate in candidates:
        cleaned = strip_cjk(strip_thinking(candidate))
        if cleaned:
            return cleaned

    return ""


def strip_cjk(content: str) -> str:
    """Drop standalone CJK-heavy noise blocks without deleting mixed-language content."""
    cleaned_lines: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        cjk_chars = re.findall(r'[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]', line)
        if not cjk_chars:
            cleaned_lines.append(raw_line)
            continue

        non_space_chars = [char for char in line if not char.isspace()]
        if non_space_chars and len(cjk_chars) / len(non_space_chars) >= 0.8:
            continue

        cleaned_lines.append(raw_line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _clean_text_for_fallback(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("<!--"):
            continue
        if line.startswith("|") and line.endswith("|"):
            continue
        if line in {"()", "( )"}:
            continue
        lines.append(line)
    compact = " ".join(lines)
    compact = re.sub(r"\s+", " ", compact).strip()
    return compact


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def build_extractive_cluster_summary(selected_chunks: list, max_sentences: int = 4) -> str:
    sentences: list[str] = []
    seen = set()
    for chunk in selected_chunks:
        cleaned = _clean_text_for_fallback(getattr(chunk, "text", ""))
        for sentence in _split_sentences(cleaned):
            key = sentence.casefold()
            if key in seen:
                continue
            if len(sentence) < 40:
                continue
            seen.add(key)
            sentences.append(sentence)
            if len(sentences) >= max_sentences:
                break
        if len(sentences) >= max_sentences:
            break
    if not sentences:
        return "Khong du noi dung de tao tom tat extractive cho cum nay."
    return " ".join(sentences)


def build_extractive_overall_summary(clusters: list, max_clusters: int = 4) -> str:
    lines = []
    for cluster in clusters[:max_clusters]:
        cluster_text = (cluster.summary or "").strip()
        if not cluster_text:
            continue
        first = _split_sentences(cluster_text)
        line = first[0] if first else cluster_text
        lines.append(f"- {cluster.title}: {line}")
    if not lines:
        return "Khong tao duoc tom tat tong the do thieu noi dung hop le."
    return "\n".join(
        [
            "Tom tat tong the (fallback extractive):",
            *lines,
        ]
    )


def prepare_messages(
    messages: list[dict],
    attachments: list[AttachmentPayload],
    available_prompt_tokens: int,
) -> list[dict]:
    enriched = [dict(message) for message in messages]
    attachment_context = build_attachment_context(attachments)

    if attachment_context:
        target_index = next(
            (index for index in range(len(enriched) - 1, -1, -1) if enriched[index]["role"] == "user"),
            None,
        )
        if target_index is None:
            enriched.append({"role": "user", "content": attachment_context})
        else:
            base = enriched[target_index]["content"].strip()
            enriched[target_index]["content"] = (
                f"{base}\n\n{attachment_context}" if base else attachment_context
            )

    return sanitize_messages(enriched, available_prompt_tokens)


def sanitize_messages(messages: list[dict], available_prompt_tokens: int) -> list[dict]:
    trimmed = []

    for message in messages:
        content = normalize_whitespace(message.get("content", ""))
        if len(content) > MAX_MESSAGE_CHARS:
            content = (
                content[: MAX_MESSAGE_CHARS // 2]
                + "\n\n[... message truncated ...]\n\n"
                + content[-MAX_MESSAGE_CHARS // 2 :]
            )
        trimmed.append({"role": message["role"], "content": content})

    if not trimmed:
        return trimmed

    # Keep the most recent messages first, but always preserve the first system message if present.
    preserved = []
    working = trimmed
    if trimmed[0]["role"] == "system":
        preserved = [trimmed[0]]
        working = trimmed[1:]

    result = []
    total_chars = sum(len(msg["content"]) for msg in preserved)

    for message in reversed(working):
        message_chars = len(message["content"])
        projected_chars = total_chars + message_chars
        projected_tokens = projected_chars // 4
        if result and (projected_chars > MAX_PROMPT_CHARS or projected_tokens > available_prompt_tokens):
            break
        result.append(message)
        total_chars = projected_chars

    if not result and working:
        latest = working[-1]
        budget_chars = min(MAX_PROMPT_CHARS, available_prompt_tokens * 4)
        content = latest["content"][:budget_chars]
        result.append({"role": latest["role"], "content": content})

    return preserved + list(reversed(result))


def normalize_whitespace(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def build_attachment_context(attachments: list[AttachmentPayload]) -> str:
    if not attachments:
        return ""

    blocks: list[str] = []

    for attachment in attachments:
        if attachment.kind == "text":
            continue
        if attachment.kind == "image":
            blocks.append(
                "\n".join(
                    [
                        f"IMAGE FILE: {attachment.path}",
                        f"SIZE: {format_size(attachment.size)}",
                        "NOTE: Anh chua duoc trich xuat noi dung pixel o backend nay.",
                    ]
                )
            )
            continue

        extracted = extract_attachment_text(
            attachment,
            MAX_ATTACHMENT_BYTES,
            MAX_ATTACHMENT_TEXT_CHARS,
        )
        if extracted:
            blocks.append(extracted)

    if not blocks:
        return ""

    return "Noi dung duoc backend trich xuat tu tep dinh kem:\n\n" + "\n\n".join(blocks)
@app.get("/gpu")
async def gpu_stats():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {"available": False}

        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 5:
                mem_used, mem_total = int(parts[2]), int(parts[3])
                gpus.append(
                    {
                        "name": parts[0],
                        "utilization": int(parts[1]),
                        "memory_used": mem_used,
                        "memory_total": mem_total,
                        "memory_percent": round(mem_used / mem_total * 100, 1),
                        "temperature": int(parts[4]),
                    }
                )
        return {"available": bool(gpus), "gpus": gpus}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


@app.get("/models")
async def list_models():
    return {"models": [{"name": model_path.name, "path": str(model_path)}]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False, loop="asyncio")
