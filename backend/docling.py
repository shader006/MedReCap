import base64
import binascii
import importlib
import importlib.metadata
import os
import tempfile
import threading
from pathlib import Path

__path__ = [str(importlib.metadata.distribution("docling").locate_file("docling"))]
_docling_document_converter = importlib.import_module("docling.document_converter")
DocumentConverter = _docling_document_converter.DocumentConverter
PdfFormatOption = _docling_document_converter.PdfFormatOption
WordFormatOption = _docling_document_converter.WordFormatOption
InputFormat = importlib.import_module("docling.datamodel.base_models").InputFormat
PdfPipelineOptions = importlib.import_module("docling.datamodel.pipeline_options").PdfPipelineOptions

MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
MAX_ATTACHMENT_TEXT_CHARS = 16000
PDF_DO_FORMULA_ENRICHMENT = os.getenv("DOCLING_PDF_FORMULA_ENRICHMENT", "1").strip().lower() in {"1", "true", "yes", "on"}

DOCLING_DIRECT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".md",
    ".csv",
    ".xml",
}
WORD_CONVERTIBLE_EXTENSIONS = {".doc", ".docm", ".rtf", ".odt"}
EXCEL_CONVERTIBLE_EXTENSIONS = {".xls", ".xlsm", ".xlsb", ".ods"}
POWERPOINT_CONVERTIBLE_EXTENSIONS = {".ppt", ".pptm", ".odp"}
DOCLING_EXTENSIONS = (
    DOCLING_DIRECT_EXTENSIONS
    | WORD_CONVERTIBLE_EXTENSIONS
    | EXCEL_CONVERTIBLE_EXTENSIONS
    | POWERPOINT_CONVERTIBLE_EXTENSIONS
)

docling_lock = threading.Lock()
office_lock = threading.Lock()
docling_converter: DocumentConverter | None = None


def extract_attachment_text(
    attachment,
    max_attachment_bytes: int = MAX_ATTACHMENT_BYTES,
    max_attachment_text_chars: int = MAX_ATTACHMENT_TEXT_CHARS,
) -> str:
    path = attachment.path or attachment.name
    suffix = Path(path).suffix.lower()

    if attachment.content:
        return attachment.content

    if suffix not in DOCLING_EXTENSIONS:
        return "\n".join(
            [
                f"FILE: {path}",
                f"SIZE: {format_size(attachment.size)}",
                f"NOTE: Dinh dang {suffix or '(khong ro)'} chua duoc Docling xu ly trong backend.",
            ]
        )

    if attachment.size > max_attachment_bytes:
        return "\n".join(
            [
                f"FILE: {path}",
                f"SIZE: {format_size(attachment.size)}",
                f"NOTE: Tep vuot qua gioi han {format_size(max_attachment_bytes)} nen khong duoc trich xuat.",
            ]
        )

    if attachment.encoding != "base64" or not attachment.data:
        return "\n".join(
            [
                f"FILE: {path}",
                f"SIZE: {format_size(attachment.size)}",
                "NOTE: Khong nhan duoc du lieu file de trich xuat.",
            ]
        )

    try:
        raw_bytes = base64.b64decode(attachment.data)
    except (ValueError, binascii.Error):
        return "\n".join(
            [
                f"FILE: {path}",
                f"SIZE: {format_size(attachment.size)}",
                "NOTE: Du lieu base64 cua file khong hop le.",
            ]
        )

    if len(raw_bytes) > max_attachment_bytes:
        return "\n".join(
            [
                f"FILE: {path}",
                f"SIZE: {format_size(len(raw_bytes))}",
                f"NOTE: Tep vuot qua gioi han {format_size(max_attachment_bytes)} nen khong duoc trich xuat.",
            ]
        )

    try:
        text, conversion_note = convert_with_docling(raw_bytes, suffix)
    except Exception as exc:
        return "\n".join(
            [
                f"FILE: {path}",
                f"SIZE: {format_size(len(raw_bytes))}",
                f"NOTE: Docling khong trich xuat duoc noi dung ({exc}).",
            ]
        )

    text = normalize_whitespace(text)
    if not text:
        return "\n".join(
            [
                f"FILE: {path}",
                f"SIZE: {format_size(len(raw_bytes))}",
                "NOTE: Docling khong tra ve noi dung van ban.",
            ]
        )

    truncated = len(text) > max_attachment_text_chars
    excerpt = text[:max_attachment_text_chars]

    return "\n".join(
        [
            f"FILE: {path}",
            f"SIZE: {format_size(len(raw_bytes))}",
            conversion_note or "",
            truncated and "NOTE: Noi dung da bi cat bot de vua context." or "",
            "CONTENT:",
            excerpt,
        ]
    ).strip()


def convert_with_docling(raw_bytes: bytes, suffix: str) -> tuple[str, str]:
    converter = get_docling_converter()
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        temp_path = temp_dir / f"input{suffix}"
        temp_path.write_bytes(raw_bytes)
        docling_path, conversion_note = prepare_docling_input(temp_path, suffix, temp_dir)
        try:
            if suffix == ".pdf":
                text, batch_note = convert_pdf_with_docling_batches(converter, docling_path, temp_dir)
                joined_note = "\n".join(part for part in [conversion_note, batch_note] if part)
                return text, joined_note

            return convert_docling_path_to_markdown(converter, docling_path), conversion_note
        except Exception:
            raise


def prepare_docling_input(input_path: Path, suffix: str, temp_dir: Path) -> tuple[Path, str]:
    if suffix in DOCLING_DIRECT_EXTENSIONS:
        return input_path, ""

    if suffix in WORD_CONVERTIBLE_EXTENSIONS:
        output_path = temp_dir / "converted.pdf"
        convert_with_word_to_pdf(input_path, output_path)
        return output_path, f"NOTE: Tep {suffix} da duoc chuyen sang .pdf truoc khi Docling trich xuat."

    if suffix in EXCEL_CONVERTIBLE_EXTENSIONS:
        output_path = temp_dir / "converted.xlsx"
        convert_with_excel(input_path, output_path)
        return output_path, f"NOTE: Tep {suffix} da duoc chuyen sang .xlsx truoc khi Docling trich xuat."

    if suffix in POWERPOINT_CONVERTIBLE_EXTENSIONS:
        output_path = temp_dir / "converted.pptx"
        convert_with_powerpoint(input_path, output_path)
        return output_path, f"NOTE: Tep {suffix} da duoc chuyen sang .pptx truoc khi Docling trich xuat."

    raise ValueError(f"Dinh dang {suffix or '(khong ro)'} chua duoc ho tro.")


def convert_with_word_to_pdf(input_path: Path, output_path: Path) -> None:
    """Convert Word documents to PDF via COM automation.

    Using PDF as the intermediate format ensures:
    - Docling processes via its PDF pipeline (with formula enrichment)
    - All math equations (OLE, OMML, images) are preserved in the PDF
    - Full document content is retained (no truncation from .docm->.docx macro stripping)
    """
    with office_lock:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        app = None
        document = None
        try:
            app = win32com.client.DispatchEx("Word.Application")
            app.Visible = False
            app.DisplayAlerts = 0
            document = app.Documents.Open(str(input_path), ReadOnly=True)
            document.SaveAs2(str(output_path), FileFormat=17)  # 17 = wdFormatPDF
        finally:
            if document is not None:
                document.Close(False)
            if app is not None:
                app.Quit()
            pythoncom.CoUninitialize()


def convert_with_word(input_path: Path, output_path: Path) -> None:
    """Convert Word documents to .docx via COM automation (legacy, kept for compatibility)."""
    with office_lock:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        app = None
        document = None
        try:
            app = win32com.client.DispatchEx("Word.Application")
            app.Visible = False
            app.DisplayAlerts = 0
            document = app.Documents.Open(str(input_path), ReadOnly=True)
            document.SaveAs2(str(output_path), FileFormat=12)  # 12 = wdFormatXMLDocument (.docx)
        finally:
            if document is not None:
                document.Close(False)
            if app is not None:
                app.Quit()
            pythoncom.CoUninitialize()


def convert_with_excel(input_path: Path, output_path: Path) -> None:
    with office_lock:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        app = None
        workbook = None
        try:
            app = win32com.client.DispatchEx("Excel.Application")
            app.Visible = False
            app.DisplayAlerts = False
            workbook = app.Workbooks.Open(str(input_path))
            workbook.SaveAs(str(output_path), FileFormat=51)
        finally:
            if workbook is not None:
                workbook.Close(False)
            if app is not None:
                app.Quit()
            pythoncom.CoUninitialize()


def convert_with_powerpoint(input_path: Path, output_path: Path) -> None:
    with office_lock:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        app = None
        presentation = None
        try:
            app = win32com.client.DispatchEx("PowerPoint.Application")
            presentation = app.Presentations.Open(str(input_path), WithWindow=False)
            presentation.SaveAs(str(output_path), 24)
        finally:
            if presentation is not None:
                presentation.Close()
            if app is not None:
                app.Quit()
            pythoncom.CoUninitialize()


def get_docling_converter() -> DocumentConverter:
    global docling_converter
    if docling_converter is None:
        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = False
        pdf_pipeline_options.force_backend_text = True
        pdf_pipeline_options.do_formula_enrichment = PDF_DO_FORMULA_ENRICHMENT

        docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
                InputFormat.DOCX: WordFormatOption(),
            }
        )
    return docling_converter


def convert_docling_path_to_markdown(converter: DocumentConverter, input_path: Path) -> str:
    with docling_lock:
        result = converter.convert(input_path)
    return result.document.export_to_markdown(
        strict_text=True,
        image_placeholder="",
        compact_tables=True,
        include_annotations=False,
    )


def convert_pdf_with_docling_batches(
    converter: DocumentConverter,
    input_path: Path,
    temp_dir: Path,
) -> tuple[str, str]:
    page_count = get_pdf_page_count(input_path)
    if page_count is None or page_count <= 1:
        return convert_docling_path_to_markdown(converter, input_path), ""

    try:
        from PyPDF2 import PdfReader, PdfWriter
    except ImportError:
        return convert_docling_path_to_markdown(converter, input_path), ""

    reader = PdfReader(str(input_path))
    initial_batch_count = get_initial_pdf_batch_count(page_count)
    initial_ranges = build_initial_page_ranges(page_count, initial_batch_count)
    batch_markdowns: list[str] = []
    batch_ranges: list[str] = []

    for start, end in initial_ranges:
        sub_markdowns, sub_ranges = convert_pdf_page_range_with_retry(
            converter=converter,
            reader=reader,
            temp_dir=temp_dir,
            start=start,
            end=end,
        )
        batch_markdowns.extend(sub_markdowns)
        batch_ranges.extend(sub_ranges)

    combined_markdown = "\n\n".join(part for part in batch_markdowns if part).strip()
    note = (
        f"NOTE: PDF da duoc tach thanh {len(batch_ranges)} batch de Docling trich xuat "
        f"(khoi tao {initial_batch_count} batch theo {page_count} trang; cac batch thanh cong: {', '.join(batch_ranges)})."
        if batch_ranges
        else ""
    )
    return combined_markdown, note


def get_initial_pdf_batch_count(page_count: int) -> int:
    pages_per_batch = 10
    return max(1, (page_count + pages_per_batch - 1) // pages_per_batch)


def build_initial_page_ranges(page_count: int, batch_count: int) -> list[tuple[int, int]]:
    batch_count = max(1, min(batch_count, page_count))
    base_size, remainder = divmod(page_count, batch_count)
    ranges: list[tuple[int, int]] = []
    start = 0
    for batch_index in range(batch_count):
        current_size = base_size + (1 if batch_index < remainder else 0)
        if current_size <= 0:
            continue
        end = start + current_size
        ranges.append((start, end))
        start = end
    return ranges


def convert_pdf_page_range_with_retry(
    converter: DocumentConverter,
    reader,
    temp_dir: Path,
    start: int,
    end: int,
) -> tuple[list[str], list[str]]:
    batch_path = write_pdf_page_range(reader, temp_dir, start, end)

    try:
        markdown = convert_docling_path_to_markdown(converter, batch_path).strip()
    except Exception as exc:
        page_count = end - start
        if page_count <= 1:
            raise RuntimeError(
                f"Docling that bai voi trang {start + 1} ({exc})"
            ) from exc

        mid = start + (page_count // 2)
        left_markdowns, left_ranges = convert_pdf_page_range_with_retry(
            converter=converter,
            reader=reader,
            temp_dir=temp_dir,
            start=start,
            end=mid,
        )
        right_markdowns, right_ranges = convert_pdf_page_range_with_retry(
            converter=converter,
            reader=reader,
            temp_dir=temp_dir,
            start=mid,
            end=end,
        )
        return left_markdowns + right_markdowns, left_ranges + right_ranges

    return ([markdown] if markdown else []), [format_page_range(start, end)]


def write_pdf_page_range(reader, temp_dir: Path, start: int, end: int) -> Path:
    from PyPDF2 import PdfWriter

    writer = PdfWriter()
    for page_index in range(start, end):
        writer.add_page(reader.pages[page_index])

    batch_path = temp_dir / f"batch_{start + 1}_{end}.pdf"
    with batch_path.open("wb") as batch_file:
        writer.write(batch_file)
    return batch_path


def format_page_range(start: int, end: int) -> str:
    if end - start == 1:
        return str(start + 1)
    return f"{start + 1}-{end}"


def get_pdf_page_count(input_path: Path) -> int | None:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        return None

    try:
        reader = PdfReader(str(input_path))
    except Exception:
        return None
    return len(reader.pages)


def normalize_whitespace(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def format_size(bytes_count: int) -> str:
    if bytes_count < 1024:
        return f"{bytes_count} B"
    if bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f} KB"
    return f"{bytes_count / (1024 * 1024):.1f} MB"
