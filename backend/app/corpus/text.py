import html
import io
import logging
import re
import zipfile

from pypdf import PdfReader

from app.core.config import settings

logger = logging.getLogger(__name__)

PREAMBLE = "Encabezado y considerandos"
WHOLE_TEXT = "Texto completo"
ORDINALS = "PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|NOVENO|D[ÉE]CIMO"
ARTICLE_START = re.compile(
    r"^(?P<label>ART[ÍI]CULO\s+(?:\d+[A-Za-z]?(?:\s*(?:o|º|°)\b)?|[ÚU]NICO)"
    r"(?:\s+(?:BIS|TRANSITORIO))?)\.?\s*(?P<rest>.*)$"
)
# Extracted documents use mixed case and ordinal words; the punctuation after the label keeps
# lines that merely start by citing an article from opening a new section.
DOCUMENT_ARTICLE_START = re.compile(
    rf"^(?P<label>ART[ÍI]CULO\s+(?:\d+[A-Za-z]?(?:\s*(?:o|º|°))?|[ÚU]NICO|{ORDINALS})"
    r"(?:\s+(?:BIS|TRANSITORIO))?)\s*[\.\-–:]\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
ANNEX_START = re.compile(r"^(?P<label>ANEXO(?:\s+[\dIVXLC]+)?)\b\.?\s*(?P<rest>.*)$")


def decode_html(content: bytes) -> str:
    # Pages declare UTF-8 but some are served as Latin-1.
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("cp1252", errors="replace")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\x00", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def document_text(content: bytes, kind: str) -> tuple[str, str]:
    """Return (text, method) for a downloaded original; empty text for unsupported formats."""
    if kind == "pdf":
        return pdf_text(content)
    if kind == "docx":
        return docx_text(content), "docx"
    return "", "none"


def pdf_text(content: bytes) -> tuple[str, str]:
    """Return (text, method) using the PDF text layer, or OCR when the layer is too thin."""
    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception:
        logger.warning("unreadable PDF", exc_info=True)
        return "", "pdf"
    text = normalize_whitespace("\n\n".join(pages))
    if pages and len(text) / len(pages) >= settings.OCR_MIN_CHARS_PER_PAGE:
        return text, "pdf"
    try:
        return normalize_whitespace(ocr_pdf(content)), "ocr"
    except Exception:
        logger.exception("OCR failed; keeping the thin text layer")
        return text, "pdf"


def ocr_pdf(content: bytes) -> str:
    import pypdfium2
    import pytesseract

    document = pypdfium2.PdfDocument(content)
    try:
        texts = []
        for index, page in enumerate(document):
            if index >= settings.OCR_MAX_PAGES:
                break
            image = page.render(scale=300 / 72).to_pil()
            texts.append(pytesseract.image_to_string(image, lang=settings.OCR_LANGUAGE))
        return "\n\n".join(texts)
    finally:
        document.close()


def docx_text(content: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, KeyError):
        logger.warning("unreadable docx")
        return ""
    paragraphs = [
        "".join(html.unescape(run) for run in re.findall(r"<w:t(?:\s[^>]*)?>([^<]*)</w:t>", part))
        for part in xml.split("</w:p>")
    ]
    return normalize_whitespace("\n".join(paragraphs))


def split_sections(lines: list[str], extracted: bool = False) -> list[tuple[str, str]]:
    """Group lines into (label, body) by article and annex headings, after a preamble."""
    article_start = DOCUMENT_ARTICLE_START if extracted else ARTICLE_START
    sections: list[tuple[str, list[str]]] = [(PREAMBLE, [])]
    for line in lines:
        article = article_start.match(line)
        annex = ANNEX_START.match(line) if not article else None
        if article:
            sections.append((article_label(article["label"]), [line]))
        elif annex and len(line) < 200:
            sections.append((annex["label"].title(), [line]))
        else:
            sections[-1][1].append(line)
    return [(label, "\n".join(body)) for label, body in sections if body]


def article_label(raw: str) -> str:
    label = re.sub(r"\s+", " ", raw).strip()
    label = re.sub(r"(\d+)\s*(?:o|º|°)\b", r"\1", label)
    return re.sub(r"^Articulo", "Artículo", label.capitalize()).replace("Unico", "Único")


def section_passages(label: str, body: str, max_chars: int) -> list[tuple[str, str]]:
    parts = split_passages(body, max_chars)
    if len(parts) == 1:
        return [(label, parts[0])]
    return [(f"{label} (parte {index})", part) for index, part in enumerate(parts, start=1)]


def document_passages(text: str, max_chars: int) -> list[tuple[str, str]]:
    """Article passages for text extracted from a PDF or DOCX."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    sections = split_sections(lines, extracted=True)
    if len(sections) == 1:
        sections = [(WHOLE_TEXT, sections[0][1])]
    return [
        passage for label, body in sections for passage in section_passages(label, body, max_chars)
    ]


def split_passages(text: str, max_chars: int) -> list[str]:
    """Split long text on paragraph boundaries into passages of at most max_chars."""
    if len(text) <= max_chars:
        return [text]
    passages: list[str] = []
    current = ""
    for paragraph in text.split("\n"):
        while len(paragraph) > max_chars:
            cut = paragraph.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if current:
                passages.append(current)
                current = ""
            passages.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        candidate = f"{current}\n{paragraph}" if current else paragraph
        if len(candidate) > max_chars:
            passages.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        passages.append(current)
    return [passage for passage in passages if passage.strip()]
