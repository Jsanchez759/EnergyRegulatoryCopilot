import io
import logging
import re

from pypdf import PdfReader

from app.core.config import settings

logger = logging.getLogger(__name__)


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


def pdf_text(content: bytes) -> tuple[str, str]:
    """Return (text, method) using the PDF text layer, or OCR when the layer is too thin."""
    reader = PdfReader(io.BytesIO(content), strict=False)
    pages = [page.extract_text() or "" for page in reader.pages]
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
        for page in document:
            image = page.render(scale=300 / 72).to_pil()
            texts.append(pytesseract.image_to_string(image, lang=settings.OCR_LANGUAGE))
        return "\n\n".join(texts)
    finally:
        document.close()


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
