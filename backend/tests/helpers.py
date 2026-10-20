import hashlib
import math
import re

from app.core.config import settings


def build_text_pdf(lines: list[str]) -> bytes:
    def escape(value: str) -> str:
        return re.sub(r"([\\()])", r"\\\1", value)

    content = "\n".join(
        ["BT /F1 12 Tf 72 720 Td 16 TL", *[f"({escape(line)}) '" for line in lines], "ET"]
    )
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content.encode('latin-1'))} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    pdf = "%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf.encode("latin-1")))
        pdf += f"{number} 0 obj\n{body}\nendobj\n"
    xref = len(pdf.encode("latin-1"))
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    pdf += "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return pdf.encode("latin-1")


def bag_of_words_embedding(text: str) -> list[float]:
    """Deterministic stand-in for a real embedding: texts sharing words point the same way."""
    vector = [0.0] * settings.EMBEDDING_DIMENSIONS
    for word in re.findall(r"\w+", text.casefold()):
        digest = hashlib.sha256(word.encode()).digest()
        vector[int.from_bytes(digest[:4], "big") % len(vector)] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]
