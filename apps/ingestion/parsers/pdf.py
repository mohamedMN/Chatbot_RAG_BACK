# backend/apps/ingestion/parsers/pdf.py
from typing import List
from io import BytesIO
from pypdf import PdfReader
from apps.ingestion.schema import Section


def _first_line_as_subject(txt: str, fallback: str) -> str:
    for line in (txt or "").splitlines():
        s = line.strip()
        if len(s) >= 3:
            return s[:120]
    return fallback


def parse_pdf_bytes(data: bytes, filename: str) -> List[Section]:
    reader = PdfReader(BytesIO(data))
    out: List[Section] = []
    ordinal = 0
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        subject = _first_line_as_subject(text, fallback=f"Page {i}")
        source = f"{filename}#page={i}"
        out.append(Section(subject=subject, source=source,
                   content=text, ordinal=ordinal))
        ordinal += 1
    return out
