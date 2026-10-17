from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Optional

import fitz

from .schemas import Chunk, ManualRecord


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _table_to_markdown(rows: List[List[Optional[str]]]) -> str:
    cleaned = [[normalize_text(cell or "") for cell in row] for row in rows]
    width = max((len(row) for row in cleaned), default=0)
    if width == 0:
        return ""
    padded = [row + [""] * (width - len(row)) for row in cleaned]
    return "[TABLE]\n" + "\n".join(" | ".join(row) for row in padded) + "\n[/TABLE]"


def extract_page_text(page: fitz.Page) -> str:
    # The manuals already define a usable content stream order. Re-sorting by
    # coordinates interleaves columns in the Yamaha tables, so preserve it and
    # append spatially reconstructed tables below.
    body = normalize_text(page.get_text("text"))
    tables = []
    try:
        for table in page.find_tables().tables:
            rendered = _table_to_markdown(table.extract())
            if rendered:
                tables.append(rendered)
    except Exception:
        tables = []
    return normalize_text("\n\n".join([body] + tables))


def printed_page_label(page: fitz.Page) -> Optional[str]:
    label = (page.get_label() or "").strip()
    if label and label != str(page.number + 1):
        return label
    height = page.rect.height
    footer = normalize_text(page.get_text("text", clip=fitz.Rect(0, height * 0.84, page.rect.width, height)))
    candidates = re.findall(r"(?<![\d-])(?:[1-9]\d{0,2}|[1-9]\d?-[1-9]\d?)(?![\d-])", footer)
    return candidates[-1] if candidates else None


def _split_text(text: str, size: int, overlap: int) -> Iterable[str]:
    if size <= overlap or overlap < 0:
        raise ValueError("chunk_size must be greater than chunk_overlap >= 0")
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind("\n", start + size // 2, end), text.rfind(". ", start + size // 2, end))
            if boundary > start:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end >= len(text):
            break
        start = end - overlap


def extract_chunks(record: ManualRecord, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    with fitz.open(record.pdf_path) as document:
        for page_index, page in enumerate(document):
            page_text = extract_page_text(page)
            if not page_text:
                continue
            label = printed_page_label(page)
            for ordinal, text in enumerate(_split_text(page_text, chunk_size, chunk_overlap)):
                stable = f"{record.manual_id}|{page_index + 1}|{ordinal}|{normalize_text(text)}"
                chunk_id = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        manual_id=record.manual_id,
                        motorcycle_id=record.motorcycle_id,
                        manual_name=record.title,
                        physical_page=page_index + 1,
                        printed_page=label,
                        text=text,
                    )
                )
    return chunks
