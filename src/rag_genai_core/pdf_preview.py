"""Locate answer-supported passages and render full manual pages without saving edits."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher
from typing import List, Set, Tuple

import fitz

_NUMBER = r"\d+(?:[.,]\d+)?"
_UNITS = (
    r"kpa|mpa|psi|bar|mm|cm|km|kg|ml|litres?|liters?|l|"
    r"n\s*[·.]?\s*m|lb\s*[·.]?\s*ft|rpm|years?|months?|hours?|"
    r"°\s*[cf]|inches|inch|in"
)
_QUANTITY = re.compile(
    rf"(?<![\w.,])(?P<values>{_NUMBER}(?:\s*(?:[-–—−\ufffd/]|to|and)\s*{_NUMBER})*)"
    rf"\s*(?P<unit>{_UNITS})(?!\w)",
    re.IGNORECASE,
)
_CITATION = re.compile(r"\[S\d+\]")


@dataclass(frozen=True)
class PagePreview:
    image: bytes
    highlight_count: int


def source_answer_text(answer: str, source_id: str) -> str:
    """Use the claim preceding this citation, excluding other sources' claims."""
    if not _CITATION.search(answer):
        return answer
    pieces = []
    for match in re.finditer(r"(.*?)(\[S\d+\](?:[ ,]*\[S\d+\])*)", answer, re.DOTALL):
        if f"[{source_id}]" in _CITATION.findall(match[2]):
            pieces.append(match[1])
    return "\n".join(pieces)


def _keys(match: re.Match) -> Set[Tuple[Decimal, str]]:
    unit = re.sub(r"[\s·.]", "", match["unit"].lower())
    aliases = {"litre": "l", "litres": "l", "liter": "l", "liters": "l", "inch": "in", "inches": "in"}
    unit = aliases.get(unit, unit)
    if unit.rstrip("s") in {"year", "month", "hour"}:
        unit = unit.rstrip("s")
    return {(Decimal(number.replace(",", ".")), unit) for number in re.findall(_NUMBER, match["values"])}


def _quantity_keys(text: str) -> Set[Tuple[Decimal, str]]:
    return {key for match in _QUANTITY.finditer(text) for key in _keys(match)}


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def highlight_quads(page: fitz.Page, evidence: str, answer: str) -> List[fitz.Quad]:
    """Match complete quantities, or short verbatim phrases, in answer AND evidence.

    There is deliberately no fallback to every number or every line in a chunk.
    Quads retain text direction on rotated PDF pages.
    """
    answer = _CITATION.sub("", answer)
    if not answer.strip() or not evidence.strip():
        return []
    supported = _quantity_keys(answer) & _quantity_keys(evidence)
    text = page.get_text("text")
    candidates = []
    if _QUANTITY.search(answer):
        for match in _QUANTITY.finditer(text):
            if _keys(match) <= supported:
                # Require the original page spelling within the retrieved evidence.
                if _compact(match[0]) in _compact(evidence):
                    candidates.append(match[0])
    else:
        # Non-numeric procedures: only short exact shared phrases, never an entire chunk.
        answer_words = re.findall(r"[\w'-]+", answer.lower())
        evidence_words = list(re.finditer(r"[\w'-]+", evidence.lower()))
        matcher = SequenceMatcher(None, answer_words, [word[0] for word in evidence_words], autojunk=False)
        blocks = sorted(matcher.get_matching_blocks(), key=lambda block: block.size, reverse=True)
        for block in blocks[:2]:
            if block.size >= 4:
                start = evidence_words[block.b].start()
                end = evidence_words[block.b + min(block.size, 12) - 1].end()
                phrase = evidence[start:end]
                if len(phrase) >= 20:
                    candidates.append(phrase)
    quads = []
    seen = set()
    words = page.get_text("words")
    for candidate in dict.fromkeys(candidates):
        for quad in page.search_for(candidate, quads=True):
            # search_for also finds substrings (220KPa within 1220KPa).
            # Reject a hit when an overlapping numeric token has different digits.
            numbers = set(re.findall(_NUMBER, candidate))
            if numbers and any(
                fitz.Rect(word[:4]).intersects(quad.rect) and not set(re.findall(_NUMBER, word[4])) <= numbers
                for word in words
            ):
                continue
            key = tuple(round(coordinate, 3) for point in quad for coordinate in point)
            if key not in seen:
                quads.append(quad)
                seen.add(key)
    return quads


def render_source_page(
    pdf_path: str, physical_page: int, evidence: str, answer: str, dpi: int
) -> PagePreview:
    with fitz.open(pdf_path) as document:
        page = document.load_page(physical_page - 1)
        quads = highlight_quads(page, evidence, answer)
        for quad in quads:
            annotation = page.add_highlight_annot(quad)
            annotation.set_colors(stroke=(1, 1, 0))
            annotation.set_opacity(0.45)
            annotation.update()
        if quads:
            page = document.reload_page(page)
        # Never crop or change the CropBox / rotation: keep all headings and context visible.
        return PagePreview(
            image=page.get_pixmap(dpi=dpi, alpha=False, annots=True).tobytes("png"),
            highlight_count=len(quads),
        )
