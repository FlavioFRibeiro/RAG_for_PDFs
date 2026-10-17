import hashlib

import fitz
import numpy as np
import pytest

from rag_genai_core.config import Settings
from rag_genai_core.pdf_preview import highlight_quads, render_source_page


@pytest.mark.manuals
@pytest.mark.parametrize(
    "filename,page_number,answer,expected_matches",
    [
        ("VOGE_500_DS.pdf", 62, "Rider: 220 kPa. Rider and passenger: 250 kPa. [S1]", 4),
        ("Manual Tenere 700.pdf", 86, "Distance A: 40.0-45.0 mm. [S1]", 1),
    ],
)
def test_entire_original_page_is_preserved_and_only_answer_regions_change(
    filename, page_number, answer, expected_matches
):
    path = Settings().root_dir / "docs" / filename
    initial_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with fitz.open(path) as doc:
        page = doc[page_number - 1]
        evidence = page.get_text()
        original = page.get_pixmap(dpi=72, alpha=False)
        quads = highlight_quads(page, evidence, answer)
        assert len(quads) == expected_matches
        permitted = np.zeros((original.height, original.width), dtype=bool)
        for quad in quads:
            rect = quad.rect * page.rotation_matrix
            # Highlight curves can extend slightly beyond the glyph boxes.
            permitted[
                max(0, int(rect.y0) - 5) : int(rect.y1) + 6, max(0, int(rect.x0) - 5) : int(rect.x1) + 6
            ] = True
        before = np.frombuffer(original.samples, dtype=np.uint8).reshape(original.height, original.width, 3)
    preview = render_source_page(str(path), page_number, evidence, answer, 72)
    rendered = fitz.Pixmap(preview.image)
    assert (rendered.width, rendered.height) == (original.width, original.height)
    after = np.frombuffer(rendered.samples, dtype=np.uint8).reshape(rendered.height, rendered.width, 3)
    # Transparency blending can change antialiased text elsewhere by 1-2 levels.
    changed = np.max(np.abs(before.astype(int) - after.astype(int)), axis=2) > 2
    assert changed.any()
    assert not changed[~permitted].any()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == initial_hash


@pytest.mark.manuals
def test_voge_marks_only_pressure_cells_with_units_even_with_full_page_evidence():
    path = Settings().root_dir / "docs" / "VOGE_500_DS.pdf"
    with fitz.open(path) as doc:
        page = doc[61]
        quads = highlight_quads(page, page.get_text(), "220 kPa and 250 kPa. [S1]")
        assert len(quads) == 4
        assert all(190 < quad.rect.y0 < 228 for quad in quads)
        assert all("kpa" in page.get_textbox(quad.rect).lower() for quad in quads)


@pytest.mark.manuals
def test_unmatched_answer_shows_original_full_page_without_fallback_highlights():
    path = Settings().root_dir / "docs" / "VOGE_500_DS.pdf"
    with fitz.open(path) as doc:
        page = doc[61]
        evidence = page.get_text()
        original = page.get_pixmap(dpi=72, alpha=False).tobytes("png")
    preview = render_source_page(str(path), 62, evidence, "999 kPa. [S1]", 72)
    assert preview.highlight_count == 0
    assert preview.image == original
