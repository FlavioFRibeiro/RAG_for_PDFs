import fitz

from rag_genai_core.pdf_preview import highlight_quads, source_answer_text


def test_pressure_matches_complete_values_and_units_not_bare_numbers_or_citation_ids():
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 40), "1. Pressure 220KPa. Wheel diameter 220 mm. Incorrect 1220KPa.")
        evidence = page.get_text()
        quads = highlight_quads(page, evidence, "The pressure is 220 kPa. [S1]")
        assert len(quads) == 1
        assert page.get_textbox(quads[0].rect) == "220KPa"


def test_answer_and_evidence_must_both_support_a_highlight():
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 40), "Pressure 220 kPa. Torque 95 Nm.")
        assert not highlight_quads(page, "Torque 95 Nm.", "Pressure is 220 kPa. [S1]")
        assert not highlight_quads(page, page.get_text(), "Pressure is 999 kPa. [S1]")
        assert not highlight_quads(page, page.get_text(), "[S1]")
        assert not highlight_quads(page, page.get_text(), "")


def test_source_focus_excludes_other_claims_and_citation_numbers():
    answer = "Pressure: 220 kPa. [S1] Torque: 95 Nm. [S2]"
    assert "95" not in source_answer_text(answer, "S1")
    assert "220" not in source_answer_text(answer, "S2")
    assert source_answer_text(answer, "S3") == ""
    assert "Pressure" in source_answer_text("Pressure: 220 kPa. [S1], [S2]", "S2")


def test_exact_procedure_phrase_can_be_highlighted_without_highlighting_whole_chunk():
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 40), "Place the motorcycle on the center stand.")
        page.insert_text((40, 70), "Unrelated maintenance procedure and warning information.")
        quads = highlight_quads(page, page.get_text(), "Place the motorcycle on the center stand. [S1]")
        assert len(quads) == 1
        assert quads[0].rect.y1 < 50


def test_partial_range_is_not_highlighted_as_a_complete_specification():
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 40), "Chain slack: 40.0-45.0 mm.")
        assert not highlight_quads(page, page.get_text(), "The value is 45 mm.")
        quads = highlight_quads(page, page.get_text(), "Chain slack: 40-45 mm. [S1]")
        assert len(quads) == 1
        assert page.get_textbox(quads[0].rect) == "40.0-45.0 mm"
