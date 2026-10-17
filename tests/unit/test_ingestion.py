import fitz
import pytest

from rag_genai_core.config import Settings
from rag_genai_core.ingestion import extract_page_text


@pytest.mark.manuals
def test_voge_pressure_table_preserves_load_and_wheel_relationships() -> None:
    path = Settings().root_dir / "docs" / "VOGE_500_DS.pdf"
    with fitz.open(path) as document:
        text = extract_page_text(document[61])
    assert "Driver | Driver & Passenger" in text
    assert "Front wheel | 220KPa | 250KPa" in text
    assert "Rear wheel | 220KPa | 250KPa" in text


@pytest.mark.manuals
def test_yamaha_chain_page_preserves_specification_and_condition() -> None:
    path = Settings().root_dir / "docs" / "Manual Tenere 700.pdf"
    with fitz.open(path) as document:
        text = extract_page_text(document[85])
    assert "40.0" in text and "45.0 mm" in text
    assert "there should be no weight" in text.lower()
