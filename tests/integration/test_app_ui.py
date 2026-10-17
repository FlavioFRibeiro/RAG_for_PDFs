import fitz
import pytest
from streamlit.testing.v1 import AppTest

from rag_genai_core.config import Settings
from rag_genai_core.pdf_preview import highlight_quads


@pytest.mark.manuals
def test_user_can_select_voge_without_upload_or_processing() -> None:
    app = AppTest.from_file("apps/streamlit_app.py").run(timeout=20)
    app.selectbox[0].select("Voge 500DS").run(timeout=20)
    assert not list(app.exception)
    assert any("Voge 500DS User Manual" in item.value for item in app.info)
    assert not list(app.get("file_uploader"))
    assert all(button.label != "Process" for button in app.button)


@pytest.mark.manuals
def test_cited_manual_text_can_be_located_for_highlighting() -> None:
    path = Settings().root_dir / "docs" / "VOGE_500_DS.pdf"
    with fitz.open(path) as document:
        rectangles = highlight_quads(
            document[61],
            "Front wheel | 220KPa | 250KPa\nRear wheel | 220KPa | 250KPa",
            "Front and rear: 220 kPa for a rider; 250 kPa with a passenger.",
        )
    assert rectangles


@pytest.mark.manuals
def test_existing_chat_sources_render_full_page_without_raw_chunk_text():
    path = Settings().root_dir / "docs" / "VOGE_500_DS.pdf"
    with fitz.open(path) as doc:
        evidence = doc[61].get_text()
    app = AppTest.from_file("apps/streamlit_app.py").run(timeout=20)
    app.selectbox[0].select("Voge 500DS").run(timeout=20)
    app.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Rider: 220 kPa. Rider and passenger: 250 kPa. [S1]",
            "sources": [
                {
                    "source_id": "S1",
                    "chunk_id": "test-pressure",
                    "physical_page": 62,
                    "printed_page": "58",
                    "evidence": evidence,
                }
            ],
        }
    ]
    # Streamlit 1.28's test driver needs the display label again on subsequent reruns.
    app.selectbox[0].select("Voge 500DS").run(timeout=20)
    assert not list(app.exception)
    assert not list(app.code)
    assert any("Full manual page" in item.value for item in app.caption)
    assert not any("cropped" in item.value for item in app.caption)
