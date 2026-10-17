import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.manuals
def test_user_can_select_voge_without_upload_or_processing() -> None:
    app = AppTest.from_file("apps/streamlit_app.py").run(timeout=20)
    app.selectbox[0].select("Voge 500DS").run(timeout=20)
    assert not list(app.exception)
    assert any("Voge 500DS User Manual" in item.value for item in app.info)
    assert not list(app.get("file_uploader"))
    assert all(button.label != "Process" for button in app.button)
