from dataclasses import replace
from pathlib import Path

import pytest

from rag_genai_core.catalog import load_catalog, validate_catalog
from rag_genai_core.config import Settings
from rag_genai_core.exceptions import CatalogError


@pytest.mark.manuals
def test_real_catalog_identifies_both_manuals() -> None:
    settings = Settings()
    records = load_catalog(settings.catalog_path, settings.root_dir)
    validate_catalog(records.values())
    assert set(records) == {
        "yamaha-tenere-700-xtz690d-2022-04-en",
        "voge-500ds-en-unknown-edition",
    }


@pytest.mark.manuals
def test_missing_file_and_hash_mismatch_are_clear(tmp_path: Path) -> None:
    settings = Settings()
    record = next(iter(load_catalog(settings.catalog_path, settings.root_dir).values()))
    with pytest.raises(CatalogError, match="missing"):
        validate_catalog([replace(record, pdf_path=tmp_path / "missing.pdf")])
    fake = tmp_path / "manual.pdf"
    fake.write_bytes(b"not the manual")
    with pytest.raises(CatalogError, match="Hash mismatch"):
        validate_catalog([replace(record, pdf_path=fake)])
