from dataclasses import replace

import pytest

from rag_genai_core.catalog import load_catalog
from rag_genai_core.config import Settings
from rag_genai_core.exceptions import IndexNotReadyError
from rag_genai_core.schemas import Chunk
from rag_genai_core.storage import build_index, index_fingerprint, load_index


def test_fingerprint_changes_with_pdf_or_chunk_configuration() -> None:
    settings = Settings()
    record = next(iter(load_catalog(settings.catalog_path, settings.root_dir).values()))
    baseline = index_fingerprint(record, settings)
    assert baseline != index_fingerprint(replace(record, sha256="0" * 64), settings)
    assert baseline != index_fingerprint(record, replace(settings, chunk_size=settings.chunk_size + 1))


def test_compatible_index_is_reused_without_embedding_again(tmp_path) -> None:
    base = Settings()
    settings = replace(base, artifacts_dir=tmp_path)
    record = next(iter(load_catalog(settings.catalog_path, settings.root_dir).values()))
    chunk = Chunk("c1", record.manual_id, record.motorcycle_id, record.title, 1, None, "content")
    calls = 0

    def embed(texts):
        nonlocal calls
        calls += 1
        return [[1.0, 0.0] for _ in texts]

    build_index(record, [chunk], settings, embed)
    build_index(record, [chunk], settings, embed)
    assert calls == 1


def test_missing_index_has_actionable_command(tmp_path) -> None:
    base = Settings()
    settings = replace(base, artifacts_dir=tmp_path)
    record = next(iter(load_catalog(settings.catalog_path, settings.root_dir).values()))
    with pytest.raises(IndexNotReadyError, match="build-index --manual"):
        load_index(record, settings)
