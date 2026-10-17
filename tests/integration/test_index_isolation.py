from pathlib import Path

from rag_genai_core.catalog import load_catalog
from rag_genai_core.config import Settings
from rag_genai_core.retrieval import ManualRetriever
from rag_genai_core.schemas import Chunk
from rag_genai_core.storage import build_index, load_index


def vector(text: str):
    return [1.0, 0.0] if "yamaha" in text.lower() else [0.0, 1.0]


def test_separate_indexes_cannot_return_other_motorcycle(tmp_path: Path) -> None:
    base = Settings()
    settings = Settings(root_dir=base.root_dir, catalog_path=base.catalog_path, artifacts_dir=tmp_path)
    records = load_catalog(settings.catalog_path, settings.root_dir)
    for record in records.values():
        chunk = Chunk(
            f"{record.manual_id}-chunk",
            record.manual_id,
            record.motorcycle_id,
            record.title,
            1,
            None,
            record.manufacturer.lower(),
        )
        build_index(record, [chunk], settings, lambda texts: [vector(text) for text in texts])
        index, chunks, _ = load_index(record, settings)
        retrieved = ManualRetriever(index, chunks, vector, 1).retrieve(record.manufacturer)
        assert {item.manual_id for item in retrieved} == {record.manual_id}
