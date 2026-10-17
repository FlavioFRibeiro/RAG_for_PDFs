from dataclasses import replace

from rag_genai_core.catalog import load_catalog
from rag_genai_core.config import Settings
from rag_genai_core.generation import GenerationResult
from rag_genai_core.rag_pipeline import RAGPipeline
from rag_genai_core.retrieval import ManualRetriever
from rag_genai_core.schemas import AnswerStatus, Chunk, GeneratedAnswer
from rag_genai_core.storage import build_index, load_index


class SimulatedProvider:
    def generate(self, question, context, history):
        assert "17.2 L" in context
        return GenerationResult(
            GeneratedAnswer(
                status="answered",
                answer="The fuel tank capacity is 17.2 L. [S1]",
                citation_ids=["S1"],
            )
        )


def test_complete_pipeline_with_simulated_providers(tmp_path) -> None:
    base = Settings()
    settings = replace(base, artifacts_dir=tmp_path, retrieval_k=1)
    records = load_catalog(settings.catalog_path, settings.root_dir)
    record = records["voge-500ds-en-unknown-edition"]
    chunk = Chunk(
        "voge-fuel",
        record.manual_id,
        record.motorcycle_id,
        record.title,
        72,
        None,
        "Fuel tank capacity: 17.2 L",
    )
    build_index(record, [chunk], settings, lambda texts: [[1.0, 0.0] for _ in texts])
    index, chunks, _ = load_index(record, settings)
    retriever = ManualRetriever(index, chunks, lambda question: [1.0, 0.0], 1)
    response = RAGPipeline(record, list(records.values()), retriever, SimulatedProvider()).ask(
        "What is the fuel tank capacity?"
    )
    assert response.status == AnswerStatus.answered
    assert response.sources[0].physical_page == 72
