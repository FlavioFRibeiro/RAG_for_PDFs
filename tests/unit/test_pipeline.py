from pathlib import Path

import pytest

from rag_genai_core.exceptions import CitationValidationError
from rag_genai_core.generation import GenerationResult
from rag_genai_core.rag_pipeline import RAGPipeline
from rag_genai_core.schemas import AnswerStatus, ChatTurn, Chunk, GeneratedAnswer, ManualRecord


def record(manual_id: str, motorcycle_id: str, maker: str, model: str) -> ManualRecord:
    return ManualRecord(
        manual_id,
        motorcycle_id,
        maker,
        model,
        "unknown",
        "unknown",
        "unknown",
        "English",
        f"{model} manual",
        Path("x.pdf"),
        "x",
        "unknown",
        "unknown",
        "test",
    )


class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, question):
        self.question = question
        return self.chunks


class Provider:
    def __init__(self, generated):
        self.generated = generated
        self.calls = 0

    def generate(self, question, context, history):
        self.calls += 1
        return GenerationResult(self.generated)


def test_wrong_motorcycle_is_blocked_before_retrieval_or_generation() -> None:
    yamaha = record("yamaha", "yamaha-tenere-700", "Yamaha", "Ténéré 700")
    voge = record("voge", "voge-500ds", "Voge", "500DS")
    provider = Provider(GeneratedAnswer(status="answered", answer="x [S1]", citation_ids=["S1"]))
    response = RAGPipeline(yamaha, [yamaha, voge], Retriever([]), provider).ask(
        "What is the Voge 500DS pressure?"
    )
    assert response.status == AnswerStatus.needs_clarification
    assert provider.calls == 0


def test_valid_citation_is_bound_to_selected_manual() -> None:
    yamaha = record("yamaha", "yamaha-tenere-700", "Yamaha", "Ténéré 700")
    chunk = Chunk("chunk-1", "yamaha", "yamaha-tenere-700", "manual", 80, "7-16", "pressure")
    provider = Provider(
        GeneratedAnswer(status="answered", answer="Use the stated pressure. [S1]", citation_ids=["S1"])
    )
    response = RAGPipeline(yamaha, [yamaha], Retriever([chunk]), provider).ask(
        "What is the chain specification?"
    )
    assert response.sources[0].chunk_id == "chunk-1"


def test_unknown_citation_is_rejected_and_valid_missing_inline_citation_is_added() -> None:
    yamaha = record("yamaha", "yamaha-tenere-700", "Yamaha", "Ténéré 700")
    chunk = Chunk("chunk-1", "yamaha", "yamaha-tenere-700", "manual", 80, None, "pressure")
    unknown = Provider(GeneratedAnswer(status="answered", answer="Claim [S9]", citation_ids=["S9"]))
    with pytest.raises(CitationValidationError, match="unknown"):
        RAGPipeline(yamaha, [yamaha], Retriever([chunk]), unknown).ask("Chain specification?")
    missing = Provider(GeneratedAnswer(status="answered", answer="Claim", citation_ids=["S1"]))
    response = RAGPipeline(yamaha, [yamaha], Retriever([chunk]), missing).ask(
        "Chain specification?"
    )
    assert response.answer == "Claim [S1]"


def test_follow_up_uses_previous_user_question_for_retrieval() -> None:
    voge = record("voge", "voge-500ds", "Voge", "500DS")
    chunk = Chunk("chunk-1", "voge", "voge-500ds", "manual", 62, None, "250 kPa")
    retriever = Retriever([chunk])
    provider = Provider(
        GeneratedAnswer(status="answered", answer="Use 250 kPa. [S1]", citation_ids=["S1"])
    )
    history = [ChatTurn(role="user", content="What tire pressure is specified for a rider only?")]
    RAGPipeline(voge, [voge], retriever, provider).ask("And for two-up riding?", history)
    assert "What tire pressure" in retriever.question


def test_ambiguous_voge_oil_capacity_requests_clarification_without_provider() -> None:
    voge = record("voge", "voge-500ds", "Voge", "500DS")
    provider = Provider(GeneratedAnswer(status="answered", answer="x", citation_ids=[]))
    response = RAGPipeline(voge, [voge], Retriever([]), provider).ask(
        "What is the engine oil capacity?"
    )
    assert response.status == AnswerStatus.needs_clarification
    assert provider.calls == 0


def test_ambiguous_yamaha_pressure_follow_up_preserves_terrain_condition() -> None:
    yamaha = record("yamaha", "yamaha-tenere-700", "Yamaha", "Ténéré 700")
    provider = Provider(GeneratedAnswer(status="answered", answer="x", citation_ids=[]))
    history = [ChatTurn(role="user", content="What tire pressure should I use?")]
    response = RAGPipeline(yamaha, [yamaha], Retriever([]), provider).ask(
        "And with a passenger?", history
    )
    assert response.status == AnswerStatus.needs_clarification
    assert "on-road or off-road" in response.answer
    assert provider.calls == 0
