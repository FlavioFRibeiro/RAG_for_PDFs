from __future__ import annotations

import logging
import re
import time
import unicodedata
import uuid
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .exceptions import CitationValidationError
from .generation import GenerationProvider
from .observability import event
from .retrieval import ManualRetriever
from .schemas import AnswerStatus, AssistantResponse, ChatTurn, Chunk, ManualRecord, Source

LOGGER = logging.getLogger(__name__)


def _mentions_other_motorcycle(
    question: str, selected: ManualRecord, all_records: Iterable[ManualRecord]
) -> str:
    def plain(value: str) -> str:
        return "".join(
            character
            for character in unicodedata.normalize("NFKD", value.lower())
            if not unicodedata.combining(character)
        )

    lower = plain(question)
    for record in all_records:
        if record.manual_id == selected.manual_id:
            continue
        aliases = {plain(record.model), plain(record.motorcycle_id), plain(record.model).replace(" ", "")}
        if "tenere" in plain(record.model):
            aliases.add("tenere")
        if "500ds" in plain(record.model).replace(" ", ""):
            aliases.update({"500ds", "500 ds"})
        if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", lower) for alias in aliases):
            return record.display_name
    return ""


def _context(chunks: Sequence[Chunk]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        printed = f", printed page {chunk.printed_page}" if chunk.printed_page else ""
        blocks.append(
            f"[S{number}] chunk_id={chunk.chunk_id}; physical PDF page {chunk.physical_page}{printed}\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def _preflight_response(
    question: str, selected: ManualRecord, history: Sequence[ChatTurn] = ()
) -> Optional[Tuple[AnswerStatus, str]]:
    lower = question.lower()
    previous_user = next((turn.content.lower() for turn in reversed(history) if turn.role == "user"), "")
    asks_pressure = "tire pressure" in lower or "tyre pressure" in lower
    ambiguous_yamaha_follow_up = (
        "yamaha" in selected.manufacturer.lower()
        and (lower.startswith("and ") or lower.startswith("what about"))
        and "passenger" in lower
        and ("tire pressure" in previous_user or "tyre pressure" in previous_user)
        and not any(term in previous_user for term in ("off-road", "off road", "on-road", "on road"))
    )
    if ambiguous_yamaha_follow_up:
        return (
            AnswerStatus.needs_clarification,
            "Do you mean on-road or off-road riding? The manual lists a separate off-road pressure.",
        )
    if asks_pressure:
        if "yamaha" in selected.manufacturer.lower() and not any(
            term in lower
            for term in ("off-road", "off road", "one person", "two people", "passenger", "rider")
        ):
            return (
                AnswerStatus.needs_clarification,
                "Which riding condition do you mean: off-road riding, one-person road riding, or riding with a passenger?",
            )
        if "voge" in selected.manufacturer.lower() and not any(
            term in lower for term in ("driver", "rider", "passenger", "two-up", "two up")
        ):
            return (
                AnswerStatus.needs_clarification,
                "Do you need the tire pressure for a rider only or for a rider with a passenger?",
            )
    if (
        "voge" in selected.manufacturer.lower()
        and "oil" in lower
        and any(term in lower for term in ("capacity", "how much", "quantity"))
        and not any(term in lower for term in ("parameter", "table", "replacement", "fill", "change"))
    ):
        return (
            AnswerStatus.needs_clarification,
            "The manual presents different engine-oil quantities in its parameter table and replacement procedure. Which context do you need?",
        )
    if any(term in lower for term in ("model year", "manual edition", "how old")):
        if selected.model_year == "unknown":
            return (
                AnswerStatus.not_found,
                "The selected manual does not confirm the motorcycle model year, so it cannot be inferred from the document date.",
            )
    return None


def _retrieval_question(question: str, history: Sequence[ChatTurn]) -> str:
    words = question.split()
    is_follow_up = len(words) <= 6 or question.lower().startswith(("and ", "what about", "how about"))
    if not is_follow_up:
        return question
    previous_user = next((turn.content for turn in reversed(history) if turn.role == "user"), "")
    return f"{previous_user}\nFollow-up: {question}" if previous_user else question


def _validate_sources(
    citation_ids: Sequence[str], chunks: Sequence[Chunk], selected: ManualRecord
) -> List[Source]:
    available: Dict[str, Chunk] = {f"S{index}": chunk for index, chunk in enumerate(chunks, start=1)}
    sources = []
    for source_id in dict.fromkeys(citation_ids):
        chunk = available.get(source_id)
        if chunk is None:
            raise CitationValidationError(f"Generated answer cited unknown source [{source_id}].")
        if chunk.manual_id != selected.manual_id or chunk.motorcycle_id != selected.motorcycle_id:
            raise CitationValidationError(
                f"Generated answer cited a source from another manual: [{source_id}]."
            )
        sources.append(
            Source(
                source_id=source_id,
                manual_id=chunk.manual_id,
                motorcycle_id=chunk.motorcycle_id,
                chunk_id=chunk.chunk_id,
                manual_name=chunk.manual_name,
                physical_page=chunk.physical_page,
                printed_page=chunk.printed_page,
                evidence=chunk.text,
            )
        )
    return sources


class RAGPipeline:
    def __init__(
        self,
        selected: ManualRecord,
        all_records: Sequence[ManualRecord],
        retriever: ManualRetriever,
        provider: GenerationProvider,
        history_turns: int = 3,
    ):
        self.selected = selected
        self.all_records = list(all_records)
        self.retriever = retriever
        self.provider = provider
        self.history_turns = history_turns

    def retrieve(self, question: str, history: Sequence[ChatTurn] = ()) -> List[Chunk]:
        bounded_history = list(history)[-self.history_turns * 2 :]
        return self.retriever.retrieve(_retrieval_question(question, bounded_history))

    def ask(self, question: str, history: Sequence[ChatTurn] = ()) -> AssistantResponse:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        other = _mentions_other_motorcycle(question, self.selected, self.all_records)
        if other:
            latency_ms = int((time.perf_counter() - started) * 1000)
            event(
                LOGGER,
                "rag_request_completed",
                request_id=request_id,
                manual_id=self.selected.manual_id,
                latency_ms=latency_ms,
            )
            return AssistantResponse(
                status=AnswerStatus.needs_clarification,
                answer=f"This chat is using the {self.selected.display_name} manual. Please switch the selection to {other} before asking about that motorcycle.",
                request_id=request_id,
                latency_ms=latency_ms,
            )
        bounded_history = list(history)[-self.history_turns * 2 :]
        preflight = _preflight_response(question, self.selected, bounded_history)
        if preflight:
            status, answer = preflight
            latency_ms = int((time.perf_counter() - started) * 1000)
            event(
                LOGGER,
                "rag_request_completed",
                request_id=request_id,
                manual_id=self.selected.manual_id,
                latency_ms=latency_ms,
            )
            return AssistantResponse(
                status=status,
                answer=answer,
                request_id=request_id,
                latency_ms=latency_ms,
            )
        retrieval_started = time.perf_counter()
        chunks = self.retrieve(question, bounded_history)
        event(
            LOGGER,
            "retrieval_completed",
            request_id=request_id,
            manual_id=self.selected.manual_id,
            latency_ms=int((time.perf_counter() - retrieval_started) * 1000),
        )
        generation_started = time.perf_counter()
        result = self.provider.generate(question, _context(chunks), bounded_history)
        event(
            LOGGER,
            "generation_completed",
            request_id=request_id,
            manual_id=self.selected.manual_id,
            latency_ms=int((time.perf_counter() - generation_started) * 1000),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        sources = _validate_sources(result.answer.citation_ids, chunks, self.selected)
        if result.answer.status == AnswerStatus.answered and not sources:
            raise CitationValidationError("An answered response must cite at least one retrieved source.")
        if result.answer.status == AnswerStatus.answered:
            missing_inline = [
                source.source_id for source in sources if f"[{source.source_id}]" not in result.answer.answer
            ]
            if missing_inline:
                result.answer.answer = f"{result.answer.answer.rstrip()} {' '.join(f'[{item}]' for item in missing_inline)}"
        latency_ms = int((time.perf_counter() - started) * 1000)
        response = AssistantResponse(
            status=result.answer.status,
            answer=result.answer.answer,
            sources=sources,
            conditions=result.answer.conditions,
            request_id=request_id,
            latency_ms=latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        event(
            LOGGER,
            "rag_request_completed",
            request_id=request_id,
            manual_id=self.selected.manual_id,
            latency_ms=latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        return response
