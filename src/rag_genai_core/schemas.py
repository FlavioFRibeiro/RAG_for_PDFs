from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class AnswerStatus(str, Enum):
    answered = "answered"
    needs_clarification = "needs_clarification"
    not_found = "not_found"


@dataclass(frozen=True)
class ManualRecord:
    manual_id: str
    motorcycle_id: str
    manufacturer: str
    model: str
    model_year: str
    edition: str
    market: str
    language: str
    title: str
    pdf_path: Path
    sha256: str
    origin: str
    redistribution: str
    identity_basis: str

    @property
    def display_name(self) -> str:
        return f"{self.manufacturer} {self.model}"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    manual_id: str
    motorcycle_id: str
    manual_name: str
    physical_page: int
    printed_page: Optional[str]
    text: str


@dataclass(frozen=True)
class Source:
    source_id: str
    manual_id: str
    motorcycle_id: str
    chunk_id: str
    manual_name: str
    physical_page: int
    printed_page: Optional[str]
    evidence: str


@dataclass
class AssistantResponse:
    status: AnswerStatus
    answer: str
    sources: List[Source] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    request_id: str = ""
    latency_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class GeneratedAnswer(BaseModel):
    status: AnswerStatus
    answer: str = Field(description="A direct English answer using only the supplied context.")
    citation_ids: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ChatTurn:
    role: str
    content: str
