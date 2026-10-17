from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    catalog_path: Path = ROOT_DIR / "configs" / "manuals.yaml"
    artifacts_dir: Path = ROOT_DIR / "artifacts"
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"
    chunk_size: int = 1400
    chunk_overlap: int = 220
    retrieval_k: int = 6
    history_turns: int = 3
    page_dpi: int = 140

    def __post_init__(self) -> None:
        if not self.embedding_model.strip() or not self.generation_model.strip():
            raise ValueError("OpenAI model names must not be empty.")
        if self.chunk_overlap < 0 or self.chunk_size <= self.chunk_overlap:
            raise ValueError("MMA_CHUNK_SIZE must be greater than MMA_CHUNK_OVERLAP >= 0.")
        if self.retrieval_k <= 0:
            raise ValueError("MMA_RETRIEVAL_K must be greater than zero.")
        if self.history_turns < 0:
            raise ValueError("MMA_HISTORY_TURNS must not be negative.")
        if self.page_dpi <= 0:
            raise ValueError("MMA_PAGE_DPI must be greater than zero.")

    @classmethod
    def from_env(cls) -> "Settings":
        root = ROOT_DIR
        return cls(
            root_dir=root,
            catalog_path=Path(os.getenv("MMA_CATALOG_PATH", root / "configs" / "manuals.yaml")),
            artifacts_dir=Path(os.getenv("MMA_ARTIFACTS_DIR", root / "artifacts")),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            generation_model=os.getenv("OPENAI_GENERATION_MODEL", "gpt-4o-mini"),
            chunk_size=int(os.getenv("MMA_CHUNK_SIZE", "1400")),
            chunk_overlap=int(os.getenv("MMA_CHUNK_OVERLAP", "220")),
            retrieval_k=int(os.getenv("MMA_RETRIEVAL_K", "6")),
            history_turns=int(os.getenv("MMA_HISTORY_TURNS", "3")),
            page_dpi=int(os.getenv("MMA_PAGE_DPI", "140")),
        )


PARSER_VERSION = "pymupdf-text-tables-v1"
TABLE_RULES_VERSION = "table-markdown-v1"
