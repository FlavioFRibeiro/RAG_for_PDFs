from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import faiss
import numpy as np

from .config import PARSER_VERSION, TABLE_RULES_VERSION, Settings
from .exceptions import IndexNotReadyError
from .schemas import Chunk, ManualRecord

EmbeddingFunction = Callable[[List[str]], List[List[float]]]


def index_fingerprint(record: ManualRecord, settings: Settings) -> str:
    payload = {
        "pdf_sha256": record.sha256,
        "parser": PARSER_VERSION,
        "table_rules": TABLE_RULES_VERSION,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (model TEXT, content_hash TEXT, vector TEXT, PRIMARY KEY(model, content_hash))"
        )

    def embed(self, texts: Sequence[str], model: str, function: EmbeddingFunction) -> List[List[float]]:
        hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        result: List[List[float]] = [[] for _ in texts]
        missing: List[int] = []
        for index, content_hash in enumerate(hashes):
            row = self.connection.execute(
                "SELECT vector FROM embeddings WHERE model=? AND content_hash=?", (model, content_hash)
            ).fetchone()
            if row:
                result[index] = json.loads(row[0])
            else:
                missing.append(index)
        for offset in range(0, len(missing), 96):
            positions = missing[offset : offset + 96]
            vectors = function([texts[position] for position in positions])
            if len(vectors) != len(positions):
                raise ValueError("Embedding provider returned an unexpected vector count.")
            for position, vector in zip(positions, vectors):
                result[position] = vector
                self.connection.execute(
                    "INSERT OR REPLACE INTO embeddings(model, content_hash, vector) VALUES(?,?,?)",
                    (model, hashes[position], json.dumps(vector, separators=(",", ":"))),
                )
            self.connection.commit()
        return result

    def close(self) -> None:
        self.connection.close()


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def build_index(
    record: ManualRecord,
    chunks: Sequence[Chunk],
    settings: Settings,
    embed_function: EmbeddingFunction,
) -> Dict[str, object]:
    if not chunks:
        raise ValueError(f"No text chunks were extracted from {record.manual_id}.")
    fingerprint = index_fingerprint(record, settings)
    manual_root = settings.artifacts_dir / "indexes" / record.manual_id
    destination = manual_root / fingerprint
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        _write_json_atomic(manual_root / "current.json", {"fingerprint": fingerprint})
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    cache = EmbeddingCache(settings.artifacts_dir / "embedding-cache.sqlite3")
    try:
        vectors = cache.embed([chunk.text for chunk in chunks], settings.embedding_model, embed_function)
    finally:
        cache.close()
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise ValueError("Invalid embedding matrix.")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    staging = manual_root / f".{fingerprint}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        faiss.write_index(index, str(staging / "index.faiss"))
        _write_json_atomic(staging / "chunks.json", [asdict(chunk) for chunk in chunks])
        manifest = {
            "schema_version": 1,
            "manual_id": record.manual_id,
            "motorcycle_id": record.motorcycle_id,
            "fingerprint": fingerprint,
            "pdf_sha256": record.sha256,
            "embedding_model": settings.embedding_model,
            "parser_version": PARSER_VERSION,
            "table_rules_version": TABLE_RULES_VERSION,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "chunk_count": len(chunks),
            "vector_dimensions": int(matrix.shape[1]),
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json_atomic(staging / "manifest.json", manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
        _write_json_atomic(manual_root / "current.json", {"fingerprint": fingerprint})
        return manifest
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()


def load_index(record: ManualRecord, settings: Settings):
    fingerprint = index_fingerprint(record, settings)
    directory = settings.artifacts_dir / "indexes" / record.manual_id / fingerprint
    required = [directory / "index.faiss", directory / "chunks.json", directory / "manifest.json"]
    if not all(path.is_file() for path in required):
        raise IndexNotReadyError(
            f"Index for {record.display_name} is missing or incompatible. Run: "
            f"python -m rag_genai_core.cli build-index --manual {record.manual_id}"
        )
    manifest = json.loads(required[2].read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != fingerprint or manifest.get("manual_id") != record.manual_id:
        raise IndexNotReadyError(f"Index manifest is incompatible for {record.manual_id}.")
    chunks = [Chunk(**item) for item in json.loads(required[1].read_text(encoding="utf-8"))]
    index = faiss.read_index(str(required[0]))
    if index.ntotal != len(chunks):
        raise IndexNotReadyError(f"Index and metadata count differ for {record.manual_id}.")
    return index, chunks, manifest
