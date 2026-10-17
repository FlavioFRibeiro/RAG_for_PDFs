from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable

import yaml

from .exceptions import CatalogError
from .schemas import ManualRecord

REQUIRED_FIELDS = {
    "manual_id",
    "motorcycle_id",
    "manufacturer",
    "model",
    "model_year",
    "edition",
    "market",
    "language",
    "title",
    "pdf_path",
    "sha256",
    "origin",
    "redistribution",
    "identity_basis",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_catalog(path: Path, root_dir: Path) -> Dict[str, ManualRecord]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogError(f"Catalog not found: {path}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise CatalogError("Catalog must be an object with version: 1.")
    records: Dict[str, ManualRecord] = {}
    for raw in payload.get("manuals", []):
        missing = REQUIRED_FIELDS - set(raw)
        if missing:
            raise CatalogError(f"Catalog entry is missing fields: {sorted(missing)}")
        manual_id = str(raw["manual_id"])
        if manual_id in records:
            raise CatalogError(f"Duplicate manual_id: {manual_id}")
        resolved = (root_dir / str(raw["pdf_path"])).resolve()
        try:
            resolved.relative_to(root_dir.resolve())
        except ValueError as exc:
            raise CatalogError(f"Manual path escapes project root: {raw['pdf_path']}") from exc
        values = {key: str(raw[key]) for key in REQUIRED_FIELDS if key != "pdf_path"}
        records[manual_id] = ManualRecord(**values, pdf_path=resolved)
    if not records:
        raise CatalogError("Catalog contains no manuals.")
    return records


def validate_catalog(records: Iterable[ManualRecord], verify_hashes: bool = True) -> None:
    motorcycle_ids = set()
    for record in records:
        if record.motorcycle_id in motorcycle_ids:
            raise CatalogError(f"Duplicate motorcycle_id: {record.motorcycle_id}")
        motorcycle_ids.add(record.motorcycle_id)
        if not record.pdf_path.is_file():
            raise CatalogError(f"Manual file is missing for {record.manual_id}: {record.pdf_path}")
        if verify_hashes:
            actual = sha256_file(record.pdf_path)
            if actual != record.sha256:
                raise CatalogError(
                    f"Hash mismatch for {record.manual_id}: expected {record.sha256}, got {actual}."
                )
