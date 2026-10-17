from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from dotenv import load_dotenv

from .catalog import load_catalog, validate_catalog
from .config import Settings
from .ingestion import extract_chunks
from .observability import configure_logging
from .runtime import create_openai_client, openai_embed_documents
from .storage import build_index


def validate_command(settings: Settings) -> int:
    catalog = load_catalog(settings.catalog_path, settings.root_dir)
    validate_catalog(catalog.values())
    for record in catalog.values():
        print(f"OK {record.manual_id}: {record.display_name} ({record.pdf_path.name})")
    return 0


def build_command(settings: Settings, manual: str) -> int:
    catalog = load_catalog(settings.catalog_path, settings.root_dir)
    validate_catalog(catalog.values())
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required to build indexes.", file=sys.stderr)
        return 2
    if manual != "all" and manual not in catalog:
        print(f"Unknown manual_id: {manual}", file=sys.stderr)
        return 2
    selected = list(catalog.values()) if manual == "all" else [catalog[manual]]
    client = create_openai_client()
    for record in selected:
        chunks = extract_chunks(record, settings.chunk_size, settings.chunk_overlap)
        manifest = build_index(
            record,
            chunks,
            settings,
            lambda texts: openai_embed_documents(client, settings.embedding_model, texts),
        )
        print(json.dumps(manifest, indent=2))
    return 0


def main(argv: List[str] = None) -> int:
    load_dotenv()
    configure_logging()
    parser = argparse.ArgumentParser(prog="motorcycle-manual-assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-catalog")
    build_parser = subparsers.add_parser("build-index")
    build_parser.add_argument("--manual", required=True, help="A catalog manual_id or 'all'.")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    if args.command == "validate-catalog":
        return validate_command(settings)
    return build_command(settings, args.manual)


if __name__ == "__main__":
    raise SystemExit(main())
