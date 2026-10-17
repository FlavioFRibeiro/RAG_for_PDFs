from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
from openai import OpenAI

from .catalog import load_catalog, validate_catalog
from .config import Settings
from .exceptions import ProviderError
from .generation import OpenAIGenerationProvider
from .observability import event
from .rag_pipeline import RAGPipeline
from .retrieval import ManualRetriever
from .storage import load_index

LOGGER = logging.getLogger(__name__)


def create_openai_client() -> OpenAI:
    ca_bundle = os.getenv("MMA_CA_BUNDLE", "").strip()
    if ca_bundle:
        path = Path(ca_bundle)
        if not path.is_file():
            raise ValueError(f"MMA_CA_BUNDLE does not exist: {path}")
        return OpenAI(http_client=httpx.Client(verify=str(path)))
    return OpenAI()


def openai_embed_documents(client: OpenAI, model: str, texts):
    try:
        response = client.embeddings.create(model=model, input=texts)
    except Exception as exc:
        raise ProviderError(f"OpenAI embedding failed: {exc}") from exc
    usage = getattr(response, "usage", None)
    event(
        LOGGER,
        "embedding_request_completed",
        model=model,
        input_count=len(texts),
        input_tokens=getattr(usage, "prompt_tokens", None),
    )
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def build_runtime(manual_id: str, settings: Settings = None) -> RAGPipeline:
    settings = settings or Settings.from_env()
    catalog = load_catalog(settings.catalog_path, settings.root_dir)
    validate_catalog(catalog.values())
    selected = catalog[manual_id]
    index, chunks, _ = load_index(selected, settings)
    client = create_openai_client()
    retriever = ManualRetriever(
        index=index,
        chunks=chunks,
        embed_query=lambda question: openai_embed_documents(client, settings.embedding_model, [question])[0],
        k=settings.retrieval_k,
    )
    provider = OpenAIGenerationProvider(settings.generation_model, client=client)
    return RAGPipeline(selected, list(catalog.values()), retriever, provider, settings.history_turns)
