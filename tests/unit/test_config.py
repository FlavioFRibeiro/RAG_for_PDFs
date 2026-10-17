from dataclasses import replace

import pytest

from rag_genai_core.config import Settings


def test_settings_reject_invalid_chunk_and_retrieval_parameters() -> None:
    settings = Settings()
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        replace(settings, chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError, match="RETRIEVAL_K"):
        replace(settings, retrieval_k=0)
