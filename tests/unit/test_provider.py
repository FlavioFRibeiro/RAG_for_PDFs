import pytest

from rag_genai_core.exceptions import ProviderError
from rag_genai_core.generation import OpenAIGenerationProvider


class BrokenResponses:
    def parse(self, **kwargs):
        raise RuntimeError("provider unavailable")


class BrokenClient:
    responses = BrokenResponses()


def test_generation_provider_error_is_recoverable() -> None:
    provider = OpenAIGenerationProvider("test-model", client=BrokenClient())
    with pytest.raises(ProviderError, match="provider unavailable"):
        provider.generate("question", "context", [])
