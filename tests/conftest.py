from types import SimpleNamespace

import pytest

from app.config import Settings


class FakeModels:
    def __init__(self):
        self.generate_calls: list[dict] = []
        self.embed_calls: list[dict] = []
        self.generate_responses: list[object] = []
        self.embed_responses: list[object] = []

    async def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        response = self.generate_responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    async def embed_content(self, **kwargs):
        self.embed_calls.append(kwargs)
        response = self.embed_responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeClient:
    def __init__(self):
        self.models = FakeModels()
        self.aio = SimpleNamespace(models=self.models)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        gemini_api_key="test-key",
        gemini_model="gemini-2.5-flash",
        gemini_embedding_model="gemini-embedding-001",
        gemini_embedding_dimensions=4,
        gemini_max_retries=0,
        gemini_retry_initial_wait=0.0,
        gemini_retry_max_wait=0.0,
    )
