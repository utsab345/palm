from types import SimpleNamespace

import pytest

from app.services.rag_service import RagService


class FakeEmbeddings:
    def __init__(self):
        self.query_calls: list[str] = []

    async def embed_query(self, text):
        self.query_calls.append(text)
        return [1.0, 0.0, 0.0, 0.0]


class FakeVectorStore:
    collection_name = "test_collection"

    def __init__(self):
        self.ensured: list[int] = []
        self.search_args = None
        self.results = []

    async def ensure_collection(self, dimensions):
        self.ensured.append(dimensions)

    async def search(self, vector, limit):
        self.search_args = (vector, limit)
        return self.results


@pytest.mark.asyncio
async def test_search_returns_sources(settings):
    embeddings = FakeEmbeddings()
    store = FakeVectorStore()
    store.results = [
        SimpleNamespace(
            id="point-1",
            score=0.87,
            payload={
                "text": "Interviews are held on weekdays.",
                "metadata": {"source": "faq"},
            },
        ),
        SimpleNamespace(id="point-2", score=0.5, payload={"text": "Bring ID"}),
    ]
    service = RagService(settings, embeddings, store)

    sources = await service.search("interview schedule", top_k=2)

    assert embeddings.query_calls == ["interview schedule"]
    assert store.ensured == [settings.gemini_embedding_dimensions]
    assert store.search_args == ([1.0, 0.0, 0.0, 0.0], 2)
    assert sources[0].id == "point-1"
    assert sources[0].score == pytest.approx(0.87)
    assert sources[0].metadata == {"source": "faq"}
    assert sources[1].metadata == {}
