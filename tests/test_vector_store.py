from types import SimpleNamespace

import pytest

from app.core.errors import CollectionDimensionMismatch, VectorStoreError
from app.services.vector_store import VectorStore


class FakeQdrant:
    def __init__(self, exists: bool, size: int | None = None):
        self.exists = exists
        self.size = size
        self.created = None
        self.upsert_args = None
        self.query_args = None
        self.fail_upsert = False

    async def collection_exists(self, collection_name):
        return self.exists

    async def create_collection(self, collection_name, vectors_config):
        self.created = (collection_name, vectors_config)

    async def get_collection(self, collection_name):
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=self.size)))
        )

    async def upsert(self, collection_name, points, wait):
        if self.fail_upsert:
            raise RuntimeError("boom")
        self.upsert_args = (collection_name, points, wait)

    async def query_points(self, **kwargs):
        self.query_args = kwargs
        return SimpleNamespace(points=[SimpleNamespace(id="1", score=0.5, payload={"text": "hi"})])

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_creates_collection_when_missing(settings):
    client = FakeQdrant(exists=False)
    store = VectorStore(settings, client=client)

    await store.ensure_collection(768)

    assert client.created is not None
    assert client.created[0] == settings.qdrant_collection
    assert client.created[1].size == 768


@pytest.mark.asyncio
async def test_matching_dimensions_do_not_recreate(settings):
    client = FakeQdrant(exists=True, size=4)
    store = VectorStore(settings, client=client)

    await store.ensure_collection(4)

    assert client.created is None


@pytest.mark.asyncio
async def test_dimension_mismatch_raises(settings):
    client = FakeQdrant(exists=True, size=1536)
    store = VectorStore(settings, client=client)

    with pytest.raises(CollectionDimensionMismatch):
        await store.ensure_collection(768)


@pytest.mark.asyncio
async def test_upsert_failure_wraps_error(settings):
    client = FakeQdrant(exists=True, size=4)
    client.fail_upsert = True
    store = VectorStore(settings, client=client)

    with pytest.raises(VectorStoreError):
        await store.upsert([])


@pytest.mark.asyncio
async def test_search_passes_dimensions_and_limit(settings):
    client = FakeQdrant(exists=True, size=4)
    store = VectorStore(settings, client=client)

    points = await store.search([1.0, 0.0, 0.0, 0.0], 3)

    assert len(points) == 1
    assert client.query_args["limit"] == 3
    assert client.query_args["collection_name"] == settings.qdrant_collection
