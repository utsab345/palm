import math
from types import SimpleNamespace

import pytest
from google.genai import errors

from app.core.errors import (
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    EmbeddingUnavailableError,
)
from app.services.embedding_service import EmbeddingService
from tests.conftest import FakeClient


def build_service(settings):
    client = FakeClient()
    return EmbeddingService(settings, client=client), client


def embedding_response(vectors):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=values) for values in vectors])


@pytest.mark.asyncio
async def test_embed_documents_uses_document_task_and_dimensions(settings):
    service, client = build_service(settings)
    client.models.embed_responses.append(
        embedding_response([[3.0, 0.0, 0.0, 4.0], [0.0, 1.0, 0.0, 0.0]])
    )

    vectors = await service.embed_documents(["first", "second"])

    assert len(vectors) == 2
    assert math.isclose(sum(v * v for v in vectors[0]), 1.0)
    assert vectors[0][:3] == pytest.approx([0.6, 0.0, 0.0])
    call = client.models.embed_calls[0]
    assert call["model"] == settings.gemini_embedding_model
    assert call["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert call["config"].output_dimensionality == settings.gemini_embedding_dimensions


@pytest.mark.asyncio
async def test_embed_query_uses_query_task(settings):
    service, client = build_service(settings)
    client.models.embed_responses.append(embedding_response([[1.0, 0.0, 0.0, 0.0]]))

    vector = await service.embed_query("check-in time")

    assert vector == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert client.models.embed_calls[0]["config"].task_type == "RETRIEVAL_QUERY"


@pytest.mark.asyncio
async def test_embed_batches_large_inputs(settings):
    service, client = build_service(settings.model_copy(update={"embedding_batch_size": 2}))
    client.models.embed_responses.append(
        embedding_response([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    )
    client.models.embed_responses.append(embedding_response([[1.0, 0.0, 0.0, 0.0]]))

    vectors = await service.embed_documents(["a", "b", "c"])

    assert len(vectors) == 3
    assert len(client.models.embed_calls) == 2


@pytest.mark.asyncio
async def test_wrong_dimension_raises(settings):
    service, client = build_service(settings)
    client.models.embed_responses.append(embedding_response([[1.0, 0.0]]))

    with pytest.raises(EmbeddingDimensionError):
        await service.embed_query("too short")


@pytest.mark.asyncio
async def test_wrong_embedding_count_raises(settings):
    service, client = build_service(settings)
    client.models.embed_responses.append(embedding_response([[1.0, 0.0, 0.0, 0.0]]))

    with pytest.raises(EmbeddingError):
        await service.embed_documents(["a", "b"])


@pytest.mark.asyncio
async def test_embedding_api_error_is_wrapped(settings):
    service, client = build_service(settings)
    client.models.embed_responses.append(
        errors.APIError(503, {"error": {"code": 503, "message": "down"}})
    )

    with pytest.raises(EmbeddingUnavailableError) as exc_info:
        await service.embed_query("hello")

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_embedding_rate_limit_maps_to_429(settings):
    service, client = build_service(settings)
    client.models.embed_responses.append(
        errors.APIError(429, {"error": {"code": 429, "message": "slow down"}})
    )

    with pytest.raises(EmbeddingRateLimitError) as exc_info:
        await service.embed_query("hello")

    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_embedding_timeout_maps_to_504(settings):
    service, client = build_service(settings)
    client.models.embed_responses.append(TimeoutError())

    with pytest.raises(EmbeddingTimeoutError) as exc_info:
        await service.embed_query("hello")

    assert exc_info.value.status_code == 504
