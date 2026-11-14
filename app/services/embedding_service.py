import logging
import math

import httpx
from google import genai
from google.genai import errors, types

from app.config import Settings
from app.core.errors import (
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    EmbeddingUnavailableError,
)
from app.core.retry import with_retry
from app.integrations.client import create_genai_client

logger = logging.getLogger(__name__)

DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"


class EmbeddingService:
    def __init__(self, settings: Settings, client: genai.Client | None = None):
        self._settings = settings
        self._client = client if client is not None else create_genai_client(settings)
        self._dimensions = settings.gemini_embedding_dimensions
        self._model = settings.gemini_embedding_model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, DOCUMENT_TASK_TYPE)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([text], QUERY_TASK_TYPE)
        return vectors[0]

    async def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        batch_size = self._settings.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(await self._embed_batch(batch, task_type))
        return vectors

    async def _embed_batch(self, batch: list[str], task_type: str) -> list[list[float]]:
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self._dimensions,
        )

        async def operation() -> types.EmbedContentResponse:
            return await self._client.aio.models.embed_content(
                model=self._model,
                contents=batch,
                config=config,
            )

        try:
            response = await with_retry(
                operation,
                max_attempts=self._settings.gemini_max_retries + 1,
                initial_wait=self._settings.gemini_retry_initial_wait,
                max_wait=self._settings.gemini_retry_max_wait,
            )
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            if code == 429:
                raise EmbeddingRateLimitError("Gemini embedding rate limit exceeded") from exc
            if code is not None and 500 <= code < 600:
                raise EmbeddingUnavailableError(
                    f"Gemini embeddings are unavailable ({code})"
                ) from exc
            raise EmbeddingError(f"Gemini embedding request failed ({code})") from exc
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise EmbeddingTimeoutError("Gemini embedding request timed out") from exc
        except httpx.TransportError as exc:
            raise EmbeddingUnavailableError("Could not reach the Gemini embedding API") from exc

        embeddings = response.embeddings or []
        if len(embeddings) != len(batch):
            raise EmbeddingError(f"Expected {len(batch)} embeddings, received {len(embeddings)}")

        vectors: list[list[float]] = []
        for embedding in embeddings:
            values = list(embedding.values or [])
            if len(values) != self._dimensions:
                raise EmbeddingDimensionError(
                    f"Expected {self._dimensions} dimensions, received {len(values)}"
                )
            vectors.append(self._normalize(values))
        return vectors

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
