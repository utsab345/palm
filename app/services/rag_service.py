from typing import Any

from app.config import Settings
from app.schemas import Source
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


class RagService:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
    ):
        self._settings = settings
        self._embeddings = embeddings
        self._vector_store = vector_store

    async def search(self, query: str, top_k: int = 4) -> list[Source]:
        await self._vector_store.ensure_collection(self._settings.gemini_embedding_dimensions)
        vector = await self._embeddings.embed_query(query)
        points = await self._vector_store.search(vector, top_k)

        sources: list[Source] = []
        for point in points:
            payload = point.payload or {}
            text = payload.get("text")
            metadata = payload.get("metadata")
            sources.append(
                Source(
                    id=str(point.id),
                    text=text if isinstance(text, str) else "",
                    score=float(point.score) if point.score is not None else None,
                    metadata=self._as_metadata(metadata),
                )
            )
        return sources

    @staticmethod
    def _as_metadata(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if value is None:
            return {}
        return {"value": value}
