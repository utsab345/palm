import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from app.config import Settings
from app.core.errors import CollectionDimensionMismatch, VectorStoreError

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        settings: Settings,
        client: AsyncQdrantClient | None = None,
    ):
        self._settings = settings
        self._client = (
            client
            if client is not None
            else AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key_value,
            )
        )

    @property
    def collection_name(self) -> str:
        return self._settings.qdrant_collection

    async def ensure_collection(self, dimensions: int) -> None:
        try:
            exists = await self._client.collection_exists(self.collection_name)
            if not exists:
                await self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=dimensions,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created Qdrant collection %s with %d dimensions",
                    self.collection_name,
                    dimensions,
                )
                return

            info = await self._client.get_collection(self.collection_name)
            vectors = info.config.params.vectors
            size = getattr(vectors, "size", None)
            if size is None:
                raise VectorStoreError("Collection uses named vectors, which are not supported")
            if int(size) != dimensions:
                raise CollectionDimensionMismatch(
                    f"Qdrant collection {self.collection_name} uses {size} dimensions "
                    f"but GEMINI_EMBEDDING_DIMENSIONS is {dimensions}"
                )
        except CollectionDimensionMismatch:
            raise
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to ensure Qdrant collection {self.collection_name}"
            ) from exc

    async def upsert(self, points: list[qdrant_models.PointStruct]) -> None:
        try:
            await self._client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError("Failed to write points to Qdrant") from exc

    async def search(self, vector: list[float], limit: int) -> list[qdrant_models.ScoredPoint]:
        try:
            response = await self._client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise VectorStoreError("Failed to query Qdrant") from exc
        return list(response.points)

    async def close(self) -> None:
        await self._client.close()
