import json
import logging

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import Settings
from app.core.errors import ChatMemoryError
from app.schemas import ChatMessage

logger = logging.getLogger(__name__)


class ChatMemoryService:
    def __init__(self, settings: Settings, client: Redis | None = None):
        self._settings = settings
        self._client = client or Redis.from_url(settings.redis_url, decode_responses=True)

    @staticmethod
    def _key(session_id: str) -> str:
        return f"chat:memory:{session_id}"

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        try:
            raw_messages = await self._client.lrange(
                self._key(session_id),
                -self._settings.chat_memory_max_messages,
                -1,
            )
        except RedisError as exc:
            raise ChatMemoryError("Could not read chat memory from Redis") from exc

        history: list[ChatMessage] = []
        for raw in raw_messages:
            try:
                message = ChatMessage.model_validate(json.loads(raw))
            except (ValueError, TypeError):
                logger.warning("Skipping malformed chat memory entry")
                continue
            history.append(message)
        return history

    async def append(self, session_id: str, message: ChatMessage) -> None:
        key = self._key(session_id)
        try:
            pipeline = self._client.pipeline()
            pipeline.rpush(key, message.model_dump_json())
            pipeline.ltrim(key, -self._settings.chat_memory_max_messages, -1)
            pipeline.expire(key, self._settings.chat_memory_ttl_seconds)
            await pipeline.execute()
        except RedisError as exc:
            raise ChatMemoryError("Could not write chat memory to Redis") from exc

    async def clear(self, session_id: str) -> None:
        try:
            await self._client.delete(self._key(session_id))
        except RedisError as exc:
            raise ChatMemoryError("Could not clear chat memory in Redis") from exc

    async def close(self) -> None:
        await self._client.aclose()
