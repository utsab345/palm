import json

import pytest
from redis.exceptions import RedisError

from app.core.errors import ChatMemoryError
from app.schemas import ChatMessage
from app.services.memory_service import ChatMemoryService


class FakePipeline:
    def __init__(self, store: "FakeRedis"):
        self._store = store

    def rpush(self, key, value):
        self._store.lists.setdefault(key, []).append(value)
        return self

    def ltrim(self, key, start, end):
        items = self._store.lists.get(key, [])
        length = len(items)
        if start < 0:
            start = max(length + start, 0)
        if end < 0:
            end = length + end + 1
        else:
            end = end + 1
        self._store.lists[key] = items[start:end]
        return self

    def expire(self, key, seconds):
        self._store.expirations.append((key, seconds))
        return self

    async def execute(self):
        return []


class FakeRedis:
    def __init__(self):
        self.lists: dict[str, list[str]] = {}
        self.expirations: list[tuple[str, int]] = []
        self.deleted: list[str] = []
        self.fail = False

    async def lrange(self, key, start, end):
        if self.fail:
            raise RedisError("down")
        items = self.lists.get(key, [])
        length = len(items)
        if start < 0:
            start = max(length + start, 0)
        if end < 0:
            end = length + end + 1
        return items[start:end]

    def pipeline(self):
        return FakePipeline(self)

    async def delete(self, key):
        if self.fail:
            raise RedisError("down")
        self.deleted.append(key)
        self.lists.pop(key, None)

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_append_and_read_history(settings):
    client = FakeRedis()
    memory = ChatMemoryService(settings, client=client)

    await memory.append("s1", ChatMessage(role="user", content="Hello"))
    await memory.append("s1", ChatMessage(role="assistant", content="Hi"))

    history = await memory.get_history("s1")

    assert history == [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi"),
    ]
    assert client.expirations[0][1] == settings.chat_memory_ttl_seconds


@pytest.mark.asyncio
async def test_history_is_trimmed(settings):
    client = FakeRedis()
    memory = ChatMemoryService(
        settings.model_copy(update={"chat_memory_max_messages": 2}),
        client=client,
    )

    for index in range(5):
        await memory.append("s1", ChatMessage(role="user", content=str(index)))

    history = await memory.get_history("s1")

    assert [message.content for message in history] == ["3", "4"]


@pytest.mark.asyncio
async def test_malformed_entries_are_skipped(settings):
    client = FakeRedis()
    client.lists["chat:memory:s1"] = [
        "not json",
        json.dumps({"role": "user", "content": "valid"}),
    ]
    memory = ChatMemoryService(settings, client=client)

    history = await memory.get_history("s1")

    assert history == [ChatMessage(role="user", content="valid")]


@pytest.mark.asyncio
async def test_clear_removes_session(settings):
    client = FakeRedis()
    memory = ChatMemoryService(settings, client=client)
    await memory.append("s1", ChatMessage(role="user", content="Hello"))

    await memory.clear("s1")

    assert client.deleted == ["chat:memory:s1"]
    assert await memory.get_history("s1") == []


@pytest.mark.asyncio
async def test_redis_failure_is_wrapped(settings):
    client = FakeRedis()
    client.fail = True
    memory = ChatMemoryService(settings, client=client)

    with pytest.raises(ChatMemoryError):
        await memory.get_history("s1")
