from datetime import UTC, date, datetime, time
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.core.errors import LLMRateLimitError, UnsupportedDocumentError
from app.dependencies import (
    get_booking_repository,
    get_chat_service,
    get_document_service,
    get_memory,
    get_rag_service,
)
from app.main import create_app
from app.schemas import (
    ChatMessage,
    ChatResponse,
    ChunkingStrategy,
    DocumentChunkSummary,
    DocumentDetail,
    DocumentIngestResponse,
    DocumentSummary,
    InterviewBookingResult,
    Source,
)


def build_settings() -> Settings:
    return Settings(
        gemini_api_key="test-key",
        gemini_model="gemini-2.5-flash",
        gemini_embedding_model="gemini-embedding-001",
        gemini_embedding_dimensions=4,
    )


class FakeChatService:
    def __init__(self):
        self.failure = None

    async def chat(self, request):
        if self.failure is not None:
            raise self.failure
        return ChatResponse(
            session_id=request.session_id or "generated-session",
            answer="Interviews are held on weekdays.",
            rewritten_query=request.message,
            is_booking_request=False,
            sources=[
                Source(
                    id="1",
                    text="Interviews are held on weekdays.",
                    score=0.9,
                    metadata={},
                )
            ],
        )


class FakeDocumentService:
    def __init__(self):
        self.get_returns_none = False
        self.failure = None

    async def ingest(self, **kwargs):
        if self.failure is not None:
            raise self.failure
        return DocumentIngestResponse(
            document_id=uuid4(),
            filename=kwargs["filename"],
            content_type=kwargs["content_type"],
            size_bytes=len(kwargs["content"]),
            chunking_strategy=kwargs["strategy"],
            chunk_size=kwargs["chunk_size"] or 1000,
            chunk_overlap=kwargs["chunk_overlap"] or 200,
            chunk_count=2,
            embedding_model="gemini-embedding-001",
            embedding_dimensions=4,
        )

    async def list_documents(self, limit=50, offset=0):
        return [self._summary()]

    async def get_document(self, document_id):
        if self.get_returns_none:
            return None
        return DocumentDetail(
            **self._summary().model_dump(),
            chunks=[
                DocumentChunkSummary(
                    id=uuid4(),
                    chunk_index=0,
                    vector_id="vector-1",
                    text="Interview policy.",
                    chunk_metadata={"source": "faq"},
                )
            ],
        )

    @staticmethod
    def _summary():
        return DocumentSummary(
            id=uuid4(),
            filename="faq.txt",
            content_type="text/plain",
            size_bytes=100,
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=1000,
            chunk_overlap=200,
            chunk_count=2,
            status="indexed",
            created_at=datetime.now(UTC),
        )


class FakeMemory:
    def __init__(self):
        self.history = [ChatMessage(role="user", content="Hello")]
        self.cleared = []

    async def get_history(self, session_id):
        return self.history

    async def clear(self, session_id):
        self.cleared.append(session_id)


class FakeBookingRepository:
    async def list_by_session(self, session_id):
        return []

    async def get(self, booking_id):
        return None


class FakeRagService:
    async def search(self, query, top_k=4):
        return [Source(id="1", text=query, score=0.8, metadata={})]


def build_client(
    *,
    chat_service: FakeChatService | None = None,
    document_service: FakeDocumentService | None = None,
    memory: FakeMemory | None = None,
) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = build_settings
    app.dependency_overrides[get_chat_service] = lambda: chat_service or FakeChatService()
    app.dependency_overrides[get_document_service] = lambda: (
        document_service or FakeDocumentService()
    )
    app.dependency_overrides[get_memory] = lambda: memory or FakeMemory()
    app.dependency_overrides[get_booking_repository] = FakeBookingRepository
    app.dependency_overrides[get_rag_service] = FakeRagService
    return TestClient(app)


def test_health_endpoint():
    client = build_client()
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "gemini-2.5-flash"
    assert body["embedding_model"] == "gemini-embedding-001"
    assert body["embedding_dimensions"] == 4
    assert body["gemini_configured"] is True
    assert body["database"] == "sqlite+aiosqlite"


def test_chat_endpoint_returns_session_id():
    client = build_client()
    response = client.post("/chat", json={"message": "When are interviews?"})

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "generated-session"
    assert body["answer"] == "Interviews are held on weekdays."
    assert body["is_booking_request"] is False
    assert len(body["sources"]) == 1


def test_chat_returns_booking_when_detected():
    class BookingChatService(FakeChatService):
        async def chat(self, request):
            return ChatResponse(
                session_id=request.session_id or "booked-session",
                answer="What is your email?",
                rewritten_query="book an interview",
                is_booking_request=True,
                booking=InterviewBookingResult(
                    id=uuid4(),
                    status="confirmed",
                    name="Ada Lovelace",
                    email="ada@example.com",
                    interview_date=date(2026, 6, 1),
                    interview_time=time(10, 30),
                ),
                sources=[],
            )

    client = build_client(chat_service=BookingChatService())
    response = client.post("/chat", json={"message": "Book an interview"})

    assert response.status_code == 200
    body = response.json()
    assert body["is_booking_request"] is True
    assert body["booking"]["status"] == "confirmed"
    assert body["booking"]["name"] == "Ada Lovelace"


def test_chat_validates_empty_message():
    client = build_client()
    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 422


def test_chat_rejects_whitespace_only_message():
    client = build_client()
    response = client.post("/chat", json={"message": "   "})

    assert response.status_code == 422


def test_rate_limit_error_is_returned_as_429():
    service = FakeChatService()
    service.failure = LLMRateLimitError("Gemini rate limit exceeded")
    client = build_client(chat_service=service)
    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 429
    assert response.json()["code"] == "llm_rate_limited"


def test_upload_document_endpoint():
    client = build_client()
    response = client.post(
        "/documents",
        files={"file": ("faq.txt", b"Interview policy text", "text/plain")},
        data={
            "strategy": "fixed",
            "chunk_size": "100",
            "chunk_overlap": "10",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "faq.txt"
    assert body["chunking_strategy"] == "fixed"
    assert body["chunk_size"] == 100
    assert body["chunk_count"] == 2


def test_upload_rejects_unsupported_file_type():
    service = FakeDocumentService()
    service.failure = UnsupportedDocumentError("Only .pdf and .txt files are supported")
    client = build_client(document_service=service)
    response = client.post(
        "/documents",
        files={"file": ("resume.docx", b"data", "application/msword")},
    )

    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_document_type"


def test_list_and_get_documents():
    client = build_client()

    listed = client.get("/documents")
    assert listed.status_code == 200
    assert listed.json()[0]["filename"] == "faq.txt"

    detail = client.get(f"/documents/{uuid4()}")
    assert detail.status_code == 200
    assert len(detail.json()["chunks"]) == 1


def test_missing_document_returns_404():
    service = FakeDocumentService()
    service.get_returns_none = True
    client = build_client(document_service=service)

    response = client.get(f"/documents/{uuid4()}")

    assert response.status_code == 404


def test_chat_memory_endpoints():
    memory = FakeMemory()
    client = build_client(memory=memory)

    history = client.get("/chat/session-1/history")
    assert history.status_code == 200
    assert history.json()[0]["content"] == "Hello"

    cleared = client.delete("/chat/session-1/history")
    assert cleared.status_code == 204
    assert memory.cleared == ["session-1"]


def test_missing_booking_returns_404():
    client = build_client()
    response = client.get(f"/bookings/{uuid4()}")

    assert response.status_code == 404


def test_search_endpoint():
    client = build_client()
    response = client.post("/search", json={"query": "interview", "top_k": 3})

    assert response.status_code == 200
    assert response.json()[0]["text"] == "interview"
