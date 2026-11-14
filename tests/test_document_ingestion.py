import uuid

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.errors import (
    DocumentProcessingError,
    DocumentTooLargeError,
    UnsupportedDocumentError,
)
from app.db.base import Base
from app.db.repositories import DocumentRepository
from app.schemas import ChunkingStrategy
from app.services import document_ingestion_service as ingestion_module
from app.services.document_ingestion_service import DocumentIngestionService


class FakeEmbeddings:
    def __init__(self):
        self.calls: list[list[str]] = []

    async def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    async def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]


class FakeVectorStore:
    def __init__(self):
        self.ensured: list[int] = []
        self.points = []

    async def ensure_collection(self, dimensions):
        self.ensured.append(dimensions)

    async def upsert(self, points):
        self.points = points


class FakePdfPage:
    def extract_text(self):
        return "Extracted PDF interview policy text."


class FakePdfReader:
    def __init__(self, stream):
        self.is_encrypted = False
        self.pages = [FakePdfPage(), FakePdfPage()]


def build_session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
    )
    return engine, async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
async def repository():
    engine, factory = build_session_factory()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with factory() as session:
        yield DocumentRepository(session)
    await engine.dispose()


def test_extract_txt_handles_utf8_bom():
    text = DocumentIngestionService.extract_text(
        "notes.txt", "\ufeffInterview details".encode("utf-8")
    )

    assert text == "Interview details"


def test_extract_pdf_uses_pdf_reader(monkeypatch):
    monkeypatch.setattr(ingestion_module, "PdfReader", FakePdfReader)

    text = DocumentIngestionService.extract_text("policy.pdf", b"%PDF-fake")

    assert "Extracted PDF" in text


def test_unsupported_extension_is_rejected():
    with pytest.raises(UnsupportedDocumentError):
        DocumentIngestionService.extract_text("resume.docx", b"data")


def test_empty_text_is_rejected():
    with pytest.raises(DocumentProcessingError):
        DocumentIngestionService.extract_text("empty.txt", b"   ")


@pytest.mark.asyncio
async def test_ingest_stores_metadata_and_vectors(settings, repository):
    embeddings = FakeEmbeddings()
    store = FakeVectorStore()
    service = DocumentIngestionService(settings, embeddings, store, repository)
    text = " ".join(f"word{index}" for index in range(120)).encode()

    response = await service.ingest(
        filename="policy.txt",
        content_type="text/plain",
        content=text,
        strategy=ChunkingStrategy.FIXED,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert response.chunk_count > 1
    assert response.document_id is not None
    assert store.ensured == [settings.gemini_embedding_dimensions]
    assert len(store.points) == response.chunk_count
    assert store.points[0].payload["metadata"]["document_id"] == str(response.document_id)

    stored = await repository.get(response.document_id)
    chunks = await repository.get_chunks(response.document_id)

    assert stored is not None
    assert stored.filename == "policy.txt"
    assert stored.chunk_count == response.chunk_count
    assert stored.chunking_strategy == "fixed"
    assert len(chunks) == response.chunk_count
    assert chunks[0].chunk_index == 0

    summaries = await service.list_documents()
    detail = await service.get_document(response.document_id)

    assert summaries[0].id == response.document_id
    assert detail is not None
    assert len(detail.chunks) == response.chunk_count


@pytest.mark.asyncio
async def test_ingest_rejects_empty_and_oversized_files(settings, repository):
    service = DocumentIngestionService(settings, FakeEmbeddings(), FakeVectorStore(), repository)

    with pytest.raises(DocumentProcessingError):
        await service.ingest(
            filename="empty.txt",
            content_type="text/plain",
            content=b"",
            strategy=ChunkingStrategy.FIXED,
        )

    small_limit = settings.model_copy(update={"max_upload_bytes": 5})
    service = DocumentIngestionService(small_limit, FakeEmbeddings(), FakeVectorStore(), repository)
    with pytest.raises(DocumentTooLargeError):
        await service.ingest(
            filename="big.txt",
            content_type="text/plain",
            content=b"too large",
            strategy=ChunkingStrategy.FIXED,
        )


@pytest.mark.asyncio
async def test_small_chunk_size_caps_default_overlap(settings, repository):
    service = DocumentIngestionService(settings, FakeEmbeddings(), FakeVectorStore(), repository)

    response = await service.ingest(
        filename="small.txt",
        content_type="text/plain",
        content=b"word " * 50,
        strategy=ChunkingStrategy.FIXED,
        chunk_size=40,
    )

    assert response.chunk_size == 40
    assert response.chunk_overlap == 39
    assert response.chunk_count > 1


@pytest.mark.asyncio
async def test_get_missing_document_returns_none(settings, repository):
    service = DocumentIngestionService(settings, FakeEmbeddings(), FakeVectorStore(), repository)

    assert await service.get_document(uuid.uuid4()) is None
