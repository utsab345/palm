from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.repositories import BookingRepository, DocumentRepository
from app.db.session import get_db_session
from app.integrations.llm import GeminiLLM
from app.services.booking_service import BookingService
from app.services.chat_service import ChatService
from app.services.document_ingestion_service import DocumentIngestionService
from app.services.embedding_service import EmbeddingService
from app.services.memory_service import ChatMemoryService
from app.services.rag_service import RagService
from app.services.vector_store import VectorStore


@lru_cache(maxsize=1)
def _llm() -> GeminiLLM:
    return GeminiLLM(get_settings())


@lru_cache(maxsize=1)
def _embeddings() -> EmbeddingService:
    return EmbeddingService(get_settings())


@lru_cache(maxsize=1)
def _vector_store() -> VectorStore:
    return VectorStore(get_settings())


@lru_cache(maxsize=1)
def _memory() -> ChatMemoryService:
    return ChatMemoryService(get_settings())


def get_llm() -> GeminiLLM:
    return _llm()


def get_embeddings() -> EmbeddingService:
    return _embeddings()


def get_vector_store() -> VectorStore:
    return _vector_store()


def get_memory() -> ChatMemoryService:
    return _memory()


def get_rag_service(
    settings: Settings = Depends(get_settings),
    embeddings: EmbeddingService = Depends(get_embeddings),
    vector_store: VectorStore = Depends(get_vector_store),
) -> RagService:
    return RagService(settings, embeddings, vector_store)


def get_document_repository(
    session: AsyncSession = Depends(get_db_session),
) -> DocumentRepository:
    return DocumentRepository(session)


def get_booking_repository(
    session: AsyncSession = Depends(get_db_session),
) -> BookingRepository:
    return BookingRepository(session)


def get_document_service(
    settings: Settings = Depends(get_settings),
    embeddings: EmbeddingService = Depends(get_embeddings),
    vector_store: VectorStore = Depends(get_vector_store),
    repository: DocumentRepository = Depends(get_document_repository),
) -> DocumentIngestionService:
    return DocumentIngestionService(settings, embeddings, vector_store, repository)


def get_booking_service(
    repository: BookingRepository = Depends(get_booking_repository),
) -> BookingService:
    return BookingService(repository)


def get_chat_service(
    llm: GeminiLLM = Depends(get_llm),
    rag: RagService = Depends(get_rag_service),
    memory: ChatMemoryService = Depends(get_memory),
    bookings: BookingService = Depends(get_booking_service),
) -> ChatService:
    return ChatService(llm, rag, memory, bookings)
