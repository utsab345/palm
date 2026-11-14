from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)

from app.config import Settings, get_settings
from app.db.repositories import BookingRepository
from app.dependencies import (
    get_booking_repository,
    get_chat_service,
    get_document_service,
    get_memory,
    get_rag_service,
)
from app.schemas import (
    BookingRecord,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChunkingStrategy,
    DocumentDetail,
    DocumentIngestResponse,
    DocumentSummary,
    HealthResponse,
    SearchRequest,
    Source,
)
from app.services.chat_service import ChatService
from app.services.document_ingestion_service import DocumentIngestionService
from app.services.memory_service import ChatMemoryService
from app.services.rag_service import RagService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        model=settings.gemini_model,
        embedding_model=settings.gemini_embedding_model,
        embedding_dimensions=settings.gemini_embedding_dimensions,
        qdrant_collection=settings.qdrant_collection,
        gemini_configured=settings.gemini_api_key_value is not None,
        database=settings.database_url.split(":", 1)[0],
        redis_configured=bool(settings.redis_url),
    )


@router.post(
    "/documents",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    strategy: ChunkingStrategy = Form(ChunkingStrategy.RECURSIVE),
    chunk_size: int | None = Form(default=None),
    chunk_overlap: int | None = Form(default=None),
    settings: Settings = Depends(get_settings),
    service: DocumentIngestionService = Depends(get_document_service),
) -> DocumentIngestResponse:
    content = await file.read(settings.max_upload_bytes + 1)
    return await service.ingest(
        filename=file.filename or "upload",
        content_type=file.content_type,
        content=content,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: DocumentIngestionService = Depends(get_document_service),
) -> list[DocumentSummary]:
    return await service.list_documents(limit=limit, offset=offset)


@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: UUID,
    service: DocumentIngestionService = Depends(get_document_service),
) -> DocumentDetail:
    document = await service.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return await service.chat(request)


@router.get("/chat/{session_id}/history", response_model=list[ChatMessage])
async def get_chat_history(
    session_id: str,
    memory: ChatMemoryService = Depends(get_memory),
) -> list[ChatMessage]:
    return await memory.get_history(session_id)


@router.delete("/chat/{session_id}/history", status_code=status.HTTP_204_NO_CONTENT)
async def clear_chat_history(
    session_id: str,
    memory: ChatMemoryService = Depends(get_memory),
) -> Response:
    await memory.clear(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/bookings", response_model=list[BookingRecord])
async def list_bookings(
    session_id: str = Query(min_length=1),
    repository: BookingRepository = Depends(get_booking_repository),
) -> list[BookingRecord]:
    bookings = await repository.list_by_session(session_id)
    return [BookingRecord.model_validate(booking) for booking in bookings]


@router.get("/bookings/{booking_id}", response_model=BookingRecord)
async def get_booking(
    booking_id: UUID,
    repository: BookingRepository = Depends(get_booking_repository),
) -> BookingRecord:
    booking = await repository.get(booking_id)
    if booking is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Booking not found",
        )
    return BookingRecord.model_validate(booking)


@router.post("/search", response_model=list[Source])
async def search(
    request: SearchRequest,
    rag: RagService = Depends(get_rag_service),
) -> list[Source]:
    return await rag.search(request.query, request.top_k)
