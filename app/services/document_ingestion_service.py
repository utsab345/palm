import re
import uuid
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from qdrant_client.http import models as qdrant_models

from app.config import Settings
from app.core.errors import (
    DocumentProcessingError,
    DocumentTooLargeError,
    UnsupportedDocumentError,
)
from app.db.models import Document, DocumentChunk
from app.db.repositories import DocumentRepository
from app.schemas import (
    ChunkingStrategy,
    DocumentChunkSummary,
    DocumentDetail,
    DocumentIngestResponse,
    DocumentSummary,
)
from app.services.chunking import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class DocumentIngestionService:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
        repository: DocumentRepository,
        chunker: ChunkingService | None = None,
    ):
        self._settings = settings
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._repository = repository
        self._chunker = chunker or ChunkingService()

    async def ingest(
        self,
        *,
        filename: str,
        content_type: str | None,
        content: bytes,
        strategy: ChunkingStrategy,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> DocumentIngestResponse:
        if not content:
            raise DocumentProcessingError("Uploaded file is empty")
        if len(content) > self._settings.max_upload_bytes:
            raise DocumentTooLargeError(
                f"File exceeds the {self._settings.max_upload_bytes} byte limit"
            )

        size_bytes = len(content)
        resolved_size = chunk_size or self._settings.default_chunk_size
        resolved_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else min(
                self._settings.default_chunk_overlap,
                max(0, resolved_size - 1),
            )
        )

        text = self.extract_text(filename, content)
        chunks = self._chunker.chunk(text, strategy, resolved_size, resolved_overlap)
        if not chunks:
            raise DocumentProcessingError("Document produced no text chunks")

        await self._vector_store.ensure_collection(self._settings.gemini_embedding_dimensions)
        document_id = uuid.uuid4()
        vectors = await self._embeddings.embed_documents([chunk.text for chunk in chunks])

        vector_ids = [str(uuid.uuid4()) for _ in chunks]
        points = [
            qdrant_models.PointStruct(
                id=vector_id,
                vector=vector,
                payload={
                    "text": chunk.text,
                    "metadata": {
                        "document_id": str(document_id),
                        "filename": filename,
                        "chunk_index": chunk.index,
                        "chunking_strategy": strategy.value,
                        "chunk_size": resolved_size,
                        "chunk_overlap": resolved_overlap,
                    },
                },
            )
            for chunk, vector_id, vector in zip(chunks, vector_ids, vectors, strict=True)
        ]
        await self._vector_store.upsert(points)

        document = Document(
            id=document_id,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            chunking_strategy=strategy.value,
            chunk_size=resolved_size,
            chunk_overlap=resolved_overlap,
            chunk_count=len(chunks),
            status="indexed",
        )
        chunk_rows = [
            DocumentChunk(
                document_id=document_id,
                chunk_index=chunk.index,
                vector_id=vector_id,
                text=chunk.text,
                chunk_metadata={
                    "filename": filename,
                    "chunk_index": chunk.index,
                    "chunking_strategy": strategy.value,
                },
            )
            for chunk, vector_id in zip(chunks, vector_ids, strict=True)
        ]
        await self._repository.create(document, chunk_rows)

        return DocumentIngestResponse(
            document_id=document_id,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            chunking_strategy=strategy,
            chunk_size=resolved_size,
            chunk_overlap=resolved_overlap,
            chunk_count=len(chunks),
            embedding_model=self._settings.gemini_embedding_model,
            embedding_dimensions=self._settings.gemini_embedding_dimensions,
        )

    async def list_documents(self, limit: int = 50, offset: int = 0) -> list[DocumentSummary]:
        documents = await self._repository.list(limit=limit, offset=offset)
        return [DocumentSummary.model_validate(document) for document in documents]

    async def get_document(self, document_id: uuid.UUID) -> DocumentDetail | None:
        document = await self._repository.get(document_id)
        if document is None:
            return None
        chunks = await self._repository.get_chunks(document_id)
        return DocumentDetail(
            **DocumentSummary.model_validate(document).model_dump(),
            chunks=[DocumentChunkSummary.model_validate(chunk) for chunk in chunks],
        )

    @staticmethod
    def extract_text(filename: str, content: bytes) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise UnsupportedDocumentError("Only .pdf and .txt files are supported")

        if suffix == ".txt":
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = content.decode("latin-1")
        else:
            text = DocumentIngestionService._extract_pdf(content)

        cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not cleaned:
            raise DocumentProcessingError("No extractable text found in document")
        return cleaned

    @staticmethod
    def _extract_pdf(content: bytes) -> str:
        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as exc:
                    raise DocumentProcessingError("Encrypted PDFs are not supported") from exc
            pages = [page.extract_text() or "" for page in reader.pages]
        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError("Could not read PDF document") from exc

        if not pages:
            raise DocumentProcessingError("PDF contains no pages")
        return "\n\n".join(pages)
