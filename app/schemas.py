from datetime import date, datetime, time
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"


class RewrittenQuery(BaseModel):
    query: str = Field(
        description=(
            "A standalone search query that preserves the user's intent and "
            "any context from the conversation."
        )
    )


class BookingIntent(BaseModel):
    is_booking_request: bool = Field(
        description="True only when the user wants to schedule, reschedule, or cancel an interview."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the classification, from 0 to 1.",
    )
    reasoning: str = Field(
        default="",
        description="One short sentence explaining the classification.",
    )


class InterviewBookingExtraction(BaseModel):
    name: str | None = Field(
        default=None, description="Full name of the candidate booking the interview."
    )
    email: str | None = Field(
        default=None, description="Email address for the interview confirmation."
    )
    interview_date: str | None = Field(
        default=None, description="Interview date in ISO 8601 format (YYYY-MM-DD)."
    )
    interview_time: str | None = Field(
        default=None, description="Interview time in 24-hour ISO 8601 format (HH:MM)."
    )
    position: str | None = Field(
        default=None, description="Role or position the candidate is interviewing for."
    )
    notes: str | None = Field(default=None, description="Any additional notes or special requests.")
    missing_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Required fields not provided by the user: name, email, interview_date, interview_time."
        ),
    )
    summary: str = Field(
        default="", description="A short summary of the requested interview booking."
    )


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    top_k: int = Field(default=4, ge=1, le=20)


class Source(BaseModel):
    id: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InterviewBookingResult(BaseModel):
    id: UUID | None = None
    status: Literal["confirmed", "incomplete"]
    name: str | None = None
    email: str | None = None
    interview_date: date | None = None
    interview_time: time | None = None
    position: str | None = None
    notes: str | None = None
    missing_fields: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    rewritten_query: str
    is_booking_request: bool
    booking: InterviewBookingResult | None = None
    sources: list[Source] = Field(default_factory=list)


class DocumentIngestResponse(BaseModel):
    document_id: UUID
    filename: str
    content_type: str | None
    size_bytes: int
    chunking_strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    embedding_model: str
    embedding_dimensions: int


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    content_type: str | None
    size_bytes: int
    chunking_strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    status: str
    created_at: datetime


class DocumentChunkSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    chunk_index: int
    vector_id: str
    text: str
    chunk_metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentDetail(DocumentSummary):
    chunks: list[DocumentChunkSummary] = Field(default_factory=list)


class BookingRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: str
    name: str
    email: str
    interview_date: date
    interview_time: time
    position: str | None
    notes: str | None
    status: str
    created_at: datetime


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class HealthResponse(BaseModel):
    status: str
    app: str
    model: str
    embedding_model: str
    embedding_dimensions: int
    qdrant_collection: str
    gemini_configured: bool
    database: str
    redis_configured: bool
