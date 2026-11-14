import uuid
from collections.abc import Sequence
from datetime import date, time

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import PersistenceError
from app.db.models import Document, DocumentChunk, InterviewBooking


class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, document: Document, chunks: Sequence[DocumentChunk]) -> Document:
        self._session.add(document)
        self._session.add_all(chunks)
        try:
            await self._session.commit()
            await self._session.refresh(document)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise PersistenceError("Failed to save document metadata") from exc
        return document

    async def list(self, limit: int = 50, offset: int = 0) -> Sequence[Document]:
        try:
            result = await self._session.execute(
                select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
            )
        except SQLAlchemyError as exc:
            raise PersistenceError("Failed to list documents") from exc
        return result.scalars().all()

    async def get(self, document_id: uuid.UUID) -> Document | None:
        try:
            return await self._session.get(Document, document_id)
        except SQLAlchemyError as exc:
            raise PersistenceError("Failed to load document") from exc

    async def get_chunks(self, document_id: uuid.UUID) -> Sequence[DocumentChunk]:
        try:
            result = await self._session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
            )
        except SQLAlchemyError as exc:
            raise PersistenceError("Failed to load document chunks") from exc
        return result.scalars().all()


class BookingRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(
        self,
        *,
        session_id: str,
        name: str,
        email: str,
        interview_date: date,
        interview_time: time,
        position: str | None,
        notes: str | None,
    ) -> InterviewBooking:
        booking = InterviewBooking(
            session_id=session_id,
            name=name,
            email=email,
            interview_date=interview_date,
            interview_time=interview_time,
            position=position,
            notes=notes,
        )
        self._session.add(booking)
        try:
            await self._session.commit()
            await self._session.refresh(booking)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise PersistenceError("Failed to save interview booking") from exc
        return booking

    async def get(self, booking_id: uuid.UUID) -> InterviewBooking | None:
        try:
            return await self._session.get(InterviewBooking, booking_id)
        except SQLAlchemyError as exc:
            raise PersistenceError("Failed to load interview booking") from exc

    async def list_by_session(self, session_id: str) -> Sequence[InterviewBooking]:
        try:
            result = await self._session.execute(
                select(InterviewBooking)
                .where(InterviewBooking.session_id == session_id)
                .order_by(InterviewBooking.created_at.desc())
            )
        except SQLAlchemyError as exc:
            raise PersistenceError("Failed to list interview bookings") from exc
        return result.scalars().all()
