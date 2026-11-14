import uuid
from datetime import date, time

import pytest

from app.schemas import (
    BookingIntent,
    ChatMessage,
    ChatRequest,
    InterviewBookingExtraction,
    InterviewBookingResult,
    Source,
)
from app.services.chat_service import ChatService


class FakeLLM:
    def __init__(self, is_booking: bool = True):
        self.is_booking = is_booking
        self.rewrite_history = None
        self.intent_history = None
        self.extract_history = None
        self.answer_sources = None

    async def rewrite_query(self, message, history):
        self.rewrite_history = history
        return f"rewritten: {message}"

    async def detect_booking_intent(self, message, history):
        self.intent_history = history
        return BookingIntent(
            is_booking_request=self.is_booking,
            confidence=0.95,
            reasoning="test",
        )

    async def extract_interview_booking(self, message, history):
        self.extract_history = history
        return InterviewBookingExtraction(
            name="Ada Lovelace",
            email="ada@example.com",
            interview_date="2026-06-01",
            interview_time="10:30",
        )

    async def generate_answer(self, message, history, sources, is_booking_request):
        self.answer_sources = sources
        return "Interview scheduled."


class FakeRag:
    async def search(self, query, top_k=4):
        return [Source(id="1", text="Policy text", score=0.9, metadata={})]


class FakeMemory:
    def __init__(self):
        self.history = [ChatMessage(role="user", content="Earlier turn")]
        self.appended: list[tuple[str, ChatMessage]] = []

    async def get_history(self, session_id):
        return self.history

    async def append(self, session_id, message):
        self.appended.append((session_id, message))


class FakeBookings:
    def __init__(self):
        self.calls = []

    async def handle(self, session_id, extraction):
        self.calls.append((session_id, extraction))
        return InterviewBookingResult(
            id=uuid.uuid4(),
            status="confirmed",
            name=extraction.name,
            email=extraction.email,
            interview_date=date(2026, 6, 1),
            interview_time=time(10, 30),
        )


def build_service(is_booking: bool = True):
    llm = FakeLLM(is_booking)
    memory = FakeMemory()
    bookings = FakeBookings()
    service = ChatService(
        llm=llm,  # type: ignore[arg-type]
        rag=FakeRag(),  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        bookings=bookings,  # type: ignore[arg-type]
    )
    return service, llm, memory, bookings


@pytest.mark.asyncio
async def test_chat_orchestrates_memory_rag_and_booking():
    service, llm, memory, bookings = build_service()

    response = await service.chat(ChatRequest(session_id="session-1", message="Book an interview"))

    assert response.session_id == "session-1"
    assert response.answer == "Interview scheduled."
    assert response.rewritten_query == "rewritten: Book an interview"
    assert response.is_booking_request is True
    assert response.booking is not None
    assert response.booking.status == "confirmed"
    assert llm.rewrite_history == memory.history
    assert llm.intent_history == memory.history
    assert llm.extract_history == memory.history
    assert llm.answer_sources == response.sources
    assert [message.role for _, message in memory.appended] == [
        "user",
        "assistant",
    ]
    assert bookings.calls[0][0] == "session-1"


@pytest.mark.asyncio
async def test_chat_generates_session_id_when_missing():
    service, _, memory, _ = build_service()

    response = await service.chat(ChatRequest(message="Hello"))

    assert response.session_id
    assert all(session_id == response.session_id for session_id, _ in memory.appended)


@pytest.mark.asyncio
async def test_non_booking_skips_extraction_and_persistence():
    service, _, _, bookings = build_service(is_booking=False)

    response = await service.chat(ChatRequest(message="What is the process?"))

    assert response.is_booking_request is False
    assert response.booking is None
    assert bookings.calls == []
