import uuid

from app.integrations.llm import GeminiLLM
from app.schemas import ChatMessage, ChatRequest, ChatResponse
from app.services.booking_service import BookingService
from app.services.memory_service import ChatMemoryService
from app.services.rag_service import RagService


class ChatService:
    def __init__(
        self,
        llm: GeminiLLM,
        rag: RagService,
        memory: ChatMemoryService,
        bookings: BookingService,
    ):
        self._llm = llm
        self._rag = rag
        self._memory = memory
        self._bookings = bookings

    async def chat(self, request: ChatRequest) -> ChatResponse:
        session_id = request.session_id or uuid.uuid4().hex
        history = await self._memory.get_history(session_id)
        await self._memory.append(session_id, ChatMessage(role="user", content=request.message))

        rewritten_query = await self._llm.rewrite_query(request.message, history)
        intent = await self._llm.detect_booking_intent(request.message, history)
        sources = await self._rag.search(rewritten_query, request.top_k)

        answer = await self._llm.generate_answer(
            message=request.message,
            history=history,
            sources=sources,
            is_booking_request=intent.is_booking_request,
        )

        booking = None
        if intent.is_booking_request:
            extraction = await self._llm.extract_interview_booking(request.message, history)
            booking = await self._bookings.handle(session_id, extraction)

        await self._memory.append(session_id, ChatMessage(role="assistant", content=answer))

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            rewritten_query=rewritten_query,
            is_booking_request=intent.is_booking_request,
            booking=booking,
            sources=sources,
        )
