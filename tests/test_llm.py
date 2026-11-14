import json
from types import SimpleNamespace

import pytest
from google.genai import errors

from app.core.errors import LLMError, LLMRateLimitError, LLMResponseError
from app.integrations.llm import GeminiLLM
from app.schemas import (
    BookingIntent,
    ChatMessage,
    InterviewBookingExtraction,
    RewrittenQuery,
)
from tests.conftest import FakeClient


def build_llm(settings):
    client = FakeClient()
    return GeminiLLM(settings, client=client), client


@pytest.mark.asyncio
async def test_rewrite_query_uses_structured_schema(settings):
    llm, client = build_llm(settings)
    client.models.generate_responses.append(
        SimpleNamespace(parsed=RewrittenQuery(query="interview policy"), text=None)
    )

    result = await llm.rewrite_query(
        "what about that?",
        [ChatMessage(role="user", content="Tell me about interviews")],
    )

    assert result == "interview policy"
    call = client.models.generate_calls[0]
    assert call["model"] == settings.gemini_model
    assert call["config"].response_schema is RewrittenQuery
    assert call["config"].response_mime_type == "application/json"


@pytest.mark.asyncio
async def test_rewrite_query_falls_back_to_message_when_empty(settings):
    llm, client = build_llm(settings)
    client.models.generate_responses.append(
        SimpleNamespace(parsed=RewrittenQuery(query="   "), text=None)
    )

    result = await llm.rewrite_query("original message", [])

    assert result == "original message"


@pytest.mark.asyncio
async def test_detect_booking_intent_accepts_dict_payload(settings):
    llm, client = build_llm(settings)
    client.models.generate_responses.append(
        SimpleNamespace(
            parsed={
                "is_booking_request": True,
                "confidence": 0.9,
                "reasoning": "User wants to schedule an interview.",
            },
            text=None,
        )
    )

    intent = await llm.detect_booking_intent(
        "10 AM works", [ChatMessage(role="user", content="Book an interview")]
    )

    assert intent == BookingIntent(
        is_booking_request=True,
        confidence=0.9,
        reasoning="User wants to schedule an interview.",
    )
    assert "Book an interview" in client.models.generate_calls[0]["contents"]


@pytest.mark.asyncio
async def test_extract_interview_booking_parses_fenced_json(settings):
    llm, client = build_llm(settings)
    payload = {
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "interview_date": "2026-06-01",
        "interview_time": "10:30",
        "position": "Backend Engineer",
        "notes": None,
        "missing_fields": [],
        "summary": "Backend interview on June 1.",
    }
    client.models.generate_responses.append(
        SimpleNamespace(parsed=None, text=f"```json\n{json.dumps(payload)}\n```")
    )

    details = await llm.extract_interview_booking("Book me for June 1", [])

    assert isinstance(details, InterviewBookingExtraction)
    assert details.name == "Ada Lovelace"
    assert details.interview_date == "2026-06-01"
    assert details.interview_time == "10:30"


@pytest.mark.asyncio
async def test_malformed_json_raises_response_error(settings):
    llm, client = build_llm(settings)
    client.models.generate_responses.append(SimpleNamespace(parsed=None, text="not json at all"))

    with pytest.raises(LLMResponseError):
        await llm.detect_booking_intent("hello", [])


@pytest.mark.asyncio
async def test_invalid_schema_raises_response_error(settings):
    llm, client = build_llm(settings)
    client.models.generate_responses.append(
        SimpleNamespace(parsed={"is_booking_request": "yes"}, text=None)
    )

    with pytest.raises(LLMResponseError):
        await llm.detect_booking_intent("hello", [])


@pytest.mark.asyncio
async def test_empty_answer_raises_response_error(settings):
    llm, client = build_llm(settings)
    client.models.generate_responses.append(SimpleNamespace(parsed=None, text="   "))

    with pytest.raises(LLMResponseError):
        await llm.generate_answer("hi", [], [], False)


@pytest.mark.asyncio
async def test_generate_answer_returns_text_and_includes_sources(settings):
    llm, client = build_llm(settings)
    client.models.generate_responses.append(
        SimpleNamespace(parsed=None, text="Interviews are held on weekdays.")
    )

    answer = await llm.generate_answer(
        "When are interviews?",
        [],
        [
            SimpleNamespace(
                text="Interviews are held on weekdays.",
                id="1",
                score=0.9,
                metadata={},
            )
        ],
        False,
    )

    assert answer == "Interviews are held on weekdays."
    prompt = client.models.generate_calls[0]["contents"]
    assert "Interviews are held on weekdays." in prompt


@pytest.mark.asyncio
async def test_rate_limit_error_is_mapped(settings):
    llm, client = build_llm(settings)
    client.models.generate_responses.append(
        errors.APIError(429, {"error": {"code": 429, "message": "quota"}})
    )

    with pytest.raises(LLMRateLimitError):
        await llm.detect_booking_intent("hello", [])


@pytest.mark.asyncio
async def test_server_error_maps_to_llm_error(settings):
    llm, client = build_llm(settings)
    client.models.generate_responses.append(
        errors.APIError(500, {"error": {"code": 500, "message": "boom"}})
    )

    with pytest.raises(LLMError):
        await llm.detect_booking_intent("hello", [])
