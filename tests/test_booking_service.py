import uuid
from datetime import UTC, date, datetime, time

import pytest

from app.db.models import InterviewBooking
from app.schemas import InterviewBookingExtraction
from app.services.booking_service import BookingService


class FakeBookingRepository:
    def __init__(self):
        self.created = []

    async def create(self, **kwargs):
        self.created.append(kwargs)
        return InterviewBooking(
            id=uuid.uuid4(),
            status="confirmed",
            created_at=datetime.now(UTC),
            **kwargs,
        )


def complete_extraction() -> InterviewBookingExtraction:
    return InterviewBookingExtraction(
        name=" Ada Lovelace ",
        email="ada@example.com",
        interview_date="2026-06-01",
        interview_time="10:30",
        position="Backend Engineer",
        notes="Remote",
        summary="Interview booking",
    )


@pytest.mark.asyncio
async def test_complete_booking_is_persisted():
    repository = FakeBookingRepository()
    service = BookingService(repository)

    result = await service.handle("session-1", complete_extraction())

    assert result.status == "confirmed"
    assert result.id is not None
    assert result.name == "Ada Lovelace"
    assert result.email == "ada@example.com"
    assert result.interview_date == date(2026, 6, 1)
    assert result.interview_time == time(10, 30)
    assert len(repository.created) == 1
    assert repository.created[0]["session_id"] == "session-1"


@pytest.mark.asyncio
async def test_missing_required_fields_are_reported():
    repository = FakeBookingRepository()
    service = BookingService(repository)

    result = await service.handle(
        "session-1",
        InterviewBookingExtraction(name="Ada", interview_date="2026-06-01"),
    )

    assert result.status == "incomplete"
    assert result.id is None
    assert result.missing_fields == ["email", "interview_time"]
    assert repository.created == []


@pytest.mark.asyncio
async def test_invalid_email_date_and_time_become_missing():
    repository = FakeBookingRepository()
    service = BookingService(repository)
    extraction = InterviewBookingExtraction(
        name="Ada",
        email="not-an-email",
        interview_date="June 1",
        interview_time="25:99",
    )

    result = await service.handle("session-1", extraction)

    assert result.email is None
    assert result.interview_date is None
    assert result.interview_time is None
    assert set(result.missing_fields) == {
        "email",
        "interview_date",
        "interview_time",
    }


def test_normalize_trims_and_parses_values():
    result = BookingService.normalize(
        InterviewBookingExtraction(
            name="  Ada  ",
            email=" ADA@EXAMPLE.COM ",
            interview_date="2026-06-01",
            interview_time="10:30 AM",
            position="  Engineer ",
        )
    )

    assert result.name == "Ada"
    assert result.email == "ADA@example.com"
    assert result.interview_date == date(2026, 6, 1)
    assert result.interview_time == time(10, 30)
    assert result.position == "Engineer"
