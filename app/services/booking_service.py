from datetime import date, datetime, time

from email_validator import EmailNotValidError, validate_email

from app.db.repositories import BookingRepository
from app.schemas import InterviewBookingExtraction, InterviewBookingResult

REQUIRED_FIELDS = ("name", "email", "interview_date", "interview_time")


class BookingService:
    def __init__(self, repository: BookingRepository):
        self._repository = repository

    async def handle(
        self, session_id: str, extraction: InterviewBookingExtraction
    ) -> InterviewBookingResult:
        normalized = self.normalize(extraction)
        missing = [field for field in REQUIRED_FIELDS if getattr(normalized, field) is None]

        if missing:
            normalized.missing_fields = missing
            normalized.status = "incomplete"
            return normalized

        record = await self._repository.create(
            session_id=session_id,
            name=normalized.name or "",
            email=normalized.email or "",
            interview_date=normalized.interview_date,  # type: ignore[arg-type]
            interview_time=normalized.interview_time,  # type: ignore[arg-type]
            position=normalized.position,
            notes=normalized.notes,
        )

        return InterviewBookingResult(
            id=record.id,
            status="confirmed",
            name=record.name,
            email=record.email,
            interview_date=record.interview_date,
            interview_time=record.interview_time,
            position=record.position,
            notes=record.notes,
            missing_fields=[],
        )

    @staticmethod
    def normalize(
        extraction: InterviewBookingExtraction,
    ) -> InterviewBookingResult:
        return InterviewBookingResult(
            status="incomplete",
            name=BookingService._clean(extraction.name),
            email=BookingService._valid_email(extraction.email),
            interview_date=BookingService._parse_date(extraction.interview_date),
            interview_time=BookingService._parse_time(extraction.interview_time),
            position=BookingService._clean(extraction.position),
            notes=BookingService._clean(extraction.notes),
            missing_fields=[],
        )

    @staticmethod
    def _clean(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @staticmethod
    def _valid_email(value: str | None) -> str | None:
        cleaned = BookingService._clean(value)
        if cleaned is None:
            return None
        try:
            return validate_email(cleaned, check_deliverability=False).normalized
        except EmailNotValidError:
            return None

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        cleaned = BookingService._clean(value)
        if cleaned is None:
            return None
        try:
            return date.fromisoformat(cleaned)
        except ValueError:
            return None

    @staticmethod
    def _parse_time(value: str | None) -> time | None:
        cleaned = BookingService._clean(value)
        if cleaned is None:
            return None
        try:
            return time.fromisoformat(cleaned)
        except ValueError:
            pass
        for pattern in ("%H:%M", "%I:%M %p", "%I:%M%p"):
            try:
                return datetime.strptime(cleaned, pattern).time()
            except ValueError:
                continue
        return None
