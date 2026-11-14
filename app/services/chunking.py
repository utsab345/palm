import re
from dataclasses import dataclass

from app.core.errors import InvalidChunkingConfig
from app.schemas import ChunkingStrategy

MIN_CHUNK_SIZE = 20
MAX_CHUNK_SIZE = 8000


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str


class ChunkingService:
    def chunk(
        self,
        text: str,
        strategy: ChunkingStrategy,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[TextChunk]:
        self._validate(chunk_size, chunk_overlap)
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        if strategy is ChunkingStrategy.FIXED:
            pieces = self._fixed(normalized, chunk_size, chunk_overlap)
        elif strategy is ChunkingStrategy.RECURSIVE:
            pieces = self._recursive(normalized, chunk_size, chunk_overlap)
        else:
            raise InvalidChunkingConfig(f"Unknown chunking strategy: {strategy}")

        return [TextChunk(index=index, text=piece) for index, piece in enumerate(pieces) if piece]

    @staticmethod
    def _validate(chunk_size: int, chunk_overlap: int) -> None:
        if not MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE:
            raise InvalidChunkingConfig(
                f"chunk_size must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}"
            )
        if not 0 <= chunk_overlap < chunk_size:
            raise InvalidChunkingConfig(
                "chunk_overlap must be at least 0 and smaller than chunk_size"
            )

    @staticmethod
    def _fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        compact = re.sub(r"[ \t]+", " ", text)
        pieces: list[str] = []
        length = len(compact)
        start = 0

        while start < length:
            end = min(start + chunk_size, length)
            if end < length:
                boundary = compact.rfind("\n", start, end)
                if boundary <= start:
                    boundary = compact.rfind(" ", start, end)
                if boundary > start + chunk_size // 2:
                    end = boundary

            piece = compact[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= length:
                break

            next_start = end - chunk_overlap
            start = next_start if next_start > start else start + 1

        return pieces

    def _recursive(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        units = self._units(text, chunk_size)
        pieces: list[str] = []
        current = ""

        for unit in units:
            candidate = f"{current} {unit}".strip() if current else unit
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                pieces.append(current)

            if chunk_overlap > 0 and current:
                room = max(0, chunk_size - len(unit) - 1)
                tail = self._tail(current, min(chunk_overlap, room))
                current = f"{tail} {unit}".strip() if tail else unit
            else:
                current = unit

        if current:
            pieces.append(current)
        return pieces

    def _units(self, text: str, chunk_size: int) -> list[str]:
        units: list[str] = []
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) <= chunk_size:
                units.append(paragraph)
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence) <= chunk_size:
                    units.append(sentence)
                else:
                    units.extend(self._split_words(sentence, chunk_size))
        return units

    @staticmethod
    def _split_words(text: str, chunk_size: int) -> list[str]:
        pieces: list[str] = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip() if current else word
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = word[:chunk_size]
        if current:
            pieces.append(current)
        return pieces

    @staticmethod
    def _tail(text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        tail = text[-limit:]
        boundary = tail.find(" ")
        if boundary != -1:
            return tail[boundary + 1 :]
        return tail
