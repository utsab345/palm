import pytest

from app.core.errors import InvalidChunkingConfig
from app.schemas import ChunkingStrategy
from app.services.chunking import ChunkingService


def test_fixed_chunking_respects_size_and_overlap():
    text = "word " * 200
    chunks = ChunkingService().chunk(text, ChunkingStrategy.FIXED, chunk_size=80, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 80 for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_fixed_chunking_preserves_overlap_content():
    text = "abcdefghij " * 20
    chunks = ChunkingService().chunk(text, ChunkingStrategy.FIXED, chunk_size=50, chunk_overlap=15)

    assert chunks[0].text[-10:] in chunks[1].text


def test_recursive_chunking_prefers_paragraph_boundaries():
    text = (
        "First paragraph about the interview process. It has two sentences.\n\n"
        "Second paragraph about preparation tips. Bring your identification.\n\n"
        "Third paragraph about the schedule and location."
    )
    chunks = ChunkingService().chunk(
        text, ChunkingStrategy.RECURSIVE, chunk_size=90, chunk_overlap=0
    )

    assert len(chunks) == 3
    assert chunks[0].text.startswith("First paragraph")
    assert chunks[1].text.startswith("Second paragraph")
    assert chunks[2].text.startswith("Third paragraph")


def test_recursive_chunking_splits_long_sentences():
    text = " ".join(f"word{index}" for index in range(100))
    chunks = ChunkingService().chunk(
        text, ChunkingStrategy.RECURSIVE, chunk_size=60, chunk_overlap=0
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 60 for chunk in chunks)


def test_empty_text_returns_no_chunks():
    assert ChunkingService().chunk("   ", ChunkingStrategy.FIXED, 100, 10) == []


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(5, 1), (10_000, 100), (100, 100), (100, -1)],
)
def test_invalid_chunking_configuration(size, overlap):
    with pytest.raises(InvalidChunkingConfig):
        ChunkingService().chunk("some text", ChunkingStrategy.FIXED, size, overlap)
