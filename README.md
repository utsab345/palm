# Interview RAG Backend

FastAPI backend for document ingestion and conversational RAG with interview
booking. Generation and embeddings use **Google Gemini only**. Chat memory lives
in **Redis**, vectors in **Qdrant**, and metadata/bookings in a **SQL database**
(SQLite by default; any SQLAlchemy async driver you install can be used).

No FAISS, no Chroma, no RetrievalQAChain, and no UI.

## Task coverage

| Requirement | Implementation |
| --- | --- |
| Upload `.pdf` / `.txt` files | `POST /documents` multipart upload; `pypdf` for PDFs, UTF-8/latin-1 for text |
| Extract text | `DocumentIngestionService.extract_text` |
| Two selectable chunking strategies | `fixed` and `recursive` via the `strategy` form field |
| Generate embeddings and store in a vector DB | Gemini `gemini-embedding-001` into Qdrant |
| Save metadata in a SQL/NoSQL DB | SQLAlchemy async models: `documents`, `document_chunks`, `interview_bookings` |
| Custom RAG (no RetrievalQAChain) | Explicit pipeline: rewrite → intent → embed query → Qdrant search → Gemini answer |
| Redis chat memory | `ChatMemoryService` stores per-session turns with TTL and trimming |
| Multi-turn queries | Redis history feeds query rewriting, intent detection, extraction, and answering |
| Interview booking via LLM | Gemini structured extraction for name, email, date, time |
| Store booking info | `interview_bookings` table + `GET /bookings` endpoints |
| Typed, modular code | Pydantic schemas, typed services/repositories, async I/O |

## Architecture

```
app/
  api/routes.py                     HTTP endpoints
  config.py                         Environment settings
  db/
    models.py                       Document, DocumentChunk, InterviewBooking
    repositories.py                 Persistence layer with typed errors
    session.py                      Async engine/session + schema init
  integrations/
    client.py                       Gemini client factory
    llm.py                          Chat, rewriting, intent, extraction
  services/
    chunking.py                     fixed + recursive chunkers
    document_ingestion_service.py   Extract → chunk → embed → Qdrant → SQL
    embedding_service.py            Gemini document/query embeddings
    vector_store.py                 Qdrant collection/upsert/search
    rag_service.py                  Query embedding + retrieval
    memory_service.py               Redis chat memory
    booking_service.py              Normalize/validate/persist bookings
    chat_service.py                 End-to-end multi-turn orchestration
  main.py                           FastAPI app + lifespan DB init
tests/                              Mocked Gemini, Qdrant, Redis, DB tests
```

## Configuration

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | required | Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Generation model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `GEMINI_EMBEDDING_DIMENSIONS` | `768` | Vector size for documents and queries |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `documents` | Qdrant collection |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/app.db` | Metadata/bookings DB |
| `REDIS_URL` | `redis://localhost:6379/0` | Chat memory |
| `CHAT_MEMORY_TTL_SECONDS` | `86400` | Session TTL |
| `CHAT_MEMORY_MAX_MESSAGES` | `40` | Max turns retained per session |
| `DEFAULT_CHUNK_SIZE` | `1000` | Default characters per chunk |
| `DEFAULT_CHUNK_OVERLAP` | `200` | Default overlap |
| `MAX_UPLOAD_BYTES` | `20000000` | Upload size limit |

Models, dimensions, chunking defaults, and infrastructure URLs are all
environment-configurable. The Qdrant collection is created with
`GEMINI_EMBEDDING_DIMENSIONS`; a mismatch raises a 503 instead of corrupting
retrieval. Change the collection name or re-ingest after changing dimensions.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Or run Qdrant, Redis, and the API together:

```bash
export GEMINI_API_KEY=your_gemini_api_key
docker compose up --build
```

## API

### Upload a document

`POST /documents` accepts multipart form data:

- `file`: `.pdf` or `.txt`
- `strategy`: `fixed` or `recursive` (default `recursive`)
- `chunk_size`: optional integer, default from settings
- `chunk_overlap`: optional integer, default from settings

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@interview-policy.pdf" \
  -F "strategy=recursive" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200"
```

Response:

```json
{
  "document_id": "0f4c...",
  "filename": "interview-policy.pdf",
  "content_type": "application/pdf",
  "size_bytes": 48211,
  "chunking_strategy": "recursive",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "chunk_count": 14,
  "embedding_model": "gemini-embedding-001",
  "embedding_dimensions": 768
}
```

Each chunk is embedded once and written to Qdrant with `document_id`,
`filename`, `chunk_index`, `chunking_strategy`, `chunk_size`, and
`chunk_overlap` metadata. Document/chunk metadata is stored in SQL.

### Chunking strategies

- `fixed`: character windows with overlap, preferring whitespace/newline
  boundaries near the end of each window.
- `recursive`: splits on paragraph and sentence boundaries, packs units up to
  the chunk size, and falls back to word splitting for oversized sentences.

### Chat (multi-turn RAG + interview booking)

`POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"candidate-42","message":"I would like to book an interview"}'
```

Follow-up turns reuse the same `session_id`; history is loaded from Redis:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"candidate-42","message":"Ada Lovelace, ada@example.com, 2026-06-01 at 10:30"}'
```

Pipeline per request:

1. Load Redis history and append the user message.
2. Gemini rewrites the message into a standalone retrieval query.
3. Gemini detects interview-booking intent using the conversation.
4. The rewritten query is embedded and searched in Qdrant.
5. Gemini answers using the retrieved context and history.
6. If the message is a booking request, Gemini extracts `name`, `email`,
   `interview_date`, and `interview_time`; valid complete bookings are stored in
   SQL and returned with an id.
7. The assistant reply is appended to Redis.

Response:

```json
{
  "session_id": "candidate-42",
  "answer": "Your interview is confirmed for June 1 at 10:30. What role are you interviewing for?",
  "rewritten_query": "interview booking for Ada Lovelace on 2026-06-01 at 10:30",
  "is_booking_request": true,
  "booking": {
    "id": "7b3c...",
    "status": "confirmed",
    "name": "Ada Lovelace",
    "email": "ada@example.com",
    "interview_date": "2026-06-01",
    "interview_time": "10:30:00",
    "position": null,
    "notes": null,
    "missing_fields": []
  },
  "sources": [
    {"id": "vector-id", "text": "Interviews are held on weekdays.", "score": 0.82, "metadata": {"filename": "interview-policy.pdf"}}
  ]
}
```

If required fields are missing or invalid, `booking.status` is `"incomplete"`
and `missing_fields` lists what to ask for. Nothing incomplete is persisted.

### Other endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Models, DB backend, and service configuration flags |
| `GET` | `/documents?limit=&offset=` | Document metadata list |
| `GET` | `/documents/{document_id}` | Document metadata + stored chunks |
| `GET` | `/chat/{session_id}/history` | Redis chat memory |
| `DELETE` | `/chat/{session_id}/history` | Clear Redis chat memory |
| `GET` | `/bookings?session_id=` | Bookings for a session |
| `GET` | `/bookings/{booking_id}` | Single booking |
| `POST` | `/search` | Retrieval-only query |

## Error handling

- `413` oversized upload, `415` unsupported file type, `400` unreadable/empty file
- `422` invalid chunking configuration or request body
- `429` Gemini rate limits after retries
- `502` invalid Gemini output or failed embedding/generation
- `503` Qdrant, Gemini outage, or Redis unavailability
- `504` Gemini timeout

Gemini structured outputs are validated against Pydantic schemas
(`RewrittenQuery`, `BookingIntent`, `InterviewBookingExtraction`). Malformed or
schema-invalid JSON returns `llm_invalid_response` instead of bad data.

## Tests

```bash
pip install -r requirements-dev.txt
ruff check app tests
pytest
```

All external services are mocked: Gemini, Qdrant, Redis, and the SQL layer (the
ingestion tests use an in-memory async SQLite database).
