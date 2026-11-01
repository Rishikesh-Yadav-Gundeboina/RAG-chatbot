# PDF Chatbot

Upload PDFs and ask questions about them in a chat. Answers come from Google Gemini and are grounded **only** in the text of your documents, with the source page shown for every answer. If the documents don't contain the answer, the bot says so instead of making one up.

Under the hood it is a **retrieval-augmented generation (RAG)** pipeline with **hybrid search**: every question is matched against the documents both by meaning (vector embeddings) and by exact keywords (BM25), and the two result lists are merged before the best passages are sent to the LLM.

```
 Streamlit UI  ──HTTP──▶  FastAPI backend
 (upload, chat,            ├─ upload ─▶ validate ─▶ save ─▶ background ingestion
  pick documents)          │                                  PDF → pages → clean → chunk
                           │                                  ├─▶ embeddings ─▶ ChromaDB   (dense index)
                           │                                  └─▶ raw text   ─▶ SQLite FTS5 (BM25 index)
                           └─ chat ─▶ rewrite follow-up ─▶ dense search ┐
                                                          BM25 search  ┴▶ RRF fusion ─▶ relevance gate
                                                          ─▶ prompt + history ─▶ Gemini ─▶ answer + sources
```

## Tech stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| API | FastAPI + Uvicorn |
| PDF parsing | PyMuPDF |
| Embeddings | sentence-transformers, `all-MiniLM-L6-v2` (runs locally) |
| Vector store | ChromaDB (persistent, L2 distance) |
| Keyword search | SQLite FTS5 with BM25 ranking (`porter` stemming) |
| LLM | Google Gemini (`gemini-3.6-flash`) via `google-genai` |
| App data | SQLite (documents, chat sessions, messages) |
| Tests | pytest |

## Getting started

**1. Install** (Python 3.10+):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure** — create a `.env` file in the project root:

```
API_KEY=your-gemini-api-key
```

**3. Run** — two terminals, **both from the project root** (the code uses paths relative to it, such as `data/` and `backend.…` imports):

```bash
# Terminal 1 – backend: http://localhost:8000  (interactive API docs at /docs)
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Terminal 2 – frontend: http://localhost:8501
source .venv/bin/activate
streamlit run frontend/app.py
```

The first backend start downloads the embedding model, so it takes a little longer.

**4. Test:**

```bash
python -m pytest tests/ -q --ignore=tests/test_ingestion.py
```

`tests/test_ingestion.py` is a manual script, not a test: `python tests/test_ingestion.py some.pdf` ingests a PDF from the command line.

### Configuration

| Setting | Where | Default | Meaning |
|---|---|---|---|
| `API_KEY` | `.env` | — | Gemini API key (required) |
| `RETRIEVAL_MODE` | env var (backend) | `hybrid` | `hybrid` = dense + BM25; `dense` = embeddings only |
| `API_BASE_URL` | env var (frontend) | `http://localhost:8000` | Where the UI finds the backend |
| `MAX_DISTANCE` | `backend/rag/relevance.py` | `1.0` | Max embedding distance for a chunk to count as relevant |
| `MIN_KEYWORD_COVERAGE` | `backend/rag/relevance.py` | `1.0` | Share of question keywords a far-away chunk must contain to still count |
| `chunk_size` / `overlap` | `backend/rag/chunker.py` | `1000` / `200` chars | Chunking |
| `HISTORY_LIMIT` | `backend/main.py` | `10` messages | Chat history used per question |
| `LLM_TIMEOUT_SECONDS` | `backend/rag/llm_service.py` | `30` | Gemini timeout |

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/documents` | Upload a PDF (multipart `file`). Returns `document_id` and `status: "processing"`, or `duplicate: true` if the same file was uploaded before |
| `GET` | `/documents` | List all documents with status (`processing` / `ready` / `failed`) and chunk count |
| `GET` | `/documents/{id}` | One document's status |
| `POST` | `/chat?question=…&session_id=…&document_ids=…` | Ask a question. Omit `session_id` to start a new session; repeat `document_ids` to search only those documents. Returns `session_id`, `answer`, `sources` |

## Project structure

```
backend/
  main.py                     FastAPI app: endpoints, service wiring, startup maintenance
  rag/
    ingestion_service.py      PDF → pages → clean → chunk → embed → both indexes; rolls back on failure
    chunker.py                Fixed-size character chunks with overlap
    embedding_service.py      Local sentence-transformer embeddings (normalised)
    vector_store.py           ChromaDB wrapper: upsert, delete, search, fetch embeddings
    keyword_store.py          SQLite FTS5 BM25 index: upsert, delete, search, backfill from Chroma
    fusion.py                 Reciprocal Rank Fusion of ranked result lists
    retrieval_service.py      Dense-only or hybrid retrieval; keyword coverage scoring
    extract_chunks_from_retrieve.py  Flattens search results into chunk dicts
    relevance.py              Relevance gate (distance threshold + keyword coverage)
    prompt_builder.py         Answer prompt and follow-up-question rewrite prompt
    rag_service.py            Orchestrates rewrite → retrieve → filter → prompt → LLM
    llm_service.py            Gemini client with timeout
    models.py                 DocumentChunk dataclass
  services/
    pdf_service.py            PDF validation and per-page text extraction (PyMuPDF)
    text_cleaner.py           Removes null bytes, collapses whitespace
    file_service.py           Saves uploads as data/uploads/<uuid>.pdf
    log_service.py            SQLite: documents, chat sessions, messages
frontend/
  app.py                      Streamlit UI: upload, document list/selection, chat with sources
tests/                        pytest suite (fakes for embeddings, stores and the LLM)
data/                         Runtime data (created automatically)
  uploads/                    Uploaded PDFs
  chroma/                     Vector index
  keyword_index.db            BM25 index
  logs/app.db                 Documents, sessions, messages
```

## How it was built, phase by phase

### Phase 1 — Upload and read PDFs
- `POST /documents` accepts a file and saves it under a fresh UUID in `data/uploads/` (`file_service.py`).
- `pdf_service.extract_pages` uses PyMuPDF to pull the text out page by page, keeping page numbers for citations later.
- `text_cleaner.clean_text` removes null bytes and collapses whitespace so PDF layout noise doesn't end up in the index.

### Phase 2 — Chunk and embed
- `chunker.chunk_text` splits each page into 1,000-character chunks with 200 characters of overlap, so a sentence cut at a boundary still appears whole in one chunk.
- Each chunk becomes a `DocumentChunk` with the id `<document_id>_<page>_<index>`.
- `EmbeddingService` turns chunks into normalised vectors with the local `all-MiniLM-L6-v2` model — no API cost for embeddings.

### Phase 3 — Vector store and retrieval
- `VectorStore` keeps chunks, embeddings and metadata (`document_id`, `page_number`) in a persistent ChromaDB collection under `data/chroma`. Writes are upserts, so re-ingesting is safe.
- `RetrievalService` embeds the question and asks Chroma for the 5 nearest chunks.
- `extract_chunks` flattens Chroma's nested result format into simple chunk dicts with text, ids, page and distance.

### Phase 4 — Answer generation (RAG)
- `prompt_builder.build_prompt` puts the retrieved chunks into a prompt that tells the model to answer **only** from that context and to say so when the answer isn't there.
- `LLMService` sends it to Gemini; `RagService.answer` ties retrieval and generation together and returns the answer with its source chunks.
- `POST /chat` exposes it over HTTP.

### Phase 5 — Ingestion pipeline and document tracking
- `IngestionService` runs the whole pipeline for one document. If any step fails, it deletes that document's partial chunks and marks it `failed`; on success it marks it `ready` with its chunk count.
- `log_service` records every document in SQLite (`data/logs/app.db`) with filename, path, status and chunk count; `GET /documents` and `GET /documents/{id}` expose it.

### Phase 6 — Conversation memory
- Chat sessions and every user/assistant message are stored in SQLite. `/chat` creates a session when none is given and loads the last 10 messages.
- **Question rewriting:** a follow-up like "how do I install *it*?" is useless for search. When there is history, `RagService` first asks Gemini to rewrite the question into a standalone one (`build_rewrite_prompt`) and searches with that. If the rewrite fails, the original question is used.
- The history is also included in the answer prompt, labelled as context only — the documents remain the source of truth.

### Phase 7 — Hardening
- **Upload validation:** only `.pdf` files with the right content type, non-empty, and actually parseable by PyMuPDF are accepted.
- **Duplicate detection:** uploads are hashed (SHA-256); uploading the same file again returns the existing document instead of indexing it twice. Older records get their hash backfilled at startup.
- **Background ingestion:** uploads return immediately with `status: "processing"` while ingestion runs as a FastAPI background task. Documents left in `processing` by a crash or restart are marked `failed` at startup.
- **Relevance gate:** chunks whose embedding distance is above `1.0` are dropped. If nothing is left, the bot answers "I could not find the answer to that in the uploaded documents." **without calling the LLM**. The threshold was measured on the ingested documents: on-topic questions' best chunk landed at 0.38–0.80 and off-topic ones at 1.17–1.57.
- **LLM timeout:** Gemini calls time out after 30 seconds, and `/chat` returns HTTP 504 with a friendly message instead of hanging.
- **Document filtering:** `/chat` accepts `document_ids` to search only selected documents.

### Phase 8 — Streamlit frontend
- Sidebar: upload a PDF, see every document with a status icon (⏳ processing, ✅ ready, ❌ failed) and chunk count, refresh the list, and tick the documents to search (none ticked = search all).
- Main area: a chat that keeps the session across questions and shows the source snippets and page numbers under each answer.

### Phase 9 — Hybrid search (BM25 + dense)
Embeddings are good at meaning and paraphrases but weak at exact terms — function names like `read_csv`, acronyms, error codes, names, numbers. BM25 keyword search is strong exactly there, so the two are combined:

1. **Keyword index.** `KeywordStore` keeps every chunk in a SQLite FTS5 table (`data/keyword_index.db`) ranked with BM25 and Porter stemming. It is persistent, updated incrementally, supports the `document_ids` filter, and needs no extra dependency. Questions are tokenised, stopwords removed, and each term quoted so user input can never break the FTS query syntax.
2. **Dual-write ingestion.** `IngestionService` writes each document to both Chroma and the keyword index, and rolls back both if anything fails. At startup, `backfill_keyword_index` adds any documents that exist in Chroma but not yet in the keyword index, so existing data was indexed automatically.
3. **Fusion.** `RetrievalService` takes the top 20 from each search and merges them with **Reciprocal Rank Fusion** (`fusion.py`): each chunk scores `1 / (60 + rank)` in every list it appears in, so chunks both searches agree on rise to the top. The best 5 go on to the LLM.
4. **One distance scale.** Chunks found only by BM25 have no embedding distance from the search, so it is computed from their stored embedding. Every chunk is then judged by the same calibrated threshold.
5. **Relevance gate, extended.** A chunk is kept if its distance is ≤ `1.0` **or** it contains *all* of the question's keywords (`keyword_coverage`). This lets exact-term matches through even when the embedding scores them as distant, while off-topic questions still get "not found". A looser 50% keyword threshold was tried and rejected: it let "What is the capital of France?" match a chunk that merely mentioned Gensim.
6. **Switchable.** `RETRIEVAL_MODE=dense` restores the original embeddings-only behaviour for comparison. The API and frontend did not change.

Measured on the three indexed books (3,481 chunks):

| Question | Dense only | Hybrid |
|---|---|---|
| "how do I read a csv file" | General CSV chunks, none mentioning `read_csv` | 4 of the top 5 are `pandas.read_csv` passages |
| "groupby" | 5 relevant | 5 relevant, 3 of them containing the term itself |
| "What is the capital of France?" / "who won the world cup" | Not found | Still not found |

## Tests

The suite uses small fake objects in place of the embedding model, vector store, keyword store and LLM, so most tests run without a model download or an API key:

- **Ingestion:** success, rollback on every failure point, both indexes written and cleaned up (`test_ingestion_rollback.py`)
- **Retrieval:** keyword store behaviour (ranking, filtering, upsert, delete, query escaping, backfill), RRF fusion, hybrid retrieval shape and distances, dense mode unchanged (`test_keyword_store.py`, `test_fusion.py`, `test_hybrid_retrieval.py`)
- **Relevance:** distance threshold and keyword-coverage rules, no-answer short-circuit (`test_relevance_filter.py`, `test_rag_service_relevance.py`)
- **Conversation:** history storage, question rewriting and its fallback, prompt building (`test_log_service_history.py`, `test_rag_service_history.py`, `test_prompt_builder.py`)
- **LLM timeout:** timeout errors are raised with the right client config, propagated from answering, and fall back gracefully during rewriting (`test_llm_timeout.py`)

A few files in `tests/` are manual scripts rather than assertions: `test_retrieve.py` runs a real retrieval against the local index and prints each chunk's distance (handy for re-checking `MAX_DISTANCE` after the document set changes a lot), and `test_pdf_service.py` / `test_embedding.py` print extraction and embedding output for inspection.
