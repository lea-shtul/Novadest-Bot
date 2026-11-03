# NovaDesk Lite: Server

A bot that answers Nova Tech employees' IT questions **only** from the company's internal
knowledge (guides, glossary, resolved tickets), using RAG. If nothing relevant is found,
it says so honestly, without calling the LLM at all.

```
Ingest (once):    documents → chunking → embedding → Chroma
Every question:   question → embedding → retrieval → threshold → LLM → answer with sources
```

Full design: [SPEC.md](SPEC.md).

## Requirements

- Windows + PowerShell, Python 3.12+ (tested on 3.13)
- A Gemini API key (free tier works)

## Setup (once)

```powershell
cd novadesk
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # your prompt now starts with (.venv)
pip install -r requirements.txt
Copy-Item .env.example .env           # then put your key in GOOGLE_API_KEY
```

If `Activate.ps1` is blocked ("running scripts is disabled"), run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Every new terminal:** `cd novadesk` and `.\.venv\Scripts\Activate.ps1` before running anything.

## Build the knowledge base

```powershell
python -m app.ingest
```

Deletes and rebuilds `storage/chroma` from `data/`. Expected summary: 69 guide, 22 glossary,
27 ticket chunks. Run it again whenever `data/` changes.

The free tier allows ~100 embeddings per minute, so ingest will pause and retry
(`provider error ... Waiting Ns`). That's normal. Wait a minute between runs.

## Run the server

```powershell
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs (Swagger) to try it.

| Endpoint | Description |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `POST /ask` | `{"question": "..."}` → `answered`, `answer`, `sources`, `top_score` |

`POST /ask` responses:
- **200**, `answered: true`: an answer in Hebrew, with sources.
- **200**, `answered: false`, `answer: null`: nothing relevant enough was found. **No LLM call was made.**
- **422**: empty question.
- **503**: Gemini is over quota or unreachable ("השירות לא זמין כרגע, נסו שוב בעוד דקה").

The console logs each decision, e.g. `top_score=0.685, 0/4 chunks >= 0.75 -> no LLM call`.

## Chat UI

A minimal chat page (Streamlit) in `chat_ui.py`. It calls the RAG pipeline directly, so the
FastAPI server does **not** need to be running. Build the knowledge base first (above).

```powershell
cd novadesk
.\.venv\Scripts\Activate.ps1
streamlit run chat_ui.py
```

Then open http://localhost:8501 (Streamlit usually opens it for you). Stop it with Ctrl+C.

Run it from `novadesk/` (the project root): that's where `.streamlit/config.toml` is read from,
and Streamlit puts the script's folder on the import path, which makes `from app...` work.

## Debug retrieval

```powershell
python -m app.search "יוצאים דפים לבנים מהמדפסת בקומה 3"
```

Prints the 4 closest chunks and their similarity scores. Uses the embedding API only, never the LLM.

## Settings (`.env`)

| Key | Meaning |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key (secret, never commit `.env`) |
| `LLM_MODEL`, `EMBED_MODEL` | Gemini model names |
| `TOP_K` | How many chunks to retrieve per question |
| `SIMILARITY_THRESHOLD` | Minimum score to count a chunk as relevant. **0.75 is provisional**: set it with the threshold experiment (SPEC section 8) |
| `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS` | Safety net for long sections |
| `EMBED_BATCH_SIZE`, `EMBED_MAX_RETRIES` | Ingest batching and retries after quota errors |
| `CHROMA_PATH`, `CHROMA_COLLECTION` | Where the vector store lives |
| `CORS_ORIGINS` | Comma-separated frontend origins allowed to call the API |

Changing a setting requires restarting the server. Changing `EMBED_MODEL` or the chunking
settings also requires re-running `python -m app.ingest`.

## Project structure

```
chat_ui.py             # streamlit run chat_ui.py  (the chat UI)
.streamlit/config.toml # Streamlit settings (no usage stats, no email prompt)
app/
├── config.py          # .env → one Settings object
├── main.py            # FastAPI app, CORS, /health, router wiring
├── ingest.py          # python -m app.ingest
├── search.py          # python -m app.search "question"  (retrieval only)
├── routers/ask.py     # POST /ask: thin, calls the pipeline
└── rag/               # all RAG logic lives here
    ├── models.py      # the only file that knows it's Gemini (+ embedding prefixes)
    ├── chunking.py    # documents → chunks
    ├── store.py       # the only file that talks to Chroma
    ├── prompts.py     # all text the LLM sees
    └── pipeline.py    # retrieve → threshold → generate → sources
```
