# NovaDesk Lite

A bot that answers Nova Tech employees' IT questions **only** from the company's internal
knowledge (guides, glossary, resolved tickets), using RAG. If nothing relevant is found,
it says so honestly, without calling the LLM at all. You talk to it through a Streamlit chat page.

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

## Run the chat

```powershell
cd novadesk
.\.venv\Scripts\Activate.ps1
streamlit run chat_ui.py
```

Then open http://localhost:8501 (Streamlit usually opens it for you). Stop it with Ctrl+C.

Run it from `novadesk/` (the project root): that's where `.streamlit/config.toml` is read from,
and Streamlit puts the script's folder on the import path, which makes `from app...` work.

What you'll see for a question:
- **An answer** in Hebrew, with the retrieved chunks under "קטעים שנבדקו".
- **"לא מצאתי מידע..."**: nothing passed the threshold (**no LLM call was made**), or the LLM replied `NO_ANSWER`.
- **"השירות לא זמין כרגע..."**: Gemini is over quota or unreachable.

The terminal logs each decision, e.g. `top_score=0.685, 0/4 chunks >= 0.75 -> no LLM call`.

## Tools (retrieval only, no LLM)

Run from `novadesk/`:

```powershell
python -m tools.search "יוצאים דפים לבנים מהמדפסת בקומה 3"   # the 4 closest chunks and their scores
python -m tools.threshold_experiment                         # SPEC section 8: pick SIMILARITY_THRESHOLD
```

Both use the embedding API only, never the LLM.

## Settings (`.env`)

| Key | Meaning |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key (secret, never commit `.env`) |
| `LLM_MODEL`, `EMBED_MODEL` | Gemini model names |
| `TOP_K` | How many chunks to retrieve per question |
| `SIMILARITY_THRESHOLD` | Minimum score to count a chunk as relevant. Set it with `python -m tools.threshold_experiment` |
| `EMBED_BATCH_SIZE`, `EMBED_MAX_RETRIES` | Ingest batching and retries after quota errors |
| `CHROMA_PATH`, `CHROMA_COLLECTION` | Where the vector store lives |

Changing a setting requires restarting the chat. Changing `EMBED_MODEL` also requires
re-running `python -m app.ingest`.

## Project structure

```
chat_ui.py             # streamlit run chat_ui.py  (the chat UI)
.streamlit/config.toml # Streamlit settings (no usage stats, no email prompt)
app/
├── config.py          # .env → one Settings object
├── ingest.py          # python -m app.ingest
└── rag/               # all RAG logic lives here
    ├── models.py      # the only file that knows it's Gemini (+ embedding prefixes)
    ├── chunking.py    # documents → chunks
    ├── store.py       # the only file that talks to Chroma
    ├── prompts.py     # all text the LLM sees
    └── pipeline.py    # retrieve → threshold → generate → sources
tools/
├── search.py               # python -m tools.search "question"
└── threshold_experiment.py # python -m tools.threshold_experiment
```
