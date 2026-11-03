# NovaDesk Lite: Spec

> A small, focused version of NovaDesk: a bot that answers employees' questions **only** from the internal knowledge of a fictional company, using RAG.
> The bot runs as a **Streamlit chat page** (`chat_ui.py`) that calls the RAG pipeline directly. There is no HTTP server: the original FastAPI server was removed (section 9, stage 7).

---

## 0. Instructions for Claude Code

- **This is my first AI project, and the goal is for me to understand every line.** At the end of each stage (section 9), stop and explain in simple terms what was built, why, and how the pieces connect. No "magic" shortcuts.
- **Build in the order of section 9.** Don't start a stage before the previous one works and has been verified.
- **Simplicity first.** Don't add libraries, files, layers or features that aren't listed here without asking first. In particular: no database, no users, no tickets, no conversation history (section 10).
- **Every tunable value** comes from `.env`, never hardcoded.
- **If anything is unclear or contradictory**, ask before deciding on your own.
- **Windows + PowerShell:** remind me how to activate the venv when needed.
- **Language:** the data files and the bot's answers are in Hebrew. Code, comments and explanations to me are in English unless I ask otherwise.

---

## 1. What the system does

Nova Tech is a fictional software company of ~80 employees. Its IT team has internal guides, a glossary of internal terms, and old resolved tickets.

An employee asks a free-text question (e.g. "יוצאים דפים לבנים מהמדפסת בקומה 3"). The bot:
1. Searches the knowledge base for the chunks closest in meaning
2. If relevant-enough chunks are found: answers **only from them**, and cites sources
3. If not: honestly says it didn't find an answer, **without calling the LLM at all**

**The knowledge base is built once at ingest time, and is read-only after that.** Nothing is written to it at runtime.

```
Ingest (once):    documents → chunking → embedding → Chroma
Every question:   question → embedding → retrieval → threshold → LLM → answer with sources
```

---

## 2. Technology decisions

| Component | Choice |
|---|---|
| Language | Python 3.12+ |
| UI | Streamlit (`chat_ui.py`), calls the pipeline directly, no HTTP |
| RAG library | LlamaIndex |
| Vector store | ChromaDB, local (`PersistentClient`), **cosine** distance |
| LLM | Gemini free tier: `gemini-3.5-flash-lite` (via `llama-index-llms-google-genai`) |
| Embeddings | Gemini: `gemini-embedding-2` (via `llama-index-embeddings-google-genai`) |
| Config | `.env` (python-dotenv) |

**Packages:** `streamlit`, `chromadb`, `llama-index-core`, `llama-index-vector-stores-chroma`, `llama-index-llms-google-genai`, `llama-index-embeddings-google-genai`, `python-dotenv`, `python-frontmatter`.

Once stage 1 works, pin versions in `requirements.txt` (`pip freeze`) so the project installs identically on defense day.

---

## 3. Project structure

```
novadesk/
├── SPEC.md
├── .env.example
├── requirements.txt
├── data/                     # the internal knowledge (files I wrote)
│   ├── guides/               # 12 guides, Markdown with frontmatter
│   ├── glossary.md           # glossary of internal terms
│   ├── seed_users.json       # only used to get technicians' names for tickets
│   └── seed_tickets.json     # resolved tickets
├── storage/chroma/           # created automatically, not in git
├── chat_ui.py                # the chat UI: streamlit run chat_ui.py
├── .streamlit/config.toml    # Streamlit settings
├── app/
│   ├── config.py             # reads .env into a single Settings class
│   ├── ingest.py             # ingest script: python -m app.ingest
│   └── rag/                  # all RAG logic, and only here
│       ├── models.py
│       ├── chunking.py
│       ├── store.py
│       ├── prompts.py
│       └── pipeline.py
└── tools/                    # retrieval-only helpers (no LLM)
    ├── search.py             # python -m tools.search "question"
    └── threshold_experiment.py  # python -m tools.threshold_experiment (section 8)
```

`storage/`, `.env` and `.venv/` go in `.gitignore`.

---

## 4. Settings (`.env`)

```env
GOOGLE_API_KEY=...
LLM_MODEL=gemini-3.5-flash-lite
EMBED_MODEL=gemini-embedding-2

TOP_K=4
SIMILARITY_THRESHOLD=0.6     # temporary! to be set by experiment (section 8)

EMBED_BATCH_SIZE=10
EMBED_MAX_RETRIES=6

CHROMA_PATH=storage/chroma
CHROMA_COLLECTION=knowledge
```

Also create `.env.example` with the same keys and no secret.

---

## 5. RAG files (`app/rag/`)

**The rule:** all RAG logic lives only in `app/rag/`. `ingest.py`, `chat_ui.py` and `tools/` only call functions from there.
**`models.py` is the only file that knows it's Gemini. `store.py` is the only file that talks to Chroma.**

| File | Role | Functions |
|---|---|---|
| `models.py` | "The translator": builds models from `.env` | `get_llm()`, `get_embed_model()` |
| `chunking.py` | "The scissors": document → chunks | `guide_to_nodes(path)`, `glossary_to_nodes(path)`, `ticket_to_node(ticket, resolved_by_name)` |
| `store.py` | "The gatekeeper": all access to Chroma | `get_index()`, `reset_knowledge()`, `add_nodes(nodes)`, `retrieve(question)` |
| `prompts.py` | All text the LLM sees | `ANSWER_PROMPT`, `format_context(nodes)` |
| `pipeline.py` | The question pipeline | `answer_question(question)` |

### 5.1 `models.py`
- `get_llm()`: returns the LLM per `LLM_MODEL`.
- `get_embed_model()`: returns the embedding model per `EMBED_MODEL`.

**Embedding prefixes (required):** `gemini-embedding-2` **ignores** the `task_type` parameter (verified in a previous attempt: identical vectors with and without task_type). Google instead recommends a prefix inside the text. But the LlamaIndex integration only sends `task_type`. So `get_embed_model()` wraps `GoogleGenAIEmbedding` in a small subclass that prepends a prefix before sending:
- queries: `task: search result | query: {text}`
- documents: `title: none | text: {text}`

No other file knows about the prefixes.

### 5.2 `chunking.py`
**Guides and glossary:**
1. Read the frontmatter (`python-frontmatter`): `title`, `category`, `updated`. These go into **metadata**, not the text. (Titles are quoted, because a colon inside a title breaks YAML.)
2. Split on **`##`** headings. Each `##` is one chunk. The top `#` heading is not its own chunk.
3. **0 overlap between sections.** Each section is a complete unit of meaning.
4. **Context header** at the start of each chunk's text (in Hebrew, to match the content):
   - guide: `מדריך: {title} | סעיף: {## heading}`
   - glossary: `מילון מונחים | מונח: {## heading}`
5. **No size limit.** Every section is one chunk. (A safety net that split sections over 512 tokens was removed in stage 7: no section came close to needing it; the largest chunk is ~450 tokens.)

**Tickets:**
- Each ticket is exactly one chunk (problem and solution stay together).
- Only tickets with `added_to_knowledge: true`.
- `ticket_to_node` receives the technician's name from the caller (`chunking.py` doesn't read user files).
- Text format (Hebrew labels, to match the content):
```
טיקט {id}: {title}
בעיה: {problem}
פתרון: {solution}
```

**Stable IDs:** `guide-{filename without extension}-{section number}`, `glossary-{term number}`, `ticket-{id}`. For a part of a long section: `-part{n}`.

**Metadata per chunk:** `source_type` (`guide` / `glossary` / `ticket`), `source_id`, `title`, `section` (guide and glossary only), `category`, `date` (guide `updated`, or ticket `resolved_at`), `resolved_by_name` (tickets only).

### 5.3 `store.py`
- `get_index()`: the single connection to the Chroma collection, with **cosine**. Check the installed chromadb version's docs for how to set cosine (the default may be L2).
- `reset_knowledge()`: delete the collection and recreate it empty.
- `add_nodes(nodes)`: embed and insert **in batches** (e.g. 10). On a rate-limit error (429): wait and retry (backoff).
- `retrieve(question)`: fetch `TOP_K` chunks using a **Retriever** (not `query_engine`, which calls the LLM even when no chunk is relevant). Return a **similarity** score (higher = more similar). If a cosine distance comes back, convert: `similarity = 1 - distance`.

### 5.4 `prompts.py`
`ANSWER_PROMPT` (**kept in Hebrew**, because the bot answers employees in Hebrew):
```
אתה בוט התמיכה הטכנית של נובה טק. אתה עונה לעובדי החברה בעברית.

כללים:
1. ענה רק על סמך הקטעים שלמטה. אל תשתמש בידע כללי, ואל תמציא שמות של מערכות, שלוחות או נתיבים.
2. אם הקטעים לא עונים על השאלה במלואה, אמור מה כן ידוע, ושאר הפרטים כדאי לברר מול צוות IT בשלוחה 7700.
3. אם יש סתירה בין מקורות: מדריך רשמי גובר על טיקט, ובין שני טיקטים, החדש גובר.
4. בסוף התשובה ציין את המקורות שהשתמשת בהם, למשל: "לפי המדריך 'מדפסות', ולפי טיקט 1043 שנפתר ע״י דני".
5. כתוב בקצרה ובבהירות. כשיש שלבים, כתוב אותם כרשימה ממוספרת.

הקטעים:
{context}

השאלה: {question}
```

The prompt's rules, in English: answer only from the chunks below; never use general knowledge or invent system names, extensions or paths; if the chunks only partly answer, say what is known and refer the rest to IT at extension 7700; on conflicting sources, an official guide beats a ticket, and a newer ticket beats an older one; end with the sources used; be short and clear, with numbered steps where relevant.

`format_context(nodes)` builds `{context}`, each chunk formatted as:
```
[מקור: מדריך | 04-printers | עודכן 2026-10-14]
{text}
```
or:
```
[מקור: טיקט 1043 | נפתר ע״י דני | 2026-10-12]
{text}
```

### 5.5 `pipeline.py`
`answer_question(question)` calls small single-purpose functions, in order:
```
store.retrieve()     ← fetch TOP_K chunks with similarity scores
threshold            ← keep only chunks with score >= SIMILARITY_THRESHOLD
                       if none remain: stop here, no LLM call
LLM call             ← one call with ANSWER_PROMPT; a reply of NO_ANSWER means "not found"
sources              ← the sources list for the answer
```
Returns an `AnswerResult` with: `answered`, `answer`, `sources`, `top_score` (the highest score, even if it didn't pass the threshold).

**Provider errors** (429 quota, network): raise `ProviderUnavailableError`, which the chat shows as "השירות לא זמין כרגע, נסו שוב בעוד דקה". Any other exception is a bug: it is not hidden behind that message.

---

## 6. Ingest script (`python -m app.ingest`)

**A script, not an endpoint.** It deletes and rebuilds all knowledge, which shouldn't be reachable over the network.

1. `store.reset_knowledge()`
2. Chunk: all guides, the glossary, and tickets from `seed_tickets.json` with `added_to_knowledge: true`. Technicians' names come from `seed_users.json`, by `resolved_by`.
3. `store.add_nodes(...)`
4. Print a summary: chunk count per type (expected: ~70 guide, 22 glossary, 27 ticket).

---

## 7. Chat UI (`chat_ui.py`)

Run from the project root: `streamlit run chat_ui.py`, then open http://localhost:8501.

- Calls `answer_question(question)` directly. Each question is sent on its own; the history is display-only.
- Answer found: the answer, and the retrieved chunks (type, title, section or "נפתר ע״י", score) in an expander "קטעים שנבדקו".
- `answered: false`: "לא מצאתי מידע על זה במאגר הידע של נובה טק. כדאי לפנות לצוות IT בשלוחה 7700."
- `ProviderUnavailableError`: the "service unavailable" message. Any other exception: a generic error, logged in the terminal.
- Empty chat: 4 example questions as buttons. "שיחה חדשה" clears the history. The page is right-to-left.

---

## 8. Threshold experiment (after stage 5, before stage 6)

`SIMILARITY_THRESHOLD=0.6` is a guess. Scores depend on the embedding model, so the threshold is set by experiment:

A script `python -m tools.threshold_experiment` that reads `data/test_questions.json` (some answerable from the knowledge base, some not, including near-misses like "המדפסת **בבית** שלי לא מדפיסה" — "my printer **at home** won't print"), runs **retrieval only** (no LLM, to save quota), and prints a table: question, score, closest source. Pick a threshold that separates the two groups.

---

## 9. Build order

Each stage: build, run, verify, and explain to me before moving on. **In stages 2–5, start by listing the stage's functions** (name, input, output, one sentence on what it does), get my approval, and only then write code. At the end of the stage, walk me through the code function by function.

| Stage | What to build | How to verify |
|---|---|---|
| 1 | Skeleton: `config.py`, `main.py`, `/health`, `requirements.txt`, `.env.example`, `.gitignore` | `/health` in Swagger (`http://localhost:8000/docs`) returns ok |
| 2 | `rag/chunking.py` only. **No Gemini, no Chroma** | A small script prints the chunk count per type, plus 2–3 full example chunks (text + metadata) |
| 3 | `rag/models.py`, `rag/store.py`, `ingest.py` | `python -m app.ingest` prints the summary; `storage/chroma` exists |
| 4 | Retrieval only: a small script that takes a question and prints the 4 chunks and scores | "יוצאים דפים לבנים מהמדפסת בקומה 3" returns ticket 1043 or the printers guide at the top. **Also:** show me the same question with and without the prefixes, scores side by side, so I understand what the prefixes do |
| 5 | `rag/prompts.py`, `rag/pipeline.py`, `routers/ask.py` | An answerable question returns an answer with sources. "איפה חונים ליד המשרד?" ("where do I park near the office?") returns `answered: false` — **and show me no LLM call was made** |
| 5b | Threshold experiment (section 8): `tools/threshold_experiment.py` + a questions JSON file. Moved here from after stage 6, because stage 4 showed the guessed 0.6 lets unrelated questions through (provisional value until then: 0.75) | The table separates answerable from unanswerable questions; set `SIMILARITY_THRESHOLD` in `.env` accordingly |
| 6 | Error handling (503), cleanup, a short README with run instructions | |
| 7 | Streamlit chat UI (`chat_ui.py`). Then simplification: FastAPI removed (`main.py`, `routers/`, CORS), safety net removed, scripts moved to `tools/` | Same chunks, same retrieval scores, same pipeline results as before |

---

## 10. Out of scope (possible later additions)

Not built now, but the structure allows adding them without breaking anything:
1. Opening a ticket when there's no answer (SQLite)
2. A technician closing a ticket and adding it to the knowledge base (`upsert_ticket` in `store.py`)
3. Conversation, with rewriting of follow-up questions
4. Users and permissions
5. An HTTP API or a separate frontend. The UI is Streamlit (`chat_ui.py`, not React), which calls `answer_question` directly, without HTTP
