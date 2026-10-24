# Financial AI Agent

**Ask a question about a stock, get a sourced, structured answer.** A local NLP pipeline
(NER, intent classification, sentiment) analyses each question first, then an LLM agent
with live market-data, web-search, charting, Excel and RAG tools answers it and streams
the result into a React chat UI.

Built as a CSE495B (Natural Language Processing) course project, so it ships with
evaluation tooling: a gold-standard test set, an ablation study, calibration and
bootstrap confidence intervals.

> Educational project, **not financial advice**.

---

## Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Running tests](#running-tests)
- [Evaluation](#evaluation)
- [API reference](#api-reference)
- [Deployment](#deployment)
- [Project structure](#project-structure)
- [Design decisions and limitations](#design-decisions-and-limitations)
- [License](#license)

---

## Features

- **Streaming answers** over Server-Sent Events, with staged progress messages, a stop button,
  and visible "Retrying in Ns…" notices when the LLM is rate-limited.
- **Copy and Regenerate** on every answer (Regenerate re-asks the previous question).
- **NLP pre-processing on every question**, shown in a collapsible panel:
  - spaCy NER plus a company→ticker map and regex entities (TICKER, FIN_METRIC, DATE_REF, MONEY)
  - an 8-class intent classifier (TF-IDF + logistic regression, with a keyword fallback)
  - VADER sentiment
- **Intent-aware prompting.** Each intent (price, chart, comparison, news, fundamentals,
  recommendation, Excel, general) gets its own response-format rules. For price, news and
  recommendation questions, VADER sentiment over recent headlines is added to the prompt.
- **Low-latency agent loop:** at most 3 tool calls per answer, compact cached tool output
  (60 s TTL per ticker + function), a short system prompt and trimmed history.
- **Agent tools (agno):** Yahoo Finance prices, fundamentals, analyst ratings and news;
  DuckDuckGo web and news search; matplotlib charts shown inline; downloadable Excel reports;
  RAG over a small financial-concepts knowledge base with citations.
- **Groq or local Ollama** as the LLM, chosen with one environment variable.
- **Saved conversations** in SQLite (rename, delete, reload). Each conversation is tied to an
  anonymous per-browser id, so users only see their own history.
- **Dark/light theme and voice input** (Web Speech API).
- **API-only extras:** watchlists, price/news alerts, company snapshots, multi-symbol
  comparison and PDF export (see [API reference](#api-reference)).

## Screenshots

> _Placeholder: no screenshots are committed yet._ Suggested captures, to be saved under
> `docs/screenshots/` and linked here:
>
> 1. Empty state with starter prompts (dark theme)
> 2. A streamed answer with the NLP panel expanded
> 3. A generated chart rendered inline
> 4. Light theme and the conversation-history sidebar
> 5. `/docs` (FastAPI OpenAPI UI)

---

## Architecture

```
┌──────────────────────────┐   SSE / JSON (X-Client-Id header)
│ React 19 + Vite (SPA)    │ ────────────────────────────────────┐
│ src/App.jsx, src/api.js  │                                     │
└──────────────────────────┘                                     ▼
                                     ┌───────────────────────────────────────────────┐
                                     │ FastAPI  — api.py → backend/app/main.py        │
                                     │  middleware: CORS · gzip · request log · key  │
                                     │  routers:  health · chat · sessions · export  │
                                     │            watchlist · alerts · market · files│
                                     └───────┬───────────────┬───────────────┬───────┘
                                             │               │               │
                 ┌───────────────────────────▼──┐  ┌─────────▼─────────┐  ┌──▼──────────────┐
                 │ services/chat.py             │  │ services/sessions │  │ persistence.py  │
                 │  nlp_pipeline.py             │  │  ownership checks │  │  SQLite         │
                 │   ├ spaCy NER (SPACY_MODEL)  │  │  rate limits      │  │  (DATABASE_PATH)│
                 │   ├ intent_classifier.py     │  └───────────────────┘  └─────────────────┘
                 │   └ VADER sentiment          │
                 │  prompt = intent rules       │
                 │         + sentiment          │
                 │         + session history    │
                 └──────────────┬───────────────┘
                                │ fresh agno Agent per request
                 ┌──────────────▼───────────────────────────────────────┐
                 │ backend/app/agent  (Groq Llama-4-Scout  |  Ollama)   │
                 │  tools: YFinance · DuckDuckGo · charts · Excel · RAG │
                 │         (cache.py TTL cache in front of yfinance)    │
                 └──────────────────────────────────────────────────────┘
```

**Request flow (`POST /api/chat/stream`).**

1. The server resolves the session and checks it belongs to the caller's `X-Client-Id`.
2. It applies the per-session and per-IP rate limits.
3. The NLP pipeline runs, and its result is sent as the first SSE event.
4. The server builds the prompt: intent instructions, optional headline sentiment, and the last
   10 messages of *this* session only.
5. A fresh agno `Agent` streams the answer. It shares the model client and toolkits with other
   requests but keeps its own run state, so nothing crosses sessions.
6. The cleaned answer is saved and sent as `[FINAL]`, followed by a structured research card
   and `[DONE]`.

**SSE protocol.** Each event is one `data:` line:

| Payload | Meaning |
|---|---|
| `[NLP_META]{json}` | NLP analysis (always first) |
| `[STATUS]{json}` | Transient notice before any text, e.g. `{"type":"status","message":"Rate limit reached — retrying in 4s…"}` or `"Groq busy — switched to backup model"` |
| `"text chunk"` | JSON-encoded piece of the answer |
| `[ERROR] message` | Generation failed; the message is user-facing (the stream still ends with `[DONE]`) |
| `[FINAL]"text"` | The saved answer. Clients replace the streamed text with it |
| `[RESEARCH_DATA]{json}` | Structured research card |
| `[MODEL_INFO]{json}` | Who answered, e.g. `{"provider":"openrouter","model":"openai/gpt-oss-120b","fallback":true}` |
| `[DONE]` | End of stream |

**Backup provider.** When Groq answers 429 (rate limit) or 5xx, or can't be reached, the same
request goes to OpenRouter right away, without waiting for Groq's `Retry-After`. The backup is an
agno `OpenRouter` agent with the same tools and instructions. It tries `OPENROUTER_MODEL` first,
then each model in `OPENROUTER_FALLBACK_MODELS`. Streams send
`[STATUS] Groq busy — switched to backup model`, and the UI labels the answer "via backup".
`/api/chat` returns `llm_provider`, `llm_model` and `llm_fallback`. `/api/health` shows the
fallback config (`fallback`) and who answered last (`last_llm`). The backups are tried once per
question. If they all fail, the normal Groq retries below take over. Other errors, such as a
400, don't trigger the fallback. It is on whenever `OPENROUTER_API_KEY` is set;
`FALLBACK_PROVIDER=none` turns it off.

If generation fails before any text is sent, the server retries up to `LLM_MAX_RETRIES` times
(default 2), sending a `[STATUS]` event before each wait. On Groq 429s it waits for the
`Retry-After` the API suggests (capped by `LLM_RETRY_MAX_DELAY`); the Groq SDK's own silent
retries are disabled. When retries run out, a clear `[ERROR]` follows. It never retries after
partial output, because that would duplicate text. The UI never re-asks via `/api/chat`, so
each question costs at most `1 + LLM_MAX_RETRIES` Groq runs, plus one run per backup model.

| Layer | Technology |
|---|---|
| LLM | Groq `openai/gpt-oss-120b` (default; `openai/gpt-oss-20b` is fastest) or Ollama `qwen2.5:3b`; OpenRouter free models as the automatic backup |
| Agent framework | agno 1.8.4 |
| API | FastAPI + uvicorn, SSE streaming, SQLite |
| NLP | spaCy `en_core_web_sm`, VADER, scikit-learn TF-IDF + LogReg |
| RAG | sentence-transformers `all-MiniLM-L6-v2` (TF-IDF fallback) |
| Frontend | React 19, Vite 7, react-markdown and Recharts (lazy-loaded chunks), Vitest |

---

## Quick start

### Prerequisites

- Python **3.12+**
- Node.js **20+** (CI and Docker use 22)
- A [Groq API key](https://console.groq.com/) (free tier available), **or** a local
  [Ollama](https://ollama.com/) install

### 1. Backend

```bash
git clone <your-fork-url> Financial_AI_Agent
cd Financial_AI_Agent

python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate

# Optional but recommended: CPU-only torch (much smaller than the default CUDA build)
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-dev.txt    # runtime + test deps (requirements.txt = runtime only)
python -m spacy download en_core_web_sm

cp .env.example .env                   # then set GROQ_API_KEY (or LLM_PROVIDER=ollama)
python api.py                          # → http://localhost:8001  (PORT to change)
```

The RAG embedding model (about 90 MB) downloads the first time a question uses RAG. Use
`uvicorn api:app --reload --port 8001` while developing the backend.

### 2. Frontend (dev server)

```bash
cd frontend
npm ci
npm run dev                            # → http://localhost:5173 (proxies /api and /outputs to :8001)
```

If the backend runs on another port, set `API_PROXY_TARGET=http://localhost:<port>` before
`npm run dev`.

### 3. Single-port production build

```bash
cd frontend && npm run build && cd ..
python api.py                          # serves the SPA from frontend/dist at http://localhost:8001
```

### Docker

```bash
cp .env.example .env                   # set GROQ_API_KEY
docker compose up --build              # → http://localhost:8001
```

The image builds the frontend and installs CPU-only torch. It also bakes in the spaCy and
sentence-transformers models the app loads, so it runs offline after the build. To bake
different models, use `--build-arg SPACY_MODEL=… --build-arg EMBEDDING_MODEL=…`. Charts and
reports are kept in `./outputs`, and the SQLite database in the `financial-ai-agent-data`
volume.

### Using Ollama instead of Groq

```bash
ollama pull qwen2.5:3b
# in .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_HOST=http://localhost:11434
```

With Docker: `docker compose --profile ollama up --build`, then
`docker exec financial-ai-ollama ollama pull qwen2.5:3b`. Compose already points
`OLLAMA_HOST` at the `ollama` service. If Ollama is unreachable, the agent falls back to Groq
when `GROQ_API_KEY` is set. Otherwise chat returns 503.

---

## Configuration

All settings are environment variables, read in `config.py`. `.env` is loaded with
`override=False`, so variables set in your shell, Docker or PaaS always win.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` or `ollama` |
| `GROQ_API_KEY` | (unset) | **Secret.** Needed for Groq. Without it the server still starts, `/api/health` reports `degraded`, and chat returns 503 |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model id. `openai/gpt-oss-20b` is the fastest; `qwen/qwen3.8-27b` is also supported |
| `LLM_MAX_RETRIES` | `2` | Retries after the first attempt when the LLM fails before any text (e.g. 429). Each retry is announced to the UI |
| `LLM_RETRY_MAX_DELAY` | `20` | Longest wait (seconds) before a retry, even if Groq's `Retry-After` is longer |
| `OPENROUTER_API_KEY` | (unset) | **Secret.** Turns on the automatic backup used when Groq is rate-limited or down |
| `FALLBACK_PROVIDER` | `openrouter` if `OPENROUTER_API_KEY` is set, else `none` | Backup LLM provider. `none` disables it |
| `OPENROUTER_MODEL` | `openai/gpt-oss-120b` | First backup model (tool calling tested) |
| `OPENROUTER_FALLBACK_MODELS` | `google/gemini-2.5-flash` | Comma-separated models tried in order if the previous one errors or returns 429. Empty means none |
| `AGENT_TOOL_CALL_LIMIT` | `3` | Max tool calls per answer; every tool round is another full model call |
| `TOOL_OUTPUT_MAX_CHARS` | `2000` | Tool results are truncated to this many characters before the model sees them |
| `CACHE_TTL_SECONDS` | `60` | Thread-safe TTL cache for yfinance data and tool results, per ticker + function. `0` disables it |
| `AGNO_TELEMETRY` | `false` | agno usage telemetry. Defaulted to `false` in code, `.env.example` and the Dockerfile |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Ollama model tag |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy model for NER and lemmatisation (`python -m spacy download <name>`) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model for RAG (`all-mpnet-base-v2` is a little better and uses about 5× the RAM) |
| `DATABASE_PATH` | `data/financial_agent.db` | SQLite file (`SQLITE_PATH` is still accepted) |
| `OUTPUTS_DIR` | `outputs/` | Where charts and Excel files are written |
| `HOST` / `PORT` | `0.0.0.0` / `8001` | Bind address for `python api.py` and the Docker image |
| `FORWARDED_ALLOW_IPS` | `127.0.0.1` | Proxies trusted for `X-Forwarded-For`. Set `*` behind a PaaS load balancer so rate limits see real client IPs |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins (e.g. `https://app.example.com,https://www.example.com`) |
| `RATE_LIMIT_PER_MINUTE` | `20` | Chat messages per minute per browser (`X-Client-Id`, or client IP without one). Over the limit: `429` with JSON `{"detail": ...}` and `Retry-After`. `0` disables |
| `RATE_LIMIT_MAX` / `IP_RATE_LIMIT_MAX` / `RATE_LIMIT_WINDOW` | `10` / `30` / `60` | Chat requests per window (seconds), per session and per IP |
| `SESSION_TTL` / `MAX_SESSIONS` | `1800` / `100` | Inactivity expiry (seconds) and session cap |
| `API_KEY` | (unset) | **Secret.** If set, `/api/*` (except `/api/health`) requires the `X-API-Key` header. The bundled UI does not send it, so use it for API-only deployments |
| `LOG_LEVEL` | `INFO` | Python logging level |

Frontend (build-time, `frontend/.env`): `VITE_API_URL` (default `/api`) and
`VITE_OUTPUTS_URL` (default `/outputs`). Set both when the API runs on a different origin.
Both are public: anything `VITE_*` ends up in the browser bundle, so never put secrets there.

**OpenRouter (paid backup).** Defaults are paid models with tested tool calling: `openai/gpt-oss-120b` (~$0.15/$0.60 per 1M input/output tokens) then `google/gemini-2.5-flash` (~$0.30/$2.50). A backup answer costs roughly $0.001–0.003. Paid models have no daily request cap (rate limits scale with credits). Every tool round of an answer is a separate request. Avoid `:free` models: they are capped (50–1000/day) and often rate-limited upstream.

**Secrets.** Only `GROQ_API_KEY`, `OPENROUTER_API_KEY` and `API_KEY` are secret. Keep them in `.env` (git-ignored
and excluded from the Docker build context) or in your platform's secret store, and pass
them at runtime (`docker run --env-file .env`, compose `env_file`, PaaS dashboard). They are
never baked into the image or exposed by `/api/health` (which only reports whether a key is
configured).

---

## Running tests

```bash
pytest -q                               # backend suite, no API key or network needed
pytest --cov=. --cov-report=term-missing

cd frontend
npm test                                # Vitest + Testing Library
npm run lint && npm run build
```

The backend tests use a temporary SQLite database. `tests/conftest.py` sets `DATABASE_PATH`
before importing the app, and `persistence.reset_for_tests()` refuses to touch the default
DB. The tests also remove `GROQ_API_KEY` and the OpenRouter settings, mock the LLM (including
the fallback), and stub live market data.

Coverage includes:

- API endpoints and SSE framing, including retry without duplicated text
- session ownership across clients
- session-id validation and PDF-export path safety
- CORS, JSON 404s and degraded health
- persistence
- agent construction and provider selection
- NLP pipeline, intent classifier, evaluation helpers, chart and Excel tools
- on the frontend: the SSE parser (events split across reads), client-id handling, the App
  streaming flow, and the UI components

---

## Evaluation

```bash
python evaluation.py        # intent (keyword vs ML), NER, RAG, calibration, bootstrap CIs
python ablation_study.py    # contribution of each NLP component
python error_analysis.py    # misclassified examples
python generate_visualizations.py   # figures → outputs/figures/
```

Results from `python evaluation.py` with the default models (`en_core_web_sm`,
`all-MiniLM-L6-v2`, scikit-learn 1.9):

| Metric | Value |
|---|---|
| Intent accuracy, ML (TF-IDF + LogReg), 80-example gold set | **95.0%** (macro-F1 95.1%) |
| Intent accuracy, keyword baseline, same set | 81.3% |
| NER precision / recall / F1 (gold set) | 71.7% / 91.5% / 80.4% |
| RAG MRR / Recall@5 (gold queries) | 100% / 100% |

Read these numbers with the caveats under
[Design decisions and limitations](#design-decisions-and-limitations). The test set is small
and written by the author, and the RAG knowledge base is tiny. Re-run the scripts after
changing models or dependencies, because the numbers will move.

---

## API reference

Interactive docs: **`/docs`** (Swagger UI) and `/openapi.json`.

Every endpoint except `/api/health`, `/api/stats`, `/api/snapshot` and `/api/compare` needs
an **`X-Client-Id`** header: 16–128 characters from `[A-Za-z0-9_-]`. The web UI creates one
per browser and keeps it in `localStorage`. Sessions, watchlists and alerts belong to the
client id that created them. Other clients get `404`. This keeps users' histories apart but is
**not** authentication: anyone who has the id can act as that browser.

Session ids must match `^[A-Za-z0-9_-]{1,64}$`.

| Method & path | Purpose |
|---|---|
| `GET /api/health` | Status (`ok` / `degraded`), LLM provider and config, backup (`fallback`) config, who answered last (`last_llm`), NLP availability |
| `GET /api/stats` | In-process usage counters (reset on restart) |
| `POST /api/chat` | Non-streaming chat → `{response, session_id, nlp_metadata, structured_response, llm_provider, llm_model, llm_fallback}` |
| `POST /api/chat/stream` | Streaming chat (SSE, see the protocol above) |
| `DELETE /api/chat/{session_id}` | Delete a session |
| `GET /api/sessions` · `GET/PATCH /api/sessions/{id}` | List, read and rename your sessions |
| `POST /api/chat/export` | PDF transcript (rendered in a temp dir and deleted after sending) |
| `POST/DELETE /api/watchlist` · `GET /api/watchlist/{id}` | Per-session watchlist with live prices |
| `POST /api/alerts` · `GET /api/alerts/{id}` · `GET /api/alerts/{id}/check` · `PATCH/DELETE /api/alerts/item/{alert_id}` | Price, digest and news alerts |
| `GET /api/snapshot/{symbol}` · `GET /api/compare?symbols=AAPL,MSFT` | Company snapshot and comparison table |
| `GET /outputs/{charts,sheets}/{file}` | Generated files |

Unknown `/api/*` paths return a JSON `404`. All other unknown `GET` paths serve the SPA.

---

## Deployment

The backend needs long-lived SSE connections, a writable disk (SQLite and `outputs/`) and
about 1 GB of RAM to be safe with the default models. Deploy it as **one container with one worker**:
the session cache, rate limits and stats live in process memory.

- **Render / Railway / Fly.io / Azure Container Apps / App Service (container):** build from
  the `Dockerfile`. Then:
  - set `GROQ_API_KEY`, set `FORWARDED_ALLOW_IPS=*` (the platform proxy sets `X-Forwarded-For`),
    and set `CORS_ORIGINS` if the frontend is hosted elsewhere
  - mount a persistent disk at `/app/data` (and optionally `/app/outputs`)
  - the image listens on `$PORT` (default 8001), which most platforms inject automatically
  - point the health check at `/api/health` (the image has a Docker `HEALTHCHECK` on it). It
    returns 200 even when `degraded`, so check the `status` field when you want readiness.
- **Vercel / Netlify:** suitable for the **frontend only**. Build `frontend/` with
  `VITE_API_URL=https://<api-host>/api` and `VITE_OUTPUTS_URL=https://<api-host>/outputs`,
  and add the frontend origin to the backend's `CORS_ORIGINS`. The API does not fit
  serverless functions (SSE duration limits, no persistent disk).
- **Memory:** with the defaults, the container uses about 530 MB idle and peaks at about
  0.7 GB RSS once the RAG model is loaded (measured with the Docker image, CPU torch). If RAM
  is tight, `EMBEDDING_MODEL` can point at any small sentence-transformers model. If it fails
  to load, RAG falls back to TF-IDF. The image is about 2 GB.

No deploy manifests (`render.yaml`, `fly.toml`, Bicep) are included yet.

---

## Project structure

```
Financial_AI_Agent/
├── api.py                     # Entry-point: `uvicorn api:app` / `python api.py`
├── config.py                  # All environment settings
├── nlp_pipeline.py            # spaCy NER + regex entities + VADER + intent
├── intent_classifier.py       # TF-IDF + LogisticRegression (8 intents)
├── persistence.py             # SQLite: sessions (with client_id), messages, watchlist, alerts
├── cache.py                   # thread-safe TTL + LRU cache (yfinance data, tool results)
├── market_service.py          # Snapshot / comparison / watchlist data
├── research_service.py        # Structured research card builder
├── financial_agent.py         # Back-compat shim → backend/app/agent
├── evaluation.py · ablation_study.py · error_analysis.py · generate_visualizations.py
├── backend/app/
│   ├── main.py                # FastAPI app factory
│   ├── middleware.py          # CORS, gzip, request logging, optional API key
│   ├── security.py            # X-Client-Id dependency, client IP
│   ├── state.py               # In-memory session cache, rate-limit windows, stats
│   ├── nlp_models.py          # Shared spaCy loader
│   ├── routers/               # health, chat, sessions, watchlist, alerts, market, export, files
│   ├── services/              # chat (prompting/NLP), sessions (ownership/rate limits), export (PDF)
│   └── agent/                 # agno Agent factory, tool registry, prompts
├── tools/                     # market_tools (cached Yahoo data), chart/excel/rag tools (agno toolkits)
├── schemas/                   # Pydantic request/response models
├── tests/                     # pytest suite (temp DB, no API key, offline)
├── data/company_tickers.json  # Company → ticker map (the SQLite DB is created here, git-ignored)
├── outputs/                   # Generated charts / sheets (git-ignored)
├── frontend/                  # React app (src/App.jsx, src/api.js, components, __tests__)
├── Dockerfile · docker-compose.yml · .github/workflows/ci.yml
└── requirements.txt (runtime, pinned) · requirements-dev.txt (tests)
```

---

## Design decisions and limitations

- **TF-IDF + logistic regression for intent,** not an LLM classifier. It is fast, free,
  deterministic and explainable (the confidence is shown in the UI), and it degrades to
  keyword rules. The trade-off is that the training data is small (about 200 examples) and
  synthetic.
- **The evaluation set is author-written,** in the same style as the training data. Treat the
  gold-set accuracy as an optimistic in-distribution estimate. `evaluation.py` reports
  bootstrap confidence intervals for this reason.
- **The RAG knowledge base is tiny** (16 hand-written concept chunks), so the perfect
  RAG scores mostly show that the pipeline works, not retrieval quality at scale.
- **Market data comes from unofficial Yahoo Finance endpoints** (via `yfinance`) and
  DuckDuckGo. It can be delayed, rate-limited or wrong.
- **Anonymous client ids, not accounts.** They separate users' histories but don't
  authenticate anyone.
- **Single process.** In-memory rate limits, the TTL data cache and stats reset on restart. Scaling out would need
  Redis (or similar) and a real database.
- **Prompt injection:** user text and web results reach an agent that can write chart and
  Excel files. Filenames are sanitised, but output size and count are not capped.

---

## License

[MIT](LICENSE) © 2026 Md. Mahamudul Hasan. Originally developed for CSE495B (NLP), Spring 2026.
