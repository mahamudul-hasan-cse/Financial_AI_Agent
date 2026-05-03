# Financial AI Agent — CSE495B NLP Project

A full-stack financial research assistant that combines a production-grade NLP pipeline with a Groq-hosted LLM and an interactive React chat interface. Built for CSE495B (Natural Language Processing).

---

## Architecture

| Layer | Technology |
|---|---|
| **LLM Backend** | Groq API — `meta-llama/llama-4-scout-17b-16e-instruct` |
| **Local LLM** | Ollama — `qwen2.5:3b` (offline fallback) |
| **Agent Framework** | agno 1.8.4 |
| **API** | FastAPI + SSE streaming |
| **NLP Pipeline** | spaCy `en_core_web_md` + VADER + TF-IDF Intent Classifier |
| **RAG Embeddings** | `sentence-transformers/all-mpnet-base-v2` (768-dim) |
| **Frontend** | React 19 + Vite 7 + Recharts |

---

## NLP Components

| # | Component | Model / Library | Purpose |
|---|---|---|---|
| 1 | **Named Entity Recognition** | spaCy `en_core_web_md` | Extracts ORG, PERSON, GPE, PRODUCT entities |
| 2 | **Financial NER** | Custom regex patterns | Extracts TICKER, FIN_METRIC, DATE_REF, MONEY |
| 3 | **Sentiment Analysis** | VADER (`nltk`) | Compound score + positive/neutral/negative label |
| 4 | **Intent Classification (ML)** | TF-IDF + Logistic Regression | 8-class intent; 93.75% accuracy on gold-standard set |
| 5 | **Intent Classification (KW)** | Keyword matching | Fallback when ML confidence < threshold |
| 6 | **RAG Retrieval** | `all-mpnet-base-v2` + BM25 re-ranking | Query expansion + source citations |

---

## Agent Tools (14 total)

**Stock Data (YFinance)**
- `get_current_stock_price` — real-time price and change
- `get_stock_fundamentals` — P/E, EPS, market cap, revenue
- `get_analyst_recommendations` — buy/hold/sell breakdown
- `get_company_news` — recent headlines

**Web Search (DuckDuckGo)**
- `duckduckgo_search` — general web search
- `duckduckgo_news` — news search

**Chart Generation (matplotlib)**
- `create_stock_chart` — line chart with fill and price annotation
- `create_comparison_bar_chart` — side-by-side stock price comparison
- `create_comparison_overlay_chart` — normalized overlay chart
- `generate_line_chart` — generic line chart from data
- `generate_bar_chart` — generic bar chart from data
- `generate_pie_chart` — generic pie chart from data

**Excel Reports (openpyxl + pandas)**
- `create_stock_excel_report` — multi-sheet report (Summary, Fundamentals, Price History)
- `create_comparison_excel` — side-by-side comparison workbook
- `create_excel_sheet` — generic data-to-Excel export

---

## Features

- **Streaming responses** via SSE with real-time token delivery
- **Intent-aware formatting** — each intent (stock_price, chart, comparison, news, fundamentals, recommendation, excel, general) has its own response format rules
- **NLP metadata panel** — shows detected intent + confidence %, entities, sentiment in a collapsible UI panel
- **Chart display** — generated PNG charts appear inline in the chat
- **Excel download** — generated reports are directly downloadable
- **Session management** — persistent chat history with rename and delete
- **Dark / light mode** with localStorage persistence
- **Voice output** via Web Speech API
- **Starter prompts** and follow-up suggestions
- **RAG** with query expansion and source citations
- **Quantitative evaluation** — 80-example gold-standard test set, per-class P/R/F1

---

## Academic Results

| Metric | Value |
|---|---|
| Intent classifier accuracy | **93.75%** (80-example gold-standard set) |
| Cross-validation accuracy | 75.5% (TF-IDF n-gram 1-2) |
| Test suite | **310 tests passing** |
| Test coverage | **66%** |
| LLM latency (Groq) | ~500 ms |
| LLM latency (Ollama qwen2.5:3b) | ~2 500 ms |
| RAG embedding model | all-mpnet-base-v2 (768-dim) |
| spaCy model | en_core_web_md |

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- A [Groq API key](https://console.groq.com/) (free tier available)
- *(Optional)* [Ollama](https://ollama.ai/) for local inference

### Backend

```bash
cd Ai_Agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### Frontend

```bash
cd Ai_Agent/frontend
npm install
```

### Environment Variables

Create `Ai_Agent/.env`:

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key |
| `OLLAMA_BASE_URL` | No | Ollama URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | No | Ollama model (default: `qwen2.5:3b`) |
| `LLM_PROVIDER` | No | `groq` or `ollama` (default: `groq`) |

### Running

```bash
# Terminal 1 — Backend
cd Ai_Agent
python api.py
# → http://localhost:8001

# Terminal 2 — Frontend (dev)
cd Ai_Agent/frontend
npm run dev
# → http://localhost:5173

# Production (single port)
cd Ai_Agent/frontend && npm run build && cd ..
python api.py
# → http://localhost:8001
```

### Running with Ollama (local, no API key needed)

```bash
# Install and pull the model
ollama pull qwen2.5:3b

# Set provider in .env
echo "LLM_PROVIDER=ollama" >> .env

# Start backend as normal
python api.py
```

---

## Running Tests

```bash
cd Ai_Agent
pytest --tb=short -q
# 310 passed

# With coverage
pytest --cov=. --cov-report=term-missing
```

### Evaluation scripts

```bash
# Quantitative NLP evaluation (80-example gold-standard set)
python evaluation.py

# Ablation study (impact of each NLP component)
python ablation_study.py
```

---

## Project Structure

```
Ai_Agent/
├── api.py                        # FastAPI backend (SSE streaming, sessions, NLP)
├── financial_agent.py            # Agent shim → backend/app/agent/
├── nlp_pipeline.py               # spaCy NER + VADER + intent pipeline
├── intent_classifier.py          # TF-IDF + LogisticRegression (8-class)
├── evaluation.py                 # Quantitative NLP evaluation
├── ablation_study.py             # NLP component ablation study
├── research_service.py           # Structured response builder
├── market_service.py             # Market context helpers
├── persistence.py                # SQLite session/alert persistence
├── cache.py                      # In-memory cache layer
├── config.py                     # App settings (pydantic)
├── requirements.txt
├── pyproject.toml                # pytest config
├── Dockerfile
├── docker-compose.yml
│
├── backend/app/agent/
│   ├── financial_agent.py        # agno Agent construction
│   ├── tool_registry.py          # Toolkit factory with graceful degradation
│   └── prompts.py                # System prompt + intent format rules
│
├── tools/
│   ├── chart_tools.py            # matplotlib PNG generation (self-contained)
│   ├── excel_tools.py            # openpyxl Excel reports (self-contained)
│   └── rag_tools.py              # RAG: query expansion + re-ranking + citations
│
├── schemas/
│   ├── chat.py                   # Request/response models
│   ├── nlp.py                    # NLPMetadata TypedDict
│   └── streaming.py              # SSE event models
│
├── tests/
│   ├── test_nlp_pipeline.py      # 45 NLP tests
│   ├── test_intent_classifier.py # 16 classifier tests
│   ├── test_evaluation.py        # 13 evaluation tests
│   ├── test_api.py               # API endpoint tests
│   ├── test_agent.py             # Agent config tests
│   ├── test_chart_tools.py       # Chart tool tests
│   └── test_excel_tools.py       # Excel tool tests
│
├── data/
│   └── company_tickers.json      # Ticker → company name mapping
│
├── outputs/
│   ├── charts/                   # Generated PNG charts (git-ignored)
│   └── sheets/                   # Generated Excel files (git-ignored)
│
└── frontend/
    ├── src/
    │   ├── App.jsx               # Main chat UI
    │   ├── App.css               # Design tokens + dark/light theme
    │   ├── ResearchCard.jsx      # Message renderer + inline chart display
    │   ├── NLPPanel.jsx          # Intent/entity/sentiment display panel
    │   ├── StockChart.jsx        # Auto-recharts from markdown tables
    │   ├── SessionsSidebar.jsx   # Chat history sidebar
    │   └── ErrorBoundary.jsx
    ├── package.json
    └── vite.config.js            # Dev proxy: /api + /outputs → :8001
```

---

## License

Academic project — CSE495B, Spring 2026.
