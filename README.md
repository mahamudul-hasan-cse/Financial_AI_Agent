# Financial AI Agent

A full-stack financial analysis assistant powered by LLMs. It combines real-time stock market data with web search intelligence to deliver comprehensive financial insights through a custom-built chat interface.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│  React Frontend  :5173 (dev) / :8000 (prod)                                │
│  ┌─ Chat UI (App.jsx) ─────────────────────────────────────────────────┐   │
│  │  User message → POST /api/chat/stream (SSE)                         │   │
│  │  SSE event 1: [NLP_META] { entities, intent, sentiment }            │   │
│  │  SSE event 2…N: streamed LLM response tokens                        │   │
│  │  NLPPanel: shows intent / entities / sentiment per message          │   │
│  │  StockChart: auto-converts markdown tables to interactive charts    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ SSE / REST
┌──────────────────────────────▼─────────────────────────────────────────────┐
│  FastAPI Backend  (api.py)                                                  │
│                                                                             │
│  ① NLP Preprocessing Pipeline  (nlp_pipeline.py)                           │
│     ├─ Named Entity Recognition  — spaCy en_core_web_sm                    │
│     ├─ Intent Classification     — keyword pattern scoring (7 intents)      │
│     └─ Sentiment Analysis        — VADER compound polarity score            │
│                                                                             │
│  ② agno Agent  (financial_agent.py)                                         │
│     ├─ YFinanceTools    — stock price, fundamentals, news, recommendations  │
│     ├─ DuckDuckGoTools  — web search + financial news                       │
│     ├─ ChartTools       — matplotlib stock & comparison charts              │
│     ├─ ExcelTools       — multi-sheet styled Excel reports                  │
│     └─ RAGTools         — dense/sparse retrieval over financial knowledge   │
│                                                                             │
│  ③ Session management • Rate limiting • Analytics (/api/stats)              │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────────────┐
│  External Services                                                          │
│  Groq API (Llama 4 Scout)  •  Yahoo Finance  •  DuckDuckGo Search          │
└────────────────────────────────────────────────────────────────────────────┘
```

**Single Agent Design** — one unified agent handles all tools directly, avoiding multi-agent delegation overhead that Groq models handle poorly.

In production, FastAPI serves both the API and the React build on a single port (8000).

---

## NLP Techniques

This project implements a multi-stage NLP pipeline that runs on every user query:

| Technique | Implementation | Purpose |
|-----------|---------------|---------|
| **Named Entity Recognition (NER)** | spaCy `en_core_web_sm` | Extract company names, products, and map to ticker symbols |
| **Intent Classification** | Keyword pattern scoring (7 intents) | Route queries: stock_price / chart / comparison / news / fundamentals / recommendation / excel |
| **Sentiment Analysis** | VADER (Valence Aware Dictionary and sEntiment Reasoner) | Polarity scoring without model download; tuned for financial short text |
| **Retrieval-Augmented Generation (RAG)** | sentence-transformers `all-MiniLM-L6-v2` + cosine similarity (TF-IDF fallback) | Ground LLM responses in a curated financial knowledge base |
| **Tool-Augmented LLM** | agno agent framework + function calling | Live stock data retrieval instead of hallucinated answers |
| **Streaming Inference** | Server-Sent Events (SSE) | Real-time token-by-token response delivery |
| **Prompt Engineering** | Structured system instructions | Consistent output format and tool usage patterns |

The NLP pipeline results (intent, entities, sentiment) are visible in the **NLP Analysis panel** displayed above each agent response in the chat UI.

---

## Features

- **NLP Analysis panel** — shows detected entities, query intent, and sentiment for every message
- **Real-time stock prices** — live quotes via YFinance
- **Analyst recommendations** — buy/hold/sell consensus data
- **Company fundamentals** — key financial metrics and ratios
- **Web search** — latest financial news via DuckDuckGo
- **RAG knowledge retrieval** — grounding via dense/sparse document retrieval
- **Interactive stock charts** — automatic chart rendering when price data is returned
- **Markdown rendering** — tables, bold, links rendered in chat
- **Streaming responses** — SSE-based real-time token streaming
- **Conversation memory** — multi-turn conversations with context
- **Chat export** — download conversation as markdown file
- **Contextual suggestions** — smart follow-up suggestions based on conversation
- **Dark-themed chat UI** — responsive design with loading indicators
- **Analytics endpoint** — `/api/stats` for usage statistics and intent distribution
- **Rate limiting** — per-session rate limiting to prevent abuse
- **Session management** — automatic cleanup of idle sessions

---

## Tech Stack

| Layer     | Technology                                                    |
|-----------|---------------------------------------------------------------|
| LLM       | Groq — `meta-llama/llama-4-scout-17b-16e-instruct`           |
| Agent     | agno 1.8.4                                                    |
| NER       | spaCy `en_core_web_sm`                                        |
| Sentiment | VADER (`vaderSentiment`)                                      |
| RAG       | `sentence-transformers` (all-MiniLM-L6-v2) + scikit-learn    |
| Backend   | FastAPI + Uvicorn                                             |
| Frontend  | React 19 + Vite 7 + Recharts                                 |
| Data      | YFinance, DuckDuckGo Search                                   |
| Rendering | react-markdown + remark-gfm                                   |
| Testing   | pytest + httpx                                                |
| CI/CD     | GitHub Actions                                                |
| Deploy    | Docker + Docker Compose                                       |

---

## Project Structure

```
Ai_Agent/
├── financial_agent.py       # Agent definition (single source of truth)
├── api.py                   # FastAPI backend (NLP pipeline + sessions + rate limiting)
├── nlp_pipeline.py          # NLP: spaCy NER + VADER sentiment + intent classifier
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Pytest configuration
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # Single-command deployment
├── .env.example             # Environment variable template
├── .gitignore
├── LICENSE
├── README.md
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI pipeline
├── tools/
│   ├── chart_tools.py       # matplotlib chart generation (5 tools)
│   ├── excel_tools.py       # openpyxl Excel report generation (3 tools)
│   └── rag_tools.py         # RAG: dense/sparse retrieval over financial knowledge
├── tests/
│   ├── conftest.py          # Shared test fixtures
│   ├── test_api.py          # Backend API tests
│   ├── test_agent.py        # Agent configuration tests
│   ├── test_nlp_pipeline.py # NLP pipeline unit tests (intent/sentiment/NER)
│   ├── test_chart_tools.py  # Chart tool tests (yfinance mocked)
│   └── test_excel_tools.py  # Excel tool tests (yfinance mocked)
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js       # Dev proxy: /api → localhost:8000
    └── src/
        ├── main.jsx         # Entry point with ErrorBoundary
        ├── App.jsx          # Chat UI component
        ├── App.css          # Dark theme styles (CSS variables)
        ├── ErrorBoundary.jsx # React error boundary
        ├── NLPPanel.jsx     # NLP analysis panel (intent + entities + sentiment)
        └── StockChart.jsx   # Interactive price chart component
```

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- A [Groq API key](https://console.groq.com)

### 1. Clone the repository

```bash
git clone https://github.com/mahamudul-hasan-cse/Financial_AI_Agent.git
cd Financial_AI_Agent
```

### 2. Create Python virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3a. Download the spaCy NER model (required for entity extraction)

```bash
python -m spacy download en_core_web_sm
```

> The sentence-transformers RAG model (`all-MiniLM-L6-v2`, ~90 MB) is downloaded
> automatically on first run. The agent works without it but falls back to TF-IDF retrieval.

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## Running the App

### Development (two terminals)

```bash
# Terminal 1 — Backend
python api.py
# API running at http://localhost:8000

# Terminal 2 — Frontend
cd frontend && npm run dev
# UI running at http://localhost:5173
```

### Production (single port)

```bash
cd frontend && npm run build && cd ..
python api.py
# Everything at http://localhost:8000
```

### Docker

```bash
docker-compose up --build
# App at http://localhost:8000
```

### CLI mode (no web UI)

```bash
# Interactive mode
python financial_agent.py

# Single query
python financial_agent.py "What is the current price of AAPL?"

# Streaming output
python financial_agent.py --stream "Compare NVDA vs AMD"
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_api.py
```

---

## API Endpoints

| Method | Endpoint              | Description                                    |
|--------|-----------------------|------------------------------------------------|
| GET    | `/api/health`         | Health check (Groq key + NLP pipeline status)  |
| GET    | `/api/stats`          | Analytics: message counts, intent distribution |
| POST   | `/api/chat`           | Send message (non-streaming, includes NLP meta)|
| POST   | `/api/chat/stream`    | Send message (SSE streaming + NLP meta event)  |
| DELETE | `/api/chat/{session}` | Clear session history                          |

### Example request

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the current price of AAPL?"}'
```

### Example response (includes NLP metadata)

```json
{
  "response": "The current price of AAPL is **$265.72** ...",
  "session_id": "a1b2c3d4-...",
  "nlp_metadata": {
    "entities": [{"text": "AAPL", "label": "TICKER", "ticker": "AAPL"}],
    "intent": "stock_price",
    "sentiment": {"label": "neutral", "score": 0.831, "compound": 0.0},
    "spacy_available": true,
    "vader_available": true
  }
}
```

### Streaming SSE events

```
data: [NLP_META]{"entities":[...],"intent":"stock_price","sentiment":{...}}

data: The current price of AAPL is

data:  **$265.72**...

data: [DONE]
```

---

## Example Use Cases

- Stock market research and live price checks
- Analyst recommendation summaries
- Company financial fundamentals analysis
- Real-time financial news monitoring
- AI-powered investment research assistant

---

## Academic Relevance (CSE495B — NLP)

This project demonstrates a comprehensive range of NLP and AI techniques:

### Core NLP Concepts
- **Named Entity Recognition (NER)** — spaCy pipeline extracts ORG, PRODUCT, and TICKER entities from free-form text, with a ticker symbol resolution lookup table
- **Text Classification / Intent Detection** — scored keyword-pattern approach classifies queries into 7 domain-specific financial intents without requiring a labelled dataset
- **Sentiment Analysis** — VADER lexicon-based polarity scoring; appropriate for short financial queries and news headlines
- **Retrieval-Augmented Generation (RAG)** — dense retrieval (sentence-transformers cosine similarity) grounds LLM responses in verified financial knowledge, reducing hallucination

### Advanced NLP / ML
- **Transformer Embeddings** — `all-MiniLM-L6-v2` encodes document chunks and queries into 384-dimensional dense vectors for semantic similarity matching
- **Sparse Retrieval (TF-IDF)** — fallback retriever shows understanding of both dense and sparse retrieval trade-offs
- **Tool-Augmented LLMs / Function Calling** — agent dynamically selects and calls external tools (function calling), grounding answers in real-time data
- **Prompt Engineering** — structured system instructions guide consistent formatting, tool usage patterns, and output style

### System Design
- **Full-Stack AI Application** — React frontend + FastAPI backend + LLM agent
- **Streaming Inference** — Server-Sent Events deliver tokens in real-time
- **Explainability** — NLP pipeline outputs (intent, entities, sentiment) are visible in the UI, making the AI decision process transparent
- **Production Readiness** — Docker, GitHub Actions CI/CD, pytest test suite, rate limiting, session management
- **Analytics** — `/api/stats` endpoint exposes intent distribution and usage metrics

---

## Author

**Md. Mahamudul Hasan**
CSE Student | AI & NLP Enthusiast

GitHub: [mahamudul-hasan-cse](https://github.com/mahamudul-hasan-cse)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- [agno](https://github.com/agno-agi/agno) — AI Agent Framework
- [Groq](https://groq.com) — LLM Inference Platform
- [Yahoo Finance](https://finance.yahoo.com) — Stock Market Data
- [DuckDuckGo](https://duckduckgo.com) — Web Search API
