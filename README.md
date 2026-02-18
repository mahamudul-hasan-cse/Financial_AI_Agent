# Financial AI Agent

A full-stack financial analysis assistant powered by LLMs. It combines real-time stock market data with web search intelligence to deliver comprehensive financial insights through a custom-built chat interface.

---

## Architecture

```
React (Vite)  ──POST /api/chat──▶  FastAPI (api.py)  ──▶  Groq LLM API
  :5173 (dev)                          :8000                (Llama 4 Scout)
                                         │
                                         ├── YFinance (stock data)
                                         └── DuckDuckGo (web search)
```

**Single Agent Design** — one unified agent handles all tools (stock prices, analyst recommendations, company fundamentals, news search) directly, avoiding multi-agent delegation overhead.

In production, FastAPI serves both the API and the React build on a single port (8000).

---

## Features

- **Real-time stock prices** — live quotes via YFinance
- **Analyst recommendations** — buy/hold/sell consensus data
- **Company fundamentals** — key financial metrics and ratios
- **Web search** — latest financial news via DuckDuckGo
- **Markdown rendering** — tables, bold, links rendered in chat
- **Dark-themed chat UI** — responsive design with loading indicators
- **Source citations** — agent always includes data sources

---

## Tech Stack

| Layer     | Technology                                     |
|-----------|------------------------------------------------|
| LLM       | Groq — `meta-llama/llama-4-scout-17b-16e-instruct` |
| Agent     | agno 1.8.4                                     |
| Backend   | FastAPI + Uvicorn                              |
| Frontend  | React 19 + Vite 7                              |
| Data      | YFinance, DuckDuckGo Search                   |
| Rendering | react-markdown + remark-gfm                   |

---

## Project Structure

```
Ai_Agent/
├── financial_agent.py    # Agent definition (single source of truth)
├── api.py                # FastAPI backend (imports agent, serves API + SPA)
├── requirements.txt      # Python dependencies
├── .env                  # API keys (not in git)
├── .gitignore
├── LICENSE
├── README.md
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js    # Dev proxy: /api → localhost:8000
    └── src/
        ├── main.jsx
        ├── App.jsx       # Chat UI component
        └── App.css       # Dark theme styles
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

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 5. Configure environment variables

Create a `.env` file in the project root:

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
cd frontend && npm run build
cd ..
python api.py
# Everything at http://localhost:8000
```

### CLI mode (no web UI)

```bash
python financial_agent.py
```

---

## API Endpoints

| Method | Endpoint       | Description              |
|--------|----------------|--------------------------|
| GET    | `/api/health`  | Health check             |
| POST   | `/api/chat`    | Send message to agent    |

### Example request

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the current price of AAPL?"}'
```

### Example response

```json
{
  "response": "The current price of AAPL is $265.72.\nSource: Yahoo Finance"
}
```

---

## Example Use Cases

- Stock market research and live price checks
- Analyst recommendation summaries
- Company financial fundamentals analysis
- Real-time financial news monitoring
- AI-powered investment research assistant

---

## Academic Relevance

This project demonstrates:

- **Tool-Augmented LLMs** — agent uses external tools (YFinance, DuckDuckGo) to ground responses in real data
- **Full-Stack AI Application** — React frontend + FastAPI backend + LLM agent
- **Real-Time Data Integration** — live stock market data and web search
- **Prompt Engineering** — structured agent instructions for consistent output formatting

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
