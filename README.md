# 💰 Financial AI Agent (Multi-Agent System)

An intelligent multi-agent financial analysis assistant built using the **Phidata framework**. This system combines real-time stock market data with web intelligence using Large Language Models (LLMs).

---

## 🚀 Project Overview

The Financial AI Agent is a **multi-agent AI system** designed to perform financial analysis using real-time data, analyst recommendations, and web-based financial news.

The system integrates multiple specialized agents coordinated by a central AI controller to deliver comprehensive financial insights.

---

## 🧠 Architecture

This project uses a **Multi-Agent Architecture** consisting of:

### 🔎 Web Search Agent
- Uses Groq LLM (Llama 3.3 70B Versatile)
- Performs real-time financial web searches
- Retrieves latest company news and market updates
- Includes source citations for transparency

---

### 📊 Finance Agent
- Uses YFinance Tools
- Retrieves:
  - Real-time stock prices
  - Analyst recommendations
  - Company fundamentals
  - Company news
- Presents data in structured table format

---

### 🤖 Multi-AI Coordinator Agent
- Combines outputs from Web Search and Finance Agents
- Performs comprehensive stock analysis
- Currently configured to analyze **NVIDIA (NVDA)**

---

## 🖥️ Application Modes

### ✅ Command Line Interface (CLI)
Run financial queries directly from terminal.
<img width="1066" height="501" alt="image" src="https://github.com/user-attachments/assets/e36ba18a-805a-4c3b-82b0-10ed84c1e069" />


---

### 🌐 Web Playground Interface
Interactive browser-based UI for real-time analysis.


Built using FastAPI and Phidata Playground.
<img width="1913" height="904" alt="image" src="https://github.com/user-attachments/assets/a91e9447-0a4f-42a5-9f0d-66aab0d82760" />

---

## 🛠️ Tech Stack

- Python
- Phidata (AI Agent Framework)
- Groq LLM (Llama 3.3 70B)
- YFinance API
- DuckDuckGo Search API
- FastAPI
- Uvicorn
- Python Dotenv

---

## 📂 Project Structure
Financial_AI_Agent
│
├── financial_agent.py # CLI based AI financial agent
├── playground.py # Web UI playground interface
├── requirements.txt # Dependencies
├── .env # API keys (ignored in Git)
└── README.md # Project documentation


---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
https://github.com/mahamudul-hasan-cse/Financial_AI_Agent.git


---

### 2️⃣ Create Virtual Environment

Activate environment:
python -m venv .venv
Windows:
.venv\Scripts\activate

---

### 3️⃣ Install Dependencies
pip install -r requirements.txt
---

### 4️⃣ Configure Environment Variables

Create `.env` file:

PHI_API_KEY=your_phi_key
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key


---

## ▶️ Running The Project

### CLI Mode


---

## ✨ Features

- Multi-Agent AI coordination
- Real-time stock market analysis
- Financial news aggregation
- Analyst recommendation insights
- Structured financial reporting
- Interactive web playground
- Source-backed AI responses

---

## 🔐 Security Note

API keys are stored in `.env` and excluded from version control for security.

---

## 📈 Example Use Cases

- Stock market research
- Financial investment analysis
- Real-time market news monitoring
- AI-based financial assistant development
- Multi-agent LLM orchestration research

---

## 🎓 Academic Relevance

This project demonstrates:

- Multi-Agent AI Systems
- LLM Orchestration
- Real-Time Data Integration
- Financial AI Applications
- Tool-Augmented Language Models

---

## 🔮 Future Improvements

- Add support for multiple stock comparisons
- Portfolio risk analysis
- Historical trend prediction
- Deployment to cloud platform
- Dashboard visualization
- Voice interaction support

---

## 👨‍💻 Author

**Mahmudul Hasan**  
CSE Student | AI & NLP Enthusiast  

GitHub: https://github.com/mahamudul-hasan-cse

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgements

- Phidata Framework
- Groq LLM Platform
- Yahoo Finance API
- DuckDuckGo Search API

