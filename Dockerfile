# =============================================================================
# Financial AI Agent — Multi-stage Dockerfile
# =============================================================================
#
# Build:  docker build -t financial-ai-agent .
# Run:    docker compose up
#
# Stage 1 — Node 22: compile React/Vite frontend to static files
# Stage 2 — Python 3.12-slim: FastAPI backend + pre-downloaded NLP models
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Build React frontend
# ─────────────────────────────────────────────────────────────────────────────
FROM node:22-alpine AS frontend-build

WORKDIR /build/frontend

# Install deps first — layer is cached until package-lock.json changes
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --frozen-lockfile

COPY frontend/ ./
RUN npm run build


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Python runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Financial AI Agent"
LABEL org.opencontainers.image.description="FastAPI + Groq/Ollama LLM + React — full-stack financial assistant"

# ── System packages ───────────────────────────────────────────────────────────
# build-essential + libffi-dev : compile any C-extension wheels
# curl                         : HEALTHCHECK + outbound API calls
# ca-certificates              : TLS verification for Groq/HuggingFace APIs
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
        curl \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Matplotlib must use non-interactive backend in a headless container
    MPLBACKEND=Agg \
    # Pin HuggingFace/sentence-transformers cache inside /app so it is
    # owned by appuser and can optionally be volume-mounted for persistence
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/hub \
    # Silence HF Hub "token not set" warning in logs
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TOKENIZERS_PARALLELISM=false

# ── Layer 1: Python dependencies (cached until requirements.txt changes) ──────
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Layer 2: Pre-download NLP models (cached unless deps change) ──────────────
# Baking models into the image avoids cold-start downloads on every container
# restart and makes the app work fully offline after the first build.
RUN python -m spacy download en_core_web_sm \
 && python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('all-MiniLM-L6-v2'); \
print('sentence-transformers: all-MiniLM-L6-v2 cached OK')"

# ── Layer 3: Application source (changes most often — last for cache efficiency)
# Core app files
COPY api.py \
     financial_agent.py \
     nlp_pipeline.py \
     intent_classifier.py \
     cache.py \
     config.py \
     market_service.py \
     persistence.py \
     research_service.py \
     ./
# Python packages
COPY backend/  ./backend/
COPY schemas/  ./schemas/
COPY tools/    ./tools/
# Static data (company_tickers.json etc. — *.db excluded via .dockerignore)
COPY data/     ./data/

# ── Output directories (will be bind-mounted at runtime — pre-create for perms)
RUN mkdir -p outputs/charts outputs/sheets

# ── Frontend static files from Stage 1 ───────────────────────────────────────
COPY --from=frontend-build /build/frontend/dist ./frontend/dist

# ── Non-root user (least-privilege principle) ─────────────────────────────────
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup \
             --shell /bin/false \
             --no-create-home \
             appuser \
 && chown -R appuser:appgroup /app
USER appuser

EXPOSE 8001

# start_period gives uvicorn + agno time to fully initialise before checks begin
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fs http://localhost:8001/api/health || exit 1

# Single worker is correct — sessions are stored in-memory (no shared state)
CMD ["uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--log-level", "info", \
     "--access-log"]
