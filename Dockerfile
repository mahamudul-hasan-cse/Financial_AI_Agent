# =============================================================================
# Financial AI Agent — Multi-stage Dockerfile
# =============================================================================
#
# Build:  docker build -t financial-ai-agent .
# Run:    docker run -p 8001:8001 --env-file .env financial-ai-agent
#         (or: docker compose up --build)
#
# Stage 1 — Node 22: compile the React/Vite frontend to static files
# Stage 2 — Python 3.12: build a virtualenv with CPU-only torch + NLP models
# Stage 3 — Python 3.12-slim runtime: venv + app source, no compilers
#
# The spaCy and sentence-transformers models baked into the image are the
# same ones the code loads at runtime (SPACY_MODEL / EMBEDDING_MODEL), so the
# container starts fully offline. Override both with --build-arg.
# =============================================================================

ARG SPACY_MODEL=en_core_web_sm
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
ARG TORCH_VERSION=2.14.1


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Build React frontend
# ─────────────────────────────────────────────────────────────────────────────
FROM node:22-alpine AS frontend-build

WORKDIR /build/frontend

# Install deps first — layer is cached until package-lock.json changes
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Python dependencies + models
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS python-build

ARG SPACY_MODEL
ARG EMBEDDING_MODEL
ARG TORCH_VERSION

# Compilers are only needed here, in case a dependency has no wheel.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libffi-dev \
 && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache/sentence_transformers \
    HF_HUB_DISABLE_PROGRESS_BARS=1
RUN python -m venv /opt/venv

# CPU-only torch first: the default PyPI wheel bundles CUDA (~2+ GB extra).
RUN pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Bake the exact models the app loads into the image.
RUN python -m spacy download "${SPACY_MODEL}" \
 && python -c "import spacy; spacy.load('${SPACY_MODEL}'); print('spaCy ${SPACY_MODEL} OK')" \
 && python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('${EMBEDDING_MODEL}'); print('sentence-transformers ${EMBEDDING_MODEL} OK')"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ARG SPACY_MODEL
ARG EMBEDDING_MODEL

LABEL org.opencontainers.image.title="Financial AI Agent"
LABEL org.opencontainers.image.description="FastAPI + OpenRouter/Groq/Ollama LLM + React — full-stack financial assistant"

# curl: HEALTHCHECK · ca-certificates: TLS to OpenRouter / Groq / Yahoo / DuckDuckGo
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache/sentence_transformers \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    SPACY_MODEL=${SPACY_MODEL} \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    DATABASE_PATH=/app/data/financial_agent.db \
    AGNO_TELEMETRY=false \
    PORT=8001 \
    FORWARDED_ALLOW_IPS=127.0.0.1

COPY --from=python-build /opt/venv /opt/venv
COPY --from=python-build /opt/hf-cache /opt/hf-cache

# ── Application source ──────────────────────────────────────────────────────
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
COPY backend/  ./backend/
COPY schemas/  ./schemas/
COPY tools/    ./tools/
# Static data (company_tickers.json — *.db excluded via .dockerignore)
COPY data/     ./data/

COPY --from=frontend-build /build/frontend/dist ./frontend/dist

# ── Non-root user; data/ and outputs/ are writable (mount volumes there) ────
RUN mkdir -p outputs/charts outputs/sheets data \
 && groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --shell /bin/false --no-create-home appuser \
 && chown -R appuser:appgroup /app
USER appuser

EXPOSE 8001

# Liveness: /api/health answers 200 even when "degraded" (no LLM key), so an
# unconfigured container is reported unhealthy only if the server is down.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/api/health" > /dev/null || exit 1

# Single worker on purpose: sessions cache, rate limits, the TTL data cache
# and stats live in process memory (SQLite holds the durable data).
# Secrets (GROQ_API_KEY, OPENROUTER_API_KEY, API_KEY) come from the runtime env (--env-file /
# platform secrets) — never bake them into the image.
# Behind a PaaS load balancer set FORWARDED_ALLOW_IPS='*' so rate limits use
# the real client IP from X-Forwarded-For.
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port \"${PORT}\" --proxy-headers --forwarded-allow-ips \"${FORWARDED_ALLOW_IPS}\" --log-level info"]
