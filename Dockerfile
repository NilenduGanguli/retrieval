# =============================================================================
# Multi-stage: build the React bundle, then ship FastAPI + bundle in one image.
# =============================================================================

# ---- Stage 1: frontend ------------------------------------------------------
FROM node:20-alpine AS frontend-build
WORKDIR /work/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: runtime -------------------------------------------------------
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /tmp/req.txt
RUN pip install --upgrade pip && pip install -r /tmp/req.txt

# App code
COPY ingest.py get_llm.py entrypoint.sh ./
COPY backend/ ./backend/
COPY frontend/package.json ./frontend/package.json
COPY --from=frontend-build /work/frontend/dist ./frontend/dist
RUN chmod +x ./entrypoint.sh

ENV APP_HOST=0.0.0.0 \
    APP_PORT=8080

EXPOSE 8080

# Health check hits /api/health
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8080/api/health || exit 1

# Inside the image, deps are pre-installed and the frontend is pre-built,
# so we tell entrypoint to skip both and just launch the server.
CMD ["./entrypoint.sh", "--skip-install", "--skip-build", "--no-venv"]
