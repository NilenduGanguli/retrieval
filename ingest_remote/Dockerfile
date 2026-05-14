FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Public deps first (cacheable layer)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Internal wheel: drop wega_chunker-*.whl into ./wheels/ before building.
# Skipped silently when the directory is empty (vertex-only deployments don't
# need a WEGA wheel).
COPY wheels/ ./wheels/
RUN if ls ./wheels/*.whl >/dev/null 2>&1; then \
        pip install ./wheels/*.whl; \
    else \
        echo "no wheel files in ./wheels/ — wega mode will be unavailable"; \
    fi

# Stellar embeddings reuse get_llm.py from the main repo. Drop a copy next to
# ./app/ before building (or set STELLAR_GETLLM_PATH at runtime). Optional in
# vertex-only deployments.
COPY app/ ./app/
COPY --chown=root:root get_llm.p[y] ./get_llm.py

EXPOSE 8090
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
