# Remote Ingestion Service

A standalone FastAPI service that runs the WEGA chunker + Stellar embeddings
pipeline and writes chunks to the shared pgvector database.

It's meant to live on a separate VM (typically the only one with access to the
internal WEGA SDK + Stellar gateway). The main RAG backend reaches it over HTTP
when `REMOTE_INGEST=true` is set.

## When to use this

| Mode                              | What happens                                                 |
|-----------------------------------|--------------------------------------------------------------|
| `REMOTE_INGEST=false` (default)   | The backend runs ingestion in-process (local Vertex or WEGA) |
| `REMOTE_INGEST=true`              | The backend proxies uploads to this service via SSE          |

## Provider modes

The remote service itself has two ingestion implementations, selected with
the `LLM_PROVIDER` env var:

| `LLM_PROVIDER` | Chunker                | Embeddings                         | Requires           |
|----------------|------------------------|------------------------------------|--------------------|
| `wega`         | `WegaChunker` (wheel)  | `StellarGenAI` from `get_llm.py`   | The WEGA chunker wheel + a copy of the repo's `get_llm.py` |
| `vertex`       | `pypdf` + char chunker | Google Vertex AI (genai)           | `pip install pypdf google-genai` + a service-account JSON |

Both modes write to the same `vector.chunk_embeddings` table, so the rest of
the RAG pipeline (retrieve, rerank, generate) doesn't care which one ran.

### Quick local test with Vertex

```bash
cd ingest_remote
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # public deps only
cp .env.example .env                   # edit: set LLM_PROVIDER=vertex,
                                       # GOOGLE_APPLICATION_CREDENTIALS, VERTEX_PROJECT
uvicorn app.main:app --port 8090 --reload

# In another shell — point the main backend at it:
export REMOTE_INGEST=true REMOTE_INGEST_URL=http://localhost:8090
./entrypoint.sh
```

## Setting up the WEGA mode

Two things are needed only for `LLM_PROVIDER=wega`:

### 1. The WEGA chunker wheel

The WEGA chunker (`wega_chunker`) is shipped as a pyarmoured wheel you can't
publish to a public index. Drop it into `wheels/`:

```text
ingest_remote/
└── wheels/
    └── wega_chunker-1.2.3-py3-none-any.whl
```

`wheels/` and `*.whl` are gitignored — never commit these.

### 2. `get_llm.py`

Stellar embeddings are not a separate wheel. The remote service ships its
own `app/stellar_client.py` (a portable copy of the main backend's client)
which in turn imports `get_llm.StellarGenAI` to handle the COIN OAuth token
and the OpenAI-compatible httpx client. Copy `get_llm.py` from the project
root into the `ingest_remote/` deployable (or set `STELLAR_GETLLM_PATH` to
its full path).

Discovery order for `get_llm.py`:

1. `$STELLAR_GETLLM_PATH` (when set)
2. `ingest_remote/get_llm.py`
3. `<repo-root>/get_llm.py` (handy when running from the retrieval repo)

The embedding model is locked to `gte-large-en-v1.5` inside `stellar_client.py`.

`get_llm.py` is gitignored — copy it during deploy, don't commit it.

## Running locally

```bash
cd ingest_remote
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install ./wheels/*.whl          # internal wheels
cp .env.example .env                 # then fill in real secrets
uvicorn app.main:app --port 8090 --reload
```

## Running in Docker

```bash
# 1. Place the internal wheels in ./wheels/ first.
docker build -t rag-ingest-remote .
docker run -p 8090:8090 --env-file .env rag-ingest-remote
```

The Dockerfile fails fast if no wheel files are present in `./wheels/`.

## API

### `GET /health`

```json
{
  "status": "ok",
  "service": "ingest_remote",
  "azure_di_configured": true,
  "pg_database": "chunker_db",
  "pg_index": "chunk_embeddings"
}
```

### `POST /ingest`

Multipart form upload. Streams Server-Sent Events back to the caller, one
event per pipeline stage (`chunk` → `embed` → `store`) plus a final `done`
event with the full summary.

Fields:

| Field             | Required | Description                                           |
|-------------------|----------|-------------------------------------------------------|
| `file`            | yes      | The PDF to ingest                                     |
| `document_name`   | no       | Logical name to store in `documentName`               |
| `overrides_json`  | no       | JSON dict overriding any `Settings` key per-request   |

Headers:

| Header              | Notes                                                   |
|---------------------|---------------------------------------------------------|
| `X-Ingest-Secret`   | Required only if `SHARED_SECRET` is set in the service. |

Example with curl:

```bash
curl -N -X POST http://ingest-vm:8090/ingest \
  -H 'X-Ingest-Secret: hunter2' \
  -F 'file=@./mydoc.pdf' \
  -F 'document_name=mydoc.pdf'
```

Example event stream:

```text
event: start
data: {"file": "/tmp/.../mydoc.pdf", "document_name": "mydoc.pdf"}

event: stage
data: {"type": "stage", "stage": "chunk", "status": "start", ...}

event: stage
data: {"type": "stage", "stage": "embed", "status": "progress", "done": 16, "total": 42}

event: done
data: {"type": "done", "chunks_count": 42, "rows_inserted": 42, "elapsed_seconds": 18.4, ...}
```
