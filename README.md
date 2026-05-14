# RAG Studio

> Production-grade Retrieval-Augmented Generation on top of WEGA-chunked PDFs
> and Stellar-hosted LLMs. Hybrid search, Anthropic-style contextual retrieval,
> listwise LLM reranking, CRAG self-correction, streaming citations, and a
> built-in benchmark harness — all behind a single React UI.

![status](https://img.shields.io/badge/status-prototype-blueviolet) ![python](https://img.shields.io/badge/python-3.11%2B-blue) ![stack](https://img.shields.io/badge/stack-FastAPI%20%2B%20pgvector%20%2B%20React-7c5cff)

---

## Why this project exists

We already chunk and embed PDFs into Postgres via [`ingest.py`](ingest.py)
(WEGA chunker + Stellar `gte-large-en-v1.5`). What was missing was the
retrieval-and-generation layer that turns those vectors into answers our
users actually trust.

This project is that layer — built to be **best-in-class** on three axes
that matter to a principal engineer:

| Axis | What we do |
|---|---|
| **Retrieval quality** | Hybrid dense+BM25 fusion, Anthropic contextual prefixes, listwise LLM rerank (RankGPT family), HyDE, query rewriting, MMR diversification, CRAG self-grader. Each is a toggle. |
| **Observability** | Latency breakdown per pipeline stage, query audit log, top-doc analytics, p50/p95/p99, strategy-mix dashboard. |
| **Provability** | Built-in benchmark harness: golden Q/A set, recall@k, MRR, nDCG, faithfulness, context precision — *with run history* so we can show "X strategy moved recall@5 from 0.62 → 0.81". |

---

## The retrieval pipeline (one screen)

```
   query
     │
     ├──► rewrite (LLM)  ─►   3 variants
     ├──► HyDE (LLM)     ─►   hypothetical answer
     ▼
   embed (gte-large-en-v1.5)
     │
     ├──► dense (pgvector HNSW, cosine)        — 50 hits
     └──► sparse (Postgres FTS / ts_rank_cd)   — 50 hits
     │
   Reciprocal Rank Fusion  ───────────────────► 20 candidates
     │
   Listwise rerank (Llama-3.3-70B, RankGPT)  ─► 20 re-ordered
     │
   MMR diversify  ────────────────────────────► top-K (default 8)
     │
   CRAG self-grader (Llama-3.1-8B)           ─► confidence score
     │
   Generate answer w/ inline [N] citations    (Llama-4-Scout, streaming SSE)
```

Every stage is independently toggleable in the UI — perfect for the
"strategy A/B" demo: ask the same question with rerank on vs off, watch
the citation set change.

### Headline technique: Contextual Retrieval (Anthropic, Sept 2024)

Before embedding, each chunk gets a 50–100 token *contextual prefix*
written by an LLM with the entire document in scope (Llama-4-Scout's 256K
window makes this trivial). We then embed `(prefix + chunk)` and store
the contextual embedding next to the original. Anthropic reported a
**49% reduction in retrieval failures** with this single change. It's the
biggest quality lever we have, and we run it as a one-click ingestion
post-step from the UI.

---

## Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI (async), asyncpg, pydantic v2, sse-starlette |
| Storage | The existing pgvector schema (no new infra) |
| LLMs | Stellar gateway (`gte-large-en-v1.5`, `Llama-4-Scout`, `Llama-3.3-70B`, `Llama-3.1-8B`) — pluggable via env |
| Frontend | Vite + React 18 + TypeScript + Tailwind + Radix + Recharts |
| Deploy | Single Dockerfile → single container, talks to your existing PG |

We deliberately **do not** ship a local cross-encoder model. The reranker
is the LLM itself (listwise / RankGPT) — denser to demo and cited
recent literature.

---

## Project layout

```
.
├── ingest.py                         # (existing) WEGA chunker + embedder → pgvector
├── get_llm.py                        # (existing) Stellar/Vertex client factory
├── backend/
│   ├── migrations/001_rag_extensions.sql
│   ├── requirements.txt
│   └── app/
│       ├── main.py                   # FastAPI app + lifespan
│       ├── config.py                 # env-driven settings
│       ├── db.py                     # asyncpg pool, dim discovery
│       ├── schemas.py                # pydantic API contracts
│       ├── stellar_client.py         # thin async wrapper over get_llm.py
│       ├── pipeline/
│       │   ├── retrieve.py           # orchestrator
│       │   ├── dense.py              # pgvector HNSW
│       │   ├── sparse.py             # Postgres FTS BM25-like
│       │   ├── rrf.py                # Reciprocal Rank Fusion
│       │   ├── rerank.py             # LLM listwise reranker
│       │   ├── mmr.py                # diversity reranker
│       │   ├── hyde.py               # hypothetical doc embeddings
│       │   ├── rewrite.py            # multi-query rewrites
│       │   ├── contextual.py         # Anthropic contextual prefix gen
│       │   ├── crag.py               # corrective-RAG self-grader
│       │   └── generate.py           # streaming answer w/ citations
│       ├── routers/
│       │   ├── retrieve.py           # /api/retrieve + /api/chat (SSE)
│       │   ├── ingest.py             # /api/ingest/* (SSE log)
│       │   ├── documents.py          # /api/documents
│       │   ├── analytics.py          # /api/analytics/*
│       │   └── bench.py              # /api/bench/*
│       └── scoring/
│           └── metrics.py            # recall@k, MRR, nDCG, faithfulness, ctx precision
└── frontend/
    └── src/
        ├── App.tsx                   # tab shell
        ├── components/
        │   ├── RetrievalTab.tsx      # demo screen: chat + citations + sources + Sankey
        │   ├── IngestionTab.tsx      # upload PDF, run contextual gen
        │   ├── AnalyticsTab.tsx      # p50/p95, strategy mix, top docs
        │   ├── BenchmarkTab.tsx      # golden set + run + trend chart
        │   ├── StrategyToggles.tsx   # the demo's killer feature
        │   ├── LatencyBar.tsx        # per-stage breakdown
        │   └── SourceCard.tsx
        ├── lib/{api,cn}.ts
        └── types/index.ts
```

---

## Database extensions

The migration is idempotent and only adds — never modifies your existing
`vector.chunk_embeddings`:

| Object | Purpose |
|---|---|
| `chunk_embeddings.content_tsv` | Generated `tsvector` for FTS hybrid |
| `idx_chunk_content_tsv` | GIN index on the above |
| `vector.chunk_context` | Anthropic contextual prefix + its embedding (separate HNSW) |
| `vector.queries` | Audit log of every retrieval call |
| `vector.golden_questions` | Q/A golden set for the benchmark harness |
| `vector.bench_runs` | Historical metrics per benchmark batch |

---

## Setup

One command brings it all up — `entrypoint.sh` handles env loading, deps,
frontend build, migrations, and serves.

```bash
# 1) Configure (do NOT commit .env)
cp .env.example .env
# Fill in PG_*, COIN_*, STELLAR_* per your VDI environment

# 2) Production: install + build + serve
./entrypoint.sh              # → http://localhost:8080

# 2b) Dev mode: vite hot-reload + uvicorn --reload
./entrypoint.sh --dev        # frontend on :5173, backend on :8080

# 2c) Fast restart (deps already installed, frontend already built)
./entrypoint.sh --skip-install --skip-build

# 2d) Single container
docker compose up -d --build # → http://localhost:8080
```

Flags:

| flag | effect |
|---|---|
| `--dev` | spawn Vite dev server alongside uvicorn `--reload` |
| `--skip-install` | skip `pip install` + `npm install` |
| `--skip-build` | skip `npm run build` (uses existing `frontend/dist`) |
| `--no-venv` | don't auto-create a `.venv` (useful in Docker / managed envs) |
| `--host`, `--port` | override `APP_HOST` / `APP_PORT` |

The migration is applied automatically on backend startup. Postgres
credentials are read from env vars only — nothing sensitive in code.

### Long-running auth (no token-expiry surprises)

`get_llm.py` now refreshes its underlying client every
`COIN_TOKEN_TTL_SECONDS` (default 14 min) using a property-based factory
pattern. The RAG service can stay up for days without ever hitting an
expired COIN token mid-request.

---

## Demo script for the principal-engineer pitch

A 7-minute walk-through with hard numbers.

### 1. "Show me ingestion" (1 min)
- Open **Ingestion tab**. Stat bar: docs / chunks / contextualised% / dim.
- Drag a PDF in. Live SSE log of WEGA chunking → embedding → upsert.
- Click "Generate context" → live progress bar as each chunk gets an
  Anthropic-style prefix. **Mention "49% recall lift from a paper that
  shipped in September"**.

### 2. "Show me retrieval" (2 min)
- Open **Retrieval tab**. Ask a real question from the corpus.
- Streamed answer with `[1] [2] [3]` chips. Click a chip → source highlights.
- Right rail: ranked chunks with RRF score + per-source contribution.
- Below: latency Sankey — every stage in milliseconds.

### 3. "Show me the A/B" (1 min)
- Toggle off **listwise rerank**. Re-ask the same question.
- Citation set changes; latency drops ~200 ms; document the trade.
- Toggle **HyDE** on, **rerank** back on. Run again.

### 4. "Show me the dashboard" (1 min)
- Open **Analytics tab**. p50/p95/p99 latencies, strategy mix bar chart,
  top retrieved documents, full audit log.

### 5. "Show me proof, not vibes" (2 min)
- Open **Benchmark tab**. Add 5 golden questions with their
  ground-truth chunk IDs.
- Run with `(hybrid + rerank + contextual)` — get recall@5, MRR, nDCG,
  faithfulness, context-precision.
- Toggle off contextual, re-run with label "no-contextual" — the trend
  chart shows the delta. **This is the slide.**

### Talking points

- **"Production techniques only"** — Anthropic Contextual Retrieval, RankGPT
  listwise rerank, CRAG, HyDE, RRF. All recent (2023–24) papers.
- **"No new infra"** — sits on the pgvector instance we already have.
- **"Stellar-only, no model file drift"** — every model is hosted; no
  ONNX artefact to ship into the VDI.
- **"Roadmap"** — `vlm2vec-full` for figure/chart embeddings,
  `Whisper-large-v3-turbo` for meeting audio ingestion, GraphRAG-lite
  for entity-centric retrieval, JWT/RBAC for multi-tenant.

---

## API quick reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | DB + Stellar status |
| GET | `/api/config` | model assignments & defaults |
| POST | `/api/retrieve` | retrieval only (no generation) |
| POST | `/api/chat` (SSE) | retrieval + streamed answer |
| GET | `/api/documents` | document list with contextual coverage |
| GET | `/api/documents/{name}/chunks` | inspect a doc |
| POST | `/api/ingest/wega` (SSE) | upload PDF + run WEGA ingest |
| POST | `/api/ingest/contextual` (SSE) | run contextual prefix generation |
| GET | `/api/analytics/summary` | p50/p95, strategy mix, top docs |
| GET | `/api/analytics/queries` | recent query log |
| GET/POST | `/api/bench/questions` | golden set CRUD |
| POST | `/api/bench/run` | run a benchmark batch |
| GET | `/api/bench/runs` | historical runs |

---

## What's intentionally left for v2

- AuthN/AuthZ (JWT, RBAC) — out of scope for the funded-prototype phase
- Conversation history persistence beyond in-request
- `vlm2vec-full` figure embeddings (separate ingestion adapter)
- `Whisper-large-v3-turbo` audio ingestion adapter
- GraphRAG entity extraction + graph-walk retrieval
- Multi-tenant sharding on document_name

Each of these has a clear slot in the existing architecture — the
backend pipeline is a directory of independent stages, the UI is a
directory of independent tabs, the DB is additive migrations.

---

## Credits / inspiration

- Anthropic — [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (Sept 2024)
- Sun et al., 2023 — *Is ChatGPT Good at Search? LLMs as Re-Ranking Agents*
- Gao et al., 2022 — *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE)
- Yan et al., 2024 — *Corrective Retrieval Augmented Generation* (CRAG)
- Cormack et al., 2009 — *Reciprocal Rank Fusion outperforms Condorcet*
