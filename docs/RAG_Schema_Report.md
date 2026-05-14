# RAG Studio — Database Schema Report

A complete inventory of every table, column, index, and view introduced or
modified by the RAG framework, who writes to each, and at which pipeline
stage.

All objects live in the `vector` schema of the `chunker_db` Postgres
database. Migration files: `backend/migrations/*.sql`. Run automatically
at startup by [backend/app/main.py:lifespan](../retrieval/backend/app/main.py).

---

## 1. Summary

### 1.1 New tables

| Table                          | Migration | Purpose                                          |
|--------------------------------|-----------|--------------------------------------------------|
| `vector.chunk_context`         | 001       | LLM-generated context prefixes + contextual embeddings (Anthropic Contextual Retrieval) |
| `vector.queries`               | 001       | Audit log of every `/api/retrieve` & `/api/chat` call |
| `vector.golden_questions`      | 001       | Curated Q/A pairs for offline evaluation         |
| `vector.bench_runs`            | 001       | Aggregate metrics from each benchmark run        |
| `vector.documents`             | 002       | One row per uploaded source file (size, S3 URI, soft-delete) |

### 1.2 Modified tables

| Table                          | Change                              | Migration | Purpose                                          |
|--------------------------------|-------------------------------------|-----------|--------------------------------------------------|
| `vector.chunk_embeddings`      | `+ content_tsv tsvector` (generated stored) | 001 | Sparse / BM25-like full-text search              |
| `vector.chunk_embeddings`      | `+ deleted_at TIMESTAMPTZ`          | 002       | Soft-delete (retrieval filters `deleted_at IS NULL`) |
| `vector.chunk_context`         | `+ context_embedding vector(D)`     | dynamic ([db.py](../retrieval/backend/app/db.py)) | Vector for hybrid search, dim discovered at runtime |
| `vector.queries`               | `+ stage_tokens JSONB`              | 003       | Per-stage token spend breakdown for analytics    |

### 1.3 Views

| View                          | Migration | Definition                                                |
|-------------------------------|-----------|-----------------------------------------------------------|
| `vector.active_chunks`        | 002       | `SELECT * FROM chunk_embeddings WHERE deleted_at IS NULL` |

### 1.4 Pre-existing table — referenced but not created by us

| Table                          | Owner            | Why we care                                      |
|--------------------------------|------------------|--------------------------------------------------|
| `vector.chunk_embeddings`      | Original `ingest.py` (WEGA path) | The base table that holds every chunk + embedding. We extend it via migration 001 + 002. |

---

## 2. Migration order and idempotency

```
backend/migrations/
├── 001_rag_extensions.sql       ← FTS column, chunk_context, queries,
│                                  golden_questions, bench_runs
├── 002_documents_and_softdelete.sql  ← documents table (was a view),
│                                       deleted_at + partial index,
│                                       active_chunks view
└── 003_stage_tokens.sql         ← queries.stage_tokens JSONB
```

Every migration uses one of:

- `CREATE TABLE IF NOT EXISTS`
- `CREATE INDEX IF NOT EXISTS`
- `DO $$ … IF NOT EXISTS … ALTER TABLE … END $$` (for column adds)
- `CREATE OR REPLACE VIEW`

so re-running them is safe. Migration 002 also handles the unusual case
where migration 001 historically declared `vector.documents` as a **view**:
it inspects `pg_class.relkind` and only drops the view if that's still
what's there, before creating the real table.

---

## 3. Table reference

### 3.1 `vector.chunk_embeddings` — chunks + dense embedding

**Origin**: pre-existing (created by the original [ingest.py](../retrieval/ingest.py)).
We added two columns via migrations 001 + 002.

```sql
CREATE TABLE vector.chunk_embeddings (
    id                BIGSERIAL PRIMARY KEY,
    content           TEXT     NOT NULL,
    "chunkUUID"       TEXT     UNIQUE,
    "pageNumber"      INTEGER,
    "tokenCount"      INTEGER,
    "chunkType"       TEXT,
    "chunkBoundingBox" JSONB,
    "documentName"    TEXT,
    "jobId"           TEXT,
    embedding         vector(D),
    content_tsv       tsvector  -- ADDED in 001 (generated stored)
                      GENERATED ALWAYS AS
                      (to_tsvector('english', COALESCE(content, ''))) STORED,
    deleted_at        TIMESTAMPTZ           -- ADDED in 002
);
```

**Indexes**
| Name                           | Type | Columns                                  | Source        |
|--------------------------------|------|------------------------------------------|---------------|
| `chunk_embeddings_pkey`        | btree| `id` PK                                  | pre-existing  |
| `chunk_embeddings_chunkUUID_key` | unique btree | `chunkUUID`                       | pre-existing  |
| `chunk_embeddings_hnsw_idx`    | HNSW | `embedding` `vector_cosine_ops`          | pre-existing  |
| `idx_chunk_content_tsv`        | GIN  | `content_tsv`                            | migration 001 |
| `idx_chunk_active`             | btree partial | `("documentName") WHERE deleted_at IS NULL` | migration 002 |

**Writers**
| Stage / route                                 | Operation                            | Source file |
|-----------------------------------------------|--------------------------------------|-------------|
| **Ingestion — local Vertex** (backend)        | `INSERT … ON CONFLICT ("chunkUUID") DO NOTHING` | [backend/app/pipeline/local_ingest.py](../retrieval/backend/app/pipeline/local_ingest.py) |
| **Ingestion — Stellar WEGA** (project root)   | `INSERT … ON CONFLICT (id) DO UPDATE`           | [ingest.py](../retrieval/ingest.py) |
| **Ingestion — remote WEGA** (separate VM)     | `INSERT … ON CONFLICT (id) DO UPDATE`           | [ingest_remote/app/ingest_core.py](../retrieval/ingest_remote/app/ingest_core.py) |
| **Ingestion — remote Vertex** (separate VM)   | `INSERT …`, narrow / wide-shape auto-detect     | [ingest_remote/app/local_ingest.py](../retrieval/ingest_remote/app/local_ingest.py) |
| **Document soft-delete** (DELETE route)       | `UPDATE … SET deleted_at = now()`               | [backend/app/routers/documents.py](../retrieval/backend/app/routers/documents.py) |

**Readers (heavy)**
- Dense retrieval — `pipeline/dense.py`
- Sparse retrieval — `pipeline/sparse.py`
- Chunks-of-doc endpoint — `routers/documents.py`
- Health endpoint — `db.py:healthcheck`

**Lifecycle**

```
                 INSERT during ingest
                          │
                          ▼
                ┌──────────────────┐    UPDATE deleted_at = now()
   ingestion ──►│  active chunk    │──────────────────┐
                └────────┬─────────┘                  │
                         │                            ▼
                  dense / sparse              ┌──────────────────┐
                  retrieval reads             │  soft-deleted    │
                         │                    │  chunk (hidden)  │
                         ▼                    └──────────────────┘
                ┌──────────────────┐
                │  retrieval hits  │
                └──────────────────┘
```

---

### 3.2 `vector.chunk_context` — Anthropic Contextual Retrieval

```sql
CREATE TABLE vector.chunk_context (
    chunk_id           BIGINT PRIMARY KEY
                       REFERENCES vector.chunk_embeddings(id) ON DELETE CASCADE,
    context_text       TEXT        NOT NULL,
    context_embedding  vector(D),                   -- added dynamically by db.py
    generated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    generator_model    TEXT
);
```

The `context_embedding` column is **added at app startup**, not in the SQL
migration, because the embedding dimension depends on which provider is
configured (`gte-large-en-v1.5` → 1024, `text-embedding-005` → 768). See
[backend/app/db.py:ensure_contextual_embedding_column](../retrieval/backend/app/db.py).

**Indexes**
| Name                                | Type | Columns                                       | Source        |
|-------------------------------------|------|-----------------------------------------------|---------------|
| `chunk_context_pkey`                | btree| `chunk_id` PK                                 | migration 001 |
| `idx_chunk_context_chunk`           | btree| `chunk_id`                                    | migration 001 |
| `idx_chunk_context_embedding_hnsw`  | HNSW | `context_embedding` `vector_cosine_ops`       | dynamic (db.py) |

**Writers**
| Stage                                          | Operation                          | Source file |
|------------------------------------------------|------------------------------------|-------------|
| Contextual generation — POST `/api/ingest/contextual` | `INSERT … ON CONFLICT (chunk_id) DO UPDATE` | [backend/app/pipeline/contextual.py:_persist_context](../retrieval/backend/app/pipeline/contextual.py) |

The contextual stage is run **after** ingestion, against existing chunks
that don't yet have a context row. The loop in
[`contextual.py:generate_context_batch`](../retrieval/backend/app/pipeline/contextual.py)
walks chunks with `LEFT JOIN chunk_context WHERE ctx.chunk_id IS NULL`,
generates a 50–100 token prefix per chunk, embeds the combined text, and
writes the row.

**Readers**
- Dense retrieval with `use_contextual=True` toggle — `pipeline/dense.py` joins this table and prefers `context_embedding <=> q.v` when present.
- Document explorer panel — shows the `context_text` next to each chunk.
- Coverage % — `db.py:healthcheck` reports `contextual_chunks` count.

**Lifecycle**

```
   chunk_embeddings.id ───►(FK ON DELETE CASCADE)──► chunk_context.chunk_id
                                       │
                                       │ INSERT during
                                       │ /api/ingest/contextual
                                       ▼
                            ┌────────────────────────┐
                            │  context_text (TEXT)   │
                            │  context_embedding (V) │
                            └────────────────────────┘
                                       │
                                       │ JOINed in dense.py
                                       │ when use_contextual=True
                                       ▼
                            ┌──── contextual hits ───┐
                            └────────────────────────┘
```

---

### 3.3 `vector.queries` — every retrieval / chat call

```sql
CREATE TABLE vector.queries (
    id               BIGSERIAL    PRIMARY KEY,
    query_text       TEXT         NOT NULL,
    strategy         JSONB        NOT NULL DEFAULT '{}'::jsonb,
    latency_ms       JSONB        NOT NULL DEFAULT '{}'::jsonb,
    top_chunk_ids    BIGINT[]     NOT NULL DEFAULT '{}',
    chunk_scores     JSONB        NOT NULL DEFAULT '{}'::jsonb,
    answer_text      TEXT,
    citations        JSONB        NOT NULL DEFAULT '[]'::jsonb,
    token_usage      JSONB        NOT NULL DEFAULT '{}'::jsonb,
    crag_confidence  REAL,
    stage_tokens     JSONB        NOT NULL DEFAULT '{}'::jsonb,   -- migration 003
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT now()
);
```

**Indexes**
| Name                          | Type | Columns          | Source        |
|-------------------------------|------|------------------|---------------|
| `queries_pkey`                | btree| `id` PK          | migration 001 |
| `idx_queries_created`         | btree DESC | `created_at`     | migration 001 |
| `idx_queries_strategy`        | GIN  | `strategy`       | migration 001 |
| `idx_queries_stage_tokens`    | GIN  | `stage_tokens`   | migration 003 |

**Writers**
| Stage                                    | Operation | Source file |
|------------------------------------------|-----------|-------------|
| Every `/api/retrieve` and `/api/chat` (after pipeline finishes) | `INSERT` | [backend/app/pipeline/retrieve.py:_log_query](../retrieval/backend/app/pipeline/retrieve.py) |

Each row is one full retrieval invocation. Field provenance:

| Field             | Filled from                                                |
|-------------------|------------------------------------------------------------|
| `query_text`      | user's `query`                                             |
| `strategy`        | the `Strategy` Pydantic model serialised to JSON           |
| `latency_ms`      | per-stage timings (rewrite/hyde/embed/dense/sparse/fuse/rerank/mmr/crag/generate/total) |
| `top_chunk_ids`   | the final K chunk ids returned to the UI                   |
| `chunk_scores`    | `{chunk_id: rrf_score}` for the returned hits              |
| `answer_text`     | only for `/api/chat` — null for retrieve-only calls        |
| `citations`       | the parsed `[N]` markers + chunk_id + character span        |
| `token_usage`     | total {prompt, completion} across the request              |
| `crag_confidence` | only when CRAG was on                                      |
| `stage_tokens`    | per-stage `{prompt, completion}` map for the analytics chart |

**Readers**
- Analytics tab — `routers/analytics.py` aggregates over `queries`
- Benchmark harness — replays historical queries against new strategies

---

### 3.4 `vector.golden_questions` — curated evaluation set

```sql
CREATE TABLE vector.golden_questions (
    id                       BIGSERIAL   PRIMARY KEY,
    question                 TEXT        NOT NULL,
    ground_truth_chunk_ids   BIGINT[]    NOT NULL DEFAULT '{}',
    ground_truth_answer      TEXT,
    tags                     TEXT[]      NOT NULL DEFAULT '{}',
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Indexes**
| Name                  | Type | Columns | Source        |
|-----------------------|------|---------|---------------|
| `golden_questions_pkey` | btree | `id` PK | migration 001 |
| `idx_goldens_tags`    | GIN  | `tags`  | migration 001 |

**Writers**
| Stage                                                 | Operation | Source file |
|-------------------------------------------------------|-----------|-------------|
| Manual add via `POST /api/bench/questions`            | `INSERT` | [backend/app/routers/bench.py](../retrieval/backend/app/routers/bench.py) |
| Auto-seed via `POST /api/bench/seed-from-docs`        | `INSERT` (LLM-generated Q's from existing chunks) | bench.py |

**Readers**
- `POST /api/bench/run` reads the entire set (filtered by `tags` if supplied)
  and executes each question through the live retrieval pipeline to compute
  recall@k, MRR, faithfulness.

---

### 3.5 `vector.bench_runs` — benchmark history

```sql
CREATE TABLE vector.bench_runs (
    id           BIGSERIAL   PRIMARY KEY,
    label        TEXT,
    strategy     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    metrics      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    n_questions  INT         NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Indexes**
| Name                  | Type | Columns                | Source        |
|-----------------------|------|------------------------|---------------|
| `bench_runs_pkey`     | btree| `id` PK                | migration 001 |
| `idx_bench_runs_created` | btree DESC | `created_at`     | migration 001 |

**Writers**
| Stage                                  | Operation | Source file |
|----------------------------------------|-----------|-------------|
| End of `POST /api/bench/run`           | `INSERT` (final aggregate row after all questions processed) | [backend/app/routers/bench.py](../retrieval/backend/app/routers/bench.py) |

The `metrics` JSONB carries: `recall@1/5/10`, `mrr`, `faithfulness_score`,
`avg_latency_ms`, `avg_tokens`, etc. The `strategy` JSONB records the exact
toggles used so historical runs can be diffed.

**Readers**
- Benchmark tab — lists past runs for comparison
- Analytics tab — optional time-series of recall@k

---

### 3.6 `vector.documents` — uploaded source files

Originally a **view** in migration 001 (just a `SELECT DISTINCT documentName
… FROM chunk_embeddings`); migration 002 drops the view and promotes it to
a real table so we can track S3 URIs and soft-delete state.

```sql
CREATE TABLE vector.documents (
    name          TEXT PRIMARY KEY,
    s3_uri        TEXT,
    size_bytes    BIGINT,
    content_type  TEXT,
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at    TIMESTAMPTZ,
    extra         JSONB        NOT NULL DEFAULT '{}'::jsonb
);
```

**Indexes**
| Name                         | Type | Columns                                | Source        |
|------------------------------|------|----------------------------------------|---------------|
| `documents_pkey`             | btree| `name` PK                              | migration 002 |
| `idx_documents_active`       | btree partial | `(uploaded_at DESC) WHERE deleted_at IS NULL` | migration 002 |

**Writers**
| Stage                                                   | Operation | Source file |
|---------------------------------------------------------|-----------|-------------|
| Local-Vertex ingestion — backend                        | `INSERT … ON CONFLICT (name) DO UPDATE` (sets `s3_uri`, `size_bytes`, `uploaded_at`, clears `deleted_at`) | [backend/app/pipeline/local_ingest.py](../retrieval/backend/app/pipeline/local_ingest.py) |
| **Remote-proxy ingestion** — backend BEFORE proxying    | same upsert pattern, runs in [routers/ingest.py:_persist_source](../retrieval/backend/app/routers/ingest.py) | ingest.py |
| Soft-delete                                             | `UPDATE … SET deleted_at = now() RETURNING s3_uri` | [backend/app/routers/documents.py](../retrieval/backend/app/routers/documents.py) |

**Readers**
- `GET /api/documents/{name}/view` and `/download` — look up `s3_uri`, fetch from S3, stream back
- `db.py:healthcheck` — `COUNT(*) WHERE deleted_at IS NULL`

---

### 3.7 `vector.active_chunks` (view)

```sql
CREATE OR REPLACE VIEW vector.active_chunks AS
SELECT * FROM vector.chunk_embeddings WHERE deleted_at IS NULL;
```

Convenience view for ad-hoc SQL — the production retrieval pipeline still
filters `WHERE deleted_at IS NULL` directly so the optimiser can hit
`idx_chunk_active`.

---

## 4. Dynamic schema — `db.py` runtime additions

Some columns can't live in static SQL because they depend on the active
embedding model's dimension. They're added at app startup by
`backend/app/db.py:ensure_contextual_embedding_column`.

```python
async def ensure_contextual_embedding_column() -> None:
    """Add a vector column to chunk_context with the live embedding dim."""
    dim = await _discover_embedding_dim()        # SELECT atttypmod FROM pg_attribute …
    async with acquire() as conn:
        has_col = await conn.fetchval("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='vector' AND table_name='chunk_context'
                  AND column_name='context_embedding'
        """)
        if not has_col:
            await conn.execute(
                f'ALTER TABLE "{settings.pg_schema}".chunk_context '
                f'ADD COLUMN context_embedding vector({dim})'
            )
        await conn.execute(
            f'CREATE INDEX IF NOT EXISTS idx_chunk_context_embedding_hnsw '
            f'ON "{settings.pg_schema}".chunk_context '
            f'USING hnsw (context_embedding vector_cosine_ops)'
        )
```

The same approach handles the embedding column on `chunk_embeddings` if
the table doesn't already have one — necessary because the WEGA-original
`chunk_embeddings` table comes pre-sized to `gte-large-en-v1.5` (1024-D),
whereas the local Vertex path uses `text-embedding-005` (768-D).

---

## 5. Pipeline-stage → table write matrix

Quick reference: which pipeline stage writes to which table.

```
ingest (any path) ────────► chunk_embeddings        (INSERT one row per chunk)
ingest (any path) ────────► documents               (UPSERT name, s3_uri)
contextual stage ─────────► chunk_context           (UPSERT per chunk)
retrieve / chat finish ───► queries                 (INSERT one row per call)
bench_run finish ─────────► bench_runs              (INSERT aggregate)
bench seed-from-docs ─────► golden_questions        (INSERT per Q)
document soft-delete ─────► chunk_embeddings.UPDATE (deleted_at = now())
document soft-delete ─────► documents.UPDATE        (deleted_at = now())
```

And read targets (top 5):

```
dense  retrieval ─► chunk_embeddings + chunk_context
sparse retrieval ─► chunk_embeddings (content_tsv)
documents list ──► chunk_embeddings (aggregates)  +  chunk_context (coverage)
view/download ───► documents (lookup) → S3 (fetch)
analytics tab ───► queries + bench_runs
```

---

## 6. Pipeline-stage → table sequence diagram

For a single end-to-end flow — *upload a PDF, then chat against it*:

```
TIME ─────────────────────────────────────────────────────────────────────►

1. POST /api/ingest/wega    file=UploadFile
       │
       ├─►  S3.put  (rag-uploads/docs/<uid>/file.pdf)        ─► MinIO
       │
       ├─►  documents UPSERT (name, s3_uri, size_bytes …)    ─► vector.documents
       │
       ├─►  (remote proxy) httpx.stream → ingest_remote
       │       │
       │       ├─►  pypdf extract pages
       │       │
       │       ├─►  Vertex embed batches      (network)
       │       │
       │       └─►  chunk_embeddings INSERT × N               ─► vector.chunk_embeddings
       │
       └─►  SSE event: done


2. POST /api/ingest/contextual   document_name="file.pdf"
       │
       ├─►  for each chunk in chunk_embeddings WHERE LEFT JOIN chunk_context IS NULL:
       │       │
       │       ├─►  Stellar / Vertex chat → context prefix
       │       │
       │       ├─►  Stellar / Vertex embed (prefix + chunk)
       │       │
       │       └─►  chunk_context UPSERT                      ─► vector.chunk_context
       │
       └─►  SSE event: done


3. POST /api/chat               query="..."
       │
       ├─►  pipeline:
       │       embed query                  ── Vertex/Stellar
       │       dense_search(chunk_embeddings + chunk_context) ─► READ chunk_embeddings, chunk_context
       │       sparse_search(content_tsv)                     ─► READ chunk_embeddings
       │       RRF fuse
       │       rerank(top-N)                ── Vertex/Stellar
       │       MMR(top-K)
       │       generate(prompt+context)     ── Vertex/Stellar (streaming)
       │
       ├─►  SSE tokens stream while generation produces them
       │
       └─►  on done: queries INSERT (text, strategy, latency_ms,
                                     top_chunk_ids, chunk_scores,
                                     answer_text, citations,
                                     token_usage, stage_tokens)  ─► vector.queries


4. DELETE /api/documents/file.pdf
       │
       ├─►  chunk_embeddings UPDATE deleted_at = now()       ─► vector.chunk_embeddings
       │     (this also "deletes" chunk_context rows because of
       │      ON DELETE CASCADE — though the chunk row itself stays,
       │      so we only flip the chunks' deleted_at flag)
       │
       ├─►  documents UPDATE deleted_at = now() RETURNING s3_uri  ─► vector.documents
       │
       └─►  S3.delete(<key>)                                 ─► MinIO
```

---

## 7. Index design rationale

| Index                              | Why                                                          |
|------------------------------------|--------------------------------------------------------------|
| `chunk_embeddings_hnsw_idx` (HNSW) | O(log n) cosine ANN over millions of chunks — the dense retrieval workhorse |
| `idx_chunk_context_embedding_hnsw` | Same, but for the contextual embeddings (the Anthropic Sept 2024 lift) |
| `idx_chunk_content_tsv` (GIN)      | Inverted index for tsvector — sub-ms BM25-like lookups       |
| `idx_chunk_active` (btree partial) | Filters `deleted_at IS NULL` while indexing only by `documentName` — keeps the index slim and `EXPLAIN` knows to use it for soft-delete-aware queries |
| `idx_documents_active`             | Lists newest non-deleted documents without scanning soft-deleted ones |
| `idx_queries_created` DESC         | Analytics tab pulls the last N queries by `created_at` — index is reverse-sorted to match the predicate |
| `idx_queries_strategy` (GIN)       | Filtering by `strategy ? 'rerank'` etc. for ablation analytics |
| `idx_queries_stage_tokens` (GIN)   | Token-spend rollups by stage                                 |
| `idx_goldens_tags` (GIN)           | Filter `golden_questions WHERE tags && ARRAY['microsoft']`   |
| `idx_bench_runs_created` DESC      | Most-recent-first listing in the Benchmark tab               |

---

## 8. Soft-delete pattern

Used on **two** tables: `chunk_embeddings` and `documents`. The pattern:

```sql
-- WRITE
UPDATE vector.chunk_embeddings
   SET deleted_at = now()
 WHERE "documentName" = $1 AND deleted_at IS NULL;

-- READ (in every retrieval SQL)
SELECT … FROM vector.chunk_embeddings
 WHERE … AND deleted_at IS NULL;

-- RECOVER (manual op)
UPDATE vector.chunk_embeddings
   SET deleted_at = NULL
 WHERE "documentName" = $1;
```

Combined with the **partial indexes** (`WHERE deleted_at IS NULL`), this
costs nothing at query time relative to a hard `DELETE` while remaining
fully reversible. Each retrieval call also pays no extra index cost
because the partial index *is* the index it would have used anyway.

---

## 9. ER diagram

```
                    ┌────────────────────────────────────┐
                    │ vector.documents                   │
                    │ ─────────────────                  │
                    │ name (PK)                          │
                    │ s3_uri                             │
                    │ size_bytes, content_type           │
                    │ uploaded_at, deleted_at            │
                    └──────────────┬─────────────────────┘
                                   │ name = documentName
                                   │ (logical, not enforced)
                                   ▼
                    ┌────────────────────────────────────┐
                    │ vector.chunk_embeddings            │
                    │ ─────────────────────              │
                    │ id (PK, bigserial)                 │
                    │ content, "chunkUUID" (unique)      │
                    │ "documentName", "pageNumber"       │
                    │ "tokenCount", "chunkType"          │
                    │ embedding vector(D)                │
                    │ content_tsv tsvector (generated)   │
                    │ deleted_at timestamptz             │
                    └──────────────┬─────────────────────┘
                                   │ chunk_id ── FK ON DELETE CASCADE
                                   ▼
                    ┌────────────────────────────────────┐
                    │ vector.chunk_context               │
                    │ ─────────────────                  │
                    │ chunk_id (PK + FK)                 │
                    │ context_text                       │
                    │ context_embedding vector(D)        │
                    │ generated_at, generator_model      │
                    └────────────────────────────────────┘


       ┌────────────────────────────────┐    ┌─────────────────────────────────┐
       │ vector.queries                 │    │ vector.golden_questions         │
       │ ─────────────                  │    │ ─────────────────────           │
       │ id (PK, bigserial)             │    │ id (PK)                         │
       │ query_text, strategy(jsonb)    │    │ question                        │
       │ latency_ms(jsonb)              │    │ ground_truth_chunk_ids[]        │
       │ top_chunk_ids bigint[]         │◄─┐ │ ground_truth_answer             │
       │ chunk_scores(jsonb)            │  │ │ tags[]                          │
       │ answer_text, citations(jsonb)  │  │ └─────────────────────────────────┘
       │ token_usage(jsonb)             │  │           │
       │ stage_tokens(jsonb)            │  │           │ chunk_ids reference
       │ crag_confidence                │  │           │
       │ created_at                     │  │           ▼
       └────────────────────────────────┘  │   (logical link to chunk_embeddings.id)
                ▲                          │
                │ used by analytics +      │
                │ benchmark replay         │
                │                          │
       ┌────────────────────────────────┐  │
       │ vector.bench_runs              │  │
       │ ─────────────                  │  │
       │ id (PK)                        │  │
       │ label, strategy(jsonb)         │  │
       │ metrics(jsonb)                 │──┘
       │ n_questions                    │
       │ created_at                     │
       └────────────────────────────────┘
```

Foreign-key relationships actually enforced by the schema:
- `chunk_context.chunk_id` → `chunk_embeddings.id` (ON DELETE CASCADE)

Logical (non-enforced) relationships:
- `chunk_embeddings."documentName"` ↔ `documents.name`
- `queries.top_chunk_ids` ↔ `chunk_embeddings.id`
- `golden_questions.ground_truth_chunk_ids` ↔ `chunk_embeddings.id`

(These aren't FKs because the chunk ids can change across re-ingestions
and we don't want benchmark history to break.)

---

## 10. Object-store layout (out-of-band but related)

S3 / MinIO carries the original source files referenced by the `documents`
table:

```
s3://rag-uploads/
└── docs/
    ├── 0ca2bc7f/
    │   └── Form_D.pdf
    ├── 10c7dd70/
    │   └── 20-F.pdf
    └── …
```

The 8-hex prefix is `uuid4().hex[:8]` — enough collision-resistance for
this scale, and short enough to keep keys readable. The key is recorded
verbatim in `documents.s3_uri = "s3://rag-uploads/docs/<uid>/<name>"`.
The view / download endpoints (`backend/app/routers/documents.py`)
re-split that URI and call `s3_get(key)`.

---

## 11. Quick checklist for adding a new table

1. New SQL file `backend/migrations/00N_*.sql` — use `CREATE TABLE IF NOT EXISTS`.
2. Indexes in the same file with `CREATE INDEX IF NOT EXISTS`.
3. If a column depends on the runtime embedding dim, add a runtime ALTER in
   `backend/app/db.py:_post_migration_dynamic_schema()`.
4. New Pydantic model in `backend/app/schemas.py`.
5. Router in `backend/app/routers/<name>.py`, registered in `main.py`.
6. **Document the writer** in this file under §3 / §5.

---

*Generated for the RAG Studio framework — `~/retrieval/`.
Authored 2026-05-14.*
