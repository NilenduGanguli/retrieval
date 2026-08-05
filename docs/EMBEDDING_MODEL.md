# Embedding model — `gte-large-en-v1.5` (1024-D), fleet-wide

**Status:** authoritative as of this change. Applies to every service and
**every environment — production, local, and test.**

## What and why

| | |
|---|---|
| **Model** | `gte-large-en-v1.5` (Alibaba-NLP) |
| **Dimension** | **1024** |
| **Previously (local/Vertex only)** | `text-embedding-005`, **768**-D |

`retrieval` and `document-enrichment-services` read and write the **same**
`chunk_embeddings.embedding` pgvector column. One index means one embedding
model: vectors from two different models are not comparable, so a mixed column
returns quietly wrong neighbours rather than an error.

Until now only the production (Stellar) path used gte. The local/Vertex
fallback still used `text-embedding-005`, so **local and prod produced
incompatible vectors for the same column.** That mismatch is what this change
removes.

The authoritative settings now live in both
`backend/app/config.py` and `ingest_remote/app/config.py`:

```python
embedding_model: str = "gte-large-en-v1.5"   # fleet-wide
embedding_dim:   int = 1024                  # cross-service contract
```

`vertex_embedding_model` is **kept but marked LEGACY / fallback-only.** It is
768-D and is not fleet-compatible; it exists so anyone mid-migration can still
read an old index. Do not point it at a gte-sized column.

## The dimension change is a migration, not a flag flip

pgvector types a column as `vector(N)`. **Vectors of different dimensions
cannot coexist in one column** — there is no mixed mode, and no implicit
conversion. Switching 768 → 1024 therefore requires **re-embedding every
existing row**; the stored 768-D vectors are dead weight, not convertible.

### Production / anything with data worth keeping

Re-embed first, cut over second.

```sql
-- 1. Stage the new vectors alongside the old ones.
ALTER TABLE "<schema>".chunk_embeddings
    ADD COLUMN embedding_1024 vector(1024);

-- 2. Re-embed every row with gte-large-en-v1.5 and write into
--    embedding_1024 (batch job, outside SQL). Verify:
--       SELECT count(*) FROM "<schema>".chunk_embeddings
--       WHERE deleted_at IS NULL AND embedding_1024 IS NULL;   -- must be 0

-- 3. Swap. (Do this in one transaction, with writes paused.)
BEGIN;
DROP INDEX IF EXISTS "<schema>".chunk_embeddings_hnsw_idx;
ALTER TABLE "<schema>".chunk_embeddings DROP COLUMN embedding;
ALTER TABLE "<schema>".chunk_embeddings
    RENAME COLUMN embedding_1024 TO embedding;
COMMIT;

-- 4. Rebuild the HNSW index — it is built over the column's vectors and
--    does NOT survive the type/column change. Retrieval is a sequential
--    scan until this finishes.
CREATE INDEX chunk_embeddings_hnsw_idx
    ON "<schema>".chunk_embeddings
    USING hnsw (embedding vector_cosine_ops);
```

An in-place `ALTER COLUMN` is only valid **after** the values are already
1024-D, and still drops the index:

```sql
ALTER TABLE "<schema>".chunk_embeddings
    ALTER COLUMN embedding TYPE vector(1024);
-- then recreate chunk_embeddings_hnsw_idx as above
```

The same applies to `chunk_context.context_embedding`, which is sized from the
live `chunk_embeddings` dimension at bootstrap.

### Dev / disposable data

Just drop and re-ingest:

```sql
DROP TABLE IF EXISTS "<schema>".chunk_context;
DROP TABLE IF EXISTS "<schema>".chunk_embeddings;
```

Bootstrap recreates `chunk_embeddings` at `vector(1024)` from
`settings.embedding_dim` on next startup.

## Serving gte locally

Two options. Prefer the first — it keeps the local path byte-identical in shape
to production (an HTTP embedding endpoint).

### 1. HTTP endpoint (recommended)

Run gte behind an OpenAI-compatible server or HuggingFace **TEI**, then set:

```
EMBEDDING_BASE_URL=http://localhost:8080/v1   # or the TEI root
EMBEDDING_API_KEY=...                         # blank if unauthenticated
EMBEDDING_API_STYLE=openai                    # or: tei
```

- `openai` → `POST {base_url}/embeddings`, body `{"model": ..., "input": [...]}`
- `tei` → `POST {base_url}/embed`, body `{"inputs": [...]}`

**TEI on Apple Silicon:** the official TEI images publish **`linux/amd64`
only**. On an M-series Mac they run under emulation (`--platform linux/amd64`)
— functional, but slow. Budget accordingly, or use option 2 locally.

### 2. `sentence-transformers` in-process

Only these pins are verified to work with this model:

```
sentence-transformers==3.0.1
transformers==4.44.2
einops
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "Alibaba-NLP/gte-large-en-v1.5",
    trust_remote_code=True,   # gte ships custom modelling code
    device="cpu",             # NOT "mps"
)
vectors = model.encode(["hello"], normalize_embeddings=True)  # 1024-D
```

Two hard-won constraints, both non-obvious:

- **`device="cpu"`.** Apple **MPS crashes** on this model.
- **`transformers==4.44.2`.** Newer transformers **breaks gte's custom RoPE
  code**, which is loaded via `trust_remote_code=True`.

## Scope of this change

This is a **configuration and documentation change only.**

- It **re-embeds nothing by itself.** No data is read, written, or migrated.
- It changes what *new* deployments create: bootstrap now sizes
  `chunk_embeddings.embedding` from `settings.embedding_dim` (1024) instead of
  a hard-coded 768.
- Against a **stale 768-D column it fails loudly** — an INSERT of a 1024-D
  vector into `vector(768)` is rejected by pgvector — rather than silently
  corrupting the index with two incompatible models. That is the intended
  behaviour: a failed write is recoverable, a poisoned index is not.

Run the migration above before pointing a gte-configured service at an existing
768-D database.
