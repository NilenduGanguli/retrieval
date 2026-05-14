-- ============================================================================
-- 001_rag_extensions.sql
-- Extends the existing vector.chunk_embeddings table with everything the
-- cutting-edge RAG framework needs: contextual prefixes, full-text search
-- column, query audit log, and a golden eval set.
--
-- Idempotent — safe to re-run.
-- ============================================================================

SET search_path TO vector, public;

-- ---------------------------------------------------------------------------
-- 1. Full-text search column on the existing chunks table (BM25-like hybrid)
--    Generated column is auto-maintained on INSERT/UPDATE — no triggers.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'vector'
          AND table_name = 'chunk_embeddings'
          AND column_name = 'content_tsv'
    ) THEN
        ALTER TABLE vector.chunk_embeddings
        ADD COLUMN content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED;
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_chunk_content_tsv
    ON vector.chunk_embeddings
    USING GIN (content_tsv);

-- ---------------------------------------------------------------------------
-- 2. Contextual Retrieval (Anthropic, Sept 2024)
--    For each chunk, an LLM generates a ~50-100 token prefix that situates
--    the chunk within the broader document. We store the prefix AND an
--    embedding of (context_prefix + chunk_content) for retrieval purposes.
--    Retrieval can compare against either the original embedding or the
--    contextual embedding via the UI toggle.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vector.chunk_context (
    chunk_id           BIGINT PRIMARY KEY
                       REFERENCES vector.chunk_embeddings(id) ON DELETE CASCADE,
    context_text       TEXT NOT NULL,
    -- Embedding dim is variable across models; we store the vector via a
    -- separate "register" step from Python that knows the live dim.
    -- See backend/app/db.py:ensure_contextual_embedding_column for the ALTER.
    generated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    generator_model    TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunk_context_chunk
    ON vector.chunk_context (chunk_id);

-- ---------------------------------------------------------------------------
-- 3. Query audit / analytics log
--    Every retrieval call writes one row. Powers the Analytics tab and the
--    eval harness (we can re-score historical retrievals against the
--    golden set without re-running them live).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vector.queries (
    id              BIGSERIAL PRIMARY KEY,
    query_text      TEXT NOT NULL,
    strategy        JSONB NOT NULL DEFAULT '{}'::jsonb,
    latency_ms      JSONB NOT NULL DEFAULT '{}'::jsonb,
    top_chunk_ids   BIGINT[] NOT NULL DEFAULT '{}',
    chunk_scores    JSONB NOT NULL DEFAULT '{}'::jsonb,
    answer_text     TEXT,
    citations       JSONB NOT NULL DEFAULT '[]'::jsonb,
    token_usage     JSONB NOT NULL DEFAULT '{}'::jsonb,
    crag_confidence REAL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_queries_created
    ON vector.queries (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_queries_strategy
    ON vector.queries
    USING GIN (strategy);

-- ---------------------------------------------------------------------------
-- 4. Golden eval set — Question / expected-chunks / expected-answer tuples
--    The eval harness picks rows from here and runs them through the live
--    retrieval pipeline, computing recall@k, MRR, faithfulness.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vector.golden_questions (
    id                       BIGSERIAL PRIMARY KEY,
    question                 TEXT NOT NULL,
    ground_truth_chunk_ids   BIGINT[] NOT NULL DEFAULT '{}',
    ground_truth_answer      TEXT,
    tags                     TEXT[] NOT NULL DEFAULT '{}',
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_goldens_tags
    ON vector.golden_questions USING GIN (tags);

-- ---------------------------------------------------------------------------
-- 5. Eval runs — each batch of evals produces a run row with aggregate
--    metrics. Lets us track recall@k / MRR / faithfulness trends over time
--    (the "are we getting better?" curve we'll show the principal engineer).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vector.bench_runs (
    id          BIGSERIAL PRIMARY KEY,
    label       TEXT,
    strategy    JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics     JSONB NOT NULL DEFAULT '{}'::jsonb,
    n_questions INT  NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bench_runs_created
    ON vector.bench_runs (created_at DESC);

-- (Convenience view 'vector.documents' moved to migration 002 where it
--  becomes a real table to record S3 URIs and soft-delete state.)
