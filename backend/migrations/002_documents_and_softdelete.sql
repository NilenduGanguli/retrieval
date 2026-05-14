-- ============================================================================
-- 002_documents_and_softdelete.sql
-- Adds a documents table (source-of-truth metadata + S3 URI) and a soft-delete
-- column on chunk_embeddings. Idempotent.
-- ============================================================================

SET search_path TO vector, public;

-- Migration 001 historically created vector.documents as a VIEW. CREATE
-- TABLE IF NOT EXISTS doesn't replace views, so we drop it only when it
-- IS still a view (idempotent against re-runs after the table exists).
DO $$
DECLARE
    rel_kind char;
BEGIN
    SELECT relkind INTO rel_kind
    FROM pg_class
    WHERE relnamespace = 'vector'::regnamespace
      AND relname = 'documents';
    IF rel_kind = 'v' THEN
        EXECUTE 'DROP VIEW vector.documents';
    END IF;
END$$;

-- Documents table — one row per uploaded source file
CREATE TABLE IF NOT EXISTS vector.documents (
    name          TEXT PRIMARY KEY,
    s3_uri        TEXT,
    size_bytes    BIGINT,
    content_type  TEXT,
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at    TIMESTAMPTZ,
    extra         JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_documents_active
    ON vector.documents (uploaded_at DESC)
    WHERE deleted_at IS NULL;

-- Soft-delete on chunks
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'vector'
          AND table_name = 'chunk_embeddings'
          AND column_name = 'deleted_at'
    ) THEN
        ALTER TABLE vector.chunk_embeddings
        ADD COLUMN deleted_at TIMESTAMPTZ;
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_chunk_active
    ON vector.chunk_embeddings ("documentName")
    WHERE deleted_at IS NULL;

-- View that hides deleted chunks (handy for ad-hoc queries; the pipeline
-- still filters in SQL).
CREATE OR REPLACE VIEW vector.active_chunks AS
SELECT * FROM vector.chunk_embeddings WHERE deleted_at IS NULL;
