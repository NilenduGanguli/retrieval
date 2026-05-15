-- ============================================================================
-- 005_document_ids.sql
--
-- Replace the documents PK with a UUID `id`, add a SHA-256 column, and
-- carry the id down into chunk_embeddings via a new `document_id` column.
--
-- Motivation: name is no longer guaranteed unique — two users can upload
-- different files that happen to share a filename, and re-uploading the
-- same name (with different content) must not collide with the existing
-- row. The new UUID is the authoritative join key; chunk_embeddings.
-- "documentName" stays around as a human-readable label.
--
-- Idempotent — safe to re-run on a database where this has already been
-- applied. SHA-256 backfill for legacy rows is left NULL; ingestion only
-- populates it for new uploads.
-- ============================================================================

SET search_path TO vector, public;

-- pgcrypto gives us gen_random_uuid() without extra setup.
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ---------------------------------------------------------------------------
-- documents: add id + sha256, drop PK on (name), re-add PK on (id)
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'vector' AND table_name = 'documents'
          AND column_name = 'id'
    ) THEN
        ALTER TABLE vector.documents
            ADD COLUMN id UUID DEFAULT gen_random_uuid();
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'vector' AND table_name = 'documents'
          AND column_name = 'sha256'
    ) THEN
        ALTER TABLE vector.documents ADD COLUMN sha256 TEXT;
    END IF;
END$$;

-- Backfill ids for rows that pre-date this column.
UPDATE vector.documents SET id = gen_random_uuid() WHERE id IS NULL;

ALTER TABLE vector.documents ALTER COLUMN id SET NOT NULL;
ALTER TABLE vector.documents ALTER COLUMN id SET DEFAULT gen_random_uuid();

-- Drop whichever PK currently exists (was on name) so we can put it on id.
DO $$
DECLARE
    pk_name TEXT;
BEGIN
    SELECT c.conname INTO pk_name
    FROM pg_constraint c
    JOIN pg_class t       ON t.oid = c.conrelid
    JOIN pg_namespace n   ON n.oid = t.relnamespace
    WHERE n.nspname = 'vector'
      AND t.relname = 'documents'
      AND c.contype = 'p';
    IF pk_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE vector.documents DROP CONSTRAINT %I', pk_name);
    END IF;
END$$;

-- New PK on id. Use a fixed name so re-runs don't double-add it.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint c
        JOIN pg_class t     ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = 'vector'
          AND t.relname = 'documents'
          AND c.conname = 'documents_pkey'
    ) THEN
        ALTER TABLE vector.documents
            ADD CONSTRAINT documents_pkey PRIMARY KEY (id);
    END IF;
END$$;

-- name is no longer unique — add a non-unique index for lookups by name.
CREATE INDEX IF NOT EXISTS idx_documents_name
    ON vector.documents (name)
    WHERE deleted_at IS NULL;

-- sha256 lookup index (non-unique on purpose; the user controls dedup policy).
CREATE INDEX IF NOT EXISTS idx_documents_sha256
    ON vector.documents (sha256)
    WHERE deleted_at IS NULL;

-- ---------------------------------------------------------------------------
-- chunk_embeddings: add document_id and backfill from name
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'vector' AND table_name = 'chunk_embeddings'
          AND column_name = 'document_id'
    ) THEN
        ALTER TABLE vector.chunk_embeddings ADD COLUMN document_id UUID;
    END IF;
END$$;

-- Backfill: every existing chunk row whose documentName matches a single
-- documents row gets that document's id. Pre-existing data has unique
-- names by construction, so this join is deterministic.
UPDATE vector.chunk_embeddings c
   SET document_id = d.id
  FROM vector.documents d
 WHERE c."documentName" = d.name
   AND c.document_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_chunk_document_id
    ON vector.chunk_embeddings (document_id)
    WHERE deleted_at IS NULL;
