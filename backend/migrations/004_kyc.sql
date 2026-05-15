-- ============================================================================
-- 004_kyc.sql
-- KYC-specific tables — separate from the generic chunk_embeddings so the
-- KYC retrieval surface doesn't pollute (or get polluted by) generic
-- retrieval. Two tables:
--
--   kyc_documents  one row per uploaded source PDF, holds normalized
--                  owner + doc-type metadata + LLM-extracted fields.
--   kyc_chunks     one row per chunk produced by the RecursiveChar splitter,
--                  carrying its own embedding and a FK to kyc_documents.
--
-- Idempotent. Schema is rewritten at runtime by run_migrations() to honour
-- the active settings.pg_schema.
-- ============================================================================

SET search_path TO vector, public;

-- ---------------------------------------------------------------------------
-- kyc_documents
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vector.kyc_documents (
    id                       BIGSERIAL PRIMARY KEY,
    document_name            TEXT      NOT NULL UNIQUE,
    s3_uri                   TEXT,
    size_bytes               BIGINT,

    -- Owner / entity that the document is about
    owner                    TEXT,
    owner_normalized         TEXT,
    owner_first_token        TEXT,

    -- Classification output
    document_type            TEXT,
    document_category        TEXT,
    confidence_score         REAL,
    classification_signals   JSONB     NOT NULL DEFAULT '[]'::jsonb,
    source_platform          TEXT,
    report_date              TEXT,

    -- Per-doc-type extracted fields. Schema is intentionally flexible
    -- (Orbis vs Aadhaar vs Bank Statement have very different fields).
    extracted_data           JSONB     NOT NULL DEFAULT '{}'::jsonb,

    -- Raw OCR text (kept for re-extraction / debugging; can be large)
    ocr_text                 TEXT,

    created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at               TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_kyc_owner_norm
    ON vector.kyc_documents (owner_normalized)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_kyc_owner_first
    ON vector.kyc_documents (owner_first_token)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_kyc_doc_type
    ON vector.kyc_documents (document_type)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_kyc_category
    ON vector.kyc_documents (document_category)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_kyc_extracted_data
    ON vector.kyc_documents USING GIN (extracted_data);

CREATE INDEX IF NOT EXISTS idx_kyc_created
    ON vector.kyc_documents (created_at DESC)
    WHERE deleted_at IS NULL;


-- ---------------------------------------------------------------------------
-- kyc_chunks  (embedding column added by db.py at runtime once we know
-- the active embedding dim — same pattern as chunk_context).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vector.kyc_chunks (
    id                 BIGSERIAL PRIMARY KEY,
    kyc_document_id    BIGINT REFERENCES vector.kyc_documents(id) ON DELETE CASCADE,
    chunk_index        INTEGER NOT NULL,
    content            TEXT    NOT NULL,
    token_count        INTEGER,
    page_number        INTEGER,
    -- Generated FTS column for keyword-based universal search
    content_tsv        tsvector
                       GENERATED ALWAYS AS
                       (to_tsvector('english', COALESCE(content, ''))) STORED,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at         TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_kyc_chunks_doc
    ON vector.kyc_chunks (kyc_document_id)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_kyc_chunks_tsv
    ON vector.kyc_chunks USING GIN (content_tsv);
