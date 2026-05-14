-- ============================================================================
-- 003_stage_tokens.sql
-- Add per-stage token breakdown column to vector.queries so the analytics
-- tab can chart token spend by pipeline stage across history.
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'vector'
          AND table_name = 'queries'
          AND column_name = 'stage_tokens'
    ) THEN
        ALTER TABLE vector.queries
        ADD COLUMN stage_tokens JSONB NOT NULL DEFAULT '{}'::jsonb;
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_queries_stage_tokens
    ON vector.queries USING GIN (stage_tokens);
