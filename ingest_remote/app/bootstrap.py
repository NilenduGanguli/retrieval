"""
Startup schema/table bootstrap for the remote ingest service.

Runs once at app startup (via FastAPI lifespan). For each connection
target (the configured Postgres database) it:

  1. Checks the target schema exists.
  2. If not, SETs ROLE to the configured owner role (when set), then
     CREATEs the schema.
  3. Checks the required tables exist (chunk_embeddings + documents).
  4. If a table is missing, CREATEs it from the embedded DDL below.

Designed to be safe to run repeatedly and to never blow up the service
on startup — if Postgres is unreachable, we log and continue (the
ingest endpoint will surface the real error on first call).
"""
from __future__ import annotations

import logging
from typing import Any

import psycopg2

from .config import settings

logger = logging.getLogger(__name__)


# --- DDL templates (schema name is interpolated at runtime from settings) ----
# SCHEMA_NAME is read lazily via settings.pg_schema so any env override
# (PG_SCHEMA=...) flows through to every CREATE/INSERT.
def _schema() -> str:
    return settings.pg_schema


CHUNK_EMBEDDINGS_DDL = """
CREATE TABLE IF NOT EXISTS "{schema}".chunk_embeddings (
    id                BIGSERIAL PRIMARY KEY,
    content           TEXT NOT NULL,
    "chunkUUID"       TEXT UNIQUE,
    "pageNumber"      INTEGER,
    "tokenCount"      INTEGER,
    "chunkType"       TEXT,
    "chunkBoundingBox" JSONB,
    "documentName"    TEXT,
    "jobId"           TEXT,
    embedding         vector({embedding_dim}),
    content_tsv       tsvector
                      GENERATED ALWAYS AS
                      (to_tsvector('english', COALESCE(content, ''))) STORED,
    deleted_at        TIMESTAMPTZ
);
"""

# Ensure the embedding column exists even when the table predates this
# bootstrap (e.g. created by an old script without the column).
ENSURE_EMBEDDING_COL_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name = 'chunk_embeddings'
          AND column_name = 'embedding'
    ) THEN
        ALTER TABLE "{schema}".chunk_embeddings
        ADD COLUMN embedding vector({embedding_dim});
    END IF;
END$$;
"""

EMBEDDING_HNSW_IDX_SQL = """
CREATE INDEX IF NOT EXISTS chunk_embeddings_hnsw_idx
ON "{schema}".chunk_embeddings
USING hnsw (embedding vector_cosine_ops);
"""

DOCUMENTS_DDL = """
CREATE TABLE IF NOT EXISTS "{schema}".documents (
    name          TEXT PRIMARY KEY,
    s3_uri        TEXT,
    size_bytes    BIGINT,
    content_type  TEXT,
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at    TIMESTAMPTZ,
    extra         JSONB NOT NULL DEFAULT '{{}}'::jsonb
);
"""

INDEXES_DDL_TEMPLATES = [
    'CREATE INDEX IF NOT EXISTS idx_chunk_content_tsv '
    'ON "{schema}".chunk_embeddings USING GIN (content_tsv)',
    'CREATE INDEX IF NOT EXISTS idx_chunk_active '
    'ON "{schema}".chunk_embeddings ("documentName") WHERE deleted_at IS NULL',
    'CREATE INDEX IF NOT EXISTS idx_documents_active '
    'ON "{schema}".documents (uploaded_at DESC) WHERE deleted_at IS NULL',
]


def _connect(dbname: str | None = None):
    return psycopg2.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        dbname=dbname or settings.pg_database,
    )


def _maybe_set_role(cur: Any) -> None:
    role = settings.pg_app_owner_role
    if not role:
        return
    try:
        cur.execute(f"SET ROLE {role};")
        logger.info("SET ROLE %s", role)
    except Exception as exc:
        logger.warning("SET ROLE %s failed (continuing): %s", role, exc)


def _ensure_schema(conn) -> bool:
    """Returns True if schema exists (or was just created).

    Always SET ROLE before CREATE SCHEMA so the new schema lands under
    the configured owner role (not the connecting user). The SET ROLE
    runs even when the schema appears to exist — cheap and harmless.
    """
    schema = _schema()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
            (schema,),
        )
        if cur.fetchone():
            return True
        # Schema is missing — set role first, then create
        _maybe_set_role(cur)
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')
        logger.info("created schema %s", schema)
        return True


def _ensure_pgvector(conn) -> bool:
    """Make sure the pgvector extension is installed."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )
        if cur.fetchone():
            return True
        try:
            _maybe_set_role(cur)
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("installed pgvector extension")
            return True
        except Exception as exc:
            logger.warning("pgvector install failed (may need superuser): %s", exc)
            return False


def _has_table(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
            """,
            (_schema(), name),
        )
        return cur.fetchone() is not None


def _ensure_tables(conn) -> dict[str, bool]:
    schema = _schema()
    # Default to the embedding dim of the active provider's output. The
    # remote service knows nothing about the dim ahead of time, so we
    # use the widest commonly-deployed Vertex/Stellar default; the
    # column can be re-created at higher dims by the main backend's
    # smart-detect path if needed.
    emb_dim = 768
    state: dict[str, bool] = {}
    with conn.cursor() as cur:
        _maybe_set_role(cur)
        if not _has_table(conn, "chunk_embeddings"):
            cur.execute(CHUNK_EMBEDDINGS_DDL.format(schema=schema, embedding_dim=emb_dim))
            logger.info("created %s.chunk_embeddings (embedding vector(%d))", schema, emb_dim)
            state["chunk_embeddings"] = True
        else:
            # Table existed — make sure the embedding column is present.
            try:
                cur.execute(ENSURE_EMBEDDING_COL_SQL.format(schema=schema, embedding_dim=emb_dim))
            except Exception as exc:
                logger.warning("ensure_embedding_column failed (continuing): %s", exc)
            state["chunk_embeddings"] = False
        if not _has_table(conn, "documents"):
            cur.execute(DOCUMENTS_DDL.format(schema=schema))
            logger.info("created %s.documents", schema)
            state["documents"] = True
        else:
            state["documents"] = False
        for ddl_tpl in INDEXES_DDL_TEMPLATES:
            try:
                cur.execute(ddl_tpl.format(schema=schema))
            except Exception as exc:
                logger.warning("index DDL failed (continuing): %s", exc)
        # HNSW index on embedding (needs pgvector — best-effort)
        try:
            cur.execute(EMBEDDING_HNSW_IDX_SQL.format(schema=schema))
        except Exception as exc:
            logger.warning("HNSW index DDL failed (continuing): %s", exc)
    conn.commit()
    return state


def bootstrap_schema() -> dict[str, Any]:
    """Idempotent schema/table bootstrap. Runs at startup.

    Returns a dict describing what happened (or what went wrong) so the
    caller can log it.
    """
    summary: dict[str, Any] = {
        "ok": False,
        "schema_existed": None,
        "tables_created": {},
        "error": None,
    }
    try:
        conn = _connect()
        conn.autocommit = True
        summary["schema_existed"] = _ensure_schema(conn)
        _ensure_pgvector(conn)
        summary["tables_created"] = _ensure_tables(conn)
        conn.close()
        summary["ok"] = True
        summary["schema"] = _schema()
        logger.info(
            "schema bootstrap ok (schema=%s, tables_created=%s)",
            _schema(), summary["tables_created"],
        )
    except Exception as exc:
        summary["error"] = str(exc)
        logger.exception("schema bootstrap failed — service will start anyway")
    return summary
