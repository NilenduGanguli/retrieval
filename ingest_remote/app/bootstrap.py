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
# {embedding_dim} is likewise interpolated from settings.embedding_dim — the
# fleet-wide vector width shared with the retrieval backend (see _ensure_tables).
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

# Fallback DDL — same chunk_embeddings shape minus the `embedding`
# column, used when pgvector isn't available on this DB. The column
# can be added later with:
#   ALTER TABLE "<schema>".chunk_embeddings ADD COLUMN embedding vector(D);
CHUNK_EMBEDDINGS_DDL_NO_VECTOR = """
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
    content_tsv       tsvector
                      GENERATED ALWAYS AS
                      (to_tsvector('english', COALESCE(content, ''))) STORED,
    deleted_at        TIMESTAMPTZ
);
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


def _ensure_pgvector(conn) -> str | None:
    """Make sure the pgvector extension is installed AND on the search_path.

    Returns the schema name where pgvector's `vector` type lives (or
    None if pgvector still isn't usable after a CREATE attempt).

    Why this matters: production DBs often have pgvector installed in a
    schema that isn't on the connecting user's default search_path, so
    a bare `vector(D)` reference in DDL fails with
    `type "vector" does not exist`. We discover the schema, prepend it
    to search_path for this session, and return it.
    """
    with conn.cursor() as cur:
        # 1. Try to find the vector type
        cur.execute(
            """
            SELECT nspname FROM pg_type t
            JOIN pg_namespace n ON t.typnamespace = n.oid
            WHERE t.typname = 'vector'
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            # Not installed — try to install (may fail without superuser)
            try:
                _maybe_set_role(cur)
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("installed pgvector extension")
            except Exception as exc:
                logger.warning("pgvector install failed (may need superuser): %s", exc)
                return None
            # Re-query
            cur.execute(
                """
                SELECT nspname FROM pg_type t
                JOIN pg_namespace n ON t.typnamespace = n.oid
                WHERE t.typname = 'vector'
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if not row:
                logger.warning("pgvector still not visible after CREATE EXTENSION")
                return None
        ext_schema = row[0]
        # 2. Make the type visible without a schema qualifier so the DDL
        #    in this session can use `vector(D)` directly.
        target_schema = _schema()
        cur.execute(
            f'SET search_path TO "{target_schema}", "{ext_schema}", public, pg_catalog;'
        )
        logger.info(
            "pgvector found in schema %s; search_path = %s, %s, public, pg_catalog",
            ext_schema, target_schema, ext_schema,
        )
        return ext_schema


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


def _ensure_tables(conn, pgvector_schema: str | None) -> dict[str, bool]:
    """Create the required tables. If pgvector is not usable, fall back
    to creating chunk_embeddings WITHOUT the embedding column — the
    table still serves retrieval lookups via FTS / contextual paths."""
    schema = _schema()
    # The vector dimension is part of the CROSS-SERVICE CONTRACT: this
    # service, the retrieval backend, and document-enrichment-services all
    # read/write the same chunk_embeddings.embedding column, so the column
    # must be sized for the fleet-wide embedding model (gte-large-en-v1.5 →
    # 1024). Never hard-code it here — a mismatch either fails the INSERT
    # outright or, worse, silently splits the index across two models.
    # See docs/EMBEDDING_MODEL.md.
    emb_dim = settings.embedding_dim
    state: dict[str, bool] = {}
    with conn.cursor() as cur:
        _maybe_set_role(cur)
        if not _has_table(conn, "chunk_embeddings"):
            if pgvector_schema:
                cur.execute(CHUNK_EMBEDDINGS_DDL.format(schema=schema, embedding_dim=emb_dim))
                logger.info("created %s.chunk_embeddings (embedding vector(%d))", schema, emb_dim)
            else:
                # pgvector not available — create the narrow table without
                # the embedding column; admins can add it later.
                cur.execute(CHUNK_EMBEDDINGS_DDL_NO_VECTOR.format(schema=schema))
                logger.info(
                    "created %s.chunk_embeddings (no embedding column — "
                    "pgvector unavailable; install + ALTER ADD COLUMN to enable dense retrieval)",
                    schema,
                )
            state["chunk_embeddings"] = True
        else:
            state["chunk_embeddings"] = False
            # Existing table — only try to backfill the embedding column
            # when pgvector is usable for *this* connection.
            if pgvector_schema:
                try:
                    cur.execute(ENSURE_EMBEDDING_COL_SQL.format(schema=schema, embedding_dim=emb_dim))
                except Exception as exc:
                    logger.warning("ensure_embedding_column failed (continuing): %s", exc)
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
        # HNSW index on embedding — only when pgvector is usable
        if pgvector_schema:
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
        pgv_schema = _ensure_pgvector(conn)
        summary["pgvector_schema"] = pgv_schema
        summary["tables_created"] = _ensure_tables(conn, pgv_schema)
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
