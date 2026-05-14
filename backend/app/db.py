"""Async Postgres pool + helpers for pgvector + dynamic schema discovery."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg

from .config import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None
_embedding_dim: int | None = None


# pgvector text format: "[0.1,0.2,...]"
def vec_to_pg(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


async def _init_conn(conn: asyncpg.Connection) -> None:
    """Per-connection initialisation: jsonb codec + search_path."""
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    await conn.execute(f'SET search_path TO "{settings.pg_schema}", public')


async def init_pool() -> asyncpg.Pool:
    global _pool
    if _pool is not None:
        return _pool
    _pool = await asyncpg.create_pool(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        database=settings.pg_database,
        min_size=2,
        max_size=10,
        init=_init_conn,
        command_timeout=60,
    )
    logger.info("PG pool ready on %s:%s/%s", settings.pg_host, settings.pg_port, settings.pg_database)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def acquire() -> AsyncIterator[asyncpg.Connection]:
    pool = await init_pool()
    async with pool.acquire() as conn:
        yield conn


async def discover_embedding_dim() -> int:
    """Discover the embedding vector dimension from the existing column."""
    global _embedding_dim
    if _embedding_dim is not None:
        return _embedding_dim
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = ($1 || '.' || $2)::regclass
              AND attname = 'embedding'
              AND NOT attisdropped
            """,
            settings.pg_schema,
            settings.pg_table,
        )
        if not row:
            raise RuntimeError(
                f"Could not introspect embedding dim on {settings.fq_table}. "
                f"Has ingest.py been run yet?"
            )
        # pgvector encodes dim in atttypmod
        _embedding_dim = int(row["atttypmod"])
        logger.info("Embedding dim discovered: %d", _embedding_dim)
        return _embedding_dim


async def ensure_contextual_embedding_column() -> None:
    """Add a vector column to chunk_context with the live embedding dim."""
    dim = await discover_embedding_dim()
    async with acquire() as conn:
        exists = await conn.fetchval(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = $1
              AND table_name = 'chunk_context'
              AND column_name = 'context_embedding'
            """,
            settings.pg_schema,
        )
        if not exists:
            await conn.execute(
                f'ALTER TABLE "{settings.pg_schema}".chunk_context '
                f"ADD COLUMN context_embedding vector({dim})"
            )
            # HNSW index on the contextual embedding too
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS idx_chunk_context_embedding_hnsw '
                f'ON "{settings.pg_schema}".chunk_context '
                f"USING hnsw (context_embedding vector_cosine_ops)"
            )
            logger.info("Added context_embedding vector(%d) + HNSW index", dim)


async def ensure_schema_and_tables() -> dict:
    """Idempotent: ensure the configured schema + base tables exist.

    Called BEFORE migrations run so a fresh DB doesn't fail the
    migration ALTERs (which assume chunk_embeddings already exists).

    Returns a summary dict for logging.
    """
    summary: dict = {
        "schema_existed": None,
        "pgvector_installed": None,
        "table_created": False,
    }
    schema = settings.pg_schema
    table = settings.pg_table
    role = settings.app_owner_role

    async with acquire() as conn:
        # 1. Schema
        schema_exists = await conn.fetchval(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = $1",
            schema,
        )
        summary["schema_existed"] = bool(schema_exists)
        if not schema_exists:
            if role:
                try:
                    await conn.execute(f'SET ROLE "{role}"')
                    logger.info("SET ROLE %s", role)
                except Exception as exc:
                    logger.warning("SET ROLE %s failed: %s", role, exc)
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            logger.info("created schema %s", schema)

        # 2. pgvector extension
        ext = await conn.fetchval("SELECT 1 FROM pg_extension WHERE extname='vector'")
        summary["pgvector_installed"] = bool(ext)
        if not ext:
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                summary["pgvector_installed"] = True
                logger.info("installed pgvector extension")
            except Exception as exc:
                logger.warning(
                    "pgvector extension install failed (run 'CREATE EXTENSION vector' "
                    "as superuser): %s", exc,
                )

        # 3. chunk_embeddings table — only the narrow shape (migration 001/002
        #    add the rest of the columns via ALTER if needed)
        has_table = await conn.fetchval(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = $1 AND table_name = $2
            """,
            schema, table,
        )
        if not has_table:
            if role:
                try:
                    await conn.execute(f'SET ROLE "{role}"')
                except Exception:
                    pass
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS "{schema}"."{table}" (
                    id                BIGSERIAL PRIMARY KEY,
                    content           TEXT NOT NULL,
                    "chunkUUID"       TEXT UNIQUE,
                    "pageNumber"      INTEGER,
                    "tokenCount"      INTEGER,
                    "chunkType"       TEXT,
                    "chunkBoundingBox" JSONB,
                    "documentName"    TEXT,
                    "jobId"           TEXT
                )
            """)
            summary["table_created"] = True
            logger.info("created %s.%s", schema, table)
            # embedding column added once we know the dim (post-ingestion)
            # — but if pgvector is available, add a default 768-D column
            #   so an empty DB still answers retrieval queries.
            if summary["pgvector_installed"]:
                try:
                    await conn.execute(
                        f'ALTER TABLE "{schema}"."{table}" '
                        f"ADD COLUMN IF NOT EXISTS embedding vector(768)"
                    )
                    await conn.execute(
                        f'CREATE INDEX IF NOT EXISTS {table}_hnsw_idx '
                        f'ON "{schema}"."{table}" '
                        f"USING hnsw (embedding vector_cosine_ops)"
                    )
                except Exception as exc:
                    logger.warning("default embedding column add failed: %s", exc)

    logger.info("schema/table bootstrap: %s", summary)
    return summary


async def run_migrations(sql_path: str) -> None:
    """Apply a SQL migration file.

    Migrations are checked into the repo with `vector.` as a sentinel
    schema name. At runtime we substitute it with the configured
    `settings.pg_schema` so a single set of migration files works against
    any schema (e.g. `vector_ng12499`).
    """
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()
    schema = settings.pg_schema
    if schema != "vector":
        # Replace standalone `vector.` and `'vector'` literals. The
        # migrations only ever use these to refer to the target schema —
        # they don't reference the pgvector extension's `vector` type
        # qualifier (which would look like `vector(D)` with parens).
        import re
        sql = re.sub(r"\bvector\.(?=[a-zA-Z_])", f'"{schema}".', sql)
        # Any literal 'vector' (used as a schema-name string in
        # information_schema lookups) → the configured schema.
        sql = re.sub(r"'vector'", f"'{schema}'", sql)
        # SET search_path TO vector, ...  →  SET search_path TO "<schema>", ...
        sql = re.sub(r"(SET\s+search_path\s+TO\s+)vector\b",
                     rf'\1"{schema}"', sql, flags=re.IGNORECASE)
    async with acquire() as conn:
        await conn.execute(sql)
    logger.info("Applied migration: %s (schema=%s)", sql_path, schema)


async def healthcheck() -> dict[str, Any]:
    """Basic DB health + counts for the UI."""
    async with acquire() as conn:
        chunk_count = await conn.fetchval(
            f'SELECT COUNT(*) FROM "{settings.pg_schema}"."{settings.pg_table}" '
            f'WHERE deleted_at IS NULL'
        )
        doc_count = await conn.fetchval(
            f'SELECT COUNT(DISTINCT "documentName") '
            f'FROM "{settings.pg_schema}"."{settings.pg_table}" '
            f'WHERE deleted_at IS NULL'
        )
        ctx_count = await conn.fetchval(
            f'SELECT COUNT(*) FROM "{settings.pg_schema}".chunk_context ctx '
            f'JOIN "{settings.pg_schema}"."{settings.pg_table}" c ON c.id = ctx.chunk_id '
            f'WHERE c.deleted_at IS NULL'
        )
        dim = await discover_embedding_dim()
        return {
            "ok": True,
            "chunks": int(chunk_count or 0),
            "documents": int(doc_count or 0),
            "contextual_chunks": int(ctx_count or 0),
            "embedding_dim": dim,
            "schema": settings.pg_schema,
            "table": settings.pg_table,
        }
