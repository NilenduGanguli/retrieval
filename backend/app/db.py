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
# Cached schema where pgvector lives (e.g. "public" or "extensions").
# Discovered once at bootstrap and reused so every pool connection's
# search_path includes it — otherwise `vector(D)` references in DDL
# fail with "type 'vector' does not exist".
_pgvector_schema: str | None = None


# pgvector text format: "[0.1,0.2,...]"
def vec_to_pg(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


async def _discover_pgvector_schema(conn: asyncpg.Connection) -> str | None:
    """Return the schema where pgvector's `vector` type is registered."""
    row = await conn.fetchrow(
        "SELECT nspname FROM pg_type t "
        "JOIN pg_namespace n ON t.typnamespace = n.oid "
        "WHERE t.typname = 'vector' LIMIT 1"
    )
    return row["nspname"] if row else None


async def _init_conn(conn: asyncpg.Connection) -> None:
    """Per-connection initialisation: jsonb codec + search_path.

    The search_path always includes the configured target schema, the
    pgvector schema (discovered once at bootstrap), and public. Without
    the pgvector schema, any `vector(D)` reference in DDL fails on
    connections that were acquired fresh from the pool.
    """
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    # If we haven't discovered the pgvector schema yet (very first pool
    # init before ensure_schema_and_tables runs), look it up now so the
    # bootstrap conn already has it.
    global _pgvector_schema
    if _pgvector_schema is None:
        try:
            _pgvector_schema = await _discover_pgvector_schema(conn)
        except Exception:
            _pgvector_schema = None

    parts: list[str] = [f'"{settings.pg_schema}"']
    if _pgvector_schema and _pgvector_schema not in ("public", settings.pg_schema):
        parts.append(f'"{_pgvector_schema}"')
    parts.append("public")
    await conn.execute("SET search_path TO " + ", ".join(parts))


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
    """Add a vector column to chunk_context with the live embedding dim.

    Runs inside SET ROLE <app_owner_role> when configured so the ALTER
    works even when chunk_context is owned by the app role rather than
    the connecting user. The chunk_context table itself only exists
    after migration 001 has run, so if the table is missing we just
    return — the migration step is the right place to surface that.
    """
    schema = settings.pg_schema
    role = settings.app_owner_role
    async with acquire() as conn:
        # Make sure chunk_context exists before doing anything else.
        has_table = await conn.fetchval(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = $1 AND table_name = 'chunk_context'
            """,
            schema,
        )
        if not has_table:
            logger.warning(
                "skip ensure_contextual_embedding_column: %s.chunk_context not present yet",
                schema,
            )
            return
        dim = await discover_embedding_dim()
        exists = await conn.fetchval(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = $1
              AND table_name = 'chunk_context'
              AND column_name = 'context_embedding'
            """,
            schema,
        )
        if exists:
            return
        # The connection acquired from the pool may have been created
        # before _pgvector_schema was discovered. Force the search_path
        # locally so `vector(D)` resolves.
        global _pgvector_schema
        if _pgvector_schema is None:
            try:
                _pgvector_schema = await _discover_pgvector_schema(conn)
            except Exception:
                _pgvector_schema = None
        if _pgvector_schema and _pgvector_schema not in ("public", schema):
            try:
                await conn.execute(
                    f'SET search_path TO "{schema}", "{_pgvector_schema}", public, pg_catalog'
                )
            except Exception as exc:
                logger.warning("failed to extend search_path with pgvector schema: %s", exc)
        role_set = False
        if role:
            try:
                await conn.execute(f'SET ROLE "{role}"')
                role_set = True
            except Exception as exc:
                logger.warning("SET ROLE %s failed for context column: %s", role, exc)
        try:
            await conn.execute(
                f'ALTER TABLE "{schema}".chunk_context '
                f"ADD COLUMN context_embedding vector({dim})"
            )
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS idx_chunk_context_embedding_hnsw '
                f'ON "{schema}".chunk_context '
                f"USING hnsw (context_embedding vector_cosine_ops)"
            )
            logger.info("Added context_embedding vector(%d) + HNSW index", dim)
        finally:
            if role_set:
                try:
                    await conn.execute("RESET ROLE")
                except Exception:
                    pass


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

        # 2. pgvector extension — and discover which schema it lives in
        #    so we can put it on the connection's search_path. Otherwise
        #    `vector(D)` references in DDL fail with "type vector does
        #    not exist" when pgvector isn't in public.
        ext_row = await conn.fetchrow(
            """
            SELECT nspname FROM pg_type t
            JOIN pg_namespace n ON t.typnamespace = n.oid
            WHERE t.typname = 'vector'
            LIMIT 1
            """
        )
        if not ext_row:
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                logger.info("installed pgvector extension")
                ext_row = await conn.fetchrow(
                    "SELECT nspname FROM pg_type t "
                    "JOIN pg_namespace n ON t.typnamespace = n.oid "
                    "WHERE t.typname = 'vector' LIMIT 1"
                )
            except Exception as exc:
                logger.warning(
                    "pgvector extension install failed (run 'CREATE EXTENSION vector' "
                    "as superuser): %s", exc,
                )
        summary["pgvector_installed"] = bool(ext_row)
        if ext_row:
            ext_schema = ext_row["nspname"]
            summary["pgvector_schema"] = ext_schema
            # Cache module-globally so every pool connection's _init_conn
            # also gets this schema on its search_path. Otherwise a fresh
            # conn acquired later (e.g. during ensure_contextual_embedding_column)
            # won't be able to resolve `vector(D)`.
            global _pgvector_schema
            _pgvector_schema = ext_schema
            # Prepend the pgvector schema to search_path for THIS connection
            # so the rest of the bootstrap's DDL resolves immediately.
            try:
                await conn.execute(
                    f'SET search_path TO "{schema}", "{ext_schema}", public, pg_catalog'
                )
                logger.info("pgvector discovered in schema %s; search_path set", ext_schema)
            except Exception as exc:
                logger.warning("failed to add pgvector schema to search_path: %s", exc)

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

        # 4. Defensive column adds — make sure soft-delete + FTS columns
        #    exist on the chunk_embeddings table regardless of how / by
        #    whom it was created (the original ingest.py creates the
        #    table without these). Healthcheck + retrieval queries all
        #    filter on `deleted_at IS NULL`, so a missing column wedges
        #    the whole app.
        added_cols: list[str] = []
        if role:
            try:
                await conn.execute(f'SET ROLE "{role}"')
            except Exception:
                pass
        # deleted_at — soft-delete marker
        try:
            has_deleted = await conn.fetchval(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = $1 AND table_name = $2 AND column_name = 'deleted_at'
                """,
                schema, table,
            )
            if not has_deleted:
                await conn.execute(
                    f'ALTER TABLE "{schema}"."{table}" ADD COLUMN deleted_at TIMESTAMPTZ'
                )
                await conn.execute(
                    f'CREATE INDEX IF NOT EXISTS idx_chunk_active '
                    f'ON "{schema}"."{table}" ("documentName") WHERE deleted_at IS NULL'
                )
                added_cols.append("deleted_at")
                logger.info("added missing column deleted_at on %s.%s", schema, table)
        except Exception as exc:
            logger.warning("ensure deleted_at failed: %s", exc)
        # content_tsv — generated tsvector for FTS / sparse retrieval
        try:
            has_tsv = await conn.fetchval(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = $1 AND table_name = $2 AND column_name = 'content_tsv'
                """,
                schema, table,
            )
            if not has_tsv:
                await conn.execute(
                    f'ALTER TABLE "{schema}"."{table}" '
                    f"ADD COLUMN content_tsv tsvector "
                    f"GENERATED ALWAYS AS (to_tsvector('english', COALESCE(content, ''))) STORED"
                )
                await conn.execute(
                    f'CREATE INDEX IF NOT EXISTS idx_chunk_content_tsv '
                    f'ON "{schema}"."{table}" USING GIN (content_tsv)'
                )
                added_cols.append("content_tsv")
                logger.info("added missing column content_tsv on %s.%s", schema, table)
        except Exception as exc:
            logger.warning("ensure content_tsv failed: %s", exc)
        if added_cols:
            summary["columns_added"] = added_cols

    logger.info("schema/table bootstrap: %s", summary)
    return summary


async def run_migrations(sql_path: str) -> None:
    """Apply a SQL migration file.

    Migrations are checked into the repo with `vector.` as a sentinel
    schema name. At runtime we substitute it with the configured
    `settings.pg_schema` so a single set of migration files works against
    any schema (e.g. `vector_ng12499`).

    The migration runs with `SET ROLE <app_owner_role>` first so that
    `ALTER TABLE` statements can modify tables that aren't owned by the
    connecting user (very common in shared/managed Postgres where the
    table was created by a service account). We RESET ROLE at the end
    so the connection is clean when it returns to the pool.
    """
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()
    schema = settings.pg_schema
    role = settings.app_owner_role
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
        # SET ROLE up-front — required when the existing table is owned
        # by the app_owner_role rather than the connecting user, otherwise
        # ALTER TABLE inside the migration fails with InsufficientPrivilege.
        role_set = False
        if role:
            try:
                await conn.execute(f'SET ROLE "{role}"')
                role_set = True
                logger.info("SET ROLE %s for migration %s", role, sql_path)
            except Exception as exc:
                logger.warning(
                    "SET ROLE %s failed before migration %s (continuing as connecting user): %s",
                    role, sql_path, exc,
                )
        try:
            await conn.execute(sql)
        finally:
            if role_set:
                try:
                    await conn.execute("RESET ROLE")
                except Exception:
                    pass
    logger.info("Applied migration: %s (schema=%s, role=%s)", sql_path, schema, role or "(none)")


async def healthcheck() -> dict[str, Any]:
    """Basic DB health + counts for the UI.

    Each query is wrapped in a try so a missing optional column
    (e.g. `deleted_at` or the `chunk_context` table not yet created
    on a freshly-deployed DB) doesn't fail the whole endpoint —
    we just degrade gracefully and report what we can.
    """
    schema = settings.pg_schema
    table = settings.pg_table

    # Detect once whether deleted_at exists so we can short-circuit the
    # WHERE clause when it doesn't (avoids per-query UndefinedColumnError).
    async with acquire() as conn:
        has_deleted = await conn.fetchval(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2 AND column_name = 'deleted_at'
            """,
            schema, table,
        )
        deleted_filter = "WHERE deleted_at IS NULL" if has_deleted else ""

        async def _safe_fetchval(sql: str) -> int:
            try:
                v = await conn.fetchval(sql)
                return int(v or 0)
            except Exception as exc:
                logger.warning("healthcheck sub-query failed (%s): %s", sql.split()[0], exc)
                return 0

        chunk_count = await _safe_fetchval(
            f'SELECT COUNT(*) FROM "{schema}"."{table}" {deleted_filter}'
        )
        doc_count = await _safe_fetchval(
            f'SELECT COUNT(DISTINCT "documentName") FROM "{schema}"."{table}" {deleted_filter}'
        )
        # chunk_context may not exist yet on a partially-bootstrapped DB
        ctx_filter = "AND c.deleted_at IS NULL" if has_deleted else ""
        ctx_count = await _safe_fetchval(
            f'SELECT COUNT(*) FROM "{schema}".chunk_context ctx '
            f'JOIN "{schema}"."{table}" c ON c.id = ctx.chunk_id '
            f'WHERE 1=1 {ctx_filter}'
        )
        try:
            dim = await discover_embedding_dim()
        except Exception:
            dim = 0
        return {
            "ok": True,
            "chunks": chunk_count,
            "documents": doc_count,
            "contextual_chunks": ctx_count,
            "embedding_dim": dim,
            "schema": schema,
            "table": table,
            "soft_delete_enabled": bool(has_deleted),
        }
