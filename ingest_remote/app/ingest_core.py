"""
Core WEGA ingestion logic — adapted from the original retrieval/ingest.py.

This is intentionally a small refactor of the reference script so that:
  * `wega_chunker` (installed from the internal pyarmoured wheel) is imported
    lazily, so the FastAPI app boots even when the wheel hasn't been
    installed in dev.
  * Stellar embeddings come from the local `app.stellar_client` copy of the
    main backend's StellarClient — no `llm_gateway` wheel dependency.
  * Progress is reported via an injected `progress_cb(payload: dict)`
    callback rather than via logging side-effects.
  * Config is taken from the service `Settings` plus per-request overrides.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .config import settings
from .stellar_client import get_stellar

logger = logging.getLogger(__name__)

ProgressCB = Optional[Callable[[Dict[str, Any]], Awaitable[None]]]


# ---------------------------------------------------------------------------
# Database helpers — mirror the original script
# ---------------------------------------------------------------------------
def _connect(cfg: dict, dbname: str | None = None):
    import psycopg2

    return psycopg2.connect(
        host=cfg["pg_host"],
        port=cfg["pg_port"],
        user=cfg["pg_user"],
        password=cfg["pg_password"],
        dbname=dbname or cfg["pg_database"],
    )


def _ensure_database(cfg: dict) -> None:
    conn = _connect(cfg, dbname="postgres")
    conn.autocommit = True
    with conn.cursor() as cur:
        if settings.pg_app_owner_role:
            try:
                cur.execute(f"SET ROLE {settings.pg_app_owner_role};")
            except Exception as role_err:  # pragma: no cover - role-only env
                logger.warning("could not SET ROLE %s: %s", settings.pg_app_owner_role, role_err)
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (cfg["pg_database"],))
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{cfg["pg_database"]}"')
            logger.info("created database %s", cfg["pg_database"])
    conn.close()


def _get_connection(cfg: dict):
    from pgvector.psycopg2 import register_vector

    conn = _connect(cfg)
    conn.autocommit = True
    with conn.cursor() as cur:
        try:
            cur.execute("SET ROLE citi_pg_app_owner;")
        except Exception as role_err:  # pragma: no cover
            logger.warning("could not SET ROLE citi_pg_app_owner: %s", role_err)
        cur.execute(
            """
            SELECT nspname
            FROM pg_type t
            JOIN pg_namespace n ON t.typnamespace = n.oid
            WHERE t.typname = 'vector'
            LIMIT 1;
            """
        )
        row = cur.fetchone()
        ext_schema = row[0] if row else "public"
        cur.execute(f'SET search_path TO "{settings.pg_schema}", {ext_schema}, public, pg_catalog;')
    register_vector(conn)
    return conn


def _create_table(conn, index_name: str, embedding_dim: int) -> None:
    with conn.cursor() as cur:
        # SET ROLE before any CREATE so the new objects land under the
        # configured owner role (when one is set). Then CREATE SCHEMA so
        # the table+index DDL below can rely on the schema existing.
        if settings.pg_app_owner_role:
            try:
                cur.execute(f"SET ROLE {settings.pg_app_owner_role};")
            except Exception:
                pass
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{settings.pg_schema}";')
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{settings.pg_schema}".{index_name} (
                id                TEXT PRIMARY KEY,
                "documentClass"   TEXT,
                title             TEXT,
                "sectionHeading"  TEXT,
                content           TEXT,
                "partNumber"      INTEGER,
                "totalPartNumber" INTEGER,
                "chunkUUID"       TEXT,
                "pageNumber"      INTEGER,
                "tokenCount"      INTEGER,
                "chunkType"       TEXT,
                "chunkBoundingBox" DOUBLE PRECISION[],
                "documentName"    TEXT,
                "jobId"           TEXT,
                embedding         vector({embedding_dim}),
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            SELECT atttypmod FROM pg_attribute
            JOIN pg_class ON attrelid = pg_class.oid
            JOIN pg_namespace ON relnamespace = pg_namespace.oid
            WHERE nspname = %s AND relname = %s AND attname = 'embedding';
            """,
            (settings.pg_schema, index_name),
        )
        dim_row = cur.fetchone()
        if dim_row and dim_row[0] != embedding_dim:
            logger.warning(
                "embedding dim mismatch (table=%s, model=%s) — recreating",
                dim_row[0],
                embedding_dim,
            )
            cur.execute(f'DROP TABLE IF EXISTS "{settings.pg_schema}".{index_name} CASCADE;')
            _create_table(conn, index_name, embedding_dim)
            return
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {index_name}_hnsw_idx
            ON "{settings.pg_schema}".{index_name}
            USING hnsw (embedding vector_cosine_ops)
            """
        )
    conn.commit()


# ---------------------------------------------------------------------------
# WEGA + Stellar wrappers — imported lazily so the wheel is optional in dev
# ---------------------------------------------------------------------------
def _run_chunker(cfg: dict) -> dict:
    from wega_chunker import WegaChunker  # provided by the internal wheel

    chunker = WegaChunker(
        azure_di_endpoint=cfg["azure_di_endpoint"],
        azure_di_key=cfg["azure_di_key"],
    )
    chunker_config = {
        "JobID": cfg["job_id"],
        "data": {
            "chunkTokenSize": cfg["chunk_token_size"],
            "extractImages": cfg["extract_images"],
            "piiExtraction": {"detectPII": cfg["detect_pii"]},
        },
        "documentType": cfg["document_type"],
        "fileName": cfg["file_name"],
        "inputFolder": cfg["input_folder"],
    }
    return chunker.chunk(chunker_config)


def _embed(texts: List[str]) -> List[List[float]]:
    """Embed a batch with Stellar (gte-large-en-v1.5).

    Delegates to the local stellar_client copy so this service has no
    dependency on the backend package or an external `llm_gateway` wheel.
    Called from inside `run_in_executor`, so we use the sync variant.
    """
    return get_stellar().embed_sync(list(texts))


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
async def emit(cb: ProgressCB, **payload: Any) -> None:
    if cb is None:
        return
    try:
        await cb(payload)
    except Exception:  # pragma: no cover
        logger.exception("progress_cb raised; continuing")


async def ingest_pdf(
    pdf_path: str,
    *,
    document_name: str,
    overrides: Optional[Dict[str, Any]] = None,
    progress_cb: ProgressCB = None,
) -> Dict[str, Any]:
    """Run the full WEGA → embed → pgvector pipeline.

    Yields events through `progress_cb` so the caller can stream them as SSE.
    Returns a summary dict identical in shape to the original CLI script.
    """
    overrides = overrides or {}
    cfg: Dict[str, Any] = {
        "azure_di_endpoint": settings.azure_di_endpoint,
        "azure_di_key": settings.azure_di_key,
        "chunk_token_size": settings.chunk_token_size,
        "document_type": settings.document_type,
        "detect_pii": settings.detect_pii,
        "extract_images": settings.extract_images,
        "pg_host": settings.pg_host,
        "pg_port": settings.pg_port,
        "pg_user": settings.pg_user,
        "pg_password": settings.pg_password,
        "pg_database": settings.pg_database,
        "pg_index": settings.pg_index,
        "embedding_batch_size": settings.embedding_batch_size,
        "job_id": f"ingest_{uuid.uuid4().hex[:8]}",
        # Per-request fields:
        "file_name": Path(pdf_path).name,
        "input_folder": str(Path(pdf_path).parent),
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    t0 = time.time()
    loop = asyncio.get_running_loop()

    # ── Step 1 — Chunk ────────────────────────────────────────────────────
    await emit(
        progress_cb,
        type="stage",
        stage="chunk",
        status="start",
        file=cfg["file_name"],
        job_id=cfg["job_id"],
    )
    result = await loop.run_in_executor(None, _run_chunker, cfg)

    chunker_result = result.get("chunkerResult", [{}])
    if not chunker_result:
        raise RuntimeError("Chunker returned empty result")

    file_result = chunker_result[0]
    chunks = file_result.get(cfg["file_name"], {}).get("chunkData", [])
    if not chunks:
        raise RuntimeError(f"No chunks produced for '{cfg['file_name']}'")
    await emit(
        progress_cb,
        type="stage",
        stage="chunk",
        status="done",
        chunks=len(chunks),
        elapsed=round(time.time() - t0, 2),
    )

    # ── Step 2 — Embed ────────────────────────────────────────────────────
    await emit(progress_cb, type="stage", stage="embed", status="start", total=len(chunks))
    contents = [c["content"] for c in chunks]
    batch_size = cfg["embedding_batch_size"]
    all_embeddings: List[List[float]] = []
    for i in range(0, len(contents), batch_size):
        batch = contents[i : i + batch_size]
        embs = await loop.run_in_executor(None, _embed, batch)
        all_embeddings.extend(embs)
        await emit(
            progress_cb,
            type="stage",
            stage="embed",
            status="progress",
            done=len(all_embeddings),
            total=len(contents),
        )
    embedding_dim = len(all_embeddings[0])
    await emit(
        progress_cb,
        type="stage",
        stage="embed",
        status="done",
        dim=embedding_dim,
        embeddings=len(all_embeddings),
    )

    # ── Step 3 — Store ────────────────────────────────────────────────────
    await emit(progress_cb, type="stage", stage="store", status="start")

    def _store_sync() -> Dict[str, int]:
        _ensure_database(cfg)
        conn = _get_connection(cfg)
        index_name = cfg["pg_index"]
        _create_table(conn, index_name, embedding_dim)

        inserted = 0
        with conn.cursor() as cur:
            for chunk, embedding in zip(chunks, all_embeddings):
                bbox = chunk.get("chunkBoundingBox")
                if bbox is not None:
                    bbox = [float(v) for v in bbox]
                cur.execute(
                    f"""
                    INSERT INTO "{settings.pg_schema}".{index_name}
                        (id, "documentClass", title, "sectionHeading", content,
                         "partNumber", "totalPartNumber", "chunkUUID", "pageNumber",
                         "tokenCount", "chunkType", "chunkBoundingBox",
                         "documentName", "jobId", embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        chunk["id"],
                        chunk.get("documentClass"),
                        chunk.get("title"),
                        chunk.get("sectionHeading"),
                        chunk["content"],
                        chunk.get("partNumber"),
                        chunk.get("totalPartNumber"),
                        chunk.get("chunkUUID"),
                        chunk.get("pageNumber"),
                        chunk.get("tokenCount"),
                        chunk.get("chunkType"),
                        bbox,
                        document_name,
                        cfg["job_id"],
                        embedding,
                    ),
                )
                inserted += 1
            conn.commit()
            cur.execute(f'SELECT COUNT(*) FROM "{settings.pg_schema}".{index_name}')
            total_rows = cur.fetchone()[0]
        conn.close()
        return {"inserted": inserted, "total_rows": total_rows}

    store_result = await loop.run_in_executor(None, _store_sync)
    await emit(
        progress_cb,
        type="stage",
        stage="store",
        status="done",
        inserted=store_result["inserted"],
        total_rows=store_result["total_rows"],
    )

    elapsed = round(time.time() - t0, 2)
    return {
        "status": "success",
        "document_name": document_name,
        "file_name": cfg["file_name"],
        "job_id": cfg["job_id"],
        "chunks_count": len(chunks),
        "embedding_dim": embedding_dim,
        "rows_inserted": store_result["inserted"],
        "total_rows": store_result["total_rows"],
        "elapsed_seconds": elapsed,
    }
