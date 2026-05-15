"""
Local (Vertex) ingestion path for the remote service.

Used when LLM_PROVIDER=vertex. Replaces the WEGA SDK with:
  1. pypdf text extraction
  2. Paragraph-aware char chunker
  3. google-genai embeddings (text-embedding-005 by default)
  4. psycopg2 + pgvector UPSERTs into the same table the WEGA path writes to

This lets you test the remote-ingestion flow on a laptop without needing the
internal WEGA wheel.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .config import settings

logger = logging.getLogger(__name__)

ProgressCB = Optional[Callable[[Dict[str, Any]], Awaitable[None]]]


async def _emit(cb: ProgressCB, **payload: Any) -> None:
    if cb is None:
        return
    try:
        await cb(payload)
    except Exception:  # pragma: no cover
        logger.exception("progress_cb raised; continuing")


# ---------------------------------------------------------------------------
# Text extraction + chunking
# ---------------------------------------------------------------------------
def _extract_pdf_pages(file_bytes: bytes) -> List[Tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    out: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("pypdf failed on page %d: %s", i, exc)
            text = ""
        out.append((i, text))
    return out


def _extract_text(file_path: str, data: bytes | None = None) -> List[Tuple[int, str]]:
    p = Path(file_path)
    if data is None:
        data = p.read_bytes()
    if p.suffix.lower() == ".pdf":
        return _extract_pdf_pages(data)
    return [(1, data.decode("utf-8", errors="replace"))]


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    cur = ""
    for para in paras:
        if len(cur) + len(para) + 2 <= max_chars:
            cur = f"{cur}\n\n{para}" if cur else para
            continue
        if cur:
            chunks.append(cur)
        if len(para) > max_chars:
            step = max(1, max_chars - overlap)
            for i in range(0, len(para), step):
                chunks.append(para[i : i + max_chars])
            cur = ""
        else:
            cur = para
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# Vertex embeddings — keep this module-level so we initialise the client once
# ---------------------------------------------------------------------------
_genai_client = None


def _get_genai_client():
    global _genai_client
    if _genai_client is not None:
        return _genai_client
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for Vertex mode. Install with: pip install google-genai"
        ) from exc

    kwargs: dict = {"vertexai": True, "location": settings.vertex_location}
    if settings.vertex_project:
        kwargs["project"] = settings.vertex_project

    creds_path = settings.google_application_credentials
    if creds_path and os.path.isfile(creds_path):
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        kwargs["credentials"] = credentials
        logger.info("Vertex client: service-account JSON loaded from %s", creds_path)
    else:
        logger.info(
            "Vertex client: using ADC / GOOGLE_APPLICATION_CREDENTIALS env (project=%s, location=%s)",
            settings.vertex_project or "(default)",
            settings.vertex_location,
        )

    _genai_client = genai.Client(**kwargs)
    return _genai_client


def _embed_sync(texts: List[str]) -> List[List[float]]:
    client = _get_genai_client()
    resp = client.models.embed_content(
        model=settings.vertex_embedding_model, contents=texts
    )
    return [list(getattr(e, "values", []) or []) for e in resp.embeddings]


# ---------------------------------------------------------------------------
# Storage — psycopg2 sync, run inside an executor
# ---------------------------------------------------------------------------
def _connect(dbname: str | None = None):
    import psycopg2

    return psycopg2.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        dbname=dbname or settings.pg_database,
    )


def _ensure_database() -> None:
    conn = _connect(dbname="postgres")
    conn.autocommit = True
    with conn.cursor() as cur:
        try:
            cur.execute(f"SET ROLE {settings.pg_app_owner_role};")
        except Exception as role_err:  # pragma: no cover
            logger.debug("SET ROLE skipped: %s", role_err)
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (settings.pg_database,))
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{settings.pg_database}"')
    conn.close()


def _open_target_conn():
    from pgvector.psycopg2 import register_vector

    conn = _connect()
    conn.autocommit = True
    with conn.cursor() as cur:
        try:
            cur.execute(f"SET ROLE {settings.pg_app_owner_role};")
        except Exception:
            pass
        cur.execute(
            """
            SELECT nspname FROM pg_type t
            JOIN pg_namespace n ON t.typnamespace = n.oid
            WHERE t.typname = 'vector'
            LIMIT 1;
            """
        )
        row = cur.fetchone()
        ext_schema = row[0] if row else "public"
        cur.execute(f'SET search_path TO "{settings.pg_schema}", {ext_schema}, public, pg_catalog;')
        if not row:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except Exception as exc:
                raise RuntimeError(
                    "pgvector is not installed in the target DB and could not be created. "
                    f"Run 'CREATE EXTENSION vector;' as superuser first. ({exc})"
                )
    register_vector(conn)
    return conn


def _detect_table_shape(conn, index_name: str) -> str:
    """Return "wide" (WEGA-style) or "narrow" (main-backend migration) or "missing"."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s
            """,
            (settings.pg_schema, index_name),
        )
        cols = {r[0] for r in cur.fetchall()}
    if not cols:
        return "missing"
    if "documentClass" in cols:
        return "wide"
    return "narrow"


def _ensure_table_wide(conn, index_name: str, embedding_dim: int) -> None:
    """Create the WEGA-shape table when running against a fresh DB.

    SET ROLE first (when configured) so the new schema + table land under
    the right owner; then CREATE SCHEMA IF NOT EXISTS.
    """
    with conn.cursor() as cur:
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
            f"""
            CREATE INDEX IF NOT EXISTS {index_name}_hnsw_idx
            ON "{settings.pg_schema}".{index_name}
            USING hnsw (embedding vector_cosine_ops)
            """
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
async def ingest_pdf_local(
    pdf_path: str,
    *,
    document_name: str,
    document_id: str | None = None,
    overrides: Optional[Dict[str, Any]] = None,
    progress_cb: ProgressCB = None,
) -> Dict[str, Any]:
    overrides = overrides or {}
    job_id = f"ingest_{uuid.uuid4().hex[:8]}"
    batch_size = int(overrides.get("embedding_batch_size", settings.embedding_batch_size))
    max_chars = int(overrides.get("local_chunk_max_chars", settings.local_chunk_max_chars))
    overlap = int(overrides.get("local_chunk_overlap", settings.local_chunk_overlap))
    index_name = overrides.get("pg_index", settings.pg_index)

    t0 = time.time()
    loop = asyncio.get_running_loop()
    p = Path(pdf_path)

    # ── Step 1 — Extract ──────────────────────────────────────────────────
    await _emit(
        progress_cb,
        type="stage",
        stage="chunk",
        status="start",
        file=p.name,
        job_id=job_id,
        mode="vertex",
    )
    file_bytes = p.read_bytes()
    pages = await loop.run_in_executor(None, _extract_text, str(p), file_bytes)
    total_chars = sum(len(t) for _, t in pages)

    pairs: List[Tuple[int, str]] = []
    for page_num, text in pages:
        for chunk in chunk_text(text, max_chars=max_chars, overlap=overlap):
            pairs.append((page_num, chunk))

    if not pairs:
        raise RuntimeError("no chunks extracted from PDF")

    await _emit(
        progress_cb,
        type="stage",
        stage="chunk",
        status="done",
        chunks=len(pairs),
        pages=len(pages),
        total_chars=total_chars,
        elapsed=round(time.time() - t0, 2),
    )

    # ── Step 2 — Embed ────────────────────────────────────────────────────
    await _emit(progress_cb, type="stage", stage="embed", status="start", total=len(pairs))
    all_embeddings: List[List[float]] = []
    contents = [c for _, c in pairs]
    for i in range(0, len(contents), batch_size):
        batch = contents[i : i + batch_size]
        embs = await loop.run_in_executor(None, _embed_sync, batch)
        all_embeddings.extend(embs)
        await _emit(
            progress_cb,
            type="stage",
            stage="embed",
            status="progress",
            done=len(all_embeddings),
            total=len(contents),
        )
    embedding_dim = len(all_embeddings[0])
    await _emit(
        progress_cb,
        type="stage",
        stage="embed",
        status="done",
        dim=embedding_dim,
        embeddings=len(all_embeddings),
    )

    # ── Step 3 — Store ────────────────────────────────────────────────────
    await _emit(progress_cb, type="stage", stage="store", status="start")

    def _store_sync() -> Dict[str, int]:
        _ensure_database()
        conn = _open_target_conn()
        shape = _detect_table_shape(conn, index_name)
        if shape == "missing":
            _ensure_table_wide(conn, index_name, embedding_dim)
            shape = "wide"

        inserted = 0
        with conn.cursor() as cur:
            for (page_num, content), embedding in zip(pairs, all_embeddings):
                chunk_uuid = str(uuid.uuid4())
                token_count = max(1, len(content) // 4)
                if shape == "wide":
                    cur.execute(
                        f"""
                        INSERT INTO "{settings.pg_schema}".{index_name}
                            (id, "documentClass", title, "sectionHeading", content,
                             "partNumber", "totalPartNumber", "chunkUUID", "pageNumber",
                             "tokenCount", "chunkType", "chunkBoundingBox",
                             "documentName", document_id, "jobId", embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding
                        """,
                        (
                            chunk_uuid,
                            None, None, None,
                            content,
                            None, None,
                            chunk_uuid,
                            page_num,
                            token_count,
                            "paragraph",
                            None,
                            document_name,
                            document_id,
                            job_id,
                            embedding,
                        ),
                    )
                else:  # narrow — the main backend's migration shape
                    cur.execute(
                        f"""
                        INSERT INTO "{settings.pg_schema}".{index_name}
                            (content, "chunkUUID", "pageNumber", "tokenCount", "chunkType",
                             "documentName", document_id, "jobId", embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::uuid, %s, %s)
                        ON CONFLICT ("chunkUUID") DO NOTHING
                        """,
                        (
                            content,
                            chunk_uuid,
                            page_num,
                            token_count,
                            "paragraph",
                            document_name,
                            document_id,
                            job_id,
                            embedding,
                        ),
                    )
                inserted += 1
            conn.commit()
            cur.execute(f'SELECT COUNT(*) FROM "{settings.pg_schema}".{index_name}')
            total_rows = cur.fetchone()[0]
        conn.close()
        return {"inserted": inserted, "total_rows": total_rows, "shape": shape}

    store_result = await loop.run_in_executor(None, _store_sync)
    await _emit(
        progress_cb,
        type="stage",
        stage="store",
        status="done",
        inserted=store_result["inserted"],
        total_rows=store_result["total_rows"],
        table_shape=store_result.get("shape", "wide"),
    )

    return {
        "status": "success",
        "document_name": document_name,
        "file_name": p.name,
        "job_id": job_id,
        "chunks_count": len(pairs),
        "embedding_dim": embedding_dim,
        "rows_inserted": store_result["inserted"],
        "total_rows": store_result["total_rows"],
        "elapsed_seconds": round(time.time() - t0, 2),
        "mode": "vertex",
    }
