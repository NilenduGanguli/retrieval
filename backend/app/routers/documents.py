"""Documents listing + soft-delete endpoints."""
from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ..config import settings
from ..db import acquire
from ..s3_store import s3_delete, s3_get
from ..schemas import DocumentSummary

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("", response_model=list[DocumentSummary])
async def list_documents() -> list[DocumentSummary]:
    schema = settings.pg_schema
    table = settings.pg_table
    sql = f"""
        WITH base AS (
            SELECT
                "documentName"    AS document_name,
                COUNT(*)::int     AS chunk_count,
                SUM("tokenCount")::int AS total_tokens,
                MIN("pageNumber") AS first_page,
                MAX("pageNumber") AS last_page,
                MAX("jobId")      AS latest_job_id,
                ARRAY_AGG(id)     AS chunk_ids
            FROM "{schema}"."{table}"
            WHERE deleted_at IS NULL
            GROUP BY "documentName"
        )
        SELECT
            base.document_name,
            base.chunk_count,
            base.total_tokens,
            base.first_page,
            base.last_page,
            base.latest_job_id,
            COALESCE(
                (SELECT COUNT(*)::float / NULLIF(base.chunk_count, 0)
                 FROM "{schema}".chunk_context ctx
                 WHERE ctx.chunk_id = ANY(base.chunk_ids)),
                0
            ) AS contextual_coverage
        FROM base
        ORDER BY base.document_name
    """
    async with acquire() as conn:
        rows = await conn.fetch(sql)
    return [
        DocumentSummary(
            document_name=r["document_name"],
            chunk_count=r["chunk_count"],
            total_tokens=r["total_tokens"],
            first_page=r["first_page"],
            last_page=r["last_page"],
            latest_job_id=r["latest_job_id"],
            contextual_coverage=float(r["contextual_coverage"] or 0.0),
        )
        for r in rows
    ]


@router.delete("/{document_name}")
async def soft_delete_document(document_name: str) -> dict:
    """
    Soft-delete every chunk for a document (sets deleted_at = now()) and
    marks the documents row deleted. Also removes the S3 object if present.
    Retrieval queries already filter `deleted_at IS NULL`.
    """
    name = unquote(document_name)
    schema = settings.pg_schema
    table = settings.pg_table

    async with acquire() as conn:
        n_chunks = await conn.fetchval(
            f'UPDATE "{schema}"."{table}" SET deleted_at = now() '
            f'WHERE "documentName" = $1 AND deleted_at IS NULL '
            f'RETURNING id',
            name,
        )
        # fetchval returns the first row's id (or None); we want the count
        count = await conn.fetchval(
            f'SELECT COUNT(*) FROM "{schema}"."{table}" '
            f'WHERE "documentName" = $1 AND deleted_at IS NOT NULL',
            name,
        )
        s3_uri = await conn.fetchval(
            f'UPDATE "{schema}".documents SET deleted_at = now() '
            f'WHERE name = $1 AND deleted_at IS NULL RETURNING s3_uri',
            name,
        )

    # Best-effort S3 cleanup
    if s3_uri and s3_uri.startswith("s3://") and settings.s3_enabled:
        try:
            key = s3_uri.split("/", 3)[-1]
            await s3_delete(key)
        except Exception:
            logger.exception("s3 delete failed for %s", s3_uri)

    return {
        "document_name": name,
        "soft_deleted_chunks": int(count or 0),
        "s3_uri_removed": s3_uri,
    }


async def _fetch_document_blob(name: str) -> tuple[bytes, str]:
    """Pull the source bytes from S3. Returns (bytes, content_type)."""
    async with acquire() as conn:
        row = await conn.fetchrow(
            f'SELECT s3_uri, content_type FROM "{settings.pg_schema}".documents '
            f'WHERE name = $1 AND deleted_at IS NULL',
            name,
        )
    if not row or not row["s3_uri"]:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No source persisted for '{name}'. "
                "This document was ingested before S3 persistence was enabled, "
                "or S3 is disabled."
            ),
        )
    s3_uri: str = row["s3_uri"]
    if not s3_uri.startswith("s3://"):
        raise HTTPException(status_code=500, detail=f"unexpected s3_uri: {s3_uri}")
    key = s3_uri.split("/", 3)[-1]
    try:
        data = await s3_get(key)
    except Exception as exc:
        logger.exception("s3_get failed for %s", key)
        raise HTTPException(status_code=502, detail=f"S3 fetch failed: {exc}")
    ctype = row["content_type"] or mimetypes.guess_type(name)[0] or "application/octet-stream"
    return data, ctype


@router.get("/{document_name}/view", response_class=Response)
async def view_document(document_name: str) -> Response:
    """Stream the source PDF inline (Content-Disposition: inline)."""
    name = unquote(document_name)
    data, ctype = await _fetch_document_blob(name)
    return Response(
        content=data,
        media_type=ctype,
        headers={
            "Content-Disposition": f'inline; filename="{quote(Path(name).name)}"',
            "Cache-Control": "private, max-age=300",
        },
    )


@router.get("/{document_name}/download", response_class=Response)
async def download_document(document_name: str) -> Response:
    """Stream the source PDF as attachment (Content-Disposition: attachment)."""
    name = unquote(document_name)
    data, ctype = await _fetch_document_blob(name)
    return Response(
        content=data,
        media_type=ctype,
        headers={
            "Content-Disposition": f'attachment; filename="{quote(Path(name).name)}"',
        },
    )


@router.get("/{document_name}/chunks")
async def list_chunks(document_name: str, limit: int = 200) -> dict:
    """Page through chunks of a document. Used by the doc explorer panel."""
    schema = settings.pg_schema
    table = settings.pg_table
    sql = f"""
        SELECT
            c.id,
            c.content,
            c."pageNumber"   AS page_number,
            c."tokenCount"   AS token_count,
            c."chunkType"    AS chunk_type,
            ctx.context_text
        FROM "{schema}"."{table}" c
        LEFT JOIN "{schema}".chunk_context ctx ON ctx.chunk_id = c.id
        WHERE c."documentName" = $1 AND c.deleted_at IS NULL
        ORDER BY c."pageNumber" NULLS LAST, c.id
        LIMIT $2
    """
    async with acquire() as conn:
        rows = await conn.fetch(sql, document_name, limit)
    return {
        "document_name": document_name,
        "chunks": [
            {
                "id": int(r["id"]),
                "content": r["content"] or "",
                "page_number": r["page_number"],
                "token_count": r["token_count"],
                "chunk_type": r["chunk_type"],
                "context_text": r["context_text"],
            }
            for r in rows
        ],
    }
