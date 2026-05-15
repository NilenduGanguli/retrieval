"""Documents listing + soft-delete endpoints."""
from __future__ import annotations

import logging
import mimetypes
import re
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

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _is_uuid(s: str) -> bool:
    """True when `s` looks like a UUID — distinguishes document_id from name."""
    return bool(_UUID_RE.match(s))


@router.get("", response_model=list[DocumentSummary])
async def list_documents() -> list[DocumentSummary]:
    """One row per uploaded document.

    Post-migration 005 the join key is `document_id` (UUID), so two uploads
    with the same filename surface as distinct rows. Chunks ingested before
    005 have a NULL document_id — they collapse under a single row keyed by
    name (COALESCE) to keep the legacy listing intact.
    """
    schema = settings.pg_schema
    table = settings.pg_table
    sql = f"""
        WITH base AS (
            SELECT
                COALESCE(c.document_id::text, '__legacy__:' || c."documentName") AS group_key,
                c.document_id,
                c."documentName"          AS document_name,
                COUNT(*)::int             AS chunk_count,
                SUM(c."tokenCount")::int  AS total_tokens,
                MIN(c."pageNumber")       AS first_page,
                MAX(c."pageNumber")       AS last_page,
                MAX(c."jobId")            AS latest_job_id,
                ARRAY_AGG(c.id)           AS chunk_ids
            FROM "{schema}"."{table}" c
            WHERE c.deleted_at IS NULL
            GROUP BY c.document_id, c."documentName"
        )
        SELECT
            base.document_id,
            base.document_name,
            d.sha256,
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
        LEFT JOIN "{schema}".documents d ON d.id = base.document_id
        ORDER BY base.document_name, base.document_id NULLS LAST
    """
    async with acquire() as conn:
        rows = await conn.fetch(sql)
    return [
        DocumentSummary(
            document_id=str(r["document_id"]) if r["document_id"] is not None else None,
            document_name=r["document_name"],
            sha256=r["sha256"],
            chunk_count=r["chunk_count"],
            total_tokens=r["total_tokens"],
            first_page=r["first_page"],
            last_page=r["last_page"],
            latest_job_id=r["latest_job_id"],
            contextual_coverage=float(r["contextual_coverage"] or 0.0),
        )
        for r in rows
    ]


@router.delete("/{ident}")
async def soft_delete_document(ident: str) -> dict:
    """Soft-delete a document and its chunks (sets deleted_at = now()).

    `ident` is either a UUID `document_id` (preferred — unambiguous when two
    uploads share a name) or a legacy `document_name` (for chunks ingested
    before migration 005, where document_id is NULL). Falls back automatically.
    """
    ident = unquote(ident)
    schema = settings.pg_schema
    table = settings.pg_table

    by_id = _is_uuid(ident)
    async with acquire() as conn:
        if by_id:
            chunk_where = 'document_id = $1::uuid'
            doc_where = 'id = $1::uuid'
        else:
            chunk_where = '"documentName" = $1'
            doc_where = 'name = $1'

        await conn.execute(
            f'UPDATE "{schema}"."{table}" SET deleted_at = now() '
            f'WHERE {chunk_where} AND deleted_at IS NULL',
            ident,
        )
        count = await conn.fetchval(
            f'SELECT COUNT(*) FROM "{schema}"."{table}" '
            f'WHERE {chunk_where} AND deleted_at IS NOT NULL',
            ident,
        )
        doc_row = await conn.fetchrow(
            f'UPDATE "{schema}".documents SET deleted_at = now() '
            f'WHERE {doc_where} AND deleted_at IS NULL '
            f'RETURNING id, name, s3_uri',
            ident,
        )

    s3_uri = doc_row["s3_uri"] if doc_row else None
    # Best-effort S3 cleanup
    if s3_uri and s3_uri.startswith("s3://") and settings.s3_enabled:
        try:
            key = s3_uri.split("/", 3)[-1]
            await s3_delete(key)
        except Exception:
            logger.exception("s3 delete failed for %s", s3_uri)

    return {
        "document_id": str(doc_row["id"]) if doc_row else (ident if by_id else None),
        "document_name": doc_row["name"] if doc_row else (None if by_id else ident),
        "soft_deleted_chunks": int(count or 0),
        "s3_uri_removed": s3_uri,
    }


async def _fetch_document_blob(ident: str) -> tuple[bytes, str, str]:
    """Pull the source bytes from S3. Returns (bytes, content_type, display_name).

    `ident` can be a UUID (`id`) or a legacy `name`.
    """
    by_id = _is_uuid(ident)
    where = "id = $1::uuid" if by_id else "name = $1"
    async with acquire() as conn:
        row = await conn.fetchrow(
            f'SELECT name, s3_uri, content_type FROM "{settings.pg_schema}".documents '
            f'WHERE {where} AND deleted_at IS NULL',
            ident,
        )
    if not row or not row["s3_uri"]:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No source persisted for '{ident}'. "
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
    display_name = row["name"] or ident
    ctype = row["content_type"] or mimetypes.guess_type(display_name)[0] or "application/octet-stream"
    return data, ctype, display_name


@router.get("/{ident}/view", response_class=Response)
async def view_document(ident: str) -> Response:
    """Stream the source PDF inline. `ident` is a document_id (UUID) or legacy name."""
    ident = unquote(ident)
    data, ctype, display_name = await _fetch_document_blob(ident)
    return Response(
        content=data,
        media_type=ctype,
        headers={
            "Content-Disposition": f'inline; filename="{quote(Path(display_name).name)}"',
            "Cache-Control": "private, max-age=300",
        },
    )


@router.get("/{ident}/download", response_class=Response)
async def download_document(ident: str) -> Response:
    """Stream the source PDF as attachment. `ident` is a document_id (UUID) or legacy name."""
    ident = unquote(ident)
    data, ctype, display_name = await _fetch_document_blob(ident)
    return Response(
        content=data,
        media_type=ctype,
        headers={
            "Content-Disposition": f'attachment; filename="{quote(Path(display_name).name)}"',
        },
    )


@router.get("/{ident}/chunks")
async def list_chunks(ident: str, limit: int = 200) -> dict:
    """Page through chunks of a document. `ident` is a document_id (UUID) or legacy name."""
    schema = settings.pg_schema
    table = settings.pg_table
    by_id = _is_uuid(ident)
    where = 'c.document_id = $1::uuid' if by_id else 'c."documentName" = $1'
    sql = f"""
        SELECT
            c.id,
            c.content,
            c."pageNumber"   AS page_number,
            c."tokenCount"   AS token_count,
            c."chunkType"    AS chunk_type,
            c."documentName" AS document_name,
            c.document_id,
            ctx.context_text
        FROM "{schema}"."{table}" c
        LEFT JOIN "{schema}".chunk_context ctx ON ctx.chunk_id = c.id
        WHERE {where} AND c.deleted_at IS NULL
        ORDER BY c."pageNumber" NULLS LAST, c.id
        LIMIT $2
    """
    async with acquire() as conn:
        rows = await conn.fetch(sql, ident, limit)
    display_name = rows[0]["document_name"] if rows else (None if by_id else ident)
    display_id = (
        str(rows[0]["document_id"]) if rows and rows[0]["document_id"] is not None
        else (ident if by_id else None)
    )
    return {
        "document_id": display_id,
        "document_name": display_name,
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
