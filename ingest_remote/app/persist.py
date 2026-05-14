"""
Shared "persist source PDF + documents row" helper used by both the WEGA
and Vertex ingestion paths in the remote service.

Best-effort: any failure here is logged and the ingestion continues.
The chunks are the load-bearing part; S3 persistence is a nice-to-have
that gates the View/Download buttons in the UI.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import psycopg2

from .config import settings
from .s3_store import s3_put

logger = logging.getLogger(__name__)


def _connect():
    return psycopg2.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        dbname=settings.pg_database,
    )


async def persist_source(pdf_path: str, document_name: str) -> str | None:
    """Upload the source PDF to S3 + UPSERT the {settings.pg_schema}.documents row.

    Returns the s3:// URI on success, None when S3 is disabled or a
    non-fatal failure happened.
    """
    if not settings.s3_enabled:
        return None
    try:
        data = Path(pdf_path).read_bytes()
        s3_key = f"docs/{uuid.uuid4().hex[:8]}/{document_name}"
        content_type = (
            "application/pdf"
            if Path(document_name).suffix.lower() == ".pdf"
            else "application/octet-stream"
        )
        s3_uri = await s3_put(s3_key, data, content_type=content_type)

        conn = _connect()
        conn.autocommit = True
        with conn.cursor() as cur:
            if settings.pg_app_owner_role:
                try:
                    cur.execute(f"SET ROLE {settings.pg_app_owner_role};")
                except Exception:
                    pass
            cur.execute(
                f"""
                INSERT INTO "{settings.pg_schema}".documents
                  (name, s3_uri, size_bytes, content_type, uploaded_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (name) DO UPDATE SET
                    s3_uri       = EXCLUDED.s3_uri,
                    size_bytes   = EXCLUDED.size_bytes,
                    content_type = EXCLUDED.content_type,
                    uploaded_at  = now(),
                    deleted_at   = NULL
                """,
                (document_name, s3_uri, len(data), content_type),
            )
        conn.close()
        return s3_uri
    except Exception:
        logger.exception("persist_source failed for %s (continuing)", document_name)
        return None
