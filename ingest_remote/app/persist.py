"""
Shared "persist source PDF + documents row" helper used by both the WEGA
and Vertex ingestion paths in the remote service.

Best-effort: any failure here is logged and the ingestion continues.
The chunks are the load-bearing part; S3 persistence is a nice-to-have
that gates the View/Download buttons in the UI.

Post-migration 005 the documents table is keyed on a UUID `id` and carries
a `sha256` checksum. Each call mints a fresh id so two uploads of the same
filename produce two distinct rows. The id is returned so the caller can
stamp it onto every chunk_embeddings row that belongs to this upload.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import NamedTuple

import psycopg2

from .config import settings
from .s3_store import s3_put

logger = logging.getLogger(__name__)


class PersistedSource(NamedTuple):
    s3_uri: str | None
    document_id: str
    sha256: str


def _connect():
    return psycopg2.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        dbname=settings.pg_database,
    )


async def persist_source(pdf_path: str, document_name: str) -> PersistedSource:
    """Upload the source PDF to S3 + INSERT a {settings.pg_schema}.documents row.

    Each upload gets a brand-new UUID — no UPSERT on name, because name is
    not unique. Always returns a `document_id` so the caller can write it
    onto every chunk for this ingest.
    """
    data = Path(pdf_path).read_bytes()
    document_id = str(uuid.uuid4())
    sha256 = hashlib.sha256(data).hexdigest()

    s3_uri: str | None = None
    if settings.s3_enabled:
        try:
            s3_key = f"docs/{uuid.uuid4().hex[:8]}/{document_name}"
            content_type = (
                "application/pdf"
                if Path(document_name).suffix.lower() == ".pdf"
                else "application/octet-stream"
            )
            s3_uri = await s3_put(s3_key, data, content_type=content_type)
        except Exception:
            logger.exception("persist_source S3 upload failed for %s (continuing)", document_name)

    content_type = (
        "application/pdf"
        if Path(document_name).suffix.lower() == ".pdf"
        else "application/octet-stream"
    )

    try:
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
                  (id, name, s3_uri, size_bytes, content_type, sha256, uploaded_at)
                VALUES (%s::uuid, %s, %s, %s, %s, %s, now())
                """,
                (document_id, document_name, s3_uri, len(data), content_type, sha256),
            )
        conn.close()
    except Exception:
        logger.exception("persist_source INSERT failed for %s (continuing)", document_name)

    return PersistedSource(s3_uri=s3_uri, document_id=document_id, sha256=sha256)
