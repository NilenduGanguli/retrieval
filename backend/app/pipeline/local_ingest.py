"""
Local ingestion path — used when LLM_PROVIDER=vertex (no Stellar/WEGA available).

Pipeline:
  1. Extract text per page with pypdf (no Azure Document Intelligence).
  2. Paragraph-aware char chunker.
  3. Embed batches with the active provider (Vertex on local).
  4. UPSERT into vector.chunk_embeddings.
  5. Stream progress events out so the UI bar can move in real time.

The WEGA path stays untouched — this only kicks in when we're on a laptop.
"""
from __future__ import annotations

import io
import logging
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..config import settings
from ..db import acquire, vec_to_pg, vector_type
from ..s3_store import s3_put
from ..stellar_client import get_stellar

logger = logging.getLogger(__name__)

ProgressCb = Callable[[dict[str, Any]], Awaitable[None]] | None


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def _extract_pdf_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    out: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:  # noqa: BLE001
            logger.warning("pypdf failed on page %d: %s", i, e)
            text = ""
        out.append((i, text))
    return out


def _extract_text(file_path: str, data: bytes | None = None) -> list[tuple[int, str]]:
    p = Path(file_path)
    if data is None:
        data = p.read_bytes()
    if p.suffix.lower() == ".pdf":
        return _extract_pdf_pages(data)
    # Plain-text fallback
    return [(1, data.decode("utf-8", errors="replace"))]


# ---------------------------------------------------------------------------
# Chunker — paragraph-aware char buckets
# ---------------------------------------------------------------------------
def chunk_text(text: str, max_chars: int = 1500, overlap: int = 150) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    cur = ""
    for para in paras:
        if len(cur) + len(para) + 2 <= max_chars:
            cur = f"{cur}\n\n{para}" if cur else para
        else:
            if cur:
                chunks.append(cur)
            if len(para) > max_chars:
                step = max(1, max_chars - overlap)
                for i in range(0, len(para), step):
                    chunks.append(para[i:i + max_chars])
                cur = ""
            else:
                cur = para
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
async def ingest_document_local(
    file_path: str,
    *,
    document_name: str | None = None,
    batch_size: int = 8,
    max_chars: int = 1500,
    progress_cb: ProgressCb = None,
) -> dict[str, Any]:
    async def emit(**payload: Any) -> None:
        if progress_cb:
            await progress_cb(payload)

    p = Path(file_path)
    if not p.is_file():
        await emit(type="error", message=f"file not found: {file_path}")
        return {"processed": 0, "total": 0}

    doc_name = document_name or p.name
    job_id = str(uuid.uuid4())
    file_bytes = p.read_bytes()
    size_bytes = len(file_bytes)

    # ---- S3 persistence (best-effort) ----
    s3_uri: str | None = None
    if settings.s3_enabled:
        try:
            s3_key = f"docs/{job_id[:8]}/{p.name}"
            await emit(type="log", line=f"uploading to S3 ({s3_key})")
            s3_uri = await s3_put(s3_key, file_bytes, content_type="application/pdf" if p.suffix.lower() == ".pdf" else "application/octet-stream")
            await emit(type="log", line=f"s3 stored: {s3_uri}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("S3 upload failed: %s", exc)
            await emit(type="log", line=f"S3 upload failed (continuing): {exc}")

    # ---- documents row ----
    try:
        async with acquire() as conn:
            await conn.execute(
                f'INSERT INTO "{settings.pg_schema}".documents '
                f'(name, s3_uri, size_bytes, content_type, uploaded_at) '
                f'VALUES ($1, $2, $3, $4, now()) '
                f'ON CONFLICT (name) DO UPDATE SET '
                f'  s3_uri = EXCLUDED.s3_uri, size_bytes = EXCLUDED.size_bytes, '
                f'  content_type = EXCLUDED.content_type, uploaded_at = now(), '
                f'  deleted_at = NULL',
                doc_name, s3_uri, size_bytes,
                "application/pdf" if p.suffix.lower() == ".pdf" else "application/octet-stream",
            )
    except Exception:  # noqa: BLE001
        logger.exception("documents upsert failed")

    await emit(type="log", line=f"reading {p.name}")
    pages = _extract_text(str(p), data=file_bytes)
    total_chars = sum(len(t) for _, t in pages)
    await emit(type="log", line=f"extracted {len(pages)} pages, {total_chars:,} chars")

    # Chunk each page
    pairs: list[tuple[int, str]] = []
    for page_num, text in pages:
        for chunk in chunk_text(text, max_chars=max_chars):
            pairs.append((page_num, chunk))

    total = len(pairs)
    if total == 0:
        await emit(type="error", message="no chunks extracted")
        return {"processed": 0, "total": 0}

    await emit(
        type="log",
        line=f"created {total} chunks (avg ~{total_chars // max(1, total):,} chars/chunk)",
    )

    llm = get_stellar()
    schema = settings.pg_schema
    table = settings.pg_table

    processed = 0
    failed = 0

    async with acquire() as conn:
        for batch_start in range(0, total, batch_size):
            batch = pairs[batch_start:batch_start + batch_size]
            texts = [c for _, c in batch]
            try:
                embeddings = await llm.embed(texts)
            except Exception as e:  # noqa: BLE001
                logger.exception("embed batch failed @ %d", batch_start)
                failed += len(batch)
                await emit(
                    type="error",
                    message=f"embed batch {batch_start//batch_size + 1} failed: {e}",
                )
                continue

            for (page_num, content), emb in zip(batch, embeddings):
                chunk_uuid = str(uuid.uuid4())
                token_count = max(1, len(content) // 4)
                try:
                    await conn.execute(
                        f'INSERT INTO "{schema}"."{table}" '
                        f'(content, "chunkUUID", "pageNumber", "tokenCount", "chunkType", '
                        f' "documentName", "jobId", embedding) '
                        f'VALUES ($1, $2, $3, $4, $5, $6, $7, $8::{vector_type()}) '
                        f'ON CONFLICT ("chunkUUID") DO NOTHING',
                        content,
                        chunk_uuid,
                        page_num,
                        token_count,
                        "paragraph",
                        doc_name,
                        job_id,
                        vec_to_pg(emb),
                    )
                    processed += 1
                except Exception as e:  # noqa: BLE001
                    logger.exception("insert failed")
                    failed += 1
                    await emit(type="error", message=f"insert failed: {e}")

            await emit(
                type="progress",
                message=f"embedded {processed}/{total}",
                progress=processed / total,
                processed=processed,
                failed=failed,
                total=total,
            )

    summary = {
        "processed": processed,
        "failed": failed,
        "total": total,
        "job_id": job_id,
        "document": doc_name,
    }
    await emit(type="done", message=f"done — {processed}/{total} chunks", **summary)
    return summary
