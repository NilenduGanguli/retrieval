"""
Remote ingestion FastAPI service.

Lives on a separate VM that has access to:
  * The internal WEGA chunker wheel (pyarmoured)
  * The Stellar LLM gateway for embeddings
  * The pgvector Postgres database

Endpoints
---------
  GET  /health
  POST /ingest   (multipart upload — streams SSE progress back to caller)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from sse_starlette.sse import EventSourceResponse

from .bootstrap import bootstrap_schema
from .config import settings
from .ingest_core import ingest_pdf
from .local_ingest import ingest_pdf_local
from .s3_store import s3_ensure_bucket

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 1. Schema bootstrap — make sure vector.* tables exist
    summary = bootstrap_schema()
    logger.info("startup schema bootstrap: %s", summary)
    # 2. S3 bucket bootstrap — only if S3 is enabled on this service
    if settings.s3_enabled:
        try:
            ok = await s3_ensure_bucket()
            logger.info("startup s3 bucket bootstrap: ok=%s bucket=%s",
                        ok, settings.s3_bucket)
        except Exception:
            logger.exception("s3 bucket bootstrap failed (continuing)")
    yield


app = FastAPI(title="RAG Remote Ingestion Service", lifespan=lifespan)

UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _check_secret(value: str | None) -> None:
    """Reject calls when a shared secret is configured but missing/wrong."""
    if not settings.shared_secret:
        return
    if value != settings.shared_secret:
        raise HTTPException(status_code=401, detail="invalid X-Ingest-Secret")


@app.get("/health")
async def health() -> dict:
    provider = (settings.llm_provider or "").lower()
    emb_model = (
        settings.internal_vertex_embedding_model
        if provider in ("vertex_internal", "internal_vertex")
        else settings.vertex_embedding_model
    )
    return {
        "status": "ok",
        "service": "ingest_remote",
        "llm_provider": settings.llm_provider,
        "embedding_model": emb_model,
        "azure_di_configured": bool(settings.azure_di_key),
        "vertex_project": settings.vertex_project or None,
        "vertex_location": settings.vertex_location,
        "pg_database": settings.pg_database,
        "pg_schema": settings.pg_schema,
        "pg_index": settings.pg_index,
        "s3_enabled": settings.s3_enabled,
        "s3_bucket": settings.s3_bucket if settings.s3_enabled else None,
    }


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    document_name: str | None = Form(default=None),
    overrides_json: str | None = Form(default=None),
    x_ingest_secret: str | None = Header(default=None, alias="X-Ingest-Secret"),
) -> EventSourceResponse:
    _check_secret(x_ingest_secret)

    overrides: dict = {}
    if overrides_json:
        try:
            overrides = json.loads(overrides_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"bad overrides_json: {exc}")

    # Save the upload to disk — the WEGA SDK reads from `inputFolder`.
    saved = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    saved.write_bytes(await file.read())
    doc_name = document_name or file.filename or saved.name
    logger.info("received upload %s (%d bytes)", saved, saved.stat().st_size)

    queue: asyncio.Queue = asyncio.Queue()
    done_sentinel: object = object()

    async def progress_cb(payload: dict) -> None:
        await queue.put(payload)

    # Provider dispatch:
    #   vertex          → pypdf chunker + Vertex (service-account) — local dev
    #   vertex_internal → WEGA chunker + InternalVertex embeddings — production
    #   wega / stellar   → WEGA chunker + Stellar embeddings — legacy
    provider = (settings.llm_provider or "vertex_internal").lower()
    if provider == "vertex":
        runner = ingest_pdf_local
    else:
        # Both vertex_internal and wega/stellar go through ingest_pdf;
        # ingest_core._embed picks the right embedding backend.
        runner = ingest_pdf

    async def _worker() -> None:
        try:
            # Persist source PDF to S3 + UPSERT documents row (when enabled
            # on THIS service; skipped silently otherwise).
            from .persist import persist_source
            s3_uri = await persist_source(str(saved), doc_name)
            if s3_uri:
                await progress_cb({"type": "stage", "stage": "s3", "status": "done", "s3_uri": s3_uri})

            summary = await runner(
                str(saved),
                document_name=doc_name,
                overrides=overrides,
                progress_cb=progress_cb,
            )
            if s3_uri:
                summary = {**summary, "s3_uri": s3_uri}
            await queue.put({"type": "done", **summary})
        except Exception as exc:
            logger.exception("remote ingest failed (%s mode)", provider)
            await queue.put({"type": "error", "message": str(exc), "mode": provider})
        finally:
            await queue.put(done_sentinel)
            try:
                saved.unlink(missing_ok=True)
            except Exception:
                pass

    async def _events():
        task = asyncio.create_task(_worker())
        try:
            yield {
                "event": "start",
                "data": json.dumps(
                    {
                        "file": str(saved),
                        "filename": file.filename,
                        "document_name": doc_name,
                        "mode": provider,
                    }
                ),
            }
            while True:
                item = await queue.get()
                if item is done_sentinel:
                    break
                yield {"event": item.get("type", "info"), "data": json.dumps(item)}
        finally:
            if not task.done():
                task.cancel()

    return EventSourceResponse(_events())


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "ingest_remote.app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
