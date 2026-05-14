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
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from sse_starlette.sse import EventSourceResponse

from .config import settings
from .ingest_core import ingest_pdf
from .local_ingest import ingest_pdf_local

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Remote Ingestion Service")

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
    return {
        "status": "ok",
        "service": "ingest_remote",
        "llm_provider": settings.llm_provider,
        "azure_di_configured": bool(settings.azure_di_key),
        "vertex_project": settings.vertex_project or None,
        "vertex_location": settings.vertex_location,
        "vertex_embedding_model": settings.vertex_embedding_model,
        "pg_database": settings.pg_database,
        "pg_index": settings.pg_index,
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

    provider = (settings.llm_provider or "wega").lower()
    if provider == "vertex":
        runner = ingest_pdf_local
    else:
        runner = ingest_pdf

    async def _worker() -> None:
        try:
            summary = await runner(
                str(saved),
                document_name=doc_name,
                overrides=overrides,
                progress_cb=progress_cb,
            )
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
