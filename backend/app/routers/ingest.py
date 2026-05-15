"""
Ingestion router.

  * GET  /api/ingest/health        DB + Stellar status
  * POST /api/ingest/contextual    generate Anthropic-style contextual prefixes
                                   over already-ingested chunks (SSE progress)
  * POST /api/ingest/wega          run WEGA ingestion against an uploaded PDF
                                   (SSE log stream)

We avoid spawning the existing ingest.py as a separate OS process — instead
we import its `ingest()` callable and run it in a thread so the existing
sync implementation doesn't block our asyncio event loop. Progress is
emitted via a queue.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import sys
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, File, UploadFile
from sse_starlette.sse import EventSourceResponse

from ..config import settings
from ..db import acquire, healthcheck
from ..pipeline.contextual import generate_context_batch
from ..pipeline.local_ingest import ingest_document_local
from ..s3_store import s3_put
from ..usage import UsageScope, snapshot

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ingest", tags=["ingest"])


@router.get("/health")
async def ingest_health() -> dict:
    return await healthcheck()


# ---------------------------------------------------------------------------
# Contextual-prefix generation (Anthropic recipe) over existing chunks
# ---------------------------------------------------------------------------
@router.post("/contextual")
async def contextual_stream(
    document_name: str | None = None,
    limit: int = 100,
    concurrency: int = 4,
) -> EventSourceResponse:
    queue: asyncio.Queue = asyncio.Queue()
    done_sentinel: object = object()

    async def progress_cb(progress: float, info: dict) -> None:
        await queue.put({"type": "context", "progress": progress, "payload": info, "tokens": snapshot()})

    async def _worker() -> None:
        usage = UsageScope()
        usage.__enter__()
        try:
            stats = await generate_context_batch(
                document_name=document_name,
                limit=limit,
                concurrency=concurrency,
                progress_cb=progress_cb,
            )
            await queue.put({"type": "done", "stats": stats, "tokens": usage.snapshot()})
        except Exception as exc:
            logger.exception("contextual_stream worker failed")
            await queue.put({"type": "error", "message": str(exc), "tokens": usage.snapshot()})
        finally:
            usage.__exit__(None, None, None)
            await queue.put(done_sentinel)

    async def _events():
        task = asyncio.create_task(_worker())
        try:
            yield {
                "event": "start",
                "data": json.dumps(
                    {"document_name": document_name, "limit": limit, "concurrency": concurrency}
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


# ---------------------------------------------------------------------------
# WEGA ingestion (in-process, via thread)
# ---------------------------------------------------------------------------
_UPLOAD_DIR = Path(os.environ.get("APP_UPLOAD_DIR", "uploads")).resolve()
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def _save_upload(file: UploadFile) -> Path:
    out = _UPLOAD_DIR / file.filename
    data = await file.read()
    out.write_bytes(data)
    return out


def _run_ingest_in_thread(pdf_path: str, line_callback) -> int:
    """
    Drives the existing ingest.py inside the current process.
    `line_callback(str)` is invoked for every log line produced by ingest.
    Returns 0 on success, 1 on exception.
    """
    # Capture logs via a custom stream handler
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import ingest as ingest_mod  # the user's existing ingest.py

    class _LineHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                line_callback(self.format(record))
            except Exception:
                pass

    fmt = logging.Formatter("%(levelname)s %(message)s")
    handler = _LineHandler()
    handler.setFormatter(fmt)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    prev_level = root_logger.level
    root_logger.setLevel(logging.INFO)

    try:
        # ingest.py exposes a `main()` or `ingest(cfg)` entrypoint.
        # Pass the pdf path via env so we don't depend on its CLI parser.
        os.environ["INGEST_PDF_PATH"] = pdf_path
        if hasattr(ingest_mod, "main"):
            ingest_mod.main()  # type: ignore[attr-defined]
        else:
            line_callback("ERROR: ingest.py has no main() entrypoint")
            return 1
        return 0
    except SystemExit as se:
        return int(se.code or 0)
    except Exception as exc:
        line_callback(f"INGEST ERROR: {exc!r}")
        logger.exception("ingest.py raised")
        return 1
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(prev_level)


async def _stream_remote_ingest(saved: Path, filename: str):
    """Open a streaming POST to the remote ingest service and re-emit its SSE.

    The remote service owns S3 upload + documents-row insertion (with its own
    UUID + SHA-256, see ingest_remote/app/persist.py). The backend used to
    pre-insert a documents row here as well, but post-migration 005 that was
    both redundant (double insertion) and broken (its `ON CONFLICT (name)`
    referenced a unique constraint we dropped). The remote service emits a
    `stage:s3 status:done` event so the UI still sees the S3 URI.
    """
    url = settings.remote_ingest_url.rstrip("/") + "/ingest"
    headers: dict[str, str] = {}
    if settings.remote_ingest_secret:
        headers["X-Ingest-Secret"] = settings.remote_ingest_secret

    yield {
        "event": "start",
        "data": json.dumps(
            {"file": str(saved), "filename": filename, "mode": "remote"}
        ),
    }

    async with httpx.AsyncClient(timeout=settings.remote_ingest_timeout) as client:
        with open(saved, "rb") as fh:
            files = {"file": (filename, fh, "application/pdf")}
            data = {"document_name": filename}
            try:
                async with client.stream(
                    "POST", url, files=files, data=data, headers=headers
                ) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", errors="replace")
                        yield {
                            "event": "error",
                            "data": json.dumps(
                                {
                                    "type": "error",
                                    "message": f"remote {resp.status_code}: {body[:500]}",
                                }
                            ),
                        }
                        return

                    event_name = "info"
                    buf = b""
                    async for chunk in resp.aiter_bytes():
                        buf += chunk
                        while b"\n" in buf:
                            line, _, buf = buf.partition(b"\n")
                            line = line.rstrip(b"\r").decode("utf-8", errors="replace")
                            if line.startswith("event:"):
                                event_name = line[6:].strip() or "info"
                            elif line.startswith("data:"):
                                payload = line[5:]
                                if payload.startswith(" "):
                                    payload = payload[1:]
                                yield {"event": event_name, "data": payload}
                                event_name = "info"
                            elif line == "":
                                event_name = "info"
            except httpx.HTTPError as exc:
                logger.exception("remote ingest call failed")
                yield {
                    "event": "error",
                    "data": json.dumps(
                        {"type": "error", "message": f"remote call failed: {exc}"}
                    ),
                }


@router.post("/wega")
async def wega_ingest(file: UploadFile = File(...)) -> EventSourceResponse:
    saved = await _save_upload(file)

    queue: asyncio.Queue = asyncio.Queue()
    done_sentinel: object = object()
    loop = asyncio.get_running_loop()

    # ----- REMOTE path: proxy to the standalone ingest_remote service -----
    if settings.remote_ingest:
        logger.info(
            "remote_ingest=True; proxying %s to %s",
            file.filename, settings.remote_ingest_url,
        )
        return EventSourceResponse(_stream_remote_ingest(saved, file.filename or saved.name))

    # ----- LOCAL path (Vertex provider): pypdf + char chunker + Vertex embeddings -----
    if settings.llm_provider.lower() == "vertex":
        async def progress_cb(payload: dict) -> None:
            # decorate every payload with the live token snapshot
            payload = dict(payload)
            payload.setdefault("tokens", snapshot())
            await queue.put(payload)

        async def _worker_local() -> None:
            usage = UsageScope()
            usage.__enter__()
            try:
                summary = await ingest_document_local(
                    str(saved),
                    document_name=file.filename,
                    progress_cb=progress_cb,
                )
                logger.info("local ingest summary: %s tokens=%s", summary, usage.snapshot())
            except Exception as exc:
                logger.exception("local ingest failed")
                await queue.put({"type": "error", "message": str(exc), "tokens": usage.snapshot()})
            finally:
                usage.__exit__(None, None, None)
                await queue.put(done_sentinel)

        async def _events_local():
            task = asyncio.create_task(_worker_local())
            try:
                yield {
                    "event": "start",
                    "data": json.dumps({
                        "file": str(saved),
                        "filename": file.filename,
                        "mode": "local-vertex",
                    }),
                }
                while True:
                    item = await queue.get()
                    if item is done_sentinel:
                        break
                    yield {"event": item.get("type", "info"), "data": json.dumps(item)}
            finally:
                if not task.done():
                    task.cancel()

        return EventSourceResponse(_events_local())

    # ----- PROD path (Stellar provider on VDI): hand off to the existing ingest.py -----
    # The legacy ingest.py was written before migration 005 added document_id /
    # sha256, so it inserts chunks with NULL document_id. We bracket the
    # subprocess with two short SQL calls so the new schema + Upload-Session
    # panel still work end-to-end:
    #   1. before: INSERT a documents row with a fresh UUID + SHA-256
    #   2. after:  UPDATE the just-written chunks to carry that UUID
    document_name = file.filename or saved.name
    document_id = str(uuid.uuid4())
    file_bytes = saved.read_bytes()
    sha256_hex = hashlib.sha256(file_bytes).hexdigest()
    size_bytes = len(file_bytes)
    content_type = (
        "application/pdf"
        if Path(document_name).suffix.lower() == ".pdf"
        else "application/octet-stream"
    )

    s3_uri: str | None = None
    if settings.s3_enabled:
        try:
            s3_key = f"docs/{document_id[:8]}/{Path(document_name).name}"
            s3_uri = await s3_put(s3_key, file_bytes, content_type=content_type)
        except Exception:  # noqa: BLE001
            logger.exception("S3 upload failed (continuing)")

    try:
        async with acquire() as conn:
            await conn.execute(
                f'INSERT INTO "{settings.pg_schema}".documents '
                f'(id, name, s3_uri, size_bytes, content_type, sha256, uploaded_at) '
                f'VALUES ($1::uuid, $2, $3, $4, $5, $6, now())',
                document_id, document_name, s3_uri, size_bytes, content_type, sha256_hex,
            )
    except Exception:  # noqa: BLE001
        logger.exception("documents pre-insert failed (continuing)")

    def _line_cb(line: str) -> None:
        asyncio.run_coroutine_threadsafe(queue.put({"type": "log", "line": line}), loop)

    async def _backfill_chunk_document_id() -> int:
        """Stamp `document_id` onto every NULL-document_id chunk the legacy
        ingest.py just wrote under this filename. Returns the rowcount.
        """
        try:
            async with acquire() as conn:
                res = await conn.execute(
                    f'UPDATE "{settings.pg_schema}"."{settings.pg_table}" '
                    f'SET document_id = $1::uuid '
                    f'WHERE "documentName" = $2 AND document_id IS NULL '
                    f'  AND deleted_at IS NULL',
                    document_id, document_name,
                )
                # asyncpg returns "UPDATE N"; parse out N (best-effort).
                try:
                    return int(res.split()[-1])
                except (ValueError, IndexError):
                    return 0
        except Exception:  # noqa: BLE001
            logger.exception("backfill chunk document_id failed")
            return 0

    async def _worker() -> None:
        try:
            rc = await loop.run_in_executor(None, _run_ingest_in_thread, str(saved), _line_cb)
            stamped = await _backfill_chunk_document_id() if rc == 0 else 0
            await queue.put({
                "type": "done",
                "returncode": rc,
                "file": str(saved),
                "document": document_name,
                "document_id": document_id,
                "sha256": sha256_hex,
                "s3_uri": s3_uri,
                "processed": stamped,
                "total": stamped,
            })
        except Exception as exc:
            await queue.put({"type": "error", "message": str(exc)})
        finally:
            await queue.put(done_sentinel)

    async def _events():
        task = asyncio.create_task(_worker())
        try:
            yield {
                "event": "start",
                "data": json.dumps({
                    "file": str(saved),
                    "filename": document_name,
                    "mode": "wega-stellar",
                    "document_id": document_id,
                    "sha256": sha256_hex,
                    "s3_uri": s3_uri,
                }),
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
