"""FastAPI application entrypoint."""
from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from fastapi.responses import ORJSONResponse as _DefaultResponse
except ImportError:  # orjson not installed → fall back to stdlib JSON.
    from fastapi.responses import JSONResponse as _DefaultResponse

from .config import settings
from .db import (
    close_pool,
    ensure_contextual_embedding_column,
    ensure_schema_and_tables,
    healthcheck,
    init_pool,
    run_migrations,
)
from .routers import analytics, bench, documents, ingest, kyc, retrieve

logging.basicConfig(
    level=getattr(logging, settings.app_log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting up...")
    app.state.db_ready = False
    app.state.s3_ready = False
    # Pump up the default executor — every sync LLM / embedding / OCR call
    # goes through loop.run_in_executor(None, ...). Default pool is small
    # (~6 threads on 2 vCPUs); a RAG workload wants 32+ so concurrent
    # requests don't block on the executor's queue.
    _executor_workers = int(os.environ.get("APP_EXECUTOR_WORKERS", "64"))
    try:
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(
                max_workers=_executor_workers,
                thread_name_prefix="app-exec",
            )
        )
        logger.info("default executor sized to %d threads", _executor_workers)
    except Exception:
        logger.exception("failed to size default executor (continuing with default)")
    try:
        await init_pool()
        # 1. Schema + base table bootstrap (creates schema/table on a fresh DB)
        await ensure_schema_and_tables()
        # 2. SQL migrations (idempotent — safe on existing or fresh DB)
        migration_dir = Path(__file__).resolve().parents[1] / "migrations"
        for sql_file in sorted(migration_dir.glob("*.sql")):
            try:
                await run_migrations(str(sql_file))
            except Exception:
                logger.exception("migration failed: %s", sql_file)
                raise
        # 3. Add the runtime-dim vector columns for contextual + KYC chunks
        await ensure_contextual_embedding_column()
        try:
            from .pipeline.kyc import ensure_kyc_embedding_column
            await ensure_kyc_embedding_column()
        except Exception:
            logger.exception("kyc embedding column bootstrap failed (continuing)")
        app.state.db_ready = True
        logger.info("ready (DB connected)")
    except Exception as exc:
        logger.warning(
            "DB unreachable (%s) - starting in UI-only mode. "
            "The UI will load but DB-dependent features will return errors.",
            exc,
        )
    # S3 bucket bootstrap (non-fatal)
    if settings.s3_enabled:
        try:
            from .s3_store import s3_ensure_bucket
            ok = await s3_ensure_bucket()
            app.state.s3_ready = ok
            if ok:
                logger.info("S3 bucket %s ready at %s", settings.s3_bucket, settings.s3_endpoint_url or "AWS")
        except Exception as exc:
            logger.warning("S3 init failed (continuing without S3): %s", exc)
    yield
    logger.info("shutting down...")
    try:
        await close_pool()
    except Exception:
        pass


app = FastAPI(
    title="RAG Framework",
    version="0.1.0",
    description="Hybrid retrieval + LLM listwise rerank + contextual chunks + CRAG.",
    lifespan=lifespan,
    # orjson when available — significantly faster JSON serialization on big
    # responses (retrieve, chunks endpoints). Falls back to stdlib JSON.
    default_response_class=_DefaultResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, **await healthcheck()}


@app.get("/api/config")
async def config() -> dict:
    """Expose the safe-to-share config so the UI can render strategy defaults."""
    from .stellar_client import model_for
    return {
        "provider": settings.llm_provider,
        "embedding_model": model_for("embedding"),
        "final_gen_model": model_for("final_gen"),
        "rerank_model": model_for("rerank"),
        "fast_model": model_for("fast"),
        "contextual_model": model_for("contextual"),
        "defaults": {
            "rewrite": settings.rag_rewrite_default,
            "hyde": settings.rag_hyde_default,
            "rerank": settings.rag_rerank_default,
            "crag": settings.rag_crag_default,
            "contextual": settings.rag_contextual_default,
            "top_k": settings.rag_final_k,
            "rerank_topn": settings.rag_rerank_topn,
            "mmr_lambda": settings.rag_mmr_lambda,
        },
        "s3": {
            "enabled": settings.s3_enabled,
            "endpoint": settings.s3_endpoint_url or None,
            "bucket": settings.s3_bucket,
        },
        "remote_ingest": settings.remote_ingest,
    }


# Routers
app.include_router(retrieve.router)
app.include_router(ingest.router)
app.include_router(documents.router)
app.include_router(analytics.router)
app.include_router(bench.router)
app.include_router(kyc.router)


# ----------------------------------------------------------------------------
# Static frontend — single-container deploy.
#
# Build output: frontend/dist/
#   ├── index.html
#   ├── assets/index-<hash>.js     ← hashed, immutable: cache forever
#   ├── assets/index-<hash>.css    ← ditto
#   └── (anything else from public/)
#
# Routing strategy (FastAPI matches routes in registration order):
#   1. /api/*                   → API routers (registered above)
#   2. /assets/*                → StaticFiles mount with long cache headers
#   3. /                        → serves dist/index.html
#   4. /{any}                   → file-from-dist if exists, else index.html
#                                 (so React-Router-style client routes work)
#
# In dev mode the user runs Vite separately on :5173 and `frontend/dist/`
# typically doesn't exist — we expose a helpful hint instead.
# ----------------------------------------------------------------------------
_frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
_dist_index = _frontend_dist / "index.html"
_dist_assets = _frontend_dist / "assets"


class _CachedStatic(StaticFiles):
    """StaticFiles with long-lived cache headers for hashed Vite bundles."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        # The Vite output filenames already include a content hash, so we
        # can serve them as immutable.
        if response.status_code == 200:
            response.headers["cache-control"] = "public, max-age=31536000, immutable"
        return response


if _dist_index.is_file():
    if _dist_assets.is_dir():
        app.mount(
            "/assets",
            _CachedStatic(directory=str(_dist_assets)),
            name="assets",
        )
        logger.info("Serving frontend assets from %s", _dist_assets)

    @app.get("/", include_in_schema=False)
    async def root_index() -> FileResponse:
        return FileResponse(_dist_index, media_type="text/html")

    @app.get("/{full_path:path}", include_in_schema=False, response_model=None)
    async def spa_fallback(full_path: str):
        # Unknown /api/* paths must 404, not return SPA HTML
        if full_path.startswith("api/") or full_path.startswith("api"):
            if full_path == "api" or full_path.startswith("api/"):
                return JSONResponse({"detail": "API route not found"}, status_code=404)

        # Resolve safely against dist/ (path-traversal guard)
        candidate = (_frontend_dist / full_path).resolve()
        try:
            candidate.relative_to(_frontend_dist.resolve())
        except ValueError:
            # Path tried to escape dist/ — refuse and serve the shell instead.
            return FileResponse(_dist_index, media_type="text/html")

        if candidate.is_file():
            return FileResponse(candidate)

        # Client-side route (e.g. /retrieval, /benchmark) — let React handle it.
        return FileResponse(_dist_index, media_type="text/html")

    logger.info("FastAPI is hosting frontend bundle from %s", _frontend_dist)
else:
    @app.get("/", include_in_schema=False)
    async def root_dev() -> dict:
        return {
            "ok": True,
            "frontend": "not_built",
            "expected_path": str(_frontend_dist),
            "message": (
                "Frontend bundle not found. Either run `./entrypoint.sh` to build it, "
                "or run `./entrypoint.sh --dev` to use the Vite dev server on :5173."
            ),
        }
    logger.warning(
        "frontend/dist/index.html not found at %s — only the JSON API is being served",
        _dist_index,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=os.environ.get("APP_RELOAD", "false").lower() == "true",
    )
