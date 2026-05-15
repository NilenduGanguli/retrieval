"""
KYC Intelligence router.

  GET  /api/kyc/taxonomy                        → DOC_TYPE_TAXONOMY
  POST /api/kyc/ingest         (SSE)            → 2-pass classify + extract + embed + store
  GET  /api/kyc/owners                          → unique owners + per-owner doc counts
  GET  /api/kyc/doc-types?owner=...             → doc types in corpus (or for one owner)
  POST /api/kyc/list-by-owner                   → metadata listing
  POST /api/kyc/extract                         → owner + doc_type → vector + LLM extraction
  POST /api/kyc/universal-search                → keyword across metadata + content
  GET  /api/kyc/browse?category=...             → all docs (optionally filtered)
  DELETE /api/kyc/{document_name}               → soft-delete a KYC document
"""
from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import unquote

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..pipeline import kyc as kyc_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kyc", tags=["kyc"])


# ============================================================
# Pydantic request bodies
# ============================================================
class OwnerSearchRequest(BaseModel):
    owner: str = Field(..., description="owner / company / person name")
    document_type: str | None = Field(None, description="optional doc type filter")


class ExtractRequest(BaseModel):
    owner: str
    document_type: str


class UniversalSearchRequest(BaseModel):
    keyword: str
    top_k: int = Field(default=8, ge=1, le=50)


# ============================================================
# 1) Static taxonomy
# ============================================================
@router.get("/taxonomy")
async def taxonomy() -> dict:
    return {
        "categories": kyc_pipeline.DOC_TYPE_TAXONOMY,
        "all_doc_types": kyc_pipeline.ALL_DOC_TYPES,
    }


# ============================================================
# 2) Ingest (SSE)
# ============================================================
@router.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> EventSourceResponse:
    data = await file.read()
    filename = file.filename or "upload.pdf"

    queue: asyncio.Queue = asyncio.Queue()
    done_sentinel: object = object()

    async def progress_cb(payload: dict) -> None:
        await queue.put(payload)

    async def _worker() -> None:
        try:
            summary = await kyc_pipeline.ingest_kyc_pdf(
                data, filename=filename, progress_cb=progress_cb,
            )
            await queue.put({"type": "done", **summary})
        except Exception as exc:
            logger.exception("kyc ingest failed")
            await queue.put({"type": "error", "message": str(exc)})
        finally:
            await queue.put(done_sentinel)

    async def _events():
        task = asyncio.create_task(_worker())
        try:
            yield {
                "event": "start",
                "data": json.dumps({"filename": filename, "size_bytes": len(data)}),
            }
            while True:
                item = await queue.get()
                if item is done_sentinel:
                    break
                yield {"event": item.get("type", "info"), "data": json.dumps(item, default=str)}
        finally:
            if not task.done():
                task.cancel()

    return EventSourceResponse(_events())


# ============================================================
# 3) Owners (+ counts)
# ============================================================
@router.get("/owners")
async def owners() -> list[dict]:
    return await kyc_pipeline.list_owners()


# ============================================================
# 4) Doc types (optionally filtered by owner)
# ============================================================
@router.get("/doc-types")
async def doc_types(owner: str | None = Query(default=None)) -> list[dict]:
    return await kyc_pipeline.list_doc_types(owner)


# ============================================================
# 5) List by owner (metadata only)
# ============================================================
@router.post("/list-by-owner")
async def list_by_owner(req: OwnerSearchRequest) -> dict:
    if not req.owner.strip():
        raise HTTPException(400, "owner is required")
    docs = await kyc_pipeline.list_by_owner(req.owner, req.document_type)
    return {"owner": req.owner, "document_type": req.document_type, "results": docs}


# ============================================================
# 6) Extract — owner + doc_type → vector + LLM extraction
# ============================================================
@router.post("/extract")
async def extract(req: ExtractRequest) -> dict:
    if not req.owner.strip() or not req.document_type.strip():
        raise HTTPException(400, "owner and document_type are required")
    result = await kyc_pipeline.extract_for_owner_type(req.owner, req.document_type)
    if not result:
        return {
            "owner": req.owner,
            "document_type": req.document_type,
            "result": None,
            "message": "no matching documents",
        }
    return {"owner": req.owner, "document_type": req.document_type, "result": result}


# ============================================================
# 7) Universal keyword search
# ============================================================
@router.post("/universal-search")
async def universal_search(req: UniversalSearchRequest) -> dict:
    if not req.keyword.strip():
        raise HTTPException(400, "keyword is required")
    results = await kyc_pipeline.universal_search(req.keyword, top_k=req.top_k)
    return {"keyword": req.keyword, "results": results}


# ============================================================
# 8) Browse all KYC docs (optionally filtered by category)
# ============================================================
@router.get("/browse")
async def browse(
    category: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict:
    docs = await kyc_pipeline.browse(category, limit)
    # Group by owner for convenience
    groups: dict[str, list] = {}
    for d in docs:
        groups.setdefault(d.get("owner") or "Unknown", []).append(d)
    return {
        "category": category,
        "total": len(docs),
        "groups": [{"owner": k, "docs": v} for k, v in sorted(groups.items())],
    }


# ============================================================
# 9) Soft-delete
# ============================================================
@router.delete("/{document_name}")
async def soft_delete(document_name: str) -> dict:
    name = unquote(document_name)
    result = await kyc_pipeline.soft_delete(name)
    if not result.get("ok"):
        raise HTTPException(404, result.get("message", "not found"))
    return result
