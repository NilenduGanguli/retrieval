"""Retrieve + Chat (SSE streaming) endpoints."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import AsyncIterator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from ..pipeline.generate import GenerationContext, stream_answer
from ..pipeline.retrieve import log_query, run_retrieval
from ..schemas import ChatRequest, RetrieveRequest, RetrieveResponse
from ..usage import UsageScope, delta, snapshot

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["retrieve"])


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """Run retrieval only - no generation. Useful for the strategy A/B demo."""
    with UsageScope() as scope:
        response = await run_retrieval(req.query, req.strategy)
        try:
            await log_query(req.query, response, token_usage=scope.snapshot())
        except Exception:
            logger.exception("query log write failed")
    return response


def _extract_citations(text: str, hits) -> list[dict]:
    """Pull all [N] markers from the answer; map to chunk_id."""
    out: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for m in re.finditer(r"\[(\d+)\]", text):
        n = int(m.group(1))
        if 1 <= n <= len(hits):
            cid = hits[n - 1].chunk_id
            span = (m.start(), m.end())
            if (cid, span[0]) in seen:
                continue
            seen.add((cid, span[0]))
            out.append({"marker": n, "chunk_id": cid, "span": list(span)})
    return out


@router.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    """
    Streamed RAG chat. Emits over SSE in order:

      stage   - one per pipeline transition (rewrite/hyde/embed/dense/sparse/
                fuse/rerank/mmr/crag)  with status: start | done | skip
      meta    - full retrieved hits + latency map, fires once retrieval is done
      token   - tokens of the streamed answer
      done    - final payload: citations, total + generation latency
      error   - if generation blows up
    """
    queue: asyncio.Queue = asyncio.Queue()
    done_sentinel: object = object()

    async def stage_cb(payload: dict) -> None:
        await queue.put({"event": "stage", "data": json.dumps(payload)})

    async def _worker() -> None:
        scope = UsageScope()
        scope.__enter__()
        try:
            retr_t0 = time.perf_counter()
            response = await run_retrieval(req.query, req.strategy, progress_cb=stage_cb)
            retr_ms = (time.perf_counter() - retr_t0) * 1000.0

            await queue.put({
                "event": "meta",
                "data": json.dumps({
                    "strategy": response.strategy_used,
                    "latency_ms": response.latency_ms.model_dump(),
                    "retrieval_ms": retr_ms,
                    "rewritten_queries": response.rewritten_queries,
                    "hyde_text": response.hyde_text,
                    "crag_confidence": response.crag_confidence,
                    "hits": [
                        {
                            "rank": h.rank,
                            "chunk_id": h.chunk_id,
                            "document_name": h.document_name,
                            "page_number": h.page_number,
                            "score": h.score,
                            "score_breakdown": h.score_breakdown,
                            "content": h.content,
                            "context": h.context,
                        }
                        for h in response.hits
                    ],
                }),
            })

            tok_before_gen = scope.snapshot()
            await queue.put({
                "event": "stage",
                "data": json.dumps({"stage": "generate", "status": "start"}),
            })

            gen_contexts = [
                GenerationContext(
                    chunk_id=h.chunk_id,
                    content=h.content,
                    document_name=h.document_name,
                    page_number=h.page_number,
                    context_text=h.context,
                )
                for h in response.hits
            ]
            full_text_parts: list[str] = []
            gen_t0 = time.perf_counter()
            try:
                async for tok in stream_answer(req.query, gen_contexts, history=req.history):
                    full_text_parts.append(tok)
                    await queue.put({"event": "token", "data": tok})
            except Exception as exc:
                logger.exception("stream_answer failed")
                await queue.put({"event": "error", "data": str(exc)})

            gen_ms = (time.perf_counter() - gen_t0) * 1000.0
            full_text = "".join(full_text_parts)
            citations = _extract_citations(full_text, response.hits)

            # final token totals are now correct (stream_answer recorded
            # its final usage in the finally block of the producer).
            total_tokens = scope.snapshot()
            # Generate-stage tokens = whatever was added during generation;
            # we can approximate by snapshotting before generate started
            # but here we just send the totals — UI already shows per-stage
            # tokens via the stage events for the earlier LLM stages.
            gen_tokens = delta(tok_before_gen, scope.snapshot())
            await queue.put({
                "event": "stage",
                "data": json.dumps({
                    "stage": "generate",
                    "status": "done",
                    "detail": {
                        "ms": gen_ms,
                        "chunks_streamed": len(full_text_parts),
                        "tokens": gen_tokens,
                    },
                }),
            })

            try:
                await log_query(
                    req.query,
                    response,
                    answer_text=full_text,
                    citations=citations,
                    generate_ms=gen_ms,
                    token_usage=total_tokens,
                    stage_tokens_extra={"generate": gen_tokens} if (gen_tokens.get("total") or 0) > 0 else None,
                )
            except Exception:
                logger.exception("query log write failed (post-stream)")

            await queue.put({
                "event": "done",
                "data": json.dumps({
                    "citations": citations,
                    "generate_ms": gen_ms,
                    "total_ms": (response.latency_ms.total or 0.0) + gen_ms,
                    "token_usage": total_tokens,
                }),
            })
        except Exception as exc:
            logger.exception("chat pipeline failed")
            await queue.put({"event": "error", "data": str(exc)})
        finally:
            scope.__exit__(None, None, None)
            await queue.put(done_sentinel)

    async def _gen() -> AsyncIterator[dict]:
        task = asyncio.create_task(_worker())
        try:
            while True:
                item = await queue.get()
                if item is done_sentinel:
                    break
                yield item
        finally:
            if not task.done():
                task.cancel()

    return EventSourceResponse(_gen())
