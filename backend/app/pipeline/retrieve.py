"""
Pipeline orchestrator — composes every stage based on the Strategy flags.

Stage order:
    rewrite ─┐
    hyde ───►├─► embed ─► (dense ║ sparse) ─► RRF ─► rerank ─► MMR
             │                                    │
             └─► (only if rewrite/hyde on)        └─► CRAG self-grade
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..config import settings
from ..db import acquire
from ..schemas import ChunkHit, LatencyMs, RetrieveResponse, Strategy
from ..stellar_client import get_stellar
from ..usage import UsageScope, delta, snapshot
from . import dense, sparse
from .crag import crag_grade
from .hyde import hyde_generate
from .mmr import MmrCandidate, mmr
from .rerank import RerankInput, llm_listwise_rerank
from .rewrite import rewrite_query
from .rrf import reciprocal_rank_fusion

ProgressCb = Callable[[dict[str, Any]], Awaitable[None]] | None

logger = logging.getLogger(__name__)


@dataclass
class _Timer:
    t0: float = field(default_factory=time.perf_counter)
    laps: dict[str, float] = field(default_factory=dict)


def _resolve_strategy(s: Strategy) -> dict[str, Any]:
    return {
        "rewrite": s.rewrite if s.rewrite is not None else settings.rag_rewrite_default,
        "hyde": s.hyde if s.hyde is not None else settings.rag_hyde_default,
        "use_contextual": (
            s.use_contextual
            if s.use_contextual is not None
            else settings.rag_contextual_default
        ),
        "hybrid": s.hybrid,
        "rerank": s.rerank if s.rerank is not None else settings.rag_rerank_default,
        "mmr": s.mmr,
        "crag": s.crag if s.crag is not None else settings.rag_crag_default,
        "top_k": s.top_k or settings.rag_final_k,
    }


async def _fetch_chunk_payload(chunk_ids: list[int]) -> dict[int, dict]:
    if not chunk_ids:
        return {}
    schema = settings.pg_schema
    table = settings.pg_table
    sql = f"""
        SELECT
            c.id,
            c.content,
            c."documentName" AS document_name,
            c."pageNumber"   AS page_number,
            ctx.context_text
        FROM "{schema}"."{table}" c
        LEFT JOIN "{schema}".chunk_context ctx ON ctx.chunk_id = c.id
        WHERE c.id = ANY($1::bigint[])
    """
    async with acquire() as conn:
        rows = await conn.fetch(sql, chunk_ids)
    out: dict[int, dict] = {}
    for r in rows:
        out[int(r["id"])] = {
            "content": r["content"] or "",
            "document_name": r["document_name"],
            "page_number": r["page_number"],
            "context_text": r["context_text"],
        }
    return out


async def run_retrieval(
    query: str,
    strategy: Strategy,
    *,
    progress_cb: ProgressCb = None,
) -> RetrieveResponse:
    """
    Stage-aware retrieval. When `progress_cb` is supplied it receives one
    payload per stage transition:
        {"stage": "rewrite", "status": "start"}
        {"stage": "rewrite", "status": "done", "ms": 80.1, "detail": {...}}
    """
    s = _resolve_strategy(strategy)
    stellar = get_stellar()
    laps: dict[str, float] = {}
    rewritten: list[str] = []
    hyde_text: str | None = None
    stage_tokens_map: dict[str, dict[str, int]] = {}

    async def emit(stage: str, status: str, **detail: Any) -> None:
        if progress_cb:
            await progress_cb({"stage": stage, "status": status, "detail": detail})

    def stage_tokens(before: dict[str, int], name: str | None = None) -> dict[str, int]:
        """Token delta consumed since `before` snapshot; also stash by stage name."""
        d = delta(before, snapshot())
        if name and (d.get("total") or 0) > 0:
            stage_tokens_map[name] = d
        return d

    # -------- 1. Rewrite --------
    if s["rewrite"]:
        await emit("rewrite", "start")
        t_start = time.perf_counter()
        tok_before = snapshot()
        rewritten = await rewrite_query(query, n=3)
        laps["rewrite"] = (time.perf_counter() - t_start) * 1000.0
        await emit("rewrite", "done", ms=laps["rewrite"], variants=rewritten, tokens=stage_tokens(tok_before, name="rewrite"))
    else:
        await emit("rewrite", "skip")

    queries_for_embed: list[str] = rewritten if rewritten else [query]

    # -------- 2. HyDE --------
    if s["hyde"]:
        await emit("hyde", "start")
        t_start = time.perf_counter()
        tok_before = snapshot()
        hyde_text = await hyde_generate(query)
        laps["hyde"] = (time.perf_counter() - t_start) * 1000.0
        await emit("hyde", "done", ms=laps["hyde"], hypothetical=hyde_text, tokens=stage_tokens(tok_before, name="hyde"))
        queries_for_embed = list({hyde_text, *queries_for_embed})
    else:
        await emit("hyde", "skip")

    # -------- 3. Embed --------
    await emit("embed", "start", n_queries=len(queries_for_embed))
    t_start = time.perf_counter()
    query_embeddings = await stellar.embed(queries_for_embed)
    laps["embed"] = (time.perf_counter() - t_start) * 1000.0
    await emit("embed", "done", ms=laps["embed"], dim=len(query_embeddings[0]) if query_embeddings else 0)

    # -------- 4. Dense --------
    async def _dense_one(vec: list[float]) -> list[dense.DenseHit]:
        return await dense.dense_search(
            vec, top_k=settings.rag_topk_dense, use_contextual=s["use_contextual"]
        )

    await emit("dense", "start", top_k=settings.rag_topk_dense, contextual=s["use_contextual"])
    t_start = time.perf_counter()
    dense_lists = await asyncio.gather(*(_dense_one(v) for v in query_embeddings))
    laps["dense"] = (time.perf_counter() - t_start) * 1000.0
    await emit("dense", "done", ms=laps["dense"], hits=sum(len(x) for x in dense_lists))

    # -------- 5. Sparse --------
    if s["hybrid"]:
        await emit("sparse", "start", top_k=settings.rag_topk_sparse)
        t_start = time.perf_counter()
        sparse_hits = await sparse.sparse_search(query, top_k=settings.rag_topk_sparse)
        laps["sparse"] = (time.perf_counter() - t_start) * 1000.0
        await emit("sparse", "done", ms=laps["sparse"], hits=len(sparse_hits))
    else:
        await emit("sparse", "skip")
        sparse_hits = []

    # -------- 6. RRF fusion --------
    await emit("fuse", "start")
    t_start = time.perf_counter()
    rankings: dict[str, list[tuple[int, dict]]] = {}
    for i, hits in enumerate(dense_lists):
        rankings[f"dense_v{i}"] = [
            (h.chunk_id, {
                "content": h.content,
                "document_name": h.document_name,
                "page_number": h.page_number,
                "context_text": h.context_text,
                "dense_score": h.score,
            }) for h in hits
        ]
    if sparse_hits:
        rankings["sparse"] = [
            (h.chunk_id, {
                "content": h.content,
                "document_name": h.document_name,
                "page_number": h.page_number,
                "sparse_score": h.score,
            }) for h in sparse_hits
        ]
    fused = reciprocal_rank_fusion(
        rankings,
        k=settings.rag_rrf_k,
        top_k=max(settings.rag_rerank_topn, s["top_k"] * 3),
    )
    laps["fuse"] = (time.perf_counter() - t_start) * 1000.0
    await emit("fuse", "done", ms=laps["fuse"], candidates=len(fused))

    # -------- 7. Listwise rerank --------
    if s["rerank"] and fused:
        await emit("rerank", "start", candidates=min(settings.rag_rerank_topn, len(fused)))
        t_start = time.perf_counter()
        tok_before_rr = snapshot()
        rerank_inputs = [
            RerankInput(
                chunk_id=f.chunk_id,
                content=f.payload.get("content", ""),
                document_name=f.payload.get("document_name"),
                page_number=f.payload.get("page_number"),
            )
            for f in fused[: settings.rag_rerank_topn]
        ]
        ranked_ids = await llm_listwise_rerank(
            query, rerank_inputs, top_n=settings.rag_rerank_topn, model=None,
        )
        order = {cid: i for i, cid in enumerate(ranked_ids)}
        fused.sort(key=lambda f: order.get(f.chunk_id, 10_000))
        laps["rerank"] = (time.perf_counter() - t_start) * 1000.0
        await emit("rerank", "done", ms=laps["rerank"], reordered=len(ranked_ids), tokens=stage_tokens(tok_before_rr, name="rerank"))
    else:
        await emit("rerank", "skip")

    # -------- 8. MMR --------
    final_k = int(s["top_k"])
    if s["mmr"] and fused:
        await emit("mmr", "start", final_k=final_k)
        t_start = time.perf_counter()
        mmr_in = [
            MmrCandidate(
                chunk_id=f.chunk_id,
                relevance=f.rrf_score,
                document_name=f.payload.get("document_name"),
                page_number=f.payload.get("page_number"),
            ) for f in fused
        ]
        kept = mmr(mmr_in, k=final_k, lambda_=settings.rag_mmr_lambda)
        kept_ids = [c.chunk_id for c in kept]
        by_id = {f.chunk_id: f for f in fused}
        fused = [by_id[i] for i in kept_ids if i in by_id]
        laps["mmr"] = (time.perf_counter() - t_start) * 1000.0
        await emit("mmr", "done", ms=laps["mmr"], kept=len(fused))
    else:
        fused = fused[:final_k]
        await emit("mmr", "skip")

    # -------- 9. CRAG --------
    crag_confidence: float | None = None
    if s["crag"] and fused:
        await emit("crag", "start")
        t_start = time.perf_counter()
        tok_before_crag = snapshot()
        verdict = await crag_grade(query, [f.payload.get("content", "") for f in fused])
        crag_confidence = verdict.confidence
        laps["crag"] = (time.perf_counter() - t_start) * 1000.0
        await emit("crag", "done", ms=laps["crag"], confidence=verdict.confidence, action=verdict.action, tokens=stage_tokens(tok_before_crag, name="crag"))
    else:
        await emit("crag", "skip")

    hits: list[ChunkHit] = []
    for i, f in enumerate(fused):
        hits.append(
            ChunkHit(
                chunk_id=f.chunk_id,
                document_name=f.payload.get("document_name"),
                page_number=f.payload.get("page_number"),
                content=f.payload.get("content", ""),
                context=f.payload.get("context_text"),
                score=f.rrf_score,
                score_breakdown=f.components,
                rank=i + 1,
            )
        )

    lat = LatencyMs(
        rewrite=laps.get("rewrite", 0.0),
        hyde=laps.get("hyde", 0.0),
        embed=laps.get("embed", 0.0),
        dense=laps.get("dense", 0.0),
        sparse=laps.get("sparse", 0.0),
        fuse=laps.get("fuse", 0.0),
        rerank=laps.get("rerank", 0.0),
        mmr=laps.get("mmr", 0.0),
        crag=laps.get("crag", 0.0),
        generate=0.0,
        total=sum(laps.values()),
    )

    return RetrieveResponse(
        query=query,
        hits=hits,
        latency_ms=lat,
        strategy_used=s,
        crag_confidence=crag_confidence,
        rewritten_queries=rewritten,
        hyde_text=hyde_text,
        stage_tokens=stage_tokens_map,
    )


async def log_query(
    query: str,
    response: RetrieveResponse,
    *,
    answer_text: str | None = None,
    token_usage: dict | None = None,
    citations: list[dict] | None = None,
    generate_ms: float = 0.0,
    stage_tokens_extra: dict | None = None,
) -> int:
    schema = settings.pg_schema
    chunk_ids = [h.chunk_id for h in response.hits]
    chunk_scores = {str(h.chunk_id): h.score for h in response.hits}
    latency = response.latency_ms.model_dump()
    latency["generate"] = generate_ms
    latency["total"] = latency.get("total", 0.0) + generate_ms

    # Fold per-stage tokens emitted by the pipeline (rewrite/hyde/rerank/crag)
    # together with whatever generate-stage tokens the caller hands us.
    stage_tokens = dict(response.stage_tokens or {})
    if stage_tokens_extra:
        stage_tokens.update(stage_tokens_extra)

    async with acquire() as conn:
        row = await conn.fetchrow(
            f"""
            INSERT INTO "{schema}".queries
                (query_text, strategy, latency_ms, top_chunk_ids,
                 chunk_scores, answer_text, citations, token_usage,
                 crag_confidence, stage_tokens)
            VALUES ($1, $2::jsonb, $3::jsonb, $4, $5::jsonb, $6, $7::jsonb,
                    $8::jsonb, $9, $10::jsonb)
            RETURNING id
            """,
            query,
            response.strategy_used,
            latency,
            chunk_ids,
            chunk_scores,
            answer_text,
            citations or [],
            token_usage or {},
            response.crag_confidence,
            stage_tokens,
        )
    return int(row["id"])
