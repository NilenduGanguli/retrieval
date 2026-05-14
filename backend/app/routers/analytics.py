"""Analytics endpoints — feeds the Analytics tab dashboard."""
from __future__ import annotations

from collections import Counter
from typing import Any

from fastapi import APIRouter

from ..config import settings
from ..db import acquire
from ..schemas import AnalyticsSummary, QueryLogEntry

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


@router.get("/summary", response_model=AnalyticsSummary)
async def summary() -> AnalyticsSummary:
    schema = settings.pg_schema
    async with acquire() as conn:
        total = await conn.fetchval(f'SELECT COUNT(*) FROM "{schema}".queries')
        total_24h = await conn.fetchval(
            f'SELECT COUNT(*) FROM "{schema}".queries '
            f"WHERE created_at > now() - INTERVAL '24 hours'"
        )
        token_totals = await conn.fetchrow(
            f"""
            SELECT
                COALESCE(SUM(COALESCE((token_usage->>'prompt')::bigint, 0)), 0)     AS prompt_total,
                COALESCE(SUM(COALESCE((token_usage->>'completion')::bigint, 0)), 0) AS completion_total,
                COALESCE(SUM(COALESCE((token_usage->>'total')::bigint, 0)), 0)      AS total_total,
                COALESCE(SUM(
                    CASE WHEN created_at > now() - INTERVAL '24 hours'
                         THEN COALESCE((token_usage->>'total')::bigint, 0)
                         ELSE 0
                    END), 0) AS total_24h
            FROM "{schema}".queries
            """
        )
        # Per-stage token aggregation across the whole history
        stage_rows = await conn.fetch(
            f"""
            SELECT key AS stage,
                   SUM(COALESCE((value->>'total')::bigint, 0))      AS total,
                   SUM(COALESCE((value->>'prompt')::bigint, 0))     AS prompt,
                   SUM(COALESCE((value->>'completion')::bigint, 0)) AS completion,
                   COUNT(*)                                          AS uses
            FROM "{schema}".queries, jsonb_each(stage_tokens)
            WHERE stage_tokens <> '{{}}'::jsonb
            GROUP BY key
            ORDER BY total DESC
            """
        )
        stage_breakdown = [
            {
                "stage": r["stage"],
                "total": int(r["total"] or 0),
                "prompt": int(r["prompt"] or 0),
                "completion": int(r["completion"] or 0),
                "uses": int(r["uses"] or 0),
            }
            for r in stage_rows
        ]
        latencies = await conn.fetch(
            f'SELECT (latency_ms->>\'total\')::float AS t, '
            f'(token_usage->>\'total\')::float AS tok, '
            f'strategy, top_chunk_ids '
            f'FROM "{schema}".queries '
            f'ORDER BY created_at DESC LIMIT 500'
        )

    times: list[float] = []
    tokens: list[float] = []
    strategy_counter: Counter[str] = Counter()
    doc_counter: Counter[str] = Counter()
    chunk_ids_acc: list[int] = []
    for r in latencies:
        t = r["t"]
        if t is not None:
            times.append(float(t))
        tok = r["tok"]
        if tok is not None:
            tokens.append(float(tok))
        strat = r["strategy"] or {}
        for k, v in strat.items():
            if v is True:
                strategy_counter[k] += 1
        for cid in (r["top_chunk_ids"] or []):
            chunk_ids_acc.append(int(cid))

    # Top documents from accumulated chunk ids
    top_docs: list[dict[str, Any]] = []
    if chunk_ids_acc:
        schema = settings.pg_schema
        table = settings.pg_table
        async with acquire() as conn:
            rows = await conn.fetch(
                f'SELECT "documentName" AS doc, COUNT(*)::int AS n '
                f'FROM "{schema}"."{table}" WHERE id = ANY($1::bigint[]) '
                f'GROUP BY "documentName" ORDER BY n DESC LIMIT 5',
                chunk_ids_acc,
            )
        top_docs = [
            {"document_name": r["doc"], "appearances": int(r["n"])} for r in rows
        ]

    times_sorted = sorted(times)
    return AnalyticsSummary(
        total_queries=int(total or 0),
        queries_24h=int(total_24h or 0),
        avg_latency_ms=(sum(times) / len(times)) if times else None,
        p50_latency_ms=_percentile(times_sorted, 0.5),
        p95_latency_ms=_percentile(times_sorted, 0.95),
        p99_latency_ms=_percentile(times_sorted, 0.99),
        avg_tokens=(sum(tokens) / len(tokens)) if tokens else None,
        top_documents=top_docs,
        strategy_mix=dict(strategy_counter),
        token_totals={
            "prompt": int(token_totals["prompt_total"] or 0) if token_totals else 0,
            "completion": int(token_totals["completion_total"] or 0) if token_totals else 0,
            "total": int(token_totals["total_total"] or 0) if token_totals else 0,
            "total_24h": int(token_totals["total_24h"] or 0) if token_totals else 0,
        },
        stage_token_breakdown=stage_breakdown,
    )


@router.get("/queries", response_model=list[QueryLogEntry])
async def recent_queries(limit: int = 50) -> list[QueryLogEntry]:
    schema = settings.pg_schema
    async with acquire() as conn:
        rows = await conn.fetch(
            f'SELECT id, query_text, strategy, latency_ms, top_chunk_ids, '
            f'answer_text, crag_confidence, created_at '
            f'FROM "{schema}".queries '
            f'ORDER BY created_at DESC LIMIT $1',
            min(max(limit, 1), 500),
        )
    return [
        QueryLogEntry(
            id=int(r["id"]),
            query_text=r["query_text"],
            strategy=r["strategy"] or {},
            latency_ms=r["latency_ms"] or {},
            top_chunk_ids=[int(c) for c in (r["top_chunk_ids"] or [])],
            answer_text=r["answer_text"],
            crag_confidence=r["crag_confidence"],
            created_at=r["created_at"].isoformat(),
        )
        for r in rows
    ]
