"""Benchmark / golden-set HTTP routes - CRUD + run-batch endpoint.

Routes mounted at /api/bench/*. Persists into vector.golden_questions and
vector.bench_runs tables (see migration 001_rag_extensions.sql).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from ..config import settings
from ..db import acquire
from ..pipeline.generate import GenerationContext, generate_answer
from ..pipeline.retrieve import run_retrieval
from ..schemas import BenchRunRequest, BenchRunResult, GoldenQuestion
from ..scoring.metrics import (
    context_precision,
    faithfulness,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from ..stellar_client import get_stellar, model_for
from ..usage import UsageScope, delta, snapshot

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/bench", tags=["bench"])

SCHEMA = settings.pg_schema


@router.get("/questions", response_model=list[GoldenQuestion])
async def list_questions() -> list[GoldenQuestion]:
    async with acquire() as conn:
        rows = await conn.fetch(
            f'SELECT id, question, ground_truth_chunk_ids, ground_truth_answer, tags '
            f'FROM "{SCHEMA}".golden_questions ORDER BY id'
        )
    return [
        GoldenQuestion(
            id=int(r["id"]),
            question=r["question"],
            ground_truth_chunk_ids=[int(c) for c in (r["ground_truth_chunk_ids"] or [])],
            ground_truth_answer=r["ground_truth_answer"],
            tags=list(r["tags"] or []),
        )
        for r in rows
    ]


@router.post("/questions", response_model=GoldenQuestion)
async def add_question(q: GoldenQuestion) -> GoldenQuestion:
    async with acquire() as conn:
        row = await conn.fetchrow(
            f'INSERT INTO "{SCHEMA}".golden_questions '
            f'(question, ground_truth_chunk_ids, ground_truth_answer, tags) '
            f'VALUES ($1, $2, $3, $4) '
            f'RETURNING id, question, ground_truth_chunk_ids, ground_truth_answer, tags',
            q.question,
            list(q.ground_truth_chunk_ids),
            q.ground_truth_answer,
            list(q.tags),
        )
    return GoldenQuestion(
        id=int(row["id"]),
        question=row["question"],
        ground_truth_chunk_ids=[int(c) for c in (row["ground_truth_chunk_ids"] or [])],
        ground_truth_answer=row["ground_truth_answer"],
        tags=list(row["tags"] or []),
    )


@router.delete("/questions/{qid}")
async def delete_question(qid: int) -> dict:
    async with acquire() as conn:
        result = await conn.execute(
            f'DELETE FROM "{SCHEMA}".golden_questions WHERE id = $1', qid
        )
    return {"ok": True, "deleted": result.endswith("1")}


_SEED_PROMPT = """You are creating a small evaluation dataset for a retrieval-augmented
generation system. Given the passage below, write ONE short, specific
factual question that can be answered ONLY by reading this passage.
The question must be concrete (a date, a number, a name, a definition,
etc.) — NOT generic.

Respond with STRICT JSON only: {{"question": "..."}}.

Document: {doc_name}
Passage:
{passage}"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    # strip ```json ... ``` fences if present
    fm = _FENCE_RE.search(text)
    candidate = fm.group(1).strip() if fm else text
    # find first { ... last }
    m = _JSON_RE.search(candidate)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


async def _generate_question_for_chunk(doc_name: str, passage: str) -> str | None:
    try:
        text, _ = await get_stellar().chat(
            model=model_for("fast"),
            messages=[
                {"role": "system", "content": "You generate concrete factual questions for retrieval evaluation."},
                {"role": "user", "content": _SEED_PROMPT.format(doc_name=doc_name, passage=passage[:2000])},
            ],
            temperature=0.4,
            max_tokens=300,
        )
    except Exception:
        logger.exception("seed question gen failed")
        return None
    obj = _extract_json_object(text or "")
    if not obj:
        logger.warning("seed: unparseable response: %r", (text or "")[:200])
        return None
    q = (obj.get("question") or "").strip()
    return q if q else None


@router.post("/seed-from-docs")
async def seed_from_docs(per_document: int = 2, limit_docs: int | None = None) -> dict:
    """
    Auto-populate the golden set: for each document (up to limit_docs),
    pick `per_document` random chunks, ask the LLM to draft a concrete
    factual question, and store it with that chunk as ground truth.
    """
    per_document = max(1, min(10, per_document))
    async with acquire() as conn:
        docs = await conn.fetch(
            f'SELECT DISTINCT "documentName" AS name '
            f'FROM "{SCHEMA}"."{settings.pg_table}" '
            f'WHERE deleted_at IS NULL '
            f'ORDER BY 1'
        )
    if limit_docs is not None:
        docs = docs[:limit_docs]

    inserted: list[dict] = []
    for d in docs:
        name = d["name"]
        async with acquire() as conn:
            rows = await conn.fetch(
                f'SELECT id, content '
                f'FROM "{SCHEMA}"."{settings.pg_table}" '
                f'WHERE "documentName" = $1 AND deleted_at IS NULL '
                f'  AND length(content) >= 80 '
                f'ORDER BY random() LIMIT $2',
                name, per_document,
            )
        for r in rows:
            chunk_id = int(r["id"])
            content = r["content"] or ""
            question = await _generate_question_for_chunk(name, content)
            if not question:
                continue
            async with acquire() as conn:
                row = await conn.fetchrow(
                    f'INSERT INTO "{SCHEMA}".golden_questions '
                    f'(question, ground_truth_chunk_ids, tags) '
                    f'VALUES ($1, $2, $3) RETURNING id',
                    question, [chunk_id], ["auto-seed", name],
                )
            inserted.append({
                "id": int(row["id"]),
                "question": question,
                "document_name": name,
                "ground_truth_chunk_ids": [chunk_id],
            })
    return {"inserted": len(inserted), "questions": inserted}


@router.post("/run")
async def run_batch(req: BenchRunRequest) -> EventSourceResponse:
    """
    Stream the benchmark run via SSE. Events emitted:

      start          - {n_questions, total_steps}
      question_start - {index, id, question, n_total}
      stage          - forwarded from run_retrieval; current pipeline stage
      judge_start    - {kind: "faithfulness" | "context_precision"}
      judge_done     - {kind, score, ms}
      question_done  - {index, id, question, metrics, retrieved, ground_truth,
                        elapsed_ms, answer}
      progress       - {processed, total, percent}
      done           - {run_id, n_questions, metrics}  (final aggregate)
      error          - {message}
    """
    async with acquire() as conn:
        if req.question_ids:
            rows = await conn.fetch(
                f'SELECT id, question, ground_truth_chunk_ids, ground_truth_answer '
                f'FROM "{SCHEMA}".golden_questions WHERE id = ANY($1::bigint[])',
                req.question_ids,
            )
        else:
            rows = await conn.fetch(
                f'SELECT id, question, ground_truth_chunk_ids, ground_truth_answer '
                f'FROM "{SCHEMA}".golden_questions'
            )

    n_q = len(rows)
    queue: asyncio.Queue = asyncio.Queue()
    done_sentinel: object = object()

    async def put(event: str, payload: dict) -> None:
        await queue.put({"event": event, "data": json.dumps(payload)})

    async def stage_cb(payload: dict) -> None:
        await put("stage", payload)

    async def _worker() -> None:
        try:
            await put("start", {"n_questions": n_q})
            sums = {
                "recall@5": 0.0, "recall@10": 0.0,
                "mrr@10": 0.0, "ndcg@10": 0.0,
                "faithfulness": 0.0, "context_precision": 0.0,
            }
            aggregate_tokens = {"prompt": 0, "completion": 0, "total": 0}

            for i, r in enumerate(rows):
                q = r["question"]
                gt = [int(x) for x in (r["ground_truth_chunk_ids"] or [])]
                qid = int(r["id"])

                await put("question_start", {
                    "index": i,
                    "id": qid,
                    "question": q,
                    "n_total": n_q,
                })

                # Per-question token scope — isolates this question's spend.
                q_scope = UsageScope()
                q_scope.__enter__()

                t0 = time.perf_counter()
                retrieval = await run_retrieval(q, req.strategy, progress_cb=stage_cb)
                retrieved = [h.chunk_id for h in retrieval.hits]

                r5 = recall_at_k(retrieved, gt, 5)
                r10 = recall_at_k(retrieved, gt, 10)
                mrr = mrr_at_k(retrieved, gt, 10)
                ndcg = ndcg_at_k(retrieved, gt, 10)

                faith = ctxp = 0.0
                ans_text = ""
                if retrieval.hits:
                    ctxs = [
                        GenerationContext(
                            chunk_id=h.chunk_id,
                            content=h.content,
                            document_name=h.document_name,
                            page_number=h.page_number,
                            context_text=h.context,
                        )
                        for h in retrieval.hits
                    ]
                    await put("stage", {"stage": "generate", "status": "start"})
                    gen_t0 = time.perf_counter()
                    ans_text, _ = await generate_answer(q, ctxs)
                    gen_ms = (time.perf_counter() - gen_t0) * 1000.0
                    await put("stage", {"stage": "generate", "status": "done", "detail": {"ms": gen_ms}})

                    try:
                        await put("judge_start", {"kind": "faithfulness"})
                        j_t0 = time.perf_counter()
                        fres = await faithfulness(q, ans_text, [h.content for h in retrieval.hits])
                        faith = fres.score
                        await put("judge_done", {"kind": "faithfulness", "score": faith, "ms": (time.perf_counter() - j_t0) * 1000.0})

                        await put("judge_start", {"kind": "context_precision"})
                        j_t0 = time.perf_counter()
                        cres = await context_precision(q, [h.content for h in retrieval.hits])
                        ctxp = cres.score
                        await put("judge_done", {"kind": "context_precision", "score": ctxp, "ms": (time.perf_counter() - j_t0) * 1000.0})
                    except Exception as exc:
                        logger.exception("judge metric failed")
                        await put("error", {"message": f"judge failed: {exc}"})

                sums["recall@5"] += r5
                sums["recall@10"] += r10
                sums["mrr@10"] += mrr
                sums["ndcg@10"] += ndcg
                sums["faithfulness"] += faith
                sums["context_precision"] += ctxp

                q_tokens = q_scope.snapshot()
                q_scope.__exit__(None, None, None)
                aggregate_tokens["prompt"] += q_tokens["prompt"]
                aggregate_tokens["completion"] += q_tokens["completion"]
                aggregate_tokens["total"] = aggregate_tokens["prompt"] + aggregate_tokens["completion"]

                qd = {
                    "index": i,
                    "id": qid,
                    "question": q,
                    "retrieved": retrieved,
                    "ground_truth": gt,
                    "metrics": {
                        "recall@5": r5,
                        "recall@10": r10,
                        "mrr@10": mrr,
                        "ndcg@10": ndcg,
                        "faithfulness": faith,
                        "context_precision": ctxp,
                    },
                    "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                    "answer": ans_text,
                    "tokens": q_tokens,
                }
                await put("question_done", qd)
                await put("progress", {
                    "processed": i + 1,
                    "total": n_q,
                    "percent": (i + 1) / max(1, n_q),
                    "tokens_so_far": dict(aggregate_tokens),
                })

            metrics = {k: (v / n_q if n_q else 0.0) for k, v in sums.items()}

            async with acquire() as conn:
                row = await conn.fetchrow(
                    f'INSERT INTO "{SCHEMA}".bench_runs (label, strategy, metrics, n_questions) '
                    f'VALUES ($1, $2::jsonb, $3::jsonb, $4) RETURNING id',
                    req.label,
                    req.strategy.model_dump(),
                    metrics,
                    n_q,
                )

            await put("done", {
                "run_id": int(row["id"]),
                "label": req.label,
                "n_questions": n_q,
                "metrics": metrics,
                "tokens": aggregate_tokens,
            })
        except Exception as exc:
            logger.exception("bench run failed")
            await put("error", {"message": str(exc)})
        finally:
            await queue.put(done_sentinel)

    async def _events():
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

    return EventSourceResponse(_events())


@router.get("/runs")
async def list_runs(limit: int = 30) -> list[dict]:
    async with acquire() as conn:
        rows = await conn.fetch(
            f'SELECT id, label, strategy, metrics, n_questions, created_at '
            f'FROM "{SCHEMA}".bench_runs ORDER BY created_at DESC LIMIT $1',
            min(max(limit, 1), 200),
        )
    return [
        {
            "id": int(r["id"]),
            "label": r["label"],
            "strategy": r["strategy"] or {},
            "metrics": r["metrics"] or {},
            "n_questions": int(r["n_questions"] or 0),
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]
