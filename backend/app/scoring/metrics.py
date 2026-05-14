"""
Retrieval & generation quality metrics.

  * recall@k        — did we put any ground-truth chunk in top-k?
  * MRR@k           — reciprocal-rank of the FIRST ground-truth chunk
  * nDCG@k          — graded ranking
  * faithfulness    — LLM-judged: do all claims in the answer trace to context?
  * context_prec    — LLM-judged: of retrieved chunks, how many are relevant?

The LLM-judged metrics use the fast 8B model so a full eval pass over a
golden set of ~50 questions runs in ~30 seconds.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass

from ..stellar_client import get_stellar, model_for

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-Python ranking metrics
# ---------------------------------------------------------------------------
def recall_at_k(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    top = set(retrieved[:k])
    return len(top & set(ground_truth)) / len(set(ground_truth))


def mrr_at_k(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    gt = set(ground_truth)
    for i, cid in enumerate(retrieved[:k], start=1):
        if cid in gt:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    gt = set(ground_truth)
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k], start=1):
        rel = 1.0 if cid in gt else 0.0
        dcg += rel / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gt), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# LLM-as-judge metrics (cheap, fast 8B model)
# ---------------------------------------------------------------------------
_FAITHFULNESS_PROMPT = """You evaluate whether an answer is *faithful* to the provided context.

For each factual claim in the answer, decide if it is supported by the
context. Then output STRICT JSON: {{"supported": N, "total": M, "score": N/M}}

Question: {q}
Answer:
{a}

Context:
{ctx}"""


_CONTEXT_PRECISION_PROMPT = """You evaluate retrieval relevance. Given a question and a list of retrieved
passages, mark each passage as 'relevant' (1) or 'not relevant' (0) to
answering the question. Output STRICT JSON like {{"marks": [1,0,1,1,0]}}
with exactly one entry per passage in the input order.

Question: {q}

Passages:
{passages}"""


def _parse_json_blob(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


@dataclass
class JudgedScore:
    score: float
    detail: dict


async def faithfulness(
    query: str,
    answer: str,
    contexts: list[str],
    *,
    model: str | None = None,
) -> JudgedScore:
    if not answer.strip() or not contexts:
        return JudgedScore(score=0.0, detail={"reason": "empty inputs"})
    chosen = model or model_for("fast")
    ctx = "\n\n---\n\n".join(contexts[:10])
    text, _ = await get_stellar().chat(
        model=chosen,
        messages=[
            {"role": "system", "content": "You strictly evaluate factual support."},
            {"role": "user", "content": _FAITHFULNESS_PROMPT.format(q=query, a=answer, ctx=ctx)},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    obj = _parse_json_blob(text) or {}
    supported = float(obj.get("supported", 0))
    total = float(obj.get("total", 0))
    score = float(obj.get("score", (supported / total) if total > 0 else 0.0))
    return JudgedScore(score=max(0.0, min(1.0, score)), detail=obj)


async def context_precision(
    query: str,
    contexts: list[str],
    *,
    model: str | None = None,
) -> JudgedScore:
    if not contexts:
        return JudgedScore(score=0.0, detail={"reason": "no contexts"})
    chosen = model or model_for("fast")
    block = "\n\n".join(f"[{i+1}] {(c or '').strip()[:600]}" for i, c in enumerate(contexts))
    text, _ = await get_stellar().chat(
        model=chosen,
        messages=[
            {"role": "system", "content": "You judge retrieval relevance. Strict JSON."},
            {"role": "user", "content": _CONTEXT_PRECISION_PROMPT.format(q=query, passages=block)},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    obj = _parse_json_blob(text) or {}
    marks = [int(m) for m in obj.get("marks", []) if isinstance(m, (int, float))]
    if not marks:
        return JudgedScore(score=0.0, detail={"reason": "no marks", "raw": text})
    score = sum(marks) / len(marks)
    return JudgedScore(score=score, detail={"marks": marks})
