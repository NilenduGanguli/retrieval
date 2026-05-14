"""
Listwise LLM reranker (RankGPT / RankZephyr style).

Instead of a local cross-encoder, we feed the top-N candidates to a chat
model with a strict ranking prompt. The model returns a JSON list of
candidate IDs ordered by relevance, which we use to re-sort.

References:
  * Sun et al. 2023 — "Is ChatGPT Good at Search? Investigating LLMs as
    Re-Ranking Agents" (https://arxiv.org/abs/2304.09542)
  * Pradeep et al. 2024 — "RankZephyr"
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ..config import settings
from ..stellar_client import get_stellar, model_for

logger = logging.getLogger(__name__)

# How many characters of each candidate to show the model. Keep this small
# to fit a large top-N in one call.
_PASSAGE_CHAR_BUDGET = 700


@dataclass
class RerankInput:
    chunk_id: int
    content: str
    document_name: str | None = None
    page_number: int | None = None


_PROMPT = """You are a precise relevance ranker. Given the user query and a list of
candidate passages, return a JSON array of the passage IDs sorted from MOST
relevant to LEAST relevant to the query. Only include IDs you've seen.

Query:
{query}

Candidates (id ▸ passage):
{passages}

Output STRICT JSON only, no prose. Example: [3, 7, 1, 4, 2]"""


def _build_passages_block(items: list[RerankInput]) -> str:
    lines = []
    for it in items:
        snippet = (it.content or "").strip().replace("\n", " ")
        if len(snippet) > _PASSAGE_CHAR_BUDGET:
            snippet = snippet[:_PASSAGE_CHAR_BUDGET] + "…"
        loc = ""
        if it.document_name:
            loc = f" [{it.document_name}"
            if it.page_number is not None:
                loc += f", p.{it.page_number}"
            loc += "]"
        lines.append(f"{it.chunk_id} ▸{loc} {snippet}")
    return "\n".join(lines)


_JSON_ARRAY_RE = re.compile(r"\[\s*(?:-?\d+\s*,\s*)*-?\d+\s*\]")


def _parse_ranked_ids(raw: str) -> list[int]:
    """
    LLMs occasionally wrap the JSON in prose. Be robust: find the first
    JSON-looking array and parse that.
    """
    if not raw:
        return []
    m = _JSON_ARRAY_RE.search(raw)
    if m:
        try:
            return [int(x) for x in json.loads(m.group(0))]
        except Exception:  # noqa: BLE001
            pass
    # last-resort: extract all integers in order
    return [int(x) for x in re.findall(r"-?\d+", raw)]


async def llm_listwise_rerank(
    query: str,
    candidates: list[RerankInput],
    *,
    top_n: int | None = None,
    model: str | None = None,
) -> list[int]:
    """
    Returns chunk_ids sorted by LLM-judged relevance. If parsing fails or
    the LLM returns nothing usable, falls back to the input order.
    """
    if not candidates:
        return []
    if top_n is not None:
        candidates = candidates[:top_n]

    prompt = _PROMPT.format(
        query=query.strip(),
        passages=_build_passages_block(candidates),
    )
    chosen_model = model or model_for("rerank")
    try:
        text, _ = await get_stellar().chat(
            model=chosen_model,
            messages=[
                {"role": "system", "content": "You re-rank passages. Be exact."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Listwise rerank LLM call failed; falling back to input order")
        return [c.chunk_id for c in candidates]

    ranked = _parse_ranked_ids(text)
    candidate_ids = {c.chunk_id for c in candidates}
    seen: set[int] = set()
    out: list[int] = []
    for cid in ranked:
        if cid in candidate_ids and cid not in seen:
            out.append(cid)
            seen.add(cid)
    # Append any candidates the LLM forgot, preserving original order
    for c in candidates:
        if c.chunk_id not in seen:
            out.append(c.chunk_id)
    return out
