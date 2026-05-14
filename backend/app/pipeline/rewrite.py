"""
Query rewriting — generate paraphrased variants for multi-query retrieval.

Each variant is embedded separately; their candidate lists feed RRF. This
catches cases where the user's phrasing differs from the document's.
"""
from __future__ import annotations

import json
import logging
import re

from ..stellar_client import get_stellar, model_for

logger = logging.getLogger(__name__)

_PROMPT = """Rewrite the following user question in {n} different ways. Each rewrite
should preserve intent but vary vocabulary, phrasing, or specificity (a
broader version, a more specific version, a keyword-oriented version,
etc.). Return ONLY a JSON array of strings — no prose.

Question: {q}"""


_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


async def rewrite_query(
    query: str,
    *,
    n: int = 3,
    model: str | None = None,
) -> list[str]:
    chosen_model = model or model_for("fast")
    try:
        text, _ = await get_stellar().chat(
            model=chosen_model,
            messages=[
                {"role": "system", "content": "You generate diverse query paraphrases. Strict JSON output."},
                {"role": "user", "content": _PROMPT.format(n=n, q=query.strip())},
            ],
            temperature=0.5,
            max_tokens=400,
        )
    except Exception:  # noqa: BLE001
        logger.exception("rewrite_query LLM call failed; returning original")
        return [query]

    m = _JSON_ARRAY_RE.search(text)
    raw = m.group(0) if m else text
    try:
        items = json.loads(raw)
        out = [str(x).strip() for x in items if isinstance(x, str) and str(x).strip()]
    except Exception:  # noqa: BLE001
        out = []
    # Always include the original query first
    if not out:
        return [query]
    return [query] + out[:n]
