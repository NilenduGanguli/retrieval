"""
Corrective-RAG (CRAG) self-grader.

Yan et al. 2024 — after retrieval, ask an LLM whether the retrieved
context is sufficient to answer the question. If confidence is low we
expose that signal in the UI ("retrieval was weak — consider rewriting")
and optionally trigger a fallback round of retrieval with a rewritten
query.

We keep this small and fast: it's a single call to the 8B model.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ..stellar_client import get_stellar, model_for

logger = logging.getLogger(__name__)


@dataclass
class CragVerdict:
    confidence: float        # 0..1
    action: str              # "use" | "rewrite" | "refuse"
    rationale: str


_PROMPT = """You evaluate retrieval quality for a RAG system. Given a user question and
the retrieved passages, judge whether the passages contain enough
information to answer the question well.

Respond with STRICT JSON only:
{{
  "confidence": <float 0..1>,
  "action": "use" | "rewrite" | "refuse",
  "rationale": "<short reason>"
}}

Guidelines:
  * "use" → confidence ≥ 0.6, passages clearly contain the answer.
  * "rewrite" → confidence 0.2..0.6, passages are tangentially related;
     a different phrasing might retrieve better.
  * "refuse" → confidence < 0.2, passages are off-topic; the system
     should tell the user it doesn't know rather than hallucinate.

Question: {q}

Passages:
{passages}"""


def _trim(s: str, n: int = 500) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


async def crag_grade(
    query: str,
    passages: list[str],
    *,
    model: str | None = None,
) -> CragVerdict:
    chosen_model = model or model_for("fast")
    block = "\n\n".join(f"[{i+1}] {_trim(p)}" for i, p in enumerate(passages))
    try:
        text, _ = await get_stellar().chat(
            model=chosen_model,
            messages=[
                {"role": "system", "content": "You are a strict retrieval-quality grader."},
                {"role": "user", "content": _PROMPT.format(q=query.strip(), passages=block)},
            ],
            temperature=0.0,
            max_tokens=200,
        )
    except Exception:  # noqa: BLE001
        logger.exception("CRAG grading failed; defaulting to use")
        return CragVerdict(confidence=0.5, action="use", rationale="grader call failed")

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    raw = m.group(0) if m else text
    try:
        obj = json.loads(raw)
        return CragVerdict(
            confidence=float(obj.get("confidence", 0.5)),
            action=str(obj.get("action", "use")).strip().lower(),
            rationale=str(obj.get("rationale", "")).strip(),
        )
    except Exception:  # noqa: BLE001
        logger.warning("CRAG parse failed for output: %r", text)
        return CragVerdict(confidence=0.5, action="use", rationale="unparseable grader output")
