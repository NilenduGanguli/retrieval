"""
HyDE — Hypothetical Document Embeddings.

Gao et al. 2022 — instead of embedding the question, ask an LLM to draft
a fake answer, embed that, and use it for retrieval. The fake answer
shares vocabulary and structure with real answers, giving a measurable
recall lift on underspecified questions.
"""
from __future__ import annotations

import logging

from ..stellar_client import get_stellar, model_for

logger = logging.getLogger(__name__)

_PROMPT = """Write a short, factual passage that would answer the following question.
Write it as if you were quoting an authoritative document. Do not hedge or
say you are unsure — produce a confident hypothetical paragraph. Three to
five sentences.

Question: {q}

Passage:"""


async def hyde_generate(query: str, *, model: str | None = None) -> str:
    chosen_model = model or model_for("fast")
    try:
        text, _ = await get_stellar().chat(
            model=chosen_model,
            messages=[
                {"role": "system", "content": "You write concise, confident hypothetical answer passages."},
                {"role": "user", "content": _PROMPT.format(q=query.strip())},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        return text.strip()
    except Exception:  # noqa: BLE001
        logger.exception("HyDE generation failed; falling back to raw query")
        return query
