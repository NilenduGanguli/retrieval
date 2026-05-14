"""
Final answer generation with citation grounding.

We prompt the model to write an answer with inline [N] citation markers
mapping to the retrieved chunks. The frontend renders each [N] as a
clickable chip that scrolls to the source.

The model is instructed to *refuse* if the context doesn't support the
answer — this is the second line of defence after CRAG.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import AsyncIterator

from ..stellar_client import get_stellar, model_for

logger = logging.getLogger(__name__)


@dataclass
class GenerationContext:
    chunk_id: int
    content: str
    document_name: str | None = None
    page_number: int | None = None
    context_text: str | None = None


_SYSTEM = """You are a careful, grounded research assistant for a Retrieval-Augmented
Generation (RAG) system. Answer the user's question using ONLY the
information in the numbered Context items provided below.

REQUIREMENTS

1. Be thorough and analytical. Aim for a multi-paragraph answer when the
   question is open-ended. Lead with a 1-2 sentence direct answer, then
   expand with details, comparisons, numbers, and any nuance the context
   supports. Use markdown structure (paragraphs, lists, bold for key terms).

2. Every factual claim MUST be followed by at least one inline citation in
   square brackets, e.g. [1] or [1][3]. Cite the EXACT source number you
   used; never invent numbers. If a claim is not supported by any context
   item, do not make the claim - say "the provided sources do not cover
   that" instead.

3. Quote concrete data when present (figures, dates, percentages, names).
   Where two sources agree or disagree, point it out and cite both.

4. End the answer with a "## Sources" section (markdown H2 — exactly two
   hash characters). Render the sources as a markdown bullet list, ONE
   bullet per cited source, in this exact format:

       - **[N] document_name** — page X · chunk #ID
         _short paraphrase of what this chunk contributed (1 line, italics)_

   Where N is the bracket number, document_name comes from the context
   header, page X is the page number if shown, and ID is the integer
   chunk identifier shown in the header. Do NOT include sources you did
   not cite. Do NOT use raw HTML — markdown only.

5. If the context truly does not contain the answer, say so plainly in a
   single short paragraph; do NOT emit a Sources section in that case.

Style: clear, grounded, no hedging, no speculation, no chain-of-thought.
"""


def _format_context(contexts: list[GenerationContext]) -> str:
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        # Header that exposes EVERY identifier so the model can attribute precisely.
        header_parts = [
            f"[{i}]",
            f"document_name: {ctx.document_name or 'unknown'}",
        ]
        if ctx.page_number is not None:
            header_parts.append(f"page: {ctx.page_number}")
        header_parts.append(f"chunk_id: {ctx.chunk_id}")
        header = " | ".join(header_parts)

        body = ctx.content.strip()
        prefix = (ctx.context_text or "").strip()
        if prefix:
            body = f"_doc-context: {prefix}_\n{body}"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def _build_messages(
    query: str,
    contexts: list[GenerationContext],
    history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    history = history or []
    ctx_text = _format_context(contexts)
    msgs: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM}]
    # Keep the history short - last 6 turns.
    for h in history[-6:]:
        if h.get("role") in {"user", "assistant"} and h.get("content"):
            msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append(
        {
            "role": "user",
            "content": (
                f"### Question\n{query.strip()}\n\n"
                f"### Context\n{ctx_text}\n\n"
                f"### Task\n"
                f"Answer the question using ONLY the context above. Be "
                f"thorough, cite [N] inline, and end with a `## Sources` "
                f"section listing every cited source with document_name, "
                f"page, and chunk_id."
            ),
        }
    )
    return msgs


async def generate_answer(
    query: str,
    contexts: list[GenerationContext],
    *,
    history: list[dict[str, str]] | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> tuple[str, dict]:
    chosen = model or model_for("final_gen")
    msgs = _build_messages(query, contexts, history)
    text, usage = await get_stellar().chat(
        model=chosen,
        messages=msgs,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return text, usage.as_dict()


async def stream_answer(
    query: str,
    contexts: list[GenerationContext],
    *,
    history: list[dict[str, str]] | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> AsyncIterator[str]:
    chosen = model or model_for("final_gen")
    msgs = _build_messages(query, contexts, history)
    async for tok in get_stellar().chat_stream(
        model=chosen,
        messages=msgs,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        yield tok
