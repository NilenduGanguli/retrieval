"""
Contextual Retrieval (Anthropic, Sept 2024).

For each chunk, we ask the LLM to write a 50–100 token prefix that
situates the chunk inside the broader document. We then embed
(context_prefix + chunk_content) and store that embedding alongside
the original. Retrieval can use either embedding via a UI toggle.

Original technique: https://www.anthropic.com/news/contextual-retrieval
Reported recall lift: ~49% reduction in retrieval failures.

This is the highest-impact stage in the whole pipeline; the
principal engineer will recognise it.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ..config import settings
from ..db import acquire, vec_to_pg, vector_type
from ..stellar_client import get_stellar, model_for

logger = logging.getLogger(__name__)


@dataclass
class ContextRow:
    chunk_id: int
    document_name: str
    content: str
    page_number: int | None


_PROMPT = """You are augmenting search retrieval. Below is the FULL DOCUMENT followed
by ONE CHUNK from that document. Write a short context (50-100 tokens)
that situates this chunk inside the document so that, on its own, the
chunk's purpose, topic and surrounding section are clear. Do not repeat
the chunk content. Output the context as a single paragraph, no
headings, no preamble like "This chunk".

<document>
{document}
</document>

<chunk>
{chunk}
</chunk>

Context:"""


async def _list_chunks_missing_context(
    document_name: str | None = None,
    limit: int = 100,
) -> list[ContextRow]:
    schema = settings.pg_schema
    table = settings.pg_table
    args: list = []
    where = []
    if document_name:
        where.append(f'c."documentName" = ${len(args)+1}')
        args.append(document_name)
    where_sql = (" AND " + " AND ".join(where)) if where else ""
    args.append(limit)
    sql = f"""
        SELECT c.id, c."documentName" AS document_name, c.content, c."pageNumber" AS page_number
        FROM "{schema}"."{table}" c
        LEFT JOIN "{schema}".chunk_context ctx ON ctx.chunk_id = c.id
        WHERE ctx.chunk_id IS NULL {where_sql}
        ORDER BY c.id
        LIMIT ${len(args)}
    """
    async with acquire() as conn:
        rows = await conn.fetch(sql, *args)
    return [
        ContextRow(
            chunk_id=int(r["id"]),
            document_name=r["document_name"] or "",
            content=r["content"] or "",
            page_number=r["page_number"],
        )
        for r in rows
    ]


async def _load_document_text(document_name: str, max_chars: int = 60_000) -> str:
    schema = settings.pg_schema
    table = settings.pg_table
    async with acquire() as conn:
        rows = await conn.fetch(
            f'SELECT content FROM "{schema}"."{table}" '
            f'WHERE "documentName" = $1 ORDER BY "pageNumber", id',
            document_name,
        )
    pieces = []
    n = 0
    for r in rows:
        c = (r["content"] or "").strip()
        if not c:
            continue
        pieces.append(c)
        n += len(c)
        if n >= max_chars:
            break
    return "\n\n".join(pieces)


async def _generate_context_for_chunk(
    document_text: str,
    chunk_content: str,
    model: str,
) -> str:
    prompt = _PROMPT.format(document=document_text, chunk=chunk_content)
    try:
        text, _ = await get_stellar().chat(
            model=model,
            messages=[
                {"role": "system", "content": "You write concise retrieval-augmenting context paragraphs."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=180,
        )
        return text.strip()
    except Exception:  # noqa: BLE001
        logger.exception("contextual chunk gen failed")
        return ""


async def _persist_context(
    chunk_id: int,
    context_text: str,
    context_embedding: list[float],
    model: str,
) -> None:
    schema = settings.pg_schema
    vec_lit = vec_to_pg(context_embedding)
    async with acquire() as conn:
        await conn.execute(
            f"""
            INSERT INTO "{schema}".chunk_context
                (chunk_id, context_text, context_embedding, generator_model, generated_at)
            VALUES ($1, $2, $3::{vector_type()}, $4, now())
            ON CONFLICT (chunk_id) DO UPDATE
                SET context_text = EXCLUDED.context_text,
                    context_embedding = EXCLUDED.context_embedding,
                    generator_model = EXCLUDED.generator_model,
                    generated_at = now()
            """,
            chunk_id, context_text, vec_lit, model,
        )


async def generate_context_batch(
    *,
    document_name: str | None = None,
    limit: int = 100,
    concurrency: int = 4,
    model: str | None = None,
    progress_cb=None,  # async callable(progress: float, info: dict) -> None
) -> dict[str, int]:
    """
    Generate contextual prefixes for chunks that don't have one yet.
    Embeds the (context + content) string and stores both in chunk_context.

    Returns {processed: N, skipped: M, failed: K}.
    """
    chosen_model = model or model_for("contextual")
    rows = await _list_chunks_missing_context(document_name=document_name, limit=limit)
    if not rows:
        return {"processed": 0, "skipped": 0, "failed": 0, "total": 0}

    # Cache document text — many chunks share a doc.
    doc_cache: dict[str, str] = {}

    sem = asyncio.Semaphore(concurrency)
    stats = {"processed": 0, "skipped": 0, "failed": 0, "total": len(rows)}
    stellar = get_stellar()

    async def _process(idx: int, row: ContextRow) -> None:
        async with sem:
            if not row.content.strip() or not row.document_name:
                stats["skipped"] += 1
                return
            doc_text = doc_cache.get(row.document_name)
            if doc_text is None:
                doc_text = await _load_document_text(row.document_name)
                doc_cache[row.document_name] = doc_text
            context_text = await _generate_context_for_chunk(
                doc_text, row.content, chosen_model
            )
            if not context_text:
                stats["failed"] += 1
                return
            to_embed = f"{context_text}\n\n{row.content}"
            try:
                emb = await stellar.embed_one(to_embed)
            except Exception:  # noqa: BLE001
                logger.exception("contextual embed failed")
                stats["failed"] += 1
                return
            try:
                await _persist_context(row.chunk_id, context_text, emb, chosen_model)
                stats["processed"] += 1
            except Exception:  # noqa: BLE001
                logger.exception("contextual persist failed")
                stats["failed"] += 1
            if progress_cb is not None:
                done = stats["processed"] + stats["skipped"] + stats["failed"]
                await progress_cb(
                    done / max(1, stats["total"]),
                    {
                        "chunk_id": row.chunk_id,
                        "document": row.document_name,
                        "page": row.page_number,
                        "stats": dict(stats),
                    },
                )

    await asyncio.gather(*(_process(i, r) for i, r in enumerate(rows)))
    return stats
