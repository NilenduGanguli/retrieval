"""Dense vector search against pgvector with HNSW."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import settings
from ..db import acquire, vec_to_pg, vector_type


@dataclass
class DenseHit:
    chunk_id: int
    content: str
    document_name: str | None
    page_number: int | None
    context_text: str | None
    score: float           # 1 - cosine_distance (so higher is better)
    rank: int


async def dense_search(
    query_vec: list[float],
    top_k: int,
    *,
    use_contextual: bool = False,
) -> list[DenseHit]:
    """
    Return top_k chunks by cosine similarity.

    use_contextual=True uses chunk_context.context_embedding (Anthropic-style
    contextual retrieval). Falls back to chunk_embeddings.embedding if no
    contextual row exists for a given chunk (LEFT JOIN preserves coverage).
    """
    schema = settings.pg_schema
    table = settings.pg_table
    vec_lit = vec_to_pg(query_vec)

    if use_contextual:
        sql = f"""
            WITH joined AS (
                SELECT
                    c.id,
                    c.content,
                    c."documentName"   AS document_name,
                    c."pageNumber"     AS page_number,
                    ctx.context_text,
                    COALESCE(ctx.context_embedding, c.embedding) AS effective_embedding
                FROM "{schema}"."{table}" c
                LEFT JOIN "{schema}".chunk_context ctx ON ctx.chunk_id = c.id
                WHERE c.deleted_at IS NULL
            )
            SELECT
                id, content, document_name, page_number, context_text,
                1 - (effective_embedding <=> $1::{vector_type()}) AS score
            FROM joined
            ORDER BY effective_embedding <=> $1::{vector_type()}
            LIMIT $2
        """
    else:
        sql = f"""
            SELECT
                c.id,
                c.content,
                c."documentName" AS document_name,
                c."pageNumber"   AS page_number,
                NULL::text       AS context_text,
                1 - (c.embedding <=> $1::{vector_type()}) AS score
            FROM "{schema}"."{table}" c
            WHERE c.deleted_at IS NULL
            ORDER BY c.embedding <=> $1::{vector_type()}
            LIMIT $2
        """

    async with acquire() as conn:
        rows = await conn.fetch(sql, vec_lit, top_k)

    return [
        DenseHit(
            chunk_id=int(r["id"]),
            content=r["content"] or "",
            document_name=r["document_name"],
            page_number=r["page_number"],
            context_text=r["context_text"],
            score=float(r["score"] or 0.0),
            rank=i + 1,
        )
        for i, r in enumerate(rows)
    ]
