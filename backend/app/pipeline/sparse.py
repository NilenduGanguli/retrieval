"""Sparse / lexical search using Postgres FTS (ts_rank_cd ~ BM25-ish)."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import settings
from ..db import acquire


@dataclass
class SparseHit:
    chunk_id: int
    content: str
    document_name: str | None
    page_number: int | None
    score: float
    rank: int


# Postgres FTS doesn't accept arbitrary user text via to_tsquery (it'd raise on
# punctuation). Build a websearch_to_tsquery — far more permissive, ignores
# stop-words, treats space as AND, supports "phrases" and -negation.
async def sparse_search(query: str, top_k: int) -> list[SparseHit]:
    schema = settings.pg_schema
    table = settings.pg_table

    sql = f"""
        WITH q AS (
            SELECT websearch_to_tsquery('english', $1) AS tsq
        )
        SELECT
            c.id,
            c.content,
            c."documentName" AS document_name,
            c."pageNumber"   AS page_number,
            ts_rank_cd(c.content_tsv, q.tsq, 32) AS score
        FROM "{schema}"."{table}" c, q
        WHERE c.content_tsv @@ q.tsq AND c.deleted_at IS NULL
        ORDER BY score DESC
        LIMIT $2
    """

    async with acquire() as conn:
        rows = await conn.fetch(sql, query, top_k)

    return [
        SparseHit(
            chunk_id=int(r["id"]),
            content=r["content"] or "",
            document_name=r["document_name"],
            page_number=r["page_number"],
            score=float(r["score"] or 0.0),
            rank=i + 1,
        )
        for i, r in enumerate(rows)
    ]
