"""Reciprocal Rank Fusion — merge multiple ranked lists."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class FusedHit:
    chunk_id: int
    rrf_score: float
    components: dict[str, float]   # source-name -> contribution
    payload: dict                  # carry original fields (content, doc, page, ...)


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[int, dict]]],
    k: int = 60,
    top_k: int | None = None,
) -> list[FusedHit]:
    """
    Merge several ranked lists of `(chunk_id, payload_dict)` tuples.

    Score per chunk = Σ_source  1 / (k + rank_in_source)
    The constant k smooths out tail rankings; 60 is the value from the
    original Cormack et al. paper, also used by Vespa / Weaviate.

    `payload` lets each ranker pass through its own content/doc fields; we
    merge them with last-write-wins so the most recent ranker's payload
    wins per field.
    """
    contributions: dict[int, dict[str, float]] = {}
    payloads: dict[int, dict] = {}

    for source, ranked in rankings.items():
        for rank, (chunk_id, payload) in enumerate(ranked, start=1):
            contrib = 1.0 / (k + rank)
            contributions.setdefault(chunk_id, {})[source] = contrib
            base = payloads.get(chunk_id, {})
            # Don't clobber non-empty fields with empties
            for key, val in (payload or {}).items():
                if val is not None and val != "":
                    base[key] = val
            payloads[chunk_id] = base

    fused = [
        FusedHit(
            chunk_id=cid,
            rrf_score=sum(comps.values()),
            components=comps,
            payload=payloads[cid],
        )
        for cid, comps in contributions.items()
    ]
    fused.sort(key=lambda h: h.rrf_score, reverse=True)
    if top_k is not None:
        fused = fused[:top_k]
    return fused
