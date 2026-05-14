"""
Maximal Marginal Relevance — diversify the final top-K by penalising
chunks that look too similar to ones already selected.

If we have embeddings for every candidate, we use cosine sim. Otherwise
we fall back to "different doc / different page" as a cheap diversity
signal so the function is always callable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MmrCandidate:
    chunk_id: int
    relevance: float                       # higher is better
    embedding: list[float] | None = None
    document_name: str | None = None
    page_number: int | None = None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mmr(
    candidates: list[MmrCandidate],
    k: int,
    lambda_: float = 0.7,
) -> list[MmrCandidate]:
    """
    Greedy MMR: arg max_{d ∈ R \\ S}  λ * relevance(d) - (1-λ) * max_{s∈S} sim(d,s)
    """
    if k <= 0 or not candidates:
        return []
    pool = list(candidates)
    selected: list[MmrCandidate] = [pool.pop(0)]  # most relevant first

    # Precompute embedding matrix when available
    have_embeddings = all(c.embedding is not None for c in candidates)
    emb_map: dict[int, np.ndarray] = {}
    if have_embeddings:
        for c in candidates:
            if c.embedding is not None:
                emb_map[c.chunk_id] = np.asarray(c.embedding, dtype=np.float32)

    while pool and len(selected) < k:
        best_idx = 0
        best_score = -1e9
        for i, c in enumerate(pool):
            if have_embeddings:
                sims = [
                    _cosine(emb_map[c.chunk_id], emb_map[s.chunk_id])
                    for s in selected
                ]
                redundancy = max(sims) if sims else 0.0
            else:
                # Cheap fallback: same doc and same page → fully redundant.
                redundancy = 0.0
                for s in selected:
                    if s.document_name and c.document_name == s.document_name:
                        redundancy = max(redundancy, 0.6)
                        if s.page_number is not None and c.page_number == s.page_number:
                            redundancy = max(redundancy, 0.9)
            score = lambda_ * c.relevance - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(pool.pop(best_idx))
    return selected
