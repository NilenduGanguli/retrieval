"""
Embedding-only client wrapping `get_llm.VertexGenAI` — Vertex AI reached
through the internal corporate proxy (`VERTEX_BASE_URL`) authenticated
with a COIN token.

Used when `LLM_PROVIDER=vertex_internal` in the remote ingest service.
Exposes the same `embed_sync(...)` method shape as the local
`stellar_client.StellarClient`, so the ingestion path stays
provider-agnostic.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, List

from .config import settings

logger = logging.getLogger(__name__)


def _ensure_get_llm_importable() -> None:
    """Mirror of stellar_client._ensure_get_llm_importable.

    Order:
      1. $STELLAR_GETLLM_PATH (env override)
      2. ingest_remote/get_llm.py  ← preferred for standalone deploys
      3. <repo-root>/get_llm.py    ← dev fallback
    Force-promotes the chosen dir to the front of sys.path.
    """
    if "get_llm" in sys.modules:
        return
    candidates: list[Path] = []
    env_path = os.environ.get("STELLAR_GETLLM_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        candidates.append(p if p.is_dir() else p.parent)
    here = Path(__file__).resolve()
    candidates.append(here.parents[1])  # ingest_remote/
    candidates.append(here.parents[2])  # retrieval/
    for c in candidates:
        if (c / "get_llm.py").is_file():
            s = str(c)
            while s in sys.path:
                sys.path.remove(s)
            sys.path.insert(0, s)
            logger.info("get_llm.py pinned to front of sys.path from %s", c)
            return


class InternalVertexClient:
    """Thin wrapper around get_llm.VertexGenAI for embeddings only."""

    def __init__(self) -> None:
        _ensure_get_llm_importable()
        import get_llm  # type: ignore
        self._gw = get_llm.VertexGenAI()
        # Force the embedding model from our settings (overrides whatever
        # default get_llm.py picked up from VERTEX_EMBEDDING_MODEL — they
        # may differ).
        configured = settings.internal_vertex_embedding_model
        if configured:
            self._gw.embedding_model = configured

    # ---------- sync (called from inside run_in_executor) ----------
    def embed_sync(self, texts: str | list[str]) -> List[List[float]]:
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        resp = self._gw.client.models.embed_content(
            model=self._gw.embedding_model, contents=items,
        )
        return [list(getattr(e, "values", []) or []) for e in resp.embeddings]

    # ---------- async (handy if used outside an executor) ----------
    async def embed(self, texts: str | list[str]) -> List[List[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_sync, texts)

    async def embed_one(self, text: str) -> List[float]:
        out = await self.embed([text])
        return out[0] if out else []


_client: InternalVertexClient | None = None


def get_internal_vertex() -> InternalVertexClient:
    global _client
    if _client is None:
        _client = InternalVertexClient()
    return _client
