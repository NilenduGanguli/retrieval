"""
Stellar client for the remote ingestion service.

This is a portable copy of backend/app/stellar_client.py so the remote service
can be deployed on a different VM without dragging in the backend package
(it depends on backend-only modules — pydantic settings, the usage
accumulator, etc.).

Differences from the backend copy
---------------------------------
* No dependency on `backend.app.config` or `backend.app.usage`. The remote
  service has no UsageScope / analytics tab to feed.
* No Vertex dispatch — that lives in `local_ingest.py` for this service.
* Same public surface: `StellarClient.embed`, `embed_one`, `chat`, `chat_stream`,
  plus `TokenUsage` and the `get_stellar()` singleton helper.

The underlying COIN-OAuth + OpenAI-compatible-client wiring is reused exactly
from the project-root `get_llm.py` (StellarGenAI). That file must be
importable — either copied next to `ingest_remote/app/`, in `ingest_remote/`,
in the parent retrieval/ root, or accessible via `STELLAR_GETLLM_PATH`.

Embedding model is locked to `gte-large-en-v1.5` to stay in lockstep with
the main backend.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, List

logger = logging.getLogger(__name__)

STELLAR_EMBEDDING_MODEL = "gte-large-en-v1.5"


# ---------------------------------------------------------------------------
# Locate the project-root get_llm.py
# ---------------------------------------------------------------------------
def _ensure_get_llm_importable() -> None:
    if "get_llm" in sys.modules:
        return
    candidates: List[Path] = []
    env_path = os.environ.get("STELLAR_GETLLM_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())
    here = Path(__file__).resolve()
    candidates.append(here.parents[1])  # ingest_remote/
    candidates.append(here.parents[2])  # retrieval/  (when running from the repo)
    for c in candidates:
        if (c / "get_llm.py").is_file() and str(c) not in sys.path:
            sys.path.insert(0, str(c))
            logger.info("get_llm.py discovered at %s", c)
            return
    logger.debug("get_llm.py not found in known locations; relying on PYTHONPATH")


# ---------------------------------------------------------------------------
# Token usage (kept for parity with the backend; analytics tab not wired here)
# ---------------------------------------------------------------------------
@dataclass
class TokenUsage:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def as_dict(self) -> dict[str, int]:
        return {"prompt": self.prompt, "completion": self.completion, "total": self.total}


# ---------------------------------------------------------------------------
# StellarClient
# ---------------------------------------------------------------------------
class StellarClient:
    """Reusable Stellar client. One instance per service lifetime."""

    def __init__(self) -> None:
        _ensure_get_llm_importable()
        import get_llm  # type: ignore  # comes from the repo, not a wheel
        # StellarGenAI's __init__ wires the OpenAI() client + COIN token.
        # We never call its hard-coded helpers — we use its `.client` and
        # `.create_embedding` directly.
        self._gw = get_llm.StellarGenAI()
        if getattr(self._gw, "embedding_model_name", None) != STELLAR_EMBEDDING_MODEL:
            self._gw.embedding_model_name = STELLAR_EMBEDDING_MODEL

    # ---------- embeddings ----------
    async def embed(self, texts: str | list[str]) -> list[list[float]]:
        """Return embeddings for one or many texts. Always returns list-of-lists."""
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._gw.create_embedding, items)
        # create_embedding returns a flat vector for a single str input,
        # list-of-lists otherwise. Always normalise.
        if result and isinstance(result, list) and isinstance(result[0], (int, float)):
            return [list(result)]
        return [list(v) for v in result]

    def embed_sync(self, texts: str | list[str]) -> list[list[float]]:
        """Synchronous equivalent — used from threads spun up via run_in_executor."""
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        result = self._gw.create_embedding(items)
        if result and isinstance(result, list) and isinstance(result[0], (int, float)):
            return [list(result)]
        return [list(v) for v in result]

    async def embed_one(self, text: str) -> list[float]:
        out = await self.embed([text])
        return out[0]

    # ---------- chat / generation ----------
    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> tuple[str, TokenUsage]:
        """Synchronous-style chat completion. Returns (text, usage)."""
        loop = asyncio.get_running_loop()

        def _call() -> Any:
            return self._gw.client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

        resp = await loop.run_in_executor(None, _call)
        text = (resp.choices[0].message.content or "").strip()
        usage = TokenUsage(
            prompt=int(getattr(resp.usage, "prompt_tokens", 0) or 0),
            completion=int(getattr(resp.usage, "completion_tokens", 0) or 0),
        )
        return text, usage

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Yield token deltas as they arrive."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _produce() -> None:
            try:
                stream = self._gw.client.chat.completions.create(  # type: ignore[attr-defined]
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                for event in stream:
                    if event.choices:
                        delta_obj = event.choices[0].delta
                        content = getattr(delta_obj, "content", None) if delta_obj else None
                        if content:
                            asyncio.run_coroutine_threadsafe(queue.put(content), loop)
            except Exception as exc:  # noqa: BLE001
                logger.exception("chat_stream producer failed")
                asyncio.run_coroutine_threadsafe(queue.put(f"\n[stream-error: {exc}]"), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        loop.run_in_executor(None, _produce)
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_stellar: StellarClient | None = None


def get_stellar() -> StellarClient:
    """Cached StellarClient. COIN token + httpx client are built on first use."""
    global _stellar
    if _stellar is None:
        _stellar = StellarClient()
    return _stellar
