"""
Thin async wrapper over the existing get_llm.py classes.

get_llm.py has two synchronous classes (`StellarGenAI`, `VertexGenAI`) with
hard-coded model names. For the RAG framework we need:

  * Async access (so FastAPI doesn't block on Stellar calls)
  * Per-call model selection (rerank uses a different model than gen)
  * Streaming generation (SSE)
  * Token-usage capture for the analytics tab

This module *reuses* StellarGenAI's auth + httpx client wiring by importing
it once at startup, then talks to the underlying OpenAI-compatible client
directly with our per-call kwargs.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

# Make the existing get_llm.py importable. We look in a few places so the
# import works in both the dev tree and the Docker deployment:
#   1. $STELLAR_GETLLM_PATH (env override)
#   2. backend/get_llm.py     ← committed copy, ships with the backend
#   3. <repo-root>/get_llm.py ← canonical dev location (../../get_llm.py)
#   4. /app/get_llm.py         ← Docker image (COPY get_llm.py /app/)
#
# The found directory is force-promoted to the front of sys.path so that
# a stray get_llm.py somewhere else on the path can't shadow our copy.
def _ensure_get_llm_importable() -> None:
    if "get_llm" in sys.modules:
        return
    candidates: list[Path] = []
    env_path = os.environ.get("STELLAR_GETLLM_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        candidates.append(p if p.is_dir() else p.parent)
    here = Path(__file__).resolve()
    candidates.append(here.parents[1])  # backend/
    candidates.append(here.parents[2])  # <repo-root>/
    candidates.append(Path("/app"))       # canonical Docker root
    for c in candidates:
        if (c / "get_llm.py").is_file():
            s = str(c)
            while s in sys.path:
                sys.path.remove(s)
            sys.path.insert(0, s)
            return
    # Fall through — let `import get_llm` raise ModuleNotFoundError
    # with the standard message; the user will see the candidate list
    # in our log warning.
    import logging as _l
    _l.getLogger(__name__).warning(
        "get_llm.py not found in any of: %s — set STELLAR_GETLLM_PATH or "
        "place the file inside backend/ or the repo root.",
        [str(c) for c in candidates],
    )


_ensure_get_llm_importable()

# Note: get_llm.py validates env vars at import time, so we lazy-import it
# only when the client is actually instantiated. This keeps modules
# importable on dev machines without Stellar creds configured.

from .config import settings
from .usage import _add_to, current_acc, record_usage

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def as_dict(self) -> dict[str, int]:
        return {"prompt": self.prompt, "completion": self.completion, "total": self.total}


class StellarClient:
    """Reusable Stellar client. One instance per app lifetime."""

    def __init__(self) -> None:
        # Lazy import — only at first instantiation. Prevents env-var checks
        # in get_llm.py from blowing up at module load time.
        import get_llm  # type: ignore
        # StellarGenAI's __init__ wires the OpenAI() client + COIN token.
        # We never call its `generate_content_llama` (it's hard-coded to one
        # model) — we use its underlying `.client` directly.
        self._gw = get_llm.StellarGenAI()

    # ---------- embeddings ----------
    async def embed(self, texts: str | list[str]) -> list[list[float]]:
        """Return embeddings for one or many texts. Always returns list-of-lists."""
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        # StellarGenAI.create_embedding is sync — push it off-loop.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._gw.create_embedding, items)
        # The underlying API may return a flat list when input was a single str.
        if items and isinstance(result, list) and result and isinstance(result[0], (int, float)):
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
        record_usage(usage.prompt, usage.completion)
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

        final_usage = {"prompt": 0, "completion": 0}
        acc_ref = current_acc()

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
                    u = getattr(event, "usage", None)
                    if u is not None:
                        final_usage["prompt"] = int(getattr(u, "prompt_tokens", 0) or 0)
                        final_usage["completion"] = int(getattr(u, "completion_tokens", 0) or 0)
            except Exception as exc:  # noqa: BLE001
                logger.exception("chat_stream producer failed")
                asyncio.run_coroutine_threadsafe(queue.put(f"\n[stream-error: {exc}]"), loop)
            finally:
                # Record stream-aggregate usage into the captured accumulator
                # (ContextVar doesn't propagate to executor threads, so we
                # use the reference captured on the main task).
                try:
                    if final_usage["prompt"] or final_usage["completion"]:
                        _add_to(acc_ref, final_usage["prompt"], final_usage["completion"])
                except Exception:
                    pass
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        # Drive the sync generator in a thread.
        loop.run_in_executor(None, _produce)
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item


# Singleton — instantiated lazily so import order is forgiving.
_stellar: StellarClient | None = None


def get_stellar() -> "StellarClient | object":
    """
    Returns either a StellarClient (production) or a VertexClient
    (local-test) depending on settings.llm_provider.

    Both implement the same public surface:
        embed(texts) -> list[list[float]]
        embed_one(text) -> list[float]
        chat(model, messages, ...) -> (text, TokenUsage)
        chat_stream(model, messages, ...) -> AsyncIterator[str]

    Pipeline modules call this exactly once via the alias and the rest of
    the code is provider-agnostic.
    """
    if settings.llm_provider.lower() == "vertex":
        from .vertex_client import get_vertex
        return get_vertex()
    global _stellar
    if _stellar is None:
        _stellar = StellarClient()
    return _stellar


def model_for(task: str) -> str:
    """
    Resolve the model name for a pipeline task, honouring the active provider.

    task is one of: "final_gen", "rerank", "fast", "contextual", "embedding"
    """
    provider = settings.llm_provider.lower()
    attr = f"{provider}_{task}_model" if task != "embedding" else f"{provider}_embedding_model"
    name = getattr(settings, attr, None)
    if name:
        return name
    # Fallback to Stellar defaults if Vertex setting missing
    return getattr(settings, f"stellar_{task}_model", "")
