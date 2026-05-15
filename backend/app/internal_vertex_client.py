"""
Async wrapper around `get_llm.VertexGenAI` — i.e. Vertex AI reached through
the internal corporate proxy (`VERTEX_BASE_URL`) authenticated with a COIN
token. Used when `LLM_PROVIDER=vertex_internal`.

Exposes the same public surface as `StellarClient` and `VertexClient` so
every pipeline module that already calls `get_stellar()` keeps working:

    embed(texts) -> list[list[float]]
    embed_one(text) -> list[float]
    chat(model, messages, ...) -> (text, TokenUsage)
    chat_stream(model, messages, ...) -> AsyncIterator[str]

Token TTL refresh is handled inside `get_llm.VertexGenAI` (the `.client`
property rebuilds when stale), so this wrapper stays thin.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

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


# --------------------------------------------------------------------------- #
# Helpers — Vertex message ↔ OpenAI message shape
# --------------------------------------------------------------------------- #
def _build_contents(messages: list[dict[str, str]]) -> tuple[str, list[Any]]:
    """Translate OpenAI-style messages to (system_text, list[types.Content])."""
    from google.genai import types

    system_text = ""
    contents: list[Any] = []
    for msg in messages:
        role = msg.get("role", "user")
        text = str(msg.get("content", ""))
        if role == "system":
            system_text = text
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=text)]))
        else:
            contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
    return system_text, contents


def _usage_from_response(resp: Any) -> TokenUsage:
    """Pull token counts off a genai response.usage_metadata when present."""
    meta = getattr(resp, "usage_metadata", None)
    if not meta:
        return TokenUsage()
    return TokenUsage(
        prompt=int(getattr(meta, "prompt_token_count", 0) or 0),
        completion=int(getattr(meta, "candidates_token_count", 0) or 0),
    )


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class InternalVertexClient:
    """Reusable async wrapper. One instance per app lifetime."""

    def __init__(self) -> None:
        # Lazy import — only on first instantiation. get_llm.py validates env
        # vars at module-load time, so don't pay that cost at app import.
        import get_llm  # type: ignore
        self._gw = get_llm.VertexGenAI()

    # ---------- embeddings ----------
    async def embed(self, texts: str | list[str]) -> list[list[float]]:
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        loop = asyncio.get_running_loop()
        model = settings.internal_vertex_embedding_model

        def _call() -> Any:
            return self._gw.client.models.embed_content(model=model, contents=items)

        resp = await loop.run_in_executor(None, _call)
        out: list[list[float]] = []
        for e in resp.embeddings:
            out.append(list(getattr(e, "values", []) or []))
        return out

    async def embed_one(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0] if result else []

    # ---------- chat ----------
    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> tuple[str, TokenUsage]:
        from google.genai import types

        system_text, contents = _build_contents(messages)
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        if system_text:
            cfg.system_instruction = system_text

        loop = asyncio.get_running_loop()

        def _call() -> Any:
            return self._gw.client.models.generate_content(
                model=model, contents=contents, config=cfg,
            )

        resp = await loop.run_in_executor(None, _call)
        text = (getattr(resp, "text", "") or "").strip()
        usage = _usage_from_response(resp)
        record_usage(usage.prompt, usage.completion)
        return text, usage

    # ---------- chat_stream ----------
    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        from google.genai import types

        system_text, contents = _build_contents(messages)
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        if system_text:
            cfg.system_instruction = system_text

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        acc_ref = current_acc()
        final = {"prompt": 0, "completion": 0}

        def _produce() -> None:
            try:
                stream = self._gw.client.models.generate_content_stream(
                    model=model, contents=contents, config=cfg,
                )
                for chunk in stream:
                    piece = getattr(chunk, "text", None)
                    if piece:
                        asyncio.run_coroutine_threadsafe(queue.put(piece), loop)
                    meta = getattr(chunk, "usage_metadata", None)
                    if meta:
                        final["prompt"] = int(getattr(meta, "prompt_token_count", 0) or 0)
                        final["completion"] = int(getattr(meta, "candidates_token_count", 0) or 0)
            except Exception as exc:  # noqa: BLE001
                logger.exception("internal-vertex chat_stream failed")
                asyncio.run_coroutine_threadsafe(queue.put(f"\n[stream-error: {exc}]"), loop)
            finally:
                try:
                    if final["prompt"] or final["completion"]:
                        _add_to(acc_ref, final["prompt"], final["completion"])
                except Exception:
                    pass
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        loop.run_in_executor(None, _produce)
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item


# --------------------------------------------------------------------------- #
# Singleton accessor
# --------------------------------------------------------------------------- #
_client: InternalVertexClient | None = None


def get_internal_vertex() -> InternalVertexClient:
    global _client
    if _client is None:
        _client = InternalVertexClient()
    return _client
