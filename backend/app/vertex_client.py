"""
Vertex AI client used for *local* development and demos.

Why this exists:
  * The production path uses Stellar via get_llm.py (COIN-OAuth, VDI-only).
  * Outside the VDI we can't reach COIN, so we need a different LLM gateway.
  * Google Vertex AI accepts a service-account JSON for auth — perfect for
    local laptop testing.

This client implements the SAME public surface as StellarClient
(`embed`, `embed_one`, `chat`, `chat_stream`, `TokenUsage`) so every
pipeline module that already calls `get_stellar()` keeps working.

Auth strategies (in priority order):
  1. settings.google_application_credentials path → service_account.Credentials
  2. GOOGLE_APPLICATION_CREDENTIALS env var (picked up by google-genai SDK)
  3. Application Default Credentials (gcloud auth application-default login)

Like the Stellar client, the underlying genai.Client is rebuilt every
COIN_TOKEN_TTL_SECONDS (default 14 min) for parity, even though the
service-account JWT auto-refresh would technically make it unnecessary.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
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


_TTL_SECONDS = int(os.getenv("COIN_TOKEN_TTL_SECONDS", str(14 * 60)))


class VertexClient:
    """Provider-parallel sibling of StellarClient. Same public surface."""

    def __init__(self, ttl_seconds: int = _TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._client: Any = None
        self._client_created: float = 0.0
        self._build_client()

    # ---------- auth + client construction ----------
    def _build_client(self) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai not installed; pip install google-genai") from exc

        kwargs: dict[str, Any] = {"vertexai": True, "location": settings.vertex_location}
        if settings.vertex_project:
            kwargs["project"] = settings.vertex_project

        # Prefer explicit service-account JSON when provided
        creds_path = settings.google_application_credentials
        if creds_path and os.path.isfile(creds_path):
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            kwargs["credentials"] = credentials
            if not settings.vertex_project:
                # service-account JSON carries the project_id
                with open(creds_path, "r", encoding="utf-8") as f:
                    import json
                    data = json.load(f)
                    if data.get("project_id"):
                        kwargs["project"] = data["project_id"]

        self._client = genai.Client(**kwargs)
        self._client_created = time.monotonic()
        logger.info(
            "VertexClient built (project=%s, location=%s, TTL=%ds)",
            kwargs.get("project"), settings.vertex_location, self._ttl,
        )

    @property
    def client(self) -> Any:
        if self._client is None or (time.monotonic() - self._client_created) > self._ttl:
            self._build_client()
        return self._client

    # ---------- embeddings ----------
    async def embed(self, texts: str | list[str]) -> list[list[float]]:
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        loop = asyncio.get_running_loop()
        model = settings.vertex_embedding_model

        def _call() -> Any:
            return self.client.models.embed_content(model=model, contents=items)

        resp = await loop.run_in_executor(None, _call)
        out: list[list[float]] = []
        for e in resp.embeddings:
            # genai embedding objects expose `.values`
            out.append(list(getattr(e, "values", []) or []))
        return out

    async def embed_one(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0] if result else []

    # ---------- chat ----------
    def _build_contents(self, messages: list[dict[str, str]]) -> tuple[str, list[Any]]:
        """Translate OpenAI-style messages to Vertex Content list + system text."""
        from google.genai import types  # local import — avoids hard dep at module load
        system_text = ""
        contents: list[Any] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_text = (system_text + "\n\n" + content).strip() if system_text else content
            elif role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=content)]))
        return system_text, contents

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
        loop = asyncio.get_running_loop()
        system_text, contents = self._build_contents(messages)
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            # gemini-2.5-* spends "thinking" tokens against max_output_tokens
            # before emitting visible output — that starves short structured
            # responses. Disable thinking entirely; the prompt does the work.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        if system_text:
            cfg.system_instruction = system_text

        def _call() -> Any:
            return self.client.models.generate_content(model=model, contents=contents, config=cfg)

        resp = await loop.run_in_executor(None, _call)
        text = (getattr(resp, "text", None) or "").strip()
        usage = TokenUsage()
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            usage = TokenUsage(
                prompt=int(getattr(meta, "prompt_token_count", 0) or 0),
                completion=int(getattr(meta, "candidates_token_count", 0) or 0),
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
        from google.genai import types
        loop = asyncio.get_running_loop()
        system_text, contents = self._build_contents(messages)
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        if system_text:
            cfg.system_instruction = system_text

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        final_usage = {"prompt": 0, "completion": 0}
        # ContextVar isn't propagated to executor threads — capture the
        # accumulator reference here on the main task and mutate it directly.
        acc_ref = current_acc()

        def _produce() -> None:
            try:
                stream = self.client.models.generate_content_stream(
                    model=model, contents=contents, config=cfg,
                )
                for event in stream:
                    text = getattr(event, "text", None)
                    if text:
                        asyncio.run_coroutine_threadsafe(queue.put(text), loop)
                    meta = getattr(event, "usage_metadata", None)
                    if meta is not None:
                        # Gemini's stream emits running totals; the last
                        # event we see is the authoritative count.
                        final_usage["prompt"] = int(getattr(meta, "prompt_token_count", 0) or 0)
                        final_usage["completion"] = int(getattr(meta, "candidates_token_count", 0) or 0)
            except Exception as exc:  # noqa: BLE001
                logger.exception("vertex chat_stream failed")
                asyncio.run_coroutine_threadsafe(queue.put(f"\n[stream-error: {exc}]"), loop)
            finally:
                try:
                    if final_usage["prompt"] or final_usage["completion"]:
                        _add_to(acc_ref, final_usage["prompt"], final_usage["completion"])
                except Exception:
                    pass
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        loop.run_in_executor(None, _produce)
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item


_vertex_singleton: VertexClient | None = None


def get_vertex() -> VertexClient:
    global _vertex_singleton
    if _vertex_singleton is None:
        _vertex_singleton = VertexClient()
    return _vertex_singleton
