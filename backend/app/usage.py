"""
Token-usage tracking for LLM calls — cross-cutting.

The pattern: each request enters a `UsageScope` context, every chat call
auto-records its prompt + completion counts into the active accumulator
via `record_usage(...)`. Pipeline stages can `snapshot()` the accumulator
before/after a stage to get per-stage deltas.

ContextVar isolation means concurrent requests don't pollute each other,
and stages running under asyncio.gather inside the same request share
the same accumulator (which is what we want).
"""
from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TokenCounts:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return int(self.prompt + self.completion)

    def as_dict(self) -> dict[str, int]:
        return {"prompt": int(self.prompt), "completion": int(self.completion), "total": self.total}


_USAGE: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "rag_token_usage", default=None
)


def record_usage(prompt: int, completion: int) -> None:
    """Add a (prompt, completion) delta to the active accumulator (no-op if none)."""
    acc = _USAGE.get()
    if acc is None:
        return
    _add_to(acc, prompt, completion)


def _add_to(acc: dict | None, prompt: int, completion: int) -> None:
    """Same as record_usage, but writes to an explicitly-given accumulator
    reference. Use when crossing into a thread executor where ContextVar
    propagation is not guaranteed (capture `acc = current_acc()` before
    `loop.run_in_executor`, then call `_add_to(acc, ...)` inside the thread)."""
    if acc is None:
        return
    try:
        acc["prompt"] += int(prompt or 0)
        acc["completion"] += int(completion or 0)
    except Exception:
        logger.exception("_add_to failed")


def current_acc() -> dict | None:
    """Return the *live* accumulator reference (or None). Cross-thread safe
    if the caller mutates the returned dict atomically enough."""
    return _USAGE.get()


def snapshot() -> dict[str, int]:
    """Return a *copy* of the current accumulator (zeros if no active scope)."""
    acc = _USAGE.get()
    if acc is None:
        return {"prompt": 0, "completion": 0, "total": 0}
    return {
        "prompt": int(acc.get("prompt", 0)),
        "completion": int(acc.get("completion", 0)),
        "total": int(acc.get("prompt", 0)) + int(acc.get("completion", 0)),
    }


class UsageScope:
    """
    Context manager that installs a fresh accumulator on entry and lets
    callers read the totals after the block exits.

        with UsageScope() as scope:
            ...
            scope.snapshot()    # {prompt: int, completion: int, total: int}
    """

    def __init__(self) -> None:
        self.acc: dict[str, int] = {"prompt": 0, "completion": 0}
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "UsageScope":
        self._token = _USAGE.set(self.acc)
        return self

    def __exit__(self, *_args) -> None:
        if self._token is not None:
            _USAGE.reset(self._token)
            self._token = None

    def snapshot(self) -> dict[str, int]:
        return {
            "prompt": int(self.acc["prompt"]),
            "completion": int(self.acc["completion"]),
            "total": int(self.acc["prompt"]) + int(self.acc["completion"]),
        }


def delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Return after - before (clamped to zero) — useful for per-stage tokens."""
    p = max(0, int(after.get("prompt", 0)) - int(before.get("prompt", 0)))
    c = max(0, int(after.get("completion", 0)) - int(before.get("completion", 0)))
    return {"prompt": p, "completion": c, "total": p + c}
