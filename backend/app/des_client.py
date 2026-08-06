"""Thin async HTTP client for document-enrichment-services (DES).

DES is a *separate* service that performs Azure Document Intelligence layout
OCR → structure-aware chunking → gte-large-en-v1.5 embeddings and writes the
resulting rows straight into the pgvector tables this service reads. So this
module deliberately does very little: it triggers a DES run and reports on it.
No chunks, no embeddings and no SQL cross this boundary.

Public API
----------
    await submit(file_bytes, filename, content_type) -> dict   # the 202 body
    await get_run(run_id)                            -> dict
    await get_events(run_id)                         -> list[dict]
    await health()                                   -> dict   # never raises

Everything except :func:`health` raises :class:`DesError` on any non-happy
answer, so callers have exactly one exception type to translate into an SSE
``error`` event.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger(__name__)

# Polling and readiness are cheap calls — they must not inherit the (very long)
# ingestion timeout, otherwise one wedged poll stalls the whole SSE stream.
_POLL_TIMEOUT = 30.0
_HEALTH_TIMEOUT = 5.0

# Keys under which a DES build might report the pgvector schema it writes into.
# Older builds report none of them, in which case the caller gets None and
# reports `schema_match: null` rather than guessing.
_SCHEMA_KEYS = ("vector_schema", "pg_vector_schema", "schema")


class DesError(RuntimeError):
    """DES was unreachable, or answered with something we cannot use."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _base_url() -> str:
    return settings.des_url.rstrip("/")


def _headers() -> dict[str, str]:
    """X-API-Key only when DES is configured with an API_KEY."""
    return {"X-API-Key": settings.des_api_key} if settings.des_api_key else {}


def _snippet(resp: Any, limit: int = 500) -> str:
    """Best-effort short body text for error messages."""
    try:
        return str(resp.text)[:limit]
    except Exception:  # noqa: BLE001 - error paths must not raise
        return ""


def _json_body(resp: Any, what: str) -> Any:
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001 - any decode failure is a DES failure
        raise DesError(
            f"DES {what} returned non-JSON body: {_snippet(resp, 200)!r}"
        ) from exc


async def submit(
    file_bytes: bytes,
    filename: str,
    content_type: str | None = None,
) -> dict[str, Any]:
    """POST the upload to DES and return its 202 acknowledgement.

    Args:
        file_bytes: Raw file bytes.
        filename: Original filename (DES stores it as the document name).
        content_type: MIME type; ``application/octet-stream`` when unknown.

    Returns:
        The 202 body: ``{"document_id", "run_id", "status", "name",
        "size_bytes", "sha256", "s3_uri"}``.

    Raises:
        DesError: On any transport failure, any status other than 202, or a
            body we cannot decode.
    """
    url = f"{_base_url()}/api/documents"
    files = {"file": (filename, file_bytes, content_type or "application/octet-stream")}
    try:
        async with httpx.AsyncClient(timeout=settings.des_timeout) as client:
            resp = await client.post(url, files=files, headers=_headers())
    except httpx.HTTPError as exc:
        raise DesError(f"DES submit failed ({url}): {exc}") from exc

    if resp.status_code != 202:
        body = _snippet(resp)
        raise DesError(
            f"DES POST /api/documents returned {resp.status_code} (expected 202): {body}",
            status_code=resp.status_code,
            body=body,
        )

    payload = _json_body(resp, "POST /api/documents")
    if not isinstance(payload, dict):
        raise DesError(f"DES POST /api/documents returned {type(payload).__name__}, expected object")
    return payload


async def get_run(run_id: str) -> dict[str, Any]:
    """Fetch one run's status/stages/counters.

    Raises:
        DesError: On transport failure, a non-200 status, or a non-object body.
    """
    url = f"{_base_url()}/api/runs/{run_id}"
    try:
        async with httpx.AsyncClient(timeout=_POLL_TIMEOUT) as client:
            resp = await client.get(url, headers=_headers())
    except httpx.HTTPError as exc:
        raise DesError(f"DES get_run failed ({url}): {exc}") from exc

    if resp.status_code != 200:
        body = _snippet(resp)
        raise DesError(
            f"DES GET /api/runs/{run_id} returned {resp.status_code}: {body}",
            status_code=resp.status_code,
            body=body,
        )

    payload = _json_body(resp, f"GET /api/runs/{run_id}")
    if not isinstance(payload, dict):
        raise DesError(f"DES GET /api/runs/{run_id} returned {type(payload).__name__}, expected object")
    return payload


async def get_events(run_id: str) -> list[dict[str, Any]]:
    """Fetch a run's ordered event log (``seq``-ascending).

    Raises:
        DesError: On transport failure or a non-200 status.
    """
    url = f"{_base_url()}/api/runs/{run_id}/events"
    try:
        async with httpx.AsyncClient(timeout=_POLL_TIMEOUT) as client:
            resp = await client.get(url, headers=_headers())
    except httpx.HTTPError as exc:
        raise DesError(f"DES get_events failed ({url}): {exc}") from exc

    if resp.status_code != 200:
        body = _snippet(resp)
        raise DesError(
            f"DES GET /api/runs/{run_id}/events returned {resp.status_code}: {body}",
            status_code=resp.status_code,
            body=body,
        )

    payload = _json_body(resp, f"GET /api/runs/{run_id}/events")
    events = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        return []
    return [e for e in events if isinstance(e, dict)]


def extract_vector_schema(body: Any) -> str | None:
    """Pull the pgvector schema DES writes into out of a /api/readyz body.

    Checked at the top level and inside ``db`` / ``embedding``; returns None
    when this DES build does not report it (older builds do not), which the
    caller must render as "unknown", never as "matches".
    """
    if not isinstance(body, dict):
        return None
    candidates: list[Any] = [body]
    for nested in ("db", "embedding", "vector", "postgres"):
        section = body.get(nested)
        if isinstance(section, dict):
            candidates.append(section)
    for section in candidates:
        for key in _SCHEMA_KEYS:
            value = section.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


async def health() -> dict[str, Any]:
    """Probe DES readiness. Never raises — failures come back as ``ok: False``.

    Returns:
        ``{"ok", "status_code", "vector_schema", "body", "error"}``.
        ``vector_schema`` is None when DES does not report one.
    """
    url = f"{_base_url()}/api/readyz"
    out: dict[str, Any] = {
        "ok": False,
        "status_code": None,
        "vector_schema": None,
        "body": None,
        "error": None,
    }
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT) as client:
            resp = await client.get(url, headers=_headers())
    except Exception as exc:  # noqa: BLE001 - health must never raise
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    out["status_code"] = getattr(resp, "status_code", None)
    body: Any = None
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 - a non-JSON body is just an unready DES
        body = None
    out["body"] = body if isinstance(body, dict) else None
    out["vector_schema"] = extract_vector_schema(body)
    ready = body.get("ready") if isinstance(body, dict) else None
    out["ok"] = out["status_code"] == 200 and (ready is not False)
    if not out["ok"] and out["error"] is None:
        out["error"] = f"DES /api/readyz status={out['status_code']} ready={ready}"
    return out
