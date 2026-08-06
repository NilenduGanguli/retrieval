"""Fleet-wide embeddings over HTTP (gte-large-en-v1.5, 1024-D).

Why this exists
---------------
The index is embedded with ``settings.embedding_model`` (gte-large-en-v1.5, 1024-D) —
that is the cross-service contract shared with document-enrichment-services. The *LLM*
provider is a separate choice: ``llm_provider=vertex`` is a perfectly good local setup for
generation and reranking, but Vertex does not serve gte, so its embedding model
(``text-embedding-005``) is 768-D.

Left alone, those two facts collide at query time: the pipeline embedded the query with the
provider's model and compared it against the 1024-D column, and Postgres answered

    asyncpg.exceptions.DataError: different vector dimensions 1024 and 768

so retrieval could not read its own index. Embeddings therefore have to be resolved
independently of the LLM provider: whenever ``embedding_base_url`` is configured, every
embedding — query or ingest — goes through that endpoint, and the provider client is used
only for chat/generation.

Wire formats
------------
``openai`` (default): ``POST {base}/v1/embeddings`` ``{"model", "input": [...]}`` ->
``data[].embedding``. Works with Stellar, vLLM, and anything OpenAI-compatible.
``tei``: ``POST {base}/embed`` ``{"inputs": [...]}`` -> a bare list of vectors.
"""
from __future__ import annotations

import logging

import httpx

from .config import settings

logger = logging.getLogger(__name__)

_TIMEOUT = 120.0


class EmbeddingError(RuntimeError):
    """The embedding endpoint was unreachable, or returned something unusable."""


def http_embeddings_enabled() -> bool:
    """True when a fleet embedding endpoint is configured and should take precedence."""
    return bool(settings.embedding_base_url)


def _headers() -> dict[str, str]:
    key = getattr(settings, "embedding_api_key", "") or ""
    return {"Authorization": f"Bearer {key}"} if key else {}


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed ``texts`` with the configured fleet endpoint, preserving input order.

    Args:
        texts: Strings to embed.

    Returns:
        One vector per input text, in order.

    Raises:
        EmbeddingError: on transport failure, a non-2xx reply, an unparseable body, or a
            dimension that disagrees with ``settings.embedding_dim`` — a silent dimension
            mismatch corrupts the index rather than failing, so it is raised loudly here.
    """
    if not texts:
        return []
    base = settings.embedding_base_url.rstrip("/")
    style = (getattr(settings, "embedding_api_style", "openai") or "openai").lower()
    if style == "tei":
        url, payload, pick = f"{base}/embed", {"inputs": texts}, None
    else:
        url = f"{base}/v1/embeddings"
        payload = {"model": settings.embedding_model, "input": texts}
        pick = "data"

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=_headers())
            resp.raise_for_status()
            body = resp.json()
    except httpx.HTTPError as exc:
        raise EmbeddingError(f"embedding endpoint {url} failed: {exc}") from exc

    if pick is None:
        vectors = body if isinstance(body, list) else body.get("embeddings")
    else:
        rows = (body or {}).get(pick)
        if not isinstance(rows, list):
            raise EmbeddingError(f"embedding endpoint {url} returned no '{pick}' array")
        vectors = [r.get("embedding") for r in rows]

    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise EmbeddingError(
            f"embedding endpoint {url} returned {len(vectors or [])} vectors "
            f"for {len(texts)} inputs"
        )
    out = [[float(x) for x in v] for v in vectors]
    expected = settings.embedding_dim
    if out and expected and len(out[0]) != expected:
        raise EmbeddingError(
            f"embedding dimension mismatch: {settings.embedding_model} via {url} returned "
            f"{len(out[0])}-D but EMBEDDING_DIM={expected}. The index and the query "
            "embedder must agree or every search fails."
        )
    return out


async def embed_one(text: str) -> list[float]:
    """Embed a single string; see :func:`embed_texts`."""
    return (await embed_texts([text]))[0]
