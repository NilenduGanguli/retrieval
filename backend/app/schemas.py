"""Pydantic models for the request/response layer."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================================================
# Strategy flags — these toggle every cutting-edge stage of the pipeline.
# Defaults are server-side (config), client can override per-query.
# ============================================================================
class Strategy(BaseModel):
    rewrite: bool | None = None
    hyde: bool | None = None
    use_contextual: bool | None = None  # use chunk_context embeddings if available
    hybrid: bool = True                  # dense+sparse always on; flip to False for dense-only
    rerank: bool | None = None
    mmr: bool = True
    crag: bool | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


# ============================================================================
# Retrieve (search-only, no generation)
# ============================================================================
class RetrieveRequest(BaseModel):
    query: str
    strategy: Strategy = Strategy()


class ChunkHit(BaseModel):
    chunk_id: int
    document_name: str | None = None
    page_number: int | None = None
    content: str
    context: str | None = None
    score: float
    score_breakdown: dict[str, float] = {}
    rank: int


class LatencyMs(BaseModel):
    rewrite: float = 0.0
    hyde: float = 0.0
    embed: float = 0.0
    dense: float = 0.0
    sparse: float = 0.0
    fuse: float = 0.0
    rerank: float = 0.0
    mmr: float = 0.0
    crag: float = 0.0
    generate: float = 0.0
    total: float = 0.0


class RetrieveResponse(BaseModel):
    query: str
    hits: list[ChunkHit]
    latency_ms: LatencyMs
    strategy_used: dict[str, Any]
    crag_confidence: float | None = None
    rewritten_queries: list[str] = []
    hyde_text: str | None = None
    stage_tokens: dict[str, dict[str, int]] = {}


# ============================================================================
# Chat (RAG with generation) — request schema same, response is streamed.
# ============================================================================
class ChatRequest(BaseModel):
    query: str
    history: list[dict[str, str]] = []  # [{role: "user"|"assistant", content: "..."}]
    strategy: Strategy = Strategy()


# ============================================================================
# Documents
# ============================================================================
class DocumentSummary(BaseModel):
    document_name: str
    chunk_count: int
    total_tokens: int | None = None
    first_page: int | None = None
    last_page: int | None = None
    latest_job_id: str | None = None
    contextual_coverage: float | None = None  # 0..1


# ============================================================================
# Analytics
# ============================================================================
class AnalyticsSummary(BaseModel):
    total_queries: int
    queries_24h: int
    avg_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    avg_tokens: float | None = None
    top_documents: list[dict[str, Any]] = []
    strategy_mix: dict[str, int] = {}
    token_totals: dict[str, int] = {}
    stage_token_breakdown: list[dict[str, Any]] = []


class QueryLogEntry(BaseModel):
    id: int
    query_text: str
    strategy: dict[str, Any]
    latency_ms: dict[str, Any]
    top_chunk_ids: list[int]
    answer_text: str | None = None
    crag_confidence: float | None = None
    created_at: str


# ============================================================================
# Ingestion
# ============================================================================
class IngestionEvent(BaseModel):
    type: Literal["info", "chunk", "embedding", "context", "upsert", "error", "done"]
    message: str = ""
    progress: float | None = None  # 0..1
    payload: dict[str, Any] = {}


# ============================================================================
# Benchmark / golden-set
# ============================================================================
class GoldenQuestion(BaseModel):
    id: int | None = None
    question: str
    ground_truth_chunk_ids: list[int] = []
    ground_truth_answer: str | None = None
    tags: list[str] = []


class BenchRunRequest(BaseModel):
    question_ids: list[int] | None = None  # if None: run all
    label: str | None = None
    strategy: Strategy = Strategy()


class BenchRunResult(BaseModel):
    run_id: int
    label: str | None
    n_questions: int
    metrics: dict[str, float]
    per_question: list[dict[str, Any]]
