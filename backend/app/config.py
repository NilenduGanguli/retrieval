"""Centralised settings — all env-var driven, never hard-coded secrets."""
from __future__ import annotations

from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- Postgres ----------
    pg_host: str = Field(default="localhost")
    pg_port: int = Field(default=5432)
    pg_user: str = Field(default="")
    pg_password: str = Field(default="")
    pg_database: str = Field(default="chunker_db")
    pg_schema: str = Field(default="vector_ng12499")
    pg_table: str = Field(default="chunk_embeddings")
    app_owner_role: str = Field(default="")

    # ---------- LLM provider selection ----------
    # "vertex_internal" → production: Vertex via the internal corporate
    #                     proxy (VERTEX_BASE_URL) using a COIN token,
    #                     wrapped by get_llm.VertexGenAI. Use this on the VDI.
    # "vertex"          → local dev: google-genai with a service-account JSON,
    #                     talks directly to Google's Vertex API.
    # "stellar"         → legacy: OpenAI-compatible Stellar (Llama/Mistral).
    llm_provider: str = Field(default="vertex_internal")

    # ---------- Stellar models (legacy — only used when llm_provider="stellar") ----------
    stellar_embedding_model: str = Field(default="gte-large-en-v1.5")
    stellar_final_gen_model: str = Field(default="Llama-4-Scout-17B-16E-Instruct")
    stellar_rerank_model: str = Field(default="Meta-Llama-3.3-70B-Instruct")
    stellar_fast_model: str = Field(default="Meta-Llama-3.1-8B-Instruct")
    stellar_contextual_model: str = Field(default="Llama-4-Scout-17B-16E-Instruct")

    # ---------- Vertex (local dev, service-account JSON) ----------
    google_application_credentials: str = Field(default="")
    vertex_project: str = Field(default="")
    vertex_location: str = Field(default="us-central1")
    vertex_embedding_model: str = Field(default="text-embedding-005")
    vertex_final_gen_model: str = Field(default="gemini-2.5-flash")
    vertex_rerank_model: str = Field(default="gemini-2.5-flash")
    vertex_fast_model: str = Field(default="gemini-2.5-flash")
    vertex_contextual_model: str = Field(default="gemini-2.5-flash")

    # ---------- Internal-Vertex (production, COIN-token via get_llm.VertexGenAI) ----------
    # Everything defaults to gemini-2.0-flash-001 — small, fast, supports
    # structured outputs + tool calling, plenty for retrieval orchestration.
    # Override individual roles in .env if you want a heavier model for the
    # final answer.
    internal_vertex_embedding_model: str = Field(default="text-embedding-005")
    internal_vertex_final_gen_model: str = Field(default="gemini-2.0-flash-001")
    internal_vertex_rerank_model: str = Field(default="gemini-2.0-flash-001")
    internal_vertex_fast_model: str = Field(default="gemini-2.0-flash-001")
    internal_vertex_contextual_model: str = Field(default="gemini-2.0-flash-001")

    # ---------- Retrieval defaults ----------
    rag_topk_dense: int = 50
    rag_topk_sparse: int = 50
    rag_rrf_k: int = 60
    rag_rerank_topn: int = 20
    rag_final_k: int = 8
    rag_mmr_lambda: float = 0.7

    rag_hyde_default: bool = False
    rag_rewrite_default: bool = False
    rag_rerank_default: bool = True
    rag_crag_default: bool = False
    rag_contextual_default: bool = True

    # ---------- Remote ingestion service ----------
    # When True, /api/ingest/wega proxies the upload to a separate FastAPI
    # service (typically running inside the VDI alongside the WEGA wheel).
    remote_ingest: bool = False
    remote_ingest_url: str = "http://localhost:8090"
    remote_ingest_secret: str = ""
    remote_ingest_timeout: int = 1800  # seconds; ingestion can be slow

    # ---------- Object storage (uploaded source PDFs) ----------
    s3_enabled: bool = True
    s3_endpoint_url: str = "http://localhost:9000"       # MinIO locally; blank = AWS
    s3_region: str = "us-east-1"
    s3_bucket: str = "rag-uploads"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_use_ssl: bool = False

    # ---------- Server ----------
    app_host: str = "0.0.0.0"
    app_port: int = 8080
    app_cors_origins: str = "http://localhost:5173,http://localhost:8080"
    app_log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.app_cors_origins.split(",") if o.strip()]

    @property
    def pg_dsn(self) -> str:
        # asyncpg DSN; password is intentionally not surfaced in repr/logs
        return (
            f"postgres://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def fq_table(self) -> str:
        return f'"{self.pg_schema}"."{self.pg_table}"'


settings = Settings()
