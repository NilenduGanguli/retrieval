"""Settings for the remote ingestion service (env-var driven)."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RemoteIngestSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- Server ----------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8090)
    log_level: str = Field(default="INFO")

    # Auth: optional shared secret the calling backend must present
    shared_secret: str = Field(default="")

    # ---------- Provider selection ----------
    # "wega"   → run the internal WegaChunker + StellarGenAI embeddings
    # "vertex" → pypdf-based chunking + Google Vertex AI embeddings (local test)
    llm_provider: str = Field(default="wega")

    # ---------- WEGA chunker (used when llm_provider="wega") ----------
    azure_di_endpoint: str = Field(default="http://sd-jibs-35nc.nam.nsroot.net:5000")
    azure_di_key: str = Field(default="")
    chunk_token_size: int = Field(default=500)
    document_type: str = Field(default="General")
    detect_pii: bool = Field(default=True)
    extract_images: bool = Field(default=False)

    # ---------- Vertex (used when llm_provider="vertex") ----------
    google_application_credentials: str = Field(default="")
    vertex_project: str = Field(default="")
    vertex_location: str = Field(default="us-central1")
    vertex_embedding_model: str = Field(default="text-embedding-005")

    # Local pypdf chunker tuning
    local_chunk_max_chars: int = Field(default=1500)
    local_chunk_overlap: int = Field(default=150)

    # ---------- Postgres / pgvector (where chunks are stored) ----------
    pg_host: str = Field(default="KYC164283DEV.pgaas.dyn.nsroot.net")
    pg_port: int = Field(default=1524)
    pg_user: str = Field(default="kyc164283devpgaas")
    pg_password: str = Field(default="")
    pg_database: str = Field(default="chunker_db")
    pg_index: str = Field(default="chunk_embeddings")
    pg_app_owner_role: str = Field(default="citi_pg_app_owner")

    # ---------- Embedding ----------
    embedding_batch_size: int = Field(default=16)

    # ---------- S3 / MinIO — source-PDF persistence ----------
    # When enabled, the service uploads every ingested PDF to S3 and
    # UPSERTs the documents row, so the source is recoverable for
    # View/Download even when the remote service is deployed standalone.
    # Turn this OFF when the main backend already uploads (the usual
    # "backend in front" topology) to avoid double-upload.
    s3_enabled: bool = Field(default=False)
    s3_endpoint_url: str = Field(default="")
    s3_region: str = Field(default="us-east-1")
    s3_bucket: str = Field(default="rag-uploads")
    s3_access_key: str = Field(default="")
    s3_secret_key: str = Field(default="")
    s3_use_ssl: bool = Field(default=False)

    # ---------- Working dir ----------
    upload_dir: str = Field(default="/tmp/remote_ingest_uploads")


settings = RemoteIngestSettings()
