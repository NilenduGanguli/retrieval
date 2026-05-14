"""
Async S3-compatible object store wrapper.

Defaults to a local MinIO instance (http://localhost:9000) using
the well-known minioadmin/minioadmin credentials. Production switches
to real AWS S3 by clearing S3_ENDPOINT_URL and supplying real keys.

Public API:
  await s3_put(key, body, content_type)  -> uri (e.g. s3://bucket/key)
  await s3_get(key)                       -> bytes
  await s3_delete(key)                    -> None
  await s3_ensure_bucket()                -> bool
"""
from __future__ import annotations

import logging
from typing import Any

import aioboto3
from botocore.config import Config as BotoConfig

from .config import settings

logger = logging.getLogger(__name__)


def _client_kwargs() -> dict[str, Any]:
    kw: dict[str, Any] = {
        "region_name": settings.s3_region,
        "aws_access_key_id": settings.s3_access_key,
        "aws_secret_access_key": settings.s3_secret_key,
        "config": BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 3, "mode": "standard"},
        ),
        "use_ssl": settings.s3_use_ssl,
    }
    if settings.s3_endpoint_url:
        kw["endpoint_url"] = settings.s3_endpoint_url
    return kw


_session: aioboto3.Session | None = None


def _session_instance() -> aioboto3.Session:
    global _session
    if _session is None:
        _session = aioboto3.Session()
    return _session


async def s3_ensure_bucket() -> bool:
    """Create the configured bucket if it doesn't exist. Idempotent."""
    if not settings.s3_enabled:
        return False
    async with _session_instance().client("s3", **_client_kwargs()) as s3:
        try:
            await s3.head_bucket(Bucket=settings.s3_bucket)
            return True
        except Exception:
            pass
        try:
            if settings.s3_region == "us-east-1":
                await s3.create_bucket(Bucket=settings.s3_bucket)
            else:
                await s3.create_bucket(
                    Bucket=settings.s3_bucket,
                    CreateBucketConfiguration={"LocationConstraint": settings.s3_region},
                )
            logger.info("created S3 bucket %s", settings.s3_bucket)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("ensure_bucket failed: %s", exc)
            return False


async def s3_put(key: str, body: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload bytes; return the s3 URI."""
    if not settings.s3_enabled:
        raise RuntimeError("S3 disabled — set S3_ENABLED=true")
    async with _session_instance().client("s3", **_client_kwargs()) as s3:
        await s3.put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
        )
    uri = f"s3://{settings.s3_bucket}/{key}"
    logger.info("s3 put %s (%d bytes)", uri, len(body))
    return uri


async def s3_get(key: str) -> bytes:
    async with _session_instance().client("s3", **_client_kwargs()) as s3:
        resp = await s3.get_object(Bucket=settings.s3_bucket, Key=key)
        async with resp["Body"] as stream:
            return await stream.read()


async def s3_delete(key: str) -> None:
    if not settings.s3_enabled:
        return
    async with _session_instance().client("s3", **_client_kwargs()) as s3:
        await s3.delete_object(Bucket=settings.s3_bucket, Key=key)


async def s3_list_prefix(prefix: str = "") -> list[dict]:
    async with _session_instance().client("s3", **_client_kwargs()) as s3:
        out: list[dict] = []
        paginator = s3.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                out.append({"key": obj["Key"], "size": obj["Size"], "modified": obj["LastModified"].isoformat()})
        return out
