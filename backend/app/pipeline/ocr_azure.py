"""
OCR helper for KYC ingestion.

Primary path: Azure Document Intelligence (Form Recognizer) — used when
AZURE_DI_ENDPOINT + AZURE_DI_KEY are configured. Handles complex layouts,
multi-column docs, tables, and scanned PDFs.

Fallback: pypdf (text-layer only) — used in local dev or when Azure DI
is unreachable. Quality drops on scanned PDFs but is fine for clean
digital PDFs while you're testing without VDI access.

Public API:
    extract_pages(pdf_bytes, *, filename="") -> list[dict]
        Returns: [{"page_number": int, "text": str}, ...]
"""
from __future__ import annotations

import io
import logging
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback — pypdf
# ---------------------------------------------------------------------------
def _extract_pages_pypdf(pdf_bytes: bytes) -> list[dict]:
    """Text-layer extraction. Bad on scanned PDFs, fine on digital ones."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf not installed; pip install pypdf") from exc

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: list[dict] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("pypdf failed on page %d: %s", i, exc)
            text = ""
        pages.append({"page_number": i, "text": text})
    return pages


# ---------------------------------------------------------------------------
# Primary — Azure Document Intelligence (prebuilt-read model)
# ---------------------------------------------------------------------------
def _extract_pages_azure_di(pdf_bytes: bytes) -> list[dict]:
    """Use Azure Document Intelligence's prebuilt-read model.

    Requires settings.azure_di_endpoint + settings.azure_di_key.
    Returns per-page text in the same shape as the pypdf fallback.
    """
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError as exc:
        raise RuntimeError(
            "azure-ai-documentintelligence not installed; "
            "pip install azure-ai-documentintelligence"
        ) from exc

    client = DocumentIntelligenceClient(
        endpoint=settings.azure_di_endpoint,
        credential=AzureKeyCredential(settings.azure_di_key),
    )
    poller = client.begin_analyze_document("prebuilt-read", body=pdf_bytes)
    result = poller.result()

    pages: list[dict] = []
    for p in result.pages or []:
        lines = [getattr(line, "content", "") for line in (p.lines or [])]
        text = "\n".join(filter(None, lines))
        pages.append({
            "page_number": getattr(p, "page_number", len(pages) + 1),
            "text": text,
        })
    return pages


# ---------------------------------------------------------------------------
# Public entry — picks Azure DI when configured, falls back to pypdf
# ---------------------------------------------------------------------------
def extract_pages(pdf_bytes: bytes, *, filename: str = "") -> list[dict]:
    """Best-effort per-page text extraction.

    Tries Azure DI first when configured; on any failure (or when not
    configured) falls back to pypdf. Always returns at least an empty
    list — never raises.
    """
    if settings.azure_di_endpoint and settings.azure_di_key:
        try:
            pages = _extract_pages_azure_di(pdf_bytes)
            if pages:
                logger.info("OCR (Azure DI) %s: %d pages", filename or "<bytes>", len(pages))
                return pages
            logger.warning("Azure DI returned 0 pages for %s — falling back to pypdf", filename)
        except Exception as exc:
            logger.warning(
                "Azure DI extraction failed for %s (%s) — falling back to pypdf",
                filename, exc,
            )

    try:
        pages = _extract_pages_pypdf(pdf_bytes)
        logger.info("OCR (pypdf fallback) %s: %d pages", filename or "<bytes>", len(pages))
        return pages
    except Exception as exc:
        logger.exception("pypdf fallback also failed for %s", filename)
        return []
