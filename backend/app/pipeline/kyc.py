"""
KYC ingestion + search pipeline.

A specialised flow on top of the generic retrieval stack:

    upload PDF
        ↓
    OCR (Azure DI primary, pypdf fallback)
        ↓
    PASS 1 — classify document type + extract owner
        ↓
    PASS 2 — type-specific structured-field extraction
        ↓
    chunk + embed + persist to kyc_documents / kyc_chunks

Plus three search modes:

  * list_by_owner(owner)                   metadata-only listing
  * extract_for_owner_type(owner, doc_type) vector search + LLM extraction
  * universal_search(keyword)              metadata scan + vector + LLM

Reuses settings.pg_schema, get_stellar() for chat + embed, S3 helpers
for source-PDF persistence, and asyncpg pool for storage.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..config import settings
from ..db import acquire, vec_to_pg
from ..s3_store import s3_put
from ..stellar_client import get_stellar, model_for
from .ocr_azure import extract_pages

logger = logging.getLogger(__name__)

ProgressCb = Callable[[dict[str, Any]], Awaitable[None]] | None


# ============================================================
# DOCUMENT TYPE TAXONOMY
# ============================================================
DOC_TYPE_TAXONOMY: dict[str, list[str]] = {
    "KYC Intelligence Reports": [
        "Orbis KYC Report",
        "D&B Market Intelligence Report",
        "LexisNexis AML Insight Report",
    ],
    "Ownership & Beneficial Ownership": [
        "Ownership Structure Report",
        "Beneficial Ownership Declaration",
        "General Information Sheet (GIS)",
    ],
    "Corporate Formation": [
        "Certificate of Incorporation",
        "Articles of Continuance",
        "Articles of Amendment",
        "Certificate of Registration",
        "Memorandum of Association",
        "Articles of Association",
    ],
    "Corporate Governance": [
        "Banking Resolution",
        "Board Resolution",
        "Power of Attorney",
        "Notarial Certificate",
        "Resignation Letter",
        "Document Lodgement Form",
        "Signature Card",
    ],
    "Banking & Financial": [
        "Bank Statement",
    ],
    "Personal Identity": [
        "Passport",
        "Aadhaar Card",
        "PAN Card",
        "Driver's License",
        "Voter ID",
        "National ID",
    ],
    "Financial Records": [
        "Salary Slip",
        "Utility Bill",
        "Insurance Document",
        "Tax Return",
    ],
    "Legal": [
        "Agreement / Contract",
        "Loan Agreement",
        "Guarantee Document",
    ],
}

ALL_DOC_TYPES: list[str] = [
    dt for types in DOC_TYPE_TAXONOMY.values() for dt in types
]


# ============================================================
# Owner normalization
# ============================================================
def normalize_name(name: str) -> str:
    """Return the first whitespace-separated token of a normalized name."""
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    parts = name.split()
    return parts[0] if parts else ""


def normalize_name_full(name: str) -> str:
    """Lowercase + strip punctuation; keep whitespace."""
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return name.strip()


def _sanitize_value(value: Any, max_len: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        clean = re.sub(r"<[^>]+>", "", value).strip()
        return clean[:max_len]
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False)[:max_len]
        except (TypeError, ValueError):
            return str(value)[:max_len]
    return str(value)[:max_len]


def _parse_confidence(raw: Any) -> float:
    try:
        if isinstance(raw, str):
            raw = re.sub(r"<[^>]+>", "", raw).strip()
        val = float(raw) if raw is not None else 0.5
        return max(0.0, min(1.0, val))
    except (TypeError, ValueError):
        return 0.5


def extract_json_from_response(response: str) -> dict | None:
    """Tolerant JSON extraction from LLM output."""
    if not response:
        return None
    response = re.sub(r"```(?:json)?", "", response).strip().replace("```", "").strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    start = response.find("{")
    if start == -1:
        return None
    brace = 0
    end = -1
    for i, ch in enumerate(response[start:], start):
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0:
                end = i + 1
                break
    if end == -1:
        return None
    try:
        return json.loads(response[start:end])
    except json.JSONDecodeError:
        return None


# ============================================================
# Prompts (ported from rag_streamlit.py)
# ============================================================
def build_classification_prompt(text: str, filename: str = "") -> str:
    truncated = text[:5000]
    taxonomy_str = "\n".join(
        f"  [{cat}]: {', '.join(types)}"
        for cat, types in DOC_TYPE_TAXONOMY.items()
    )
    return f"""You are an expert KYC document analyst. Classify this document and extract its owner and key metadata.

FILENAME: {filename}

=== DOCUMENT TYPE TAXONOMY ===
{taxonomy_str}

=== CLASSIFICATION RULES (apply in strict priority order) ===

PRIORITY 1 — KYC INTELLIGENCE REPORTS:
- Contains "Moody's API" AND "orbis" AND "COMPANY SUMMARY" → "Orbis KYC Report"
- Contains "LexisNexis" AND "D&B Global Market Identifiers" AND "Worldbase" → "D&B Market Intelligence Report"
- Contains "LexisNexis" AND "AML Insight" (without D&B) → "LexisNexis AML Insight Report"

PRIORITY 2 — CORPORATE FORMATION:
- Contains "CERTIFICATE OF INCORPORATION" or "ARTICLES OF INCORPORATION" → "Certificate of Incorporation"
- Contains "ARTICLES OF CONTINUANCE" → "Articles of Continuance"
- Contains "ARTICLES OF AMENDMENT" → "Articles of Amendment"
- Contains "CERTIFICATE OF REGISTRATION" → "Certificate of Registration"
- Contains "MEMORANDUM OF ASSOCIATION" → "Memorandum of Association"

PRIORITY 3 — CORPORATE GOVERNANCE:
- Contains "RESOLUTION AS TO BANKING" → "Banking Resolution"
- Contains "BOARD RESOLUTION" → "Board Resolution"
- Contains "POWER OF ATTORNEY" → "Power of Attorney"
- Contains "NOTARIAL CERTIFICATE" or "NOTARY PUBLIC" → "Notarial Certificate"
- Contains "RESIGNATION" + officer context → "Resignation Letter"
- Contains "DOCUMENT LODGEMENT" → "Document Lodgement Form"
- Contains "SIGNATURE CARD" or "AUTHORIZED SIGNATURES" with bank account → "Signature Card"

PRIORITY 4 — KYC/OWNERSHIP:
- Contains "CONTROLLING SHAREHOLDERS" or "BENEFICIAL OWNER" (NOT Orbis) → "Ownership Structure Report"
- Contains "General Information Sheet" or "GIS" as form title → "General Information Sheet (GIS)"

PRIORITY 5 — BANKING:
- Contains "BANK STATEMENT" or transaction list → "Bank Statement"

PRIORITY 6 — PERSONAL IDENTITY:
- "Aadhaar" → "Aadhaar Card"
- "PAN" + income tax → "PAN Card"
- "PASSPORT" → "Passport"
- "DRIVING LICENCE" → "Driver's License"
- "VOTER" + "ID" → "Voter ID"
- "NATIONAL IDENTITY CARD" → "National ID"

PRIORITY 7 — FINANCIAL RECORDS:
- salary/payslip/earnings/net pay → "Salary Slip"
- utility/electricity/water/gas bill → "Utility Bill"
- insurance policy/premium → "Insurance Document"

=== OWNER EXTRACTION RULES ===
- "Orbis KYC Report" → owner = "Inquiry Name" field (SUBJECT company)
- "D&B" / "LexisNexis AML" → owner = bold company at top of Worldbase section
- "Certificate of Incorporation" / "Articles of *" → owner = COMPANY being incorporated
- "Banking Resolution" / "Signature Card" / "Power of Attorney" → COMPANY on the account
- "Document Lodgement Form" → COMPANY in "Full Legal Entity Name"
- "Ownership Structure" / "GIS" → COMPANY in form header
- "Bank Statement" → account holder name
- Personal ID docs → person's FULL NAME as printed
- "Resignation Letter" → COMPANY receiving the resignation
- "Notarial Certificate" → COMPANY being certified

Return ONLY this JSON (no markdown):
{{
  "document_type": "<exact type from taxonomy>",
  "document_category": "<category from taxonomy>",
  "owner": "<company or person name — exact as appears>",
  "confidence_score": <0.0-1.0>,
  "classification_signals": ["<signal1>", "<signal2>"],
  "source_platform": "<e.g. Moody's Orbis, LexisNexis, D&B Worldbase, or empty>",
  "report_date": "<delivery/report date if found, else empty>"
}}

DOCUMENT TEXT:
{truncated}"""


def build_extraction_prompt(doc_type: str, text: str, filename: str) -> str:
    truncated = text[:6000]
    t = doc_type.lower()

    if "orbis" in t:
        fields = (
            "inquiry_name, gfcid, delivery_date, update_date, company_name, status, "
            "legal_form, national_legal_form, bvd_id, orbis_id, activity, operating_revenue, "
            "num_employees, date_of_incorporation, owners_count, global_ultimate_owner, "
            "address, city, country, postcode, province_state, phone, email, website, "
            "type_of_entity, bvd_sector, nace_main_section, nace_core_code, naics_core_code, "
            "us_sic_core_code, lei, isin, ticker, swift_code, directors_list, "
            "controlling_shareholders_list, beneficial_owners_list, corporate_group_list"
        )
    elif any(x in t for x in ["d&b", "lexisnexis", "aml", "worldbase", "market intelligence"]):
        fields = (
            "report_source, report_date, document_number, search_terms, company_name, "
            "address, city, state_province, postcode, country, telephone, email, website, "
            "duns, national_id, founded_year, legal_status, employees_total, "
            "business_description, sic_codes, annual_sales_usd"
        )
    elif any(x in t for x in ["incorporation", "continuance", "amendment", "registration", "memorandum", "articles"]):
        fields = (
            "company_name, registration_number, date_of_incorporation, jurisdiction, "
            "registered_address, company_type, directors_list, officers_list, "
            "share_structure, fiscal_year_end, incorporator"
        )
    elif any(x in t for x in ["banking resolution", "board resolution"]):
        fields = (
            "company_name, resolution_date, meeting_type, authorized_signatories, "
            "signing_conditions, bank_name, account_details, effective_date, expiry_date"
        )
    elif "signature card" in t:
        fields = "company_name, account_number, bank_name, branch, authorized_signatories, signing_authority, date"
    elif "power of attorney" in t:
        fields = "grantor, attorney, scope, effective_date, expiry_date, jurisdiction, notarized"
    elif "bank statement" in t:
        fields = (
            "account_holder, account_number, bank_name, branch, statement_period, "
            "opening_balance, closing_balance, currency, total_credits, total_debits, transaction_count"
        )
    elif any(x in t for x in ["passport", "aadhaar", "pan", "driver", "voter", "national id"]):
        fields = (
            "full_name, id_number, date_of_birth, gender, nationality, expiry_date, "
            "issue_date, issuing_authority, address, father_name, aadhaar_number, "
            "pan_number, voter_id_number, licence_number, passport_number"
        )
    elif "gis" in t or "general information" in t:
        fields = (
            "company_name, sec_registration_number, date_of_registration, fiscal_year_end, "
            "principal_office_address, nature_of_business, authorized_capital, "
            "subscribed_capital, paid_up_capital, top_stockholders, directors_officers, external_auditor"
        )
    elif any(x in t for x in ["ownership", "beneficial"]):
        fields = (
            "company_name, inquiry_date, registration_number, country, status, "
            "industry, shareholders, ultimate_beneficial_owner, subsidiaries"
        )
    elif "salary" in t or "payslip" in t:
        fields = (
            "employee_name, employee_id, employer_name, pay_period, basic_salary, "
            "gross_salary, net_salary, deductions, allowances, currency"
        )
    elif "notarial" in t:
        fields = "notary_name, notary_number, company_name, document_certified, certification_date, jurisdiction"
    elif "resignation" in t:
        fields = "resigning_person, position, company_name, resignation_date, effective_date, reason"
    else:
        fields = (
            "names, dates, numbers, addresses, amounts, identifiers — extract any "
            "fields you can confidently identify"
        )

    return f"""You are a KYC document data extraction specialist. Extract ALL structured data from the document below.

DOCUMENT TYPE: {doc_type}
FILENAME: {filename}

EXTRACTION INSTRUCTIONS:
Extract these fields (use empty string "" when missing): {fields}

CRITICAL RULES:
1. Values EXACTLY as they appear in the document.
2. Dates: preserve original format (e.g., "13 JUN 2025").
3. Names: full name as printed.
4. Amounts: include currency symbol/code.
5. If a field is not found: empty string "" — never null.
6. Lists/arrays: convert to JSON string or comma-separated string.
7. ALL values in "data" must be plain strings.

Return ONLY this JSON (no markdown):
{{
  "owner": "<primary owner / company / person>",
  "score": <0.0-1.0>,
  "data": {{
    "field_name": "string value"
  }}
}}

DOCUMENT TEXT:
{truncated}"""


def build_retrieval_prompt(query: str, context: list, doc_type: str = "") -> str:
    return f"""You are a KYC document retrieval specialist. Extract precise information from the provided document context.

SEARCH QUERY: {query}
TARGET DOCUMENT TYPE: {doc_type or "Any"}

TASK: Find the best matching document and extract ALL available structured data from it.

RULES:
1. Return data EXACTLY as it appears in the source text.
2. Do not infer or fabricate any values.
3. Empty string "" for missing fields — never null.
4. ALL values must be plain strings.
5. For lists: convert to a JSON string.
6. score is how well the document matches the query (0.0-1.0).

Return ONLY this JSON:
{{
  "owner": "<name>",
  "document_name": "<filename>",
  "document_type": "<type>",
  "score": 0.9,
  "data": {{ "field": "string value" }}
}}

CONTEXT DOCUMENTS:
{json.dumps(context, indent=2)}"""


def build_universal_search_prompt(keyword: str, context: list) -> str:
    return f"""You are a KYC document intelligence specialist. A user is searching for a specific keyword or value across all indexed documents.

SEARCH KEYWORD / VALUE: "{keyword}"

TASK:
1. Search through ALL provided document chunks for any occurrence of the keyword/value.
2. The keyword could be: phone, PAN, Aadhaar, address, email, registration #, DUNS, account #, name, date, amount, etc.
3. For EACH document where the keyword is found (or closely matches), return a result entry.
4. Extract the matched field name and its value.

RULES:
- Match case-insensitively; allow partial matches (e.g., "9876" matches "9876543210").
- Do NOT fabricate matches — only genuine occurrences.
- If no match: empty results array.
- ALL values must be plain strings.

Return ONLY this JSON:
{{
  "query": "{keyword}",
  "results": [
    {{
      "owner": "<owner>",
      "document_name": "<filename>",
      "document_type": "<type>",
      "document_category": "<category>",
      "matched_field": "<field name>",
      "matched_value": "<exact value>",
      "relevance_score": <0.0-1.0>
    }}
  ]
}}

CONTEXT DOCUMENTS:
{json.dumps(context, indent=2)}"""


# ============================================================
# Chunking
# ============================================================
def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    """Simple paragraph-aware char chunker."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    cur = ""
    for para in paras:
        if len(cur) + len(para) + 2 <= max_chars:
            cur = f"{cur}\n\n{para}" if cur else para
            continue
        if cur:
            chunks.append(cur)
        if len(para) > max_chars:
            step = max(1, max_chars - overlap)
            for i in range(0, len(para), step):
                chunks.append(para[i : i + max_chars])
            cur = ""
        else:
            cur = para
    if cur:
        chunks.append(cur)
    return chunks


# ============================================================
# LLM helpers
# ============================================================
async def _chat_simple(prompt: str, *, max_tokens: int = 2048) -> str:
    """Single-shot chat with the project's active LLM provider."""
    client = get_stellar()
    model = model_for("final_gen") or model_for("fast")
    text, _usage = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        timeout=120.0,
    )
    return text


# ============================================================
# Ingestion driver — 2-pass classify + extract + chunk + embed
# ============================================================
async def ingest_kyc_pdf(
    pdf_bytes: bytes,
    *,
    filename: str,
    progress_cb: ProgressCb = None,
) -> dict[str, Any]:
    """Run the full KYC ingestion pipeline on a single PDF.

    Emits progress events:
        stage=upload         status=start|done
        stage=ocr            status=start|done   pages=N
        stage=classify       status=start|done   document_type, owner, confidence
        stage=extract        status=start|done   fields_count
        stage=embed          status=progress|done done/total
        stage=store          status=done         chunks=N, kyc_document_id=...

    Returns the persisted kyc_documents row id + summary.
    """
    async def emit(**payload: Any) -> None:
        if progress_cb:
            try:
                await progress_cb(payload)
            except Exception:
                logger.exception("kyc progress_cb raised")

    schema = settings.pg_schema

    # ── 1. S3 upload (best-effort)
    s3_uri: str | None = None
    if settings.s3_enabled:
        await emit(type="stage", stage="upload", status="start")
        try:
            s3_key = f"kyc/{uuid.uuid4().hex[:8]}/{filename}"
            s3_uri = await s3_put(s3_key, pdf_bytes, content_type="application/pdf")
            await emit(type="stage", stage="upload", status="done", s3_uri=s3_uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning("kyc s3_put failed: %s", exc)
            await emit(type="stage", stage="upload", status="error", message=str(exc))

    # ── 2. OCR
    await emit(type="stage", stage="ocr", status="start")
    loop = asyncio.get_running_loop()
    pages = await loop.run_in_executor(None, lambda: extract_pages(pdf_bytes, filename=filename))
    if not pages:
        await emit(type="error", message=f"OCR returned no pages for {filename}")
        return {"ok": False, "error": "ocr_empty"}
    page_texts = [f"[PAGE {p['page_number']}]\n{p['text']}".strip() for p in pages if p.get("text")]
    ocr_text = "\n\n".join(page_texts)
    await emit(type="stage", stage="ocr", status="done", pages=len(pages), chars=len(ocr_text))
    if not ocr_text.strip():
        await emit(type="error", message=f"OCR text is empty for {filename}")
        return {"ok": False, "error": "ocr_empty_text"}

    # ── 3. Pass 1 — Classify
    await emit(type="stage", stage="classify", status="start")
    cls_raw = await _chat_simple(build_classification_prompt(ocr_text, filename), max_tokens=1024)
    cls = extract_json_from_response(cls_raw) or {}
    owner = str(cls.get("owner") or "Unknown Owner").strip()
    doc_type = str(cls.get("document_type") or "Unknown Type").strip()
    doc_category = str(cls.get("document_category") or "").strip()
    confidence = _parse_confidence(cls.get("confidence_score", 0.5))
    signals = cls.get("classification_signals", []) or []
    source_platform = str(cls.get("source_platform") or "").strip()
    report_date = str(cls.get("report_date") or "").strip()
    await emit(
        type="stage", stage="classify", status="done",
        document_type=doc_type, document_category=doc_category,
        owner=owner, confidence=confidence, source_platform=source_platform,
    )

    # ── 4. Pass 2 — Extract type-specific fields
    await emit(type="stage", stage="extract", status="start")
    ext_raw = await _chat_simple(build_extraction_prompt(doc_type, ocr_text, filename), max_tokens=2048)
    ext = extract_json_from_response(ext_raw) or {}
    extracted_data: dict[str, str] = {}
    raw_data = ext.get("data", {})
    if isinstance(raw_data, dict):
        for k, v in raw_data.items():
            extracted_data[str(k)[:80]] = _sanitize_value(v, max_len=600)
    # If extraction surfaces a more specific owner, prefer it
    ext_owner = str(ext.get("owner") or "").strip()
    if ext_owner and ext_owner.lower() not in ("unknown owner", "unknown", ""):
        owner = ext_owner
    await emit(type="stage", stage="extract", status="done", fields=len(extracted_data))

    # ── 5. Chunk
    chunks = _chunk_text(ocr_text, max_chars=settings.kyc_chunk_size, overlap=settings.kyc_chunk_overlap)
    if not chunks:
        await emit(type="error", message=f"chunker produced 0 chunks for {filename}")
        return {"ok": False, "error": "no_chunks"}

    # ── 6. Embed (project's standard embedder)
    await emit(type="stage", stage="embed", status="start", total=len(chunks))
    client = get_stellar()
    batch_size = 16
    embeddings: list[list[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embs = await client.embed(batch)
        embeddings.extend(embs)
        await emit(
            type="stage", stage="embed", status="progress",
            done=len(embeddings), total=len(chunks),
        )
    await emit(type="stage", stage="embed", status="done", embeddings=len(embeddings))

    # ── 7. Store (UPSERT kyc_documents + insert kyc_chunks)
    await emit(type="stage", stage="store", status="start")
    owner_norm = normalize_name_full(owner)
    owner_first = normalize_name(owner)
    async with acquire() as conn:
        # Best-effort SET ROLE for the writes
        if settings.app_owner_role:
            try:
                await conn.execute(f'SET ROLE "{settings.app_owner_role}"')
            except Exception:
                pass
        kyc_doc_id = await conn.fetchval(
            f"""
            INSERT INTO "{schema}".kyc_documents
                (document_name, s3_uri, size_bytes,
                 owner, owner_normalized, owner_first_token,
                 document_type, document_category, confidence_score,
                 classification_signals, source_platform, report_date,
                 extracted_data, ocr_text, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, $12, $13::jsonb, $14, now())
            ON CONFLICT (document_name) DO UPDATE SET
                s3_uri                 = EXCLUDED.s3_uri,
                size_bytes             = EXCLUDED.size_bytes,
                owner                  = EXCLUDED.owner,
                owner_normalized       = EXCLUDED.owner_normalized,
                owner_first_token      = EXCLUDED.owner_first_token,
                document_type          = EXCLUDED.document_type,
                document_category      = EXCLUDED.document_category,
                confidence_score       = EXCLUDED.confidence_score,
                classification_signals = EXCLUDED.classification_signals,
                source_platform        = EXCLUDED.source_platform,
                report_date            = EXCLUDED.report_date,
                extracted_data         = EXCLUDED.extracted_data,
                ocr_text               = EXCLUDED.ocr_text,
                updated_at             = now(),
                deleted_at             = NULL
            RETURNING id
            """,
            filename, s3_uri, len(pdf_bytes),
            owner, owner_norm, owner_first,
            doc_type, doc_category, float(confidence),
            json.dumps(signals if isinstance(signals, list) else [signals]),
            source_platform, report_date,
            json.dumps(extracted_data), ocr_text[:200000],
        )

        # Clear existing chunks for this doc, then re-insert
        await conn.execute(
            f'DELETE FROM "{schema}".kyc_chunks WHERE kyc_document_id = $1',
            kyc_doc_id,
        )
        for idx, (content, vec) in enumerate(zip(chunks, embeddings)):
            token_count = max(1, len(content) // 4)
            await conn.execute(
                f"""
                INSERT INTO "{schema}".kyc_chunks
                    (kyc_document_id, chunk_index, content, token_count, embedding)
                VALUES ($1, $2, $3, $4, $5::vector)
                """,
                kyc_doc_id, idx, content, token_count, vec_to_pg(vec),
            )

    await emit(
        type="stage", stage="store", status="done",
        kyc_document_id=kyc_doc_id, chunks=len(chunks),
    )

    return {
        "ok": True,
        "kyc_document_id": kyc_doc_id,
        "document_name": filename,
        "owner": owner,
        "document_type": doc_type,
        "document_category": doc_category,
        "confidence_score": confidence,
        "source_platform": source_platform,
        "chunks": len(chunks),
        "s3_uri": s3_uri,
    }


# ============================================================
# Embedding column bootstrap — called at app startup
# ============================================================
async def ensure_kyc_embedding_column() -> None:
    """Add the embedding column on kyc_chunks once we know the active dim."""
    schema = settings.pg_schema
    role = settings.app_owner_role
    async with acquire() as conn:
        has_table = await conn.fetchval(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = $1 AND table_name = 'kyc_chunks'",
            schema,
        )
        if not has_table:
            return
        has_col = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = $1 AND table_name = 'kyc_chunks' "
            "  AND column_name = 'embedding'",
            schema,
        )
        if has_col:
            return
        from ..db import discover_embedding_dim
        try:
            dim = await discover_embedding_dim()
        except Exception:
            dim = 768
        if role:
            try:
                await conn.execute(f'SET ROLE "{role}"')
            except Exception:
                pass
        await conn.execute(
            f'ALTER TABLE "{schema}".kyc_chunks ADD COLUMN embedding vector({dim})'
        )
        await conn.execute(
            f'CREATE INDEX IF NOT EXISTS idx_kyc_chunks_hnsw '
            f'ON "{schema}".kyc_chunks USING hnsw (embedding vector_cosine_ops)'
        )
        logger.info("Added embedding vector(%d) + HNSW index on kyc_chunks", dim)


# ============================================================
# Owner & doc-type discovery
# ============================================================
async def list_owners() -> list[dict[str, Any]]:
    schema = settings.pg_schema
    async with acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT owner, owner_normalized, COUNT(*)::int AS doc_count
            FROM "{schema}".kyc_documents
            WHERE deleted_at IS NULL AND owner IS NOT NULL AND owner <> ''
            GROUP BY owner, owner_normalized
            ORDER BY owner
            """
        )
    return [
        {"owner": r["owner"], "owner_normalized": r["owner_normalized"],
         "doc_count": r["doc_count"]}
        for r in rows
    ]


async def list_doc_types(owner: str | None = None) -> list[dict[str, Any]]:
    """Distinct document types in the KYC corpus, optionally narrowed to one owner."""
    schema = settings.pg_schema
    if owner:
        norm = normalize_name_full(owner)
        first = normalize_name(owner)
        sql = (
            f'SELECT document_type, document_category, COUNT(*)::int AS doc_count '
            f'FROM "{schema}".kyc_documents '
            f'WHERE deleted_at IS NULL AND document_type IS NOT NULL '
            f'  AND ( owner_normalized = $1 OR owner_first_token = $2 '
            f'        OR owner_normalized LIKE $3 OR owner_first_token LIKE $3 ) '
            f'GROUP BY document_type, document_category '
            f'ORDER BY document_type'
        )
        args: list[Any] = [norm, first, f"%{first}%"]
    else:
        sql = (
            f'SELECT document_type, document_category, COUNT(*)::int AS doc_count '
            f'FROM "{schema}".kyc_documents '
            f'WHERE deleted_at IS NULL AND document_type IS NOT NULL '
            f'GROUP BY document_type, document_category '
            f'ORDER BY document_type'
        )
        args = []
    async with acquire() as conn:
        rows = await conn.fetch(sql, *args)
    return [
        {"document_type": r["document_type"],
         "document_category": r["document_category"],
         "doc_count": r["doc_count"]}
        for r in rows
    ]


# ============================================================
# Search — LIST mode (metadata only)
# ============================================================
async def list_by_owner(owner: str, doc_type: str | None = None) -> list[dict[str, Any]]:
    schema = settings.pg_schema
    norm = normalize_name_full(owner)
    first = normalize_name(owner)
    sql = (
        f'SELECT id, document_name, owner, document_type, document_category, '
        f'       confidence_score, source_platform, report_date, '
        f'       classification_signals, extracted_data, s3_uri, created_at '
        f'FROM "{schema}".kyc_documents '
        f'WHERE deleted_at IS NULL '
        f'  AND ( owner_normalized = $1 OR owner_first_token = $2 '
        f'        OR owner_normalized LIKE $3 OR owner_first_token LIKE $3 OR owner ILIKE $4 ) '
    )
    args: list[Any] = [norm, first, f"%{first}%", f"%{owner}%"]
    if doc_type:
        sql += "  AND document_type = $5 "
        args.append(doc_type)
    sql += "ORDER BY created_at DESC"
    async with acquire() as conn:
        rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]


# ============================================================
# Search — DETAILED mode (vector + LLM extraction)
# ============================================================
async def extract_for_owner_type(owner: str, doc_type: str) -> dict[str, Any] | None:
    """Vector search → top-K KYC chunks for the matching owner+type → LLM extraction."""
    schema = settings.pg_schema
    norm = normalize_name_full(owner)
    first = normalize_name(owner)

    # Embed query
    query_str = f"{owner} {doc_type}".strip()
    client = get_stellar()
    q_vec = await client.embed_one(query_str)

    # KNN over kyc_chunks restricted by owner+type via JOIN to kyc_documents
    sql = (
        f'WITH q AS (SELECT $1::vector AS v) '
        f'SELECT c.id AS chunk_id, c.content, c.page_number, '
        f'       d.id AS kyc_document_id, d.document_name, d.owner, '
        f'       d.document_type, d.document_category, d.confidence_score, '
        f'       d.source_platform, d.report_date, d.extracted_data, d.s3_uri, '
        f'       1 - (c.embedding <=> q.v) AS score '
        f'FROM "{schema}".kyc_chunks c '
        f'JOIN "{schema}".kyc_documents d ON d.id = c.kyc_document_id, q '
        f'WHERE c.deleted_at IS NULL AND d.deleted_at IS NULL '
        f'  AND ( d.owner_normalized = $2 OR d.owner_first_token = $3 '
        f'        OR d.owner_normalized LIKE $4 OR d.owner ILIKE $5 ) '
        f'  AND d.document_type = $6 '
        f'ORDER BY c.embedding <=> q.v '
        f'LIMIT 6'
    )
    args = [vec_to_pg(q_vec), norm, first, f"%{first}%", f"%{owner}%", doc_type]
    async with acquire() as conn:
        rows = await conn.fetch(sql, *args)
    if not rows:
        return None

    # Build context for LLM (top 3 unique docs)
    seen: set[int] = set()
    context: list[dict[str, Any]] = []
    for r in rows:
        if r["kyc_document_id"] in seen:
            continue
        seen.add(r["kyc_document_id"])
        context.append({
            "document_name": r["document_name"],
            "owner": r["owner"],
            "document_type": r["document_type"],
            "document_category": r["document_category"],
            "source_platform": r["source_platform"],
            "content": (r["content"] or "")[:2500],
        })
        if len(context) >= 3:
            break

    llm_raw = await _chat_simple(
        build_retrieval_prompt(query_str, context, doc_type),
        max_tokens=2048,
    )
    parsed = extract_json_from_response(llm_raw) or {}

    # Top match's own extracted_data is a great companion
    top = rows[0]
    return {
        "owner": parsed.get("owner") or top["owner"],
        "document_name": parsed.get("document_name") or top["document_name"],
        "document_type": parsed.get("document_type") or top["document_type"],
        "document_category": top["document_category"],
        "score": float(parsed.get("score", top["score"] or 0)),
        "s3_uri": top["s3_uri"],
        "data": parsed.get("data") or top["extracted_data"] or {},
        "stored_extracted_data": top["extracted_data"] or {},
        "confidence_score": top["confidence_score"],
        "source_platform": top["source_platform"],
        "report_date": top["report_date"],
    }


# ============================================================
# Universal keyword search (metadata scan + vector + LLM)
# ============================================================
async def universal_search(keyword: str, top_k: int = 8) -> list[dict[str, Any]]:
    if not keyword or not keyword.strip():
        return []
    schema = settings.pg_schema
    kw = keyword.strip()
    kw_lower = kw.lower()
    results: list[dict[str, Any]] = []
    seen_docs: set[int] = set()

    # Step 1 — Metadata scan (extracted_data JSONB + core columns)
    async with acquire() as conn:
        meta_hits = await conn.fetch(
            f"""
            SELECT id, document_name, owner, document_type, document_category,
                   confidence_score, source_platform, report_date,
                   extracted_data, s3_uri
            FROM "{schema}".kyc_documents
            WHERE deleted_at IS NULL
              AND (
                   owner ILIKE $1
                OR document_type ILIKE $1
                OR document_category ILIKE $1
                OR source_platform ILIKE $1
                OR report_date ILIKE $1
                OR document_name ILIKE $1
                OR extracted_data::text ILIKE $1
              )
            LIMIT 50
            """,
            f"%{kw}%",
        )
    for r in meta_hits:
        if r["id"] in seen_docs:
            continue
        seen_docs.add(r["id"])
        # Identify which field contained the match
        matched_field, matched_value = "", ""
        for col in ("owner", "document_type", "document_category",
                    "source_platform", "report_date", "document_name"):
            v = r[col]
            if v and kw_lower in str(v).lower():
                matched_field = col
                matched_value = str(v)
                break
        if not matched_field and isinstance(r["extracted_data"], dict):
            for k, v in r["extracted_data"].items():
                if v and kw_lower in str(v).lower():
                    matched_field = f"ext_{k}"
                    matched_value = str(v)
                    break
        results.append({
            "kyc_document_id": r["id"],
            "owner": r["owner"],
            "document_name": r["document_name"],
            "document_type": r["document_type"],
            "document_category": r["document_category"],
            "confidence_score": r["confidence_score"],
            "source_platform": r["source_platform"],
            "report_date": r["report_date"],
            "s3_uri": r["s3_uri"],
            "matched_field": matched_field or "metadata",
            "matched_value": matched_value or kw,
            "relevance_score": 1.0,
            "match_source": "metadata",
        })

    # Step 2 — Vector search over content (top_k) and exclude already-found docs
    client = get_stellar()
    q_vec = await client.embed_one(kw)
    async with acquire() as conn:
        vec_rows = await conn.fetch(
            f"""
            WITH q AS (SELECT $1::vector AS v)
            SELECT c.id AS chunk_id, c.content, c.page_number,
                   d.id AS kyc_document_id, d.document_name, d.owner,
                   d.document_type, d.document_category, d.confidence_score,
                   d.source_platform, d.report_date, d.s3_uri,
                   1 - (c.embedding <=> q.v) AS score
            FROM "{schema}".kyc_chunks c
            JOIN "{schema}".kyc_documents d ON d.id = c.kyc_document_id, q
            WHERE c.deleted_at IS NULL AND d.deleted_at IS NULL
            ORDER BY c.embedding <=> q.v
            LIMIT $2
            """,
            vec_to_pg(q_vec), top_k,
        )

    # Step 3 — LLM confirmation pass on vector candidates not already metadata-matched
    llm_context: list[dict[str, Any]] = []
    for r in vec_rows:
        if r["kyc_document_id"] in seen_docs:
            continue
        llm_context.append({
            "kyc_document_id": r["kyc_document_id"],
            "document_name": r["document_name"],
            "owner": r["owner"],
            "document_type": r["document_type"],
            "document_category": r["document_category"],
            "source_platform": r["source_platform"],
            "content": (r["content"] or "")[:3000],
        })
        if len(llm_context) >= 5:
            break

    if llm_context:
        llm_raw = await _chat_simple(
            build_universal_search_prompt(kw, llm_context),
            max_tokens=2048,
        )
        parsed = extract_json_from_response(llm_raw) or {}
        for r in parsed.get("results", []) or []:
            rel = float(r.get("relevance_score", 0) or 0)
            if rel < 0.3:
                continue
            # Match the docname back to a candidate to recover the id
            dn = str(r.get("document_name") or "")
            cand = next((c for c in llm_context if c["document_name"] == dn), None)
            kyc_id = cand["kyc_document_id"] if cand else None
            if kyc_id in seen_docs:
                continue
            if kyc_id is not None:
                seen_docs.add(kyc_id)
            results.append({
                "kyc_document_id": kyc_id,
                "owner": str(r.get("owner") or (cand or {}).get("owner") or ""),
                "document_name": dn or (cand or {}).get("document_name", ""),
                "document_type": str(r.get("document_type") or (cand or {}).get("document_type") or ""),
                "document_category": str(r.get("document_category") or (cand or {}).get("document_category") or ""),
                "confidence_score": None,
                "source_platform": (cand or {}).get("source_platform", ""),
                "report_date": "",
                "s3_uri": None,
                "matched_field": str(r.get("matched_field") or "content"),
                "matched_value": str(r.get("matched_value") or ""),
                "relevance_score": rel,
                "match_source": "vector+llm",
            })

    results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return results


# ============================================================
# Browse (paginated, optional category filter)
# ============================================================
async def browse(category: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    schema = settings.pg_schema
    if category and category != "All Categories":
        sql = (
            f'SELECT id, document_name, owner, document_type, document_category, '
            f'       confidence_score, source_platform, report_date, '
            f'       extracted_data, s3_uri, created_at '
            f'FROM "{schema}".kyc_documents '
            f'WHERE deleted_at IS NULL AND document_category = $1 '
            f'ORDER BY owner, document_type LIMIT $2'
        )
        args: list[Any] = [category, limit]
    else:
        sql = (
            f'SELECT id, document_name, owner, document_type, document_category, '
            f'       confidence_score, source_platform, report_date, '
            f'       extracted_data, s3_uri, created_at '
            f'FROM "{schema}".kyc_documents '
            f'WHERE deleted_at IS NULL '
            f'ORDER BY owner, document_type LIMIT $1'
        )
        args = [limit]
    async with acquire() as conn:
        rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]


# ============================================================
# Soft-delete a KYC document
# ============================================================
async def soft_delete(document_name: str) -> dict[str, Any]:
    schema = settings.pg_schema
    async with acquire() as conn:
        row = await conn.fetchrow(
            f'UPDATE "{schema}".kyc_documents '
            f'SET deleted_at = now() '
            f'WHERE document_name = $1 AND deleted_at IS NULL '
            f'RETURNING id, s3_uri',
            document_name,
        )
        if not row:
            return {"ok": False, "message": "not found or already deleted"}
        await conn.execute(
            f'UPDATE "{schema}".kyc_chunks SET deleted_at = now() '
            f'WHERE kyc_document_id = $1 AND deleted_at IS NULL',
            row["id"],
        )
    return {"ok": True, "id": row["id"], "s3_uri": row["s3_uri"]}
