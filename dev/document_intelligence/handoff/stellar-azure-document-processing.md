# Stellar + Azure Document Intelligence — Engineering Handoff

> **Audience:** an engineer or agent who is new to this repo and needs to work on
> document processing (ingestion, OCR, embedding, KYC). By the end of this doc you
> will know exactly how the two external services — **Stellar** (the LLM + embedding
> gateway) and **Azure Document Intelligence** (OCR) — are wired in, where every call
> lives, what every env var does, and which gotchas will bite you.
>
> Everything here is traceable to source. File references use clickable `path:line`.
>
> **Repo:** `retrieval` ("RAG Studio") — FastAPI + pgvector + React. This doc covers
> only the **document-processing surface**, not retrieval/rerank/generation.

---

## 0. Mental model (read this first)

There are **two external dependencies** involved in turning a PDF into searchable,
structured data:

| Service | What it does for us | Auth | Lives behind |
|---|---|---|---|
| **Stellar** | LLM gateway (chat) **and** text embeddings, OpenAI-compatible API | COIN OAuth2 + corporate SSL cert | Citi VDI only |
| **Azure Document Intelligence** (formerly Form Recognizer) | OCR — turns PDF bytes into per-page text, handles scanned docs / tables / multi-column | endpoint + API key | Azure (or an internal-hosted DI endpoint) |

Two flows consume them:

```
                              ┌─────────────────────────────────────────────┐
  PDF ──► OCR ──► chunk ──► embed ──► store          (1) MAIN CORPUS (WEGA)  │
          │              │                            Azure DI is INSIDE the │
          │ Azure DI     │ Stellar gte-large          WegaChunker wheel.     │
          │ (inside      │                            Stellar does embeddings│
          │  WegaChunker)│                            → vector.chunk_embeddings
          └──────────────┴─────────────────────────────────────────────────┘

                              ┌─────────────────────────────────────────────┐
  PDF ──► OCR ──► classify ──► extract ──► chunk ──► embed ──► store         │
          │        │           │                   │           (2) KYC FLOW │
          │ Azure  │ Stellar   │ Stellar           │ Stellar    Azure DI is │
          │ DI     │ chat      │ chat              │ gte-large  called      │
          │ direct │ (Pass 1)  │ (Pass 2)          │            DIRECTLY    │
          └────────┴───────────┴───────────────────┴──► vector.kyc_documents│
                                                        vector.kyc_chunks   │
                                                       └────────────────────┘
```

**The single most important fact:** Azure DI is integrated **two different ways**.
1. **Directly**, via [`ocr_azure.py`](../../../backend/app/pipeline/ocr_azure.py), used by the **KYC** flow.
2. **Indirectly**, via the internal **`WegaChunker`** wheel (you pass it the same
   endpoint + key and it does OCR + chunking for you), used by the **main corpus** flow.

Stellar is also integrated through **two client layers** (a low-level sync class and an
async wrapper) and there are **three near-identical copies** of the Stellar wiring in the
tree. Details below.

---

## Table of contents

- [Part A — Stellar (LLM + embedding gateway)](#part-a--stellar)
  - [A1. What Stellar is](#a1-what-stellar-is)
  - [A2. Authentication (COIN OAuth2 + SSL)](#a2-authentication-coin-oauth2--ssl)
  - [A3. The two client layers](#a3-the-two-client-layers)
  - [A4. Models and `model_for(task)`](#a4-models-and-model_fortask)
  - [A5. Public API surface](#a5-public-api-surface)
  - [A6. Token-usage accounting](#a6-token-usage-accounting)
  - [A7. The `get_llm.py` import hack](#a7-the-get_llmpy-import-hack)
  - [A8. The three copies of the Stellar client](#a8-the-three-copies)
- [Part B — Azure Document Intelligence (OCR)](#part-b--azure-document-intelligence)
  - [B1. Direct integration (`ocr_azure.py`)](#b1-direct-integration-ocr_azurepy)
  - [B2. Indirect integration (inside WegaChunker)](#b2-indirect-integration-inside-wegachunker)
  - [B3. Output shape and the never-raise contract](#b3-output-shape)
- [Part C — How they combine: the document-processing flows](#part-c--the-flows)
  - [C1. KYC ingestion (Azure DI + Stellar, end to end)](#c1-kyc-ingestion)
  - [C2. WEGA main-corpus ingestion (prod) + the 3 dispatch paths](#c2-wega-main-corpus-ingestion)
  - [C3. Contextual prefix pass (Stellar only)](#c3-contextual-prefix-pass)
  - [C4. Local / Vertex fallback (no Azure, no Stellar)](#c4-local--vertex-fallback)
  - [C5. The `ingest_remote/` microservice](#c5-the-ingest_remote-microservice)
- [Part D — Configuration & environment reference](#part-d--configuration--environment-reference)
- [Part E — Storage schema touchpoints](#part-e--storage-schema-touchpoints)
- [Part F — Operational gotchas & failure modes](#part-f--operational-gotchas--failure-modes)
- [Part G — File map](#part-g--file-map)
- [Appendix — copy-pasteable call snippets](#appendix--copy-pasteable-call-snippets)

---

# Part A — Stellar

## A1. What Stellar is

Stellar is Citi's internal **OpenAI-compatible** model gateway. You talk to it with the
stock `openai` Python SDK — `client.chat.completions.create(...)` and
`client.embeddings.create(...)` — pointed at `STELLAR_BASE_URL` with a COIN access token
as the API key. It serves both **chat/instruct models** (Llama family, Mixtral, Mistral)
and the **`gte-large-en-v1.5` embedding model**.

It is **VDI-only**: the COIN token endpoint and the Stellar base URL are reachable only
from inside the corporate environment. On a laptop you fall back to **Vertex** (see
[C4](#c4-local--vertex-fallback)); the code is provider-agnostic so the rest of the
pipeline doesn't care.

Source of truth for the low-level wiring: [`get_llm.py`](../../../get_llm.py) (class `StellarGenAI`).

## A2. Authentication (COIN OAuth2 + SSL)

Two things must be in place before any Stellar call works: a **corporate SSL cert** and a
**COIN OAuth2 token**.

### SSL cert
[`get_llm.py:31-34`](../../../get_llm.py) requires `SSL_CERT_FILE` and pins it onto both
`SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` process env. The actual API client is built with
`httpx.Client(verify=SSL_CERT_FILE)` so every request to Stellar validates against the
corporate CA bundle.

### COIN token — `get_coin_token()` ([`get_llm.py:55-83`](../../../get_llm.py))
```
POST  $COIN_URL
  headers: Content-Type: application/x-www-form-urlencoded
  data:
    grant_type    = client_credentials
    client_id     = base64-decode($COIN_CLIENT_ID)
    client_secret = base64-decode($COIN_CLIENT_SECRET)
    scope         = "coinscope" + base64-decode($COIN_SCOPE)
  → returns response.json()["access_token"]
```
Notes:
- `COIN_CLIENT_ID`, `COIN_CLIENT_SECRET`, `COIN_SCOPE` are stored **base64-encoded** in
  env and decoded at import time. The scope is prefixed with the literal `coinscope`.
- **The token POST uses `verify=False`** ([`get_llm.py:64`](../../../get_llm.py)) — TLS
  verification is disabled *for the token fetch only*. The subsequent Stellar API client
  still verifies against `SSL_CERT_FILE`. Don't "fix" this without checking the gateway;
  it is intentional for the COIN endpoint.

### Token TTL & auto-refresh (why the service can run for days)
COIN access tokens expire (~60 min). To avoid a mid-request 401, the client classes are
built around a **TTL-bounded `.client` property**:

- `COIN_TOKEN_TTL_SECONDS` (default `14 * 60` = **840 s / 14 min**) — [`get_llm.py:29`](../../../get_llm.py)
- `StellarGenAI._build_client()` mints a fresh token + new `OpenAI()` client and stamps
  `self._client_created = time.monotonic()` ([`get_llm.py:167-177`](../../../get_llm.py)).
- The `.client` property rebuilds automatically once `monotonic() - created > ttl`
  ([`get_llm.py:179-184`](../../../get_llm.py)). Callers just use `gw.client.foo.bar()`
  and never see an expired token.

This is a property-based factory pattern; the same pattern is used for `VertexGenAI`.

## A3. The two client layers

There is a **low-level sync class** and a **high-level async wrapper**. Application/pipeline
code should only ever touch the async wrapper.

### Layer 1 (low-level, sync): `get_llm.py::StellarGenAI`
- `__init__` builds the OpenAI client + COIN token eagerly.
- `create_embedding(query: str | list[str])` → calls `client.embeddings.create(model="gte-large-en-v1.5", input=...)`. Returns a **flat vector** for a single `str`, **list-of-lists** for a `list`. ([`get_llm.py:186-196`](../../../get_llm.py))
- `generate_content_llama(query)` → one-shot chat with the hard-coded `STELLAR_CHAT_MODEL`. **The RAG code never calls this** (it's hard-coded to one model); it uses `.client` directly so it can pick a model per call.
- Hard-coded `embedding_model_name = "gte-large-en-v1.5"` ([`get_llm.py:162`](../../../get_llm.py)).

### Layer 2 (high-level, async): `backend/app/stellar_client.py::StellarClient`
The wrapper the whole backend uses. It **reuses `StellarGenAI`'s auth + httpx wiring** by
instantiating it once, then drives the underlying `.client` directly with per-call kwargs.
Why it exists ([`stellar_client.py:4-15`](../../../backend/app/stellar_client.py)):
async access (so FastAPI doesn't block), per-call model selection, streaming, and
token-usage capture.

Every sync Stellar call is pushed off the event loop with
`loop.run_in_executor(None, ...)` — this is why `main.py` enlarges the default thread pool
to 64 workers ([`main.py:48-56`](../../../backend/app/main.py)).

## A4. Models and `model_for(task)`

Pipeline code never hard-codes a model name. It calls `model_for("<task>")`
([`stellar_client.py:236-248`](../../../backend/app/stellar_client.py)), which resolves
`{provider}_{task}_model` from settings, honouring the active provider.

| Task key | Stellar default | Vertex default | Used by |
|---|---|---|---|
| `embedding` | `gte-large-en-v1.5` | `text-embedding-005` | all embed calls |
| `final_gen` | `Llama-4-Scout-17B-16E-Instruct` | `gemini-2.5-flash` | answer gen, KYC classify/extract |
| `rerank` | `Meta-Llama-3.3-70B-Instruct` | `gemini-2.5-flash` | listwise rerank |
| `fast` | `Meta-Llama-3.1-8B-Instruct` | `gemini-2.5-flash` | CRAG, cheap passes |
| `contextual` | `Llama-4-Scout-17B-16E-Instruct` | `gemini-2.5-flash` | contextual prefix gen |

Defaults live in [`config.py:34-48`](../../../backend/app/config.py). Override any of them
with the matching env var (e.g. `STELLAR_FINAL_GEN_MODEL`). The full catalogue of models
the gateway can serve is in [`models.txt`](../../../models.txt) (probe them with
`python test_llm.py --models models.txt`).

> **Embedding model is locked to `gte-large-en-v1.5`** in both the backend
> ([`get_llm.py:162`](../../../get_llm.py)) and the remote service
> ([`ingest_remote/app/stellar_client.py:37`](../../../ingest_remote/app/stellar_client.py))
> so every ingestion path writes vectors of the same dimensionality. Don't change one
> without the other or KNN search will break on a dim mismatch.

## A5. Public API surface

`backend/app/stellar_client.py::StellarClient` — the surface every pipeline module relies on:

```python
async def embed(texts: str | list[str]) -> list[list[float]]   # always list-of-lists
async def embed_one(text: str) -> list[float]
async def chat(model, messages, *, temperature=0.2, max_tokens=1024, timeout=60.0)
                                                  -> tuple[str, TokenUsage]
async def chat_stream(model, messages, *, temperature=0.2, max_tokens=1024)
                                                  -> AsyncIterator[str]   # token deltas
```

- `embed` normalises the single-`str`-returns-flat-vector quirk so callers always get a
  list-of-lists ([`stellar_client.py:112-114`](../../../backend/app/stellar_client.py)).
- `chat` returns `(text, TokenUsage)` and auto-records usage.
- `chat_stream` sets `stream=True, stream_options={"include_usage": True}`, drives the sync
  generator in a thread, and pushes deltas onto an `asyncio.Queue`
  ([`stellar_client.py:151-206`](../../../backend/app/stellar_client.py)).

Get the singleton with `get_stellar()` ([`stellar_client.py:213-233`](../../../backend/app/stellar_client.py)).
It returns a `StellarClient` **or** a `VertexClient` depending on `settings.llm_provider`;
both expose the identical surface, so pipeline code is provider-agnostic.

The **remote service** copy adds one extra method:
`embed_sync(texts)` ([`ingest_remote/app/stellar_client.py:128-136`](../../../ingest_remote/app/stellar_client.py))
— a synchronous embed used from inside threads that the WEGA path spins up.

## A6. Token-usage accounting

[`backend/app/usage.py`](../../../backend/app/usage.py) implements a `ContextVar`-based
accumulator so concurrent requests don't pollute each other's token counts.

- Enter a request scope: `with UsageScope() as scope:` ([`usage.py:80-108`](../../../backend/app/usage.py)).
- `chat()` calls `record_usage(prompt, completion)` automatically.
- Pipeline stages `snapshot()` before/after and `delta(...)` to get per-stage tokens.
- **Cross-thread caveat:** `ContextVar` does **not** propagate into executor threads. For
  streaming, the code captures `acc = current_acc()` on the main task and calls
  `_add_to(acc, ...)` inside the thread ([`stellar_client.py:163-197`](../../../backend/app/stellar_client.py),
  [`usage.py:48-59`](../../../backend/app/usage.py)). Remember this if you add any new
  threaded LLM call — naive `record_usage` inside a thread is a silent no-op.

The remote service deliberately **drops** usage tracking
([`ingest_remote/app/stellar_client.py:9-13`](../../../ingest_remote/app/stellar_client.py))
— there is no analytics tab to feed there.

## A7. The `get_llm.py` import hack

`get_llm.py` validates its env vars **at import time** (`_require(...)` raises
`EnvironmentError` on a missing var). So the wrapper imports it **lazily** — only on first
client instantiation — to keep modules importable on a dev box without Stellar creds
([`stellar_client.py:67-100`](../../../backend/app/stellar_client.py)).

`_ensure_get_llm_importable()` ([`stellar_client.py:35-65`](../../../backend/app/stellar_client.py))
searches for `get_llm.py` in priority order and **force-promotes** the found dir to the
front of `sys.path` so a stray copy elsewhere can't shadow it:
1. `$STELLAR_GETLLM_PATH` (env override — file or its parent dir)
2. `backend/get_llm.py` (committed copy that ships with the backend)
3. `<repo-root>/get_llm.py` (canonical dev location)
4. `/app/get_llm.py` (Docker image root)

## A8. The three copies

The same Stellar wiring is duplicated three times. **If you change auth/model logic, change
all the relevant copies:**

| Copy | Role | Notes |
|---|---|---|
| [`get_llm.py`](../../../get_llm.py) (root) | canonical low-level `StellarGenAI` / `VertexGenAI` | the file the others import |
| [`backend/get_llm.py`](../../../backend/get_llm.py) | committed backend copy | shipped so the backend container is self-contained (207 lines, mirrors root) |
| [`backend/app/stellar_client.py`](../../../backend/app/stellar_client.py) | async wrapper + provider switch + usage | what the backend pipeline uses |
| [`ingest_remote/app/stellar_client.py`](../../../ingest_remote/app/stellar_client.py) | portable async wrapper for the remote VM | adds `embed_sync`, no Vertex dispatch, no usage |
| [`ingest_remote/get_llm.py`](../../../ingest_remote/get_llm.py) | remote copy of the low-level class | so the remote service has no backend dependency |

---

# Part B — Azure Document Intelligence

## B1. Direct integration (`ocr_azure.py`)

[`backend/app/pipeline/ocr_azure.py`](../../../backend/app/pipeline/ocr_azure.py) is a
small, self-contained OCR helper. **Public API — one function:**

```python
extract_pages(pdf_bytes: bytes, *, filename: str = "") -> list[dict]
# returns [{"page_number": int, "text": str}, ...]   — NEVER raises
```

### The Azure DI call ([`ocr_azure.py:52-82`](../../../backend/app/pipeline/ocr_azure.py))
```python
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

client = DocumentIntelligenceClient(
    endpoint=settings.azure_di_endpoint,
    credential=AzureKeyCredential(settings.azure_di_key),
)
poller = client.begin_analyze_document("prebuilt-read", body=pdf_bytes)  # long-running op
result = poller.result()

# per page: join the .content of every line with "\n"
for p in result.pages:
    text = "\n".join(line.content for line in p.lines)
```

- **Model used:** `prebuilt-read` — Azure's general OCR model (text + layout, handles
  scanned pages, multi-column, tables). Not a custom/trained model.
- **SDK package:** `azure-ai-documentintelligence` (+ `azure-core` for `AzureKeyCredential`).
  Imported lazily; a clear `RuntimeError` tells you to `pip install` it if missing
  ([`ocr_azure.py:58-65`](../../../backend/app/pipeline/ocr_azure.py)).
- **It is a synchronous SDK.** Callers wrap it in `run_in_executor` — see
  [`kyc.py:529`](../../../backend/app/pipeline/kyc.py).

### Fallback — pypdf
When Azure DI isn't configured (`azure_di_endpoint` / `azure_di_key` empty) or any call
fails, `extract_pages` falls back to **pypdf text-layer extraction**
([`ocr_azure.py:30-46`](../../../backend/app/pipeline/ocr_azure.py)). pypdf is fine on
clean digital PDFs and **bad on scanned PDFs** (no real OCR). This is what lets you develop
KYC locally without VDI/Azure access.

### Selection logic ([`ocr_azure.py:88-114`](../../../backend/app/pipeline/ocr_azure.py))
```
if azure_di_endpoint and azure_di_key:
    try Azure DI → if it returns >0 pages, use it
    on 0 pages OR any exception → log a warning, fall through
fall back to pypdf
on pypdf failure → return []   (never raises)
```

## B2. Indirect integration (inside WegaChunker)

The **main corpus** ingestion does **not** call `ocr_azure.py`. Instead it hands the Azure
DI endpoint + key to the internal **`WegaChunker`** wheel, which does OCR **and** chunking
in one shot ([`ingest_remote/app/ingest_core.py:153-171`](../../../ingest_remote/app/ingest_core.py)):

```python
from wega_chunker import WegaChunker      # internal pyarmoured wheel, imported lazily
chunker = WegaChunker(
    azure_di_endpoint=cfg["azure_di_endpoint"],
    azure_di_key=cfg["azure_di_key"],
)
result = chunker.chunk({
    "JobID": ...,
    "data": {
        "chunkTokenSize": cfg["chunk_token_size"],   # default 500
        "extractImages":  cfg["extract_images"],     # default False
        "piiExtraction":  {"detectPII": cfg["detect_pii"]},  # default True
    },
    "documentType": cfg["document_type"],            # default "General"
    "fileName": ..., "inputFolder": ...,
})
# result["chunkerResult"][0][fileName]["chunkData"] → list of chunk dicts
```

So in the WEGA path **Azure DI lives inside the wheel** — you configure it with the *same*
`AZURE_DI_ENDPOINT` / `AZURE_DI_KEY`, but you never see the DI client directly.

> The wheel is `pyarmoured` (obfuscated) and internal. It is imported **lazily** so the app
> boots in dev where the wheel isn't installed ([`ingest_core.py:153`](../../../ingest_remote/app/ingest_core.py)).
> The default DI endpoint in the remote service is an internal host:
> `http://sd-jibs-35nc.nam.nsroot.net:5000` ([`ingest_remote/app/config.py:30`](../../../ingest_remote/app/config.py)).

## B3. Output shape

Both Azure-DI and pypdf paths of `extract_pages` return the **same shape** so downstream
code is agnostic:
```python
[{"page_number": 1, "text": "..."}, {"page_number": 2, "text": "..."}, ...]
```
The KYC flow then stitches pages into one OCR string with `[PAGE N]` markers
([`kyc.py:533-534`](../../../backend/app/pipeline/kyc.py)):
```python
page_texts = [f"[PAGE {p['page_number']}]\n{p['text']}".strip() for p in pages if p.get("text")]
ocr_text = "\n\n".join(page_texts)
```

---

# Part C — The flows

## C1. KYC ingestion

The flagship "Azure + Stellar together" flow. Entry point:
`ingest_kyc_pdf(pdf_bytes, *, filename, progress_cb)`
([`kyc.py:487-671`](../../../backend/app/pipeline/kyc.py)). Triggered from the KYC router
(`backend/app/routers/kyc.py`) and streamed to the UI as SSE stage events.

| # | Stage | Service | Detail |
|---|---|---|---|
| 1 | **S3 upload** | MinIO/S3 | best-effort; key `kyc/<rand8>/<filename>` ([`kyc.py:514-524`](../../../backend/app/pipeline/kyc.py)) |
| 2 | **OCR** | **Azure DI** (pypdf fallback) | `extract_pages` in a thread → `[PAGE N]` text ([`kyc.py:526-538`](../../../backend/app/pipeline/kyc.py)) |
| 3 | **Pass 1 — Classify** | **Stellar chat** | `final_gen` model, temp 0, `max_tokens=1024`; prompt carries an 8-category **DOC_TYPE_TAXONOMY** + priority rules; returns JSON `{document_type, document_category, owner, confidence_score, classification_signals, source_platform, report_date}` ([`kyc.py:540-555`](../../../backend/app/pipeline/kyc.py), prompt at [`kyc.py:189-269`](../../../backend/app/pipeline/kyc.py)) |
| 4 | **Pass 2 — Extract** | **Stellar chat** | `final_gen` model, `max_tokens=2048`; **type-specific field list** chosen by doc type (Orbis, D&B, incorporation, bank statement, passport, …); returns JSON `{owner, score, data:{field:value}}` ([`kyc.py:557-570`](../../../backend/app/pipeline/kyc.py), prompt at [`kyc.py:272-372`](../../../backend/app/pipeline/kyc.py)) |
| 5 | **Chunk** | — | `_chunk_text`, paragraph-aware char chunker, `KYC_CHUNK_SIZE=1200` / `KYC_CHUNK_OVERLAP=200` ([`kyc.py:444-464`](../../../backend/app/pipeline/kyc.py)) |
| 6 | **Embed** | **Stellar** | `client.embed(batch)`, batch size **16** ([`kyc.py:578-591`](../../../backend/app/pipeline/kyc.py)) |
| 7 | **Store** | Postgres | UPSERT `kyc_documents` (`ON CONFLICT (document_name)`), then DELETE+INSERT `kyc_chunks` with `embedding::vector` ([`kyc.py:593-658`](../../../backend/app/pipeline/kyc.py)) |

LLM helper: `_chat_simple(prompt, max_tokens)` picks `model_for("final_gen") or model_for("fast")`, temp 0, 120 s timeout ([`kyc.py:470-481`](../../../backend/app/pipeline/kyc.py)).
LLM JSON is parsed tolerantly with `extract_json_from_response` (strips ```` ```json ```` fences, brace-matches) ([`kyc.py:156-183`](../../../backend/app/pipeline/kyc.py)).

**KYC search modes** (all use Stellar embed + chat over `kyc_chunks`/`kyc_documents`):
- `list_by_owner(owner, doc_type?)` — metadata only, owner-normalised match ([`kyc.py:777-801`](../../../backend/app/pipeline/kyc.py))
- `extract_for_owner_type(owner, doc_type)` — vector KNN (`embedding <=> q`) → top chunks → Stellar extraction ([`kyc.py:807-889`](../../../backend/app/pipeline/kyc.py))
- `universal_search(keyword)` — metadata scan → vector search → Stellar confirmation ([`kyc.py:895-1069`](../../../backend/app/pipeline/kyc.py))

**Embedding column bootstrap:** `ensure_kyc_embedding_column()` adds `embedding vector(<dim>)`
+ an HNSW (`vector_cosine_ops`) index to `kyc_chunks` at app startup, discovering the dim at
runtime (fallback 768) ([`kyc.py:677-714`](../../../backend/app/pipeline/kyc.py)). Called from
the lifespan in [`main.py:73-77`](../../../backend/app/main.py).

## C2. WEGA main-corpus ingestion

This is the production path for the general document corpus. The router
**`POST /api/ingest/wega`** ([`routers/ingest.py:234-402`](../../../backend/app/routers/ingest.py))
dispatches **three ways**, in this priority:

1. **Remote** — if `REMOTE_INGEST=true`: proxy the upload to the standalone
   `ingest_remote` service and re-stream its SSE
   ([`routers/ingest.py:164-248`](../../../backend/app/routers/ingest.py)). Auth via
   `X-Ingest-Secret`.
2. **Local/Vertex** — if `LLM_PROVIDER=vertex`: run `ingest_document_local` (pypdf + char
   chunker + Vertex embeddings). No WEGA, no Azure. See [C4](#c4-local--vertex-fallback).
3. **WEGA/Stellar (VDI prod)** — otherwise: import the legacy root
   [`ingest.py`](../../../ingest.py) and run its `main()` **in a thread**, capturing its log
   lines into the SSE stream ([`routers/ingest.py:115-161`](../../../backend/app/routers/ingest.py)).
   Because legacy `ingest.py` predates migration 005, the router brackets it: pre-inserts a
   `documents` row (UUID + SHA-256 + S3) and post-backfills `document_id` onto the chunks
   ([`routers/ingest.py:297-357`](../../../backend/app/routers/ingest.py)).

The canonical WEGA→embed→store implementation (used by the remote service, and a refactor
of the legacy script) is `ingest_pdf` in
[`ingest_remote/app/ingest_core.py:196-374`](../../../ingest_remote/app/ingest_core.py):

```
chunk  : WegaChunker(azure_di_endpoint, azure_di_key).chunk(config)  → chunkData[]
embed  : Stellar embed_sync, batch = EMBEDDING_BATCH_SIZE (16)       → vectors
store  : psycopg2 + pgvector register_vector → INSERT into <schema>.chunk_embeddings,
         create table+HNSW on the fly, dim taken from len(embeddings[0])
```

Chunk rows carry WEGA-native fields (`documentClass`, `sectionHeading`, `partNumber`,
`chunkBoundingBox`, `pageNumber`, `tokenCount`, …) ([`ingest_core.py:88-147, 303-351`](../../../ingest_remote/app/ingest_core.py)).

## C3. Contextual prefix pass

Stellar-only post-processing that implements **Anthropic Contextual Retrieval**
([`backend/app/pipeline/contextual.py`](../../../backend/app/pipeline/contextual.py),
endpoint `POST /api/ingest/contextual` SSE [`routers/ingest.py:50-98`](../../../backend/app/routers/ingest.py)).

For every chunk without a context row:
1. Load the **full document text** (cap 60 000 chars), cached per document ([`contextual.py:88-107`](../../../backend/app/pipeline/contextual.py)).
2. **Stellar chat** with the `contextual` model, temp 0.2, **`max_tokens=180`**, to write a
   50–100 token prefix situating the chunk ([`contextual.py:36-129`](../../../backend/app/pipeline/contextual.py)).
3. **Stellar embed** the `f"{context}\n\n{content}"` string ([`contextual.py:197-199`](../../../backend/app/pipeline/contextual.py)).
4. UPSERT into `chunk_context` (`context_text`, `context_embedding::vector`, `generator_model`),
   `ON CONFLICT (chunk_id)` ([`contextual.py:132-153`](../../../backend/app/pipeline/contextual.py)).

Concurrency is bounded by an `asyncio.Semaphore` (default 4) ([`contextual.py:156-222`](../../../backend/app/pipeline/contextual.py)).
No Azure here — it operates on already-ingested chunk text.

## C4. Local / Vertex fallback

When you're on a laptop (`LLM_PROVIDER=vertex`), `ingest_document_local`
([`backend/app/pipeline/local_ingest.py`](../../../backend/app/pipeline/local_ingest.py))
runs: **pypdf** text extraction (no Azure DI) → paragraph char chunker (1500/150) →
**Vertex** embeddings (via the same `get_stellar()` switch) → UPSERT `chunk_embeddings`.
This is purely for dev; the WEGA + Azure + Stellar prod path stays untouched.

## C5. The `ingest_remote/` microservice

A **standalone FastAPI service** ([`ingest_remote/app/main.py`](../../../ingest_remote/app/main.py))
meant to live on a VM that has the three things the main backend's host may not:
1. the internal **WegaChunker** wheel,
2. **Stellar** gateway access,
3. the **pgvector** Postgres.

- Endpoints: `GET /health`, `POST /ingest` (multipart upload, streams SSE)
  ([`main.py:70-179`](../../../ingest_remote/app/main.py)).
- Auth: optional shared secret via `X-Ingest-Secret` header vs `SHARED_SECRET`
  ([`main.py:62-67`](../../../ingest_remote/app/main.py)).
- Provider switch: `LLM_PROVIDER=wega` (WegaChunker + Stellar) or `vertex` (pypdf + Vertex)
  ([`main.py:115-119`](../../../ingest_remote/app/main.py)).
- Persists source PDF + a `documents` row (fresh UUID + SHA-256) via `persist_source`, so
  chunks can be stamped with a consistent `document_id` ([`main.py:121-145`](../../../ingest_remote/app/main.py)).
- Its own settings class with **real internal infra defaults** (PG host
  `KYC164283DEV.pgaas.dyn.nsroot.net:1524`, schema `vector_ng12499`, role
  `citi_pg_app_owner`) ([`ingest_remote/app/config.py`](../../../ingest_remote/app/config.py)).

Topology: the main backend sits in front and **proxies** `/api/ingest/wega` →
`ingest_remote:/ingest` when `REMOTE_INGEST=true`. Set the remote service's `S3_ENABLED=false`
in that topology to avoid double-upload (the backend already stores the PDF).

---

# Part D — Configuration & environment reference

All settings are env-driven (pydantic-settings, `case_sensitive=false`, loaded from `.env`).
Backend settings class: [`backend/app/config.py`](../../../backend/app/config.py). The
low-level `get_llm.py` reads its own set at **import time** and **raises** if any is missing.

### Stellar / COIN / SSL — required by [`get_llm.py`](../../../get_llm.py) (hard `_require`)
| Env var | Meaning |
|---|---|
| `SSL_CERT_FILE` | Path to corporate CA bundle. Pinned onto `REQUESTS_CA_BUNDLE` too. **Required.** |
| `COIN_URL` | COIN OAuth2 token endpoint. |
| `COIN_CLIENT_ID` | **base64-encoded** client id. |
| `COIN_CLIENT_SECRET` | **base64-encoded** client secret. |
| `COIN_SCOPE` | **base64-encoded** scope suffix (prefixed with `coinscope` at runtime). |
| `COIN_TOKEN_TTL_SECONDS` | Client-refresh interval. Default `840` (14 min). |
| `STELLAR_BASE_URL` | OpenAI-compatible base URL for Stellar. |
| `STELLAR_CHAT_MODEL` | Default chat model for the low-level class (RAG overrides per call). |
| `STELLAR_MAX_TOKENS` | Default max tokens for the low-level class. |

### Provider + model selection — [`config.py`](../../../backend/app/config.py)
| Env var | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `stellar` | `stellar` (VDI/prod) or `vertex` (local). Drives `get_stellar()` + `model_for`. |
| `STELLAR_EMBEDDING_MODEL` | `gte-large-en-v1.5` | embeddings |
| `STELLAR_FINAL_GEN_MODEL` | `Llama-4-Scout-17B-16E-Instruct` | answer / KYC classify+extract |
| `STELLAR_RERANK_MODEL` | `Meta-Llama-3.3-70B-Instruct` | listwise rerank |
| `STELLAR_FAST_MODEL` | `Meta-Llama-3.1-8B-Instruct` | cheap passes / CRAG |
| `STELLAR_CONTEXTUAL_MODEL` | `Llama-4-Scout-17B-16E-Instruct` | contextual prefixes |
| `STELLAR_GETLLM_PATH` | _(unset)_ | override where `get_llm.py` is found |

### Vertex (local-test) — [`config.py`](../../../backend/app/config.py)
`GOOGLE_APPLICATION_CREDENTIALS`, `VERTEX_PROJECT`, `VERTEX_LOCATION` (`us-central1`),
`VERTEX_EMBEDDING_MODEL` (`text-embedding-005`), `VERTEX_FINAL_GEN_MODEL` / `_RERANK_` /
`_FAST_` / `_CONTEXTUAL_` (all `gemini-2.5-flash`). The root `get_llm.py` also requires
`VERTEX_BASE_URL` and `VERTEX_DEFAULT_MODEL` / `VERTEX_EMBEDDING_MODEL` at import.

### Azure Document Intelligence — [`config.py:81-89`](../../../backend/app/config.py)
| Env var | Default | Meaning |
|---|---|---|
| `AZURE_DI_ENDPOINT` | `""` | DI endpoint. **Empty → pypdf fallback.** Also fed to WegaChunker. |
| `AZURE_DI_KEY` | `""` | DI API key. **Empty → pypdf fallback.** |
| `KYC_CHUNK_SIZE` | `1200` | KYC char-chunk size |
| `KYC_CHUNK_OVERLAP` | `200` | KYC char-chunk overlap |

### Remote ingest proxy — [`config.py:64-70`](../../../backend/app/config.py)
`REMOTE_INGEST` (`false`), `REMOTE_INGEST_URL` (`http://localhost:8090`),
`REMOTE_INGEST_SECRET`, `REMOTE_INGEST_TIMEOUT` (`1800` s).

### `ingest_remote` service — [`ingest_remote/app/config.py`](../../../ingest_remote/app/config.py)
Has its **own** set: `HOST`/`PORT` (8090), `SHARED_SECRET`, `LLM_PROVIDER` (`wega`),
`AZURE_DI_ENDPOINT` (internal default), `AZURE_DI_KEY`, `CHUNK_TOKEN_SIZE` (500),
`DOCUMENT_TYPE` (`General`), `DETECT_PII` (true), `EXTRACT_IMAGES` (false),
`EMBEDDING_BATCH_SIZE` (16), PG + S3 settings, `PG_APP_OWNER_ROLE` (`citi_pg_app_owner`).

### Postgres + storage — [`config.py:18-26, 72-79`](../../../backend/app/config.py)
`PG_HOST/PORT/USER/PASSWORD/DATABASE`, `PG_SCHEMA` (default `vector_ng12499`), `PG_TABLE`
(`chunk_embeddings`), `APP_OWNER_ROLE`, `PG_POOL_MIN`/`PG_POOL_MAX` (5/30, read in
[`db.py:73-75`](../../../backend/app/db.py)), and `S3_*` (MinIO defaults).

> A canonical commented template lives in `.env.example` at the repo root. (It is excluded
> from the editor's read tooling as an env file; copy it to `.env` and fill in
> `PG_*`, `COIN_*`, `STELLAR_*`, `AZURE_DI_*` per your environment.)

---

# Part E — Storage schema touchpoints

Schema is `settings.pg_schema` (default `vector_ng12499`; migrations use a `vector.`
sentinel that's rewritten at runtime — [`db.py:401-431`](../../../backend/app/db.py)).
Embedding **dimension is discovered at runtime** from the live column's `atttypmod`
(`discover_embedding_dim`, [`db.py:118-143`](../../../backend/app/db.py)); the bootstrap
default when nothing exists yet is **768**. `gte-large-en-v1.5` yields **1024-D** vectors,
Vertex `text-embedding-005` yields **768-D** — don't mix providers into one column.

| Table | Written by | Key columns | Vector index |
|---|---|---|---|
| `chunk_embeddings` | WEGA / local ingest | `content`, `chunkUUID`, `documentName`, `document_id`, `pageNumber`, `embedding`, `content_tsv` (FTS, generated), `deleted_at` | HNSW `vector_cosine_ops` |
| `chunk_context` | contextual pass | `chunk_id`, `context_text`, `context_embedding`, `generator_model` | HNSW on `context_embedding` |
| `kyc_documents` | KYC ingest | `document_name` (unique), `owner`/`owner_normalized`/`owner_first_token`, `document_type`/`category`, `confidence_score`, `extracted_data` (jsonb), `ocr_text`, `s3_uri`, `deleted_at` | — |
| `kyc_chunks` | KYC ingest | `kyc_document_id`, `chunk_index`, `content`, `token_count`, `embedding` | HNSW `vector_cosine_ops` |

`vec_to_pg(vec)` formats a Python float list into pgvector's `"[0.1,0.2,…]"` literal
([`db.py:25-26`](../../../backend/app/db.py)). Migrations: [`backend/migrations/`](../../../backend/migrations/)
(`001_rag_extensions`, `002_documents_and_softdelete`, `003_stage_tokens`, `004_kyc`,
`005_document_ids`), applied idempotently on backend startup ([`main.py:59-77`](../../../backend/app/main.py)).

---

# Part F — Operational gotchas & failure modes

1. **`get_llm.py` validates env at import.** A missing `SSL_CERT_FILE` / `COIN_*` /
   `STELLAR_*` raises `EnvironmentError` the moment the module is imported. That's why the
   import is lazy — but the first Stellar call on a misconfigured box will still blow up
   there. Symptom: app boots fine, first chat/embed throws.
2. **COIN token TTL.** Default 14-min client rebuild keeps long-running services alive.
   If you see periodic 401s mid-request, check `COIN_TOKEN_TTL_SECONDS` isn't set higher
   than the real token lifetime.
3. **Token fetch uses `verify=False`.** Intentional, for the COIN endpoint only. The API
   client still verifies against `SSL_CERT_FILE`.
4. **Azure DI never raises** — it silently degrades to pypdf. If KYC OCR quality is bad on
   scanned PDFs, confirm `AZURE_DI_ENDPOINT`/`KEY` are actually set; an empty value means
   you've been on the pypdf fallback the whole time (check logs for
   `OCR (pypdf fallback)` vs `OCR (Azure DI)`).
5. **Sync SDKs on the event loop.** Both the OpenAI client and the Azure DI SDK are sync;
   always call them through `run_in_executor`. The default executor is sized to 64 threads
   in [`main.py:48-56`](../../../backend/app/main.py) — a small pool will bottleneck
   concurrent ingest/retrieve.
6. **`ContextVar` doesn't cross into threads.** Token accounting inside a threaded LLM call
   must use the `_add_to(captured_acc, …)` pattern, not `record_usage`.
7. **Embedding dim lock-in.** Once a column exists at dim D, every vector you insert must be
   D. Switching `LLM_PROVIDER` between `stellar` (1024-D) and `vertex` (768-D) against the
   same table will fail inserts / KNN. Use separate DBs/schemas per provider.
8. **Three Stellar copies + two get_llm copies.** Keep auth/model edits in sync across
   root/backend/ingest_remote (see [A8](#a8-the-three-copies)).
9. **WegaChunker is optional in dev.** Lazily imported; the app boots without the wheel, but
   the WEGA/Stellar prod ingest path will fail at chunk time with an import error if the
   wheel isn't installed on the host.
10. **Double S3 upload.** In the backend-in-front + remote-ingest topology, set the remote
    service's `S3_ENABLED=false` so the PDF isn't stored twice.

---

# Part G — File map

| Path | What |
|---|---|
| [`get_llm.py`](../../../get_llm.py) | Low-level `StellarGenAI` / `VertexGenAI` — COIN auth, token TTL, embed/chat |
| [`backend/app/stellar_client.py`](../../../backend/app/stellar_client.py) | Async Stellar wrapper + `get_stellar()` + `model_for()` + provider switch |
| [`backend/app/vertex_client.py`](../../../backend/app/vertex_client.py) | Vertex implementation of the same surface (local provider) |
| [`backend/app/usage.py`](../../../backend/app/usage.py) | `ContextVar` token-usage accumulator |
| [`backend/app/config.py`](../../../backend/app/config.py) | All backend settings (Stellar models, Azure DI, PG, S3, remote) |
| [`backend/app/pipeline/ocr_azure.py`](../../../backend/app/pipeline/ocr_azure.py) | **Direct Azure DI OCR** (`prebuilt-read`) + pypdf fallback |
| [`backend/app/pipeline/kyc.py`](../../../backend/app/pipeline/kyc.py) | **KYC flow** — OCR → classify → extract → chunk → embed → store + search |
| [`backend/app/pipeline/contextual.py`](../../../backend/app/pipeline/contextual.py) | Stellar contextual-prefix generation |
| [`backend/app/pipeline/local_ingest.py`](../../../backend/app/pipeline/local_ingest.py) | pypdf + char chunk + provider embed (Vertex/local) |
| [`backend/app/routers/ingest.py`](../../../backend/app/routers/ingest.py) | `/api/ingest/wega` (3-way dispatch), `/api/ingest/contextual` |
| [`backend/app/routers/kyc.py`](../../../backend/app/routers/kyc.py) | KYC HTTP endpoints |
| [`backend/app/db.py`](../../../backend/app/db.py) | asyncpg pool, `vec_to_pg`, dim discovery, schema/table bootstrap, migrations |
| [`ingest.py`](../../../ingest.py) | Legacy root WEGA ingest script (run in-thread by the prod path) |
| [`ingest_remote/app/main.py`](../../../ingest_remote/app/main.py) | Standalone remote ingest FastAPI service |
| [`ingest_remote/app/ingest_core.py`](../../../ingest_remote/app/ingest_core.py) | **WegaChunker(Azure DI) → Stellar embed → pgvector** (canonical WEGA impl) |
| [`ingest_remote/app/stellar_client.py`](../../../ingest_remote/app/stellar_client.py) | Portable Stellar wrapper (`embed_sync`, no Vertex/usage) |
| [`ingest_remote/app/config.py`](../../../ingest_remote/app/config.py) | Remote service settings (internal infra defaults) |
| [`models.txt`](../../../models.txt) | Catalogue of Stellar + Vertex models to probe |
| [`test_llm.py`](../../../test_llm.py) | Probe script for the model catalogue |

---

# Appendix — copy-pasteable call snippets

### Embed text (provider-agnostic)
```python
from backend.app.stellar_client import get_stellar
client = get_stellar()                      # StellarClient or VertexClient
vecs = await client.embed(["chunk one", "chunk two"])   # -> list[list[float]]
one  = await client.embed_one("just this")              # -> list[float]
```

### One-shot chat (pick the model by task)
```python
from backend.app.stellar_client import get_stellar, model_for
client = get_stellar()
text, usage = await client.chat(
    model=model_for("final_gen"),
    messages=[{"role": "user", "content": "Summarise this."}],
    temperature=0,
    max_tokens=1024,
)
# usage.prompt, usage.completion, usage.total
```

### Stream chat tokens (SSE)
```python
async for delta in client.chat_stream(model_for("final_gen"), messages):
    send(delta)
```

### OCR a PDF (Azure DI with automatic pypdf fallback)
```python
from backend.app.pipeline.ocr_azure import extract_pages
pages = extract_pages(pdf_bytes, filename="doc.pdf")   # [{page_number, text}, ...]
# Set AZURE_DI_ENDPOINT + AZURE_DI_KEY to use Azure DI; empty → pypdf.
# This is a SYNC call — wrap it: await loop.run_in_executor(None, lambda: extract_pages(b))
```

### Full KYC ingest of one PDF
```python
from backend.app.pipeline.kyc import ingest_kyc_pdf
result = await ingest_kyc_pdf(pdf_bytes, filename="passport.pdf", progress_cb=cb)
# -> {ok, kyc_document_id, owner, document_type, document_category, chunks, s3_uri, ...}
```

---

*Generated as an engineering handoff. Every claim above is traceable to the cited
`file:line`. If a detail here disagrees with the code, the code wins — update this doc.*
