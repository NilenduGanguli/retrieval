"""
test_local.py — local LLM-only smoke test using Vertex AI.

What it does:
  1. Auths via the service-account JSON
  2. Builds an embedding (verifies dim)
  3. Runs a one-shot chat completion
  4. Runs a streaming chat completion
  5. Exercises the four pipeline LLM stages that don't need Postgres:
        - HyDE generation
        - Query rewriting
        - Listwise reranking (RankGPT-style)
        - CRAG self-grader

What it does NOT test:
  * Pgvector hybrid retrieval (needs a populated DB)
  * The benchmark harness, contextual ingest, document explorer

Run:
    python test_local.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

# ----- minimal env (Vertex provider only — get_llm.py never gets imported) -----
SERVICE_ACCOUNT = "/Users/neelu/dev/nlp2sql-491115-200db3f6447d.json"
os.environ.setdefault("LLM_PROVIDER", "vertex")
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", SERVICE_ACCOUNT)
os.environ.setdefault("VERTEX_PROJECT", "nlp2sql-491115")
os.environ.setdefault("VERTEX_LOCATION", "us-central1")
os.environ.setdefault("VERTEX_EMBEDDING_MODEL", "text-embedding-005")
os.environ.setdefault("VERTEX_FINAL_GEN_MODEL", "gemini-2.5-flash")
os.environ.setdefault("VERTEX_RERANK_MODEL", "gemini-2.5-flash")
os.environ.setdefault("VERTEX_FAST_MODEL", "gemini-2.5-flash")
os.environ.setdefault("VERTEX_CONTEXTUAL_MODEL", "gemini-2.5-flash")

# Don't override SSL_CERT_FILE — google-genai uses it as the real CA bundle
# and falls back to certifi when unset.

# PG creds — only needed so pydantic_settings doesn't complain; the test
# never opens a DB connection.
os.environ.setdefault("PG_HOST", "localhost")
os.environ.setdefault("PG_PORT", "5432")
os.environ.setdefault("PG_DATABASE", "postgres")
os.environ.setdefault("PG_USER", "postgres")
os.environ.setdefault("PG_PASSWORD", "postgres")


# ----- colour helpers -----
def c(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s
def green(s):  return c(s, "1;32")
def red(s):    return c(s, "1;31")
def cyan(s):   return c(s, "1;36")
def dim(s):    return c(s, "2")
def bold(s):   return c(s, "1")


PASSED, FAILED = 0, 0
async def step(label: str, coro):
    global PASSED, FAILED
    t0 = time.perf_counter()
    print(f"\n{bold('▶ ' + label)}")
    try:
        result = await coro
        dt = (time.perf_counter() - t0) * 1000.0
        print(green(f"  ✓ passed ({dt:.0f} ms)"))
        PASSED += 1
        return result
    except Exception as exc:  # noqa: BLE001
        dt = (time.perf_counter() - t0) * 1000.0
        print(red(f"  ✗ FAILED ({dt:.0f} ms): {exc!r}"))
        import traceback; traceback.print_exc()
        FAILED += 1
        return None


# ============================================================================
async def test_auth():
    from backend.app.vertex_client import get_vertex
    client = get_vertex()
    print(dim(f"    project={client.client._api_client.project if hasattr(client.client, '_api_client') else '<n/a>'}"))
    return client


async def test_embedding():
    from backend.app.stellar_client import get_stellar
    llm = get_stellar()
    vecs = await llm.embed(["hello world", "RAG framework with hybrid retrieval"])
    assert len(vecs) == 2, f"expected 2 vectors, got {len(vecs)}"
    dim_ = len(vecs[0])
    assert dim_ > 0, "embedding dim is zero"
    print(dim(f"    dim={dim_}, first-vec[:6]={[round(x,4) for x in vecs[0][:6]]}"))
    return vecs


async def test_chat():
    from backend.app.stellar_client import get_stellar, model_for
    llm = get_stellar()
    text, usage = await llm.chat(
        model=model_for("final_gen"),
        messages=[
            {"role": "system", "content": "Answer with one short sentence."},
            {"role": "user", "content": "What is reciprocal rank fusion in one line?"},
        ],
        temperature=0.0,
        max_tokens=120,
    )
    print(dim(f"    response: {text[:160]}{'…' if len(text)>160 else ''}"))
    print(dim(f"    usage: prompt={usage.prompt} completion={usage.completion}"))
    return text


async def test_chat_stream():
    from backend.app.stellar_client import get_stellar, model_for
    llm = get_stellar()
    chunks = []
    print(dim("    streaming: "), end="", flush=True)
    async for tok in llm.chat_stream(
        model=model_for("final_gen"),
        messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "List the 4 retrieval stages in our RAG pipeline."},
        ],
        temperature=0.0,
        max_tokens=160,
    ):
        chunks.append(tok)
        sys.stdout.write(cyan(tok))
        sys.stdout.flush()
    print()
    full = "".join(chunks)
    assert full.strip(), "empty stream"
    return full


async def test_hyde():
    from backend.app.pipeline.hyde import hyde_generate
    text = await hyde_generate("How does HNSW index work for approximate nearest neighbour?")
    print(dim(f"    hypothetical: {text[:200]}{'…' if len(text)>200 else ''}"))
    return text


async def test_rewrite():
    from backend.app.pipeline.rewrite import rewrite_query
    out = await rewrite_query("how do hybrid search systems combine BM25 and vectors", n=3)
    for i, q in enumerate(out, 1):
        print(dim(f"    [{i}] {q}"))
    assert len(out) >= 1
    return out


async def test_listwise_rerank():
    from backend.app.pipeline.rerank import RerankInput, llm_listwise_rerank
    query = "What is reciprocal rank fusion?"
    candidates = [
        RerankInput(101, "Reciprocal Rank Fusion (RRF) is a method that combines ranked lists from multiple retrieval systems by summing 1/(k+rank).", "rrf.pdf", 2),
        RerankInput(102, "HNSW (Hierarchical Navigable Small World) is a graph-based ANN index used in pgvector.", "hnsw.pdf", 1),
        RerankInput(103, "BM25 is a ranking function used in lexical retrieval — a sparse representation alternative to TF-IDF.", "bm25.pdf", 5),
        RerankInput(104, "The fusion formula in Cormack et al. 2009 outperforms Condorcet methods on TREC tasks.", "rrf.pdf", 3),
        RerankInput(105, "Vector embeddings turn text into dense floats; cosine distance is a common similarity metric.", "vec.pdf", 1),
    ]
    ranked = await llm_listwise_rerank(query, candidates, top_n=5)
    print(dim(f"    input order  : {[c.chunk_id for c in candidates]}"))
    print(dim(f"    rerank output: {ranked}"))
    # 101 and 104 should be near the top
    top2 = set(ranked[:2])
    print(dim(f"    top-2 set    : {top2}  (expected to contain 101 and/or 104)"))
    return ranked


async def test_crag():
    from backend.app.pipeline.crag import crag_grade
    # Highly relevant scenario
    good = await crag_grade(
        "What is RRF?",
        [
            "Reciprocal Rank Fusion (RRF) is a method for combining ranked lists.",
            "It sums 1/(k + rank) across each ranker.",
        ],
    )
    print(dim(f"    relevant : conf={good.confidence:.2f} action={good.action!r} — {good.rationale[:120]}"))

    # Off-topic scenario
    bad = await crag_grade(
        "What is RRF?",
        [
            "Banana bread requires ripe bananas.",
            "Sourdough relies on a live starter culture.",
        ],
    )
    print(dim(f"    off-topic: conf={bad.confidence:.2f} action={bad.action!r} — {bad.rationale[:120]}"))
    return (good, bad)


# ============================================================================
async def main():
    print(bold("\n=== RAG Studio · Local Vertex smoke test ===\n"))
    print(dim(f"creds: {SERVICE_ACCOUNT}"))
    print(dim(f"project: {os.environ['VERTEX_PROJECT']} · location: {os.environ['VERTEX_LOCATION']}"))
    print(dim(f"chat: {os.environ['VERTEX_FINAL_GEN_MODEL']} · embed: {os.environ['VERTEX_EMBEDDING_MODEL']}"))

    await step("1. Vertex auth (service-account)", test_auth())
    await step("2. Embedding (1 batch, 2 inputs)", test_embedding())
    await step("3. One-shot chat completion", test_chat())
    await step("4. Streaming chat completion", test_chat_stream())
    await step("5. HyDE generation", test_hyde())
    await step("6. Query rewriting (3 variants)", test_rewrite())
    await step("7. Listwise reranking (5 candidates)", test_listwise_rerank())
    await step("8. CRAG self-grader (relevant vs off-topic)", test_crag())

    print()
    print(bold(f"Result: {green(str(PASSED) + ' passed')} · {red(str(FAILED) + ' failed')}\n"))
    sys.exit(0 if FAILED == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
