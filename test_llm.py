#!/usr/bin/env python3
"""
test_llm.py — probe which LLM models are reachable from this host.

Reads model names from a text file (one per line, `#` comments allowed),
fires a tiny chat completion AND / OR embedding request at each, and
prints a pass/fail table with per-model latency + error detail.

Usage
-----
  python test_llm.py                                    # models.txt, both tasks, auto provider
  python test_llm.py --models my-models.txt
  python test_llm.py --provider stellar --task chat
  python test_llm.py --provider vertex  --task embed
  python test_llm.py --provider both    --json out.json
  python test_llm.py --timeout 15       # tighter per-call timeout
  python test_llm.py --prompt "ping" --text "hello"

The script loads environment from `.env` automatically (in the same
directory) so you don't need to `export` anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# .env loader (no python-dotenv dependency required)
# --------------------------------------------------------------------------- #
def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_env()

# --------------------------------------------------------------------------- #
# Make get_llm.py + backend.* importable
# --------------------------------------------------------------------------- #
for cand in (ROOT, ROOT / "backend"):
    if (cand / "get_llm.py").is_file():
        sys.path.insert(0, str(cand))
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# Model-file parsing
# --------------------------------------------------------------------------- #
def parse_models_file(path: Path) -> list[str]:
    if not path.is_file():
        sys.exit(f"models file not found: {path}")
    out: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


# --------------------------------------------------------------------------- #
# Stellar probes
# --------------------------------------------------------------------------- #
_stellar_client: Any = None


def _stellar() -> Any:
    """Return a cached StellarGenAI (one COIN token for the whole run)."""
    global _stellar_client
    if _stellar_client is None:
        import get_llm  # type: ignore  # comes from repo (root or backend/)
        _stellar_client = get_llm.StellarGenAI()
    return _stellar_client


def test_stellar_chat(model: str, prompt: str, timeout: float) -> tuple[bool, float, str]:
    t0 = time.monotonic()
    try:
        resp = _stellar().client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=32,
            timeout=timeout,
        )
        text = (resp.choices[0].message.content or "").replace("\n", " ").strip()
        return True, (time.monotonic() - t0) * 1000, text[:80]
    except Exception as exc:
        return False, (time.monotonic() - t0) * 1000, str(exc)[:200]


def test_stellar_embed(model: str, text: str, timeout: float) -> tuple[bool, float, str]:
    t0 = time.monotonic()
    try:
        resp = _stellar().client.embeddings.create(
            model=model, input=[text], timeout=timeout,
        )
        dim = len(resp.data[0].embedding)
        return True, (time.monotonic() - t0) * 1000, f"dim={dim}"
    except Exception as exc:
        return False, (time.monotonic() - t0) * 1000, str(exc)[:200]


# --------------------------------------------------------------------------- #
# Vertex probes (uses google-genai, service-account creds when available)
# --------------------------------------------------------------------------- #
_vertex_client: Any = None


def _vertex() -> Any:
    global _vertex_client
    if _vertex_client is not None:
        return _vertex_client
    from google import genai
    kwargs: dict = {
        "vertexai": True,
        "location": os.environ.get("VERTEX_LOCATION", "us-central1"),
    }
    project = os.environ.get("VERTEX_PROJECT", "")
    if project:
        kwargs["project"] = project
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds_path and os.path.isfile(creds_path):
        from google.oauth2 import service_account
        kwargs["credentials"] = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    _vertex_client = genai.Client(**kwargs)
    return _vertex_client


def test_vertex_chat(model: str, prompt: str, timeout: float) -> tuple[bool, float, str]:
    t0 = time.monotonic()
    try:
        from google.genai import types
        resp = _vertex().models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=32,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (resp.text or "").replace("\n", " ").strip()
        return True, (time.monotonic() - t0) * 1000, text[:80]
    except Exception as exc:
        return False, (time.monotonic() - t0) * 1000, str(exc)[:200]


def test_vertex_embed(model: str, text: str, timeout: float) -> tuple[bool, float, str]:
    t0 = time.monotonic()
    try:
        resp = _vertex().models.embed_content(model=model, contents=[text])
        vals = getattr(resp.embeddings[0], "values", None) or []
        return True, (time.monotonic() - t0) * 1000, f"dim={len(vals)}"
    except Exception as exc:
        return False, (time.monotonic() - t0) * 1000, str(exc)[:200]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
_internal_vertex_client: Any = None


def _internal_vertex() -> Any:
    """Vertex via the internal corporate proxy, using get_llm.VertexGenAI."""
    global _internal_vertex_client
    if _internal_vertex_client is None:
        import get_llm  # type: ignore
        _internal_vertex_client = get_llm.VertexGenAI()
    return _internal_vertex_client


def test_vertex_internal_chat(model: str, prompt: str, timeout: float) -> tuple[bool, float, str]:
    from google.genai import types
    t0 = time.monotonic()
    try:
        resp = _internal_vertex().client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=0, max_output_tokens=32,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (resp.text or "").replace("\n", " ").strip()
        return True, (time.monotonic() - t0) * 1000, text[:80]
    except Exception as exc:
        return False, (time.monotonic() - t0) * 1000, str(exc)[:200]


def test_vertex_internal_embed(model: str, text: str, timeout: float) -> tuple[bool, float, str]:
    t0 = time.monotonic()
    try:
        resp = _internal_vertex().client.models.embed_content(model=model, contents=[text])
        vals = getattr(resp.embeddings[0], "values", None) or []
        return True, (time.monotonic() - t0) * 1000, f"dim={len(vals)}"
    except Exception as exc:
        return False, (time.monotonic() - t0) * 1000, str(exc)[:200]


PROBES = {
    ("stellar", "chat"):  test_stellar_chat,
    ("stellar", "embed"): test_stellar_embed,
    ("vertex",  "chat"):  test_vertex_chat,
    ("vertex",  "embed"): test_vertex_embed,
    ("vertex_internal", "chat"):  test_vertex_internal_chat,
    ("vertex_internal", "embed"): test_vertex_internal_embed,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe which LLM models are reachable via Stellar / Vertex.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--models", "-f", type=Path, default=Path("models.txt"),
                    help="Text file of model names — one per line. '#' comments are skipped.")
    ap.add_argument("--provider", "-p",
                    choices=["stellar", "vertex", "vertex_internal", "both", "all", "auto"],
                    default="auto",
                    help="Which provider(s) to test. 'auto' picks from LLM_PROVIDER. "
                         "'both' = stellar+vertex. 'all' = stellar+vertex+vertex_internal.")
    ap.add_argument("--task", "-t",
                    choices=["chat", "embed", "both"], default="both",
                    help="What kind of probe to run against each model.")
    ap.add_argument("--prompt", default="Reply with the single word: hi",
                    help="Prompt text for chat probes.")
    ap.add_argument("--text", default="hello world",
                    help="Text to embed for embed probes.")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="Per-call HTTP timeout in seconds.")
    ap.add_argument("--json", type=Path,
                    help="Optional path to write structured results.")
    args = ap.parse_args()

    models = parse_models_file(args.models)
    if not models:
        sys.exit(f"no models found in {args.models} (after stripping blanks/comments)")

    if args.provider == "auto":
        providers = [os.environ.get("LLM_PROVIDER", "vertex_internal").lower()]
    elif args.provider == "both":
        providers = ["stellar", "vertex"]
    elif args.provider == "all":
        providers = ["stellar", "vertex", "vertex_internal"]
    else:
        providers = [args.provider]
    # Normalise "internal_vertex" alias → "vertex_internal"
    providers = ["vertex_internal" if p == "internal_vertex" else p for p in providers]

    tasks = ["chat", "embed"] if args.task == "both" else [args.task]

    # Header
    print(
        f"\nTesting {len(models)} model(s) × {len(providers)} provider(s) × {len(tasks)} task(s)"
        f"  ·  timeout={args.timeout:.0f}s\n"
    )
    print(f"{'model':<44} {'provider':<8} {'task':<6} {'':2} {'ms':>7}  detail")
    print("-" * 115)

    results: list[dict] = []
    for model in models:
        for provider in providers:
            for task in tasks:
                fn = PROBES[(provider, task)]
                probe = args.prompt if task == "chat" else args.text
                ok, ms, detail = fn(model, probe, args.timeout)
                results.append({
                    "model": model, "provider": provider, "task": task,
                    "ok": ok, "latency_ms": round(ms, 1), "detail": detail,
                })
                tag = "OK" if ok else "✗"
                print(f"{model:<44} {provider:<8} {task:<6} {tag:<2} {ms:>7.0f}  {detail[:60]}")

    # Summary
    print()
    print("=" * 60)
    summary: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = summary.setdefault(r["provider"], {"ok": 0, "total": 0})
        bucket["total"] += 1
        if r["ok"]:
            bucket["ok"] += 1
    for prov, d in summary.items():
        print(f"  {prov:<8} {d['ok']}/{d['total']} reachable")

    # Models that worked at least once (handy paste-back into config)
    print()
    working = sorted({r["model"] for r in results if r["ok"]})
    print(f"working models ({len(working)}):")
    for m in working:
        print(f"  - {m}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote JSON results to {args.json}")


if __name__ == "__main__":
    main()
