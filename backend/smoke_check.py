"""
Light smoke test that does NOT require Postgres or Stellar to be reachable.

It only verifies that every module imports cleanly and that the FastAPI app
can be instantiated. Run from the project root:

    cd backend
    python smoke_check.py
"""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

# Allow `from app.* import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent))

MODULES = [
    "app.config",
    "app.schemas",
    "app.db",
    "app.stellar_client",
    "app.scoring.metrics",
    "app.pipeline.dense",
    "app.pipeline.sparse",
    "app.pipeline.rrf",
    "app.pipeline.rerank",
    "app.pipeline.mmr",
    "app.pipeline.hyde",
    "app.pipeline.rewrite",
    "app.pipeline.contextual",
    "app.pipeline.crag",
    "app.pipeline.generate",
    "app.pipeline.retrieve",
    "app.routers.retrieve",
    "app.routers.ingest",
    "app.routers.documents",
    "app.routers.analytics",
    "app.routers.bench",
]

ok, bad = [], []
for m in MODULES:
    try:
        importlib.import_module(m)
        ok.append(m)
    except Exception as e:
        bad.append((m, e))
        traceback.print_exc()

print(f"\nimport summary: {len(ok)} ok, {len(bad)} failed")
for m, e in bad:
    print(f"  FAIL  {m}: {e}")
if bad:
    sys.exit(1)

# FastAPI app instantiation (uses lifespan but doesn't run it)
print("\ninstantiating FastAPI app...")
try:
    from app.main import app  # noqa
    print(f"app.title         = {app.title}")
    print(f"routes registered = {len(app.routes)}")
    for r in sorted(getattr(app, 'routes', []), key=lambda r: getattr(r, 'path', '')):
        path = getattr(r, "path", "?")
        methods = ",".join(sorted(getattr(r, "methods", []) or []))
        print(f"  {methods:12s}  {path}")
except Exception as e:
    print(f"FastAPI instantiation FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nsmoke check OK")
