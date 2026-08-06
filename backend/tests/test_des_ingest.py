"""Offline tests for the DES ingestion path (backend/app/des_client.py +
the /des and /sources routes in backend/app/routers/ingest.py).

NOTHING here touches the network, DES, or Postgres: `des_client`'s three call
sites are swapped out for fakes, and the one test that exercises the real
client stubs `httpx.AsyncClient` itself.

Run with the repo venv:

    .venv/bin/python -m pytest backend/tests/test_des_ingest.py -q

or standalone, with no pytest at all:

    .venv/bin/python backend/tests/test_des_ingest.py
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# The backend is a package rooted at backend/ (`python -m app.main`), and
# importing the router creates its upload dir — point that at a temp dir so a
# test run never writes into the repo.
_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
os.environ.setdefault("APP_UPLOAD_DIR", tempfile.mkdtemp(prefix="des-tests-uploads-"))

import httpx  # noqa: E402

from app import des_client  # noqa: E402
from app.config import settings  # noqa: E402
from app.des_client import DesError  # noqa: E402
from app.routers import ingest  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _run(coro):
    return asyncio.run(coro)


async def _collect(agen) -> list[tuple[str, dict]]:
    """Drain an SSE generator into [(event_name, decoded_payload), ...]."""
    out: list[tuple[str, dict]] = []
    async for item in agen:
        out.append((item["event"], json.loads(item["data"])))
    return out


def _names(events: list[tuple[str, dict]]) -> list[str]:
    return [name for name, _ in events]


@contextlib.contextmanager
def _patch(target: Any, **attrs: Any):
    """Temporarily set attributes on `target`, restoring them afterwards."""
    previous = {k: getattr(target, k) for k in attrs}
    for k, v in attrs.items():
        setattr(target, k, v)
    try:
        yield
    finally:
        for k, v in previous.items():
            setattr(target, k, v)


def _fake_des(
    *,
    runs: list[dict],
    event_polls: list[list[dict]] | None = None,
    submitted: dict | None = None,
    calls: dict | None = None,
):
    """Build submit/get_run/get_events stand-ins driven by scripted responses.

    `runs` is consumed one entry per poll; the last entry repeats forever, so a
    terminal run can be listed once. `event_polls` mirrors that for the event
    log (DES always returns the WHOLE log, hence the cumulative fixtures).
    """
    state = {"poll": 0}
    calls = calls if calls is not None else {}
    calls.setdefault("submit", 0)
    calls.setdefault("get_run", 0)
    calls.setdefault("get_events", 0)
    payload = submitted or {
        "document_id": "doc-1",
        "run_id": "run-1",
        "status": "queued",
        "name": "sample.pdf",
        "size_bytes": 11,
        "sha256": "abc123",
        "s3_uri": "s3://bucket/doc-1/sample.pdf",
    }

    async def submit(file_bytes, filename, content_type=None):
        calls["submit"] += 1
        return payload

    async def get_run(run_id):
        idx = min(state["poll"], len(runs) - 1)
        calls["get_run"] += 1
        return runs[idx]

    async def get_events(run_id):
        polls = event_polls or []
        if not polls:
            return []
        idx = min(state["poll"], len(polls) - 1)
        calls["get_events"] += 1
        state["poll"] += 1  # advance after both fetches of one iteration
        return polls[idx]

    if not event_polls:
        # Keep the poll counter moving even when the event log is unused.
        async def get_events(run_id):  # noqa: F811
            calls["get_events"] += 1
            state["poll"] += 1
            return []

    return submit, get_run, get_events


def _stage_map(*done: str) -> dict[str, int]:
    return {name: 120 for name in done}


def _sample_file() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="des-tests-src-")) / "sample.pdf"
    tmp.write_bytes(b"%PDF-1.4 fake")
    return tmp


@contextlib.contextmanager
def _fast_polling():
    with _patch(settings, des_poll_interval=0.0, des_timeout=30):
        yield


# ---------------------------------------------------------------------------
# (a) happy path: start -> info(s) -> progress -> done, with the ids the
#     Ingestion tab needs to fetch the inserted chunks
# ---------------------------------------------------------------------------
def test_stream_emits_start_info_progress_done_in_order():
    runs = [
        {"id": "run-1", "status": "queued", "stages": {}, "chunk_count": None},
        {
            "id": "run-1",
            "status": "running",
            "stages": _stage_map("upload", "raster", "ocr"),
            "chunk_count": None,
        },
        {
            "id": "run-1",
            "status": "succeeded",
            "document_id": "doc-1",
            "document_name": "sample.pdf",
            "stages": _stage_map(
                "upload", "raster", "ocr", "chunk", "context", "embed", "persist"
            ),
            "page_count": 3,
            "chunk_count": 42,
            "error": None,
        },
    ]
    event_polls = [
        [],
        [
            {"seq": 1, "stage": "upload", "status": "ok", "detail": {"bytes": 11}},
            {"seq": 2, "stage": "ocr", "status": "start", "detail": {}},
        ],
        [
            {"seq": 1, "stage": "upload", "status": "ok", "detail": {"bytes": 11}},
            {"seq": 2, "stage": "ocr", "status": "start", "detail": {}},
            {"seq": 3, "stage": "persist", "status": "ok", "detail": {"chunks": 42}},
        ],
    ]
    submit, get_run, get_events = _fake_des(runs=runs, event_polls=event_polls)

    with _fast_polling(), _patch(
        des_client, submit=submit, get_run=get_run, get_events=get_events
    ):
        events = _run(_collect(ingest._stream_des_ingest(_sample_file(), "sample.pdf")))

    names = _names(events)
    assert names[0] == "start", names
    assert events[0][1]["mode"] == "des"
    assert events[0][1]["filename"] == "sample.pdf"

    # start -> info -> ... -> progress -> ... -> done, strictly ordered
    first_info = names.index("info")
    first_progress = names.index("progress")
    assert 0 < first_info < first_progress, names
    assert names[-1] == "done", names
    assert "error" not in names, names

    # the submit info line correlates the run for an operator
    assert "run-1" in events[first_info][1]["line"]
    assert events[first_info][1]["document_id"] == "doc-1"

    # progress is completed-stages / 7, message names the current stage
    fractions = [p["progress"] for n, p in events if n == "progress"]
    assert fractions[0] == 0.0
    assert abs(fractions[1] - 3 / 7) < 1e-9, fractions
    assert fractions[-1] == 1.0, fractions
    assert [p["message"] for n, p in events if n == "progress"][1] == "chunk"

    done = events[-1][1]
    assert done["processed"] == 42 and done["total"] == 42
    assert done["document_id"] == "doc-1"
    assert done["document"] == "sample.pdf"
    assert done["page_count"] == 3


# ---------------------------------------------------------------------------
# (b) a failed run is an SSE "error", never an exception
# ---------------------------------------------------------------------------
def test_failed_run_yields_error_event():
    runs = [
        {"id": "run-1", "status": "running", "stages": _stage_map("upload")},
        {
            "id": "run-1",
            "status": "failed",
            "stages": _stage_map("upload", "raster"),
            "error": "azure DI returned 429 after 5 retries",
            "chunk_count": 0,
        },
    ]
    submit, get_run, get_events = _fake_des(runs=runs)

    with _fast_polling(), _patch(
        des_client, submit=submit, get_run=get_run, get_events=get_events
    ):
        events = _run(_collect(ingest._stream_des_ingest(_sample_file(), "sample.pdf")))

    names = _names(events)
    assert names[-1] == "error", names
    assert "done" not in names, names
    payload = events[-1][1]
    assert payload["error"] == "azure DI returned 429 after 5 retries"
    # the Ingestion tab reads `message`; the contract names `error` — both ship
    assert payload["message"] == payload["error"]


def test_submit_failure_yields_error_event():
    async def submit(*_a, **_kw):
        raise DesError("DES POST /api/documents returned 503 (expected 202): down")

    async def unused(*_a, **_kw):  # pragma: no cover - must never be reached
        raise AssertionError("polling must not start after a failed submit")

    with _fast_polling(), _patch(
        des_client, submit=submit, get_run=unused, get_events=unused
    ):
        events = _run(_collect(ingest._stream_des_ingest(_sample_file(), "sample.pdf")))

    assert _names(events) == ["start", "error"], _names(events)
    assert "503" in events[-1][1]["message"]


def test_get_run_failure_yields_error_event():
    async def submit(*_a, **_kw):
        return {"document_id": "doc-1", "run_id": "run-1", "name": "sample.pdf"}

    async def get_run(_run_id):
        raise DesError("DES get_run failed: connection refused")

    async def get_events(_run_id):
        return []

    with _fast_polling(), _patch(
        des_client, submit=submit, get_run=get_run, get_events=get_events
    ):
        events = _run(_collect(ingest._stream_des_ingest(_sample_file(), "sample.pdf")))

    assert _names(events)[-1] == "error", _names(events)
    assert "connection refused" in events[-1][1]["error"]


def test_timeout_yields_error_event():
    runs = [{"id": "run-1", "status": "running", "stages": _stage_map("upload")}]
    submit, get_run, get_events = _fake_des(runs=runs)

    # des_timeout=0 -> the deadline passes on the second loop pass.
    with _patch(settings, des_poll_interval=0.0, des_timeout=0), _patch(
        des_client, submit=submit, get_run=get_run, get_events=get_events
    ):
        events = _run(_collect(ingest._stream_des_ingest(_sample_file(), "sample.pdf")))

    assert _names(events)[-1] == "error", _names(events)
    assert "des_timeout" in events[-1][1]["message"]


# ---------------------------------------------------------------------------
# (c) DES returns the whole event log on every poll — we must not replay it
# ---------------------------------------------------------------------------
def test_events_are_not_replayed_across_polls():
    runs = [
        {"id": "run-1", "status": "running", "stages": _stage_map("upload")},
        {"id": "run-1", "status": "running", "stages": _stage_map("upload", "raster")},
        {
            "id": "run-1",
            "status": "succeeded",
            "document_id": "doc-1",
            "document_name": "sample.pdf",
            "stages": _stage_map(
                "upload", "raster", "ocr", "chunk", "context", "embed", "persist"
            ),
            "chunk_count": 7,
        },
    ]
    e1 = {"seq": 1, "stage": "upload", "status": "ok", "detail": {}}
    e2 = {"seq": 2, "stage": "raster", "status": "ok", "detail": {"pages": 3}}
    e3 = {"seq": 3, "stage": "ocr", "status": "ok", "detail": {}}
    event_polls = [[e1], [e1, e2], [e1, e2, e3]]  # cumulative, as DES serves it
    submit, get_run, get_events = _fake_des(runs=runs, event_polls=event_polls)

    with _fast_polling(), _patch(
        des_client, submit=submit, get_run=get_run, get_events=get_events
    ):
        events = _run(_collect(ingest._stream_des_ingest(_sample_file(), "sample.pdf")))

    seqs = [p["seq"] for n, p in events if n == "info" and "seq" in p]
    assert seqs == [1, 2, 3], seqs               # each event exactly once...
    assert seqs == sorted(set(seqs)), seqs       # ...and in order, no dupes


# ---------------------------------------------------------------------------
# (d) /sources
# ---------------------------------------------------------------------------
def test_sources_reports_des_unavailable_when_disabled():
    async def health():  # pragma: no cover - must never be reached
        raise AssertionError("health must not be probed while des_enabled=False")

    with _patch(settings, des_enabled=False), _patch(des_client, health=health):
        body = _run(ingest.ingest_sources())

    by_id = {s["id"]: s for s in body["sources"]}
    assert set(by_id) == {"wega", "des"}
    assert by_id["des"]["available"] is False
    assert by_id["des"]["schema_match"] is None
    assert by_id["wega"]["available"] is True
    assert by_id["wega"]["detail"] in {"remote", "local"}


def test_sources_flags_schema_mismatch():
    async def matching():
        return {"ok": True, "vector_schema": settings.pg_schema, "error": None}

    async def mismatched():
        return {"ok": True, "vector_schema": "some_other_schema", "error": None}

    async def silent():  # a DES build that does not report its schema
        return {"ok": True, "vector_schema": None, "error": None}

    with _patch(settings, des_enabled=True):
        with _patch(des_client, health=matching):
            des = {s["id"]: s for s in _run(ingest.ingest_sources())["sources"]}["des"]
            assert des["available"] is True and des["schema_match"] is True
        with _patch(des_client, health=mismatched):
            des = {s["id"]: s for s in _run(ingest.ingest_sources())["sources"]}["des"]
            assert des["schema_match"] is False   # chunks land where we never read
        with _patch(des_client, health=silent):
            des = {s["id"]: s for s in _run(ingest.ingest_sources())["sources"]}["des"]
            assert des["schema_match"] is None    # unknown, not "fine"


# ---------------------------------------------------------------------------
# (e) des_client.submit contract
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload or {})

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient; records the calls it was handed."""

    def __init__(self, response: _FakeResponse, log: list[dict]):
        self._response = response
        self._log = log

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def post(self, url, **kwargs):
        self._log.append({"method": "POST", "url": url, **kwargs})
        return self._response

    async def get(self, url, **kwargs):
        self._log.append({"method": "GET", "url": url, **kwargs})
        return self._response


def _fake_httpx(response: _FakeResponse, log: list[dict]):
    def factory(*_a, **_kw):
        return _FakeAsyncClient(response, log)

    class _Stub:
        AsyncClient = staticmethod(factory)
        HTTPError = httpx.HTTPError

    return _Stub


def test_submit_raises_des_error_on_non_202():
    log: list[dict] = []
    stub = _fake_httpx(_FakeResponse(500, text="boom: azure key missing"), log)

    raised: DesError | None = None
    with _patch(des_client, httpx=stub):
        try:
            _run(des_client.submit(b"%PDF", "sample.pdf", "application/pdf"))
        except DesError as exc:
            raised = exc

    assert raised is not None, "submit must raise DesError on a non-202 answer"
    assert raised.status_code == 500
    assert "500" in str(raised) and "boom" in str(raised)
    assert log and log[0]["url"].endswith("/api/documents")


def test_submit_returns_202_body_and_sends_api_key():
    body = {
        "document_id": "doc-9",
        "run_id": "run-9",
        "status": "queued",
        "name": "sample.pdf",
        "size_bytes": 4,
        "sha256": "deadbeef",
        "s3_uri": None,
    }
    log: list[dict] = []
    stub = _fake_httpx(_FakeResponse(202, payload=body), log)

    with _patch(settings, des_api_key="sekret"), _patch(des_client, httpx=stub):
        out = _run(des_client.submit(b"%PDF", "sample.pdf", "application/pdf"))

    assert out == body
    assert log[0]["headers"]["X-API-Key"] == "sekret"
    assert log[0]["files"]["file"][0] == "sample.pdf"

    # ...and no header at all when DES is unauthenticated
    log.clear()
    with _patch(settings, des_api_key=""), _patch(des_client, httpx=stub):
        _run(des_client.submit(b"%PDF", "sample.pdf", None))
    assert log[0]["headers"] == {}


def test_health_never_raises_when_des_is_down():
    class _Exploding:
        HTTPError = httpx.HTTPError

        @staticmethod
        def AsyncClient(*_a, **_kw):
            raise httpx.ConnectError("connection refused")

    with _patch(des_client, httpx=_Exploding):
        out = _run(des_client.health())

    assert out["ok"] is False
    assert "connection refused" in out["error"]
    assert out["vector_schema"] is None


def test_health_extracts_vector_schema_from_readyz():
    log: list[dict] = []
    stub = _fake_httpx(
        _FakeResponse(200, payload={"ready": True, "db": {"ok": True, "vector_schema": "vector"}}),
        log,
    )
    with _patch(des_client, httpx=stub):
        out = _run(des_client.health())

    assert out["ok"] is True
    assert out["vector_schema"] == "vector"
    assert log[0]["url"].endswith("/api/readyz")


# ---------------------------------------------------------------------------
# standalone runner (so this file works without pytest installed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback

    failed = 0
    for _name, _fn in list(globals().items()):
        if not _name.startswith("test_") or not callable(_fn):
            continue
        try:
            _fn()
        except Exception:  # noqa: BLE001 - this IS the test reporter
            failed += 1
            print(f"FAIL {_name}")
            traceback.print_exc()
        else:
            print(f"pass {_name}")
    print("FAILED" if failed else "OK")
    sys.exit(1 if failed else 0)
