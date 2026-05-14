#!/usr/bin/env bash
# =============================================================================
#  entrypoint.sh — single-command launcher for RAG Studio.
#
#  USAGE
#     ./entrypoint.sh              # production: build frontend if needed, then serve
#     ./entrypoint.sh --dev        # dev mode: vite hot-reload on :5173 + uvicorn --reload
#     ./entrypoint.sh --skip-install --skip-build   # fast restart inside a built image
#     ./entrypoint.sh --help
#
#  WHAT IT DOES
#     1. Source .env (if present) so PG_*, COIN_*, STELLAR_* land in the environment.
#     2. Install Python deps (pip) — skip with --skip-install.
#     3. Install + build the React frontend — skip with --skip-build.
#     4. Start FastAPI (uvicorn). In --dev, also start Vite alongside.
#     5. Apply DB migrations on first boot (handled by app/main.py lifespan).
#
#  Designed to work both bare-metal on the VDI AND as a Docker CMD.
# =============================================================================
set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────
DEV_MODE=false
SKIP_INSTALL=false
SKIP_BUILD=false
HOST="${APP_HOST:-0.0.0.0}"
PORT="${APP_PORT:-8080}"
PYTHON="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
USE_VENV="${USE_VENV:-auto}"   # auto | always | never

# ── flag parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)            DEV_MODE=true ;;
        --skip-install)   SKIP_INSTALL=true ;;
        --skip-build)     SKIP_BUILD=true ;;
        --host)           HOST="$2"; shift ;;
        --port)           PORT="$2"; shift ;;
        --no-venv)        USE_VENV=never ;;
        --venv)           USE_VENV=always ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            echo "[entrypoint] unknown option: $1" >&2
            exit 2
            ;;
    esac
    shift
done

cd "$(cd "$(dirname "$0")" && pwd)"
ROOT="$PWD"

log()  { printf '\033[1;35m[entrypoint]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[entrypoint]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[entrypoint]\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. .env loading ───────────────────────────────────────────────────────
if [[ -f .env ]]; then
    log "loading .env"
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
else
    warn ".env not found — relying on environment variables already exported"
fi

# Required minimum vars for the server to start usefully.
for var in PG_HOST PG_USER PG_PASSWORD PG_DATABASE; do
    if [[ -z "${!var:-}" ]]; then
        warn "$var is empty — the backend will start but DB calls will fail"
    fi
done

# ── 2. Python deps ────────────────────────────────────────────────────────
ACTIVATED_VENV=false
if [[ "$USE_VENV" == "always" ]] || { [[ "$USE_VENV" == "auto" ]] && [[ ! -f /.dockerenv ]]; }; then
    if [[ ! -d "$VENV_DIR" ]]; then
        log "creating venv at $VENV_DIR"
        "$PYTHON" -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PYTHON="python"
    ACTIVATED_VENV=true
fi

if [[ "$SKIP_INSTALL" == "false" ]]; then
    log "installing Python deps (backend/requirements.txt)"
    "$PYTHON" -m pip install --upgrade pip --quiet
    "$PYTHON" -m pip install -r backend/requirements.txt --quiet
fi

# ── 3. Frontend build / dev setup ─────────────────────────────────────────
FRONTEND_DIR="$ROOT/frontend"

ensure_node_modules() {
    if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
        log "installing frontend deps (npm install)"
        ( cd "$FRONTEND_DIR" && npm install --no-audit --no-fund )
    fi
}

if [[ "$DEV_MODE" == "true" ]]; then
    ensure_node_modules
    log "starting Vite dev server on :5173 (proxies /api → :$PORT)"
    ( cd "$FRONTEND_DIR" && npm run dev ) &
    VITE_PID=$!
    trap 'kill "$VITE_PID" 2>/dev/null || true' EXIT INT TERM
else
    if [[ "$SKIP_BUILD" == "false" ]]; then
        ensure_node_modules
        if [[ ! -d "$FRONTEND_DIR/dist" ]] || [[ ! -f "$FRONTEND_DIR/dist/index.html" ]]; then
            log "building frontend (npm run build)"
            ( cd "$FRONTEND_DIR" && npm run build )
        else
            log "frontend/dist exists, skipping build (use --skip-build to always skip, delete dist/ to rebuild)"
        fi
    fi
fi

# ── 4. Sanity check (optional) ────────────────────────────────────────────
if [[ "${ENTRYPOINT_SMOKE_CHECK:-false}" == "true" ]] && [[ -f backend/smoke_check.py ]]; then
    log "running smoke check"
    ( cd backend && "$PYTHON" smoke_check.py ) || warn "smoke check failed — continuing anyway"
fi

# ── 5. Launch backend ─────────────────────────────────────────────────────
RELOAD_FLAG=""
WORKERS_FLAG="--workers 1"
if [[ "$DEV_MODE" == "true" ]]; then
    RELOAD_FLAG="--reload"
    WORKERS_FLAG=""   # --reload is incompatible with multi-workers
    log "starting backend (uvicorn --reload) on $HOST:$PORT"
    log "  → frontend dev: http://localhost:5173"
    log "  → backend api : http://localhost:$PORT/api/health"
else
    log "starting backend on $HOST:$PORT"
    log "  → ui: http://localhost:$PORT"
fi

exec "$PYTHON" -m uvicorn backend.app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "${APP_LOG_LEVEL:-info}" \
    $RELOAD_FLAG $WORKERS_FLAG
