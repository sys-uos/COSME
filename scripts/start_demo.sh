#!/usr/bin/env bash
# One-shot entry point for the COSME demo: prepares one-time data (media
# assets, OSRM), then brings up the fully-dockerized stack (backend,
# frontend, endpoints, probe, OSRM) with a single command.
#
# Usage: ./scripts/start_demo.sh [--skip-osrm] [--down]
#   --skip-osrm  don't run scripts/prepare_osrm.sh even if OSRM data is
#                missing (useful for a quick non-routing demo/dev loop)
#   --down       stop and remove all COSME containers, then exit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$REPO_ROOT/docker"

SKIP_OSRM=0
DOWN=0
for arg in "$@"; do
  case "$arg" in
    --skip-osrm) SKIP_OSRM=1 ;;
    --down) DOWN=1 ;;
  esac
done

if [ "$DOWN" = "1" ]; then
  echo "==> Stopping the COSME stack..."
  (cd "$DOCKER_DIR" && docker compose down)
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found on PATH. Install Docker first." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: docker daemon not reachable (permission or not running?)." >&2
  exit 1
fi

echo "==> Media assets (Big Buck Bunny)..."
"$SCRIPT_DIR/prepare_media_assets.sh"

OSRM_DONE_MARKER="$DOCKER_DIR/osrm-data/germany-latest.osrm.cell_metrics"
if [ -f "$OSRM_DONE_MARKER" ]; then
  echo "==> OSRM data already prepared (found $OSRM_DONE_MARKER), skipping prepare_osrm.sh."
elif [ "$SKIP_OSRM" = "1" ]; then
  echo "==> --skip-osrm passed: custom-route simulation will not work until you run"
  echo "    scripts/prepare_osrm.sh yourself."
else
  echo "==> OSRM data not found -- running the one-time preprocessing."
  echo "    (Large: multi-GB download, long-running extract/partition/customize."
  echo "     Pass --skip-osrm to defer this to later.)"
  "$SCRIPT_DIR/prepare_osrm.sh"
fi

echo "==> Loading the tcp_bbr kernel module on the host (best-effort; BBR just"
echo "    won't be selectable if this fails, e.g. no root or unsupported kernel)..."
sudo modprobe tcp_bbr 2>/dev/null || echo "    (skipped: could not modprobe tcp_bbr)"

echo "==> Building and starting the full stack (this includes the backend and"
echo "    frontend containers -- nothing needs to run on the host anymore)..."
(cd "$DOCKER_DIR" && docker compose up -d --build)

echo "==> Waiting for the backend to come up..."
for _ in $(seq 1 60); do
  if curl -s -m 2 http://127.0.0.1:8731/api/health 2>/dev/null | grep -q '"status":"ok"'; then
    break
  fi
  sleep 1
done
if ! curl -s -m 2 http://127.0.0.1:8731/api/health 2>/dev/null | grep -q '"status":"ok"'; then
  echo "ERROR: backend did not become healthy -- check 'docker logs cosme-backend'." >&2
  exit 1
fi

if [ -f "$OSRM_DONE_MARKER" ]; then
  echo "==> Waiting for OSRM to load its dataset and answer a real route request"
  echo "    (mmap-ing several GB of index files can take 30-60s after a fresh start)..."
  OSRM_OK=0
  for _ in $(seq 1 90); do
    if curl -sf -m 3 "http://localhost:5000/route/v1/car/8.05,52.27;9.73,52.37?overview=false" >/dev/null 2>&1; then
      OSRM_OK=1
      break
    fi
    sleep 2
  done
  if [ "$OSRM_OK" = "0" ]; then
    echo "WARNING: OSRM did not answer a test route after 3 minutes -- check 'docker logs cosme-osrm'." >&2
  fi
fi

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo ""
echo "==> COSME demo is up."
echo "    Dashboard:  http://${HOST_IP:-<this-host>}:8732/index.html"
echo "    Backend:    http://${HOST_IP:-<this-host>}:8731/api/health"
echo "    Stop with:  ./scripts/start_demo.sh --down"
