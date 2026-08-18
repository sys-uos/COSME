#!/usr/bin/env bash
# End-to-end checklist for everything this repo could NOT verify in its
# development sandbox (no Docker there at all -- confirmed repeatedly this
# session, including with sandboxing explicitly disabled). Run this once on
# the real Docker-capable machine before the conference; each step prints
# PASS/FAIL/SKIP so failures are visible rather than silently swallowed.
#
# Usage: ./scripts/verify_docker_stack.sh [--with-osrm]
#   --with-osrm   also verify the (large, one-time) OSRM stack -- requires
#                 scripts/prepare_osrm.sh to have been run already
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WITH_OSRM=0
for arg in "$@"; do
  case "$arg" in
    --with-osrm) WITH_OSRM=1 ;;
  esac
done

PASS=0
FAIL=0

check() {
  local desc="$1"; shift
  if "$@" >/tmp/cosme_verify_out.$$ 2>&1; then
    echo "PASS: $desc"
    PASS=$((PASS+1))
  else
    echo "FAIL: $desc"
    sed 's/^/       /' /tmp/cosme_verify_out.$$
    FAIL=$((FAIL+1))
  fi
  rm -f /tmp/cosme_verify_out.$$
}

echo "=== 1. Docker basics ==="
check "docker is installed" bash -c "command -v docker"
check "docker daemon is reachable" docker info

echo "=== 2. Base cosme-client/cosme-server stack (+ probe) ==="
(cd "$REPO_ROOT/docker" && docker compose up -d --build)
check "cosme-server container is running" bash -c "docker ps --format '{{.Names}}' | grep -qx cosme-server"
check "cosme-client container is running" bash -c "docker ps --format '{{.Names}}' | grep -qx cosme-client"
check "cosme-probe container is running (shares cosme-client netns)" bash -c "docker ps --format '{{.Names}}' | grep -qx cosme-probe"
check "cosme-server has NET_ADMIN (tc works)" docker exec cosme-server tc qdisc show
check "cosme-client has NET_ADMIN (tc works)" docker exec cosme-client tc qdisc show
check "cosme-client can ping cosme-server" docker exec cosme-client ping -c 2 -W 2 10.42.0.10

# start_period is 10s, so give the healthcheck a moment to actually run once before checking --
# an immediate check right after `up -d` would spuriously see "starting" and fail.
sleep 12
check "cosme-server healthcheck reports healthy (ip route get, not just 'running')" bash -c \
  "[ \"\$(docker inspect -f '{{.State.Health.Status}}' cosme-server)\" = healthy ]"
check "cosme-client healthcheck reports healthy (ip route get, not just 'running')" bash -c \
  "[ \"\$(docker inspect -f '{{.State.Health.Status}}' cosme-client)\" = healthy ]"
check "cosme-probe healthcheck reports healthy (catches the documented netns-staleness bug)" bash -c \
  "[ \"\$(docker inspect -f '{{.State.Health.Status}}' cosme-probe)\" = healthy ]"

echo "=== 3. tc/netem shaping actually changes measured latency ==="
docker exec cosme-server tc qdisc replace dev eth0 root netem delay 200ms >/dev/null 2>&1
docker exec cosme-client tc qdisc replace dev eth0 root netem delay 200ms >/dev/null 2>&1
check "200ms netem delay makes ping RTT >= 350ms (both directions shaped)" \
  bash -c "docker exec cosme-client ping -c 3 -W 3 10.42.0.10 | tail -1 | grep -Eo 'rtt.*' | awk -F/ '{exit (\$5+0 < 350)}'"
docker exec cosme-server tc qdisc del dev eth0 root >/dev/null 2>&1
docker exec cosme-client tc qdisc del dev eth0 root >/dev/null 2>&1

echo "=== 4. TCP congestion control (sysctl, per-netns) ==="
# A plain `docker exec ... sysctl -w` always fails "permission denied" on
# modern Docker/runc (/proc/sys is read-only inside a container's mount
# namespace post-creation, even with NET_ADMIN+CAP_SYS_ADMIN) -- these checks
# go through the same nsenter-based fix as backend/showcases/file_transfer.py's
# set_congestion_control, which needs the host sudoers rule from README.md's
# "Known environment gotchas" (a passwordless sudo scoped to nsenter+sysctl).
check "cubic is available" docker exec cosme-server sysctl net.ipv4.tcp_available_congestion_control
check "can set cubic" bash -c \
  "cd '$REPO_ROOT' && source .venv/bin/activate && python3 -c 'from backend.showcases.file_transfer import set_congestion_control; set_congestion_control(\"cubic\")'"
if docker exec cosme-server sysctl net.ipv4.tcp_available_congestion_control | grep -q bbr; then
  check "can set bbr" bash -c \
    "cd '$REPO_ROOT' && source .venv/bin/activate && python3 -c 'from backend.showcases.file_transfer import set_congestion_control; set_congestion_control(\"bbr\")'"
  (cd "$REPO_ROOT" && source .venv/bin/activate && python3 -c \
    'from backend.showcases.file_transfer import set_congestion_control; set_congestion_control("cubic")') >/dev/null 2>&1
else
  echo "SKIP: bbr not available -- run 'sudo modprobe tcp_bbr' on the HOST and re-check"
fi

echo "=== 5. Backend + orchestrator against the real containers ==="
check "backend Python env has deps" bash -c "cd '$REPO_ROOT' && source .venv/bin/activate && python3 -c 'import fastapi, pandas, shapely'"
check "backend detects docker_available()" bash -c "cd '$REPO_ROOT' && source .venv/bin/activate && python3 -c 'from backend.api import _docker_available; assert _docker_available()'"

echo "=== 6. File-transfer showcase (real HTTP through the shaped link) ==="
if [ ! -f "$REPO_ROOT/docker/media/bigbuckbunny.mp4" ]; then
  echo "SKIP: run scripts/prepare_media_assets.sh first"
else
  check "file-transfer showcase runs and returns real throughput" bash -c \
    "cd '$REPO_ROOT' && source .venv/bin/activate && python3 -m backend.showcases.file_transfer --cc cubic"
fi

echo "=== 6b. WebRTC showcases (aiortc two-peer: video conferencing + VoIP) ==="
if [ ! -f "$REPO_ROOT/docker/media/bigbuckbunny.mp4" ]; then
  echo "SKIP: run scripts/prepare_media_assets.sh first"
else
  check "VoIP showcase (audio-only WebRTC) receives real audio frames" bash -c \
    "cd '$REPO_ROOT' && source .venv/bin/activate && python3 -m backend.showcases.video_conferencing --duration 10 --mode audio | grep -q \"'audio_frames_received': [1-9]\""
fi

echo "=== 6c. Surveillance showcase (MPEG-TS over UDP) ==="
if [ ! -f "$REPO_ROOT/docker/media/bigbuckbunny.mp4" ]; then
  echo "SKIP: run scripts/prepare_media_assets.sh first"
else
  check "surveillance showcase decodes real frames" bash -c \
    "cd '$REPO_ROOT' && source .venv/bin/activate && python3 -m backend.showcases.surveillance --duration 10 | grep -q \"'frames_received': [1-9]\""
fi

if [ "$WITH_OSRM" = "1" ]; then
  echo "=== 7. OSRM (whole Germany, one-time preprocessed) ==="
  (cd "$REPO_ROOT/docker" && docker compose up -d osrm)
  check "osrm responds to a real route request" \
    curl -sf "http://localhost:5000/route/v1/car/8.05,52.27;9.73,52.37?overview=false"
  check "oblos_live.py can use the self-hosted instance" bash -c \
    "cd '$REPO_ROOT' && source .venv/bin/activate && python3 -c \"
from backend.models.oblos_live import fetch_route
r = fetch_route(52.27, 8.05, 52.37, 9.73, osrm_url='http://localhost:5000')
assert len(r.segments) > 0
\""
else
  echo "=== 7. OSRM: SKIPPED (pass --with-osrm to check; needs scripts/prepare_osrm.sh run first) ==="
fi

echo "=== 8. Remote-desktop (VNC) showcase (cosme-probe container) ==="
check "remote-desktop (VNC) showcase measures real keystroke latency" bash -c \
  "cd '$REPO_ROOT' && source .venv/bin/activate && python3 -m backend.showcases.remote_desktop --duration 12 --cc cubic | grep -q \"'n_keystrokes': [1-9]\""

echo "=== 9. Integration test suite (pytest -m docker) ==="
check "pytest -m docker passes against the running stack" bash -c \
  "cd '$REPO_ROOT' && source .venv/bin/activate && python3 -m pytest backend/tests/ -m docker -q"

echo ""
echo "=== Summary: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
