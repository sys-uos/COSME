#!/usr/bin/env bash
# One-time preprocessing for the self-hosted, whole-Germany OSRM instance
# backend/models/oblos_live.py talks to (matching the ObLoS website's own
# hardcoded http://localhost:5000 default).
#
# Per project decision this covers the *entire* country, not a clipped
# regional extract -- deliberately: it's a real one-time cost (expect a
# large download, substantial RAM, and a long wall-clock run for the
# osrm-partition/osrm-customize (MLD) steps), so run this well ahead of the
# conference, on a capable machine, NOT live on-site.
#
# Each of the three preprocessing stages (extract/partition/customize) is
# individually skipped if its output already exists and is newer than its
# input -- so re-running this script after it already succeeded once (e.g.
# a fresh `start_demo.sh` on the conference laptop, or after fixing an
# unrelated script bug) completes in seconds instead of redoing hours of
# work. A newer PBF (re-downloaded, or a fresher regional extract) correctly
# invalidates everything downstream via the mtime chain.
#
# Usage: ./scripts/prepare_osrm.sh [--force]
#   --force  rebuild every stage regardless of what's already on disk
# Requires: docker, ~50GB free disk, and patience (see notes below).
set -euo pipefail

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../docker/osrm-data"
PBF_URL="https://download.geofabrik.de/europe/germany-latest.osm.pbf"
PBF_FILE="germany-latest.osm.pbf"
OSRM_BASE="${PBF_FILE%.osm.pbf}.osrm"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# True if $1 exists, and (no $2, or $2 doesn't exist, or $1 is newer than $2).
_up_to_date() {
  [ -f "$1" ] && { [ ! -f "${2:-/nonexistent}" ] || [ "$1" -nt "$2" ]; }
}

if [ ! -f "$PBF_FILE" ]; then
  echo "==> Downloading $PBF_URL (several GB, one-time)"
  curl -L -o "$PBF_FILE" "$PBF_URL"
else
  echo "==> $PBF_FILE already present, skipping download"
fi

if [ "$FORCE" = "0" ] && _up_to_date "$OSRM_BASE" "$PBF_FILE"; then
  echo "==> osrm-extract: $OSRM_BASE is up to date with $PBF_FILE, skipping"
else
  echo "==> osrm-extract (car profile) -- this reads the whole-Germany PBF, expect a long run"
  docker run --rm -t -v "$DATA_DIR:/data" osrm/osrm-backend \
    osrm-extract -p /opt/car.lua "/data/$PBF_FILE"
fi

if [ "$FORCE" = "0" ] && _up_to_date "$OSRM_BASE.partition" "$OSRM_BASE"; then
  echo "==> osrm-partition: $OSRM_BASE.partition is up to date, skipping"
else
  echo "==> osrm-partition (MLD)"
  docker run --rm -t -v "$DATA_DIR:/data" osrm/osrm-backend \
    osrm-partition "/data/$OSRM_BASE"
fi

if [ "$FORCE" = "0" ] && _up_to_date "$OSRM_BASE.cell_metrics" "$OSRM_BASE.partition"; then
  echo "==> osrm-customize: $OSRM_BASE.cell_metrics is up to date, skipping"
else
  echo "==> osrm-customize (MLD)"
  docker run --rm -t -v "$DATA_DIR:/data" osrm/osrm-backend \
    osrm-customize "/data/$OSRM_BASE"
fi

# docker-compose.yml expects the fixed name germany-latest.osrm*; rename if
# the PBF filename above ever changes.
if [ ! -f "germany-latest.osrm" ]; then
  echo "==> ERROR: expected germany-latest.osrm* output files but found:"
  ls -la "$DATA_DIR"
  exit 1
fi

echo "==> Done. Start the stack with: cd docker && docker compose up -d"
echo "==> Verify with: curl 'http://localhost:5000/route/v1/car/8.05,52.27;9.73,52.37?overview=false'"
