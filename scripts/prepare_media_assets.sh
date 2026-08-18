#!/usr/bin/env bash
# One-time media asset prep for the real-application showcases:
#   - file-transfer showcase serves the MP4 as-is over HTTP
#   - video-conferencing showcase feeds Chromium's
#     --use-file-for-fake-video-capture/--use-file-for-fake-audio-capture
#     flags, which require raw Y4M (video) and WAV (audio) -- not MP4 --
#     hence the ffmpeg conversion below.
#
# Big Buck Bunny (Blender Foundation, CC BY 3.0) is used for both showcases:
# one real, freely-licensed asset, two reuses. Confirmed reachable from this
# repo's dev environment at time of writing (~64MB, 320x180 h264/aac mp4).
#
# Usage: ./scripts/prepare_media_assets.sh
# Requires: curl, ffmpeg.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_DIR="$SCRIPT_DIR/../docker/media"
SOURCE_URL="https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4"
MP4_FILE="bigbuckbunny.mp4"
Y4M_FILE="bigbuckbunny.y4m"
WAV_FILE="bigbuckbunny.wav"

mkdir -p "$MEDIA_DIR"
# Must exist as a real directory BEFORE compose brings up cosme-server: docker-compose.yml
# mounts a writable tmpfs at /srv/media/sized (nested inside the read-only /srv/media bind
# mount, for the file-transfer showcase's exact-size asset generation, see
# backend/showcases/file_transfer.py's _ensure_sized_asset) -- Docker can only mount tmpfs
# onto a mountpoint that already exists as a directory entry in the bind-mounted source;
# it can't mkdir one itself once the parent is read-only.
mkdir -p "$MEDIA_DIR/sized"
cd "$MEDIA_DIR"

if [ ! -f "$MP4_FILE" ]; then
  echo "==> Downloading Big Buck Bunny (CC BY 3.0, Blender Foundation) from $SOURCE_URL"
  curl -L -o "$MP4_FILE" "$SOURCE_URL"
else
  echo "==> $MP4_FILE already present, skipping download"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found. Install it (e.g. 'apt install ffmpeg') and re-run." >&2
  exit 1
fi

if [ ! -f "$Y4M_FILE" ]; then
  echo "==> Converting to Y4M for Chromium's --use-file-for-fake-video-capture"
  ffmpeg -y -i "$MP4_FILE" -pix_fmt yuv420p "$Y4M_FILE"
else
  echo "==> $Y4M_FILE already present, skipping"
fi

if [ ! -f "$WAV_FILE" ]; then
  echo "==> Extracting audio to WAV for Chromium's --use-file-for-fake-audio-capture"
  ffmpeg -y -i "$MP4_FILE" -vn -acodec pcm_s16le -ar 48000 -ac 1 "$WAV_FILE"
else
  echo "==> $WAV_FILE already present, skipping"
fi

echo "==> Done. Assets in $MEDIA_DIR:"
ls -la "$MEDIA_DIR"
echo "==> These are bind-mounted read-only into cosme-server/cosme-client (/srv/media)"
echo "==> -- see docker/docker-compose.yml."
