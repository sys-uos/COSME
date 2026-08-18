"""Remote-surveillance showcase: one-way UDP video streaming through the
emulated link, with real measured freeze/bitrate QoE.

Transport decision (documented trade-off, see docs/APPLICATIONS.md): plain
MPEG-TS over UDP rather than RTSP. RTSP would add a dedicated media server
(e.g. mediamtx) plus a TCP control channel, and its RTP media path largely
duplicates territory the WebRTC showcases already cover; raw MPEG-TS/UDP is
genuinely UDP end-to-end with zero extra services, and packet loss maps
directly onto the decode gaps / freezes we want to measure. The trade-off:
no receiver-driven session setup, so this module starts the receiver probe
first, then pushes from the sender.

Topology: ffmpeg in `cosme-server` streams the Big Buck Bunny asset
(`-stream_loop -1`, stream-copied H.264 in MPEG-TS) to
`udp://10.42.0.11:9500`; `surveillance_probe.py` (bind-mounted into the
endpoint containers at /opt/cosme, see docker/docker-compose.yml) receives
and decodes it in `cosme-client`, wall-clock-stamps every decoded frame via
ffmpeg's showinfo filter, POSTs per-second {fps, bitrate_kbps, in_freeze}
samples to the backend, and prints an end-of-run JSON summary (freeze
events = frame-arrival gaps > 200ms, see backend/showcases/qoe.py:freeze_stats).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass

from backend.showcases._watchdog import ensure_containers_healthy

SENDER_CONTAINER = "cosme-server"
RECEIVER_CONTAINER = "cosme-client"
RECEIVER_IP = "10.42.0.11"  # cosme-client's fixed address (docker/docker-compose.yml)
STREAM_PORT = 9500
ASSET_IN_CONTAINER = "/srv/media/bigbuckbunny.mp4"
PROBE_IN_CONTAINER = "/opt/cosme/surveillance_probe.py"
BACKEND_STATS_URL = os.environ.get(
    "COSME_STATS_URL", "http://10.42.0.1:8731/api/showcase/app-stats"
)
BACKEND_LIVE_URL = os.environ.get(
    "COSME_LIVE_URL", "http://10.42.0.1:8731/api/showcase/live-frame"
) + "?app=surveillance"


class ShowcaseError(RuntimeError):
    pass


@dataclass
class SurveillanceResult:
    duration_s: float
    frames_received: int
    freeze_count: int
    total_freeze_s: float
    longest_freeze_s: float
    mean_bitrate_kbps: float
    log_tail: str


def _kill_sender() -> None:
    subprocess.run(
        ["docker", "exec", SENDER_CONTAINER, "pkill", "-f", "ffmpeg.*mpegts"],
        capture_output=True, text=True, timeout=10,
    )


def run_surveillance_showcase(duration_s: float = 30.0) -> SurveillanceResult:
    """One real surveillance-stream run: receiver probe first, then the sender.

    Requires the endpoint image to include ffmpeg (see docker/Dockerfile.endpoint)
    and `scripts/prepare_media_assets.sh` to have produced the MP4 asset.
    """
    ensure_containers_healthy(SENDER_CONTAINER, RECEIVER_CONTAINER)
    receiver = subprocess.Popen(
        ["docker", "exec", RECEIVER_CONTAINER, "python3", PROBE_IN_CONTAINER,
         "--port", str(STREAM_PORT), "--duration", str(duration_s),
         "--stats-endpoint", BACKEND_STATS_URL, "--live-endpoint", BACKEND_LIVE_URL],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    time.sleep(1.5)  # let the probe's ffmpeg bind the UDP port before pushing

    # -re paces to real time; -c copy avoids transcoding (H.264 into MPEG-TS);
    # pkt_size 1316 keeps TS packets within one MTU (7 x 188 bytes).
    sender_cmd = [
        "docker", "exec", "-d", SENDER_CONTAINER, "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-re", "-stream_loop", "-1", "-i", ASSET_IN_CONTAINER,
        "-c", "copy", "-f", "mpegts",
        f"udp://{RECEIVER_IP}:{STREAM_PORT}?pkt_size=1316",
    ]
    started = subprocess.run(sender_cmd, capture_output=True, text=True, timeout=15)
    if started.returncode != 0:
        receiver.kill()
        raise ShowcaseError(f"could not start ffmpeg sender: {started.stderr}")

    try:
        stdout, stderr = receiver.communicate(timeout=duration_s + 30)
    except subprocess.TimeoutExpired:
        receiver.kill()
        raise ShowcaseError(f"surveillance probe did not finish within {duration_s + 30}s")
    finally:
        _kill_sender()

    if receiver.returncode != 0:
        raise ShowcaseError(f"surveillance probe failed (rc={receiver.returncode}): {stderr[-2000:]}")

    # The probe prints exactly one JSON summary as its last stdout line.
    try:
        summary = json.loads(stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        raise ShowcaseError(f"could not parse probe summary from: {stdout[-500:]!r}") from e

    return SurveillanceResult(
        duration_s=duration_s,
        frames_received=int(summary.get("frames_received", 0)),
        freeze_count=int(summary.get("freeze_count", 0)),
        total_freeze_s=float(summary.get("total_freeze_s", 0.0)),
        longest_freeze_s=float(summary.get("longest_freeze_s", 0.0)),
        mean_bitrate_kbps=float(summary.get("mean_bitrate_kbps", 0.0)),
        log_tail=stdout[-2000:],
    )


if __name__ == "__main__":
    import argparse
    import dataclasses

    parser = argparse.ArgumentParser(description="Run the surveillance showcase.")
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    print(dataclasses.asdict(run_surveillance_showcase(duration_s=args.duration)))
