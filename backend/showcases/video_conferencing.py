"""Video-conferencing and VoIP showcases: a real two-peer WebRTC call
(aiortc) between cosme-client and cosme-server, through the emulated link.

Media plane: a direct aiortc WebRTC session (docker/endpoint-scripts/webrtc_peer.py):

  - both peers stream the real Big Buck Bunny asset (audio-only for VoIP);
  - real RTP/RTCP crosses both tc/netem-shaped interfaces (client<->server);
  - received QoE is measured, not estimated: framerate/resolution from the
    actually-decoded frames, loss/jitter from RTCP, RTT from remote-inbound;
  - per-second samples are POSTed to the backend's app-stats endpoint at the
    docker bridge gateway (requires uvicorn on 0.0.0.0, see README.md).

(History: an earlier revision drove this via a self-hosted Jitsi Meet stack
with a headless-Chromium bot; it reached joined conferences with completed
ICE/DTLS, but the JVB never sustainably forwarded media in the shared-netns
container topology. See docs/APPLICATIONS.md for a short account. The Jitsi
stack has been removed from the repo.)
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Literal

from backend.showcases._watchdog import ensure_containers_healthy

SERVER_CONTAINER = "cosme-server"
CLIENT_CONTAINER = "cosme-client"
SERVER_IP = "10.42.0.10"  # matches docker/docker-compose.yml's fixed address
PEER_SCRIPT = "/opt/cosme/webrtc_peer.py"
ASSET_IN_CONTAINER = "/srv/media/bigbuckbunny.mp4"
BACKEND_STATS_URL = os.environ.get(
    "COSME_STATS_URL", "http://10.42.0.1:8731/api/showcase/app-stats"
)
BACKEND_LIVE_URL_BASE = os.environ.get(
    "COSME_LIVE_URL", "http://10.42.0.1:8731/api/showcase/live-frame"
)


class ShowcaseError(RuntimeError):
    pass


@dataclass
class VideoConferencingResult:
    app: str
    duration_s: float
    media_plane: str
    video_frames_received: int
    audio_frames_received: int
    log_tail: str
    # Unbiased end-of-run aggregates computed inside the container by
    # webrtc_peer.py over EVERY sample it generated. The per-second samples
    # POSTed to /api/showcase/app-stats travel over the emulated link and are
    # dropped exactly when the link is worst, so anything averaged from those
    # understates the impairment. Prefer this field for reported QoE; the live
    # samples remain the dashboard's real-time feed.
    totals: dict | None = None


def run_video_conferencing_showcase(
    duration_s: float = 60.0,
    mode: Literal["video", "audio"] = "video",
    # accepted for backward compatibility with older callers/tests; the
    # aiortc peers read the MP4 asset directly.
    video_asset: str | None = None,
    audio_asset: str | None = None,
) -> VideoConferencingResult:
    """One real WebRTC call: answer peer in cosme-server, offer peer in cosme-client."""
    ensure_containers_healthy(SERVER_CONTAINER, CLIENT_CONTAINER)
    app = "video_conferencing" if mode == "video" else "voip"

    # Idempotent cleanup of any stale peer from an aborted previous run.
    subprocess.run(["docker", "exec", SERVER_CONTAINER, "pkill", "-f", "webrtc_peer"],
                   capture_output=True, timeout=10)

    started = subprocess.run(
        ["docker", "exec", "-d", SERVER_CONTAINER, "python3", PEER_SCRIPT,
         "--role", "answer", "--mode", mode, "--asset", ASSET_IN_CONTAINER,
         "--duration", str(duration_s + 10)],
        capture_output=True, text=True, timeout=15,
    )
    if started.returncode != 0:
        raise ShowcaseError(f"could not start answer peer: {started.stderr}")
    time.sleep(1.5)  # let the answer peer bind its signaling socket

    try:
        result = subprocess.run(
            ["docker", "exec", CLIENT_CONTAINER, "python3", PEER_SCRIPT,
             "--role", "offer", "--connect", SERVER_IP, "--mode", mode,
             "--asset", ASSET_IN_CONTAINER, "--duration", str(duration_s),
             "--app", app, "--stats-endpoint", BACKEND_STATS_URL,
             "--live-endpoint", f"{BACKEND_LIVE_URL_BASE}?app={app}"],
            capture_output=True, text=True, timeout=duration_s + 60, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise ShowcaseError(f"WebRTC offer peer failed: {e.stderr[-2000:]}") from e
    except subprocess.TimeoutExpired:
        raise ShowcaseError(f"WebRTC call did not finish within {duration_s + 60}s")
    finally:
        subprocess.run(["docker", "exec", SERVER_CONTAINER, "pkill", "-f", "webrtc_peer"],
                       capture_output=True, timeout=10)

    try:
        summary = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        raise ShowcaseError(f"could not parse peer summary from: {result.stdout[-500:]!r}") from e

    return VideoConferencingResult(
        app=app, duration_s=duration_s, media_plane="webrtc/aiortc (two-peer)",
        video_frames_received=int(summary.get("video_frames_received", 0)),
        audio_frames_received=int(summary.get("audio_frames_received", 0)),
        log_tail=result.stdout[-2000:],
        totals={
            "packets": summary.get("totals"),
            "n_samples_generated": summary.get("n_samples"),
            "rtt_ms_mean": summary.get("rtt_ms_mean"),
            "jitter_ms_mean": summary.get("jitter_ms_mean"),
            "audio_bitrate_kbps_mean": summary.get("audio_bitrate_kbps_mean"),
            "video_bitrate_kbps_mean": summary.get("video_bitrate_kbps_mean"),
            "framerate_mean": summary.get("framerate_mean"),
        },
    )


def run_voip_showcase(
    duration_s: float = 60.0,
    audio_asset: str | None = None,
) -> VideoConferencingResult:
    """VoIP showcase: a genuine audio-only WebRTC call (no video track at all)."""
    return run_video_conferencing_showcase(duration_s=duration_s, mode="audio")


if __name__ == "__main__":
    import argparse
    import dataclasses

    parser = argparse.ArgumentParser(description="Run the video-conferencing or VoIP showcase.")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--mode", choices=["video", "audio"], default="video")
    args = parser.parse_args()

    print(dataclasses.asdict(run_video_conferencing_showcase(duration_s=args.duration, mode=args.mode)))
