#!/usr/bin/env python3
"""Surveillance-showcase receiver probe: decodes the MPEG-TS/UDP stream and
measures real playback QoE from frame arrival times.

Runs inside cosme-client (bind-mounted at /opt/cosme, see
docker/docker-compose.yml) via `docker exec` from
backend/showcases/surveillance.py. Deliberately stdlib-only: the endpoint
image has no pip packages.

Mechanism: ffmpeg decodes the incoming stream with `-vf showinfo -f null -`,
which logs one stderr line per decoded frame (including `pos:` -- the input
byte offset, giving received bitrate without any packet parsing). A reader
thread wall-clock-stamps each such line; the main loop emits one sample per
second to the backend:

    {"app": "surveillance", "t": ..., "fps": ..., "bitrate_kbps": ..., "in_freeze": ...}

A freeze is a gap between consecutive decoded frames exceeding --freeze-threshold
(default 200ms -- the same definition as backend/showcases/qoe.py:freeze_stats,
which computes the end-of-run summary printed as this script's last stdout line).
Decoder stalls (e.g. TS desync under burst loss) count as freezes by design:
from the viewer's perspective, they are.

When --live-endpoint is given, a SECOND ffmpeg output (same input, decoded
once, standard ffmpeg multi-output pattern) writes a ~2fps still to disk;
the main loop POSTs its bytes each second -- so the dashboard's live view is
the actual decoded picture at that moment, corruption/freezes and all, not
a synthetic placeholder.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request

FRAME_RE = re.compile(r"Parsed_showinfo.*\bn:\s*(\d+).*?\bpos:\s*(\d+|N/A)")
_LIVE_JPG_PATH = "/tmp/cosme_surveillance_live.jpg"


def _post(endpoint: str, data: bytes, content_type: str) -> None:
    try:
        req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": content_type})
        urllib.request.urlopen(req, timeout=3)
    except (urllib.error.URLError, OSError):
        pass  # backend unreachable this tick -- keep measuring


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9500)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--freeze-threshold", type=float, default=0.2)
    parser.add_argument("--stats-endpoint", default=None)
    parser.add_argument("--live-endpoint", default=None,
                        help="POST a ~2fps JPEG still of the decoded stream here each second")
    args = parser.parse_args()

    # fifo_size + overrun_nonfatal keep the UDP demuxer alive under burst
    # loss; nobuffer/low_delay minimize decode-side buffering so frame
    # arrival times reflect the network, not ffmpeg's queues.
    url = f"udp://0.0.0.0:{args.port}?fifo_size=278876&overrun_nonfatal=1"
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-fflags", "nobuffer", "-flags", "low_delay",
        "-i", url,
        "-map", "0:v", "-vf", "showinfo", "-f", "null", "-",
    ]
    if args.live_endpoint:
        # Standard ffmpeg multi-output pattern: the input is decoded once and
        # shared across both output filtergraphs.
        cmd += ["-map", "0:v", "-vf", "fps=2", "-update", "1", "-y", _LIVE_JPG_PATH]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    frame_times: list[float] = []
    last_pos = [0]
    lock = threading.Lock()

    def reader() -> None:
        for line in proc.stderr:  # type: ignore[union-attr]
            m = FRAME_RE.search(line)
            if m:
                with lock:
                    frame_times.append(time.monotonic())
                    if m.group(2) != "N/A":
                        last_pos[0] = int(m.group(2))

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    start = time.monotonic()
    prev_frames = 0
    prev_pos = 0
    last_live_mtime = 0.0
    while time.monotonic() - start < args.duration:
        time.sleep(1.0)
        now = time.monotonic()
        with lock:
            n_frames = len(frame_times)
            pos = last_pos[0]
            last_frame = frame_times[-1] if frame_times else None
        sample = {
            "app": "surveillance",
            "t": time.time(),
            "fps": n_frames - prev_frames,
            "bitrate_kbps": (pos - prev_pos) * 8 / 1000,
            "in_freeze": last_frame is None or (now - last_frame) > args.freeze_threshold,
        }
        prev_frames, prev_pos = n_frames, pos
        if args.stats_endpoint:
            _post(args.stats_endpoint, json.dumps(sample).encode(), "application/json")

        if args.live_endpoint:
            try:
                mtime = os.path.getmtime(_LIVE_JPG_PATH)
                if mtime != last_live_mtime:
                    with open(_LIVE_JPG_PATH, "rb") as f:
                        _post(args.live_endpoint, f.read(), "image/jpeg")
                    last_live_mtime = mtime
            except OSError:
                pass  # ffmpeg hasn't written a still yet -- try again next tick

    proc.kill()

    with lock:
        times = list(frame_times)
        pos = last_pos[0]
    gaps = [b - a for a, b in zip(times, times[1:])]
    freezes = [g for g in gaps if g > args.freeze_threshold]
    elapsed = time.monotonic() - start
    print(json.dumps({
        "frames_received": len(times),
        "freeze_count": len(freezes),
        "total_freeze_s": round(sum(freezes), 2),
        "longest_freeze_s": round(max(freezes), 2) if freezes else 0.0,
        "mean_bitrate_kbps": round(pos * 8 / 1000 / elapsed, 1) if elapsed > 0 else 0.0,
    }))


if __name__ == "__main__":
    main()
