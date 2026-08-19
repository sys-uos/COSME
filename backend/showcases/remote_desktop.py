"""Remote-desktop (VNC) showcase: real RFB traffic through the emulated link,
with measured keystroke-round-trip latency and effective frame update rate.

Per user decision: TigerVNC server + a vncdotool-based headless Python RFB
client (chosen over a noVNC/browser approach -- native RFB over TCP is the
real application traffic, and a programmatic client gives precise timing).

Topology:
  - `cosme-server` runs `Xtigervnc :1` (rfbport 5901, SecurityTypes None --
    acceptable: the port is bridge-internal, never published to the host)
    plus an xterm printing a 10Hz timestamp loop. The deterministic 10Hz
    screen activity gives the "effective frame update rate" a known ceiling
    (10 fps) that shaping visibly degrades, and the xterm is the echo target
    for keystroke-latency measurement.
  - The probe (`docker/probe/vnc_probe.py`) runs in the dedicated
    `cosme-probe` container -- it shares cosme-client's netns, so its RFB
    traffic is shaped identically to any other client traffic, and its image
    (python:3.12-slim) pip-installs vncdotool, which isn't packaged in
    Debian and so can't go into the minimal endpoint image via apt.

TCP congestion control: the VNC server is the sender of the framebuffer
stream, so CC is set on `cosme-server` -- same sender-side mechanism and
rationale as the file-transfer showcase (see file_transfer.set_congestion_control).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass

from backend.showcases._watchdog import ensure_containers_healthy
from backend.showcases.file_transfer import set_congestion_control

VNC_SERVER_CONTAINER = "cosme-server"
PROBE_CONTAINER = "cosme-probe"
VNC_HOST = "10.42.0.10"  # cosme-server's fixed address
VNC_PORT = 5901
VNC_DISPLAY = ":1"
BACKEND_STATS_URL = os.environ.get(
    "COSME_STATS_URL", "http://10.42.0.1:8731/api/showcase/app-stats"
)
BACKEND_LIVE_URL = os.environ.get(
    "COSME_LIVE_URL", "http://10.42.0.1:8731/api/showcase/live-frame"
) + "?app=remote_desktop"


class ShowcaseError(RuntimeError):
    pass


@dataclass
class RemoteDesktopResult:
    duration_s: float
    n_keystrokes: int                       # keystrokes whose echo came back
    keystroke_latency_ms_median: float | None
    keystroke_latency_ms_p95: float | None
    effective_fps_mean: float | None
    congestion_control: str
    log_tail: str
    # The latency percentiles above cover only round trips that SUCCEEDED, and a keystroke times
    # out exactly when interactivity is worst. These make that censoring visible: on a degraded
    # link the timeout rate, not the survivors' latency, is what the user experiences.
    keystroke_attempts: int = 0
    keystroke_timeouts: int = 0
    keystroke_timeout_pct: float | None = None


def _docker_exec(container: str, *cmd: str, detach: bool = False,
                 timeout_s: float = 30.0, check: bool = True) -> subprocess.CompletedProcess:
    full = ["docker", "exec"] + (["-d"] if detach else []) + [container, *cmd]
    return subprocess.run(full, capture_output=True, text=True, timeout=timeout_s, check=check)


def ensure_vnc_server_running(container: str = VNC_SERVER_CONTAINER) -> None:
    """Idempotently starts Xtigervnc + the two xterms inside the server container.

    Two windows, matching the pixel regions vnc_probe.py samples:
      - "activity" xterm (100x30 at +0+0): prints a timestamp at 10Hz --
        deterministic screen activity giving the effective-fps metric a known
        10 fps ceiling that shaping visibly degrades.
      - "echo" xterm (80x10 at +0+500): an idle shell, the keystroke-echo
        target. Kept spatially separate so the scrolling activity window can
        never masquerade as a keystroke echo. Xvnc runs without a window
        manager, i.e. X PointerRoot focus: the probe focuses this window by
        moving the pointer into it.

    Left running between showcase runs (like the file-transfer HTTP server) --
    repeated runs reuse the same session, keeping run startup fast.
    """
    check = _docker_exec(container, "pgrep", "-f", "Xtigervnc", check=False)
    if check.returncode != 0:
        _docker_exec(
            container, "Xtigervnc", VNC_DISPLAY,
            "-geometry", "1024x768", "-depth", "24",
            "-SecurityTypes", "None", "-rfbport", str(VNC_PORT),
            "-localhost=0",
            detach=True,
        )
    check = _docker_exec(container, "pgrep", "-f", "xterm -title activity", check=False)
    if check.returncode != 0:
        _docker_exec(
            container, "sh", "-c",
            f'sleep 1; DISPLAY={VNC_DISPLAY} xterm -title activity -geometry 100x30+0+0 '
            f'-e "while true; do date +%s.%N; sleep 0.1; done"',
            detach=True,
        )
    check = _docker_exec(container, "pgrep", "-f", "xterm -title echo", check=False)
    if check.returncode != 0:
        _docker_exec(
            container, "sh", "-c",
            f'sleep 1; DISPLAY={VNC_DISPLAY} xterm -title echo -geometry 80x10+0+500 sh',
            detach=True,
        )


def wait_vnc_ready(host: str = VNC_HOST, port: int = VNC_PORT,
                    retries: int = 20, retry_interval_s: float = 0.25) -> None:
    """Polls for the real RFB protocol banner (not just a bare TCP accept) before returning.

    `ensure_vnc_server_running()` starts Xtigervnc/xterms detached with no readiness wait --
    unlike file_transfer.ensure_http_server_running()'s polling pattern, which this mirrors.
    Without it, a cold start races the vncdotool probe's connection attempt against Xtigervnc
    still initializing, which surfaces as "VNC showcase not responding".
    """
    for _ in range(retries):
        try:
            with socket.create_connection((host, port), timeout=retry_interval_s) as s:
                if s.recv(4).startswith(b"RFB "):
                    return
        except OSError:
            pass
        time.sleep(retry_interval_s)
    raise ShowcaseError(f"VNC server at {host}:{port} did not answer the RFB banner after "
                         f"{retries * retry_interval_s:.1f}s")


def run_remote_desktop_showcase(
    duration_s: float = 30.0,
    congestion_control: str = "cubic",
) -> RemoteDesktopResult:
    """One real VNC session: connect, type keystrokes, measure echo latency + fps."""
    ensure_containers_healthy(VNC_SERVER_CONTAINER, PROBE_CONTAINER)
    set_congestion_control(congestion_control, VNC_SERVER_CONTAINER)
    ensure_vnc_server_running()
    wait_vnc_ready()

    try:
        result = subprocess.run(
            ["docker", "exec", PROBE_CONTAINER, "python3", "/app/vnc_probe.py",
             "--host", VNC_HOST, "--port", str(VNC_PORT),
             "--duration", str(duration_s),
             "--stats-endpoint", BACKEND_STATS_URL, "--live-endpoint", BACKEND_LIVE_URL],
            capture_output=True, text=True, timeout=duration_s + 60, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise ShowcaseError(f"VNC probe failed: {e.stderr[-2000:]}") from e
    except subprocess.TimeoutExpired:
        raise ShowcaseError(f"VNC probe did not finish within {duration_s + 60}s")

    try:
        summary = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        raise ShowcaseError(f"could not parse probe summary from: {result.stdout[-500:]!r}") from e

    return RemoteDesktopResult(
        duration_s=duration_s,
        n_keystrokes=int(summary.get("n_keystrokes", 0)),
        keystroke_latency_ms_median=summary.get("keystroke_latency_ms_median"),
        keystroke_latency_ms_p95=summary.get("keystroke_latency_ms_p95"),
        effective_fps_mean=summary.get("effective_fps_mean"),
        congestion_control=congestion_control,
        log_tail=result.stdout[-2000:],
        keystroke_attempts=int(summary.get("keystroke_attempts", 0)),
        keystroke_timeouts=int(summary.get("keystroke_timeouts", 0)),
        keystroke_timeout_pct=summary.get("keystroke_timeout_pct"),
    )


if __name__ == "__main__":
    import argparse
    import dataclasses

    parser = argparse.ArgumentParser(description="Run the remote-desktop (VNC) showcase.")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--cc", default="cubic", choices=["cubic", "bbr", "reno"])
    args = parser.parse_args()

    print(dataclasses.asdict(run_remote_desktop_showcase(duration_s=args.duration, congestion_control=args.cc)))
