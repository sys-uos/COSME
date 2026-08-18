"""Shared container-health/readiness helpers for the showcases.

Two distinct failure classes this addresses:
  1. A container isn't RUNNING at all (crashed, never started, or was manually stopped) --
     `ensure_containers_healthy()`, meant to be the FIRST thing every `run_*_showcase()` does, so
     a missing/crashed container fails fast with a clear message instead of the underlying
     operation (curl/vncdotool/ffmpeg/aiortc) failing confusingly deep inside its own error path.
  2. A container is running but the specific SERVICE a showcase needs isn't ready yet (a cold
     start racing the showcase, or a slow-to-initialize server) -- `wait_until_ready()`,
     generalizing the polling pattern `file_transfer.ensure_http_server_running()` and
     `remote_desktop.wait_vnc_ready()` already used ad hoc (both real, confirmed-live fixes for
     real "showcase not responding" reports -- see those modules' own docstrings).

Deliberately NOT a Docker-level `healthcheck:` for `cosme-client`/`cosme-server`/`cosme-probe`
(see docker-compose.yml's comment on why) -- those containers have no persistent app of their
own, so "is it running" (checked here) is the only FIXED thing worth checking at the container
level; readiness of whatever a specific showcase provisions on top is checked per-call instead.
"""
from __future__ import annotations

import subprocess
import time
from typing import Callable


class WatchdogError(RuntimeError):
    pass


def container_health_status(name: str, timeout_s: float = 5.0) -> str:
    """One of: "healthy", "unhealthy", "starting", "none" (running, no healthcheck defined --
    the expected/normal case for cosme-client/-server/-probe), "stopped", or "missing"."""
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f",
             "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}", name],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return "missing"
    if out.returncode != 0:
        return "missing"
    status, _, health = out.stdout.strip().partition("|")
    if status != "running":
        return "stopped"
    return health or "none"


def ensure_containers_healthy(*names: str) -> None:
    """Raises WatchdogError for the first container that's missing/stopped/unhealthy.

    Call this FIRST in every run_*_showcase(), before touching curl/vncdotool/ffmpeg/aiortc --
    a clear, actionable error here beats a confusing failure several layers deep in a
    third-party client library (exactly what made the real VNC hangs hard to diagnose).
    """
    for name in names:
        status = container_health_status(name)
        if status == "missing":
            raise WatchdogError(
                f"{name} is not running (container not found) -- is the demo stack up? "
                f"See ./scripts/start_demo.sh."
            )
        if status == "stopped":
            raise WatchdogError(f"{name} is not running -- see `docker logs {name}`.")
        if status == "unhealthy":
            raise WatchdogError(
                f"{name} is unhealthy -- see `docker logs {name}`, or try "
                f"POST /api/system/reset-containers."
            )


def wait_until_ready(check_fn: Callable[[], bool], timeout_s: float, retry_interval_s: float = 0.25,
                      description: str = "service") -> None:
    """Polls `check_fn()` until it returns True or `timeout_s` elapses.

    `check_fn` should be cheap and swallow its own transient errors by returning False (a
    connection-refused/timeout while a service is still starting up is expected, not a bug) --
    this wrapper itself also treats any exception from `check_fn` as "not ready yet" so callers
    don't need their own try/except around every check.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if check_fn():
                return
        except Exception:
            pass
        time.sleep(retry_interval_s)
    raise WatchdogError(f"{description} did not become ready within {timeout_s:.0f}s")


# app_id -> [(container, pkill pattern), ...] -- the actual per-RUN process to kill to cancel
# that showcase early. Deliberately does NOT touch any long-lived service a showcase provisions
# ONCE and reuses across runs (cosme-server's http.server for file_transfer, Xtigervnc/xterms
# for remote_desktop, see those modules' own ensure_*_running()) -- killing those would break
# the NEXT run too, not just cancel this one. Patterns match each showcase's own existing
# cleanup-on-exit pkill calls exactly (video_conferencing.py/surveillance.py already had these
# for their OWN normal-exit cleanup; file_transfer/remote_desktop didn't need one before since
# nothing could cancel them mid-run).
SHOWCASE_CANCEL_TARGETS: dict[str, list[tuple[str, str]]] = {
    "file_transfer": [("cosme-client", "curl")],
    "remote_desktop": [("cosme-probe", "vnc_probe.py")],
    "video_conferencing": [("cosme-client", "webrtc_peer"), ("cosme-server", "webrtc_peer")],
    "voip": [("cosme-client", "webrtc_peer"), ("cosme-server", "webrtc_peer")],
    "surveillance": [("cosme-client", "surveillance_probe"), ("cosme-server", "ffmpeg.*mpegts")],
}


def cancel_showcase(app_id: str) -> None:
    """Best-effort: kills the in-flight per-run process(es) for `app_id`.

    This makes the showcase's own blocking subprocess call (inside run_*_showcase(), running in
    a background thread via asyncio.to_thread) fail with a real, immediate error instead of
    running to completion -- api.py's `_run_showcase_job` already catches that into a normal
    "error" job status, this doesn't need its own error handling beyond "don't crash the caller
    if a container/process is already gone" (`check=False`, swallow the resulting non-zero exit
    silently -- pkill matching nothing is the expected case once the process has already exited).
    """
    for container, pattern in SHOWCASE_CANCEL_TARGETS.get(app_id, []):
        subprocess.run(["docker", "exec", container, "pkill", "-f", pattern],
                        capture_output=True, timeout=10, check=False)
