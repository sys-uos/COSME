"""File-transfer showcase: a real HTTP download through the emulated link.

Per the user's decision to use real application traffic instead of iperf3
synthetic profiles: `cosme-server` runs a plain `python3 -m http.server`
serving a real media asset (Big Buck Bunny -- CC-BY-3.0, the same file used
by the video-conferencing showcase, see docs/APPLICATIONS.md), and
`cosme-client` performs a genuine `curl` GET of it through the netem-shaped
link. QoE is derived from the transfer's real, measured numbers (`curl -w`
timing + `ss` retransmit counters), not modeled/estimated.

TCP congestion control is a *sender*-socket property; for a download,
`cosme-server` is the sender, so CC is set there before the transfer starts.
`backend/api.py`'s `Scenario.__init__` calls this same function for a whole
scenario run; this module also exposes it standalone so the showcase can be
driven independently of the dashboard, e.g. from a CLI/notebook.

Verified end-to-end against real Docker: setting it via a plain
`docker exec <container> sysctl -w net.ipv4.tcp_congestion_control=<cc>`
always fails with "permission denied" on modern Docker/runc, even with
NET_ADMIN + CAP_SYS_ADMIN and even when re-setting the value already in
effect -- runc mounts `/proc/sys` read-only inside a container's own mount
namespace post-creation, and that mount can't be remounted read-write from
inside even with CAP_SYS_ADMIN (confirmed: "is write-protected"). The fix
used here is `nsenter -t <container_pid> -n sysctl -w ...` run on the HOST:
`-n` joins only the container's *network* namespace (where the sysctl is
namespaced), so the write goes through the host's own writable `/proc`
mount. This needs the calling process to have passwordless sudo scoped to
this exact command -- see the `/etc/sudoers.d/cosme-nsenter-cc`-style rule
documented in README.md's "Known environment gotchas" -- UNLESS it's already
running as root (e.g. the dockerized backend container, which runs
privileged), in which case `sudo` is skipped entirely (see `_nsenter_cmd`).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass

import requests

from backend.showcases._watchdog import ensure_containers_healthy

DEFAULT_SERVER_CONTAINER = "cosme-server"
DEFAULT_CLIENT_CONTAINER = "cosme-client"
SERVER_INTERNAL_IP = "10.42.0.10"  # matches docker/docker-compose.yml's fixed address
HTTP_PORT = 8081
MEDIA_DIR = "/srv/media"
# The exact server invocation, used both to start it and to recognize it via
# pgrep -- deliberately includes the port so a stale server on an old port
# (e.g. from a prior code version) is never mistaken for "already running".
_HTTP_SERVER_CMD = ["python3", "-m", "http.server", str(HTTP_PORT), "--directory", MEDIA_DIR]
DEST_PATH = "/tmp/cosme_ft_download.bin"
RESULT_PATH = "/tmp/cosme_ft_result.json"
# Unlike the other showcases, this module's polling loop runs in the BACKEND
# process itself (only the `curl` runs inside a container) -- so posting
# live progress is a loopback call to its own FastAPI app, not a cross
# -container hop, and needs no COSME_STATS_URL/bridge-gateway indirection.
BACKEND_STATS_URL = os.environ.get("COSME_STATS_URL", "http://127.0.0.1:8731/api/showcase/app-stats")


class ShowcaseError(RuntimeError):
    pass


def _docker_exec(container: str, *cmd: str, timeout_s: float = 30.0, check: bool = True) -> subprocess.CompletedProcess:
    full = ["docker", "exec", container, *cmd]
    return subprocess.run(full, capture_output=True, text=True, timeout=timeout_s, check=check)


def _container_pid(container: str) -> str:
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Pid}}", container],
        capture_output=True, text=True, timeout=10, check=True,
    )
    return out.stdout.strip()


def _nsenter_cmd(pid: str, *sysctl_args: str) -> list[str]:
    """`nsenter ... sysctl ...`, prefixed with `sudo -n` unless already root.

    Running as root (e.g. the dockerized backend, `privileged: true`) needs no sudo, and so no
    host sudoers rule either.
    """
    cmd = ["nsenter", "-t", pid, "-n", "/usr/sbin/sysctl", *sysctl_args]
    return cmd if os.geteuid() == 0 else ["sudo", "-n", *cmd]


def _current_congestion_control(container: str) -> str | None:
    """Best-effort unprivileged read; None if it can't be determined."""
    try:
        out = _docker_exec(container, "sysctl", "-n", "net.ipv4.tcp_congestion_control", timeout_s=10)
        return out.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def set_congestion_control(cc: str, container: str = DEFAULT_SERVER_CONTAINER) -> None:
    """Set the sender-side TCP congestion control algorithm (server is the sender for a download).

    Skips the privileged write entirely when the container is already on
    `cc` -- this is what makes the common case (default "cubic", already the
    kernel default) work even on a host that never set up the nsenter
    sudoers rule; see the module docstring for why a plain
    `docker exec ... sysctl -w` can't be used instead.
    """
    if _current_congestion_control(container) == cc:
        return
    try:
        pid = _container_pid(container)
        subprocess.run(
            _nsenter_cmd(pid, "-w", f"net.ipv4.tcp_congestion_control={cc}"),
            capture_output=True, text=True, timeout=10, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise ShowcaseError(
            f"Could not set congestion control to {cc!r} -- if this is BBR, confirm "
            f"`modprobe tcp_bbr` has been run on the HOST (containers can't load kernel modules "
            f"themselves); otherwise confirm the host has the passwordless-sudo nsenter rule from "
            f"README.md's 'Known environment gotchas' set up. stderr: {e.stderr}"
        ) from e


def ensure_http_server_running(container: str = DEFAULT_SERVER_CONTAINER,
                               retries: int = 20, retry_interval_s: float = 0.25) -> None:
    """Starts `python3 -m http.server` on HTTP_PORT if not already listening there.

    Matches on the exact command (including the port) rather than a bare
    "http.server" substring, and kills any OTHER http.server first: a stale
    server from a previous code version (different port/directory) would
    otherwise satisfy a loose pgrep check forever and silently starve the
    real one of the port. After starting, polls the port with a real
    in-container curl until it answers -- callers get a clean failure
    instead of racing a not-yet-listening server.
    """
    running = _docker_exec(container, "pgrep", "-af", "http.server", check=False)
    already_correct = False
    for line in running.stdout.splitlines():
        if " ".join(_HTTP_SERVER_CMD) in line:
            already_correct = True
        else:
            pid = line.split(None, 1)[0]
            _docker_exec(container, "kill", pid, check=False)
    if already_correct:
        return

    subprocess.run(
        ["docker", "exec", "-d", container, *_HTTP_SERVER_CMD],
        capture_output=True, text=True, timeout=10,
    )
    for _ in range(retries):
        probe = _docker_exec(
            container, "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
            f"http://localhost:{HTTP_PORT}/", check=False,
        )
        if probe.stdout.strip() not in ("", "000"):
            return
        time.sleep(retry_interval_s)
    raise ShowcaseError(
        f"http.server on port {HTTP_PORT} in {container} did not come up after "
        f"{retries * retry_interval_s:.1f}s"
    )


_CURL_FORMAT = (
    '{"time_total":%{time_total},"size_download":%{size_download},'
    '"speed_download":%{speed_download},"http_code":%{http_code}}'
)


@dataclass
class FileTransferResult:
    duration_s: float
    size_bytes: float
    throughput_bps: float
    http_code: int
    tcp_retransmits: int | None
    congestion_control: str


SIZED_ASSET_DIR = "sized"  # subdir of MEDIA_DIR -- auto-served by the existing http.server


_SIZED_ASSET_MIN_TIMEOUT_S = 120.0
_SIZED_ASSET_ASSUMED_MIN_MBPS = 20.0  # conservative disk-write throughput floor for the timeout budget
# Floor throughput assumed when deriving a transfer's timeout from its size. Deliberately far
# below anything the emulator produces: its only job is to catch a hung transfer, not to cap a
# slow one. At 1 GB this allows ~72 min before giving up.
_TRANSFER_ASSUMED_MIN_MBPS = 2.0


def _ensure_sized_asset(container: str, size_bytes: int, source_asset: str = "bigbuckbunny.mp4") -> str:
    """Builds (once, cached) a REAL-content file of exactly `size_bytes`.

    Repeats `source_asset`'s bytes (truncating the last copy to land on the
    exact size) rather than generating synthetic/random data -- the project's
    design choice was real application traffic over synthetic profiles (see
    module docstring), and this keeps that true even when the user picks a
    transfer size bigger than any single real media asset (the bundled
    bigbuckbunny.mp4 is only ~65MB). Idempotent: a request for a size that's
    already been built reuses the cached file at `MEDIA_DIR/sized/<bytes>.bin`
    instead of regenerating it.

    Writes to `MEDIA_DIR/sized`, which docker-compose.yml mounts as a disk-backed bind mount,
    NOT tmpfs: tmpfs caps at 50% of host RAM, which truncates large requests part-way. Each
    `cat` is checked so a write failure raises immediately instead of looping to the timeout.
    """
    rel_path = f"{SIZED_ASSET_DIR}/{size_bytes}.bin"
    dest = f"{MEDIA_DIR}/{rel_path}"
    src = f"{MEDIA_DIR}/{source_asset}"
    script = (
        f'mkdir -p "{MEDIA_DIR}/{SIZED_ASSET_DIR}" && '
        f'if [ "$(stat -c%s "{dest}" 2>/dev/null || echo 0)" != "{size_bytes}" ]; then '
        f': > "{dest}" && '
        f'cur=0 && '
        f'while [ "$cur" -lt "{size_bytes}" ]; do '
        f'cat "{src}" >> "{dest}" || {{ echo "write failed (disk full?)" >&2; exit 1; }}; '
        f'cur=$(stat -c%s "{dest}"); '
        f'done && '
        f'truncate -s {size_bytes} "{dest}"; '
        f'fi'
    )
    # Scales with size so a multi-GB request isn't cut off mid-generation by a
    # timeout sized for the original ~64MB-scale use case.
    timeout_s = max(_SIZED_ASSET_MIN_TIMEOUT_S, size_bytes / (_SIZED_ASSET_ASSUMED_MIN_MBPS * 1024 ** 2) + 30.0)
    try:
        _docker_exec(container, "sh", "-c", script, timeout_s=timeout_s)
    except subprocess.CalledProcessError as e:
        raise ShowcaseError(
            f"Could not build the {size_bytes / 1024**3:.2f}GB transfer asset on {container} "
            f"(likely out of disk space on the host/container -- check with `df -h`). stderr: {e.stderr}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise ShowcaseError(
            f"Building the {size_bytes / 1024**3:.2f}GB transfer asset on {container} didn't finish within "
            f"{timeout_s:.0f}s -- the host disk may be too slow for this size, or genuinely out of space."
        ) from e
    return rel_path


def _content_length(client_container: str, url: str) -> int | None:
    """Real asset size via a HEAD request -- lets the live view show a % progress bar."""
    head = _docker_exec(client_container, "curl", "-sI", url, timeout_s=15, check=False)
    m = re.search(r"(?im)^content-length:\s*(\d+)", head.stdout)
    return int(m.group(1)) if m else None


def _file_size(container: str, path: str) -> int:
    out = _docker_exec(container, "sh", "-c", f"stat -c%s {path} 2>/dev/null || echo 0", check=False)
    try:
        return int(out.stdout.strip() or 0)
    except ValueError:
        return 0


def _post_live_progress(cur_bytes: int, total_bytes: int | None, throughput_mbps: float) -> None:
    sample = {
        "app": "file_transfer", "t": time.time(),
        "bytes": cur_bytes, "total_bytes": total_bytes,
        "throughput_mbps": throughput_mbps,
    }
    try:
        requests.post(BACKEND_STATS_URL, json=sample, timeout=2)
    except requests.RequestException:
        pass  # best-effort -- a missed progress tick doesn't affect the real transfer


def run_file_transfer_showcase(
    asset_name: str = "bigbuckbunny.mp4",
    size_gb: float | None = None,
    congestion_control: str = "cubic",
    client_container: str = DEFAULT_CLIENT_CONTAINER,
    server_container: str = DEFAULT_SERVER_CONTAINER,
    poll_interval_s: float = 0.4,
    timeout_s: float | None = None,
) -> FileTransferResult:
    """Runs one real HTTP download from cosme-server to cosme-client.

    There's no notion of "duration" here -- unlike the other showcases, this
    one is a fixed-size download that ends when the transfer completes, so
    the dashboard exposes a transfer SIZE control instead. When `size_gb` is
    given, downloads a real-content file of exactly that size (built once,
    cached, see `_ensure_sized_asset`) instead of the fixed `asset_name` --
    `asset_name` remains the default for standalone/CLI use where a specific
    real media file (rather than an arbitrary size) is wanted.

    Requires `scripts/prepare_media_assets.sh` to have placed `asset_name`
    under `docker/media/` (bind-mounted to /srv/media in both containers)
    and both containers to be up (see docker/docker-compose.yml).

    `timeout_s=None` (the default) derives the bound from the transfer SIZE rather than using a
    flat wall-clock number: a fixed 300s cap silently truncated large or heavily-impaired transfers
    and recorded them as failures, which is exactly the case a LEO emulator is built to produce.
    The derived bound assumes a floor of `_TRANSFER_ASSUMED_MIN_MBPS`, so a genuinely hung transfer
    is still caught while a merely slow one is allowed to finish. Pass an explicit value to override.

    Unlike a plain blocking curl, this downloads to a real file (not
    /dev/null) and runs curl detached so the size can be polled while the
    transfer is in flight -- that's what feeds the dashboard's live progress
    bar/throughput sparkline. A fast/unshaped transfer may complete within a
    single poll tick (nothing to show); a shaped one -- the actual point of
    this demo -- takes long enough for the live view to be meaningful.
    """
    ensure_containers_healthy(server_container, client_container)
    if timeout_s is None:
        if size_gb:
            timeout_s = max(300.0, (size_gb * 1024 ** 3 * 8) / (_TRANSFER_ASSUMED_MIN_MBPS * 1e6) + 60.0)
        else:
            timeout_s = 300.0

    set_congestion_control(congestion_control, server_container)
    ensure_http_server_running(server_container)

    if size_gb is not None:
        size_bytes = round(size_gb * 1024 ** 3)
        asset_name = _ensure_sized_asset(server_container, size_bytes)

    url = f"http://{SERVER_INTERNAL_IP}:{HTTP_PORT}/{asset_name}"
    total_bytes = _content_length(client_container, url)

    _docker_exec(client_container, "rm", "-f", DEST_PATH, RESULT_PATH, check=False)
    # Baseline on the SERVER: it is the sending side, so it is the end that retransmits.
    retrans_before = _tcp_retrans_total(server_container)

    subprocess.run(
        ["docker", "exec", "-d", client_container, "sh", "-c",
         f"curl -s -o {DEST_PATH} -w '{_CURL_FORMAT}' {url} > {RESULT_PATH}"],
        capture_output=True, text=True, timeout=10,
    )

    start = time.monotonic()
    prev_bytes, prev_t = 0, start
    stats = None
    while time.monotonic() - start < timeout_s:
        time.sleep(poll_interval_s)
        cur_bytes = _file_size(client_container, DEST_PATH)
        now = time.monotonic()
        dt = now - prev_t
        throughput_mbps = (max(cur_bytes - prev_bytes, 0) * 8 / 1e6 / dt) if dt > 0 else 0.0
        _post_live_progress(cur_bytes, total_bytes, throughput_mbps)
        prev_bytes, prev_t = cur_bytes, now

        done = _docker_exec(client_container, "cat", RESULT_PATH, check=False)
        if done.returncode == 0 and done.stdout.strip():
            stats = json.loads(done.stdout)
            break
    if stats is None:
        raise ShowcaseError(f"file transfer did not finish within {timeout_s:.0f}s")

    retrans_after = _tcp_retrans_total(server_container)
    if retrans_before is None or retrans_after is None:
        retransmits = None
    else:
        # Negative means the counter wrapped or the container restarted mid-run: report unknown.
        delta = retrans_after - retrans_before
        retransmits = delta if delta >= 0 else None

    return FileTransferResult(
        duration_s=stats["time_total"],
        size_bytes=stats["size_download"],
        throughput_bps=stats["speed_download"] * 8,
        http_code=int(stats["http_code"]),
        tcp_retransmits=retransmits,
        congestion_control=congestion_control,
    )


def _tcp_retrans_total(container: str) -> int | None:
    """Cumulative TCP RetransSegs for `container`'s network namespace.

    /proc/net/snmp is per-netns and survives connection teardown, so callers read it either side
    of a transfer and report the difference. Must be read on the SENDING side: for a download the
    client only sends ACKs and has nothing to retransmit.

    Namespace-wide, so it includes any other TCP retransmission in that container during the
    window. On cosme-server the download is the only meaningful TCP traffic during a showcase.
    """
    try:
        out = _docker_exec(container, "sh", "-c", "grep '^Tcp:' /proc/net/snmp", check=False)
    except subprocess.TimeoutExpired:
        return None
    lines = [l for l in out.stdout.splitlines() if l.startswith("Tcp:")]
    if len(lines) < 2:
        return None
    keys, values = lines[0].split()[1:], lines[1].split()[1:]
    raw = dict(zip(keys, values)).get("RetransSegs")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    import argparse
    import dataclasses

    parser = argparse.ArgumentParser(description="Run the file-transfer showcase.")
    parser.add_argument("--asset", default="bigbuckbunny.mp4")
    parser.add_argument("--size-gb", type=float, default=None, help="download an exact-size real-content file instead of --asset")
    parser.add_argument("--cc", default="cubic", choices=["cubic", "bbr", "reno"])
    args = parser.parse_args()

    result = run_file_transfer_showcase(asset_name=args.asset, size_gb=args.size_gb, congestion_control=args.cc)
    print(dataclasses.asdict(result))
