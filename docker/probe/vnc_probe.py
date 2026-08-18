#!/usr/bin/env python3
"""Remote-desktop showcase probe: a real RFB (VNC) client measuring
user-perceived interaction QoE through the shaped link.

Runs inside the cosme-probe container (python:3.12-slim + vncdotool, shares
cosme-client's netns -- see docker/docker-compose.yml) via `docker exec` from
backend/showcases/remote_desktop.py, against the TigerVNC server in
cosme-server. Two xterms are on the remote display (started by the showcase
module):
  - the "activity" xterm at the top-left prints a timestamp at 10Hz -- the
    known-rate screen activity whose observed update rate is the
    "effective fps" metric;
  - the "echo" xterm at the bottom-left runs an idle shell -- the target for
    keystroke round-trip measurement. Xvnc without a window manager uses
    X's PointerRoot focus (focus follows the pointer), so moving the VNC
    pointer into that window is what routes our key events to it.

Both metrics are deliberately *client-perceived*: each screen sample is a
real FramebufferUpdateRequest round trip (VNC is client-pull), so keystroke
latency includes the echo plus one update round trip, and the observed
update rate is bounded by both the server's 10Hz activity and how fast the
shaped link lets a client pull -- exactly what a human user of this VNC
session would experience.

Per second the probe POSTs a stats sample
({"app": "remote_desktop", "t", "effective_fps", "keystroke_latency_ms"})
and, when --live-endpoint is given, a full-screen PNG capture for the
dashboard's live view. The last stdout line is the end-of-run JSON summary
parsed by the backend. HTTP via stdlib urllib only (no pip deps besides
vncdotool).

Reliability notes (found while root-causing real "VNC showcase not
responding" reports, vncdotool==1.3.0): `VNCDoToolClient.encoding` defaults
to RAW (fully uncompressed) -- ACTIVITY_REGION alone is ~780KB per single
uncompressed poll, sent every 0.1-1s, which is slow even against an UNSHAPED
link (measured effective_fps of 0.1-1.8 against the documented 10fps
ceiling). RAW *looks* like an obvious fix target, but both compressed
alternatives this vncdotool version exposes are confirmed BROKEN here, not
just slower to negotiate: TIGHT reliably hangs every real round trip
(`TimeoutError`, reproduced 3/3 in direct testing, never a single success),
and ZRLE has an outright decoder bug (`vncdotool/rfb.py`'s
`_handleDecodeZRLEdata` raises `IndexError: list index out of range` on
real server data, reproduced 3/3). So RAW -- slow, but the only encoding
that actually works with this vncdotool release -- stays the default; do
not "fix" this by switching encodings without re-verifying against whatever
vncdotool version is installed at the time, since this was tested against
1.3.0 specifically. What IS fixed below: individual RFB round trips are
retried a couple times (`_rfb_call`) before that one sample is skipped,
rather than one slow/lost round trip (confirmed to happen even under RAW,
via vncdotool's threaded reactor bridge) crashing the entire run and losing
every sample collected so far -- combined with backend/showcases/
remote_desktop.py's new wait_vnc_ready() (closes a real cold-start race
this session also hit), measured real-run success went from clearly flaky
in the initial investigation to 8/9 in the final measured batch (one
initial-connection failure that exhausts before the main loop's per-tick
recovery kicks in is a known remaining gap -- see remote_desktop.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

from vncdotool import api

# Individual RFB round trips should be fast (tens-hundreds of ms) on a real link, shaped or
# not -- this is deliberately much shorter than args.keystroke_timeout_s (a higher-level "did
# the keystroke's echo genuinely never arrive" budget), so one slow round trip doesn't eat the
# whole run's time budget across its retries.
RFB_CALL_TIMEOUT_S = 5.0
RFB_CALL_RETRIES = 2
RFB_CALL_BACKOFF_S = 0.3
# Worst case a single _rfb_call can cost: every attempt times out, plus backoff between them.
# Used as the run's overall grace period beyond --duration (see `deadline` below) -- real bug
# this bounds: with no hard deadline threaded through, a single struggling call could cost up to
# this much, and NOTHING stopped several such calls from stacking back-to-back, since the only
# duration check was `while time.monotonic() - start < args.duration` at the top of the OUTER
# loop -- one blocked iteration (e.g. the inner keystroke-echo-wait loop, which had its own
# independent keystroke_timeout_s budget never itself bounded by args.duration) could run long
# with nothing upstream noticing. Confirmed live: a `--duration 15` run actually took 43.2s once
# a real, concurrently-shaping scenario's periodic loss bursts (100% loss for ~100-400ms every
# ~15s -- see backend/models/reconfig_schedule.py) started landing mid-call. This was never
# exercised before showcases could run concurrently with an actively-shaping scenario.
RFB_CALL_MAX_COST_S = RFB_CALL_TIMEOUT_S * (RFB_CALL_RETRIES + 1) + RFB_CALL_BACKOFF_S * RFB_CALL_RETRIES

# Pixel regions (x, y, w, h) matching the xterm geometries started by
# backend/showcases/remote_desktop.py on the 1024x768 display.
ACTIVITY_REGION = (0, 0, 620, 420)     # 100x30-char xterm at +0+0
ECHO_REGION = (0, 500, 520, 160)       # 80x10-char xterm at +0+500
ECHO_POINTER = (250, 560)              # pointer target inside the echo xterm

_CAP_PATH = "/tmp/cosme_vnc_cap.png"   # PIL needs a real extension to pick a format
_LIVE_PATH = "/tmp/cosme_vnc_live.png"


def _post(url: str, data: bytes, content_type: str) -> None:
    try:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": content_type})
        urllib.request.urlopen(req, timeout=3)
    except (urllib.error.URLError, OSError):
        pass  # backend unreachable this tick -- keep measuring


def _rfb_call(fn, *args, deadline: float | None = None, **kwargs):
    """Retries one vncdotool round trip a couple times before giving up.

    Confirmed live (see module docstring) that a single RFB round trip can
    occasionally time out even against a healthy, unshaped link -- almost
    certainly a quirk of vncdotool's Twisted-reactor/background-thread
    bridge rather than anything wrong with the server. A short retry absorbs
    that without treating it as a fatal probe failure.

    `deadline` (a `time.monotonic()` timestamp) bounds the RETRY behavior specifically: once
    passed, stop attempting further retries rather than blindly working through the full
    RFB_CALL_RETRIES budget regardless of how much of the overall run's time is left -- see
    RFB_CALL_MAX_COST_S's own comment for the real overrun this prevents.
    """
    # Seeded with a real exception, not None: if `deadline` has already passed before even the
    # FIRST attempt (a real, reachable case -- an earlier _rfb_call in the same loop tick can eat
    # the whole remaining budget), the loop below breaks immediately with zero attempts made. A
    # bare `raise None` in that case would crash with `TypeError: exceptions must derive from
    # BaseException` instead of a meaningful error -- caught while writing this fix's own test.
    last_exc: Exception = TimeoutError("deadline already passed before any RFB attempt was made")
    for attempt in range(RFB_CALL_RETRIES + 1):
        if deadline is not None and time.monotonic() >= deadline:
            break
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < RFB_CALL_RETRIES:
                time.sleep(RFB_CALL_BACKOFF_S)
    raise last_exc


def region_digest(client, region: tuple[int, int, int, int], deadline: float | None = None) -> str:
    """One real client-pull screen sample of `region`, hashed for change detection."""
    _rfb_call(client.captureRegion, _CAP_PATH, *region, deadline=deadline)
    with open(_CAP_PATH, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="10.42.0.10")
    parser.add_argument("--port", type=int, default=5901)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--keystroke-interval-s", type=float, default=2.0)
    parser.add_argument("--keystroke-timeout-s", type=float, default=10.0)
    parser.add_argument("--stats-endpoint", default=None)
    parser.add_argument("--live-endpoint", default=None,
                        help="POST a full-screen PNG here once per second (dashboard live view)")
    args = parser.parse_args()

    client = api.connect(f"{args.host}::{args.port}", password=None)
    # Bound every individual RFB round trip (see RFB_CALL_TIMEOUT_S's own comment for why this
    # is deliberately much shorter than args.keystroke_timeout_s).
    client.timeout = RFB_CALL_TIMEOUT_S
    # Route keyboard input to the echo xterm (PointerRoot focus, see docstring).
    _rfb_call(client.mouseMove, *ECHO_POINTER)

    latencies_ms: list[float] = []
    keystroke_attempts = 0    # every attempt, echoed or not
    keystroke_timeouts = 0    # attempts whose echo never arrived in time
    start = time.monotonic()
    # Hard overall ceiling: no combination of retries/keystroke-waits may run the probe past
    # this, no matter how many individual RFB calls struggle along the way (see
    # RFB_CALL_MAX_COST_S's own comment). One call's worth of extra grace, not one PER retry
    # PER tick stacked indefinitely -- the actual bug that let a --duration 15 run take 43.2s.
    deadline = start + args.duration + RFB_CALL_MAX_COST_S
    last_activity_digest = region_digest(client, ACTIVITY_REGION, deadline=deadline)
    next_keystroke_at = start + 1.0
    second_start = time.monotonic()
    changes_this_second = 0
    total_changes = 0
    latency_this_second: float | None = None

    while time.monotonic() - start < args.duration and time.monotonic() < deadline:
        try:
            # Effective fps: does the activity area differ from the last pull?
            digest = region_digest(client, ACTIVITY_REGION, deadline=deadline)
            if digest != last_activity_digest:
                changes_this_second += 1
                total_changes += 1
                last_activity_digest = digest

            # Keystroke round trip, every --keystroke-interval-s.
            #
            # Attempts that never echo back are COUNTED, not dropped. Reporting the median only
            # over round trips that completed is survivorship bias: a keystroke times out exactly
            # when interactivity is worst, so excluding those describes "latency when it worked"
            # and makes a badly degraded link look responsive. The timeout rate is reported
            # alongside, and is arguably the more honest headline for interactivity.
            if time.monotonic() >= next_keystroke_at:
                keystroke_attempts += 1
                echoed = False
                try:
                    baseline = region_digest(client, ECHO_REGION, deadline=deadline)
                    t0 = time.monotonic()
                    _rfb_call(client.keyPress, "x", deadline=deadline)
                    keystroke_deadline = min(t0 + args.keystroke_timeout_s, deadline)
                    while time.monotonic() < keystroke_deadline:
                        if region_digest(client, ECHO_REGION, deadline=keystroke_deadline) != baseline:
                            latency = (time.monotonic() - t0) * 1000
                            latencies_ms.append(latency)
                            latency_this_second = latency
                            echoed = True
                            break
                except Exception as e:
                    print(f"[vnc_probe] keystroke round trip failed: {e}", file=sys.stderr, flush=True)
                if not echoed:
                    keystroke_timeouts += 1
                next_keystroke_at = time.monotonic() + args.keystroke_interval_s
        except Exception as e:
            # _rfb_call already retried RFB_CALL_RETRIES times -- this tick's sample is lost,
            # but one bad round trip must never crash the whole run and lose every sample
            # collected so far (confirmed live: this was the actual "VNC showcase not
            # responding" failure mode). Keep going.
            print(f"[vnc_probe] skipped one sample after RFB error: {e}", file=sys.stderr, flush=True)

        now = time.monotonic()
        if now - second_start >= 1.0:
            sample = {
                "app": "remote_desktop",
                "t": time.time(),
                "effective_fps": changes_this_second / (now - second_start),
                "keystroke_latency_ms": latency_this_second,
            }
            if args.stats_endpoint:
                _post(args.stats_endpoint, json.dumps(sample).encode(), "application/json")
            if args.live_endpoint:
                # What the user of this VNC session sees RIGHT NOW -- another
                # real client-pull round trip through the shaped link.
                try:
                    _rfb_call(client.captureScreen, _LIVE_PATH, deadline=deadline)
                    with open(_LIVE_PATH, "rb") as f:
                        _post(args.live_endpoint, f.read(), "image/png")
                except Exception:
                    pass  # capture failure must never kill the measurement
            second_start = now
            changes_this_second = 0
            latency_this_second = None

    elapsed = time.monotonic() - start
    client.disconnect()
    # Without this, vncdotool's non-daemon Twisted reactor thread keeps the
    # process alive forever after main() returns -- the summary print below
    # then sits in a stdio buffer and the docker-exec caller times out.
    api.shutdown()

    summary: dict = {
        "n_keystrokes": len(latencies_ms),          # echoed back (kept for existing callers)
        "keystroke_attempts": keystroke_attempts,
        "keystroke_timeouts": keystroke_timeouts,
        "keystroke_timeout_pct": (round(100.0 * keystroke_timeouts / keystroke_attempts, 2)
                                   if keystroke_attempts else None),
        "duration_s": round(elapsed, 1),
    }
    # NOTE: median/p95 below are CONDITIONAL ON THE KEYSTROKE ECHOING AT ALL. Read them together
    # with keystroke_timeout_pct -- on a badly degraded link the timeouts, not the latency of the
    # survivors, are what the user actually experiences.
    if latencies_ms:
        summary["keystroke_latency_ms_median"] = round(statistics.median(latencies_ms), 1)
        summary["keystroke_latency_ms_p95"] = round(
            statistics.quantiles(latencies_ms, n=20)[-1] if len(latencies_ms) >= 2 else latencies_ms[0], 1
        )
    summary["effective_fps_mean"] = round(total_changes / elapsed, 1) if elapsed > 0 else 0.0
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
