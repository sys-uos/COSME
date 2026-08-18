"""Playback engine: turns a composed trace into a real-time tc/netem schedule.

Per the paper: "The orchestrator's preprocessing adapts to the two stages
and simplifies the trace by adapting the emulation only every 15 seconds
for a short time interval for the reconfiguration. This simplification
avoids constant configuration changes in sub-millisecond timing by
configuring the calculated jitter, max-, and min-delay in netem for the
15-second non-handover slots."

Concretely, this module:
  * Quantizes Garcia's per-sample delay/jitter and WetLinks' bandwidth to
    **two representative netem configs per 15s reconfiguration slot** (one
    "normal" config, one "reconfiguration" config for the brief handover
    sub-window) -- matching the paper's stated simplification, since Garcia's
    raw model varies at sub-millisecond granularity and applying tc changes
    at that rate is neither necessary nor practical.
  * Applies loss (obstruction OR reconfiguration) at its **actual event
    resolution** instead of quantizing it to the 15s grid: obstruction
    events happen at arbitrary times along a real driven route (not aligned
    to any 15s boundary), so quantizing them to slot-level would smear out
    or miss short bursts entirely. Loss is the one dimension the original
    ObLoS emulator (models/ObLoS/emulator) also applied at exact event
    timestamps, not on a fixed grid -- this preserves that behavior.

The two update streams (slot-level delay/jitter/bandwidth,
event-level loss) are merged into one sorted schedule and played back in
real time (or fast-forwarded via `speed`) against a NetemBackend.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.compose import ComposedTrace
from backend.netem_backend import NetemBackend, NetemParams


@dataclass
class ScheduledUpdate:
    at_s: float
    endpoint: str  # "client" (upload direction) or "server" (download direction)
    params: NetemParams
    critical: bool = False  # True for a loss on/off transition -- see play()'s skip-ahead logic


@dataclass
class PlaybackStats:
    """Live, mutated-in-place progress tracking for an in-flight play() call.

    Exists because elapsed_s() (backend/api.py's Scenario) is pure wall-clock time, completely
    decoupled from how far play() has actually gotten through the plan -- if playback falls
    behind (see play()'s own docstring), elapsed_s() keeps climbing regardless, silently hiding
    the lag from anyone watching the dashboard. Surfaced via GET /api/scenarios/{id}/status so a
    real lag is visible instead of just producing confusing "loss never happens" symptoms.
    """
    lag_s: float = 0.0        # how far behind schedule the most recently processed tick was (0 = caught up)
    ticks_skipped: int = 0    # non-critical ticks jumped over while catching up from lag (loss ticks: always 0)
    ticks_total: int = 0


MAX_UPDATE_INTERVAL_S = 1.0  # netem updates must happen at least once per second (user requirement)


def _clamp_update_interval(update_interval_s: float) -> float:
    """Enforce the >=1Hz update-rate floor: interval may be finer than 1s, never coarser."""
    return min(update_interval_s, MAX_UPDATE_INTERVAL_S)


def _grid_delay_jitter_updates(df: pd.DataFrame, update_interval_s: float = MAX_UPDATE_INTERVAL_S) -> list[tuple[float, float, float]]:
    """(time_s, delay_ms, jitter_ms) representative updates on a regular update_interval_s grid.

    This grid is independent of (and much finer than) Garcia/Zimmermann's
    physical 15s reconfiguration cadence (RECONFIG_PERIOD_S) -- it's purely
    how often the *playback engine* pushes a new netem config, so it tracks
    Garcia's continuously-varying delay/jitter far more closely than one
    flat value per ~12.5s "normal" window. The brief reconfiguration windows
    themselves get an additional, sharper update at their exact start time
    via `_reconfig_event_updates()`, so a handover disruption is never
    smoothed away by landing mid-bucket.

    Real, confirmed perf bug this replaced: a per-bin `df[(timestamp >= bin_start) &
    (timestamp < bin_end)]` filter inside a `for i in range(n_bins)` loop scans the WHOLE
    dataframe on every iteration, i.e. O(n_bins * n_rows) -- both grow with route duration, so
    total cost grows roughly quadratically. Measured on the longest real drive (177 min, ~1.06M
    10ms-grid rows, 10629 one-second bins): 9.4s for this function alone with the old approach.
    `groupby()` on a vectorized bin index is a single O(n_rows) pass; same drive: 0.3s -- a ~31x
    speedup, verified numerically identical (max diff ~1e-14, pure float-summation-order noise)
    against the old implementation before replacing it. A bin with zero rows (only possible if
    `update_interval_s` is configured finer than the trace's own `dt_s` grid) is naturally absent
    from `groupby`'s output, matching the old code's explicit `if bucket.empty: continue` skip.
    """
    update_interval_s = _clamp_update_interval(update_interval_s)
    bin_idx = np.floor(df["timestamp"].to_numpy() / update_interval_s).astype(np.int64)
    grouped = df.groupby(bin_idx)[["delay_ms", "jitter_ms"]].mean()
    return [(int(idx) * update_interval_s, float(row["delay_ms"]), float(row["jitter_ms"]))
            for idx, row in grouped.iterrows()]


def _reconfig_event_updates(df: pd.DataFrame) -> list[tuple[float, float, float]]:
    """(time_s, delay_ms, jitter_ms) sharp updates at each reconfiguration burst's real start time.

    Layered on top of the coarser update grid so the ~100-400ms
    reconfiguration disruption (RECONFIG_PERIOD_S-cadenced, see
    reconfig_schedule.py) is applied precisely rather than only showing up
    as a slightly-elevated grid-bucket average.
    """
    reconfig = df["reconfig_loss"].to_numpy()
    times = df["timestamp"].to_numpy()
    updates = []
    if not reconfig.any():
        return updates
    edges = np.diff(reconfig.astype(int))
    starts_idx = (np.where(edges == 1)[0] + 1).tolist()
    if reconfig[0]:
        starts_idx.insert(0, 0)
    for start_idx in starts_idx:
        end_idx = start_idx
        while end_idx < len(reconfig) and reconfig[end_idx]:
            end_idx += 1
        run = df.iloc[start_idx:end_idx]
        updates.append((float(times[start_idx]), float(run["delay_ms"].mean()), float(run["jitter_ms"].mean())))
    return updates


def _grid_bandwidth_updates(df: pd.DataFrame, update_interval_s: float = MAX_UPDATE_INTERVAL_S) -> list[tuple[float, float, float]]:
    """(time_s, download_mbps, upload_mbps) representative updates on the same update_interval_s grid.

    See `_grid_delay_jitter_updates`'s docstring for why this is a vectorized `groupby` rather
    than a per-bin dataframe filter -- same O(n_bins*n_rows) -> O(n_rows) fix, same verification.
    """
    update_interval_s = _clamp_update_interval(update_interval_s)
    bin_idx = np.floor(df["timestamp"].to_numpy() / update_interval_s).astype(np.int64)
    grouped = df.groupby(bin_idx)[["download_mbps", "upload_mbps"]].mean()
    return [(int(idx) * update_interval_s, float(row["download_mbps"]), float(row["upload_mbps"]))
            for idx, row in grouped.iterrows()]


def _loss_event_updates(df: pd.DataFrame) -> list[tuple[float, float]]:
    """(start_s, end_s) for every contiguous loss=True run, at real event resolution."""
    loss = df["loss"].to_numpy()
    times = df["timestamp"].to_numpy()
    if not loss.any():
        return []
    edges = np.diff(loss.astype(int))
    starts = times[np.where(edges == 1)[0] + 1]
    ends = times[np.where(edges == -1)[0] + 1]
    if loss[0]:
        starts = np.insert(starts, 0, times[0])
    if loss[-1]:
        ends = np.append(ends, times[-1] + (times[1] - times[0]))
    return list(zip(starts.tolist(), ends.tolist()))


def build_playback_plan(trace: ComposedTrace, update_interval_s: float = MAX_UPDATE_INTERVAL_S) -> list[ScheduledUpdate]:
    """Merge grid-level delay/jitter/bandwidth, sharp reconfig events, and event-level loss into one schedule.

    `update_interval_s` controls how often delay/jitter/bandwidth are
    re-quantized (clamped to <=1s -- see MAX_UPDATE_INTERVAL_S); loss is
    always applied at real event resolution regardless of this setting.
    """
    update_interval_s = _clamp_update_interval(update_interval_s)
    df = trace.df
    events: dict[float, dict] = {}
    # Loss on/off transitions must NEVER be silently skipped during a catch-up (see play()) --
    # every other field (delay/jitter/bandwidth) is just a moving average that's safe to jump
    # past when stale, but a loss transition is a discrete, real "packet drops now"/"stops now"
    # event: dropping one would mean a modeled loss burst genuinely never reaches the real link.
    critical_at_s: set[float] = set()

    def _set(at_s: float, **kwargs):
        events.setdefault(round(at_s, 6), {}).update(kwargs)

    for at_s, delay_ms, jitter_ms in _grid_delay_jitter_updates(df, update_interval_s):
        _set(at_s, delay_ms=delay_ms, jitter_ms=jitter_ms)
    for at_s, delay_ms, jitter_ms in _reconfig_event_updates(df):
        _set(at_s, delay_ms=delay_ms, jitter_ms=jitter_ms)
    for at_s, dl, ul in _grid_bandwidth_updates(df, update_interval_s):
        _set(at_s, download_mbps=dl, upload_mbps=ul)
    for start_s, end_s in _loss_event_updates(df):
        _set(start_s, loss_pct=100.0)
        _set(end_s, loss_pct=0.0)
        critical_at_s.add(round(start_s, 6))
        critical_at_s.add(round(end_s, 6))

    # Forward-fill so every scheduled instant carries a complete parameter set.
    last = {"delay_ms": 0.0, "jitter_ms": 0.0, "download_mbps": 100.0, "upload_mbps": 10.0, "loss_pct": 0.0}
    plan: list[ScheduledUpdate] = []
    for at_s in sorted(events):
        last.update(events[at_s])
        critical = at_s in critical_at_s
        plan.append(ScheduledUpdate(at_s=at_s, endpoint="server", critical=critical,
                                     params=NetemParams(loss_pct=last["loss_pct"], delay_ms=last["delay_ms"],
                                                         jitter_ms=last["jitter_ms"], rate_mbit=last["download_mbps"])))
        plan.append(ScheduledUpdate(at_s=at_s, endpoint="client", critical=critical,
                                     params=NetemParams(loss_pct=last["loss_pct"], delay_ms=last["delay_ms"],
                                                         jitter_ms=last["jitter_ms"], rate_mbit=last["upload_mbps"])))
    return plan


MAX_PLAYBACK_LAG_S = 1.0  # once this far behind schedule, start catching up (see play())


async def play(plan: list[ScheduledUpdate], backend: NetemBackend, speed: float = 1.0,
                max_lag_s: float = MAX_PLAYBACK_LAG_S, stats: PlaybackStats | None = None) -> None:
    """Drive `backend` through `plan` in real time (speed=1.0) or fast-forwarded.

    **Real, confirmed bug this guards against**: `build_playback_plan` always emits a
    server+client PAIR of updates at every scheduled instant. Applying each sequentially costs two
    full `docker exec` round trips per tick (tens-to-hundreds of ms each, per
    `netem_backend.DockerNetemBackend`'s own docstring); if the plan's own update cadence
    (`update_interval_s`, user-configurable down to 0.1s) demands ticks faster than that, this
    loop can never keep up -- and once behind, it has no way to know or recover, so it falls
    FURTHER behind, unboundedly, for the rest of the run. Confirmed live against a real
    long-running scenario: `elapsed_s()` (wall-clock-only, see `api.Scenario`) read ~531s while
    what was ACTUALLY on the wire matched the trace's own ~352s content -- **178s of accumulated
    lag**. Because it grows monotonically, a brief (100-400ms) loss burst that WAS genuinely
    issued ends up applied only for as long as one more command's round trip takes before the next
    (also backlogged) update overwrites it -- on the wire for a sliver of real time, at a
    real-time offset that keeps drifting further from whatever the dashboard/operator currently
    believes "now" is. This reads exactly like "loss is never applied", even though every loss
    command was technically issued -- delay/rate look fine because their own values change slowly
    enough that a stale one still looks plausible; loss doesn't have that luxury.

    Two real mitigations, not just detection:
      1. **Server and client applies for the same tick now run concurrently**
         (`asyncio.gather`, not sequential `await`s) -- halves the real wall-clock cost per tick,
         directly raising the update rate this loop can actually sustain.
      2. **Catch-up skip-ahead for non-critical ticks only**: once more than `max_lag_s` behind
         schedule, jump straight to the latest already-due tick that is ALSO non-critical, without
         applying the ones in between -- replaying stale intermediate delay/jitter/bandwidth
         values after the fact cannot improve real-time fidelity (the link never experienced them
         at the "right" time regardless) and only makes the lag worse. Loss on/off transitions
         (`ScheduledUpdate.critical`) are NEVER skipped or reordered -- every one is applied, in
         order, even while catching up, so a modeled loss burst can never simply vanish from the
         real link the way it did before this fix.

    `stats`, if given, is mutated in place as playback progresses (not just at the end) so a
    caller (e.g. `api.Scenario`) can expose live lag/skip counts via its own status endpoint.
    """
    t0 = time.monotonic()

    # Group into ticks: build_playback_plan always emits a "server" entry immediately followed by
    # its paired "client" entry at the same at_s -- this recovers that pairing generically (also
    # tolerates a plan with only one endpoint per tick, e.g. a hand-built test plan).
    ticks: list[tuple[float, list[ScheduledUpdate]]] = []
    i = 0
    while i < len(plan):
        at_s = plan[i].at_s
        group = [plan[i]]
        i += 1
        while i < len(plan) and plan[i].at_s == at_s:
            group.append(plan[i])
            i += 1
        ticks.append((at_s, group))

    if stats is not None:
        stats.ticks_total = len(ticks)

    idx = 0
    n = len(ticks)
    while idx < n:
        at_s, group = ticks[idx]
        target = t0 + at_s / speed
        now = time.monotonic()
        is_critical = any(u.critical for u in group)

        if not is_critical and (now - target) > max_lag_s:
            # Behind schedule on a routine (non-loss) tick: fast-forward to the latest tick
            # that's both already due AND non-critical, stopping at the first critical or
            # not-yet-due tick so nothing gets reordered or dropped.
            j = idx
            while j + 1 < n:
                next_at_s, next_group = ticks[j + 1]
                if (t0 + next_at_s / speed) > now or any(u.critical for u in next_group):
                    break
                j += 1
            if j > idx:
                if stats is not None:
                    stats.ticks_skipped += j - idx
                idx = j
                at_s, group = ticks[idx]
                target = t0 + at_s / speed
                now = time.monotonic()

        if target > now:
            await asyncio.sleep(target - now)
            now = target

        # backend.apply() shells out (`docker exec ... tc qdisc replace`, tens to
        # hundreds of ms); running it inline would block uvicorn's single-threaded
        # event loop for that long on every update, starving ALL other requests
        # (health checks, showcase polling, the frontend's own status polls) for
        # the duration of the scenario -- confirmed live: /api/health hung
        # indefinitely while a scenario played. Server+client run CONCURRENTLY (see docstring).
        await asyncio.gather(*(asyncio.to_thread(backend.apply, u.endpoint, u.params) for u in group))

        if stats is not None:
            stats.lag_s = max(0.0, time.monotonic() - target)
        idx += 1


def play_sync(plan: list[ScheduledUpdate], backend: NetemBackend, speed: float = 1.0) -> None:
    asyncio.run(play(plan, backend, speed=speed))


if __name__ == "__main__":
    import argparse

    from backend.compose import compose
    from backend.models import oblos
    from backend.netem_backend import DryRunNetemBackend

    parser = argparse.ArgumentParser(description="Run the COSME orchestrator against a composed trace.")
    parser.add_argument("--drive", default=None, help="clipped_measurements drive name for real ObLoS trace")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--speed", type=float, default=20.0, help="playback speed multiplier (20x for a quick smoke test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    obstruction_trace = None
    if args.drive:
        obstruction_trace = oblos.load_obstruction_trace(args.drive)

    composed = compose(duration_s=args.duration, obstruction_trace=obstruction_trace, seed=args.seed)
    plan = build_playback_plan(composed)
    print(f"built {len(plan)} scheduled updates over {args.duration}s")

    backend = DryRunNetemBackend(verbose=True)
    play_sync(plan, backend, speed=args.speed)
    print(f"issued {len(backend.log)} tc commands (dry-run)")
