"""Regression tests for backend/orchestrator.py.

These pin play()'s real-time behaviour: without detection and recovery it can fall arbitrarily
far behind schedule (seen: 178s of accumulated lag on a long scenario), which makes brief loss
bursts statistically invisible on the real link even though they were technically applied.
"""
import asyncio
import time

import numpy as np
import pandas as pd

from backend.compose import compose
from backend.netem_backend import NetemBackend, NetemParams
from backend.orchestrator import (
    PlaybackStats,
    _grid_bandwidth_updates,
    _grid_delay_jitter_updates,
    build_playback_plan,
    play,
)


class _RecordingBackend(NetemBackend):
    """Records every apply() call (endpoint, exact params, wall-clock time); optionally slow."""

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.calls: list[tuple[str, NetemParams, float]] = []

    def apply(self, endpoint: str, params: NetemParams) -> None:
        if self.delay_s:
            time.sleep(self.delay_s)  # simulates a real `docker exec` round trip
        self.calls.append((endpoint, params, time.monotonic()))

    def reset(self, endpoint: str) -> None:
        pass


def _bursty_trace(n_bursts: int = 6, burst_s: float = 0.2, gap_s: float = 0.6, duration_s: float = 5.0):
    """A short trace with several short, precisely-timed obstruction loss bursts -- independent of
    Garcia/Zimmermann's 15s reconfig cadence, which wouldn't fire at all within a 5s duration."""
    starts = [i * (burst_s + gap_s) + 0.2 for i in range(n_bursts)]
    obstruction = pd.DataFrame({"timestamp": starts, "lossTime": [burst_s] * n_bursts})
    return compose(duration_s=duration_s, obstruction_trace=obstruction, seed=1)


def _is_subsequence(needle: list, haystack: list) -> bool:
    it = iter(haystack)
    return all(n in it for n in needle)


class TestBuildPlaybackPlanCritical:
    def test_loss_transitions_are_marked_critical(self):
        trace = _bursty_trace()
        plan = build_playback_plan(trace, update_interval_s=0.05)
        critical = [u for u in plan if u.critical]
        assert critical, "expected at least one critical (loss-transition) update"
        assert all(u.params.loss_pct in (0.0, 100.0) for u in critical)

    def test_non_loss_trace_has_no_critical_updates(self):
        trace = compose(duration_s=3, seed=1)  # too short for any reconfig burst (15s cadence), no obstruction
        assert not trace.df["loss"].any()
        plan = build_playback_plan(trace, update_interval_s=0.1)
        assert plan
        assert not any(u.critical for u in plan)

    def test_critical_updates_come_in_server_client_pairs(self):
        trace = _bursty_trace()
        plan = build_playback_plan(trace, update_interval_s=0.05)
        critical_at_s = sorted({u.at_s for u in plan if u.critical})
        for at_s in critical_at_s:
            endpoints = {u.endpoint for u in plan if u.at_s == at_s and u.critical}
            assert endpoints == {"server", "client"}


class TestPlayNeverSkipsLoss:
    def test_every_critical_transition_is_applied_in_order_even_when_backend_is_slow(self):
        trace = _bursty_trace()
        # A fine grid (0.02s) over 5s of sim time produces a dense plan -- combined with a slow
        # backend and a high playback speed, this reliably forces play() into its catch-up path.
        plan = build_playback_plan(trace, update_interval_s=0.02)
        expected = {
            "server": [u.params for u in plan if u.critical and u.endpoint == "server"],
            "client": [u.params for u in plan if u.critical and u.endpoint == "client"],
        }
        assert expected["server"], "test trace produced no critical updates -- test is meaningless"

        backend = _RecordingBackend(delay_s=0.02)
        stats = PlaybackStats()
        asyncio.run(play(plan, backend, speed=200.0, max_lag_s=0.1, stats=stats))

        applied = {"server": [], "client": []}
        for endpoint, params, _ in backend.calls:
            applied[endpoint].append(params)

        assert _is_subsequence(expected["server"], applied["server"]), \
            "a critical (loss transition) update for 'server' was skipped"
        assert _is_subsequence(expected["client"], applied["client"]), \
            "a critical (loss transition) update for 'client' was skipped"

    def test_catch_up_actually_skips_non_critical_updates_when_behind(self):
        trace = _bursty_trace()
        plan = build_playback_plan(trace, update_interval_s=0.02)

        backend = _RecordingBackend(delay_s=0.02)
        stats = PlaybackStats()
        asyncio.run(play(plan, backend, speed=200.0, max_lag_s=0.1, stats=stats))

        assert stats.ticks_total > 0
        assert stats.ticks_skipped > 0, "expected the slow backend to force at least one skip"
        # fewer real apply() calls than the full plan -- proof skipping actually reduced work,
        # not just that the counter incremented.
        assert len(backend.calls) < len(plan)

    def test_fast_backend_does_not_skip_anything(self):
        trace = _bursty_trace()
        plan = build_playback_plan(trace, update_interval_s=0.1)

        backend = _RecordingBackend(delay_s=0.0)
        stats = PlaybackStats()
        asyncio.run(play(plan, backend, speed=1000.0, stats=stats))

        assert stats.ticks_skipped == 0
        assert len(backend.calls) == len(plan)


class TestPlayStats:
    def test_stats_ticks_total_matches_number_of_distinct_at_s(self):
        trace = _bursty_trace()
        plan = build_playback_plan(trace, update_interval_s=0.1)
        n_ticks = len({u.at_s for u in plan})

        backend = _RecordingBackend()
        stats = PlaybackStats()
        asyncio.run(play(plan, backend, speed=1000.0, stats=stats))

        assert stats.ticks_total == n_ticks

    def test_lag_is_zero_when_never_behind(self):
        trace = _bursty_trace(n_bursts=2, duration_s=2.0)
        plan = build_playback_plan(trace, update_interval_s=0.5)

        backend = _RecordingBackend(delay_s=0.0)
        stats = PlaybackStats()
        asyncio.run(play(plan, backend, speed=1000.0, stats=stats))

        assert stats.lag_s < 0.5


class TestGridUpdatesVectorizedBinning:
    """`_grid_delay_jitter_updates`/`_grid_bandwidth_updates` bin via a single vectorized
    `groupby` rather than filtering the whole dataframe once per bin, which is O(n_bins * n_rows)
    and costs ~9.4s on the longest real drive (~1.06M 10ms-grid rows) against ~0.3s here.
    These tests pin the exact binning semantics on small, hand-computable data so it cannot change
    behavior."""

    def _df(self):
        # 0.0, 0.1, ..., 2.4s at 10ms resolution isn't necessary here -- a coarser, hand-pickable
        # grid is enough to make the expected bin averages easy to verify by eye.
        timestamps = [0.0, 0.3, 0.6, 1.0, 1.4, 1.8, 2.0]
        return pd.DataFrame({
            "timestamp": timestamps,
            "delay_ms": [10.0, 20.0, 12.0, 40.0, 44.0, 48.0, 50.0],
            "jitter_ms": [1.0, 2.0, 1.2, 4.0, 4.4, 4.8, 5.0],
            "download_mbps": [100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0],
            "upload_mbps": [10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0],
        })

    def test_delay_jitter_bins_average_correctly(self):
        df = self._df()
        updates = _grid_delay_jitter_updates(df, update_interval_s=1.0)
        # bin 0 = [0.0, 1.0): rows at t=0.0,0.3,0.6 -> mean(10,20,12)=14.0, mean(1,2,1.2)=1.4
        # bin 1 = [1.0, 2.0): rows at t=1.0,1.4,1.8 -> mean(40,44,48)=44.0, mean(4,4.4,4.8)=4.4
        # bin 2 = [2.0, 3.0): row at t=2.0 -> 50.0, 5.0
        assert len(updates) == 3
        at0, delay0, jitter0 = updates[0]
        assert at0 == 0.0
        assert np.isclose(delay0, 14.0)
        assert np.isclose(jitter0, 1.4)
        at1, delay1, jitter1 = updates[1]
        assert at1 == 1.0
        assert np.isclose(delay1, 44.0)
        assert np.isclose(jitter1, 4.4)
        at2, delay2, jitter2 = updates[2]
        assert at2 == 2.0
        assert np.isclose(delay2, 50.0)
        assert np.isclose(jitter2, 5.0)

    def test_bandwidth_bins_average_correctly(self):
        df = self._df()
        updates = _grid_bandwidth_updates(df, update_interval_s=1.0)
        assert len(updates) == 3
        at0, dl0, ul0 = updates[0]
        assert np.isclose(dl0, 100.0) and np.isclose(ul0, 10.0)
        at1, dl1, ul1 = updates[1]
        assert np.isclose(dl1, 50.0) and np.isclose(ul1, 5.0)

    def test_empty_bins_are_skipped_not_emitted(self):
        # A gap in the data (no rows between t=1.0 and t=3.0) must not produce a spurious
        # zero-filled or NaN update for the empty bin(s) in between -- matches the old
        # per-bin-filter code's explicit `if bucket.empty: continue`.
        df = pd.DataFrame({
            "timestamp": [0.0, 3.0],
            "delay_ms": [10.0, 20.0],
            "jitter_ms": [1.0, 2.0],
            "download_mbps": [100.0, 100.0],
            "upload_mbps": [10.0, 10.0],
        })
        updates = _grid_delay_jitter_updates(df, update_interval_s=1.0)
        assert [u[0] for u in updates] == [0.0, 3.0]

    def test_matches_naive_per_bin_filter_on_real_composed_trace(self):
        """Cross-checks the vectorized implementation against the original per-bin-filter
        approach directly, on a real (if short) composed trace -- not just hand-picked values."""
        trace = _bursty_trace(n_bursts=4, duration_s=6.0)
        df = trace.df
        update_interval_s = 0.25

        def naive(df, update_interval_s):
            updates = []
            n_bins = int(np.ceil(df["timestamp"].max() / update_interval_s)) + 1
            for i in range(n_bins):
                bin_start, bin_end = i * update_interval_s, (i + 1) * update_interval_s
                bucket = df[(df["timestamp"] >= bin_start) & (df["timestamp"] < bin_end)]
                if bucket.empty:
                    continue
                updates.append((bin_start, float(bucket["delay_ms"].mean()), float(bucket["jitter_ms"].mean())))
            return updates

        expected = naive(df, update_interval_s)
        actual = _grid_delay_jitter_updates(df, update_interval_s=update_interval_s)
        assert len(expected) == len(actual)
        for (at_e, d_e, j_e), (at_a, d_a, j_a) in zip(expected, actual):
            assert abs(at_e - at_a) < 1e-9
            assert abs(d_e - d_a) < 1e-9
            assert abs(j_e - j_a) < 1e-9
