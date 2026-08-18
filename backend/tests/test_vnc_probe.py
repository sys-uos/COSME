"""Regression tests for docker/probe/vnc_probe.py's deadline-enforcement logic.

vnc_probe.py runs inside the cosme-probe container (not the backend package) and imports
vncdotool at module level -- not installed in this project's own venv (it's a probe-container-only
pip dependency, see docker/probe/Dockerfile). `vncdotool`/`vncdotool.api` are stubbed in
sys.modules so the real script file can be imported here unmodified, purely to unit-test its
dependency-free deadline arithmetic (`_rfb_call`) -- no real VNC connection involved.

There was previously NO test coverage at all for this file -- a real, live bug shipped and went
unnoticed as a result: a single struggling RFB call, retried with no hard deadline, could cost up
to ~15.6s (RFB_CALL_MAX_COST_S), and nothing bounded how many such calls could stack within one
run. Confirmed live: a `--duration 15` run actually took 43.2s once a real, concurrently-shaping
scenario's periodic loss bursts started landing mid-call -- never exercised before showcases could
run concurrently with an actively-shaping scenario (see the recent concurrency-lock fix).
"""
import importlib.util
import sys
import time
import types
from pathlib import Path

import pytest


def _load_vnc_probe():
    if "vncdotool" not in sys.modules:
        stub = types.ModuleType("vncdotool")
        stub_api = types.ModuleType("vncdotool.api")
        stub_api.connect = lambda *a, **k: None
        stub_api.shutdown = lambda: None
        stub.api = stub_api
        sys.modules["vncdotool"] = stub
        sys.modules["vncdotool.api"] = stub_api

    path = Path(__file__).resolve().parents[2] / "docker" / "probe" / "vnc_probe.py"
    spec = importlib.util.spec_from_file_location("cosme_vnc_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vnc_probe = _load_vnc_probe()


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # These tests exercise retry/backoff timing logic without actually waiting through it.
    monkeypatch.setattr(vnc_probe.time, "sleep", lambda s: None)


class TestRfbCallSuccess:
    def test_returns_the_function_result_on_first_success(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        assert vnc_probe._rfb_call(fn) == "ok"
        assert len(calls) == 1  # no unnecessary retries once it succeeds


class TestRfbCallRetriesWithoutDeadline:
    def test_retries_up_to_the_configured_count_then_raises_the_real_exception(self):
        calls = []

        def fn():
            calls.append(1)
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            vnc_probe._rfb_call(fn)
        assert len(calls) == vnc_probe.RFB_CALL_RETRIES + 1

    def test_succeeds_on_a_later_attempt_after_earlier_failures(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ValueError("transient")
            return "recovered"

        assert vnc_probe._rfb_call(fn) == "recovered"
        assert len(calls) == 2


class TestRfbCallDeadline:
    """The actual bug fix: retries must never blindly run past a hard overall deadline,
    regardless of how much of RFB_CALL_RETRIES' budget is nominally left."""

    def test_makes_zero_attempts_once_deadline_has_already_passed(self):
        calls = []

        def fn():
            calls.append(1)
            raise ValueError("should never be called")

        already_passed = time.monotonic() - 1.0
        with pytest.raises(Exception):
            vnc_probe._rfb_call(fn, deadline=already_passed)
        assert calls == []

    def test_raising_with_zero_attempts_is_a_real_exception_not_a_bare_raise_none(self):
        # Regression for a bug caught while writing THIS test: an expired deadline with zero
        # attempts made used to leave the internal "last exception" as None, and `raise None`
        # crashes with TypeError instead of a meaningful error.
        def fn():
            raise AssertionError("should never be called")

        already_passed = time.monotonic() - 1.0
        with pytest.raises(TimeoutError, match="deadline already passed"):
            vnc_probe._rfb_call(fn, deadline=already_passed)

    def test_a_generous_future_deadline_does_not_interfere_with_normal_retries(self):
        calls = []

        def fn():
            calls.append(1)
            raise ValueError("boom")

        far_future = time.monotonic() + 3600
        with pytest.raises(ValueError):
            vnc_probe._rfb_call(fn, deadline=far_future)
        assert len(calls) == vnc_probe.RFB_CALL_RETRIES + 1

    def test_a_shared_stale_deadline_stops_a_later_call_even_after_an_earlier_one_succeeded(self):
        # Mirrors the real bug: main()'s loop computes ONE `deadline` up front and threads it
        # through several separate _rfb_call invocations per tick (activity digest, echo
        # baseline, keyPress, echo poll...). A call succeeding does NOT mean the shared deadline
        # is still valid for the NEXT call in the same tick -- each one must check it
        # independently. `vnc_probe.time.sleep` is neutered by the autouse fixture above (it's
        # the same singleton `time` module the test file itself imports), so this test uses an
        # already-past deadline value directly instead of depending on any real elapsed time.
        stale_deadline = time.monotonic() - 0.001

        def fn():
            return "first call succeeds"

        # First call uses its own generous deadline and succeeds.
        assert vnc_probe._rfb_call(fn, deadline=time.monotonic() + 10) == "first call succeeds"

        calls = []

        def fn2():
            calls.append(1)
            raise ValueError("should never be attempted -- shared deadline already stale")

        # Second call in the same "tick" reuses the earlier, already-stale deadline value.
        with pytest.raises(TimeoutError):
            vnc_probe._rfb_call(fn2, deadline=stale_deadline)
        assert calls == []


class TestRfbCallMaxCost:
    def test_max_cost_matches_the_documented_worst_case_arithmetic(self):
        expected = (vnc_probe.RFB_CALL_TIMEOUT_S * (vnc_probe.RFB_CALL_RETRIES + 1)
                    + vnc_probe.RFB_CALL_BACKOFF_S * vnc_probe.RFB_CALL_RETRIES)
        assert vnc_probe.RFB_CALL_MAX_COST_S == expected

    def test_max_cost_is_a_small_bounded_number_of_seconds(self):
        # Sanity floor/ceiling so a future change to the retry constants can't silently make the
        # per-call worst case huge again without a visible test failure here.
        assert 1.0 < vnc_probe.RFB_CALL_MAX_COST_S < 60.0
