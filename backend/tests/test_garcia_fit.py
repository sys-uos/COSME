import os

import numpy as np
import pandas as pd
import pytest

from backend.models import garcia_fit as gf
from backend.models.garcia import GarciaModel, GarciaParams


def _write_measurement_dir(root, name: str, rows: pd.DataFrame) -> str:
    meas_dir = os.path.join(root, name)
    os.makedirs(meas_dir, exist_ok=True)
    rows.to_csv(os.path.join(meas_dir, "ping_results_FHP.csv"), index=False)
    return meas_dir


def _synthetic_rows(n_normal: int, n_reconfig: int, rng: np.random.Generator) -> pd.DataFrame:
    """Real-shaped rows: normal-window RTT low, elevated RTT scattered across many
    real ~15s reconfig windows (near :12/:27/:42/:57) rather than one long burst --
    a single burst that runs longer than the +-0.5s classification window would
    drift outside it and get mislabeled, which isn't how the real handovers work
    (they recur every cycle, each briefly)."""
    base = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    n_cycles = max(n_normal, n_reconfig) + 1
    cycle_normal = rng.integers(0, n_cycles, size=n_normal)
    phase_normal = rng.uniform(1.0, 11.0, size=n_normal)  # far from the :12 mark
    t_normal = cycle_normal * 15.0 + phase_normal
    rtt_normal = rng.normal(20.0, 1.0, n_normal)  # -> OWD ~10ms

    cycle_reconfig = rng.integers(0, n_cycles, size=n_reconfig)
    phase_reconfig = rng.uniform(-0.4, 0.4, size=n_reconfig)  # inside the +-0.5s window
    t_reconfig = cycle_reconfig * 15.0 + 12.0 + phase_reconfig
    rtt_reconfig = rng.normal(60.0, 1.0, n_reconfig)  # -> OWD ~30ms, clearly elevated

    t_all = np.concatenate([t_normal, t_reconfig])
    rtt_all = np.concatenate([rtt_normal, rtt_reconfig])
    times = [base + pd.Timedelta(seconds=float(t)) for t in t_all]
    return pd.DataFrame({
        "link_name": "FHP", "repetition": 1,
        "timestamp": [t.isoformat() for t in times],
        "icmp_seq": range(len(times)), "ttl": 64, "time_ms": rtt_all,
    })


class TestLoadRttSamples:
    def test_phase_and_owd_computed_correctly(self, tmp_path):
        rng = np.random.default_rng(0)
        df = _synthetic_rows(n_normal=50, n_reconfig=10, rng=rng)
        meas_dir = _write_measurement_dir(tmp_path, "measurement-x", df)
        samples = gf.load_rtt_samples(meas_dir)
        assert len(samples) == 60
        assert np.isclose(samples["owd_ms"].iloc[0], df["time_ms"].iloc[0] / 2.0)
        # the last 10 rows sit right at :12 -- must be flagged as reconfig
        assert samples["is_reconfig"].iloc[-10:].all()
        assert not samples["is_reconfig"].iloc[:50].any()

    def test_missing_file_returns_empty_frame(self, tmp_path):
        meas_dir = os.path.join(tmp_path, "no-rtt-here")
        os.makedirs(meas_dir)
        samples = gf.load_rtt_samples(meas_dir)
        assert samples.empty
        assert list(samples.columns) == ["owd_ms", "phase_s", "is_reconfig", "slot_index"]


class TestFit:
    def test_fit_separates_normal_and_reconfig_and_is_json_roundtrippable(self, tmp_path):
        rng = np.random.default_rng(1)
        for i in range(6):
            df = _synthetic_rows(n_normal=2000, n_reconfig=400, rng=rng)
            _write_measurement_dir(tmp_path, f"measurement-{i}-multicar-onlyping", df)

        fitted = gf.fit(holdout_frac=0.2, seed=42, clipped_dir=str(tmp_path))
        assert fitted.n_normal_samples > 0
        assert fitted.n_reconfig_samples > 0
        assert len(fitted.normal_modes_ms) == 3
        assert len(fitted.normal_weights) == 3
        assert len(fitted.normal_sigmas_ms) == 3
        assert pytest.approx(sum(fitted.normal_weights), abs=1e-6) == 1.0
        # the synthetic reconfig burst (RTT~60ms) is far above the normal
        # burst (RTT~20ms) -- the fitted reconfig mean must reflect that
        # real separation, not just echo the normal distribution.
        assert fitted.reconfig_mean_ms > max(fitted.normal_modes_ms)
        assert set(fitted.train_dirs).isdisjoint(fitted.holdout_dirs)

        as_json = fitted.to_json()
        round_tripped = gf.GarciaFit.from_json(as_json)
        assert round_tripped == fitted

    def test_load_fit_generates_and_caches_when_missing(self, tmp_path, monkeypatch):
        rng = np.random.default_rng(2)
        clipped = tmp_path / "clipped"
        for i in range(6):
            df = _synthetic_rows(n_normal=1000, n_reconfig=200, rng=rng)
            _write_measurement_dir(str(clipped), f"measurement-{i}-multicar-onlyping", df)
        monkeypatch.setattr(gf, "CLIPPED_DIR", str(clipped))

        cache_path = str(tmp_path / "garcia_fit.json")
        assert not os.path.exists(cache_path)
        fitted = gf.load_fit(cache_path)
        assert os.path.exists(cache_path)
        again = gf.load_fit(cache_path)
        assert again == fitted


class TestGarciaParamsDefault:
    def test_default_falls_back_to_invented_when_no_fit_available(self, monkeypatch):
        def _raise():
            raise FileNotFoundError("no fit")
        monkeypatch.setattr("backend.models.garcia_fit.load_fit", lambda: _raise())
        params = GarciaParams.default()
        assert "invented defaults" in params.source

    def test_default_uses_real_fit_when_available(self, monkeypatch):
        fake_fit = gf.GarciaFit(
            train_dirs=["a"], holdout_dirs=["b"], delta_floor_ms=5.0,
            normal_modes_ms=[1.0, 2.0, 3.0], normal_weights=[0.5, 0.3, 0.2],
            normal_sigmas_ms=[0.1, 0.2, 0.3], reconfig_mean_ms=9.0,
            reconfig_sigma_ms=2.0, n_normal_samples=100, n_reconfig_samples=10,
        )
        monkeypatch.setattr("backend.models.garcia_fit.load_fit", lambda: fake_fit)
        params = GarciaParams.default()
        assert params.delta_floor_ms == 5.0
        assert params.reconfig_mean_ms == 9.0
        assert "fitted from" in params.source

    def test_model_uses_default_params_when_none_given(self, monkeypatch):
        fake_fit = gf.GarciaFit(
            train_dirs=["a"], holdout_dirs=["b"], delta_floor_ms=42.0,
            normal_modes_ms=[0.1], normal_weights=[1.0], normal_sigmas_ms=[0.01],
            reconfig_mean_ms=50.0, reconfig_sigma_ms=0.01,
            n_normal_samples=100, n_reconfig_samples=10,
        )
        monkeypatch.setattr("backend.models.garcia_fit.load_fit", lambda: fake_fit)
        model = GarciaModel(seed=1)
        trace = model.generate(duration_s=2, dt_s=0.01)
        assert trace.delay_ms[~trace.reconfig_flag].mean() == pytest.approx(42.1, abs=1.0)
