import numpy as np
import pandas as pd
import pytest

from backend.models import oblos, wetlinks
from backend.models.garcia import GarciaModel, GarciaParams
from backend.models.reconfig_schedule import ReconfigSchedule
from backend.models.zimmermann import ZimmermannModel, extract_reconfig_bursts, fit


class TestZimmermann:
    @pytest.mark.research_data
    def test_fit_produces_train_holdout_split_with_no_overlap(self):
        fitted = fit(holdout_frac=0.2, seed=1)
        assert set(fitted.train_dirs).isdisjoint(fitted.holdout_dirs)
        assert len(fitted.train_dirs) > len(fitted.holdout_dirs)

    def test_fit_is_deterministic_given_seed(self):
        a = fit(holdout_frac=0.2, seed=42)
        b = fit(holdout_frac=0.2, seed=42)
        assert a.holdout_dirs == b.holdout_dirs
        assert a.train_dirs == b.train_dirs

    @pytest.mark.research_data
    def test_extract_reconfig_bursts_returns_subset_of_full_trace(self):
        fitted = fit(seed=1)
        sample_dir_name = fitted.train_dirs[0]
        import os
        from backend.models.zimmermann import CLIPPED_DIR
        bursts = extract_reconfig_bursts(os.path.join(CLIPPED_DIR, sample_dir_name))
        assert "lossTime" in bursts.columns
        assert (bursts["lossTime"] >= 0).all()

    def test_sample_burst_duration_is_positive_and_from_empirical_set(self):
        model = ZimmermannModel(seed=1)
        samples = [model.sample_burst_duration() for _ in range(100)]
        assert all(s > 0 for s in samples)
        assert all(s in model._durations for s in samples)

    def test_summary_reports_expected_keys(self):
        model = ZimmermannModel(seed=1)
        summary = model.summary()
        for key in ("n_samples", "n_train_dirs", "n_holdout_dirs", "mean_s", "median_s", "p90_s", "p99_s"):
            assert key in summary


class TestReconfigSchedule:
    def test_events_spaced_by_period(self):
        sched = ReconfigSchedule(duration_s=200, seed=1, phase_offset_s=0.0)
        starts = [e.start_s for e in sched.events]
        diffs = np.diff(starts)
        assert np.allclose(diffs, sched.period_s)

    def test_is_reconfiguring_true_only_inside_events(self):
        sched = ReconfigSchedule(duration_s=100, seed=1, phase_offset_s=5.0)
        ev = sched.events[0]
        assert sched.is_reconfiguring(ev.start_s + ev.duration_s / 2)
        assert not sched.is_reconfiguring(ev.start_s - 1.0)

    def test_boolean_series_matches_is_reconfiguring(self):
        sched = ReconfigSchedule(duration_s=60, seed=2)
        times = np.arange(0, 60, 0.01)
        series = sched.as_boolean_series(times)
        # spot check against the scalar method
        for i in range(0, len(times), 500):
            assert series[i] == sched.is_reconfiguring(times[i])


class TestGarcia:
    def test_generate_produces_expected_shape(self):
        model = GarciaModel(seed=1)
        trace = model.generate(duration_s=10, dt_s=0.01)
        n = int(10 / 0.01)
        assert len(trace.time_s) == n
        assert len(trace.delay_ms) == n
        assert len(trace.jitter_smooth_ms) == n

    def test_delay_never_negative(self):
        model = GarciaModel(seed=2)
        trace = model.generate(duration_s=30, dt_s=0.01)
        assert (trace.delay_ms >= 0).all()

    def test_delay_never_below_floor(self):
        # The GMM/reconfig components are fit on residuals truncated at 0 (see
        # garcia_fit.py), so a raw Gaussian draw's left tail must be re-truncated
        # at generation time or delay_ms can fall below delta_floor_ms -- and
        # since the same delay is applied on both endpoints, RTT (~2x OWD) would
        # then dip below the real Zimmermann-measured minimum.
        model = GarciaModel(seed=5)
        trace = model.generate(duration_s=60, dt_s=0.01)
        assert (trace.delay_ms >= model.params.delta_floor_ms).all()

    def test_baseline_varies_slot_to_slot(self):
        # Each ~15s dwell between handovers should get its own baseline (a
        # handover connects to a different satellite with a different path
        # length), not just an identical distribution replayed every slot --
        # this is what makes jitter spike at reconfig boundaries, not just
        # during the reconfig window itself.
        params = GarciaParams(slot_baseline_sigma_ms=10.0, normal_sigmas_ms=(0.01, 0.01, 0.01),
                               normal_modes_ms=(0.0, 0.0, 0.0))
        model = GarciaModel(params=params, seed=6)
        schedule = ReconfigSchedule(duration_s=300, period_s=15.0, phase_offset_s=0.0, seed=6)
        trace = model.generate(duration_s=300, dt_s=0.05, schedule=schedule)
        slot_idx = (trace.time_s // 15.0).astype(int)
        non_reconfig = ~trace.reconfig_flag
        slot_means = [
            trace.delay_ms[non_reconfig & (slot_idx == i)].mean()
            for i in range(slot_idx.max() + 1)
            if (non_reconfig & (slot_idx == i)).any()
        ]
        assert len(slot_means) > 5
        assert np.std(slot_means) > 1.0  # near-zero per-sample noise, so this is almost all slot offset

    def test_reconfig_slots_have_higher_mean_delay(self):
        # The paper's core claim: reconfiguration slots carry extra queueing delay.
        model = GarciaModel(seed=3)
        trace = model.generate(duration_s=120, dt_s=0.01)
        if trace.reconfig_flag.any() and (~trace.reconfig_flag).any():
            assert trace.delay_ms[trace.reconfig_flag].mean() > trace.delay_ms[~trace.reconfig_flag].mean()

    def test_custom_params_respected(self):
        params = GarciaParams(delta_floor_ms=100.0, normal_sigmas_ms=(0.001, 0.001, 0.001))
        model = GarciaModel(params=params, seed=4)
        trace = model.generate(duration_s=5, dt_s=0.01)
        # with near-zero variance components, delay should hover close to delta_floor_ms + small terms
        assert trace.delay_ms[~trace.reconfig_flag].mean() > 95


class TestWetLinks:
    def test_fit_factors_are_between_zero_and_one_point_five(self):
        fitted = wetlinks.fit("Osnabrück")
        for b in wetlinks.BUCKET_NAMES:
            assert 0.0 < fitted.download_factor[b] < 1.5
            assert 0.0 < fitted.upload_factor[b] < 1.5

    def test_none_bucket_is_baseline(self):
        fitted = wetlinks.fit("Osnabrück")
        assert fitted.download_factor["none"] == pytest.approx(1.0)
        assert fitted.upload_factor["none"] == pytest.approx(1.0)

    def test_heavy_rain_reduces_download_more_than_light(self):
        fitted = wetlinks.fit("Osnabrück")
        assert fitted.download_factor["heavy"] < fitted.download_factor["light"] < 1.0

    def test_model_throughput_at_applies_factor(self):
        model = wetlinks.WetLinksModel(nominal_download_mbps=100, nominal_upload_mbps=10, site="Osnabrück")
        dl_dry, ul_dry = model.throughput_at(0.0)
        dl_heavy, ul_heavy = model.throughput_at(10.0)
        assert dl_dry == pytest.approx(100.0)
        assert dl_heavy < dl_dry

    def test_generate_matches_per_sample_bucketing(self):
        model = wetlinks.WetLinksModel(nominal_download_mbps=100, nominal_upload_mbps=10, site="Osnabrück")
        times = np.array([0, 1, 2, 3])
        rain = np.array([0.0, 1.0, 3.0, 10.0])
        df = model.generate(times, rain)
        expected = [model.throughput_at(r)[0] for r in rain]
        assert df["download_mbps"].tolist() == pytest.approx(expected, rel=1e-6)


class TestOblos:
    @pytest.mark.research_data
    def test_list_available_drives_nonempty(self):
        drives = oblos.list_available_drives()
        assert len(drives) > 50

    @pytest.mark.research_data
    def test_load_obstruction_trace_has_expected_columns(self):
        drives = oblos.list_available_drives()
        trace = oblos.load_obstruction_trace(drives[0])
        assert list(trace.columns) == ["timestamp", "lossTime"]

    def test_as_boolean_series_true_only_during_events(self):
        trace = pd.DataFrame({"timestamp": [1.0, 5.0], "lossTime": [0.5, 0.2]})
        times = np.array([0.0, 1.2, 2.0, 5.1, 6.0])
        series = oblos.as_boolean_series(trace, times)
        assert list(series) == [False, True, False, True, False]

    def test_generate_live_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            oblos.generate_live(52.0, 8.0, 52.1, 8.1)
