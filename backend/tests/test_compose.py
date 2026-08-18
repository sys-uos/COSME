import numpy as np
import pandas as pd

from backend.compose import compose


class TestCompose:
    def test_dry_no_obstruction_run_has_no_obstruction_loss(self):
        trace = compose(duration_s=30, seed=1)
        assert not trace.df["obstruction_loss"].any()

    def test_loss_is_boolean_or_of_components(self):
        trace = compose(duration_s=60, seed=1)
        expected = trace.df["obstruction_loss"] | trace.df["reconfig_loss"]
        assert (trace.df["loss"] == expected).all()

    def test_loss_never_exceeds_either_component_bound(self):
        # OR composition: loss can't be "more lost" than the union of causes.
        trace = compose(duration_s=60, seed=1)
        assert trace.df["loss"].mean() <= trace.df["obstruction_loss"].mean() + trace.df["reconfig_loss"].mean()

    def test_obstruction_trace_is_incorporated(self):
        obstruction = pd.DataFrame({"timestamp": [10.0], "lossTime": [2.0]})
        trace = compose(duration_s=30, obstruction_trace=obstruction, seed=1)
        window = trace.df[(trace.df["timestamp"] >= 10.0) & (trace.df["timestamp"] < 12.0)]
        assert window["obstruction_loss"].all()
        assert window["loss"].all()

    def test_rain_reduces_download_bandwidth(self):
        rain_time = np.array([0.0, 60.0])
        rain_mm_h = np.array([10.0, 10.0])
        trace = compose(duration_s=30, rain_time_s=rain_time, rain_mm_h=rain_mm_h,
                         nominal_download_mbps=100, seed=1)
        assert (trace.df["download_mbps"] < 100).all()

    def test_no_rain_keeps_nominal_bandwidth(self):
        trace = compose(duration_s=30, nominal_download_mbps=100, nominal_upload_mbps=10, seed=1)
        assert np.allclose(trace.df["download_mbps"], 100)
        assert np.allclose(trace.df["upload_mbps"], 10)

    def test_delay_and_jitter_columns_present_and_nonnegative(self):
        trace = compose(duration_s=30, seed=1)
        assert (trace.df["delay_ms"] >= 0).all()
        assert (trace.df["jitter_ms"] >= 0).all()

    def test_to_csv_round_trips(self, tmp_path):
        trace = compose(duration_s=10, seed=1)
        path = tmp_path / "trace.csv"
        trace.to_csv(str(path))
        reloaded = pd.read_csv(path)
        assert len(reloaded) == len(trace.df)
        assert set(["timestamp", "loss", "delay_ms", "jitter_ms", "download_mbps", "upload_mbps"]).issubset(reloaded.columns)

    def test_deterministic_given_seed(self):
        a = compose(duration_s=20, seed=99)
        b = compose(duration_s=20, seed=99)
        pd.testing.assert_frame_equal(a.df, b.df)
