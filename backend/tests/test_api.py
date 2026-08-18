import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import api

# Context-managed so Starlette's anyio portal/event loop stays alive across
# calls -- required for scenario.start()'s asyncio.create_task() background
# work to actually progress between separate client.get() polls (verified
# separately against a live uvicorn server; this is a TestClient lifecycle
# requirement, not an application behavior difference).
_client_cm = TestClient(api.app)
client = _client_cm.__enter__()


def teardown_module(module):
    _client_cm.__exit__(None, None, None)


@pytest.fixture(autouse=True)
def _clear_active_run_state():
    """Module-wide test isolation for the concurrency lock (api._require_free_testbed()).

    SCENARIOS/SHOWCASE_JOBS are plain module-level dicts that persist for the whole pytest
    session (mirroring real server behavior -- see api.py). Before this fixture existed, a
    scenario left "running" by one test (e.g. one that doesn't wait for completion, or a
    deliberately-long-duration test like TestConcurrencyLock's own) would make the concurrency
    lock reject a completely unrelated LATER test's own POST /api/scenarios with a spurious 409.
    Clearing before AND after each test keeps this file's tests independent of run order.
    """
    api.SCENARIOS.clear()
    api.SHOWCASE_JOBS.clear()
    yield
    api.SCENARIOS.clear()
    api.SHOWCASE_JOBS.clear()


@pytest.fixture(autouse=True)
def _default_no_real_docker(monkeypatch):
    """Real bug found and fixed this session: several test classes here (e.g.
    TestScenarioUpdateInterval, TestCustomObstructionTrace) create real scenarios with NO
    per-class `_docker_available` mock, on the (previously implicit, never-enforced) assumption
    that the environment running this file never has real Docker available. That assumption is
    FALSE in this environment -- confirmed live: running this suite measurably re-shaped the
    real cosme-client/cosme-server containers (their qdisc handle changed on every run) as a
    pure side effect of test collection, which could just as easily clobber a real user's
    actually-running scenario. This file is meant to be the hermetic "not docker" suite --
    `backend/tests/test_docker_integration.py` (a separate file, `-m docker`) is where real
    Docker interaction belongs. Default OFF here, globally; a test that specifically wants real
    (or specifically enabled) Docker behavior still monkeypatches this within its own body/
    fixture, same as several classes already do -- that continues to work, since a call to
    monkeypatch.setattr from within a test overrides this fixture's default for that test only.
    """
    monkeypatch.setattr(api, "_docker_available", lambda: False)


class TestHealthAndSystem:
    def test_health(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_congestion_controls_falls_back_without_docker(self):
        resp = client.get("/api/system/congestion-controls")
        assert resp.status_code == 200
        body = resp.json()
        assert "cubic" in body["available"]


class TestDrives:
    @pytest.mark.research_data
    def test_list_drives(self):
        resp = client.get("/api/drives")
        assert resp.status_code == 200
        assert len(resp.json()["drives"]) > 50


class TestScenarioWeatherModes:
    @pytest.fixture(autouse=True)
    def _no_docker(self, monkeypatch):
        # These tests exercise weather-preset logic, not Docker execution, so
        # pin them to the fast/deterministic DryRunNetemBackend regardless of
        # whether real cosme-* containers happen to be running on the machine
        # executing the suite -- against the real docker backend, each update
        # tick issues real `docker exec` subprocess calls whose overhead can
        # push _wait_done past its timeout even at speed=100.
        monkeypatch.setattr(api, "_docker_available", lambda: False)

    def _create(self, **overrides):
        payload = {"duration_s": 10, "speed": 100, "seed": 1}
        payload.update(overrides)
        resp = client.post("/api/scenarios", json=payload)
        assert resp.status_code == 200, resp.text
        return resp.json()["id"]

    def _wait_done(self, scenario_id, timeout_s=5):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = client.get(f"/api/scenarios/{scenario_id}/status").json()
            if not status["running"]:
                return status
            time.sleep(0.05)
        raise TimeoutError("scenario did not finish in time")

    def test_dry_preset_gives_zero_rain(self):
        sid = self._create(weather_mode="dry")
        self._wait_done(sid)
        metrics = client.get(f"/api/scenarios/{sid}/metrics").json()
        assert metrics["current"]["rain_mm_h"] == 0.0
        assert metrics["current"]["download_mbps"] == pytest.approx(150.0)

    def test_heavy_preset_reduces_bandwidth(self):
        sid = self._create(weather_mode="heavy", nominal_download_mbps=150.0)
        self._wait_done(sid)
        metrics = client.get(f"/api/scenarios/{sid}/metrics").json()
        assert metrics["current"]["rain_mm_h"] == pytest.approx(7.0)
        assert metrics["current"]["download_mbps"] < 150.0

    def test_moderate_preset_between_light_and_heavy(self):
        light_sid = self._create(weather_mode="light")
        self._wait_done(light_sid)
        light_dl = client.get(f"/api/scenarios/{light_sid}/metrics").json()["current"]["download_mbps"]

        heavy_sid = self._create(weather_mode="heavy")
        self._wait_done(heavy_sid)
        heavy_dl = client.get(f"/api/scenarios/{heavy_sid}/metrics").json()["current"]["download_mbps"]

        assert heavy_dl < light_dl

    def test_status_reports_backend_mode(self):
        sid = self._create(weather_mode="dry")
        self._wait_done(sid)
        status = client.get(f"/api/scenarios/{sid}/status").json()
        assert status["backend_mode"] in ("docker", "dry_run")


class TestScenarioUpdateInterval:
    def test_update_interval_clamped_to_one_second_default(self):
        resp = client.post("/api/scenarios", json={"duration_s": 30, "speed": 100, "seed": 2})
        body = resp.json()
        # 30s duration at <=1s cadence should yield at least ~30 grid updates (x2 endpoints, plus reconfig/loss events)
        assert body["n_updates"] >= 30

    def test_coarser_requested_interval_is_clamped(self):
        resp_fine = client.post("/api/scenarios", json={"duration_s": 30, "speed": 100, "seed": 2, "update_interval_s": 1.0})
        # n_updates is already captured in the response above (computed synchronously at
        # creation time) -- stop it before creating the second scenario so the concurrency
        # lock (only one scenario/showcase may use the shared testbed at a time) doesn't
        # reject the second POST while the first is still playing back.
        client.post(f"/api/scenarios/{resp_fine.json()['id']}/stop")
        resp_coarse = client.post("/api/scenarios", json={"duration_s": 30, "speed": 100, "seed": 2, "update_interval_s": 15.0})
        # Both should clamp to the same <=1s cadence, so update counts should match (same seed).
        assert resp_fine.json()["n_updates"] == resp_coarse.json()["n_updates"]


class TestUnknownScenario:
    def test_status_404(self):
        resp = client.get("/api/scenarios/does-not-exist/status")
        assert resp.status_code == 404

    def test_stop_404(self):
        resp = client.post("/api/scenarios/does-not-exist/stop")
        assert resp.status_code == 404


class TestScenarioStop:
    @pytest.fixture(autouse=True)
    def _no_docker(self, monkeypatch):
        monkeypatch.setattr(api, "_docker_available", lambda: False)

    def test_stop_cancels_and_reports_not_running(self):
        resp = client.post("/api/scenarios", json={"duration_s": 300, "speed": 1, "seed": 3})
        sid = resp.json()["id"]
        assert client.get(f"/api/scenarios/{sid}/status").json()["running"]

        stop_resp = client.post(f"/api/scenarios/{sid}/stop")
        assert stop_resp.status_code == 200
        assert stop_resp.json()["running"] is False
        assert client.get(f"/api/scenarios/{sid}/status").json()["running"] is False

    def test_stop_clears_netem_state_dry_run(self):
        # DryRunNetemBackend just logs commands, so "cleared" means the reset
        # command was issued -- check the log for the `tc qdisc del` call
        # NetemBackend.reset() emits.
        resp = client.post("/api/scenarios", json={"duration_s": 300, "speed": 1, "seed": 4})
        sid = resp.json()["id"]
        client.post(f"/api/scenarios/{sid}/stop")
        scenario = api.SCENARIOS[sid]
        assert any("qdisc del" in line for line in scenario.backend.log[-4:])

    def test_elapsed_s_freezes_after_stop_instead_of_climbing_forever(self):
        # Real bug: elapsed_s() was pure wall-clock math with no dependency on `running`, so it
        # kept climbing (up to duration_s) even after playback had genuinely stopped -- from the
        # dashboard this looked like "I pressed Stop but the plots keep moving," since /status and
        # /metrics kept returning a growing elapsed_s and a correspondingly sliding metrics window
        # well after Stop was pressed. speed=1000 makes the bug (if it regressed) obvious within a
        # tiny real sleep, since a working freeze must show ZERO drift regardless of speed.
        resp = client.post("/api/scenarios", json={"duration_s": 300, "speed": 1000, "seed": 5})
        sid = resp.json()["id"]
        time.sleep(0.05)
        client.post(f"/api/scenarios/{sid}/stop")
        elapsed_at_stop = client.get(f"/api/scenarios/{sid}/status").json()["elapsed_s"]
        time.sleep(0.3)
        elapsed_after_wait = client.get(f"/api/scenarios/{sid}/status").json()["elapsed_s"]
        assert elapsed_after_wait == elapsed_at_stop


class TestResetContainers:
    def test_503_without_docker(self, monkeypatch):
        monkeypatch.setattr(api, "_docker_available", lambda: False)
        resp = client.post("/api/system/reset-containers")
        assert resp.status_code == 503


class TestOblosSimulateEndpoint:
    def test_unreachable_osrm_reports_error_not_crash(self):
        resp = client.post("/api/oblos/simulate", json={
            "start_lat": 52.0, "start_lon": 8.0, "end_lat": 52.1, "end_lon": 8.1,
            "osrm_url": "http://localhost:1",  # nothing listens here
        })
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        deadline = time.monotonic() + 10
        job = client.get(f"/api/oblos/simulate/{job_id}").json()
        while job["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.1)
            job = client.get(f"/api/oblos/simulate/{job_id}").json()
        assert job["status"] == "error"
        assert "error" in job

    def test_unknown_job_404(self):
        resp = client.get("/api/oblos/simulate/does-not-exist")
        assert resp.status_code == 404


class TestShowcasesWithoutDocker:
    """Showcases should fail gracefully (503) when Docker/containers aren't available.

    Forced via monkeypatch rather than relying on the test machine actually
    lacking Docker -- real cosme-* containers may legitimately be running
    (e.g. during manual verification), and these tests should hold either way.
    """

    @pytest.fixture(autouse=True)
    def _no_docker(self, monkeypatch):
        monkeypatch.setattr(api, "_docker_available", lambda: False)

    @pytest.mark.parametrize("endpoint,body", [
        ("/api/showcase/file-transfer", {}),
        ("/api/showcase/video-conferencing", {"duration_s": 5}),
        ("/api/showcase/voip", {"duration_s": 5}),
        ("/api/showcase/remote-desktop", {}),
        ("/api/showcase/surveillance", {}),
    ])
    def test_all_showcases_503_without_docker(self, endpoint, body):
        resp = client.post(endpoint, json=body)
        assert resp.status_code == 503

    def test_unknown_job_404(self):
        resp = client.get("/api/showcase/jobs/does-not-exist")
        assert resp.status_code == 404


class TestShowcaseRegistry:
    def test_apps_listed_with_transport_and_metrics(self):
        apps = client.get("/api/showcase/apps").json()["apps"]
        assert {a["id"] for a in apps} == {
            "file_transfer", "video_conferencing", "voip", "remote_desktop", "surveillance",
        }
        for a in apps:
            assert a["transport"] in ("tcp", "udp")
            assert a["metrics"], f"{a['id']} has no metrics"
            for m in a["metrics"]:
                assert {"key", "label", "unit", "decimals"} <= set(m)

    def test_paper_promise_two_tcp_three_udp(self):
        apps = client.get("/api/showcase/apps").json()["apps"]
        transports = [a["transport"] for a in apps]
        assert transports.count("tcp") == 2 and transports.count("udp") == 3


class TestAppStats:
    def test_post_and_get_round_trip(self):
        resp = client.post("/api/showcase/app-stats", json={"app": "voip", "t": 123.0})
        assert resp.status_code == 200
        stats = client.get("/api/showcase/app-stats?app=voip&n=5").json()["stats"]
        assert any(s.get("t") == 123.0 for s in stats)

    def test_missing_or_unknown_app_rejected(self):
        assert client.post("/api/showcase/app-stats", json={"t": 1.0}).status_code == 400
        assert client.post("/api/showcase/app-stats", json={"app": "nope"}).status_code == 400
        assert client.get("/api/showcase/app-stats?app=nope").status_code == 404

    def test_ring_buffer_caps_at_max(self):
        for i in range(api.MAX_APP_STATS + 10):
            client.post("/api/showcase/app-stats", json={"app": "surveillance", "i": i})
        assert len(api.APP_STATS["surveillance"]) == api.MAX_APP_STATS


class TestShowcaseQoe:
    def test_unknown_app_404(self):
        assert client.get("/api/showcase/qoe/nope").status_code == 404

    def test_no_data_yet(self):
        api.APP_STATS["remote_desktop"] = []
        body = client.get("/api/showcase/qoe/remote_desktop").json()
        assert body["source"] == "no data yet"
        assert body["metrics"] == {}

    def test_voip_qoe_from_live_samples(self):
        api.APP_STATS["voip"] = [
            {"app": "voip", "t": 1.0, "rtt_ms": 80.0,
             "audio": {"loss_frac": 0.01, "jitter_ms": 10.0, "bitrate_kbps": 48.0}},
            {"app": "voip", "t": 2.0, "rtt_ms": 120.0,
             "audio": {"loss_frac": 0.03, "jitter_ms": 20.0, "bitrate_kbps": 44.0}},
        ]
        body = client.get("/api/showcase/qoe/voip").json()
        m = body["metrics"]
        assert 1.0 <= m["mos"] <= 4.5
        assert m["rtt_ms"] == 100.0
        assert m["loss_pct"] == 2.0
        assert "live samples" in body["source"]
        api.APP_STATS["voip"] = []


class TestShowcaseQoeSourcePrecedence:
    """A finished run must be scored from its in-container totals, not from the live samples.

    The live per-second samples are POSTed over the emulated link, so they go missing exactly
    when the link is worst -- measured on a real 600s run, the seconds that arrived averaged a
    3.8% loss duty cycle while the 34 gaps whose samples never made it averaged 36.4%. The
    endpoint originally applied `metrics.update(live)` LAST, so those biased values silently
    overwrote the unbiased ones (the dashboard showed ~16 surveillance freezes where the probe's
    own end-of-run summary said 165). While a run is still in flight there is nothing else to
    show, so live wins then -- these two tests pin both directions.
    """

    def _seed(self, running: bool):
        api.SHOWCASE_JOBS.clear()
        api.SHOWCASE_JOBS["finished"] = {
            "app": "surveillance", "status": "done", "started_at": 1.0,
            "duration_s": 600.0, "freeze_count": 165, "total_freeze_s": 66.3,
            "mean_bitrate_kbps": 842.0, "frames_received": 12401,
        }
        if running:
            api.SHOWCASE_JOBS["live"] = {"app": "surveillance", "status": "running",
                                          "started_at": 2.0}
        # Live samples imply far fewer freezes than the probe actually counted.
        api.APP_STATS["surveillance"] = [
            {"app": "surveillance", "t": float(i), "fps": 25.0, "bitrate_kbps": 900.0,
             "in_freeze": i in (3, 4)}
            for i in range(10)
        ]

    def test_finished_run_totals_win_over_live_samples(self):
        self._seed(running=False)
        m = client.get("/api/showcase/qoe/surveillance").json()
        assert m["metrics"]["freeze_count"] == 165, "live samples overwrote the in-container total"
        assert m["metrics"]["total_freeze_s"] == 66.3
        assert "in-container totals" in m["source"]
        api.SHOWCASE_JOBS.clear()
        api.APP_STATS["surveillance"] = []

    def test_live_samples_win_while_a_run_is_in_flight(self):
        self._seed(running=True)
        m = client.get("/api/showcase/qoe/surveillance").json()
        # 1 freeze episode in the seeded samples -- the PREVIOUS run's 165 must not be shown as
        # if it described the run currently on the wire.
        assert m["metrics"]["freeze_count"] == 1
        assert "run in progress" in m["source"]
        api.SHOWCASE_JOBS.clear()
        api.APP_STATS["surveillance"] = []

    def test_voip_finished_run_scored_from_packet_totals(self):
        api.SHOWCASE_JOBS.clear()
        api.APP_STATS["voip"] = []
        api.SHOWCASE_JOBS["done"] = {
            "app": "voip", "status": "done", "started_at": 1.0,
            "totals": {"packets": {"audio": {"packets_lost": 900, "packets_received": 9100,
                                              "loss_frac": 0.09}},
                        "rtt_ms_mean": 30.0, "jitter_ms_mean": {"audio": 4.0},
                        "audio_bitrate_kbps_mean": 112.0, "n_samples_generated": 598},
        }
        m = client.get("/api/showcase/qoe/voip").json()["metrics"]
        # ratio of sums over the whole run, not a mean of per-second fractions
        assert m["loss_pct"] == 9.0
        assert m["rtt_ms"] == 30.0
        api.SHOWCASE_JOBS.clear()


class TestTraceDecimation:
    def test_brief_reconfig_bursts_survive_stride_decimation(self):
        # Regression: naive iloc[::step] stride sampling silently drops most
        # short-lived loss events over a long trace -- confirmed live, a long
        # scenario with ~80 real 15s-cadence reconfig bursts showed only 3-6
        # of them in the trace timeline. Build a synthetic 20min/10ms-grid
        # trace (120,000 rows) with an 80-event, 100ms-wide reconfig burst
        # every 15s -- exactly the shape a real composed trace has -- and
        # confirm decimating to 2000 points still reports all 80 as lost.
        n = 120_000
        dt = 0.01
        timestamps = np.arange(n) * dt
        loss = np.zeros(n, dtype=bool)
        n_events = 0
        t = 8.67  # a real-ish random phase offset -- t=0 would coincidentally
        # align the burst grid with the stride-sampling grid at some max_points
        # values, masking how badly naive stride sampling actually does (confirmed:
        # with phase 0 and max_points=2000, old code visible was a coincidental
        # 80/80; with this phase it drops to 0/80 -- the real ReconfigSchedule's
        # phase_offset_s is `rng.uniform(0, period_s)`, essentially never 0).
        while t < timestamps[-1]:
            start_idx = int(t / dt)
            end_idx = min(n, start_idx + 10)  # 100ms burst
            loss[start_idx:end_idx] = True
            n_events += 1
            t += 15.0
        df = pd.DataFrame({
            "timestamp": timestamps, "loss": loss, "obstruction_loss": loss,
            "reconfig_loss": loss, "delay_ms": np.zeros(n), "jitter_ms": np.zeros(n),
            "download_mbps": np.full(n, 100.0), "upload_mbps": np.full(n, 10.0),
        })
        decimated = api._decimate_trace(df, max_points=2000)
        n_lost_buckets = sum(1 for row in decimated if row["reconfig_loss"])
        assert n_events > 75  # sanity: the synthetic trace really has ~80 events
        assert n_lost_buckets == n_events  # every single one must survive decimation

    def test_short_frame_returned_unchanged(self):
        df = pd.DataFrame({"timestamp": [0.0, 0.1], "loss": [False, True],
                            "obstruction_loss": [False, True], "reconfig_loss": [False, False],
                            "delay_ms": [10.0, 12.0], "jitter_ms": [1.0, 1.0],
                            "download_mbps": [100.0, 100.0], "upload_mbps": [10.0, 10.0]})
        assert api._decimate_trace(df, max_points=2000) == df.to_dict(orient="records")


class TestCustomObstructionTrace:
    def test_custom_trace_used_instead_of_drive(self):
        resp = client.post("/api/scenarios", json={
            "duration_s": 30, "speed": 100, "seed": 5,
            "custom_obstruction_trace": [{"timestamp": 5.0, "lossTime": 1.0}],
        })
        assert resp.status_code == 200
        sid = resp.json()["id"]
        deadline = time.monotonic() + 5
        while client.get(f"/api/scenarios/{sid}/status").json()["running"] and time.monotonic() < deadline:
            time.sleep(0.05)
        trace = client.get(f"/api/scenarios/{sid}/trace?max_points=3000").json()["trace"]
        assert any(row["obstruction_loss"] for row in trace)


class TestConcurrencyLock:
    """There's exactly one physical shaped link (cosme-client<->cosme-server) -- these confirm
    a second scenario/showcase can't silently start while one is already using it (see
    api._require_free_testbed()'s own docstring for the corruption this prevents)."""

    def test_second_scenario_rejected_while_first_running(self, monkeypatch):
        monkeypatch.setattr(api, "_docker_available", lambda: False)
        first = client.post("/api/scenarios", json={"duration_s": 300, "speed": 1, "seed": 11})
        assert first.status_code == 200, first.text
        try:
            second = client.post("/api/scenarios", json={"duration_s": 10, "speed": 100, "seed": 12})
            assert second.status_code == 409
            assert "busy" in second.json()["detail"]
        finally:
            client.post(f"/api/scenarios/{first.json()['id']}/stop")

    def test_showcase_allowed_while_scenario_running(self, monkeypatch):
        # The actual bug report this answers: a showcase's traffic is meant to flow THROUGH a
        # concurrently running scenario's live netem shaping (see api._require_free_testbed's own
        # docstring) -- showcases never touch tc/netem themselves, so this is not a resource
        # conflict. An earlier version of the lock rejected this combination outright, which
        # forced users to stop the scenario before running a showcase -- silently leaving the
        # link completely unshaped for the whole showcase (reported as "loss conditions are not
        # applied" during a VoIP call).
        monkeypatch.setattr(api, "_docker_available", lambda: False)
        scenario_resp = client.post("/api/scenarios", json={"duration_s": 300, "speed": 1, "seed": 13})
        assert scenario_resp.status_code == 200, scenario_resp.text
        try:
            # _launch_showcase's OWN precondition also needs Docker "available" to get past its
            # 503 gate and reach the concurrency check -- doesn't retroactively change the
            # already-created scenario above (its DryRunNetemBackend was fixed at creation time).
            monkeypatch.setattr(api, "_docker_available", lambda: True)
            resp = client.post("/api/showcase/file-transfer", json={})
            assert resp.status_code != 409
        finally:
            monkeypatch.setattr(api, "_docker_available", lambda: False)
            client.post(f"/api/scenarios/{scenario_resp.json()['id']}/stop")

    def test_scenario_allowed_while_showcase_running(self, monkeypatch):
        # Injects SHOWCASE_JOBS state directly rather than racing a real background-task launch
        # (whose exact start/finish timing relative to the test's own next line isn't
        # deterministic under TestClient) -- still exercises the real _active_run() code path,
        # just the SHOWCASE_JOBS branch specifically (the scenario branch is covered above).
        # A running showcase must NOT block starting a scenario -- that's the same "showcase
        # traffic flows through a live-shaped link" pattern, just started in the other order.
        monkeypatch.setattr(api, "_docker_available", lambda: False)
        api.SHOWCASE_JOBS["test-fake-job"] = {"status": "running", "app": "file_transfer"}
        try:
            resp = client.post("/api/scenarios", json={"duration_s": 10, "speed": 100, "seed": 14})
            assert resp.status_code == 200, resp.text
            client.post(f"/api/scenarios/{resp.json()['id']}/stop")
        finally:
            del api.SHOWCASE_JOBS["test-fake-job"]

    def test_second_showcase_rejected_while_first_running(self, monkeypatch):
        # The real remaining conflict: two showcases (same or different app) sharing the
        # containers/APP_STATS/LIVE_FRAMES keys -- see _require_free_testbed's docstring.
        monkeypatch.setattr(api, "_docker_available", lambda: True)
        api.SHOWCASE_JOBS["test-fake-job"] = {"status": "running", "app": "file_transfer"}
        try:
            resp = client.post("/api/showcase/file-transfer", json={})
            assert resp.status_code == 409
            assert "busy" in resp.json()["detail"]
        finally:
            del api.SHOWCASE_JOBS["test-fake-job"]

    def test_scenario_allowed_after_first_finishes(self, monkeypatch):
        monkeypatch.setattr(api, "_docker_available", lambda: False)
        first = client.post("/api/scenarios", json={"duration_s": 5, "speed": 100, "seed": 15})
        sid = first.json()["id"]
        deadline = time.monotonic() + 5
        while client.get(f"/api/scenarios/{sid}/status").json()["running"] and time.monotonic() < deadline:
            time.sleep(0.05)
        second = client.post("/api/scenarios", json={"duration_s": 5, "speed": 100, "seed": 16})
        assert second.status_code == 200, second.text

    def test_active_run_discovers_a_scenario_this_client_never_polled(self, monkeypatch):
        # The actual bug report this answers: a scenario's id only ever lived in the browser
        # tab that created it (no persistence) -- reload the page, or load it from a completely
        # different "client" (simulated here by never touching `first`'s id again), and there
        # was previously no way to discover it was still running, let alone stop it.
        monkeypatch.setattr(api, "_docker_available", lambda: False)
        first = client.post("/api/scenarios", json={"duration_s": 300, "speed": 1, "seed": 17})
        sid = first.json()["id"]
        try:
            active = client.get("/api/system/active-run").json()
            assert active["active"]["kind"] == "scenario"
            assert active["active"]["id"] == sid
            assert active["active"]["duration_s"] == 300.0
        finally:
            client.post(f"/api/scenarios/{sid}/stop")

    def test_active_run_none_when_testbed_free(self, monkeypatch):
        monkeypatch.setattr(api, "_docker_available", lambda: False)
        assert client.get("/api/system/active-run").json() == {"active": None}

    def test_stop_showcase_job_frees_the_lock(self, monkeypatch):
        # cancel_showcase() itself shells out to `docker exec ... pkill` per app -- mocked here
        # since this is a plain unit test with no real containers, but the actual point under
        # test is that stopping a "running" job flips its status and frees _active_run().
        monkeypatch.setattr(api.watchdog, "cancel_showcase", lambda app_id: None)
        api.SHOWCASE_JOBS["test-stop-job"] = {"status": "running", "app": "file_transfer"}
        try:
            assert client.get("/api/system/active-run").json()["active"]["id"] == "test-stop-job"
            resp = client.post("/api/showcase/jobs/test-stop-job/stop")
            assert resp.status_code == 200
            assert resp.json()["status"] == "error"
            assert api.SHOWCASE_JOBS["test-stop-job"]["error"] == "stopped by user"
            assert client.get("/api/system/active-run").json() == {"active": None}
        finally:
            del api.SHOWCASE_JOBS["test-stop-job"]

    def test_stop_showcase_job_404_for_unknown_job(self):
        resp = client.post("/api/showcase/jobs/does-not-exist/stop")
        assert resp.status_code == 404


class TestExportImportReplay:
    @pytest.fixture(autouse=True)
    def _no_docker(self, monkeypatch):
        monkeypatch.setattr(api, "_docker_available", lambda: False)

    def _create_and_wait(self, **overrides):
        payload = {"duration_s": 20, "speed": 100, "seed": 7}
        payload.update(overrides)
        resp = client.post("/api/scenarios", json=payload)
        assert resp.status_code == 200, resp.text
        sid = resp.json()["id"]
        deadline = time.monotonic() + 5
        while client.get(f"/api/scenarios/{sid}/status").json()["running"] and time.monotonic() < deadline:
            time.sleep(0.05)
        return sid

    def test_compose_is_deterministic_given_same_seed(self):
        # Direct proof of the claim backing the export/import design (see
        # docs/COMPOSITION.md and ScenarioConfig.imported_trace's docstring):
        # compose() is a pure function of its inputs, no hidden randomness.
        from backend.compose import compose
        a = compose(duration_s=20, seed=123)
        b = compose(duration_s=20, seed=123)
        pd.testing.assert_frame_equal(a.df, b.df)

    def test_full_trace_endpoint_is_undecimated(self):
        sid = self._create_and_wait()
        full = client.get(f"/api/scenarios/{sid}/trace?max_points=0").json()["trace"]
        scenario = api.SCENARIOS[sid]
        assert len(full) == len(scenario.composed.df)

    def test_imported_trace_reproduces_exact_composed_df(self):
        sid = self._create_and_wait()
        full_trace = client.get(f"/api/scenarios/{sid}/trace?max_points=0").json()["trace"]

        import_resp = client.post("/api/scenarios", json={
            "seed": 999999,  # deliberately different -- imported_trace must override, not blend
            "speed": 100,
            "imported_trace": full_trace,
        })
        assert import_resp.status_code == 200, import_resp.text
        imported_sid = import_resp.json()["id"]
        deadline = time.monotonic() + 5
        while client.get(f"/api/scenarios/{imported_sid}/status").json()["running"] and time.monotonic() < deadline:
            time.sleep(0.05)

        original_df = api.SCENARIOS[sid].composed.df
        replayed_df = api.SCENARIOS[imported_sid].composed.df
        pd.testing.assert_frame_equal(original_df, replayed_df, check_dtype=False)
