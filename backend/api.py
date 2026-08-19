"""FastAPI backend serving routes/traces/metrics to the COSME dashboard.

Two QoE layers, kept honestly separate:
  - The scenario-level metrics (`/api/scenarios/{id}/metrics`) are derived
    analytically from the composed trace's own signals (loss/delay/jitter/
    bandwidth) -- an always-available estimate, labeled as such in each
    response's `qoe_note` field.
  - The per-application showcase QoE (`/api/showcase/qoe/{app}`) is measured
    from real application traffic: showcase job results plus the live
    per-second samples the in-container probes/bots POST to
    `/api/showcase/app-stats`. Both layers share one R-factor/MOS formula
    (backend/showcases/qoe.py) so they differ only in their inputs.

Start with `--host 0.0.0.0`: the in-container probes reach this backend via
the docker bridge gateway (10.42.0.1:8731), which a loopback bind can't serve.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from backend import geocoding
from backend.compose import ComposedTrace, compose
from backend.models import oblos, oblos_live, wetlinks
from backend.models.dwd_weather import WeatherUnavailable, weather_for_drive
from backend.netem_backend import DockerNetemBackend, DryRunNetemBackend
from backend.orchestrator import MAX_UPDATE_INTERVAL_S, PlaybackStats, build_playback_plan, play
from backend.showcases import _watchdog as watchdog
from backend.showcases import file_transfer as file_transfer_showcase
from backend.showcases import qoe as qoe_math
from backend.showcases import remote_desktop as remote_desktop_showcase
from backend.showcases import surveillance as surveillance_showcase
from backend.showcases import video_conferencing as video_conferencing_showcase

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

WeatherMode = Literal["real_dwd", "dry", "light", "moderate", "heavy"]
CongestionControl = Literal["cubic", "bbr", "reno"]

DOCKER_CONTAINERS = {"client": "cosme-client", "server": "cosme-server"}


def _docker_available() -> bool:
    """True if `docker` is installed and both cosme-* containers are running."""
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return False
        running = set(out.stdout.split())
        return set(DOCKER_CONTAINERS.values()).issubset(running)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: clears any netem shaping left on cosme-client/cosme-server by a PREVIOUS
    backend process.

    `SCENARIOS` is an in-memory dict wiped on every backend restart, but `tc qdisc` state lives
    in the CONTAINERS' kernel state and outlives the process that set it. Without this, a restart
    leaves the previous scenario's delay/loss/rate applied indefinitely with nothing in the UI to
    explain or stop it. Always start a fresh backend from a clean, unshaped link.
    """
    if "pytest" in sys.modules:
        # Never touch real containers as a side effect of collecting the test suite:
        # `TestClient(api.app)` fires this same startup event, and the suite may run against a
        # live stack, where resetting netem would disturb whatever is actually running.
        pass
    elif _docker_available():
        backend = DockerNetemBackend(containers=DOCKER_CONTAINERS)
        for endpoint in DOCKER_CONTAINERS:
            await asyncio.to_thread(backend.reset, endpoint)
    yield


app = FastAPI(title="COSME Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScenarioConfig(BaseModel):
    drive: Optional[str] = None
    # Alternative to `drive`: a trace simulated via POST /api/oblos/simulate
    # (its "trace" field, passed straight through) for an ad-hoc user-drawn
    # route instead of one of the real clipped_measurements drives.
    custom_obstruction_trace: Optional[list[dict]] = None
    # A previously-exported full composed trace (see GET /api/scenarios/{id}/trace's full=true
    # and the frontend's Export/Import), replayed byte-for-byte instead of recomposing from
    # drive/custom_obstruction_trace/weather -- takes precedence over both when set. This is what
    # makes an exported run's replay immune to any later model refit or cache change (compose()
    # is deterministic given the same inputs, but only as long as the fit caches it reads from
    # disk stay the same -- see docs/COMPOSITION.md and the Export button's own explanation).
    imported_trace: Optional[list[dict]] = None
    duration_s: Optional[float] = None  # defaults to the full drive/trace length
    nominal_download_mbps: float = 150.0
    nominal_upload_mbps: float = 15.0
    weather_mode: WeatherMode = "real_dwd"
    tcp_congestion_control: CongestionControl = "cubic"
    update_interval_s: float = MAX_UPDATE_INTERVAL_S  # clamped to <=1.0s in build_playback_plan
    speed: float = 1.0
    seed: int = 42


class Scenario:
    def __init__(self, config: ScenarioConfig):
        self.id = str(uuid.uuid4())
        self.config = config
        self.start_wall_time: float | None = None  # time.monotonic(), drives elapsed_s()
        self.start_time: float | None = None  # time.time(), lets app-stats samples (also
        # time.time()-stamped) be mapped onto this scenario's trace timeline for the
        # end-of-run summary chart: sim_t = (sample.t - start_time) * speed
        self.running = False
        self.error: str | None = None  # set if playback dies mid-run (see start()/_run())
        self._frozen_elapsed_s: float | None = None  # see stop()/elapsed_s()

        self.backend_mode = "docker" if _docker_available() else "dry_run"
        self.backend = (
            DockerNetemBackend(containers=DOCKER_CONTAINERS) if self.backend_mode == "docker"
            else DryRunNetemBackend(containers=DOCKER_CONTAINERS)
        )

        if config.imported_trace is not None:
            # Replay an exported run byte-for-byte: skip drive/custom_obstruction_trace/weather
            # resolution and compose() entirely -- delay/loss/bandwidth are already baked into
            # the imported rows, so re-deriving them from the (possibly since-changed) model fits
            # would defeat the whole point of an exported "1:1 replay" bundle.
            imported_df = pd.DataFrame(config.imported_trace)
            self.composed = ComposedTrace(df=imported_df)
            duration_s = config.duration_s or (
                float(imported_df["timestamp"].max()) if not imported_df.empty else 300.0
            )
        else:
            obstruction_trace = None
            rain_time_s = rain_mm_h = None
            duration_s = config.duration_s

            if config.custom_obstruction_trace is not None:
                obstruction_trace = pd.DataFrame(config.custom_obstruction_trace, columns=["timestamp", "lossTime"])
                if duration_s is None and not obstruction_trace.empty:
                    duration_s = float(obstruction_trace["timestamp"].max()) + 30.0
            elif config.drive:
                obstruction_trace = oblos.load_obstruction_trace(config.drive)
                if duration_s is None:
                    duration_s = float(obstruction_trace["timestamp"].max()) + 30.0
            duration_s = duration_s or 300.0

            if config.weather_mode == "real_dwd":
                if config.drive:
                    gps_path = os.path.join(oblos.CLIPPED_DIR, config.drive, "gps_location_results_root.csv")
                    if os.path.exists(gps_path):
                        try:
                            gps = pd.read_csv(gps_path)
                            drive_start = pd.to_datetime(gps["Timestamp"].iloc[0], utc=True, format="ISO8601")
                            weather = weather_for_drive(gps_path)
                            rain_time_s = (weather["timestamp"] - drive_start).dt.total_seconds().to_numpy()
                            rain_mm_h = weather["precipitation_mm"].to_numpy()
                        except WeatherUnavailable:
                            pass
                # else: no drive selected, real_dwd has nothing to key off of -> falls through to dry weather
            else:
                rain_time_s, rain_mm_h = wetlinks.constant_rain_series(duration_s, config.weather_mode)

            self.composed = compose(
                duration_s=duration_s,
                obstruction_trace=obstruction_trace,
                rain_time_s=rain_time_s,
                rain_mm_h=rain_mm_h,
                nominal_download_mbps=config.nominal_download_mbps,
                nominal_upload_mbps=config.nominal_upload_mbps,
                seed=config.seed,
            )
        self.duration_s = duration_s
        self.plan = build_playback_plan(self.composed, update_interval_s=config.update_interval_s)
        self.playback_stats = PlaybackStats()  # live lag/skip tracking -- see orchestrator.play()
        self._task: asyncio.Task | None = None

        if self.backend_mode == "docker" and config.tcp_congestion_control != "cubic":
            # cosme-server is the sender for a download-direction file transfer;
            # CC is a sender-socket property, so it's set there, once, up front.
            # Uses file_transfer.set_congestion_control (nsenter-based, see its
            # docstring) rather than a plain docker-exec sysctl write, which
            # always fails "permission denied" on modern Docker/runc.
            try:
                file_transfer_showcase.set_congestion_control(
                    config.tcp_congestion_control, DOCKER_CONTAINERS["server"]
                )
            except file_transfer_showcase.ShowcaseError:
                pass  # e.g. tcp_bbr module not loaded on the host, or nsenter sudo rule missing --
                      # surfaced via /api/system/congestion-controls

    def start(self) -> None:
        self.start_wall_time = time.monotonic()
        self.start_time = time.time()
        self.running = True
        self.error: str | None = None
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        # This runs as an un-awaited create_task, so an exception here would otherwise be
        # discarded: playback dies after one command with `running` stuck True and nothing
        # logged. `running`/`error` must be correct even when play() raises.
        try:
            await play(self.plan, self.backend, speed=self.config.speed, stats=self.playback_stats)
        except asyncio.CancelledError:
            raise  # expected from stop() -- not a failure, don't record it as one
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.running = False
            # Same freeze as stop() -- elapsed_s() is already clamped to duration_s once playback
            # finishes naturally, so this is a defensive-consistency measure rather than fixing an
            # observed symptom here specifically, but keeps both exit paths identical.
            if self._frozen_elapsed_s is None:
                self._frozen_elapsed_s = self.elapsed_s()

    def stop(self) -> None:
        """Cancel playback and clear both endpoints' netem state (back to clean pass-through).

        Distinct from a container restart (see reset_containers()): this only
        touches tc/netem shaping, not the containers themselves, so it's fast
        and doesn't disturb any showcase process that happens to be running
        in the same containers.
        """
        if self._task is not None:
            self._task.cancel()
        # `running` must end up False even if a reset() call itself fails (e.g. now that
        # DockerNetemBackend's subprocess calls have a real timeout instead of none at all --
        # see netem_backend.py -- a hung docker exec now raises instead of blocking forever, but
        # that exception must not leave this scenario stuck "running" from the dashboard's view).
        try:
            self.backend.reset("client")
            self.backend.reset("server")
        finally:
            self.running = False
            # elapsed_s() is wall-clock math independent of `running`, so without freezing it
            # here it keeps climbing after playback stops and /status keeps reporting a sliding
            # metrics window. Freezing means every later poll returns the value at the moment of
            # stopping, not wherever wall-clock time has since wandered to.
            self._frozen_elapsed_s = self.elapsed_s()

    def elapsed_s(self) -> float:
        if self._frozen_elapsed_s is not None:
            return self._frozen_elapsed_s
        if self.start_wall_time is None:
            return 0.0
        return min((time.monotonic() - self.start_wall_time) * self.config.speed, self.duration_s)


SCENARIOS: dict[str, Scenario] = {}


def _get_scenario(scenario_id: str) -> Scenario:
    scenario = SCENARIOS.get(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"unknown scenario {scenario_id}")
    return scenario


def _active_run() -> dict | None:
    """None if the shared cosme-client/cosme-server testbed is free, else a dict describing
    what's currently using it. Prefers a running scenario in the result (matches the intended
    "showcase running through an active scenario's shaped link" demo pattern -- both can be
    legitimately active at once, see `_require_free_testbed`), falling back to a running showcase.

    Used by GET /api/system/active-run (orphan-run banner) and by `_require_free_testbed` for the
    real conflict checks below.
    """
    for sid, scenario in SCENARIOS.items():
        if scenario.running:
            return {"kind": "scenario", "id": sid}
    for jid, job in SHOWCASE_JOBS.items():
        if job.get("status") == "running":
            return {"kind": "showcase", "id": jid, "app": job.get("app")}
    return None


def _require_free_testbed(kind: str) -> None:
    """Rejects a second concurrent SCENARIO (they both call `docker exec ... tc qdisc replace
    ...` against the same containers via real OS threads/`asyncio.to_thread` -- whichever call
    lands last wins, silently interleaving both composed traces with no error) or a second
    concurrent SHOWCASE of the same/different app (`_launch_showcase` unconditionally resets
    APP_STATS[app_id]/LIVE_FRAMES[app_id] on every launch, and file_transfer deletes its own
    destination file before starting -- a second concurrent showcase would stomp the first's
    in-flight state).

    Deliberately does NOT cross-block scenario vs. showcase: showcases never touch netem/tc
    (only DockerNetemBackend does, and only Scenario calls it), so a showcase's traffic flowing
    through a concurrently running scenario's shaping is the intended demo pattern, not a
    conflict. Blocking it would push users into running showcases with no scenario, i.e. over a
    completely unshaped link.
    """
    busy = _active_run()
    if busy is None or busy["kind"] != kind:
        return
    app_note = f" ({busy['app']})" if busy.get("app") else ""
    raise HTTPException(
        status_code=409,
        detail=f"cosme-client/cosme-server are busy running {busy['kind']} {busy['id']}"
               f"{app_note} -- only one {kind} can run at a time. Stop it first, or wait for it "
               f"to finish. (A scenario and a showcase MAY run at the same time -- that's the "
               f"intended way to see an app's traffic respond to a live-shaped link.)",
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/drives")
def list_drives():
    drives = oblos.list_available_drives()
    out = []
    for d in drives:
        gps_path = os.path.join(oblos.CLIPPED_DIR, d, "gps_location_results_root.csv")
        out.append({"drive": d, "has_gps": os.path.exists(gps_path)})
    return {"drives": out}


@app.get("/api/drives/{drive}/route")
def get_route(drive: str, max_points: int = 500):
    gps_path = os.path.join(oblos.CLIPPED_DIR, drive, "gps_location_results_root.csv")
    if not os.path.exists(gps_path):
        raise HTTPException(status_code=404, detail=f"no GPS trace for drive {drive}")
    gps = pd.read_csv(gps_path)
    gps["Timestamp"] = pd.to_datetime(gps["Timestamp"], utc=True, format="ISO8601")
    drive_start = gps["Timestamp"].iloc[0]
    gps["t_s"] = (gps["Timestamp"] - drive_start).dt.total_seconds()

    step = max(1, len(gps) // max_points)
    decimated = gps.iloc[::step]
    polyline = decimated[["Latitude", "Longitude"]].to_numpy().tolist()

    obstruction_trace = oblos.load_obstruction_trace(drive)
    markers = []
    for _, row in obstruction_trace.iterrows():
        idx = (gps["t_s"] - row["timestamp"]).abs().idxmin()
        markers.append({
            "lat": float(gps.loc[idx, "Latitude"]),
            "lon": float(gps.loc[idx, "Longitude"]),
            "t_s": float(row["timestamp"]),
            "loss_time_s": float(row["lossTime"]),
        })

    return {
        "drive": drive,
        "polyline": polyline,
        "obstruction_markers": markers,
        "duration_s": float(gps["t_s"].max()),
    }


@app.post("/api/scenarios")
async def create_scenario(config: ScenarioConfig):
    _require_free_testbed("scenario")
    # Scenario(config) is pure synchronous work (compose() + build_playback_plan(), a real
    # weather-API call if weather_mode="real_dwd" and uncached, an nsenter subprocess call for
    # non-cubic CC) -- for a long real drive (10ms grid, up to ~3 hours) this can take several
    # real seconds. Running it inline would block uvicorn's single-threaded event loop for that
    # whole time, freezing every OTHER concurrent request (health checks, other users' polling,
    # in-flight showcase progress) for the whole time. Scenario.__init__ has no `await`, so it
    # moves to a worker thread safely; scenario.start() must stay on the event-loop thread and is
    # called after.
    scenario = await asyncio.to_thread(Scenario, config)
    SCENARIOS[scenario.id] = scenario
    scenario.start()
    return {"id": scenario.id, "duration_s": scenario.duration_s, "n_updates": len(scenario.plan)}


@app.get("/api/scenarios/{scenario_id}/status")
def scenario_status(scenario_id: str):
    scenario = _get_scenario(scenario_id)
    # Real bug this guards against: with only the last 5 commands shown, a loss on/off pair
    # (rare relative to the steady stream of delay/rate grid updates, and applied for as little
    # as one tick's real duration) is very unlikely to still be in that tiny window whenever
    # someone happens to look -- looks exactly like "loss is never applied" even when it is.
    # Scanned from the end so this stays cheap even for a long-running scenario's full log.
    last_loss_cmd = next((c for c in reversed(scenario.backend.log) if "loss" in c), None)
    return {
        "id": scenario_id,
        "elapsed_s": scenario.elapsed_s(),
        "duration_s": scenario.duration_s,
        "running": scenario.running,
        "error": scenario.error,
        "backend_mode": scenario.backend_mode,
        "n_tc_commands_issued": len(scenario.backend.log),
        "recent_tc_commands": scenario.backend.log[-10:],
        "last_loss_command": last_loss_cmd,
        # See orchestrator.PlaybackStats -- elapsed_s above is pure wall-clock and says nothing
        # about whether playback has actually kept up; this does. playback_lag_s > a couple
        # seconds means what's really on the wire is stale relative to what the dashboard/trace
        # implies "now" should look like.
        "playback_lag_s": round(scenario.playback_stats.lag_s, 3),
        "playback_ticks_skipped": scenario.playback_stats.ticks_skipped,
        "playback_ticks_total": scenario.playback_stats.ticks_total,
        # For aligning app-stats samples (time.time()-stamped) onto this
        # scenario's own sim-time axis: sim_t = (sample.t - start_time) * speed.
        "start_time": scenario.start_time,
        "speed": scenario.config.speed,
    }


@app.post("/api/scenarios/{scenario_id}/stop")
async def stop_scenario(scenario_id: str):
    """Cancel playback and clear netem state -- the common "wrong scenario, bail out" case.

    Runs scenario.stop() (which shells out to `tc qdisc del ...` via
    NetemBackend.reset()) in a thread, matching orchestrator.play()'s own
    reasoning for why blocking subprocess calls can't run inline on
    uvicorn's event loop (see that module's docstring).
    """
    scenario = _get_scenario(scenario_id)
    await asyncio.to_thread(scenario.stop)
    return {"id": scenario_id, "running": scenario.running}


@app.get("/api/system/congestion-controls")
def congestion_controls():
    """Real available CC algorithms if Docker is up; a documented static fallback otherwise.

    BBR needs the `tcp_bbr` kernel module loaded on the HOST (`sudo modprobe
    tcp_bbr`) -- not something a container can do for itself, so its absence
    here is a host setup step, not a bug.
    """
    if _docker_available():
        try:
            out = subprocess.run(
                ["docker", "exec", DOCKER_CONTAINERS["server"], "sysctl", "-n",
                 "net.ipv4.tcp_available_congestion_control"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            return {"available": out.stdout.strip().split(), "source": "docker exec sysctl (live)"}
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    return {
        "available": ["cubic", "reno"],
        "source": "static fallback -- Docker not detected or sysctl check failed; "
                   "bbr may still be selectable if the host's tcp_bbr module is loaded.",
    }


@app.get("/api/system/health")
def system_health():
    """Per-container status for the dashboard's health badge -- independent of any specific
    scenario/showcase, so a problem (a crashed/unhealthy container) is visible before a user
    picks something and hits a confusing failure several clicks later.

    Reuses backend.showcases._watchdog.container_health_status, the same check every
    run_*_showcase() now does first (see backend/showcases/_watchdog.py).
    """
    if not _docker_available():
        return {"docker_available": False, "containers": {}}
    containers = {}
    # cosme-osrm is monitored here for visibility (real, confirmed-live failure mode: OOM-killed
    # under memory pressure, see docker-compose.yml's own comment on it) but deliberately NOT
    # included in RESETTABLE_CONTAINERS / POST /api/system/reset-containers -- restarting it means
    # reloading the whole-Germany dataset from scratch (a real, ~30s+ cost observed live), much
    # heavier than restarting the stateless client/server/probe shells that button targets.
    for name in RESETTABLE_CONTAINERS + ["cosme-osrm"]:
        containers[name] = watchdog.container_health_status(name)
    return {"docker_available": True, "containers": containers}


@app.get("/api/system/active-run")
def active_run():
    """What (if anything) currently holds the shared-testbed lock (see `_active_run()`),
    regardless of which browser tab/session started it.

    Real gap this closes: a scenario/showcase's id only ever lived in the JS variable of the
    tab that created it (`scenarioId`, `lastShowcaseRequest` -- nothing persisted, e.g. to
    localStorage). Reload the page, open a new tab, or come back later, and that tab has no way
    to know a run is still going -- it can't even show the Stop button, since that's driven by
    polling a specific known scenario id. Confirmed live: a real ~52-minute scenario a user
    started kept blocking every new POST /api/scenarios with 409 with no way to discover or stop
    it from a fresh page load. The frontend polls this endpoint on load (and periodically) to
    detect and offer to stop a run it didn't itself start.
    """
    busy = _active_run()
    if busy is None:
        return {"active": None}
    out = {"active": busy}
    if busy["kind"] == "scenario":
        scenario = SCENARIOS.get(busy["id"])
        if scenario is not None:
            out["active"]["elapsed_s"] = scenario.elapsed_s()
            out["active"]["duration_s"] = scenario.duration_s
    return out


class OblosSimulateRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    osrm_url: str = oblos_live.DEFAULT_OSRM_URL


SIMULATION_JOBS: dict[str, dict] = {}


async def _run_oblos_simulation(job_id: str, req: OblosSimulateRequest) -> None:
    try:
        result = await asyncio.to_thread(
            oblos_live.simulate_route, req.start_lat, req.start_lon, req.end_lat, req.end_lon,
            osrm_url=req.osrm_url,
        )
        SIMULATION_JOBS[job_id] = {
            "status": "done",
            "trace": result["trace"].to_dict(orient="records"),
            "polyline": result["polyline"],
        }
    except Exception as e:
        SIMULATION_JOBS[job_id] = {"status": "error", "error": str(e)}


@app.post("/api/oblos/simulate")
async def start_oblos_simulation(req: OblosSimulateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    SIMULATION_JOBS[job_id] = {"status": "running"}
    background_tasks.add_task(_run_oblos_simulation, job_id, req)
    return {"job_id": job_id}


@app.get("/api/oblos/simulate/{job_id}")
def oblos_simulation_status(job_id: str):
    job = SIMULATION_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown simulation job {job_id}")
    return job


@app.get("/api/geocode")
async def geocode_lookup(q: str):
    """Name/address -> candidate coordinates for the custom-route inputs.

    Germany-limited (matching the self-hosted OSRM coverage), disk-cached,
    Nominatim-policy rate-limited -- see backend/geocoding.py. Empty results
    mean "no match" (HTTP 200); 502 means the live lookup failed and there
    was no cache to fall back on.
    """
    try:
        return await asyncio.to_thread(geocoding.geocode, q)
    except geocoding.GeocodingUnavailable as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Application showcases: real traffic, real measured QoE (see backend/showcases/
# and docs/APPLICATIONS.md). Five apps -- two TCP (file transfer, remote
# desktop), three UDP (video conferencing, VoIP, surveillance) -- all launched
# through one job+poll pattern and one per-app live-stats store.
# ---------------------------------------------------------------------------

# Color thresholds for the QoE tiles/gauges: {good, warn, higher_is_better}. A value at or
# beyond `good` (in the "good" direction) renders green, between `good` and `warn` amber, past
# `warn` red. Grounded in real conventions already used in this codebase where one exists
# (backend/showcases/qoe.py's r_factor()/mos_from_r() implement the standard ITU-T G.107 scale --
# MOS>=4.0/R>=80 good, MOS>=3.6/R>=70 acceptable, matching the values below) or otherwise
# documented as reasonable networking defaults, NOT measured -- honestly labeled rather than
# implying a precision they don't have. `duration_s` (file transfer) deliberately has no
# thresholds: its "good" value depends entirely on the user-chosen transfer size, so a fixed
# number would be actively misleading rather than just imprecise.
_ITU_MOS = {"good": 4.0, "warn": 3.6, "higher_is_better": True}          # ITU-T G.107
_ITU_R_FACTOR = {"good": 80.0, "warn": 70.0, "higher_is_better": True}   # ITU-T G.107
_LOSS_PCT = {"good": 0.5, "warn": 3.0, "higher_is_better": False}        # reasonable default
_RTT_MS = {"good": 150.0, "warn": 300.0, "higher_is_better": False}      # reasonable default (~2x ITU one-way)
_JITTER_MS = {"good": 30.0, "warn": 80.0, "higher_is_better": False}     # reasonable default

# Single source of truth for what the dashboard renders per application:
# transport (drives the CC-selector gating in the UI), a `primary_metric`
# (the one headline gauge drawn per app), and the measured-QoE metric tiles
# (key names match what /api/showcase/qoe/{app} produces; `good`/`warn`/
# `higher_is_better` are omitted on metrics with no sensible fixed threshold).
SHOWCASE_APPS: list[dict] = [
    {
        "id": "file_transfer", "label": "File transfer (HTTP)", "transport": "tcp",
        "endpoint": "/showcase/file-transfer", "primary_metric": "throughput_mbps",
        "metrics": [
            {"key": "throughput_mbps", "label": "Throughput", "unit": "Mbit/s", "decimals": 1,
             "good": 50.0, "warn": 10.0, "higher_is_better": True},
            {"key": "duration_s", "label": "Transfer time", "unit": "s", "decimals": 1},
            {"key": "tcp_retransmits", "label": "TCP retransmits", "unit": "", "decimals": 0,
             "good": 5, "warn": 50, "higher_is_better": False},
        ],
    },
    {
        "id": "video_conferencing", "label": "Video conference (WebRTC)", "transport": "udp",
        "endpoint": "/showcase/video-conferencing", "primary_metric": "loss_pct",
        "metrics": [
            {"key": "video_bitrate_kbps", "label": "Video bitrate", "unit": "kbit/s", "decimals": 0,
             "good": 800.0, "warn": 300.0, "higher_is_better": True},
            {"key": "framerate", "label": "Framerate", "unit": "fps", "decimals": 1,
             "good": 24.0, "warn": 15.0, "higher_is_better": True},
            {"key": "loss_pct", "label": "Packet loss", "unit": "%", "decimals": 2, **_LOSS_PCT},
            {"key": "rtt_ms", "label": "RTT", "unit": "ms", "decimals": 1, **_RTT_MS},
            {"key": "jitter_ms", "label": "Jitter", "unit": "ms", "decimals": 1, **_JITTER_MS},
        ],
    },
    {
        "id": "voip", "label": "VoIP call (WebRTC, audio-only)", "transport": "udp",
        "endpoint": "/showcase/voip", "primary_metric": "mos",
        "metrics": [
            {"key": "mos", "label": "MOS (measured)", "unit": "", "decimals": 2, **_ITU_MOS},
            {"key": "r_factor", "label": "R-factor", "unit": "", "decimals": 1, **_ITU_R_FACTOR},
            {"key": "rtt_ms", "label": "RTT", "unit": "ms", "decimals": 1, **_RTT_MS},
            {"key": "jitter_ms", "label": "Jitter", "unit": "ms", "decimals": 1, **_JITTER_MS},
            {"key": "loss_pct", "label": "Audio loss", "unit": "%", "decimals": 2, **_LOSS_PCT},
        ],
    },
    {
        "id": "remote_desktop", "label": "Remote desktop (VNC)", "transport": "tcp",
        "endpoint": "/showcase/remote-desktop", "primary_metric": "keystroke_latency_ms_median",
        "metrics": [
            {"key": "keystroke_latency_ms_median", "label": "Keystroke RTT (median)", "unit": "ms", "decimals": 1,
             "good": 100.0, "warn": 300.0, "higher_is_better": False},
            {"key": "keystroke_latency_ms_p95", "label": "Keystroke RTT (p95)", "unit": "ms", "decimals": 1,
             "good": 200.0, "warn": 500.0, "higher_is_better": False},
            {"key": "effective_fps", "label": "Screen updates", "unit": "fps", "decimals": 1,
             "good": 15.0, "warn": 5.0, "higher_is_better": True},
            # Read this next to the two percentiles above: they only cover keystrokes whose echo
            # came back, and a keystroke times out exactly when interactivity is worst. Without
            # it a session where most keystrokes never echoed still shows a healthy median.
            {"key": "keystroke_timeout_pct", "label": "Keystrokes lost", "unit": "%", "decimals": 1,
             "good": 0.0, "warn": 5.0, "higher_is_better": False},
        ],
    },
    {
        "id": "surveillance", "label": "Surveillance stream (MPEG-TS)", "transport": "udp",
        "endpoint": "/showcase/surveillance", "primary_metric": "freeze_count",
        "metrics": [
            {"key": "freeze_count", "label": "Freeze events", "unit": "", "decimals": 0,
             "good": 0, "warn": 3, "higher_is_better": False},
            {"key": "total_freeze_s", "label": "Frozen time", "unit": "s", "decimals": 1,
             "good": 1.0, "warn": 5.0, "higher_is_better": False},
            {"key": "bitrate_kbps", "label": "Received bitrate", "unit": "kbit/s", "decimals": 0,
             "good": 500.0, "warn": 200.0, "higher_is_better": True},
            {"key": "fps", "label": "Decoded frames", "unit": "fps", "decimals": 1,
             "good": 20.0, "warn": 10.0, "higher_is_better": True},
        ],
    },
]
_APP_IDS = {a["id"] for a in SHOWCASE_APPS}

SHOWCASE_JOBS: dict[str, dict] = {}
MAX_APP_STATS = 500
APP_STATS: dict[str, list[dict]] = {app_id: [] for app_id in _APP_IDS}
# Latest live-view artifact per app: {"data": bytes, "content_type": str, "t": float}.
# Only the most recent frame is kept -- this is a live view, not a recording.
LIVE_FRAMES: dict[str, dict] = {app_id: None for app_id in _APP_IDS}
LIVE_FRAME_STALE_S = 5.0

# Containers eligible for a hard reset -- deliberately NEVER cosme-backend/cosme-frontend,
# which would kill the API serving this very request (and the dashboard polling it).
RESETTABLE_CONTAINERS = [DOCKER_CONTAINERS["client"], DOCKER_CONTAINERS["server"], "cosme-probe"]


@app.post("/api/system/reset-containers")
async def reset_containers():
    """Hard reset: `docker restart` the emulated endpoints (+ the VNC probe).

    For when something is actually stuck (a wedged showcase process, a qdisc
    that won't clear) -- the common "wrong scenario/showcase picked" case
    should use POST /api/scenarios/{id}/stop instead, which is faster and
    doesn't disturb anything else running in the containers. This clears ALL
    scenario/showcase state since a container restart invalidates it anyway
    (mirrors _launch_showcase's own per-run APP_STATS/LIVE_FRAMES reset above).
    """
    if not _docker_available():
        raise HTTPException(status_code=503, detail="Docker not detected -- nothing to reset.")
    for scenario in SCENARIOS.values():
        if scenario.running:
            await asyncio.to_thread(scenario.stop)
    SCENARIOS.clear()
    for app_id in _APP_IDS:
        APP_STATS[app_id] = []
        LIVE_FRAMES[app_id] = None

    def _restart():
        subprocess.run(
            ["docker", "restart"] + RESETTABLE_CONTAINERS,
            capture_output=True, text=True, timeout=60, check=True,
        )
    try:
        await asyncio.to_thread(_restart)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"container restart failed: {e.stderr}") from e
    except subprocess.TimeoutExpired as e:
        raise HTTPException(status_code=500, detail=f"container restart timed out: {e}") from e
    return {"status": "ok", "restarted": RESETTABLE_CONTAINERS}


@app.get("/api/showcase/apps")
def showcase_apps():
    return {"apps": SHOWCASE_APPS}


async def _run_showcase_job(job_id: str, fn, kwargs: dict) -> None:
    try:
        result = await asyncio.to_thread(fn, **kwargs)
        SHOWCASE_JOBS[job_id].update({"status": "done", **result.__dict__})
    except Exception as e:
        SHOWCASE_JOBS[job_id].update({"status": "error", "error": str(e)})


def _launch_showcase(app_id: str, background_tasks: BackgroundTasks, fn, *,
                     needs: str = "the real cosme-client/cosme-server containers", **kwargs) -> dict:
    if not _docker_available():
        raise HTTPException(status_code=503, detail=f"Docker not detected -- this showcase needs {needs}.")
    _require_free_testbed("showcase")
    job_id = str(uuid.uuid4())
    APP_STATS[app_id] = []  # a new run must never show the previous run's live samples
    LIVE_FRAMES[app_id] = None  # nor its last live-view frame
    SHOWCASE_JOBS[job_id] = {"status": "running", "app": app_id, "started_at": time.time()}
    background_tasks.add_task(_run_showcase_job, job_id, fn, kwargs)
    return {"job_id": job_id}


@app.get("/api/showcase/jobs/{job_id}")
def showcase_job_status(job_id: str):
    job = SHOWCASE_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown showcase job {job_id}")
    return job


@app.post("/api/showcase/jobs/{job_id}/stop")
async def stop_showcase_job(job_id: str):
    """Cancels an in-flight showcase run (see backend.showcases._watchdog.cancel_showcase for
    what actually gets killed, per app). Marks the job "error" immediately here rather than
    waiting for _run_showcase_job's background task to notice the killed process and update it
    itself -- both converge on the same "error" status, but doing it here means
    GET /api/system/active-run frees up right away for a caller polling it.
    """
    job = SHOWCASE_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown showcase job {job_id}")
    if job.get("status") != "running":
        return {"id": job_id, "status": job.get("status")}
    app_id = job.get("app")
    if app_id:
        await asyncio.to_thread(watchdog.cancel_showcase, app_id)
    job["status"] = "error"
    job["error"] = "stopped by user"
    return {"id": job_id, "status": "error"}


class FileTransferRequest(BaseModel):
    asset_name: str = "bigbuckbunny.mp4"
    # file_transfer has no "duration" (it's a fixed-size download, not a fixed-length
    # stream) -- size_gb drives an exact-size real-content file instead; None keeps
    # the legacy fixed-asset_name behavior (e.g. for CLI/standalone use).
    size_gb: float | None = None
    congestion_control: CongestionControl = "cubic"


@app.post("/api/showcase/file-transfer")
async def start_file_transfer(req: FileTransferRequest, background_tasks: BackgroundTasks):
    return _launch_showcase(
        "file_transfer", background_tasks, file_transfer_showcase.run_file_transfer_showcase,
        asset_name=req.asset_name, size_gb=req.size_gb, congestion_control=req.congestion_control,
    )


class VideoConferencingRequest(BaseModel):
    duration_s: float = 60.0
    video_asset: str = "bigbuckbunny.y4m"
    audio_asset: str = "bigbuckbunny.wav"


@app.post("/api/showcase/video-conferencing")
async def start_video_conferencing(req: VideoConferencingRequest, background_tasks: BackgroundTasks):
    return _launch_showcase(
        "video_conferencing", background_tasks,
        video_conferencing_showcase.run_video_conferencing_showcase,
        needs="the real cosme-client/cosme-server containers (aiortc media plane)",
        duration_s=req.duration_s, video_asset=req.video_asset, audio_asset=req.audio_asset,
    )


class VoipRequest(BaseModel):
    duration_s: float = 60.0
    audio_asset: str = "bigbuckbunny.wav"


@app.post("/api/showcase/voip")
async def start_voip(req: VoipRequest, background_tasks: BackgroundTasks):
    return _launch_showcase(
        "voip", background_tasks, video_conferencing_showcase.run_voip_showcase,
        needs="the real cosme-client/cosme-server containers (aiortc media plane)",
        duration_s=req.duration_s, audio_asset=req.audio_asset,
    )


class RemoteDesktopRequest(BaseModel):
    duration_s: float = 30.0
    congestion_control: CongestionControl = "cubic"


@app.post("/api/showcase/remote-desktop")
async def start_remote_desktop(req: RemoteDesktopRequest, background_tasks: BackgroundTasks):
    return _launch_showcase(
        "remote_desktop", background_tasks, remote_desktop_showcase.run_remote_desktop_showcase,
        needs="the cosme-probe container (VNC probe)",
        duration_s=req.duration_s, congestion_control=req.congestion_control,
    )


class SurveillanceRequest(BaseModel):
    duration_s: float = 30.0


@app.post("/api/showcase/surveillance")
async def start_surveillance(req: SurveillanceRequest, background_tasks: BackgroundTasks):
    return _launch_showcase(
        "surveillance", background_tasks, surveillance_showcase.run_surveillance_showcase,
        duration_s=req.duration_s,
    )


@app.post("/api/showcase/app-stats")
async def post_app_stats(stats: dict):
    """In-container probes/bots push real per-second measured samples here.

    Each sample must carry an "app" field naming one of SHOWCASE_APPS' ids.
    Reachable from inside the containers via the bridge gateway
    (http://10.42.0.1:8731) -- hence the --host 0.0.0.0 requirement.
    """
    app_id = stats.get("app")
    if app_id not in _APP_IDS:
        raise HTTPException(status_code=400, detail=f"unknown or missing app id {app_id!r}")
    APP_STATS[app_id].append(stats)
    del APP_STATS[app_id][:-MAX_APP_STATS]
    return {"status": "ok"}


@app.get("/api/showcase/app-stats")
def get_app_stats(app: str, n: int = 60):
    if app not in _APP_IDS:
        raise HTTPException(status_code=404, detail=f"unknown app {app!r}")
    return {"app": app, "stats": APP_STATS[app][-n:]}


@app.post("/api/showcase/live-frame")
async def post_live_frame(app: str, request: Request):
    """In-container probes/peers push a live-view snapshot here (JPEG/PNG).

    Raw image bytes as the request body (not JSON -- these are genuine
    decoded video frames / screen captures, see docker/endpoint-scripts/
    webrtc_peer.py, surveillance_probe.py, docker/probe/vnc_probe.py). Only
    the latest frame per app is kept; a new run replaces whatever the
    previous run left (see `_launch_showcase`'s LIVE_FRAMES reset).
    """
    if app not in _APP_IDS:
        raise HTTPException(status_code=400, detail=f"unknown app {app!r}")
    data = await request.body()
    content_type = request.headers.get("content-type", "application/octet-stream")
    LIVE_FRAMES[app] = {"data": data, "content_type": content_type, "t": time.time()}
    return {"status": "ok"}


@app.get("/api/showcase/live-frame/{app_id}")
def get_live_frame(app_id: str):
    """Latest live-view frame for `app_id`, or 204 if there is none or it's stale.

    Staleness (LIVE_FRAME_STALE_S) matters because a finished/crashed
    showcase must stop showing its last frame forever -- the dashboard
    should visibly go blank rather than lie about what's happening now.
    """
    if app_id not in _APP_IDS:
        raise HTTPException(status_code=404, detail=f"unknown app {app_id!r}")
    frame = LIVE_FRAMES.get(app_id)
    if frame is None or (time.time() - frame["t"]) > LIVE_FRAME_STALE_S:
        return Response(status_code=204)
    return Response(content=frame["data"], media_type=frame["content_type"])


_LIVE_QOE_AGGREGATORS = {
    "file_transfer": qoe_math.file_transfer_qoe_from_samples,
    "voip": qoe_math.voip_qoe_from_samples,
    "video_conferencing": qoe_math.media_qoe_from_samples,
    "surveillance": qoe_math.surveillance_qoe_from_samples,
    "remote_desktop": qoe_math.remote_desktop_qoe_from_samples,
}


def _latest_finished_job(app_id: str) -> dict | None:
    done = [j for j in SHOWCASE_JOBS.values() if j.get("app") == app_id and j.get("status") == "done"]
    return max(done, key=lambda j: j.get("started_at", 0)) if done else None


def _job_qoe_metrics(app_id: str, job: dict) -> dict:
    """Maps a finished job's result fields onto the registry's metric keys."""
    if app_id == "file_transfer":
        return {
            "throughput_mbps": job.get("throughput_bps", 0) / 1e6,
            "duration_s": job.get("duration_s"),
            "tcp_retransmits": job.get("tcp_retransmits"),
        }
    if app_id == "remote_desktop":
        out = {
            "keystroke_latency_ms_median": job.get("keystroke_latency_ms_median"),
            "keystroke_latency_ms_p95": job.get("keystroke_latency_ms_p95"),
            "effective_fps": job.get("effective_fps_mean"),
            # The percentiles above only cover keystrokes whose echo came back. A keystroke
            # times out exactly when interactivity is worst, so without this the dashboard shows
            # a responsive-looking median for a session that is actually unusable.
            "keystroke_timeout_pct": job.get("keystroke_timeout_pct"),
        }
        return {k: v for k, v in out.items() if v is not None}
    if app_id == "surveillance":
        duration = job.get("duration_s") or 1.0
        return {
            "freeze_count": job.get("freeze_count"),
            "total_freeze_s": job.get("total_freeze_s"),
            "bitrate_kbps": job.get("mean_bitrate_kbps"),
            "fps": (job.get("frames_received") or 0) / duration,
        }
    if app_id in ("voip", "video_conferencing"):
        # Scored from webrtc_peer.py's in-container cumulative counters, not from live samples.
        # Live samples cross the emulated link and go missing exactly when it is worst, so
        # averaging the survivors describes a better link than the one under test.
        totals = job.get("totals") or {}
        fn = qoe_math.voip_qoe_from_totals if app_id == "voip" else qoe_math.media_qoe_from_totals
        return {k: v for k, v in fn(totals).items() if k != "source"}
    return {}


@app.get("/api/showcase/qoe/{app_id}")
def showcase_qoe(app_id: str, n: int = 10):
    """Measured per-application QoE: finished-run results overlaid with live samples."""
    if app_id not in _APP_IDS:
        raise HTTPException(status_code=404, detail=f"unknown app {app_id!r}")

    metrics: dict = {}
    sources: list[str] = []

    # Which source wins depends on whether a run is in flight.
    #
    # Live: the per-second samples are the only thing describing this run, so they win; a previous
    # run's totals would be actively misleading.
    #
    # Finished: the job's in-container summary wins. Live samples cross the emulated link and are
    # dropped precisely when it is worst, so averaging the arrivals reports a better link than the
    # one under test.
    running = any(j.get("app") == app_id and j.get("status") == "running"
                  for j in SHOWCASE_JOBS.values())

    job = _latest_finished_job(app_id)
    job_metrics = _job_qoe_metrics(app_id, job) if job else {}
    live = _LIVE_QOE_AGGREGATORS.get(app_id, lambda s: {})(APP_STATS[app_id][-n:])

    if running:
        if job_metrics:
            metrics.update(job_metrics)
            sources.append("previous run")
        if live:
            metrics.update(live)
            sources.append(f"live samples (last {live.get('n_samples', 0)}, run in progress)")
    else:
        if live:
            metrics.update(live)
            sources.append(f"live samples (last {live.get('n_samples', 0)})")
        if job_metrics:
            metrics.update(job_metrics)
            sources.append("last completed run (in-container totals)")

    return {
        "app": app_id,
        "metrics": metrics,
        "source": "measured: " + " + ".join(sources) if sources else "no data yet",
    }


# Aggregation per column when bucketing a composed trace down to a point budget.
# NOT stride sampling (`iloc[::step]`): a reconfig burst is only ~0.1-0.7s wide, so at ~60
# samples/bucket almost none of them land on a kept sample. Boolean loss uses "any" (a bucket is
# lost if any sample in it was); delay/jitter use "max" so a brief bump survives decimation.
_TRACE_AGG = {
    "timestamp": "mean",
    "loss": "any",
    "obstruction_loss": "any",
    "reconfig_loss": "any",
    "delay_ms": "max",
    "jitter_ms": "max",
    "download_mbps": "min",
    "upload_mbps": "min",
}


def _decimate_trace(df: pd.DataFrame, max_points: int) -> list[dict]:
    n = len(df)
    if n <= max_points:
        return df.to_dict(orient="records")
    bucket = (np.arange(n) * max_points) // n
    agg = {col: how for col, how in _TRACE_AGG.items() if col in df.columns}
    grouped = df.groupby(bucket).agg(agg).reset_index(drop=True)
    return grouped.to_dict(orient="records")


@app.get("/api/scenarios/{scenario_id}/metrics")
def scenario_metrics(scenario_id: str, window_s: float = 30.0):
    scenario = _get_scenario(scenario_id)
    elapsed = scenario.elapsed_s()
    df = scenario.composed.df
    window = df[(df["timestamp"] <= elapsed) & (df["timestamp"] > elapsed - window_s)]
    current = df[df["timestamp"] <= elapsed].tail(1)

    current_row = current.iloc[0] if not current.empty else df.iloc[0]

    return {
        "elapsed_s": elapsed,
        "current": current_row.to_dict(),
        "window": _decimate_trace(window, max_points=300),
    }


@app.get("/api/scenarios/{scenario_id}/trace")
def scenario_trace(scenario_id: str, max_points: int = 2000):
    """`max_points<=0` returns the FULL, undecimated trace -- used by the dashboard's Export
    button, since _decimate_trace's aggregation is lossy (drops/averages rows) and unsuitable
    for an exact 1:1 replay (see ScenarioConfig.imported_trace)."""
    scenario = _get_scenario(scenario_id)
    df = scenario.composed.df
    if max_points <= 0:
        return {"trace": df.to_dict(orient="records")}
    return {"trace": _decimate_trace(df, max_points=max_points)}
