"""Controlled three-arm application-impact campaign for the SIGCOMM'26 poster.

WHY THREE ARMS, AND WHY THIS REPLACES THE EARLIER TWO-ARM RUN
-------------------------------------------------------------
The first campaign compared COSME against an *unshaped* link: the baseline ran on a bare
`qdisc noqueue` (no delay, no bandwidth cap, no loss) while the COSME arm ran
`netem delay 16.8ms 6.2ms rate 25Mbit`. Every reported delta therefore mixed three causes -- the
bandwidth ceiling, the base latency, and the impairment models -- and only the last is COSME's
contribution. Bulk transfer had to be dropped from the poster entirely because its unshaped
baseline reached ~856 Mbit/s, i.e. the container's own capability rather than a link.

This campaign holds link provisioning constant and varies only the impairment:

  A  reference   329/30 Mbit/s, 12.0 ms constant delay, no jitter, no loss
  B  single      A + obstruction loss only (what an ObLoS-alone emulation predicts)
  C  composed    A's capacity + the full COSME chain: Garcia delay/jitter,
                 loss = obstruction OR reconfiguration

B is the "single impairment vs composed impairment" comparison
(see docs/VALIDATION.md). Note B->C changes two things at once (reconfiguration loss AND delay
variation), because a single-impairment ObLoS emulation genuinely has neither -- report it that way.

PARAMETERS, AND WHERE THEY COME FROM
------------------------------------
  capacity  329 / 30 Mbit/s  -- Lindenberg UDP ceiling measurements over two years
  delay     12.0 ms per direction (= 24.0 ms RTT) -- median of 4.9M real ping samples across 8
            drives in models/Zimmermann/clipped_measurements/*/ping_results_FHP.csv. The fitted
            Garcia model's own median one-way delay is 12.42 ms, agreeing within 0.4 ms.
            netem applies delay on BOTH endpoints (orchestrator emits a server+client pair per
            tick), so 12.0 ms per side gives a 24.0 ms RTT.

Arm A is a *reference link*, not "Starlink without weather": a real clear-sky Starlink link still
has jitter and residual loss. Label it accordingly wherever it is reported.

MEASUREMENT FIDELITY
--------------------
QoE is read from each showcase's END-OF-RUN, IN-CONTAINER summary, never from the per-second
samples POSTed to /api/showcase/app-stats. Those samples cross the emulated link and are dropped
exactly when it is worst: on a real 600s run the seconds that arrived averaged a 3.8% loss duty
cycle while the 34 gaps whose samples never made it averaged 36.4%, so averaging survivors
materially understates the impairment.

Usage:
    python scripts/poster_campaign.py            # full matrix (~6.5h real time)
    python scripts/poster_campaign.py --report   # aggregate what is on disk
    python scripts/poster_campaign.py --smoke    # 60s runs, one app, to check the arms
"""
from __future__ import annotations

import argparse
import json
import os
import math
import statistics
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.environ.get("COSME_API", "http://localhost:8731/api")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "traces", "poster_runs")

DRIVE = "measurement-2025-04-03_16-58-multicar-onlyping"
# 15 repeats per arm. n=3 was too thin to say anything about determinism; with 15 the standard
# deviation is worth reporting in its own right, and it is what tells a reader whether a difference
# between arms is a result or noise. The first three are unchanged so the runs already on disk stay
# valid and are reused rather than repeated.
SEEDS = [42, 1337, 2026, 7, 91, 404, 512, 808, 1234, 1618, 2718, 3141, 4096, 5150, 8191]
REPEATS = 15
DURATION_S = 600.0
FILE_TRANSFER_GB = 1.0
FILE_TRANSFER_CCS = ("cubic", "bbr")   # reno is also available; the paper names these two

DOWNLOAD_MBPS = 329.0      # Lindenberg two-year UDP ceiling
UPLOAD_MBPS = 30.0
CONST_DELAY_MS = 12.0      # per direction; 24.0 ms RTT, the measured median
UPDATE_INTERVAL_S = 0.1

ARMS = ("reference", "obstruction", "composed")

TIMED_APPS = {
    "remote_desktop": ("remote-desktop", {"congestion_control": "cubic"}),
    "video_conferencing": ("video-conferencing", {}),
    "voip": ("voip", {}),
    "surveillance": ("surveillance", {}),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- traces


def _compact(df):
    """Keep only rows that change the loss state, plus the endpoints.

    The playback engine derives loss events from transitions in the `loss` column
    (`orchestrator._loss_event_updates`) and delay/bandwidth from a groupby over a time grid.
    With delay and bandwidth held constant, every row carries the same value, so dropping rows
    that do not change `loss` reproduces exactly the same playback plan from ~1k rows instead of
    60k -- which keeps the imported_trace POST small enough to be practical.
    """
    import numpy as np
    loss = df["loss"].to_numpy().astype(bool)
    keep = np.zeros(len(df), dtype=bool)
    keep[0] = keep[-1] = True
    keep[1:] |= loss[1:] != loss[:-1]
    return df[keep]


def build_arm_trace(arm: str, seed: int, duration_s: float):
    """Rows for `imported_trace`. Arm C is not built here -- it uses the normal drive path."""
    import pandas as pd  # noqa: F401  (imported for the caller's benefit / type clarity)
    from backend.compose import compose
    from backend.models import oblos

    obstruction = oblos.load_obstruction_trace(DRIVE)
    composed = compose(duration_s=duration_s, obstruction_trace=obstruction,
                       nominal_download_mbps=DOWNLOAD_MBPS, nominal_upload_mbps=UPLOAD_MBPS,
                       seed=seed)
    df = composed.df.copy()
    df["delay_ms"] = CONST_DELAY_MS
    df["jitter_ms"] = 0.0
    df["download_mbps"] = DOWNLOAD_MBPS
    df["upload_mbps"] = UPLOAD_MBPS

    if arm == "reference":
        df["loss"] = False
        df["obstruction_loss"] = False
        df["reconfig_loss"] = False
        # constant everything: first and last row fully describe it
        df = df.iloc[[0, -1]]
    elif arm == "obstruction":
        # Single impairment: ObLoS alone. Obstruction loss does not depend on the seed.
        df["loss"] = df["obstruction_loss"].astype(bool)
        df["reconfig_loss"] = False
        df = _compact(df)
    else:
        raise ValueError(arm)

    df = df[["timestamp", "loss", "obstruction_loss", "reconfig_loss",
             "delay_ms", "jitter_ms", "download_mbps", "upload_mbps"]]
    return df.to_dict(orient="records")


# --------------------------------------------------------------------------- link state


def qdisc_state(container: str = "cosme-client") -> str:
    try:
        out = subprocess.run(["docker", "exec", container, "tc", "qdisc", "show", "dev", "eth0"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except Exception as e:  # pragma: no cover - diagnostic path
        return f"<unavailable: {e}>"


# --------------------------------------------------------------------------- API


def start_scenario(arm: str, seed: int, duration_s: float) -> dict:
    if arm == "composed":
        body = {"drive": DRIVE, "weather_mode": "dry",
                "nominal_download_mbps": DOWNLOAD_MBPS, "nominal_upload_mbps": UPLOAD_MBPS,
                "update_interval_s": UPDATE_INTERVAL_S, "speed": 1.0, "seed": seed}
    else:
        body = {"imported_trace": build_arm_trace(arm, seed, duration_s + 60),
                "update_interval_s": UPDATE_INTERVAL_S, "speed": 1.0, "seed": seed}
    r = requests.post(f"{BASE}/scenarios", json=body, timeout=300)
    r.raise_for_status()
    return r.json()


def stop_scenario(sid: str) -> None:
    try:
        requests.post(f"{BASE}/scenarios/{sid}/stop", timeout=60)
    except Exception as e:  # pragma: no cover
        log(f"  WARN: stop_scenario({sid}) failed: {e}")


def scenario_status(sid: str) -> dict:
    return requests.get(f"{BASE}/scenarios/{sid}/status", timeout=30).json()


def app_stats(app_id: str, n: int = 5000) -> list:
    r = requests.get(f"{BASE}/showcase/app-stats", params={"app": app_id, "n": n}, timeout=60)
    r.raise_for_status()
    return r.json()["stats"]


def run_showcase(endpoint: str, body: dict, budget_s: float) -> dict:
    r = requests.post(f"{BASE}/showcase/{endpoint}", json=body, timeout=60)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    deadline = time.monotonic() + budget_s
    job = {}
    while time.monotonic() < deadline:
        job = requests.get(f"{BASE}/showcase/jobs/{job_id}", timeout=30).json()
        if job.get("status") != "running":
            return job
        time.sleep(5)
    requests.post(f"{BASE}/showcase/jobs/{job_id}/stop", timeout=30)
    job["timed_out"] = True
    return job


# --------------------------------------------------------------------------- QoE extraction


def qoe_from_job(app_id: str, job: dict) -> dict:
    """Unbiased QoE from the showcase's own end-of-run, in-container summary.

    Deliberately NOT from GET /api/showcase/qoe/{app}, which averages the per-second samples that
    reached the backend -- those cross the shaped link and go missing exactly when loss is worst
    (see this module's docstring). Every field below is computed inside the container over the
    whole run.
    """
    from backend.showcases import qoe as qoe_mod

    if app_id == "voip":
        return qoe_mod.voip_qoe_from_totals(job.get("totals") or {})
    if app_id == "video_conferencing":
        return qoe_mod.media_qoe_from_totals(job.get("totals") or {})
    if app_id == "surveillance":
        # surveillance_probe.py's own end-of-run summary: freeze_count/total_freeze_s are computed
        # from raw frame arrival gaps (>0.2s) over the full run, not from per-second samples.
        return {k: job[k] for k in ("freeze_count", "total_freeze_s", "longest_freeze_s",
                                    "mean_bitrate_kbps", "frames_received") if k in job}
    if app_id == "remote_desktop":
        return {k: job[k] for k in ("keystroke_latency_ms_median", "keystroke_latency_ms_p95",
                                    "effective_fps_mean", "keystroke_timeout_pct",
                                    "keystroke_attempts", "n_keystrokes") if k in job}
    if app_id == "file_transfer":
        out = {k: job[k] for k in ("duration_s", "tcp_retransmits") if k in job}
        if job.get("throughput_bps"):
            out["throughput_mbps"] = job["throughput_bps"] / 1e6
        return out
    return {}


# --------------------------------------------------------------------------- one run


def out_path(name: str) -> str:
    return os.path.join(OUT_DIR, f"{name}.json")


def do_run(name: str, app_id: str, endpoint: str, body: dict, arm: str, seed: int,
           budget_s: float) -> None:
    if os.path.exists(out_path(name)):
        log(f"skip {name} (already on disk)")
        return

    log(f"=== {name} ===")
    sid = None
    duration = body.get("duration_s") or DURATION_S
    record: dict = {"name": name, "app": app_id, "arm": arm, "seed": seed,
                    "duration_s_requested": body.get("duration_s"),
                    "size_gb": body.get("size_gb"),
                    "cc": body.get("congestion_control"),
                    "params": {"download_mbps": DOWNLOAD_MBPS, "upload_mbps": UPLOAD_MBPS,
                               "const_delay_ms": CONST_DELAY_MS if arm != "composed" else None,
                               "drive": DRIVE, "seed": seed}}
    t0 = time.monotonic()
    try:
        scenario = start_scenario(arm, seed, duration)
        sid = scenario["id"]
        time.sleep(3)
        record["qdisc_before"] = qdisc_state()
        log(f"  scenario {sid} arm={arm} seed={seed}")
        log(f"  qdisc: {record['qdisc_before'][:120]}")

        job = run_showcase(endpoint, body, budget_s)
        record["real_elapsed_s"] = round(time.monotonic() - t0, 1)
        record["job"] = job
        record["qoe"] = qoe_from_job(app_id, job)
        record["app_stats_n_received"] = len(app_stats(app_id))
        record["qdisc_after"] = qdisc_state()
        record["scenario_status"] = scenario_status(sid)
        log(f"  status={job.get('status')} elapsed={record['real_elapsed_s']}s "
            f"qoe={json.dumps(record['qoe'])}")
        with open(out_path(name), "w") as f:
            json.dump(record, f, indent=2)
        log(f"  saved {os.path.relpath(out_path(name), REPO)}")
    finally:
        if sid:
            stop_scenario(sid)
        time.sleep(3)


# --------------------------------------------------------------------------- matrix


def build_matrix(duration_s: float, only: list[str] | None) -> list[tuple]:
    runs = []
    for app_id, (endpoint, extra) in TIMED_APPS.items():
        if only and app_id not in only:
            continue
        body = {"duration_s": duration_s, **extra}
        for arm in ARMS:
            # Arms A and B are deterministic (constant delay; the obstruction trace is not
            # seeded), so their repeats measure instrument noise. Arm C's seeds vary the
            # handover phase, so its spread is real trace variation. Same n either way.
            seeds = SEEDS[:REPEATS]
            for i, seed in enumerate(seeds, 1):
                runs.append((f"{app_id}_{arm}_{i}", app_id, endpoint, body, arm, seed,
                             duration_s + 240))
    if not only or "file_transfer" in only:
        # HTTP/TCP download; congestion control is a sender property, set on cosme-server. Both
        # flavours the demo paper names are run: reporting one unlabelled is under-specified.
        #
        # 1 GB, not 3 GB. At 3 GB the transfer runs ~85 s, long enough for both CCs to reach steady
        # state and look identical (302 vs 305 Mbit/s). At 1 GB the ramp is a real fraction of the
        # transfer and Cubic's slower one shows: measured 208 vs 290 Mbit/s on an unimpaired link.
        # That difference is a genuine property of the two algorithms, and averaging it away behind
        # a longer transfer hides it.
        for cc in FILE_TRANSFER_CCS:
            body = {"size_gb": FILE_TRANSFER_GB, "congestion_control": cc}
            for arm in ARMS:
                for i, seed in enumerate(SEEDS[:REPEATS], 1):
                    # Generous poll budget: the showcase itself now derives its timeout from the
                    # transfer size (file_transfer._TRANSFER_ASSUMED_MIN_MBPS), so this must not be
                    # the thing that truncates a slow-but-progressing transfer.
                    runs.append((f"file_transfer_{cc}_{arm}_{i}", "file_transfer", "file-transfer",
                                 body, arm, seed, 5400))
    return runs


def run_campaign(duration_s: float, only: list[str] | None) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = build_matrix(duration_s, only)
    todo = [r for r in runs if not os.path.exists(out_path(r[0]))]
    est = sum((duration_s + 45) if "file_transfer" not in r[0] else 160 for r in todo) / 60
    log(f"{len(runs)} runs, {len(todo)} remaining, ~{est:.0f} min estimated")
    for i, (name, app_id, endpoint, body, arm, seed, budget) in enumerate(runs, 1):
        log(f"--- run {i}/{len(runs)} ---")
        try:
            do_run(name, app_id, endpoint, body, arm, seed, budget)
        except Exception as e:
            log(f"  FAILED {name}: {e}")
    log("campaign complete")
    report()


# --------------------------------------------------------------------------- reporting


def _spread(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {}
    return {"n": len(vals), "mean": round(statistics.fmean(vals), 3),
            "min": round(min(vals), 3), "max": round(max(vals), 3),
            "sd": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0}


def _welch_p(a: dict, b: dict) -> float | None:
    """Two-sided Welch's t-test p-value between two summarised arms (unequal variance, unequal n).

    Computed from the summaries rather than the raw samples so it stays consistent with what is
    reported. Returns None when either arm has n<2 or both are exactly constant.
    """
    na, nb = a.get("n", 0), b.get("n", 0)
    if na < 2 or nb < 2:
        return None
    va, vb = a["sd"] ** 2 / na, b["sd"] ** 2 / nb
    denom = va + vb
    if denom <= 0:
        return 0.0 if a["mean"] != b["mean"] else 1.0
    t = abs(a["mean"] - b["mean"]) / math.sqrt(denom)
    df = denom ** 2 / ((va ** 2 / (na - 1)) + (vb ** 2 / (nb - 1))) if (va or vb) else 1.0
    try:
        from statistics import NormalDist
        # t with df>=~20 is close enough to normal for a reporting threshold; df is ~28 at n=15.
        return 2 * (1 - NormalDist().cdf(t))
    except Exception:
        return None


def _welch_separates(a: dict, b: dict, alpha: float = 0.01) -> bool:
    p = _welch_p(a, b)
    return p is not None and p < alpha


def report() -> None:
    if not os.path.isdir(OUT_DIR):
        log(f"no results at {OUT_DIR}")
        return
    runs = []
    for fn in sorted(os.listdir(OUT_DIR)):
        if fn.endswith(".json") and fn != "summary.json":
            with open(os.path.join(OUT_DIR, fn)) as f:
                r = json.load(f)
            if "arm" in r:          # skip records from the superseded two-arm campaign
                runs.append(r)
    if not runs:
        log("no three-arm results on disk yet")
        return

    by_app: dict[str, dict[str, list[dict]]] = {}
    for r in runs:
        # Only bulk transfer VARIES congestion control; remote_desktop merely records a fixed
        # cubic, so keying every app by cc split its 15 runs into two bogus groups (and the 3
        # earliest, recorded before the field existed, into a third).
        key = (f"{r['app']} ({r['cc']})"
               if r.get("cc") and r["app"] == "file_transfer" else r["app"])
        by_app.setdefault(key, {a: [] for a in ARMS})[r["arm"]].append(r)

    summary: dict = {}
    rejected: list[str] = []
    print("\n" + "=" * 96)
    print("THREE-ARM CAMPAIGN -- mean +- sd over n runs.  329/30 Mbit/s and 12.0 ms constant "
          "delay in every arm.")
    print("=" * 96)
    for app, arms in sorted(by_app.items()):
        print(f"\n## {app}")
        metrics: set[str] = set()
        for rs in arms.values():
            for r in rs:
                metrics |= set((r.get("qoe") or {}).keys())
        metrics -= {"source", "n_samples", "packets_lost", "packets_received", "n_keystrokes",
                    "keystroke_attempts", "frames_received", "tcp_retransmits",
                    "jitter_implausible_ms"}
        summary[app] = {}
        for metric in sorted(metrics):
            row = {}
            for arm in ARMS:
                vals = [(r.get("qoe") or {}).get(metric) for r in arms[arm]]
                vals = [v for v in vals if isinstance(v, (int, float))]
                # Runs already on disk had their QoE computed before qoe.MAX_PLAUSIBLE_JITTER_MS
                # existed, so apply the same bound here. A corrupt jitter estimate also corrupts
                # MOS and the R-factor, which are derived from it.
                if metric in ("jitter_ms", "mos", "r_factor"):
                    keep = []
                    for r in arms[arm]:
                        q = r.get("qoe") or {}
                        v = q.get(metric)
                        if not isinstance(v, (int, float)):
                            continue
                        j = q.get("jitter_ms")
                        if isinstance(j, (int, float)) and j > 500.0:
                            rejected.append(f"{app}/{arm}/{metric}")
                            continue
                        keep.append(v)
                    vals = keep
                row[arm] = _spread(vals)
            summary[app][metric] = row
            ref, obs, comp = row["reference"], row["obstruction"], row["composed"]
            f = lambda s: (f"{s['mean']:>9.2f}+-{s['sd']:<6.2f} n={s['n']}"
                           if s else " " * 21 + "--")
            # Separation is claimed only against BOTH other arms -- that is what the poster says
            # (composing changes the outcome, AND differs from a single-impairment model).
            # Welch's t-test rather than range overlap: at n=15 a single outlier in either arm
            # would veto a real effect, and the standard deviations are the interesting quantity
            # here anyway (how deterministic is the emulation).
            if comp and ref and obs:
                row["separated"] = all(_welch_separates(comp, other) for other in (ref, obs))
                row["p_vs_reference"] = _welch_p(comp, ref)
                row["p_vs_obstruction"] = _welch_p(comp, obs)
            print(f"  {metric:<30} ref{f(ref)}  obs{f(obs)}  cosme{f(comp)}"
                  f"{'  <- separated' if row.get('separated') else ''}")

    path = os.path.join(OUT_DIR, "summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {os.path.relpath(path, REPO)}")
    if rejected:
        print(f"\nrejected {len(rejected)} value(s) with a physically impossible jitter estimate "
              f"(>500 ms on a 12 ms link): {sorted(set(rejected))}")
    print("`separated` = composed differs from BOTH other arms, Welch's t-test p<0.01.")
    print("Values are mean +- sd over n runs; the sd is the determinism claim.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="60s runs, voip only, to check the arms")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    if args.report:
        report()
        return
    try:
        health = requests.get(f"{BASE}/system/health", timeout=15).json()
    except Exception as e:
        sys.exit(f"backend not reachable at {BASE}: {e}")
    if not health.get("docker_available"):
        sys.exit("Docker not detected by the backend")
    unhealthy = {k: v for k, v in health.get("containers", {}).items() if v != "healthy"}
    if unhealthy:
        sys.exit(f"unhealthy containers: {unhealthy}")
    if args.smoke:
        run_campaign(60.0, ["voip"])
    else:
        run_campaign(DURATION_S, args.only)


if __name__ == "__main__":
    main()
