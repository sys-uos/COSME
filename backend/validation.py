"""Validate the composed trace against real, held-out measurements.

Evidence that the composed emulator is accurate, by comparing how a workload
behaves under a single impairment versus the composed impairment.

`zimmermann.fit()` already holds out ~20% of the 57 usable
`clipped_measurements` drives (see zimmermann.py); those drives were never
used to fit the empirical burst-length distribution. For each held-out
drive we run two checks:

  Check A (does the fitted duration distribution generalize?): compare the
  held-out drive's *own real* reconfiguration-burst durations (extracted by
  the same full-vs-filtered diff used for training, just on a drive the fit
  never saw) against the train-only fitted distribution, via a
  Kolmogorov-Smirnov test. This isolates "is the duration model accurate"
  from "did we get the 15s phase right" (we don't have recovered absolute
  handover phase for these drives at all -- see docs/COMPOSITION.md -- so
  we deliberately don't evaluate timing/phase here).

  Check B (does composing beat a single impairment?): build (1) an
  obstruction-only signal directly from the real, measured obstacle-only
  trace, and (2) a composed signal = that same real obstruction trace OR'd
  with a *synthetic* reconfiguration-loss schedule sampled from the
  train-only Zimmermann distribution (random phase -- honestly labeled as
  such). Compare both against the real, fully-measured loss trace (which
  includes the real reconfiguration losses) using: event count, Hamming
  distance at the compose.py 10ms grid, and DTW similarity on a 1s-binned
  loss-occurred-in-this-second signal (same coarsening the original
  Zimmermann analysis notebook used for DTW, to keep sequences tractable).
  If composed reliably beats obstruction-only on these real drives, that is
  direct evidence that modeling+combining the
  reconfiguration-loss dimension improves realism over a single-impairment
  model -- not just an assertion.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.stats import ks_2samp

from backend.compose import compose
from backend.models import garcia_fit as gfit
from backend.models import oblos
from backend.models.garcia import GarciaParams, _gaussian_mixture
from backend.models.reconfig_schedule import ReconfigSchedule
from backend.models.zimmermann import (
    RECONFIG_PERIOD_S,
    ZimmermannModel,
    extract_reconfig_bursts,
    load_fit,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLIPPED_DIR = os.path.join(REPO_ROOT, "models", "Zimmermann", "clipped_measurements")
RESULTS_PATH = os.path.join(REPO_ROOT, "traces", "validation", "results.json")
GRID_DT_S = 0.01
DTW_BIN_S = 1.0


def _full_trace_path(meas_dir: str) -> str | None:
    for name in ("ping_results_FHP_loss_trace.csv", "ping_results_FHP_root_clip_loss_trace.csv"):
        p = os.path.join(meas_dir, name)
        if os.path.exists(p):
            return p
    return None


def _bin_loss_occurred(bool_series: np.ndarray, times: np.ndarray, bin_s: float = DTW_BIN_S) -> np.ndarray:
    """1 if loss occurred at all within each bin_s-second bin, else 0 -- for tractable DTW."""
    n_bins = int(np.ceil(times[-1] / bin_s)) + 1 if len(times) else 0
    binned = np.zeros(n_bins, dtype=np.float64)
    bin_idx = (times // bin_s).astype(int)
    np.maximum.at(binned, bin_idx, bool_series.astype(np.float64))
    return binned


def _event_count(bool_series: np.ndarray) -> int:
    if not bool_series.any():
        return 0
    edges = np.diff(bool_series.astype(int))
    return int((edges == 1).sum()) + (1 if bool_series[0] else 0)


@dataclass
class DriveResult:
    drive: str
    duration_s: float
    real_event_count: int
    obstruction_only_event_count: int
    composed_event_count: int
    obstruction_only_hamming: float
    composed_hamming: float
    obstruction_only_dtw: float
    composed_dtw: float
    n_real_reconfig_bursts: int
    ks_statistic: float | None
    ks_pvalue: float | None


def validate_drive(drive: str, seed: int) -> DriveResult | None:
    meas_dir = os.path.join(CLIPPED_DIR, drive)
    full_path = _full_trace_path(meas_dir)
    if full_path is None:
        return None
    try:
        obstruction_trace = oblos.load_obstruction_trace(drive)
    except FileNotFoundError:
        return None

    real_full = pd.read_csv(full_path)
    if real_full.empty:
        return None
    duration_s = float(real_full["timestamp"].max()) + 5.0
    times = np.arange(0, duration_s, GRID_DT_S)

    real_loss = oblos.as_boolean_series(real_full, times)
    obstruction_only_loss = oblos.as_boolean_series(obstruction_trace, times)

    # Check A: this drive's own real reconfig burst durations vs. the
    # train-only fitted distribution.
    real_reconfig_bursts_df = extract_reconfig_bursts(meas_dir)
    ks_stat, ks_p = None, None
    if len(real_reconfig_bursts_df) >= 5:
        model = ZimmermannModel(seed=seed)
        fitted_samples = model._durations
        ks_stat, ks_p = ks_2samp(real_reconfig_bursts_df["lossTime"].to_numpy(), fitted_samples)
        ks_stat, ks_p = float(ks_stat), float(ks_p)

    # Check B: obstruction-only vs. composed (synthetic reconfig, real
    # obstruction), both vs. the real fully-measured trace.
    composed = compose(duration_s=duration_s, obstruction_trace=obstruction_trace, seed=seed)
    composed_loss = composed.df["loss"].to_numpy()
    composed_times = composed.df["timestamp"].to_numpy()

    # Re-grid obstruction/real onto compose.py's own grid for a fair, aligned comparison.
    real_loss_c = oblos.as_boolean_series(real_full, composed_times)
    obstruction_only_c = oblos.as_boolean_series(obstruction_trace, composed_times)

    real_binned = _bin_loss_occurred(real_loss_c, composed_times)
    obstruction_binned = _bin_loss_occurred(obstruction_only_c, composed_times)
    composed_binned = _bin_loss_occurred(composed_loss, composed_times)

    obstruction_dtw, _ = fastdtw(obstruction_binned, real_binned)
    composed_dtw, _ = fastdtw(composed_binned, real_binned)

    return DriveResult(
        drive=drive,
        duration_s=duration_s,
        real_event_count=_event_count(real_loss_c),
        obstruction_only_event_count=_event_count(obstruction_only_c),
        composed_event_count=_event_count(composed_loss),
        obstruction_only_hamming=float(np.mean(obstruction_only_c != real_loss_c)),
        composed_hamming=float(np.mean(composed_loss != real_loss_c)),
        obstruction_only_dtw=float(obstruction_dtw) / len(real_binned),
        composed_dtw=float(composed_dtw) / len(real_binned),
        n_real_reconfig_bursts=len(real_reconfig_bursts_df),
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
    )


@dataclass
class DelayDriveResult:
    drive: str
    n_normal_samples: int
    n_reconfig_samples: int
    ks_statistic_normal: float
    ks_pvalue_normal: float
    ks_statistic_reconfig: float | None
    ks_pvalue_reconfig: float | None
    real_normal_mean_ms: float
    model_normal_mean_ms: float
    real_reconfig_mean_ms: float | None
    model_reconfig_mean_ms: float | None


def validate_delay_drive(drive: str, params: GarciaParams, seed: int) -> DelayDriveResult | None:
    """Real per-packet OWD (this held-out drive's own RTT/2) vs. i.i.d. draws
    from the fitted normal/reconfig distributions -- same KS-test logic as
    Check A above, just for delay/jitter instead of loss-burst duration:
    "does the fitted delay model generalize to a drive it never saw?"

    Compares against i.i.d. draws from the fitted per-branch distributions
    directly (not a full GarciaModel.generate() trace) since the real data
    here is i.i.d. per-packet RTT samples, not a time-correlated trace.
    """
    meas_dir = os.path.join(gfit.CLIPPED_DIR, drive)
    real = gfit.load_rtt_samples(meas_dir)
    if real.empty:
        return None
    real_normal = real.loc[~real["is_reconfig"], "owd_ms"].to_numpy()
    real_reconfig = real.loc[real["is_reconfig"], "owd_ms"].to_numpy()
    if len(real_normal) < 10:
        return None

    rng = np.random.default_rng(seed)
    model_normal = params.delta_floor_ms + _gaussian_mixture(
        rng, len(real_normal), params.normal_modes_ms, params.normal_weights, params.normal_sigmas_ms
    )
    ks_stat_n, ks_p_n = ks_2samp(real_normal, model_normal)

    ks_stat_r = ks_p_r = model_reconfig_mean = real_reconfig_mean = None
    if len(real_reconfig) >= 10:
        model_reconfig = params.delta_floor_ms + rng.normal(
            params.reconfig_mean_ms, params.reconfig_sigma_ms, len(real_reconfig)
        )
        ks_stat_r, ks_p_r = ks_2samp(real_reconfig, model_reconfig)
        ks_stat_r, ks_p_r = float(ks_stat_r), float(ks_p_r)
        model_reconfig_mean = float(np.mean(model_reconfig))
        real_reconfig_mean = float(np.mean(real_reconfig))

    return DelayDriveResult(
        drive=drive,
        n_normal_samples=int(len(real_normal)),
        n_reconfig_samples=int(len(real_reconfig)),
        ks_statistic_normal=float(ks_stat_n),
        ks_pvalue_normal=float(ks_p_n),
        ks_statistic_reconfig=ks_stat_r,
        ks_pvalue_reconfig=ks_p_r,
        real_normal_mean_ms=float(np.mean(real_normal)),
        model_normal_mean_ms=float(np.mean(model_normal)),
        real_reconfig_mean_ms=real_reconfig_mean,
        model_reconfig_mean_ms=model_reconfig_mean,
    )


def run_delay_validation(seed: int = 2024) -> dict:
    garcia_params = GarciaParams.default()
    garcia_fitted = gfit.load_fit()
    results = [
        r for r in (validate_delay_drive(d, garcia_params, seed=seed) for d in garcia_fitted.holdout_dirs)
        if r is not None
    ]
    with_reconfig = [r for r in results if r.ks_pvalue_reconfig is not None]
    return {
        "source": garcia_params.source,
        "n_holdout_drives": len(garcia_fitted.holdout_dirs),
        "n_validated": len(results),
        "ks_normal_mean_pvalue": float(np.mean([r.ks_pvalue_normal for r in results])) if results else None,
        "ks_reconfig_mean_pvalue": float(np.mean([r.ks_pvalue_reconfig for r in with_reconfig])) if with_reconfig else None,
        "mean_abs_error_normal_ms": float(np.mean(
            [abs(r.real_normal_mean_ms - r.model_normal_mean_ms) for r in results]
        )) if results else None,
        "mean_abs_error_reconfig_ms": float(np.mean(
            [abs(r.real_reconfig_mean_ms - r.model_reconfig_mean_ms) for r in with_reconfig]
        )) if with_reconfig else None,
        "per_drive": [asdict(r) for r in results],
    }


def run_validation(seed: int = 2024) -> dict:
    fitted = load_fit()
    results = []
    for drive in fitted.holdout_dirs:
        r = validate_drive(drive, seed=seed)
        if r is not None:
            results.append(r)

    def agg(field):
        vals = [getattr(r, field) for r in results]
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals))}

    summary = {
        "n_holdout_drives": len(fitted.holdout_dirs),
        "n_validated": len(results),
        "obstruction_only_hamming": agg("obstruction_only_hamming"),
        "composed_hamming": agg("composed_hamming"),
        "obstruction_only_dtw_per_sample": agg("obstruction_only_dtw"),
        "composed_dtw_per_sample": agg("composed_dtw"),
        "event_count_ratio_obstruction_only": float(np.mean(
            [r.obstruction_only_event_count / max(r.real_event_count, 1) for r in results]
        )),
        "event_count_ratio_composed": float(np.mean(
            [r.composed_event_count / max(r.real_event_count, 1) for r in results]
        )),
        "ks_tests": [
            {"drive": r.drive, "statistic": r.ks_statistic, "pvalue": r.ks_pvalue, "n_bursts": r.n_real_reconfig_bursts}
            for r in results if r.ks_statistic is not None
        ],
        "per_drive": [asdict(r) for r in results],
        "delay": run_delay_validation(seed=seed),
    }
    return summary


def save_results(summary: dict, path: str = RESULTS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    summary = run_validation()
    save_results(summary)
    print(f"validated {summary['n_validated']}/{summary['n_holdout_drives']} holdout drives")
    print(f"Hamming distance (lower=better):  obstruction-only={summary['obstruction_only_hamming']['mean']:.4f}  "
          f"composed={summary['composed_hamming']['mean']:.4f}")
    print(f"DTW/sample (lower=better):        obstruction-only={summary['obstruction_only_dtw_per_sample']['mean']:.4f}  "
          f"composed={summary['composed_dtw_per_sample']['mean']:.4f}")
    print(f"Event-count ratio (1.0=perfect):  obstruction-only={summary['event_count_ratio_obstruction_only']:.3f}  "
          f"composed={summary['event_count_ratio_composed']:.3f}")
    ks = summary["ks_tests"]
    if ks:
        mean_p = np.mean([k["pvalue"] for k in ks])
        print(f"KS test (train-fitted duration dist vs each holdout drive's own real bursts): "
              f"mean p-value={mean_p:.4f} across {len(ks)} drives")

    delay = summary["delay"]
    print(f"\nGarcia delay/jitter ({delay['source']}):")
    print(f"  validated {delay['n_validated']}/{delay['n_holdout_drives']} holdout drives")
    if delay["ks_normal_mean_pvalue"] is not None:
        print(f"  KS test (fitted normal-state dist vs each drive's own real OWD): "
              f"mean p-value={delay['ks_normal_mean_pvalue']:.4f}, "
              f"mean |error|={delay['mean_abs_error_normal_ms']:.2f}ms")
    if delay["ks_reconfig_mean_pvalue"] is not None:
        print(f"  KS test (fitted reconfig dist vs each drive's own real OWD): "
              f"mean p-value={delay['ks_reconfig_mean_pvalue']:.4f}, "
              f"mean |error|={delay['mean_abs_error_reconfig_ms']:.2f}ms")
