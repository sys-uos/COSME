"""COSME composition engine.

Compounds the four impairment models' time series into a single, unified
multivariate trace, per rules documented in ``docs/COMPOSITION.md``. This
module is where the composition method is pinned down: every combination
rule used here is enumerated and justified in that doc, not left implicit.

Summary of the rules (see docs/COMPOSITION.md for the full rationale):
  1. All series share one time grid (default dt=10ms).
  2. Obstruction-loss (ObLoS) and reconfiguration-loss (Zimmermann) are both
     binary "in-loss" signals combined via boolean OR.
  3. Garcia's reconfiguration flag (rho) and Zimmermann's loss bursts are
     driven by the *same* ReconfigSchedule instance -- a modeled dependency,
     not two independent coin flips.
  4. WetLinks' bandwidth factor is applied independently of the loss/delay
     layers (documented limitation: no rain<->obstruction/handover
     interaction is modeled).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.models import oblos
from backend.models.garcia import GarciaModel, GarciaTrace
from backend.models.reconfig_schedule import ReconfigSchedule
from backend.models.wetlinks import WetLinksModel
from backend.models.zimmermann import ZimmermannModel

DEFAULT_DT_S = 0.01  # 10ms common grid, per docs/COMPOSITION.md rule 1


@dataclass
class ComposedTrace:
    df: pd.DataFrame  # timestamp, loss, obstruction_loss, reconfig_loss, delay_ms, jitter_ms, download_mbps, upload_mbps
    # Optional: None when built directly from an imported/replayed trace rather than a fresh
    # compose() call (see backend/api.py's ScenarioConfig.imported_trace) -- nothing outside this
    # module reads .schedule/.garcia_trace off a ComposedTrace, only .df, so this widening is
    # non-breaking for every other caller.
    schedule: ReconfigSchedule | None = None
    garcia_trace: GarciaTrace | None = None

    def to_csv(self, path: str) -> None:
        self.df.to_csv(path, index=False)


def compose(
    duration_s: float,
    obstruction_trace: pd.DataFrame | None = None,
    rain_time_s: np.ndarray | None = None,
    rain_mm_h: np.ndarray | None = None,
    nominal_download_mbps: float = 150.0,
    nominal_upload_mbps: float = 15.0,
    dt_s: float = DEFAULT_DT_S,
    garcia_model: GarciaModel | None = None,
    zimmermann_model: ZimmermannModel | None = None,
    wetlinks_model: WetLinksModel | None = None,
    seed: int | None = None,
) -> ComposedTrace:
    """Compose ObLoS + Garcia + WetLinks + Zimmermann into one trace.

    `obstruction_trace` is a (timestamp, lossTime) DataFrame, e.g. from
    ``oblos.load_obstruction_trace()``; omit for a run with no obstruction
    impairment (e.g. isolating the effect of weather alone).
    `rain_time_s`/`rain_mm_h` describe a rain-rate time series (e.g. from
    ``dwd_weather``) to drive WetLinks; omit for a dry-weather run.
    """
    zimmermann_model = zimmermann_model or ZimmermannModel(seed=seed)
    garcia_model = garcia_model or GarciaModel(seed=seed)
    wetlinks_model = wetlinks_model or WetLinksModel(nominal_download_mbps, nominal_upload_mbps)

    # Rule 3: one shared schedule drives both Garcia's rho and Zimmermann's bursts.
    schedule = ReconfigSchedule(duration_s=duration_s, zimmermann_model=zimmermann_model, seed=seed)
    garcia_trace = garcia_model.generate(duration_s=duration_s, dt_s=dt_s, schedule=schedule)
    times = garcia_trace.time_s

    # Rule 2: obstruction-loss and reconfiguration-loss combined via boolean OR.
    if obstruction_trace is not None and not obstruction_trace.empty:
        obstruction_loss = oblos.as_boolean_series(obstruction_trace, times)
    else:
        obstruction_loss = np.zeros(len(times), dtype=bool)
    reconfig_loss = garcia_trace.reconfig_flag  # same schedule Garcia already used
    combined_loss = obstruction_loss | reconfig_loss

    # Rule 4: WetLinks bandwidth factor, independent dimension.
    if rain_time_s is not None and rain_mm_h is not None and len(rain_time_s) > 0:
        rain_at_grid = np.interp(times, rain_time_s, rain_mm_h, left=rain_mm_h[0], right=rain_mm_h[-1])
    else:
        rain_at_grid = np.zeros(len(times))
    bw = wetlinks_model.generate(times, rain_at_grid)

    df = pd.DataFrame({
        "timestamp": times,
        "loss": combined_loss,
        "obstruction_loss": obstruction_loss,
        "reconfig_loss": reconfig_loss,
        "delay_ms": garcia_trace.delay_ms,
        "jitter_ms": garcia_trace.jitter_smooth_ms,
        "rain_mm_h": rain_at_grid,
        "download_mbps": bw["download_mbps"].to_numpy(),
        "upload_mbps": bw["upload_mbps"].to_numpy(),
    })

    return ComposedTrace(df=df, schedule=schedule, garcia_trace=garcia_trace)


if __name__ == "__main__":
    trace = compose(duration_s=60, seed=42)
    print(trace.df.describe(include="all"))
    print(f"loss fraction: {trace.df['loss'].mean():.3%} "
          f"(obstruction {trace.df['obstruction_loss'].mean():.3%}, "
          f"reconfig {trace.df['reconfig_loss'].mean():.3%})")
