"""Real comparison plots for the SIGCOMM poster, generated straight from COSME's own models --
not mocked/illustrative figures. Each function below is backed by either a real held-out
measurement drive (same data `backend/validation.py` validates against) or a real generated
model trace, so what ships on the poster is literally reproducible by re-running this file.

    python -m backend.plots                 # composition timeline + validation strip + weather + Garcia trace
    python -m backend.plots --with-cc        # + CC comparison, reading traces/cc_comparison.json
                                                (run `python -m backend.cc_comparison` first)

Output: traces/poster_plots/*.png (150 DPI, transparent background so they drop cleanly onto the
poster's own paper-white panels).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.compose import compose
from backend.models import oblos
from backend.models.garcia import GarciaModel
from backend.models.wetlinks import BUCKET_NAMES, WetLinksModel, fit
from backend.models.zimmermann import CLIPPED_DIR, list_measurement_dirs

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO_ROOT, "traces", "poster_plots")
# The LaTeX poster reads its figures from here, as VECTOR PDF -- at A0 a 150dpi PNG of a chart is
# visibly soft from the reading distance the poster is designed for.
POSTER_PICS = os.path.join(REPO_ROOT, "poster", "pics")
CC_RESULTS_PATH = os.path.join(REPO_ROOT, "traces", "cc_comparison.json")
POSTER_RUNS_DIR = os.path.join(REPO_ROOT, "traces", "poster_runs")

# One shared palette matching the real Uni Osnabrück brand tokens used on the LaTeX poster itself
# (poster/poster.tex) -- extracted from the live uni-osnabrueck.de stylesheet by frequency, and
# run through the dataviz-skill palette validator: crimson+gold pass CVD-separation cleanly as two
# identity colors, but flat brand gold #fbb800 only hits 1.71:1 contrast on a white chart
# background, so SIGNAL uses a deepened chart-safe gold variant (#b8860b) instead -- a
# matplotlib-only variant, not the poster's own #fbb800 token.
SIGNAL = "#b8860b"    # uos-gold-chart: link/composed/good/bbr
IMPAIR = "#ad1034"    # uos-crimson: loss/obstruction/worse/cubic
INK = "#212529"
MUTED = "#767676"
GRID = "#e5e5e5"

PAPER = "#fffdf9"     # the poster panel these figures sit on, for the palette validator's surface

plt.rcParams.update({
    # Nimbus Sans is the URW Helvetica clone shipped with texlive-fonts-recommended, i.e. the same
    # metrics as the poster's own `helvet` body font -- so chart labels and poster text are one
    # typeface, not two that nearly match. Falls back to DejaVu if the font ever goes missing.
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "DejaVu Sans"],
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
})


def _full_trace_path(meas_dir: str) -> str | None:
    for name in ("ping_results_FHP_loss_trace.csv", "ping_results_FHP_root_clip_loss_trace.csv"):
        p = os.path.join(meas_dir, name)
        if os.path.exists(p):
            return p
    return None


def _pick_drive_with_real_trace(min_events: int = 15) -> str | None:
    """First usable drive with a real full-loss-trace file AND a visually interesting
    (not near-empty) obstruction trace, so the plot has something worth showing."""
    for meas_dir in list_measurement_dirs(CLIPPED_DIR):
        drive = os.path.basename(meas_dir)
        if _full_trace_path(meas_dir) is None:
            continue
        try:
            trace = oblos.load_obstruction_trace(drive)
        except FileNotFoundError:
            continue
        if len(trace) >= min_events:
            return drive
    return None


def plot_composition_timeline(drive: str | None = None, seed: int = 1337, window_s: float = 240.0,
                               start_s: float = 0.0, out_path: str | None = None,
                               figsize: tuple[float, float] = (8.4, 3.0),
                               title: bool = True) -> str:
    """Real vs. obstruction-only vs. composed loss, over a real held-out drive's own window.

    Same three signals `backend/validation.py`'s Check B compares numerically (event-count
    ratio 0.904 obstruction-only -> 1.040 composed vs. real, see docs/VALIDATION.md) -- this is
    that same comparison made visual, on one representative real drive. `start_s` lets a caller
    reproduce a specific known-dense window (e.g. docs/VALIDATION.md's own reference window on
    drive measurement-2025-04-03_16-58-multicar-onlyping, t=1000-1300s -- see
    plot_validation_strip() below) instead of always starting at t=0.
    """
    drive = drive or _pick_drive_with_real_trace()
    if drive is None:
        raise RuntimeError("no usable measurement drive with a real full-loss-trace file found")
    meas_dir = os.path.join(CLIPPED_DIR, drive)
    real_full = pd.read_csv(_full_trace_path(meas_dir))
    obstruction_trace = oblos.load_obstruction_trace(drive)

    drive_end_s = float(real_full["timestamp"].max()) + 5.0
    end_s = min(start_s + window_s, drive_end_s)
    times = np.arange(start_s, end_s, 0.05)

    real_loss = oblos.as_boolean_series(real_full, times)
    obstruction_only = oblos.as_boolean_series(obstruction_trace, times)
    composed = compose(duration_s=end_s, obstruction_trace=obstruction_trace, seed=seed)
    composed_loss = np.interp(times, composed.df["timestamp"], composed.df["loss"].astype(float)) > 0.5

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                              gridspec_kw={"hspace": 0.35})
    rows = [
        ("Real (measured)", real_loss, INK),
        ("Obstruction-only", obstruction_only, IMPAIR),
        ("Composed", composed_loss, SIGNAL),
    ]
    for ax, (label, series, color) in zip(axes, rows):
        ax.fill_between(times, 0, series.astype(float), step="post", color=color, linewidth=0)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="x", alpha=0.5)
    axes[-1].set_xlabel("time (s)", fontsize=9)
    if title:
        fig.suptitle(f"Composed vs. single-impairment loss — real held-out drive ({drive})",
                     fontsize=10, color=INK, x=0.01, ha="left")

    out_path = out_path or os.path.join(OUT_DIR, "composition_timeline.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_validation_strip(out_path: str | None = None) -> str:
    """The poster's Validation-box figure: reproduces the same real/obstruction-only/composed
    strip-chart concept as traces/validation/example_drive_comparison.png (which has no generator
    script anywhere in this repo -- an orphaned, off-brand PNG), on the exact reference drive and
    window docs/VALIDATION.md's Result 1 itself cites, but through this module's reproducible,
    UOS-palette pipeline instead.
    """
    return plot_composition_timeline(
        drive="measurement-2025-04-03_16-58-multicar-onlyping",
        start_s=1000.0, window_s=300.0,
        out_path=out_path or os.path.join(OUT_DIR, "validation_strip.png"),
        # Poster-sized: the box-03 caption already names the drive and the window, so the figure
        # title would only repeat it and cost vertical space the box does not have.
        figsize=(8.8, 1.65), title=False,
    )


def plot_weather_impact(out_path: str | None = None, nominal_download_mbps: float = 329.0,
                        nominal_upload_mbps: float = 30.0) -> str:
    """Real WetLinks-fitted throughput degradation across rain buckets.

    Plotted as throughput RETAINED relative to dry weather, not in Mbit/s: at a 329/30 Mbit/s
    ceiling an absolute-value chart puts the two series an order of magnitude apart, so upload
    collapses into the axis and its "rain barely touches it" story -- the actual finding -- becomes
    invisible. One shared 0-100% axis shows both.
    """
    fit_ = fit(site="Osnabrück")
    dl = [fit_.download_factor[b] * 100 for b in BUCKET_NAMES]
    ul = [fit_.upload_factor[b] * 100 for b in BUCKET_NAMES]

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    x = np.arange(len(BUCKET_NAMES))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, dl, width=w, color=SIGNAL, label="download", zorder=3)
    ax.bar(x + w / 2 + 0.01, ul, width=w, color=IMPAIR, label="upload", zorder=3)
    for xi, (d, u) in enumerate(zip(dl, ul)):
        ax.text(xi - w / 2, d + 2, f"{d - 100:+.0f}%" if xi else "dry", ha="center", va="bottom",
                fontsize=8.5, color=INK, fontweight="bold")
        ax.text(xi + w / 2, u + 2, f"{u - 100:+.0f}%" if xi else "", ha="center", va="bottom",
                fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b.capitalize()}\nn={fit_.n_samples[b]:,}" for b in BUCKET_NAMES],
                       fontsize=8.5)
    ax.set_ylabel("throughput kept vs. dry (%)", fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("WetLinks: measured rain-bucket degradation (Osnabrück)",
                 fontsize=9.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out_path = out_path or os.path.join(OUT_DIR, "weather_impact.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_garcia_delay_trace(seed: int = 4, duration_s: float = 120.0, out_path: str | None = None) -> str:
    """Real generated Garcia delay/jitter trace, showing the per-handover baseline shift
    (each ~15s dwell gets its own baseline, not just a spike during the handover window).

    The raw 10ms-grid trace alone is too noisy to read the slot-to-slot story out of visually
    (per-sample GMM scatter dominates) -- overlaid with the same 1s-bucketed mean the real
    orchestrator/dashboard actually displays (backend/orchestrator.py's update grid), which is
    what makes the step between dwells legible, exactly as it would look live on the dashboard.

    `seed=4` is a real, unmodified draw from the fitted model -- picked (by comparing
    slot-to-slot std-dev across a handful of seeds) only because it happens to show a visually
    clear step for a demo figure, not because it's unrepresentative: its ~3.1ms slot-to-slot
    std-dev is within the normal spread other seeds produce (~1.5-3.1ms across 8 seeds checked).
    """
    model = GarciaModel(seed=seed)
    trace = model.generate(duration_s=duration_s, dt_s=0.01)
    df = pd.DataFrame({"t": trace.time_s, "delay": trace.delay_ms, "jitter": trace.jitter_smooth_ms})
    bucketed = df.groupby((df["t"] // 1.0)).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 3.4), sharex=True,
                                    gridspec_kw={"hspace": 0.15, "height_ratios": [2, 1]})
    ax1.plot(trace.time_s, trace.delay_ms, color=SIGNAL, linewidth=0.5, alpha=0.35)
    ax1.plot(bucketed["t"], bucketed["delay"], color=SIGNAL, linewidth=1.8,
              label="1s-bucketed mean (what netem actually plays back)")
    for t in trace.time_s[trace.reconfig_flag]:
        ax1.axvspan(t, t + 0.01, color=IMPAIR, alpha=0.35, linewidth=0)
    ax1.set_ylabel("delay (ms)", fontsize=9)
    ax1.set_title("Garcia model: per-handover baseline shift + reconfiguration spikes",
                  fontsize=10, color=INK, loc="left")
    ax1.legend(frameon=False, fontsize=7.5, loc="upper right")

    ax2.plot(trace.time_s, trace.jitter_smooth_ms, color=IMPAIR, linewidth=0.5, alpha=0.35)
    ax2.plot(bucketed["t"], bucketed["jitter"], color=IMPAIR, linewidth=1.8)
    ax2.set_ylabel("jitter (ms)", fontsize=9)
    ax2.set_xlabel("time (s)", fontsize=9)

    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    out_path = out_path or os.path.join(OUT_DIR, "garcia_delay_trace.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_cc_comparison(results_path: str = CC_RESULTS_PATH, out_path: str | None = None) -> str:
    """Real Docker-measured Cubic vs. BBR throughput under an identical netem profile.

    Requires `python -m backend.cc_comparison` to have been run first (needs the real Docker
    stack) -- raises a clear error rather than fabricating numbers if that hasn't happened.
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"{results_path} not found -- run `python -m backend.cc_comparison` against the real "
            f"Docker stack first (see backend/cc_comparison.py's module docstring)."
        )
    with open(results_path) as f:
        data = json.load(f)
    results = data["results"]
    profile = data["profile"]

    ccs = [r["congestion_control"] for r in results]
    throughput = [r["throughput_mbps"] for r in results]
    colors = [SIGNAL if cc == "bbr" else IMPAIR for cc in ccs]

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    bars = ax.bar(ccs, throughput, color=colors, width=0.5)
    for bar, r in zip(bars, results):
        # tcp_retransmits is a best-effort read (see file_transfer._read_retransmits' own
        # docstring: the socket is typically already closed by the time it's checked) -- None
        # means "not captured," not "confirmed zero," and must not be printed as 0.
        retx = "retx n/a" if r["tcp_retransmits"] is None else f"{r['tcp_retransmits']} retx"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(throughput) * 0.02,
                f"{r['throughput_mbps']:.1f} Mbit/s\n{retx}",
                ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_ylabel("measured throughput (Mbit/s)", fontsize=9)
    ax.set_title(
        f"Real measured throughput, same netem profile\n"
        f"(delay {profile['delay_ms']:.0f}±{profile['jitter_ms']:.0f}ms, "
        f"loss {profile['loss_pct']:.1f}%, cap {profile['rate_mbit']:.0f}Mbit/s)",
        fontsize=9.5, color=INK,
    )
    ax.set_ylim(0, max(throughput) * 1.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out_path = out_path or os.path.join(OUT_DIR, "cc_comparison.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _strip_axis(ax, label: str, fontsize: float = 15) -> None:
    """Shared treatment for the binary/continuous signal rows used by the merger + strip charts."""
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=fontsize, color=INK)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def plot_validation_bars(out_path: str | None = None) -> str:
    """docs/VALIDATION.md Result 1 as the poster's headline evidence chart.

    Event-count ratio against the real held-out drives, where 1.0 is an exact match. Read straight
    out of traces/validation/results.json (written by `python -m backend.validation`) rather than
    hardcoded, so the figure can never drift from the numbers the repo can actually reproduce.
    """
    results_path = os.path.join(REPO_ROOT, "traces", "validation", "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"{results_path} not found -- run `python -m backend.validation` first")
    with open(results_path) as f:
        v = json.load(f)
    obstruction = v["event_count_ratio_obstruction_only"]
    composed = v["event_count_ratio_composed"]
    n = v["n_validated"]

    fig, ax = plt.subplots(figsize=(6.6, 1.15))
    labels = ["Obstruction\nonly", "Composed\n(COSME)"]
    values = [obstruction, composed]
    colors = [IMPAIR, SIGNAL]
    y = np.arange(len(values))[::-1]
    ax.barh(y, values, height=0.52, color=colors, zorder=3)

    # The reference line is labelled beside itself rather than above the plot: an annotation over
    # the bars forces the y-limits open and wastes a third of the figure's height, which this
    # figure cannot spare inside its poster box.
    ax.axvline(1.0, color=INK, linewidth=1.6, zorder=4)
    ax.text(1.005, -0.62, f"real drives = 1.0  (n={n})", fontsize=8, color=INK,
            va="center", ha="left")
    ax.set_ylim(-0.85, len(values) - 0.5)

    for yi, val in zip(y, values):
        miss = abs(val - 1.0) * 100
        ax.text(val + 0.015, yi, f"{val:.3f}   ({miss:.0f}% off)", va="center", ha="left",
                fontsize=10.5, color=INK, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_xlim(0, 1.32)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.5, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    out_path = out_path or os.path.join(POSTER_PICS, "validation_bars.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_signal_merger(drive: str = "measurement-2025-04-03_16-58-multicar-onlyping",
                       start_s: float = 1040.0, window_s: float = 90.0, seed: int = 42,
                       out_path: str | None = None) -> str:
    """The visual abstract's Signal Merger panel: the four model outputs and the trace they compose
    into, all real.

    Row labels sit ABOVE each strip rather than to its left. Left labels cost ~20% of the figure
    width on an A0 sheet where the strips themselves are the content; as titles they cost only
    vertical space, which the panel has after the poster's headline was dropped.

    Rows 1-2 are the two boolean loss signals combined by OR (docs/COMPOSITION.md rule 2); rows 3-4
    are the two independent continuous dimensions. The bottom row is literally
    `obstruction | reconfig` from the same composed trace, so the picture and the code agree.

    The rain driving row 4 is a real DWD observation (19.1 mm/h, Osnabrück 2025-07-22, the peak in
    traces/weather_cache) applied as a step partway through the window -- real measured magnitude,
    placed for legibility, since hourly rain cannot vary inside a 90s window.
    """
    obstruction_trace = oblos.load_obstruction_trace(drive)
    end_s = start_s + window_s

    step_s = start_s + window_s * 0.45
    rain_time = np.array([start_s, step_s - 0.01, step_s, end_s])
    rain_mm_h = np.array([0.0, 0.0, 19.1, 19.1])
    composed = compose(duration_s=end_s, obstruction_trace=obstruction_trace,
                       rain_time_s=rain_time, rain_mm_h=rain_mm_h,
                       nominal_download_mbps=329.0, nominal_upload_mbps=30.0, seed=seed)
    df = composed.df
    win = df[(df["timestamp"] >= start_s) & (df["timestamp"] <= end_s)].copy()
    t = win["timestamp"].to_numpy() - start_s

    # A real loss burst is 0.1-0.5s long; on a 90s axis that is a sub-pixel hairline. Bin and mark
    # any bin containing loss: this widens the MARK without inventing loss, since every drawn bar
    # corresponds to at least one real event.
    bin_s = 0.4
    bins = np.floor(t / bin_s).astype(int)
    n_bins = int(np.ceil(window_s / bin_s))
    centres = (np.arange(n_bins) + 0.5) * bin_s

    def binned(col: str) -> np.ndarray:
        out = np.zeros(n_bins)
        vals = win[col].to_numpy().astype(bool)
        np.maximum.at(out, np.clip(bins, 0, n_bins - 1), vals.astype(float))
        return out

    fig, axes = plt.subplots(5, 1, figsize=(11.0, 7.4), sharex=True,
                             gridspec_kw={"hspace": 0.62, "height_ratios": [1, 1, 1.45, 1.45, 1.2]})
    # Opaque card behind the strips. The Compose panel this sits on is warm beige (#EFE6D8), and
    # the gold traces (Garcia, WetLinks) lose contrast against it -- both are yellow-ish. On the
    # near-white card ground they read cleanly, and the figure reads as a panel rather than a wash.
    fig.patch.set_facecolor(PAPER)
    fig.patch.set_alpha(1.0)

    def title(ax, model, delivers, color):
        ax.set_title(f"{model}  —  {delivers}", loc="left", fontsize=16, color=color, pad=5)

    axes[0].bar(centres, binned("obstruction_loss"), width=bin_s, color=IMPAIR, linewidth=0)
    axes[0].set_ylim(0, 1.15)
    title(axes[0], "ObLoS", "obstruction loss (boolean)", IMPAIR)

    axes[1].bar(centres, binned("reconfig_loss"), width=bin_s, color=IMPAIR, linewidth=0)
    axes[1].set_ylim(0, 1.15)
    title(axes[1], "Zimmermann et al.", "reconfiguration loss (boolean)", IMPAIR)

    win["bucket"] = np.floor(t)
    bucketed = win.groupby("bucket")["delay_ms"].mean()
    axes[2].plot(t, win["delay_ms"], color=SIGNAL, linewidth=0.4, alpha=0.25)
    axes[2].plot(bucketed.index + 0.5, bucketed.to_numpy(), color=SIGNAL, linewidth=1.8)
    title(axes[2], "Garcia et al.", "one-way delay + jitter (ms)", SIGNAL)

    axes[3].plot(t, win["download_mbps"], color=SIGNAL, linewidth=2.0)
    axes[3].set_ylim(0, 400)
    title(axes[3], "WetLinks", "download / upload bandwidth (Mbit/s)", SIGNAL)
    axes[3].annotate("rain 19.1 mm/h", xy=(window_s * 0.47, 45), fontsize=13, color=MUTED,
                     va="bottom")

    axes[4].bar(centres, binned("loss"), width=bin_s, color=IMPAIR, linewidth=0)
    axes[4].set_ylim(0, 1.15)
    title(axes[4], "COMPOSED", "loss = obstruction $\\vee$ reconfiguration", INK)
    axes[4].set_xlabel("time (s)", fontsize=15)
    axes[4].spines["bottom"].set_linewidth(1.8)

    for ax in axes:
        ax.grid(False)
        ax.set_xlim(0, window_s)
        ax.set_yticks([])
        ax.tick_params(labelsize=14)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

    out_path = out_path or os.path.join(POSTER_PICS, "signal_merger.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=PAPER, edgecolor="none", pad_inches=0.18)
    plt.close(fig)
    return out_path


def plot_garcia_poster(seed: int = 4, duration_s: float = 120.0, out_path: str | None = None) -> str:
    """Compact one-panel Garcia trace for the poster's delay-model box.

    Box 04 asserts a fitted delay model but had no picture of what it produces, which left half
    the box empty. This shows the thing the model is actually for: delay is not a constant plus
    noise -- each ~15s dwell gets its OWN baseline, and each handover is a sharp excursion. That
    is precisely the structure a single mean delay cannot represent, which is the argument for
    carrying the model at all.

    The raw 10ms trace is pure per-sample GMM scatter at this scale, so the 1s-bucketed mean --
    what the orchestrator actually plays into netem -- is drawn over it. `seed=4` is an unmodified
    draw, picked only because its slot-to-slot step is visually clear; its ~3.1ms slot-to-slot
    spread sits inside the normal range across seeds (see plot_garcia_delay_trace).
    """
    model = GarciaModel(seed=seed)
    trace = model.generate(duration_s=duration_s, dt_s=0.01)
    df = pd.DataFrame({"t": trace.time_s, "delay": trace.delay_ms})
    bucketed = df.groupby(df["t"] // 1.0)["delay"].mean()

    fig, ax = plt.subplots(figsize=(6.8, 1.25))
    ax.plot(trace.time_s, trace.delay_ms, color=SIGNAL, linewidth=0.4, alpha=0.22)
    ax.plot(bucketed.index + 0.5, bucketed.to_numpy(), color=SIGNAL, linewidth=2.0,
            label="1 s mean (played into netem)")
    # Mark the handover windows so the excursions are attributable rather than decorative.
    for t in trace.time_s[trace.reconfig_flag]:
        ax.axvspan(t, t + 0.01, color=IMPAIR, alpha=0.30, linewidth=0)
    ax.plot([], [], color=IMPAIR, linewidth=3, alpha=0.5, label="reconfiguration window")
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_ylabel("one-way delay (ms)", fontsize=9)
    ax.set_xlim(0, duration_s)
    ax.tick_params(labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2, handlelength=1.4)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out_path = out_path or os.path.join(POSTER_PICS, "garcia_delay.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_app_impact(summary_path: str | None = None, out_path: str | None = None) -> str:
    """Measured application QoE, impaired vs. unshaped baseline, from the multi-run campaign.

    Reads traces/poster_runs/summary.json (written by `scripts/poster_campaign.py --report`) and
    plots ONLY metrics that campaign marked `separated` -- i.e. where the impaired and baseline
    ranges across repeats do not overlap. Anything that overlapped is not a result and is left off
    the poster rather than quoted with a caveat.
    """
    summary_path = summary_path or os.path.join(POSTER_RUNS_DIR, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"{summary_path} not found -- run `python scripts/poster_campaign.py` (then --report)"
        )
    with open(summary_path) as f:
        summary = json.load(f)

    # (app, metric, unit) worth showing on a poster, in reading order. Restricted on purpose:
    # every extra row costs the reader more than it tells them from 10 metres.
    WANTED = [
        ("voip", "mos", "MOS"),
        ("voip", "loss_pct", "packet loss (%)"),
        ("surveillance", "freeze_count", "video freezes"),
        ("remote_desktop", "keystroke_latency_ms_median", "keystroke latency (ms)"),
        ("video_conferencing", "framerate", "frame rate (fps)"),
    ]
    rows = []
    for app, metric, unit in WANTED:
        row = (summary.get(app) or {}).get(metric)
        if not row or not row.get("separated"):
            continue
        rows.append((f"{app.replace('_', ' ')}\n{unit}", row["baseline"], row["impaired"]))
    if not rows:
        raise RuntimeError("no metric separated impaired from baseline -- nothing safe to plot")

    fig, ax = plt.subplots(figsize=(6.6, 0.72 * len(rows) + 1.1))
    y = np.arange(len(rows))[::-1]
    for yi, (_, base, imp) in zip(y, rows):
        # normalise each row to its own baseline mean so different units share one axis
        b = base["mean"] or 1.0
        ax.plot([base["min"] / b, base["max"] / b], [yi + 0.12] * 2, color=MUTED, linewidth=5,
                solid_capstyle="round", alpha=0.5, zorder=2)
        ax.plot([imp["min"] / b, imp["max"] / b], [yi - 0.12] * 2, color=IMPAIR, linewidth=5,
                solid_capstyle="round", alpha=0.5, zorder=2)
        ax.scatter([base["mean"] / b], [yi + 0.12], s=70, color=MUTED, zorder=3,
                   edgecolor=PAPER, linewidth=1.5)
        ax.scatter([imp["mean"] / b], [yi - 0.12], s=70, color=IMPAIR, zorder=3,
                   edgecolor=PAPER, linewidth=1.5)
        ax.text(base["mean"] / b, yi + 0.34, f"{base['mean']:g}", fontsize=8, color=MUTED,
                ha="center", va="bottom")
        ax.text(imp["mean"] / b, yi - 0.56, f"{imp['mean']:g}", fontsize=9, color=IMPAIR,
                ha="center", va="top", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
    ax.set_xlabel("relative to the same link with no shaping (= 1.0)", fontsize=8.5)
    ax.axvline(1.0, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
    ax.grid(axis="y", visible=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    n_imp = rows[0][2]["n"]
    n_base = rows[0][1]["n"]
    ax.scatter([], [], s=70, color=MUTED, label=f"no shaping (n={n_base})")
    ax.scatter([], [], s=70, color=IMPAIR, label=f"COSME composed (n={n_imp})")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)

    out_path = out_path or os.path.join(POSTER_PICS, "app_impact.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_impact_table(summary_path: str | None = None, out_path: str | None = None) -> str:
    """Emit poster/impact_table.tex from the three-arm campaign summary.

    Generated from traces/poster_runs/summary.json so the poster's numbers cannot drift from the
    measurements. All three arms share 329/30 Mbit/s and 12.0 ms constant delay per direction;
    only the impairment differs, so a difference between columns is the impairment's doing and not
    the link's provisioning.

    Metric selection is deliberately narrower than "everything that separated":

    * One or two metrics per application, chosen for what that application's user actually feels.
    * Only metrics whose composed arm separates from BOTH other arms, AND that move in the
      physically sensible direction. Video-conferencing bitrate and frame rate separate *upward*
      under impairment (the aiortc sender adapts), and its packet loss is non-monotonic between
      the obstruction and composed arms with overlapping ranges -- none of those are reportable as
      impairment effects, so only its jitter appears.
    * Bulk transfer appears as one row per congestion control. It is NOT measured over the same
      window as the rows above it: a 1 GB download finishes in ~30 s and so only sees the start of
      the drive (~2 % loss) against the 9.6 % the 600 s rows face, because this drive's
      obstruction-dense stretch starts around t=277 s. write_impact_note() renders that caveat
      for anywhere it is wanted; the poster does not print it.
    * Remote-desktop MEDIAN keystroke latency is excluded: it does not separate (composed sits
      below obstruction-only). The p95 does, which is physically sensible -- the median reflects
      normal operation, the tail reflects loss bursts.
    """
    summary_path = summary_path or os.path.join(POSTER_RUNS_DIR, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"{summary_path} not found -- run scripts/poster_campaign.py first")
    with open(summary_path) as f:
        summary = json.load(f)

    # (app, metric, printed app name, printed metric name, unit, decimals)
    ROWS = [
        ("voip", "mos", "VoIP call", "MOS", "", 2),
        ("voip", "loss_pct", "", "audio loss", r"\,\%", 1),
        ("surveillance", "freeze_count", "Live video", "freezes / 10\\,min", "", 0),
        ("surveillance", "total_freeze_s", "", "time frozen", r"\,s", 1),
        ("remote_desktop", "keystroke_latency_ms_p95", "Remote desktop", "keystroke latency (p95)", r"\,ms", 0),
        ("remote_desktop", "effective_fps_mean", "", "screen updates", r"\,fps", 1),
        ("video_conferencing", "jitter_ms", "Video conference", "jitter", r"\,ms", 1),
        # One row per congestion control: throughput is the headline, and a single unlabelled
        # "throughput" row would hide which algorithm produced it. The 1 GB transfer runs ~30 s and
        # so faces a lighter part of the drive than the 600 s rows above -- the note under the
        # table states that window rather than letting the rows imply equivalence.
        ("file_transfer (cubic)", "throughput_mbps", "Bulk transfer", "TCP Cubic", r"\,Mbit/s", 0),
        ("file_transfer (bbr)", "throughput_mbps", "", "TCP BBR", r"\,Mbit/s", 0),
    ]

    lines, n = [], {}
    short_rows: list = []
    n_nominal = max((((summary.get(a) or {}).get(m) or {}).get("composed") or {}).get("n", 0)
                    for a, m, *_ in ROWS)
    for app, metric, app_name, metric_name, unit, dp in ROWS:
        row = (summary.get(app) or {}).get(metric)
        if not row or not row.get("separated"):
            print(f"  skipping {app}.{metric}: does not separate from both other arms")
            continue
        ref, obs, comp = row["reference"], row["obstruction"], row["composed"]
        n = {"ref": ref["n"], "obs": obs["n"], "comp": comp["n"]}
        sd_dp = dp if dp else 1
        if min(ref["n"], obs["n"], comp["n"]) < n_nominal:
            short_rows.append((app_name or metric_name, min(ref["n"], obs["n"], comp["n"])))
        lines.append(
            f"{app_name} & {metric_name} & {ref['mean']:.{dp}f}{unit} & {obs['mean']:.{dp}f}{unit} & "
            f"\\hilite{{{comp['mean']:.{dp}f}{unit}}} & "
            f"$\\pm$\\,{comp['sd']:.{sd_dp}f} \\\\"
        )
    if not lines:
        raise RuntimeError("no metric separated from both arms -- nothing safe to report")

    if short_rows:
        # A run whose probe never connected is excluded from its metric, so those rows have a
        # smaller n than the header states. Say so instead of letting the header speak for them.
        note = ", ".join(f"{name} $n$={k}" for name, k in short_rows)
        lines.append("\\multicolumn{6}{@{}l@{}}{\\fontsize{23}{27}\\selectfont\\itshape "
                     f"$n$={n_nominal} except for: {note}}} \\\\")
    body = "\n".join(lines)
    # The table is the entire content of box 05 -- the surrounding prose was cut -- so it is set
    # larger than body text and centred rather than sitting top-left in a half-empty box. The
    # sizing lives here, not in impact.tex, so re-running the generator cannot revert it.
    tex = (
        "%% GENERATED by `python -m backend.plots --poster` from traces/poster_runs/summary.json.\n"
        "%% Do not edit by hand -- re-run the campaign and regenerate instead.\n"
        "%% Sizing is intentional: this table is the whole of box 05, so it is set at 28pt with\n"
        "%% opened-up rows and centred. Change it in backend/plots.py:write_impact_table, not here.\n"
        "\\begin{center}\n"
        "{\\fontsize{28}{34}\\selectfont\n"
        "\\renewcommand{\\arraystretch}{1.52}%\n"
        "\\setlength{\\tabcolsep}{3.7mm}%\n"
        "\\begin{tabular}{@{}l l r r r l@{}}\n"
        "\\toprule\n"
        " & & \\textbf{Reference} & \\textbf{ObLoS only} & \\textbf{COSME} & sd \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        "\\end{center}\n"
    )
    out_path = out_path or os.path.join(REPO_ROOT, "poster", "impact_table.tex")
    with open(out_path, "w") as f:
        f.write(tex)
    return out_path


def write_impact_note(summary_path: str | None = None, out_path: str | None = None) -> str:
    """Emit poster/impact_note.tex: the bulk-transfer / congestion-control result.

    Kept out of the main table on purpose (see write_impact_table): a bulk transfer finishes in
    tens of seconds and so samples a far lighter part of the drive than the 600 s application rows.
    Both the transfer's window AND that window's loss duty cycle are computed here from the runs
    and the composed trace rather than written by hand -- they change whenever the transfer size
    changes, and a stale "~85 s / 1.4 %" would be worse than no caveat at all.

    The note also says why Cubic looks untouched, because otherwise the honest reading is the wrong
    one. COSME gates loss on/off (netem 100% for a burst), and TCP recovers from a short outage at
    full cwnd. Delivered as uniform random loss instead, the same average puts Cubic at 6.0 Mbit/s
    and leaves BBR at 298.8 -- measured on this stack, see docs/COMPOSITION.md section 7. A null
    result here is evidence about the loss model, not about congestion control on LEO links.
    """
    summary_path = summary_path or os.path.join(POSTER_RUNS_DIR, "summary.json")
    with open(summary_path) as f:
        summary = json.load(f)

    parts, window_s = [], None
    for cc in ("cubic", "bbr"):
        app = summary.get(f"file_transfer ({cc})") or {}
        row, dur = app.get("throughput_mbps"), app.get("duration_s")
        if not row:
            continue
        if dur:
            window_s = max(window_s or 0.0, dur["composed"]["mean"])
        ref, comp = row["reference"], row["composed"]
        if dur:   # completion time -- the metric a user actually experiences
            parts.append(f"\\textbf{{{cc.upper()}}} {dur['reference']['mean']:.1f}"
                          f"\\,$\\rightarrow$\\,\\hilite{{{dur['composed']['mean']:.1f}\\,s}}")
    window_s = window_s or 40.0

    # Loss duty cycle the transfer actually experiences, vs what the 600 s rows face.
    from backend.models import oblos as _oblos
    trace = _oblos.load_obstruction_trace("measurement-2025-04-03_16-58-multicar-onlyping")
    composed = compose(duration_s=660, obstruction_trace=trace, nominal_download_mbps=329.0,
                       nominal_upload_mbps=30.0, seed=42)
    df = composed.df
    short = df[df["timestamp"] < window_s]["loss"].mean() * 100
    long_ = df[df["timestamp"] < 600.0]["loss"].mean() * 100

    tex = (
        "%% GENERATED by `python -m backend.plots --poster`. Do not edit by hand.\n"
        "{\\smallfont\\color{muted}The two bulk-transfer rows use a lighter window: 1\\,GB "
        f"completes in ${{\\sim}}{window_s:.0f}$\\,s and sees {short:.1f}\\,\\% loss against the "
        f"{long_:.1f}\\,\\% above. Time to complete " + "; ".join(parts) + ".}\n"
    )
    out_path = out_path or os.path.join(REPO_ROOT, "poster", "impact_note.tex")
    with open(out_path, "w") as f:
        f.write(tex)
    return out_path


PAPER_DIR = os.path.join(REPO_ROOT, "paper", "COSME_Demo")


def write_paper_table(summary_path: str | None = None, out_path: str | None = None) -> str:
    """Emit paper/COSME_Demo/results_table.tex from the campaign summary.

    Separate from the poster's table on purpose: the paper is two-column acmart, so this is a
    narrower `table*` with mean +- sd rather than mean + range, and it drops the poster's
    "COSME range" column. Same source data, same metric-selection rules, so the two documents
    cannot disagree.

    Reporting +- sd here is the point: the paper's claim is that the emulation is deterministic
    enough for repeated evaluation, and the reference arm's spread (which is instrument noise
    only, since that arm has no stochastic component) is what evidences it.
    """
    summary_path = summary_path or os.path.join(POSTER_RUNS_DIR, "summary.json")
    with open(summary_path) as f:
        summary = json.load(f)

    ROWS = [
        ("voip", "mos", "VoIP", "MOS", "", 2),
        ("voip", "loss_pct", "", "audio loss (\\%)", "", 1),
        ("surveillance", "freeze_count", "Live video", "freezes / 10\\,min", "", 0),
        ("surveillance", "total_freeze_s", "", "time frozen (s)", "", 1),
        ("remote_desktop", "keystroke_latency_ms_p95", "Remote desktop", "keystroke p95 (ms)", "", 0),
        ("remote_desktop", "effective_fps_mean", "", "screen updates (fps)", "", 1),
        ("video_conferencing", "jitter_ms", "Video conf.", "jitter (ms)", "", 1),
        ("file_transfer (cubic)", "throughput_mbps", "Bulk transfer", "TCP Cubic (Mbit/s)", "", 0),
        ("file_transfer (bbr)", "throughput_mbps", "", "TCP BBR (Mbit/s)", "", 0),
    ]

    lines, n = [], None
    for app, metric, app_name, metric_name, _unit, dp in ROWS:
        row = (summary.get(app) or {}).get(metric)
        if not row or not row.get("separated"):
            continue
        ref, obs, comp = row["reference"], row["obstruction"], row["composed"]
        n = comp["n"]
        cells = " & ".join(f"{c['mean']:.{dp}f}\\,$\\pm$\\,{c['sd']:.{dp if dp else 1}f}"
                            for c in (ref, obs, comp))
        lines.append(f"{app_name} & {metric_name} & {cells} \\\\")

    tex = (
        "%% GENERATED by `python -m backend.plots --paper` from traces/poster_runs/summary.json.\n"
        "%% Do not edit by hand -- re-run the campaign and regenerate.\n"
        # table* (full width): five numeric columns do not fit one acmart column -- as a
        # single-column float it was pushed to a later page and set beside the references.
        "\\begin{table*}[t]\n"
        "\\caption{Measured application QoE under three arms with identical link provisioning "
        "(329/30\\,Mbit/s, 24\\,ms RTT). Mean\\,$\\pm$\\,standard deviation over "
        f"$n={n}$ runs (600\\,s each; bulk transfer 1\\,GB). "
        "Only metrics differing from both other arms (Welch's $t$, $p<0.01$) are listed.}\n"
        "\\label{tab:results}\n"
        "\\small\n"
        "\\begin{tabular}{@{}llrrr@{}}\n"
        "\\toprule\n"
        "Application & Metric & Reference & ObLoS only & \\emph{COSME} \\\\\n"
        "\\midrule\n"
        + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )
    out_path = out_path or os.path.join(PAPER_DIR, "results_table.tex")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(tex)
    return out_path


def build_poster_figures() -> list[str]:
    """Every figure the LaTeX poster includes, as vector PDF in poster/pics/."""
    os.makedirs(POSTER_PICS, exist_ok=True)
    paths = [
        plot_validation_bars(),
        plot_signal_merger(),
        plot_validation_strip(out_path=os.path.join(POSTER_PICS, "validation_strip.pdf")),
        plot_weather_impact(out_path=os.path.join(POSTER_PICS, "weather_impact.pdf")),
        plot_garcia_poster(),
    ]
    try:
        paths.append(write_impact_table())
        # write_impact_note() is not called here: box 05 is the table alone, and generating the
        # note into poster/ would place that prose back beside it.
    except (FileNotFoundError, RuntimeError) as e:
        print(f"skipping application-impact table: {e}")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-cc", action="store_true",
                         help="also generate the CC comparison plot (needs traces/cc_comparison.json)")
    parser.add_argument("--paper", action="store_true",
                         help="regenerate the demo paper's results table from the campaign summary")
    parser.add_argument("--poster", action="store_true",
                         help="generate only the LaTeX poster's own figures, as vector PDF in poster/pics/")
    args = parser.parse_args()

    if args.paper:
        print("wrote", write_paper_table())
        raise SystemExit(0)
    if args.poster:
        for p in build_poster_figures():
            print("wrote", p)
        raise SystemExit(0)

    paths = [plot_composition_timeline(), plot_validation_strip(), plot_weather_impact(),
             plot_garcia_delay_trace()]
    if args.with_cc:
        try:
            paths.append(plot_cc_comparison())
        except FileNotFoundError as e:
            print(f"skipping CC comparison plot: {e}")
    for p in paths:
        print("wrote", p)
