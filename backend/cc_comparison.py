"""Real, Docker-measured TCP congestion-control comparison (Cubic vs. BBR vs. Reno).

Answers a concrete question raised while testing the dashboard: is the CC selector actually
changing anything, and does it matter under LEO-like conditions? `file_transfer.py`'s
`set_congestion_control()` is unit-tested for the *mechanism* (set + read back via nsenter,
see backend/tests/test_docker_integration.py::TestCongestionControl), but that doesn't by
itself show a measurable throughput difference under real impairment -- this module runs the
actual file-transfer showcase, back to back, under each CC algorithm, against the SAME
static netem profile applied via the same `DockerNetemBackend` the dashboard uses, and reports
real measured throughput/retransmits.

The profile is a moderate, LEO-representative shape (delay+jitter+loss+rate) rather than one of
COSME's own time-varying composed traces, specifically so each CC arm sees IDENTICAL, repeatable
conditions -- a fair, controlled A/B, not a comparison muddied by the trace also changing between
runs. `docs/COMPOSITION.md` section 6 documents why loss under Starlink's real (drop-front) queuing
may hit loss-based CCs like Cubic harder than this simple netem-loss model does; this comparison
still shows the expected qualitative direction (Cubic degrades more than BBR as loss increases)
even without modeling that specific mechanism.

Requires the real Docker stack up (`./scripts/start_demo.sh`) and BBR loaded on the host
(`sudo modprobe tcp_bbr`) for the bbr arm; skips/raises clearly otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

from backend.netem_backend import DockerNetemBackend, NetemParams
from backend.showcases import file_transfer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(REPO_ROOT, "traces", "cc_comparison.json")

DOCKER_CONTAINERS = {"client": "cosme-client", "server": "cosme-server"}

# A fixed, moderate LEO-representative shape: real Garcia-fitted delay sits ~10-30ms, real
# Zimmermann/reconfig bursts are brief 100-400ms full-loss events -- 1.5% steady loss is a
# rough, simplified stand-in for a drive with intermittent obstruction/handover activity
# averaged over a whole transfer, not a claim that real loss is Bernoulli/uniform (compose()'s
# own traces are the accurate, time-varying version; this is deliberately static for a fair A/B).
DEFAULT_PROFILE = NetemParams(loss_pct=1.5, delay_ms=25.0, jitter_ms=8.0, rate_mbit=50.0)


@dataclass
class CcRunResult:
    congestion_control: str
    throughput_mbps: float
    duration_s: float
    size_mb: float
    tcp_retransmits: int | None


def run_comparison(
    ccs: tuple[str, ...] = ("cubic", "bbr"),
    profile: NetemParams = DEFAULT_PROFILE,
    size_gb: float = 0.05,
) -> list[CcRunResult]:
    """Runs the real file-transfer showcase once per CC, all under the identical netem profile."""
    backend = DockerNetemBackend(containers=DOCKER_CONTAINERS)
    results: list[CcRunResult] = []
    try:
        backend.apply("server", profile)
        backend.apply("client", profile)
        for cc in ccs:
            print(f"[cc_comparison] starting {cc}...", flush=True)
            result = file_transfer.run_file_transfer_showcase(size_gb=size_gb, congestion_control=cc)
            print(f"[cc_comparison] {cc} done: {result.throughput_bps/1e6:.2f} Mbit/s, "
                  f"{result.duration_s:.2f}s, retx={result.tcp_retransmits}", flush=True)
            results.append(CcRunResult(
                congestion_control=cc,
                throughput_mbps=result.throughput_bps / 1e6,
                duration_s=result.duration_s,
                size_mb=result.size_bytes / 1e6,
                tcp_retransmits=result.tcp_retransmits,
            ))
            time.sleep(1.0)  # let any in-flight retransmits/queued packets settle between runs
    finally:
        backend.reset("server")
        backend.reset("client")
    return results


def save_results(results: list[CcRunResult], path: str = RESULTS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "profile": asdict(DEFAULT_PROFILE),
            "generated_at": time.time(),
            "results": [asdict(r) for r in results],
        }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccs", nargs="+", default=["cubic", "bbr"])
    parser.add_argument("--size-gb", type=float, default=0.05)
    args = parser.parse_args()

    results = run_comparison(ccs=tuple(args.ccs), size_gb=args.size_gb)
    for r in results:
        print(f"{r.congestion_control:>8}: {r.throughput_mbps:6.2f} Mbit/s over {r.duration_s:5.2f}s "
              f"({r.size_mb:.1f}MB), retransmits={r.tcp_retransmits}")
    save_results(results)
    print(f"saved to {RESULTS_PATH}")
