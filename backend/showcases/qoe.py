"""Shared QoE math for COSME's showcases and the analytical scenario fallback.

One home for the simplified ITU-T G.107-style R-factor/MOS formula so the
scenario dashboard's analytical estimate (backend/api.py:_estimate_qoe) and
the VoIP showcase's *measured* QoE (real per-second WebRTC stats reported by
the aiortc peer, see docker/endpoint-scripts/webrtc_peer.py) provably use the
same model -- the difference between them is purely whether the inputs are
trace-derived or measured on the wire.

The formula is deliberately simplified (Ie/Bpl delay+loss terms only, no
codec-specific Ie fitted) and is documented as indicative, not a validated
MOS -- same caveat as the original _estimate_qoe.
"""
from __future__ import annotations

import statistics
from typing import Iterable


def r_factor(effective_delay_ms: float, loss_frac: float) -> float:
    """Simplified R-factor from one-way effective delay (ms) and loss fraction (0..1)."""
    r = 93.2 - (effective_delay_ms / 40.0) - (loss_frac * 100 * 2.5)
    return max(0.0, min(93.2, r))


def mos_from_r(r: float) -> float:
    mos = 1 + 0.035 * r + r * (r - 60) * (100 - r) * 7e-6
    return max(1.0, min(4.5, mos))


def _mean(values: Iterable[float | None], default: float = 0.0) -> float:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else default


# RFC 3550 interarrival jitter is an EWMA: one packet with a corrupt timestamp spikes it and it
# then decays slowly, so a single bad sample can dominate a whole run's mean. Observed once in 270
# runs -- an otherwise healthy VoIP run reported 2479 ms mean jitter on a link configured with
# 12 ms constant delay and NO jitter term, while its loss, RTT and bitrate were all normal for its
# arm. Anything above this bound cannot come from the emulated link (modelled jitter is single-digit
# ms), so it is reported as unavailable rather than propagated into MOS, which is computed from it.
MAX_PLAUSIBLE_JITTER_MS = 500.0


def voip_qoe_from_totals(totals: dict) -> dict:
    """Measured VoIP QoE from the peer's own end-of-run totals, not from live samples.

    Why this exists: `voip_qoe_from_samples` below averages the per-second samples that reached
    the backend -- but those are POSTed over the emulated link and are dropped precisely when the
    link is worst. Measured on a real 600s run: the seconds that arrived averaged a 3.8% loss duty
    cycle, while the 34 gaps whose samples never made it averaged 36.4%. Averaging survivors
    therefore reports a link that is materially better than the one under test.

    `totals` is `VideoConferencingResult.totals` (built in-container by webrtc_peer.py). Loss here
    is a ratio of sums over the whole run -- total lost / (total lost + total received) -- rather
    than a mean of per-second ratios, which additionally scores a fully blacked-out second as 0%
    because no packets arrive to compare.

    Returns {} when totals are unavailable, so callers can fall back to the sampled path.
    """
    if not totals:
        return {}
    audio = ((totals.get("packets") or {}).get("audio")) or {}
    loss_frac = audio.get("loss_frac")
    rtt_ms = totals.get("rtt_ms_mean")
    jitter_ms = (totals.get("jitter_ms_mean") or {}).get("audio")
    if loss_frac is None or rtt_ms is None:
        return {}
    if jitter_ms is not None and jitter_ms > MAX_PLAUSIBLE_JITTER_MS:
        # Corrupt jitter estimate: drop the metrics derived from it, keep the ones that are not.
        return {
            "rtt_ms": round(rtt_ms, 1),
            "loss_pct": round(loss_frac * 100, 2),
            "audio_bitrate_kbps": round(totals.get("audio_bitrate_kbps_mean") or 0.0, 0),
            "packets_lost": audio.get("packets_lost"),
            "packets_received": audio.get("packets_received"),
            "n_samples": totals.get("n_samples_generated"),
            "jitter_implausible_ms": round(jitter_ms, 1),
            "source": "in-container totals (jitter estimate rejected)",
        }
    jitter_ms = jitter_ms or 0.0

    effective_delay_ms = rtt_ms / 2.0 + 2.0 * jitter_ms
    r = r_factor(effective_delay_ms, loss_frac)
    return {
        "mos": round(mos_from_r(r), 2),
        "r_factor": round(r, 1),
        "rtt_ms": round(rtt_ms, 1),
        "jitter_ms": round(jitter_ms, 1),
        "loss_pct": round(loss_frac * 100, 2),
        "audio_bitrate_kbps": round(totals.get("audio_bitrate_kbps_mean") or 0.0, 0),
        "packets_lost": audio.get("packets_lost"),
        "packets_received": audio.get("packets_received"),
        "n_samples": totals.get("n_samples_generated"),
        "source": "in-container totals (unbiased)",
    }


def media_qoe_from_totals(totals: dict) -> dict:
    """Video-conferencing counterpart of voip_qoe_from_totals -- same rationale."""
    if not totals:
        return {}
    video = ((totals.get("packets") or {}).get("video")) or {}
    loss_frac = video.get("loss_frac")
    rtt_ms = totals.get("rtt_ms_mean")
    if loss_frac is None or rtt_ms is None:
        return {}
    return {
        "video_bitrate_kbps": round(totals.get("video_bitrate_kbps_mean") or 0.0, 0),
        "framerate": round(totals.get("framerate_mean") or 0.0, 1),
        "loss_pct": round(loss_frac * 100, 2),
        "rtt_ms": round(rtt_ms, 1),
        "jitter_ms": round((totals.get("jitter_ms_mean") or {}).get("audio") or 0.0, 1),
        "packets_lost": video.get("packets_lost"),
        "packets_received": video.get("packets_received"),
        "n_samples": totals.get("n_samples_generated"),
        "source": "in-container totals (unbiased)",
    }


def voip_qoe_from_samples(samples: list[dict]) -> dict:
    """Measured VoIP QoE from the aiortc peer's per-second app-stats samples.

    Sample shape (see docker/endpoint-scripts/webrtc_peer.py): {t, app, rtt_ms,
    audio: {loss_frac, jitter_ms, bitrate_kbps, ...}, video: {...}|None}.
    Effective one-way delay is approximated as rtt/2 + 2*jitter (same jitter
    weighting as the analytical fallback). Tolerates missing fields --
    real stats can be sparse in the first seconds of a call.
    """
    if not samples:
        return {}
    rtt_ms = _mean(s.get("rtt_ms") for s in samples)
    audio = [s.get("audio") or {} for s in samples]
    jitter_ms = _mean(a.get("jitter_ms") for a in audio)
    loss_frac = _mean(a.get("loss_frac") for a in audio)
    bitrate_kbps = _mean(a.get("bitrate_kbps") for a in audio)

    effective_delay_ms = rtt_ms / 2.0 + 2.0 * jitter_ms
    r = r_factor(effective_delay_ms, loss_frac)
    return {
        "mos": round(mos_from_r(r), 2),
        "r_factor": round(r, 1),
        "rtt_ms": round(rtt_ms, 1),
        "jitter_ms": round(jitter_ms, 1),
        "loss_pct": round(loss_frac * 100, 2),
        "audio_bitrate_kbps": round(bitrate_kbps, 0),
        "n_samples": len(samples),
    }


def media_qoe_from_samples(samples: list[dict]) -> dict:
    """Measured video-conferencing QoE from the bot's per-second samples."""
    if not samples:
        return {}
    video = [s.get("video") or {} for s in samples]
    audio = [s.get("audio") or {} for s in samples]
    return {
        "video_bitrate_kbps": round(_mean(v.get("bitrate_kbps") for v in video), 0),
        "framerate": round(_mean(v.get("framerate") for v in video), 1),
        "loss_pct": round(_mean(v.get("loss_frac") for v in video) * 100, 2),
        "rtt_ms": round(_mean(s.get("rtt_ms") for s in samples), 1),
        "jitter_ms": round(_mean(a.get("jitter_ms") for a in audio), 1),
        "n_samples": len(samples),
    }


def surveillance_qoe_from_samples(samples: list[dict]) -> dict:
    """Live surveillance QoE from the receiver probe's per-second samples.

    Sample shape (see docker/endpoint-scripts/surveillance_probe.py):
    {t, app, fps, bitrate_kbps, in_freeze}.
    """
    if not samples:
        return {}
    return {
        "fps": round(_mean(s.get("fps") for s in samples), 1),
        "bitrate_kbps": round(_mean(s.get("bitrate_kbps") for s in samples), 0),
        "freeze_count": sum(1 for a, b in zip(samples, samples[1:])
                            if not a.get("in_freeze") and b.get("in_freeze"))
                        + (1 if samples and samples[0].get("in_freeze") else 0),
        "n_samples": len(samples),
    }


def remote_desktop_qoe_from_samples(samples: list[dict]) -> dict:
    """Live remote-desktop QoE from the VNC probe's per-second samples.

    Sample shape (see docker/probe/vnc_probe.py): {t, app,
    keystroke_latency_ms|None, effective_fps}.
    """
    if not samples:
        return {}
    latencies = [s["keystroke_latency_ms"] for s in samples if s.get("keystroke_latency_ms") is not None]
    out = {
        "effective_fps": round(_mean(s.get("effective_fps") for s in samples), 1),
        "n_samples": len(samples),
    }
    if latencies:
        out["keystroke_latency_ms_median"] = round(statistics.median(latencies), 1)
        out["keystroke_latency_ms_p95"] = round(
            statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 2 else latencies[0], 1
        )
    return out


def freeze_stats(frame_times: list[float], threshold_s: float = 0.2) -> dict:
    """Freeze events from raw frame arrival timestamps: gaps > threshold_s.

    Used by the surveillance probe's end-of-run summary; kept here so the
    definition of "a freeze" lives in exactly one place.
    """
    gaps = [b - a for a, b in zip(frame_times, frame_times[1:])]
    freezes = [g for g in gaps if g > threshold_s]
    return {
        "freeze_count": len(freezes),
        "total_freeze_s": round(sum(freezes), 2),
        "longest_freeze_s": round(max(freezes), 2) if freezes else 0.0,
    }
