#!/usr/bin/env python3
"""Two-peer WebRTC call for the video-conferencing and VoIP showcases.

A real aiortc WebRTC session directly between cosme-client and cosme-server
(see docs/APPLICATIONS.md and backend/showcases/video_conferencing.py for
the media-plane rationale and history).

Both peers send real media from the Big Buck Bunny asset (audio-only in
VoIP mode) and receive the other's stream, so genuine RTP/RTCP crosses the
tc/netem-shaped bridge in both directions. Signaling is a single JSON
offer/answer exchange over a TCP socket on the unshaped-irrelevant bridge
(one round trip, before media starts).

The offer-side peer (cosme-client) measures received QoE per second --
framerate/resolution by consuming the actual decoded frames, loss/jitter
from RTCP inbound stats, RTT from RTCP remote-inbound -- POSTs samples to
the backend app-stats endpoint, and prints an end-of-run JSON summary as its
last stdout line. When --live-endpoint is given, it also POSTs a JPEG of the
latest decoded video frame each second (mode=video, the dashboard's live
view) or includes an RMS audio level in the stats sample (mode=audio, the
VoIP level meter) -- both derived from the actual media coming off the
shaped link, not a synthetic placeholder.

Runs inside the endpoint containers (bind-mounted at /opt/cosme); needs
python3-aiortc from the endpoint image (see docker/Dockerfile.endpoint).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import socket
import time
import urllib.error
import urllib.request

import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer

SIGNALING_PORT = 9600


def _strip_extmap(sdp: str) -> str:
    """Removes RTP header-extension negotiation from an SDP.

    Debian bookworm's aiortc 1.4.0 has a receiver-side RTP parser bug: with
    the (self-negotiated) abs-send-time header extension present, EVERY
    incoming packet dies with `struct.error: unpack requires a buffer of 4
    bytes` in rtp.py -- the sender counts packetsSent normally while the
    receiver decodes zero frames (verified in an in-container loopback
    call). De-negotiating all extmaps (they're only used for BWE, which the
    fixed netem-shaped link doesn't need) makes media flow. Applied to both
    the outgoing and incoming SDP by both roles.
    """
    return "\n".join(l for l in sdp.split("\n") if not l.startswith("a=extmap:"))


def _post(endpoint: str, data: bytes, content_type: str) -> None:
    try:
        req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": content_type})
        urllib.request.urlopen(req, timeout=3)
    except (urllib.error.URLError, OSError):
        pass  # backend unreachable this tick -- keep measuring


def _post_stats(endpoint: str, sample: dict) -> None:
    _post(endpoint, json.dumps(sample).encode(), "application/json")


class FrameCounter:
    """Consumes a remote track, counting really-decoded frames (the QoE truth).

    Also keeps the LAST decoded frame around so the periodic report loop can
    derive a live-view artifact from it: a JPEG snapshot for video (dashboard
    live view), or an RMS level for audio (VoIP level meter) -- both are
    genuinely what's coming off the shaped link at that instant, not a
    separate/cheaper approximation.
    """

    def __init__(self) -> None:
        self.frames = 0
        self.width = None
        self.height = None
        self.last_frame = None
        self.task: asyncio.Task | None = None

    def start(self, track) -> None:
        async def _consume() -> None:
            while True:
                try:
                    frame = await track.recv()
                except Exception:
                    return
                self.frames += 1
                self.width = getattr(frame, "width", self.width)
                self.height = getattr(frame, "height", self.height)
                self.last_frame = frame

        self.task = asyncio.get_event_loop().create_task(_consume())


def _audio_rms_level(frame) -> float | None:
    """RMS level of a decoded audio frame, normalized to roughly 0..1.

    aiortc/PyAV decodes Opus to signed 16-bit PCM, so dividing by the int16
    peak gives a reasonable 0..1-ish RMS regardless of channel layout.
    """
    try:
        arr = frame.to_ndarray().astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(np.square(arr))))
    except Exception:
        return None


def _video_live_jpeg(frame, quality: int = 70) -> bytes | None:
    """JPEG-encodes a decoded video frame for the dashboard's live view."""
    try:
        buf = io.BytesIO()
        frame.to_image().save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except Exception:
        return None


async def run(args) -> None:
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    player = MediaPlayer(args.asset)
    blackhole = MediaBlackhole()
    video_counter = FrameCounter()
    audio_counter = FrameCounter()

    if player.audio:
        pc.addTrack(player.audio)
    if args.mode == "video" and player.video:
        pc.addTrack(player.video)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            video_counter.start(track)
        elif args.role == "offer":
            audio_counter.start(track)
        else:
            blackhole.addTrack(track)

    if args.role == "answer":
        srv = socket.create_server(("0.0.0.0", args.port))
        srv.settimeout(60)
        conn, _ = srv.accept()
        f = conn.makefile("rw")
        offer = json.loads(f.readline())
        await pc.setRemoteDescription(
            RTCSessionDescription(_strip_extmap(offer["sdp"]), offer["type"]))
        await blackhole.start()
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        f.write(json.dumps({"sdp": _strip_extmap(pc.localDescription.sdp),
                            "type": pc.localDescription.type}) + "\n")
        f.flush()
    else:
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        conn = socket.create_connection((args.connect, args.port), timeout=30)
        f = conn.makefile("rw")
        f.write(json.dumps({"sdp": _strip_extmap(pc.localDescription.sdp),
                            "type": pc.localDescription.type}) + "\n")
        f.flush()
        answer = json.loads(f.readline())
        await pc.setRemoteDescription(
            RTCSessionDescription(_strip_extmap(answer["sdp"]), answer["type"]))

    start = time.monotonic()
    prev = {"vframes": 0, "aframes": 0, "bytes": 0, "lost": {}, "recv": {}}
    samples = []

    while time.monotonic() - start < args.duration:
        await asyncio.sleep(1.0)
        if args.role != "offer":
            continue

        rtt_ms = None
        jitter_ms = {}
        lost_frac = {}
        transport_bytes = 0
        stats = await pc.getStats()
        for s in stats.values():
            typ = getattr(s, "type", "")
            kind = getattr(s, "kind", None)
            if typ == "remote-inbound-rtp":
                rt = getattr(s, "roundTripTime", None)
                if rt is not None:
                    rtt_ms = rt * 1000
            elif typ == "inbound-rtp":
                j = getattr(s, "jitter", None)
                if j is not None:
                    # aiortc 1.4 reports interarrival jitter in RTP clock
                    # units (RFC 3550 native form), not seconds -- convert
                    # via the media clock rate (48kHz Opus / 90kHz video).
                    clock = 48000 if kind == "audio" else 90000
                    jitter_ms[kind] = j / clock * 1000
                lost = getattr(s, "packetsLost", 0) or 0
                recv = getattr(s, "packetsReceived", 0) or 0
                dl = lost - prev["lost"].get(kind, 0)
                dr = recv - prev["recv"].get(kind, 0)
                prev["lost"][kind], prev["recv"][kind] = lost, recv
                total = max(dl, 0) + max(dr, 0)
                lost_frac[kind] = (max(dl, 0) / total) if total > 0 else 0.0
            elif typ == "transport":
                transport_bytes += getattr(s, "bytesReceived", 0) or 0

        d_bytes = transport_bytes - prev["bytes"]
        prev["bytes"] = transport_bytes
        d_vframes = video_counter.frames - prev["vframes"]
        prev["vframes"] = video_counter.frames
        d_aframes = audio_counter.frames - prev["aframes"]
        prev["aframes"] = audio_counter.frames

        bitrate_kbps = max(d_bytes, 0) * 8 / 1000
        audio_dict = {
            "jitter_ms": jitter_ms.get("audio"),
            "loss_frac": lost_frac.get("audio"),
            "frames_per_s": d_aframes,
            # in audio-only mode the transport bitrate IS the audio bitrate
            "bitrate_kbps": bitrate_kbps if args.mode == "audio" else None,
        }
        if args.mode == "audio" and audio_counter.last_frame is not None:
            # Live level meter for the VoIP dashboard tile -- what's actually
            # coming off the shaped link this second, not a synthetic tone.
            audio_dict["level"] = _audio_rms_level(audio_counter.last_frame)
        sample = {
            "t": time.time(),
            "app": args.app,
            "rtt_ms": rtt_ms,
            "audio": audio_dict,
            "video": None if args.mode != "video" else {
                "jitter_ms": jitter_ms.get("video"),
                "loss_frac": lost_frac.get("video"),
                "framerate": d_vframes,
                "frame_width": video_counter.width,
                "frame_height": video_counter.height,
                "bitrate_kbps": bitrate_kbps,  # transport-level (audio share is small)
            },
        }
        samples.append(sample)
        if args.stats_endpoint:
            _post_stats(args.stats_endpoint, sample)

        if args.live_endpoint and args.mode == "video" and video_counter.last_frame is not None:
            jpeg = _video_live_jpeg(video_counter.last_frame)
            if jpeg is not None:
                _post(args.live_endpoint, jpeg, "image/jpeg")

    await pc.close()
    if args.role == "offer":
        # ------------------------------------------------------------------
        # Unbiased end-of-run totals, computed HERE rather than from the
        # per-second samples the backend received.
        #
        # The live samples are POSTed over the emulated link, so they are lost
        # exactly when the link is worst: measured on a real 600s run, the
        # seconds that did arrive averaged 3.8% loss duty cycle while the 34
        # gaps whose samples never made it averaged 36.4%. Averaging what
        # arrives therefore understates every QoE metric, badly.
        #
        # packetsLost/packetsReceived are cumulative RTCP counters, so the last
        # values seen are the run totals. Loss is then a ratio of sums over the
        # whole run -- not a mean of per-second ratios, which additionally
        # scores a fully blacked-out second as 0.0 (no packets => dl+dr == 0).
        # ------------------------------------------------------------------
        totals = {}
        for kind in ("audio", "video"):
            lost = prev["lost"].get(kind)
            recv = prev["recv"].get(kind)
            if lost is None and recv is None:
                continue
            lost, recv = max(lost or 0, 0), max(recv or 0, 0)
            totals[kind] = {
                "packets_lost": lost,
                "packets_received": recv,
                "loss_frac": (lost / (lost + recv)) if (lost + recv) > 0 else None,
            }

        def _mean(vals):
            vals = [v for v in vals if v is not None]
            return (sum(vals) / len(vals)) if vals else None

        print(json.dumps({
            "n_samples": len(samples),
            "video_frames_received": video_counter.frames,
            "audio_frames_received": audio_counter.frames,
            # Aggregates over EVERY sample this peer generated, including the
            # ones whose POST never arrived.
            "totals": totals,
            "rtt_ms_mean": _mean(s["rtt_ms"] for s in samples),
            "jitter_ms_mean": {
                "audio": _mean((s.get("audio") or {}).get("jitter_ms") for s in samples),
                "video": _mean((s.get("video") or {}).get("jitter_ms") for s in samples),
            },
            "audio_bitrate_kbps_mean": _mean((s.get("audio") or {}).get("bitrate_kbps") for s in samples),
            "video_bitrate_kbps_mean": _mean((s.get("video") or {}).get("bitrate_kbps") for s in samples),
            "framerate_mean": _mean((s.get("video") or {}).get("framerate") for s in samples),
            "last": samples[-1] if samples else None,
        }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["offer", "answer"], required=True)
    parser.add_argument("--connect", default="10.42.0.10", help="answer peer's address (offer role)")
    parser.add_argument("--port", type=int, default=SIGNALING_PORT)
    parser.add_argument("--mode", choices=["video", "audio"], default="video")
    parser.add_argument("--asset", default="/srv/media/bigbuckbunny.mp4")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--app", default="video_conferencing")
    parser.add_argument("--stats-endpoint", default=None)
    parser.add_argument("--live-endpoint", default=None,
                        help="POST a JPEG snapshot of the latest decoded video frame here each "
                             "second (mode=video only; dashboard live view)")
    asyncio.get_event_loop().run_until_complete(run(parser.parse_args()))
