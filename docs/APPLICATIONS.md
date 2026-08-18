# COSME Application Showcases

The COSME paper (`paper/COSME_Demo/main.tex`) promises "five diverse application showcases":
two TCP-based and three UDP-based applications, with TCP scenarios runnable under different
congestion-control algorithms. **All five are implemented and verified end-to-end on real
Docker** (see `scripts/verify_docker_stack.sh` and `pytest -m docker`).

Per project decision, every showcase uses **real application traffic** with **really measured
QoE** — no synthetic iperf3 profiles, no analytical estimates presented as measurements. All
reuse one freely-licensed media asset (Big Buck Bunny, Blender Foundation, CC BY 3.0 — prepared
by `scripts/prepare_media_assets.sh`).

## The five showcases

| Showcase | Transport | Driver | CC-selectable | Measured QoE |
|---|---|---|---|---|
| File transfer | TCP | `backend/showcases/file_transfer.py` | Yes | transfer time, throughput, TCP retransmits (curl `-w` + `ss -ti`) |
| Remote desktop (VNC) | TCP | `backend/showcases/remote_desktop.py` + `docker/probe/vnc_probe.py` | Yes | keystroke→screen-echo round trip (median/p95), effective screen update rate |
| Video conferencing | UDP (WebRTC/RTP) | `backend/showcases/video_conferencing.py` + `docker/endpoint-scripts/webrtc_peer.py` | N/A | received bitrate, framerate, resolution, RTP loss, jitter, RTCP RTT |
| VoIP call | UDP (WebRTC/RTP, audio-only) | same, `mode="audio"` (`run_voip_showcase`) | N/A | **measured MOS/R-factor** (from real loss/jitter/RTT), audio bitrate |
| Remote surveillance | UDP (MPEG-TS) | `backend/showcases/surveillance.py` + `docker/endpoint-scripts/surveillance_probe.py` | N/A | freeze events (count/total/longest), received bitrate, decoded fps |

All five run through one unified API pattern: `POST /api/showcase/<app>` returns a job id,
`GET /api/showcase/jobs/{id}` polls it, live per-second samples stream to
`POST /api/showcase/app-stats` (from inside the containers, straight to the backend's own
cosme-link address `10.42.0.2:8731` — hence uvicorn's required `--host 0.0.0.0`; file_transfer.py
is the one exception, since its polling loop runs IN the backend process itself, so it posts to
`127.0.0.1:8731`), and `GET /api/showcase/qoe/{app}` serves the aggregated measured QoE the
dashboard's per-application tiles render. The registry that drives the dashboard (labels,
transports, metric definitions) is `GET /api/showcase/apps`.

**Live view**: while a showcase runs, the dashboard's Live view tile shows what's actually
happening, not just numbers — real decoded video (video conferencing, surveillance), the real VNC
screen (remote desktop), a real audio-level meter + running MOS (VoIP), or a live progress bar +
throughput sparkline (file transfer). Image-producing showcases POST a JPEG/PNG to
`POST /api/showcase/live-frame?app=<id>` about once a second; `GET /api/showcase/live-frame/{app}`
serves the latest one back (204 if none yet or stale >5s, so a finished/crashed run doesn't leave a
stale frame showing forever). VoIP's level/MOS and file transfer's progress/throughput instead ride
the existing app-stats samples (an `audio.level` field, and `bytes`/`total_bytes`/`throughput_mbps`
fields, respectively) — no separate endpoint needed for those. On completion, the tile
auto-switches to **Run summary**: the app's measured metrics plotted against time, aligned onto the
concurrently-running scenario's own sim-time axis (`sim_t = (sample.t - scenario.start_time) *
speed`, both exposed via `/api/scenarios/{id}/status`) and shown directly above that scenario's
loss strip and delay/jitter chart over the same window.

## Per-showcase design notes

### File transfer (TCP)
`cosme-server` serves the MP4 via `python3 -m http.server` on port **8081**; `cosme-client`
fetches it with real `curl`, run detached (`docker exec -d`) so the backend can poll the
downloading file's growing size for live progress/throughput while the transfer is still in
flight, rather than only learning the result after `curl` exits. CC is set sender-side via the
nsenter mechanism (see README troubleshooting).

### Remote desktop (VNC, TCP)
TigerVNC (`Xtigervnc :1`, SecurityTypes None — bridge-internal only) in `cosme-server`, with
two xterms: an "activity" window printing a 10Hz timestamp (giving the effective-fps metric a
known ceiling that shaping visibly degrades) and an "echo" window targeted by keystrokes
(spatially separate so scrolling activity can't masquerade as an echo; keyboard focus via X
PointerRoot focus-follows-mouse). The client is a vncdotool RFB probe in the dedicated
`cosme-probe` container (shares cosme-client's netns; vncdotool is pip-only, not packaged in
Debian, hence its own minimal `python:3.12-slim` image rather than adding pip to the endpoint
image). Both metrics are deliberately client-perceived: each sample is a real
FramebufferUpdateRequest round trip, so
under 2×100ms netem the measured keystroke echo goes 60ms → ~660ms and effective fps 9.4 → ~1.1
(verified). This supersedes the earlier noVNC-in-browser idea — native RFB with programmatic
timing, no websockify layer. The probe also does a real `captureScreen()` once a second and POSTs
it as the live view's PNG when `--live-endpoint` is passed.

### Video conferencing + VoIP (UDP, WebRTC)
A real two-peer **aiortc** WebRTC session between `cosme-client` and `cosme-server`
(`docker/endpoint-scripts/webrtc_peer.py`): both peers stream the asset (audio-only in VoIP
mode — genuinely no video track), RTP/RTCP crosses both shaped interfaces, and the client peer
measures received QoE from the actually-decoded frames plus RTCP stats. VoIP MOS is computed
backend-side (`backend/showcases/qoe.py`) from the measured loss/jitter/RTT using the same
simplified G.107 R-factor formula as the scenario dashboard's analytical estimate — so the
estimated and measured numbers are directly comparable by construction. Verified degradation:
150ms±40ms + 4% loss shaping → measured RTT 308ms, loss 4.3%, MOS 4.41→3.92.

Two implementation notes: (a) this supersedes the earlier SIP-client recipe. It also supersedes
an intermediate approach that drove a full self-hosted Jitsi Meet stack (web/prosody/jicofo/JVB)
with a headless-Chromium bot — that stack was brought up and debugged extensively on real
Docker (nine distinct fixes: secure-origin requirements for headless Chromium, prejoin-bypass
stripping media tracks, the colibri bridge-channel websocket being off by default, inconsistent
XMPP domain overrides across components, a degenerate `JVB_ADVERTISE_IPS` mapping that broke all
outgoing ICE checks, missing component auth passwords, a wrong compose build context, and a
missing `playwright` pip package in the bot image) and reached joined conferences with completed
ICE/DTLS and the JVB receiving both participants' audio, but the JVB never sustainably forwarded
media onward in that shared-netns container topology (verified with tcpdump on two
docker-jitsi-meet releases — likely an upstream ice4j/JVB issue with this specific topology, not
conclusively root-caused). The Jitsi stack has been removed from the repo; the aiortc media
plane is simpler, has no signaling-server dependency, and is what's actually used. (b) Debian
bookworm's aiortc 1.4.0 has an RTP header-extension parser bug that silently discards every
received packet; `webrtc_peer.py` works around it by stripping `a=extmap:` lines during
signaling (documented in the script). (c) `FrameCounter._consume` keeps the last decoded frame
around: for video conferencing, `frame.to_image()` JPEG-encodes it once a second for the live
view; for VoIP, `frame.to_ndarray()` feeds a real RMS audio level (needs `python3-pil`/numpy in
the endpoint image, respectively).

### Remote surveillance (UDP, one-way)
ffmpeg in `cosme-server` pushes the asset as **plain MPEG-TS over UDP** (`-c copy`, TS packets
sized to fit one MTU) to a receiver probe in `cosme-client` that decodes with
`-vf showinfo` and wall-clock-stamps every decoded frame. Freezes are frame-arrival gaps
>200ms (`qoe.py:freeze_stats` is the single definition); received bitrate comes from the
demuxer's byte offsets. Transport trade-off, documented: RTSP would need a media server
(mediamtx) plus a TCP control channel and duplicates RTP territory the WebRTC showcases cover;
raw MPEG-TS is genuinely UDP end-to-end with zero extra services, and loss maps directly onto
the freeze phenomenology being measured. Verified: 12% netem loss → 3 freeze events, 1.46s
frozen, in a 20s run. The probe also runs a SECOND ffmpeg output on the same input
(`-vf fps=2 -update 1`, the standard ffmpeg multi-output-from-one-input pattern) writing a ~2fps
still that's read and POSTed for the live view once a second — genuinely the decoded picture at
that moment, corruption/freezes included, not a synthetic placeholder.

## Congestion control

Both TCP showcases (file transfer, remote desktop) set the sender-side
(`cosme-server`) algorithm via `file_transfer.set_congestion_control()` — the host-side
nsenter mechanism (plain `docker exec sysctl -w` fails on modern Docker; see README). The
dashboard's CC selector is populated from the live `GET /api/system/congestion-controls`
(BBR appears only when the host has `tcp_bbr` loaded) and is **disabled whenever a UDP
application is selected** — UDP apps have no TCP congestion control; their tunable would be an
application-level bitrate/quality knob (a possible future extension).
