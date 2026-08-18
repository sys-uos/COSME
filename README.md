# COSME — Composable Orchestrated Starlink Mobility Emulation

COSME emulates a moving Starlink link in real time, presented as Demo at SIGCOMM 2026 in Denver, CO, USA. The demo abstract is available in [ACM digital library](https://doi.org/10.1145/3789240.3830288). It composes four measurement-derived impairment
models into one time series and plays that back over Docker containers with `tc`/`netem`, so real
applications experience a modelled LEO link on a real network stack.

| Model | Contributes | Source |
|---|---|---|
| ObLoS | obstruction loss (boolean) | bridge/tunnel geometry along the route |
| Zimmermann | reconfiguration loss (boolean bursts) | 57 real Osnabrück↔Hannover drives |
| Garcia | one-way delay + jitter (ms) | fitted from the same drives' RTT measurements |
| WetLinks | download/upload bandwidth (Mbit/s) | rain-rate buckets, real DWD history or presets |

The two loss signals combine by boolean OR; bandwidth applies independently. Garcia and Zimmermann
share one handover schedule, so a handover produces both a delay bump and a loss burst at the same
instant. Full rules in [docs/COMPOSITION.md](docs/COMPOSITION.md); validation against held-out
drives in [docs/VALIDATION.md](docs/VALIDATION.md).

## Quick start

```bash
git clone <repo> && cd cosme
./scripts/start_demo.sh --skip-osrm
```

That builds and starts the whole stack and prints the dashboard URL
(`http://<host-ip>:8732/index.html`). It takes a few minutes on first run, mostly image builds and
the ~500 MB media download for the video showcases.

Drop `--skip-osrm` only if you want the "simulate a custom route" feature — it adds a one-time,
multi-hour, ~50 GB preprocessing pass over whole-Germany OSM data. Everything else, including all
five application showcases and every pre-recorded drive, works without it.

`./scripts/start_demo.sh --down` stops and removes everything.

## Requirements

- **Docker** with the Compose plugin (`docker compose version` should print a version). No further
  Docker configuration is needed: `cosme-backend` runs privileged and orchestrates its sibling
  containers itself.
- **Python 3.10–3.12** with `python3-venv` and `build-essential`, only if you want to run the model
  code or tests outside Docker. Built and tested on 3.12.
  ```bash
  sudo apt install -y python3-venv build-essential
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r backend/requirements.txt
  ```
- **~8 GB RAM.** OSRM in particular is memory-hungry during preprocessing; add swap if you hit an
  OOM kill during `prepare_osrm.sh`.
- Optional: `sudo modprobe tcp_bbr` on the host if you want BBR selectable as a congestion-control
  option. Cubic and Reno always work. Containers cannot load kernel modules themselves.

## Research data

The measurement data COSME's models are fitted from is **not in this repository** — it is ~2.7 GB
and not ours to redistribute. Without it you still get the full Docker stack, the dashboard, live
route simulation, and all five showcases; what you lose is playback of the pre-recorded drives and
the ability to re-fit or re-validate the models against real traces.

Two fitted-parameter caches derived from that data *are* checked in
(`traces/zimmermann_fit.json`, `traces/garcia_fit.json`), so Garcia's and Zimmermann's
distributions work out of the box regardless.

To use the real data, obtain it from the authors and place it under `models/`:

```
models/
  Zimmermann/clipped_measurements/measurement-*-multicar-onlyping/   # 57 usable drives
  WetLinks-main/Preprocessed_Data/analysis_data_*.csv
  ObLoS/website/public/obstacle_data.json                            # BASt/OSM bridge catalog
  ObLoS/website/meta/                                                # width-estimator fallback data
```

The test suite reports this state explicitly rather than failing: without `models/` it is
118 passed, 46 skipped; with it, 164 passed.

## Using the dashboard

Pick a **real drive**, or switch to **Simulate custom route** and set start and end by clicking the
map or typing a place name. Choose a weather preset and a TCP congestion-control algorithm, then
start the run. The map, timeline, loss strips and delay chart update as the trace plays.

The **showcases** panel runs one of five real applications through the shaped link:

| Showcase | Transport | Measured |
|---|---|---|
| File transfer | TCP/HTTP | throughput, retransmits, completion time |
| Remote desktop | TCP/VNC | keystroke round-trip latency, update rate |
| Video conference | UDP/WebRTC | bitrate, frame rate, loss |
| VoIP | UDP/WebRTC | MOS, R-factor, jitter |
| Surveillance | UDP/MPEG-TS | freeze events, bitrate |

While a showcase runs, the live view shows the actual decoded video frame, VNC screen, audio level
or transfer progress. When it finishes, the panel switches to a run summary plotting the app's
measured metrics against the scenario's own loss/delay/jitter over the same window, so you can see
whether QoE tracked the impairment. Details in [docs/APPLICATIONS.md](docs/APPLICATIONS.md).

A scenario and a showcase may run at the same time — that is the intended demo pattern. A *second*
scenario or a second showcase is rejected with a 409; there is only one physical link.

## Command-line use

```bash
python -m backend.compose                  # smoke-test the composition engine
python -m backend.validation               # validate against held-out drives
python -m backend.orchestrator --drive <name> --duration 120 --speed 20   # dry-run playback
python -m backend.models.zimmermann        # (re)fit the reconfiguration-loss distribution
python -m backend.models.garcia_fit        # (re)fit delay/jitter from real RTT measurements
python -m backend.models.wetlinks          # print rain-bucket bandwidth factors
python -m backend.geocoding "Osnabrück"    # place name -> coordinates

python -m pytest backend/tests/ -m "not docker"   # 164 unit tests, no Docker
python -m pytest backend/tests/ -m docker         # 8 integration tests vs a running stack
```

The `-m docker` tests skip automatically when the stack isn't up.
`scripts/verify_docker_stack.sh` is the fuller bring-up checklist and ends by running them.

## Layout

```
backend/
  models/        the four impairment models, each independently loadable
                 oblos.py (recorded traces) / oblos_live.py (live OSRM+Overpass route simulation)
                 garcia.py + garcia_fit.py, wetlinks.py, zimmermann.py, dwd_weather.py
                 reconfig_schedule.py — the shared handover timeline both Garcia and Zimmermann use
  showcases/     the five applications above, plus qoe.py (shared MOS/R-factor math)
  compose.py     merges the four models onto one 10 ms grid
  orchestrator.py  composed trace -> tc/netem playback schedule
  netem_backend.py Docker backend, or a dry-run backend that logs commands instead
  api.py         FastAPI backend; runs as the cosme-backend container
docker/          one compose file for all six containers
frontend/        single-file dashboard (index.html)
scripts/         start_demo.sh and the one-time data preparation scripts
docs/            composition rules, validation numbers, showcase details
traces/          fitted-parameter caches and validation output (checked in)
models/          research data — not included, see above
```

Six containers: `cosme-client` and `cosme-server` (the shaped link's endpoints), `cosme-probe` (VNC
client, shares the client's network namespace), `cosme-backend`, `cosme-frontend`, `cosme-osrm`.
Backend on port 8731, dashboard on 8732; both published, so the dashboard is reachable from the LAN
with no firewall changes.

## Development

`backend/`, `models/` and `traces/` are bind-mounted into `cosme-backend`, so after editing backend
code `docker compose restart backend` is enough — no rebuild. Running uvicorn on the host also
works, but only with the containerised backend stopped first, since both bind 8731.

## Troubleshooting

**Dashboard says dry-run mode.** The backend falls back to a logging-only netem backend when it
can't see `cosme-client` and `cosme-server`. Check `docker ps`, then
`GET /api/scenarios/{id}/status` for `backend_mode`.

**VNC showcase fails after restarting containers.** `cosme-probe` shares `cosme-client`'s network
namespace, so it dies silently whenever `cosme-client` is recreated. Always follow a recreate with:
```bash
docker compose up -d --force-recreate probe
```
Compose has no way to chain this automatically; the container healthcheck surfaces it.

**Congestion control won't change.** Modern Docker mounts `/proc/sys` read-only inside containers,
so `docker exec ... sysctl -w` always fails, even with `CAP_SYS_ADMIN`. The backend works around
this with `nsenter` into the target's network namespace. This needs no setup in normal operation.
Only if you run the backend on the host as a non-root user do you need a sudoers rule:
```
# /etc/sudoers.d/cosme-nsenter-cc (chmod 440)
<user> ALL=(root) NOPASSWD: /usr/bin/nsenter -t [0-9]* -n /usr/sbin/sysctl -w net.ipv4.tcp_congestion_control=*
```

**BBR not offered.** Run `sudo modprobe tcp_bbr` on the host, then re-check
`GET /api/system/congestion-controls`.

**`prepare_osrm.sh` restarts a stage that already succeeded.** Its extract/partition/customize
stages are cached by file mtime. If you copied `docker/osrm-data/` from another machine, `touch` the
files back into extract < partition < customize order or it will redo the multi-hour rebuild.
`--force` rebuilds regardless.

**Custom route simulation fails with HTTP 406.** The public Overpass API rejects the default
`python-requests` User-Agent. `oblos_live.py` sends its own, so this only bites if you query
Overpass yourself. Results cache to `traces/oblos_sim_cache/`.

**Offline at the venue.** DWD weather, Nominatim geocoding and Overpass responses all cache to
`traces/`. Prime them once beforehand, or prefer a weather preset over "Real DWD" for a guaranteed
offline run.

**Video showcases receive no frames.** Debian's `python3-aiortc` 1.4.0 drops all received RTP when
header extensions are negotiated. `webrtc_peer.py` strips `a=extmap:` lines during signalling to
work around it; the workaround becomes unnecessary on aiortc ≥ 1.5.

## Known limitations

- Loss is emulated as outages, not as a per-packet drop rate. That is faithful to what the models
  describe, but it means loss-based congestion control is barely exercised, and COSME does not
  reproduce the TCP goodput deficit reported for real Starlink. Measured numbers and the proposed
  fix are in [docs/COMPOSITION.md](docs/COMPOSITION.md) §7.
- No rain↔obstruction or rain↔handover interaction is modelled; bandwidth applies independently.
  We have no data showing such a correlation, so modelling one would be inventing it (§4).
- Starlink's real drop-front queue management is not emulated; `netem`'s child qdisc is drop-tail
  (§6).

## Citation
If you use this work in your research, we would be happy for a citation using the following BibTeX:
```
@inbook{10.1145/3789240.3830288,
author = {Lanfer, Eric and Laniewski, Dominic and Zimmermann, Till and Aschenbruck, Nils},
title = {DEMO: COSME -- Composable Orchestrated Starlink Mobility Emulation},
year = {2026},
isbn = {9798400724671},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3789240.3830288},
abstract = {Low Earth Orbit (LEO) satellite links are pivotal for ubiquitous vehicle connectivity, yet their performance is impacted by a complex interplay of weather, constellation dynamics, and physical obstructions. While individual models for these impairments exist, protocol and application designers lack a unified tool to evaluate behavior under realistic, composed LEO conditions. We present COSME, a route-aware, real-time mobility emulator that integrates multiple impairment models - including obstruction-based loss, constellation-induced jitter, precipitation-driven bandwidth reduction, and packet loss at handovers - into a single framework. By orchestrating Linux network namespaces via tc and netem, COSME enables the high-fidelity playback of merged impairment traces. We demonstrate COSME through five diverse application showcases, highlighting the impact of different congestion control algorithms and transport protocols on LEO connectivity.},
booktitle = {Proceedings of the ACM SIGCOMM 2026 Conference},
pages = {2221–2223},
numpages = {3}
}
``` 

## AI Disclaimer
Generative AI Disclosure. Claude (Models: Sonnet 5 and Fable 5) was used to assist with the implementation of the demonstration software by generating and suggesting portions of the source code.
