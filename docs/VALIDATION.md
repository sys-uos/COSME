# COSME Validation

Does composing the models produce a trace closer to reality than any single model alone? Numbers
below are computed by `backend/validation.py` (`python -m backend.validation`) against 11 real,
held-out `clipped_measurements` drives that were **never used to fit** the Zimmermann
reconfiguration-loss model — not simulated, not cherry-picked.

## Method

`backend/models/zimmermann.py` splits the 57 usable measurement drives 80/20 (`seed=1337`, split
by drive so no drive contributes to both train and test). The 20% holdout set (11 drives) is used
here. For each holdout drive:

- **Real, fully-measured loss trace** (`ping_results_FHP_loss_trace.csv`) is the ground truth — it
  contains both obstruction- and reconfiguration-induced losses, as actually measured on a real
  Osnabrück↔Hannover drive.
- **Obstruction-only** = the real, measured obstacle-only trace for that same drive
  (`..._loss_trace_filtered_reconfig.csv`) — a genuine single-impairment baseline, not synthetic.
- **Composed** = that same real obstruction trace, OR'd (per `docs/COMPOSITION.md` rule 2) with a
  *synthetic* reconfiguration-loss schedule sampled from the train-only Zimmermann distribution
  (random 15s-cadence phase, since absolute handover phase isn't recoverable from these
  relative-timestamped drives — an explicit, stated limitation, not a hidden one).

## Result 1 — composing fixes the systematic event-undercount

| | mean event-count ratio (1.0 = matches real) |
|---|---|
| Obstruction-only | **0.904** (undercounts events by ~10%, consistently, in all 11/11 drives) |
| Composed | **1.040** (within ~4% of real, both directions) |

A single-impairment model *systematically*
undercounts real loss events on every held-out drive, because it has no notion of
reconfiguration-induced bursts at all. Adding the (independently-derived) Zimmermann component
closes that gap. See `traces/validation/example_drive_comparison.png` for a representative 300s
window (drive `measurement-2025-04-03_16-58-multicar-onlyping`) — the composed row visibly
recovers loss events the obstruction-only row misses, both quantitatively and to the eye.

## Result 2 — instant-level alignment is worse for composed, and we know why

| | Hamming distance (lower=better) | DTW / sample (lower=better) |
|---|---|---|
| Obstruction-only | 0.0084 | 0.0275 |
| Composed | 0.0157 | 0.0423 |

Composed is *worse* on both instant-aligned metrics. This is expected, not swept under the rug:
our synthetic reconfiguration schedule places bursts at a randomized 15s-cadence phase because
**we have no way to recover a drive's absolute handover phase from these relative-timestamped
measurements** (see `docs/COMPOSITION.md` §3 and `reconfig_schedule.py`'s module docstring). So
composed correctly adds *approximately the right number* of extra loss events (Result 1), landing
at *approximately the right rate*, but not always at the *same instants* as the drive's real
handovers — which briefly desynchronizes the composed signal from the real one at 10ms
resolution, at exactly the moments a synthetic burst doesn't overlap a real one. This is the single
biggest concrete way our model is *not* yet accurate, and the concrete next step for closing it is
recovering absolute handover phase (e.g. from Starlink dish telemetry, if available) rather than
assuming a random or zero phase offset.

## Result 3 — does the burst-duration distribution generalize to unseen drives?

A Kolmogorov-Smirnov test comparing each holdout drive's own real reconfiguration-burst durations
against the pooled, train-only fitted distribution: **mean p-value 0.0133 across 11 drives**, with
individual drives ranging from p≈0.075 (statistically indistinguishable) down to p≈5e-13 (clearly
different). In plain terms: **the pooled empirical distribution is a reasonable but imperfect fit
per-drive** — several drives show burst-length distributions that differ significantly from the
pooled training set, suggesting per-session or per-weather-condition variation in handover
disruption length that a single pooled distribution doesn't fully capture. This is exactly the kind
of explanation for ObLoS's own reported loss-event-length
inaccuracy (p50 3.35× overestimate in the ObLoS paper) — we don't have a fix for it here, but we
measure and report it rather than asserting accuracy without evidence.

## Result 4 — Garcia's delay/jitter, fitted and validated against real RTT measurements

`garcia.py`'s delay/jitter parameters are no longer invented: `backend/models/garcia_fit.py` fits
them from Zimmermann's own `ping_results_FHP.csv` per-packet RTT measurements (RTT/2 as an OWD
approximation), with reconfiguration windows identified from the CSVs' real *absolute wall-clock*
timestamps against the ":12/:27/:42/:57 of the minute" handover cadence `garcia.py`'s docstring
already claimed — an independent (RTT-based, not loss-based) confirmation of the same 15s cadence
Zimmermann's own loss-burst fit relies on: binning real OWD by phase shows a clear elevation in the
~1s window straddling each mark (e.g. a 27ms baseline jumping to 30-31ms).

`backend/validation.py`'s `run_delay_validation()` (folded into the same `run_validation()` output)
KS-tests each of 11 held-out drives' own real per-packet OWD against i.i.d. draws from the fitted
normal-state and reconfig-branch distributions:

| | mean |error| on branch mean (ms) | mean KS p-value |
|---|---|---|
| Normal-state | 0.58 | ≈0.0000 |
| Reconfig | 0.63 | ≈0.0000 |

Read honestly: the branch **means** match closely (sub-1ms), but the full-distribution KS test
formally rejects on every drive. That's expected, not evidence the fit is unusable — each holdout
drive supplies 100k+ i.i.d. per-packet samples, and a KS test's power to detect even tiny real
distributional differences (a slightly heavier tail, a mode weight off by a percent) grows with
sample size; at this n, "reject" is close to guaranteed even for a good fit. The means being tight
is the more informative number here.

The fitted numbers themselves show a real but modest handover effect: `delta_floor_ms≈7.8`, normal
modes at ≈2.9/9.1/24.3ms (weights ≈0.62/0.28/0.10), reconfig branch mean≈8.3ms/sigma≈8.8ms — i.e. a
few-ms aggregate elevation during reconfiguration, honestly smaller than the paper's headline
"100-400ms disruption" figure (which describes packet-level disruption/gaps, a different and more
specific effect than a smoothed RTT elevation averaged over a ~1s window).

## What this validation does *not* cover (explicit scope limits)

- **Weather is not part of this loss-signal validation.** WetLinks affects *bandwidth*, not the
  binary loss signal validated above, so co-locating real DWD rain data (see `dwd_weather.py`,
  confirmed working against `api.brightsky.dev`, e.g. drive
  `measurement-2025-07-22_08-29-multicar-onlyping` saw a real 19.1mm/h peak) demonstrates the
  *data pipeline* for simultaneous multi-impairment composition, but we do not have simultaneous
  application-level throughput ground truth for these specific drives to validate the WetLinks
  bandwidth factor against on-route (the WetLinks factors themselves are separately validated
  against their own source measurements — see `backend/models/wetlinks.py`'s docstring).
- **Garcia's normal/reconfig split still can't be decomposed further than this from RTT-only
  data** — the paper's own delta_SLsatVar/delta_sched and delta_Base/delta_BaseVar/delta_SLfix
  sub-splits aren't separately identifiable from RTT alone (see `garcia_fit.py`'s module
  docstring), so Result 4 above validates the two branches COSME actually models (normal vs.
  reconfig), not each of the paper's finer-grained named terms individually.
- **The live ad-hoc route simulator (`backend/models/oblos_live.py`, "Simulate route" in the
  dashboard) only predicts loss from cataloged BASt/OSM bridge and tunnel crossings** — by design,
  matching the original ObLoS website exactly, since bridges/tunnels are the only obstruction
  source that's actually mappable in advance for a route nobody has driven yet. It does NOT model
  trees, buildings, terrain, or any other real-world obstruction source. Confirmed by direct
  comparison: a real 116km route (Gescher -> Wallenhorst, crossing the A1/A30/A31) predicts ~0.5
  obstruction events/km, while the real recorded drives used everywhere else in this repo (real
  measured GNSS/Starlink loss, obstruction-only, all real-world causes included) show 10-90
  events/km. This gap is expected and does not indicate under-detection: diagnostics on that same
  route confirmed Overpass found all 72 candidate bridge crossings near the route, all resolved
  correctly against `obstacle_data.json`, and 61 survived the BASt exclusive per-bridge dedup (the
  other 11 were legitimate duplicate carriageway pairs of the same physical bridge). The live
  simulator is a demonstration of the PREDICTIVE half of ObLoS (plan a route, get a *plausible*
  bridge-driven loss trace); the pre-recorded `clipped_measurements` drives used for the demo's
  default scenarios and for all of this document's validation are the GROUND-TRUTH half (real
  measured loss, all causes). Don't compare the two densities directly.
- Reproduce with `python -m backend.validation`; raw per-drive numbers are in
  `traces/validation/results.json`.
