# COSME Composition Rules

The paper states that the model outputs are "compounded into a single, unified multivariate time
series" without defining how. This document defines it, and answers the two questions that
description leaves open:

1. How are simultaneous obstruction loss and reconfiguration loss combined? (§2)
2. Is cross-impairment dependence modeled at all — rain interacting with obstruction, handover
   timing, or route geometry? (§3 for the one dependency that is, §4 for the ones that are not.)

The rules below are exactly what `backend/compose.py` implements. Nothing here is aspirational —
every rule cites the line(s) of code that enforce it.

## 1. Common time grid

All four models' outputs are resampled onto one shared grid (default `dt = 10ms`,
`backend/compose.py:DEFAULT_DT_S`). Binary/state series (loss flags) are expanded onto the grid by
interval membership; continuous series (rain rate) are linearly interpolated
(`compose.py: np.interp(...)`).

## 2. Loss combination: boolean OR, not additive probability

Obstruction loss (ObLoS) and reconfiguration loss (Zimmermann) are each represented as a boolean
"packet lost right now" signal over time, not a probability. They are combined with a logical
**OR**: a packet is lost if *either* cause is currently active.

```python
combined_loss = obstruction_loss | reconfig_loss   # compose.py
```

**Why OR and not addition**: these are two physically distinct loss mechanisms (a bridge blocking
line-of-sight vs. a satellite handover) acting on the same packet stream. Treating them as
additive probabilities (`p_obstruction + p_reconfig`) can exceed 100% and double-counts instants
where both happen to be active; OR is the correct combination for two binary "is loss happening"
indicators and never exceeds a valid loss state. It also means: **during a reconfiguration window
that overlaps an obstruction, the packet is still counted lost exactly once** — there is no
special-cased priority between the two, because from the wire's perspective there's no way to
distinguish "why" a packet did not arrive.

## 3. The modeled cross-impairment dependency: shared reconfiguration schedule

Is *any* cross-impairment dependence modeled? There is one concrete
answer in this implementation: **Garcia's reconfiguration flag (ρ) and Zimmermann's loss bursts are
driven by the same underlying event schedule**, not independently sampled.

Both effects are two observable consequences of the same physical event (a Starlink satellite
handover, occurring on a ~15s cadence per Garcia et al.). `backend/models/reconfig_schedule.py`
generates a single `ReconfigSchedule` (a list of `(start_s, duration_s)` handover windows, duration
sampled from Zimmermann's real empirical burst-length distribution). `compose.py` passes that one
schedule into `GarciaModel.generate(schedule=...)`, so a handover at t=45.0s produces *both* a
Garcia delay/jitter bump and a Zimmermann loss burst at the same instant, with a shared duration —
instead of each model flipping its own independent coin every 15 seconds.

```python
schedule = ReconfigSchedule(duration_s=duration_s, zimmermann_model=zimmermann_model, seed=seed)
garcia_trace = garcia_model.generate(duration_s=duration_s, dt_s=dt_s, schedule=schedule)
reconfig_loss = garcia_trace.reconfig_flag  # same schedule Zimmermann's bursts came from
```

## 4. What is *not* modeled as dependent (explicit limitation)

Taking the obvious candidates -- rain interacting with obstruction, handover timing, or route
geometry -- **no such interaction is modeled**. WetLinks' bandwidth factor is computed purely
from the rain-rate input series and applied as an independent multiplicative factor on nominal
throughput, with no coupling to the loss/delay layers:

```python
bw = wetlinks_model.generate(times, rain_at_grid)  # independent of obstruction_loss / reconfig_loss
```

This is a real, acknowledged simplification, not an oversight we're hiding: we have no measurement
data in this repo showing rain statistically correlated with obstruction geometry or handover
timing (they are physically different mechanisms — precipitation attenuation happens at the
dish-to-satellite RF link, obstruction loss happens from physical line-of-sight blockage, and
handover timing is a constellation-scheduling artifact), so modeling a dependency here would be
inventing a relationship, not deriving one. See `docs/VALIDATION.md` for what we *could* check with
the data on hand.

## 5. Playback simplification (bandwidth/delay update quantization)

The merged trace itself is continuous at the 10ms grid resolution. The paper's originally stated
design only issued new `tc`/`netem` parameter updates once per 15s reconfiguration slot ("adapting
the emulation only every 15 seconds for a short time interval for the reconfiguration"), but this
was superseded per an explicit user requirement to track the underlying model more closely: the
orchestrator (`backend/orchestrator.py`) now re-quantizes delay/jitter/bandwidth onto a
configurable `update_interval_s` grid (≤1s, `MAX_UPDATE_INTERVAL_S`, user-selectable down to
0.1s), plus one additional sharp update exactly at each reconfiguration burst's real start time so
the ~100-400ms handover disruption is never smoothed away by landing mid-bucket. Loss is applied at
its real event resolution regardless of `update_interval_s`. This is a playback-engine performance
simplification, not a composition rule — the underlying multivariate trace used for validation and
dashboard visualization is at full 10ms resolution regardless of playback granularity.

## 6. Known unmodeled effect: drop-front queue management

Garcia, Sundberg, and Brunstrom's follow-up measurement paper ("Characterizing the Configuration of
Starlink Queuing", IMC'26, arXiv:2605.27717 — a different paper from the OWD-characterization work
`garcia.py`/`garcia_fit.py` are fitted from) finds that Starlink's bottleneck queue uses **drop-front**
buffer management rather than the conventional drop-tail or per-flow fair queuing: incoming packets
are accepted and existing packets are dropped from the *front* of the queue once it's full. This
lowers average queuing delay, but can deliver an earlier/distorted congestion signal to loss-based
congestion control (Cubic, the Linux default) than drop-tail would, which the paper ties to
Starlink's previously-observed Cubic throughput degradation.

This is **not modeled here**. COSME's playback drives plain `tc`/`netem` (`backend/netem_backend.py`),
whose default underlying child qdisc behaves as drop-tail, not drop-front — under sustained
saturation (e.g. the file-transfer showcase's TCP Cubic ramp), loss dynamics may therefore differ
from a real Starlink terminal's. Linux's `pfifo_head_drop` qdisc could emulate drop-front in
principle, but wiring it in cleanly conflicts with how `netem` updates are currently issued (`tc
qdisc replace ... root netem ...`, up to 10Hz — a `replace` rebuilds the whole qdisc tree, so a
child qdisc would need re-attaching, or the update path changed to `tc qdisc change`, on every
single update). Flagged here as a real, evidence-based limitation and candidate future work, not
implemented in this pass.

## 7. Known limitation: loss is emulated as outages, not as a per-packet drop rate

Both loss models emit a **boolean** "is loss happening right now" signal (rule 2), so the playback
engine gates `netem` at `loss 100%` for the burst's duration and `0%` otherwise
(`orchestrator.py:_loss_event_updates`). That is physically right for what the models describe --
a bridge blocking line-of-sight, or a handover dropping a burst, are outages, not a dice roll per
packet. But it means the *average* loss rate a transport sees is delivered in a very different
shape from the per-packet loss those transports are usually characterised against, and for
loss-based congestion control the difference is enormous.

Measured on this stack (1 GB/200 MB HTTP download, 329/30 Mbit/s, 24 ms RTT, otherwise identical):

| condition (same 1.4% average loss) | Cubic | BBR |
|---|---|---|
| no loss | 304.9 Mbit/s | 304.5 Mbit/s |
| **uniform random** 1.4% (`netem loss 1.4%`) | **6.0 Mbit/s** | 298.8 Mbit/s |
| **COSME bursts** (the composed arm, ~1.4% duty cycle over the transfer window) | **301 Mbit/s** | 291 Mbit/s |

Cubic collapses by ~51x under random loss and is essentially unaffected by ours, because it
recovers from a short total outage at full congestion window while sustained random drops keep it
in repeated multiplicative decrease (the Mathis bound puts 1.4% at 24 ms RTT in the single-digit
Mbit/s range -- which is what we measure). BBR, being model-based rather than loss-based, is
insensitive either way.

Two consequences worth stating plainly:

- **COSME does not currently reproduce the TCP goodput deficit reported for real Starlink.** UDP
  through the same shaping delivers ~315 Mbit/s and TCP reaches ~302, i.e. ~96% of the ceiling,
  where real measurements commonly show TCP at roughly half of the UDP ceiling. The likely causes
  are all things this emulator does not model: continuously varying capacity, the real terminal's
  queue behaviour (see section 6's drop-front finding), and per-packet loss alongside the outages.
- **Do not read a null result on Cubic here as evidence that CC choice does not matter on LEO
  links.** It is evidence that this loss model does not exercise loss-based CC.

The obvious next step is a hybrid loss signal -- the modelled outages *plus* a residual per-packet
drop rate fitted from the same measurement traces -- which would let the emulator reproduce both
the burst structure and the steady-state penalty. Not implemented; flagged here rather than left
to be found.
