# Results — the head-to-head, as measured

Measured 8–9 Sep 2026 on the dev box (8 × RTX A5000, 48 host cores), five GPUs per arm,
other tenants resident throughout. Every number here came out of a run; nothing is scaled,
extrapolated or asserted from configuration except where the baseline itself does so and it
says which. `README.md` says how to reproduce them; this file says what came back.

**Supersedes nothing outside itself.** When a run disagrees with a number here, replace the
number and date it.

## The two arms

| | baseline | ShipInfer |
|---|---|---|
| binary | `benchmarks/baseline` (`sim_pipeline_v2`), unmodified | `csrc/build/bench`, `--source nvdec` |
| input | a folder of JPEGs, `cv::imread` per frame | RTSP → NVDEC → NV12 in VRAM |
| chain | two disjoint one-model pipelines (`det`, `seg`) | detect → conditional segment → embed person → embed ship → reassembly → JSON events |
| models per frame | 1 | 2.65 measured (106 294 requests / 40 148 frames, below) |
| what it reports | images/s, **asserted** from its configuration minus buffer growth | events/s and rows/s, **counted** |

The asymmetry is not hidden: one baseline image passes through one model, one ShipInfer frame
passes through the chain, so equal frames per second is strictly more work on our side. It
also cuts the other way — the baseline decodes on the host per frame and still retired 960
img/s.

## The like-for-like pair

Both arms on GPUs 0/1/3/4/6, 50 cameras × 20 fps × 70 s, the same busy box:

| | throughput | per GPU | rows into a model |
|---|---|---|---|
| baseline | **960.2 img/s** (saturated: det 485.8 + seg 474.4) | 192.0 rows/s | 1 row per image |
| ShipInfer | **573 events/s** (40 131 events, 0 failed) | 114.7 events/s | **6 935 rows/s** (1 387/GPU) |

The baseline is insensitive to which five GPUs it gets — 959.8 on GPUs 2–6 against 960.2
here, 0.04% apart — because at saturation it is bound by its engines. So it is a **capacity**,
not a floor.

Our stage counters for the same run, which is where the rows come from:

| model | requests | rows | rows/request | input |
|---|---|---|---|---|
| `ship_detector` | 40 148 | 40 148 | 1.0 | 640×640 |
| `ship_segmenter` | 19 805 | 54 375 | 2.7 | 640×640 |
| `person_embedder` | 26 536 | 336 569 | 12.7 | 256×128 |
| `ship_embedder` | 19 805 | 54 375 | 2.7 | 256×128 |
| total | 106 294 | 485 467 | | |

`ship_detector`'s rows equal its requests exactly, on all five devices, because one frame is
one row — so the counter is counting rows and not re-reporting requests.

## The four ratios, and they are four different claims

| ratio | value | what it claims, and what it ignores |
|---|---|---|
| frames end to end | **0.60×** | The softest. A CPU-bound stage moves it, and both runs were on a box at 25/48 cores. |
| pixels into a model | **1.87×** | An **area** proxy, not work: it treats a 640×640 detector row and a 256×128 crop as 12.5:1 and ignores that their FLOPs per pixel differ too. |
| rows into a model | **7.22×** | Counts a crop and a frame alike, and 12.7 of our rows per request are crops. |
| rows per host CPU-second | **~3.4× default, 7.2× with the knob** | The only one with a **like-for-like denominator** — the same kernel counter on both arms. A **floor** (see below). The two figures are one env var apart, from a later sitting whose control reproduces the 3.94× the section below derives — see *The one knob that moves it*. |

Corroborated on a second five-GPU set: 7.7× rows and 2.03× pixels on GPUs 2/3/6. Same
ordering, same conclusion, so the spread between the weightings is a property of the workload
and not of one run's silicon.

**Why rows-per-CPU-second is a floor.** Three interleaved pairs, 50×20×40 s on GPUs 1/3/4/5/6:

```
pass   our rows/CPU-s   baseline rows/CPU-s   ratio
a           358.7               87.2          4.11x
b           379.9              113.0          3.36x
c           367.2               84.2          4.36x
            mean 3.94x, range 3.36-4.36 (25.3% of the mean) -> ~4x, one digit
```

The spread is almost all on the baseline's side — 34% against our 5.8% — because its reported
throughput is asserted from its configuration while its CPU-seconds are measured. When the box
starves it of CPU its CPU-seconds fall and its images do not, which **inflates** its figure
exactly when the box is busy. Pass b, the busiest, is both the baseline's best and the ratio's
worst.

Per-unit host cost, from the **first (non-interleaved) sitting** — 50×20×40 s on five idle
GPUs, which is where the 313.6-against-82.5 rows-per-CPU-second figure comes from:

| | host CPU per unit |
|---|---|
| ShipInfer | 3.19 ms per row |
| baseline | 12.12 ms per image |

## The one knob that moves it, and it roughly doubles the ratio

`SHIPINFER_CUDA_BLOCKING_SYNC=1` asks the driver for `cudaDeviceScheduleBlockingSync` instead
of the `cudaDeviceScheduleAuto` default, which **spins** while a synchronise waits when the
number of contexts is at or below the core count. It is **off by default**. Nine runs, three
passes, arm order rotated between passes so a drift inside a pass cannot look like the knob —
50 cameras × 20 fps × 40 s, `--source nvdec`, GPUs 1/3/4/5/6, all nine `exit=0`:

| pass | baseline | ours, knob off | ours, knob on | off ratio | on ratio | the knob |
|---|---|---|---|---|---|---|
| a | 83.0 | 294.5 | 670.5 | 3.55× | 8.08× | 2.28× |
| b | 97.3 | 263.7 | 666.7 | 2.71× | 6.85× | 2.53× |
| c | 99.6 | 396.3 | 656.6 | 3.98× | 6.59× | 1.66× |
| **mean** | | | | **3.41×** | **7.17×** | **2.15×** |

Rows per host CPU-second in every column. **The control reproduces the sitting above** —
flag-off means 3.41× (2.71–3.98) against the 3.94× (3.36–4.36) measured a day earlier, and its
baseline column lands at 83.0/97.3/99.6 against that sitting's 87.2/113.0/84.2 — which is what
makes the flag-on column readable rather than a number from nowhere.

**It moves both terms and they compound**: host CPU 704.2 → 422.6 CPU-s (−40%) and rows
224 871 → 280 848 (+25%).

**The flag-on arm is the stable one, and that is the most telling figure here.** Its CPU
spread is 2% (419.8/428.0/419.9) against the flag-off arm's 23%, its rows 3% against 51%, and
its accepted frames 4% (22 753–23 745) against 51% (13 691–23 264). A spinning wait costs
whatever contention there is to lose, so the default arm is partly a measurement of the box's
other tenants while the knob-on arm is a measurement of the work.

**The latency half of the trade does not appear — and one pass inverts, which is how it has to
be said.** `cli/bench` prints no percentiles, but it counts `collector_timeouts`, a stage that
did not answer in time: 67/148/26 with the knob off against 10/76/72 with it on. Lower on
average, and pass c goes the other way, so at n=3 this supports "no evidence of a penalty",
not "the latency improves". What is unambiguous is shedding: **every** knob-on arm drops fewer
frames than its own pass's knob-off arm.

## Method, because one run decides nothing here

- **The noise floor is ~15%.** Four runs of one arm at identical settings spread 26 669 to
  39 375 events — 36.7% of the mean; dropping the earliest still leaves 14.9%.
- **So the arms are interleaved** (baseline, ours, baseline, ours, …) and the **pairwise
  ratios** are quoted rather than the means. A worker-count sweep looked like a 6% win on its
  first pass and lost on both repeats; the means hid the reversal.
- **Rotating the arm order is the same argument one level up.** With three arms per pass
  (baseline, knob off, knob on), always running one of them last would let a drift inside a
  pass read as that arm's property. The order is base/off/on, then on/off/base, then
  base/on/off.
- **A simultaneous pair in one log is not achievable on this box**, tried three ways: at 12×10
  and 8×5 the baseline's concurrent load starved our in-process generator below the offer
  gate; at a load small enough to avoid that, the baseline logs too few samples to bound a
  growth rate; `--topology fleet` at 50×20 failed all five shards the same way. The harness
  says why in its own words — "run one at a time to keep the GPUs uncontended".

## What is not in any number above

- **Nothing in the chain. Both exclusions are gone, and the page owes the numbers instead.**
  `track` landed 10 Sep and `mtmc` on 11 Sep, so a run of the `ship_person_cpu` chain now
  reports **7 stages, "not run here: decode output"** where it once reported 5 and named both
  among them. The binary's stamped note is derived from that list rather than asserting an
  exclusion, so it says "fused kernels are NOT in this measurement" and adds ", and neither is
  tracking" only when a run really had none. **The whole-chain measurement is its own section
  below**, because it changed what the per-worker numbers above mean.
  **What that costs, and the first answer was wrong.** The lane-in/lane-out pair at 8 cameras ×
  10 fps × 20 s on two GPUs read 1 595 complete events tracked against 1 600 untracked, and
  this page called it 0.3%, inside the noise floor above. It was not noise: it was a defect in
  the stage, which let the tracker's ordering refusal fail the stage instead of publishing the
  frame untracked, so the frame never completed. Three tracked runs lost 5, 12 and 17 events;
  after the fix two runs completed **1 598/1 598 and 1 600/1 600**, with the reordering named
  instead — `track_frames_untracked track 2` and `track 1`. So the stage's throughput cost at
  that load is **not measurable here**, and one or two frames per 1 600 really are reordered by
  the worker pool. That is not a claim about the design load, where the stage runs 50 times
  more often.
  Every ratio above still predates the stage, so they are measured on a chain one stage shorter
  than the one that now ships — in the direction that understates us.
- **The fused kernels.** `ldd csrc/build/bench` links no shipvision library. (The `shipvision`
  *lane* is in the binary now — the cross-camera tracker is compiled from that submodule's C++
  — but the fused IMAGE kernels still are not: the letterbox and the crops are ours.)
- **A baseline that reads video.** V156's fairness condition — the same bench for both arms,
  video in and targets out — cannot be met by this baseline: its input is a folder of JPEGs,
  it has no decoder and no RTSP, and the submodule is read-only by design. Our side does meet
  it (`--source nvdec` is RTSP → NV12 → VRAM end to end), which is why the two arms are
  compared on work retired rather than on the same input.
- **The previous system** (`references/`, the one this project replaces) is the only candidate
  that could be given video, and it is not runnable here: no image, no weights, and the image
  cannot be built on this kernel. It needs artefacts, not a measurement.

## The whole chain, end to end, and the number that is not the throughput

`decode → detect → crop → segment → embed_person → embed_ship → track → mtmc`, over **gstreamer
RTSP** from the offline H.264 (`--source nvdec`, which is RTSP → NV12 → VRAM with no host
round trip), 4 GPUs (0/2/5/6), 12 cameras × 200 fps × 40 s, fp16, `shipinfer-gst:jammy-nvdec`.
One variable: `workers`.

| workers | accepted img/s | untracked | **tracked img/s** | complete / incomplete |
|---|---|---|---|---|
| 24 | 265.8 | 188 (1.8%) | **261.1** | 10 633 / 0 |
| 48 | 331.4 | 2 679 (20.2%) | **264.4** | 13 256 / 0 |
| 92 | 442.6 | 7 409 (41.9%) | **257.3** | 17 662 / 40 |

**The tracked rate is flat.** Accepted frames rise 1.67× across that range and the frames that
leave with track ids do not move: every worker past ~24 buys a frame the tracker refused, and a
frame with no ids is one `mtmc` cannot associate. So **~260 img/s** is this chain's answer on
four A5000s, and the "throughput scales with workers" reading of the numbers further up this
page was counting refusals.

**Why, and it is not a defect.** One shared worker pool reorders a camera's frames, and a
per-camera tracker refuses a frame that does not advance its own stream
(`graph/stages.cpp`, and `topology/elements/track.py` catches the same refusal). More workers,
more reordering, more refusals. `track_frames_untracked` is the counter that makes it visible,
and it exists because #215's review found the stage FAILING on that refusal instead of
publishing the frame untracked.

**Neither the host nor the engines is the wall there.** At 92 workers the run used 12.4 of 48
cores (26%), and the four models' busy percentages sum to ~430% of the 800% that eight instances
per device could use. What binds is the barrier's latency trade and the ordering above.

**AND THE CHAIN ISSUES NO GLOBAL IDS AT THE REFERENCE'S DEFAULTS, which the throughput table
cannot show.** Now that a chain can state the gate (#225), its two halves can be separated —
and the first reading of these counters, which blamed the footage, was wrong. Nine arms, one
variable each: 12 cameras × 20 fps × 40 s, GPUs 0/2/5/6, 24 workers (the regime where the
per-camera tracker refuses almost nothing — 13 to 72 late frames of ~9 540 — so the gate is the
only thing under test), `--source nvdec` over gstreamer RTSP from the offline H.264.

| min_hits | min_height_fraction | window | roster | offered | admitted | ids / tracks | instants `complete` |
|---|---|---|---|---|---|---|---|
| **3** (default) | **1/9** (default) | 60 | 4 declared | 3 868 | **0** | 0 / 0 | 0 of 890 |
| 2 | 1/9 | 60 | 4 declared | 3 739 | 969 (25.9%) | 10 / 23 | 0 of 910 |
| 1 | 1/9 | 60 | 4 declared | 3 680 | **3 680 (100%)** | 17 / 52 | 0 of 894 |
| 3 | 0.02 | 60 | 4 declared | 3 507 | **0** | 0 / 0 | 0 of 909 |
| 1 | 0.02 | 60 | 4 declared | 3 317 | 3 317 (100%) | 21 / 60 | 0 of 869 |
| 3 | 1/9 | **200** | 4 declared | 3 419 | **0** | 0 / 0 | 0 of 918 |
| 3 | 1/9 | 60 | **the run's 12** | 3 480 | **0** | 0 / 0 | **421** of 905 |
| 2 | 1/9 | 60 | the run's 12 | 3 330 | 847 (25.4%) | 6 / 20 | 487 of 881 |
| 3 | 1/9 | 200 | the run's 12 | 4 211 | **0** | 0 / 0 | 449 of 850 |

**The height floor excludes nothing.** At `min_hits 1` the admitted count equals the offered
count *exactly* — 3 680 of 3 680, and 3 317 of 3 317 — so every box the tracker produced already
clears 120 px of a 1080-tall frame at the reference's own `1/9`, which is
["roughly the smallest crop its re-ID model was trained to handle"](../3rdparty/shipvision/shipvision/mtmc/gating.py).
Lowering **only** the floor admits nothing. The earlier diagnosis lowered both thresholds in one
step and concluded the footage was too small; it is not, and
`BENCH-FOOTAGE-IS-BELOW-THE-MTMC-GATE` is closed as refuted rather than fixed.

**What closes the gate is the AGE test.** `min_hits` counts *consecutive* qualifying instants and
the hit map is replaced each instant, so a track absent from one instant starts over: 3 admits
nothing, 2 admits a quarter, 1 admits everything. It is not the window — 200 ms moves the closes
from `window` to `advanced` and still admits 0 — and it is not the roster.

**The roster is wrong too, and separately.** `topology/ship_person_cpu.yaml` declares
`cameras: [cam-01 … cam-04]` while the bench fleet is `cam00 … cam11`, so the barrier waited for
four cameras that never connect: **not one instant closed `complete`** until the roster named the
run's own cameras, and then 421 of 905 did. It moves the gate's arithmetic not at all (25.4%
against 25.9% at `min_hits 2`), which is why it is a separate defect and not this one's cause.

**And the gate sees a third of a frame's rows.** 3 480 observations from 9 538 frames is 0.36 per
frame, while the two embedders processed 77 777 person crops and 13 124 ship crops — about 9.5
embedded rows per frame. An observation needs a track id **and** an embedding
(`graph/stages.cpp`), and a row only carries an id once the tracker has CONFIRMED it. That is
the thread the next section pulls, and it is the answer to this whole page's mtmc puzzle.

So the chain is proven end to end, the association runs on real observations when the gate lets
it, and the roster finding is filed with its numbers rather than paraphrased:
`MTMC-ROSTER-NAMES-NO-CAMERA-A-RUN-HAS`.

## Footage a tracker can follow, and the chain's first global ids at the defaults it ships

Every measurement above replays **ten unrelated photographs** at 20 fps: `scripts/rtsp_serve.py`
encodes a directory of JPEGs into one H.264 stream and the server loops it, so a camera's scene
changes completely every 50 ms. A per-camera tracker confirms a track by agreeing with itself
across frames, and nothing in that input agrees with anything — which is why a row so rarely
carried an id, and why an age gate counting *consecutive* instants could never be satisfied.

`benchmarks/harness/pan.py` builds the other kind of input from the same real photographs: a
1080p window panned across one 4K frame on a path that closes on itself, ~15 px per step, so
consecutive frames hold the same people a few pixels along. Generated from real data and
deterministic, which is the rule `crowd.py` states. The same rig as the table above — 12 cameras
× 20 fps × 40 s, GPUs 0/2/5/6, 24 workers, `--source nvdec` over gstreamer RTSP — at the
reference's **production defaults**, `min_hits 3` and `min_height_fraction 1/9`:

| fixture | roster | observations/frame | offered | admitted | ids / tracks |
|---|---|---|---|---|---|
| ten photographs | 4 declared | 0.41 | 3 868 | **0** | 0 / 0 |
| ten photographs | the run's 12 | 0.36 | 3 480 | **0** | 0 / 0 |
| **the pan** | 4 declared | **4.42** | 42 063 | **26 023 (61.9%)** | **13 / 72** |
| **the pan** | the run's 12 | **4.41** | 41 951 | **24 177 (57.6%)** | **12 / 72** |

**The chain associates across cameras at the configuration the deployment ships.** Nothing was
tuned for these rows: the gate is the reference's own, the window is the chain's 60 ms, and the
only thing that changed is that the camera now shows a scene instead of a slideshow. Ten times
the observations reach the gate, three fifths of them qualify, and 72 per-camera tracks resolve
to a dozen global identities.

**And the cost appears where the old runs could not see it.** With the gate admitting, the same
40 s leaves **1 483 of 9 520 events incomplete** (15.6%) against **zero** on the old fixture at
the same reassembly window — the barrier and the clusterer are doing per-instant work now, and
the reassembly window has not been re-chosen since they started
(`MTMC-REAL-WORK-COSTS-A-SIXTH-OF-THE-EVENTS`). The throughput table further up this page was
measured with `mtmc` admitting nothing, so it is a floor for this chain rather than its rate.


The fix for the *throughput* half is placement **affinity**, not more threads — a camera's frames reaching one worker, or
a per-camera sequencer in front of the tracker — and it trades load balance for ordering, which
is the trade this project exists to get right. Priced and open as
`PIPELINE-WORKERS-NEED-CAMERA-AFFINITY`.

## The verdict, and the one open question

The ≥5× target needs a ratio to be against, and the four above give opposite answers. Absent
a decision the number reported is **rows per host CPU-second** — chosen because it is the only
ratio measured the same way on both arms, because a resource ratio is what "5×" ought to mean,
and because it is a floor that errs in the baseline's favour. It is deliberately not the 7.22×
rows figure, which clears the target by counting a 256×128 crop as one 640×640 frame.

On that ratio the answer now depends on one env var: **~3.4× as shipped, so NOT MET at the
default; 7.17× with `SHIPINFER_CUDA_BLOCKING_SYNC=1`, which clears it.** Both are measured, in
one sitting, with the control reproducing the previous one. Whether the knob should be the
default is a separate decision and not this page's to make: what argues for it is −40% host
CPU, +25% rows, fewer drops in every pass and no visible latency cost; what argues against is
that `cli/bench` reports no latency percentiles, so the cost the knob is documented to trade
for has never been measured directly — only its proxy.

The headroom is our own host cost: 3.19 ms of CPU per row while our arm was host-bound and the
baseline was saturated. `.claude/TASKS.md` holds the accounting under
`NOT-GPU-BOUND-AT-FIVE-GPUS` and the decision under `C1-WHAT-IS-THE-5x-AGAINST?`.
