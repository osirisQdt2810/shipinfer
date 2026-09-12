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

**What the better fixture costs, on the arm that pays for it.** `--source replay` decodes the
whole folder into page-locked host memory once, so 400 frames of 1920×1080 are ~2.5 GiB a
library where ten photographs were ~62 MB. Measured on the default entry point (12 cameras ×
20 fps × 20 s, the bench process's own peak RSS): **5.62 GiB against 1.04 GiB**, +4.6 GiB, with
startup 2.23 s against 2.07 s and the same frames accepted (4 800 / 4 795). The RTSP arm —
every table on this page — encodes once and streams, and pays neither. On a shared box that is
the number to know before running the replay arm; `ReplayLibrary` takes a frame limit if a
smaller lap will do.

**The chain associates across cameras at the configuration the deployment ships.** Nothing was
tuned for these rows: the gate is the reference's own, the window is the chain's 60 ms, and the
only thing that changed is that the camera now shows a scene instead of a slideshow. Ten times
the observations reach the gate, three fifths of them qualify, and 72 per-camera tracks resolve
to a dozen global identities.

**And one number here was a counting artefact, corrected.** These runs reported *1 483 of 9 520
events incomplete* (15.6%) against zero on the old fixture, and this page blamed the barrier's
per-instant work against a reassembly window nobody had re-chosen. It was neither.
`collector_timeouts` was 0 the whole time, and a counter for **which** stage never answered
(`events_missing_stage`) named `crop` for every one of them: the run opened each frame expecting
`crop` unconditionally, `Dag::runnable` requires a stage's inputs to be non-empty, and a frame
the detector found nothing in therefore never makes `crop` runnable. With `crop` off that list
the same rig at 30 s answers `events_complete 7 118` of 7 118 accepted frames,
`events_incomplete 0` — **nothing was ever lost**.
The throughput table further up this page is still a floor rather than a rate, because it was
measured with `mtmc` admitting nothing.

### The design load, on four of the sixteen GPUs

50 cameras × 20 fps — the sizing every decision in this repository is arranged around — on the
pan fixture, GPUs 0/2/5/6, `--source nvdec` over gstreamer RTSP from the offline H.264, 92
workers, 40 s. The `mtmc` group stays twelve cameras, because a group is an atomic unit of
placement and the waiter budget has to cover it.

| offered | accepted | refused at the queue | untracked | **tracked img/s** | host cores | frame p50 / p95 |
|---|---|---|---|---|---|---|
| 954.5 img/s | 739.1 img/s | 23% | 3.7% | **711.5** | 15.5 of 48 (+1.0 for the RTSP servers) | 294 ms / 1.12 s |

**711.5 tracked img/s on four A5000s**, and TWO things moved between that and the ~260 this
page used to report, so the page separates them rather than crediting one:

| same four GPUs, same 92 workers | fixture | load | tracked img/s |
|---|---|---|---|
| the flat-rate table above | slideshow | 12 × 200 fps | 257.3 |
| the saturation table below | **pan** | 12 × 200 fps | 544.4 |
| this row | **pan** | **50 × 20 fps** | **711.5** |

**The fixture is worth 2.1×** (257.3 → 544.4, footage the only variable) and the remaining 1.3×
is the load SHAPE: at 200 fps per camera the tracker still refuses 37.3% of frames on the same
pan fixture, which is the reordering ceiling below rather than anything about the footage. The
fleet delivered 95% of the 1 000 img/s it was asked for; a quarter of that is refused at the
pipeline queue, evenly, and what gets through is tracked.

The run's own counters, so the row is evidence rather than a summary of it:

```
frames_read 38182   frames_accepted 29565   queue_rejected 8645
track_frames_untracked track 1105
events_complete 29565   events_incomplete 0
mtmc_observations mtmc offered 56050   admitted 170
mtmc_identities mtmc 0 0
frame_us_p50 294454   frame_us_p95 1118620
```

**What decides the 3 000 target is the host, not the devices.** 15.5 cores for 739 img/s is
21 ms of CPU per image: 63 cores at 3 000, and this box has 48. Two measured levers bring that
under the line — the blocking-sync default (−39% host CPU, #214) and the mask fold on the device
(1.44 ms of CPU per crop today, 10 µs as a kernel) — and four devices at this rate extrapolate to
~2 850 on sixteen. So the figure to quote **at the design rate** is **711.5 tracked img/s per
four A5000s**, with the host budget as the thing that has to be fixed for the full box to reach
the target — and it is not the same number as the saturation ceiling below, which is what this
chain does when it is pushed to ten times that rate.

**Identity does not survive this load, and that is a separate ceiling.** 170 of 56 050
observations admitted, zero global ids: a quarter of each camera's frames are refused at the
queue, and `min_hits` counts *consecutive* instants. Frames, yes; identities, not yet — the same
mechanism as the saturation table below, and `PIPELINE-WORKERS-NEED-CAMERA-AFFINITY`'s territory.

### Pushed to saturation, the useful rate is not the retired rate

The rows above run the design rate — 20 fps per camera, which twelve cameras deliver and this
chain keeps up with at 24 workers. Pushed to **200 fps per camera**, the load the flat-rate
finding used, on the same pan fixture, the same four A5000s (0/2/5/6), `--source nvdec`, and the
same chain with the roster naming the run's own twelve cameras — so the comparison row below is
the 57.6% arm, not the 4-camera-roster one — with workers as the only variable:

| workers | retired img/s | untracked | **tracked img/s** | mtmc admitted | ids / tracks | events incomplete † |
|---|---|---|---|---|---|---|
| 24 | 399.7 | 447 (2.8%) | **388.5** | 42 of 25 282 (0.17%) | 1 / 1 | 18.3% † |
| 48 | 659.1 | 4 262 (16.2%) | **552.5** | 120 of 34 954 (0.34%) | 0 / 0 | 18.6% † |
| 92 | 867.9 | 12 940 (37.3%) | **544.4** | 96 of 25 385 (0.38%) | 0 / 0 | 20.2% † |

† **The incomplete column is the same counting artefact as above**, at a different rate: these
arms were measured before `crop` came off the run's unconditional expected list, so every frame
the detector found nothing in is counted here as a partial loss. Nothing was lost in those
frames. Read the column as "frames with no detections", or re-run the three arms.

**The tracked rate is a ceiling and the workers do not move it.** Retired frames scale (399.7 →
867.9) while the **tracked** rate saturates at ~550: 48 workers to 92 buys 209 more retired
img/s and *no* more frames carrying ids, while the tracker's refusals rise 2.8% → 16.2% → 37.3%.
So the honest figure for this chain on four A5000s **at ten times the design rate** is **~550
tracked img/s**, and the devices are not what stops it. At the design rate itself it is 711.5
(the section above): two loads, two numbers, and neither is "the box's figure" without its load
beside it.

**Identity collapses here, and the worker count is NOT why.** 0.17 / 0.34 / 0.38% of observations
admitted, against **57.6%** at the design rate on the same footage, the same roster and the same
box. The refusal column is the reordering, and it rises **13×** across these arms while admission
does not move — and goes the wrong way if anything. The 24-worker arm settles it: 2.8% refusals,
reordering all but absent, and admission already collapsed to 0.17%. The variable is the input
**rate**, not the workers.

What the rate does is decimate each camera's stream before the barrier ever sees it: **50 601 to
76 420 frames are refused at the pipeline queue** (spread evenly — 4 000 ± 600 per camera at 92
workers, which is the fair queue working at ten times the design rate), so only 17–41% of a
camera's frames reach a tracker at all. `min_hits` counts CONSECUTIVE qualifying instants and the
gate's hit map is replaced each instant, so a track present in a sixth of them starts again almost
every time. At the design rate on TWELVE cameras nothing is refused and the same gate admits
three fifths; the fifty-camera row above is the design rate too, and it refuses 23% -- the
queue is sized per host, so the fleet's size decides how much of the rate reaches it.

**What this changes about the fix.** `PIPELINE-WORKERS-NEED-CAMERA-AFFINITY` was priced against a
flat 260 img/s measured on the slideshow fixture, where almost nothing tracked at all. Its
premise survives the better fixture and its baseline moves: the tracked rate flattens at ~550
while refusals rise 13×, which is the ordering cost the item is about. What it may NOT claim is
the identity collapse above — that one is the rate, and the item says so now, because the
measurement that would settle affinity's worth is admission at a load the queue does not
decimate.


The fix for the *throughput* half is placement **affinity**, not more threads — a camera's frames reaching one worker, or
a per-camera sequencer in front of the tracker — and it trades load balance for ordering, which
is the trade this project exists to get right. Priced and open as
`PIPELINE-WORKERS-NEED-CAMERA-AFFINITY`.

## Where the time actually goes (the first profile of the C++ chain)

V168 makes optimisation a loop -- benchmark, then **profile** -- and the profiler could not
reach the benchmarked route: `deploy/rootless/profile.sh --cpp` named
`csrc/build/shipinfer_pipeline`, which `scripts/build_csrc.py` has never produced, and it had no
way to start the RTSP servers the mandated route needs in the same container. Both fixed; this
is what the first run says. Nsight Systems, `--trace=cuda,nvtx,osrt`, the pan fixture, 12
cameras x 20 fps x 20 s, four A5000s -- the design rate, not saturation.

**The host is the wall, and the GPUs are a quarter busy.** 121.6 s of process CPU for 26.7 s of
wall is **4.55 cores** at ~240 img/s (3.7 excluding the profiler's own 18.6 s + 4.2 s). Per
thread group: the model instance threads 58.9 s (48%), the 24 pipeline workers 19.8 s (16%),
the twelve camera threads 3.4 s, the RTSP servers 6.1 s (a cost no deployment pays). Extrapolate
the 3.7 cores linearly and **3 000 img/s needs ~46 of this box's 48 cores** -- the host runs out
first, with the four devices at 25% average kernel occupancy.

| CUDA API | share of API time | calls | avg |
|---|---|---|---|
| `cudaStreamSynchronize` | **32.2%** (9.64 s) | 5 564 | 1.73 ms |
| kernel launches (`cuLaunchKernelEx` + `cudaLaunchKernel` + `cuLaunchKernel`) | 36% (10.8 s) | **1 026 905** | ~10 us |
| `cudaMemcpyAsync` | 4.5% (1.36 s) | 53 686 | 25 us |

A million kernel launches in twenty seconds is ~4 300 per frame, which is what four TensorRT
engines cost per image; the launches are the floor, the **syncs are not**. Every
`TrtEngine::execute` ends in a blocking `cudaStreamSynchronize`, so the instance thread stops
dead for 1.7 ms per batch instead of handing the batch to a completion queue and taking the next.

### Answering V168's two questions

**No, the RAM -> VRAM -> RAM -> VRAM round trip is gone.** Host-to-device for the whole run was
**14.7 MB in 6 copies** -- context setup, nothing per frame. The pixels are decoded by NVDEC into
VRAM, copied device-to-device once into a buffer the pipeline owns (`surface_intake.h` argues
that copy), letterboxed by a kernel, cropped by a kernel, and handed to TensorRT bindings that
already live on the device.

**But the reverse leg is enormous, and that is the finding:**

| direction | volume | copies | GPU memcpy time |
|---|---|---|---|
| Device-to-Host | **39.6 GiB** | 7 086 | **2.07 s (73.7%)** |
| Device-to-Device | 192 GiB | 46 115 | 0.73 s (26.1%) |
| Host-to-Device | 14.7 MB | 6 | 3.7 ms |

`TrtEngine::execute` copies **every** output to host memory whether a host consumer reads it or
not, and one of them is the segmenter's `(32, 160, 160)` prototype bank -- **3.1 MB per crop**,
copied down so that a host loop can reduce it to **one float**: the mask's area. Measured on this
box at the shipped shapes, that fold costs **1.44 ms of CPU per crop** (one core sustains 693),
and it is the single largest host item after the engines themselves.

### Which steps run on the CPU

| stage | device | host work |
|---|---|---|
| decode | **GPU** (NVDEC) | one driver thread per camera (3.4 s / 12 cameras) |
| letterbox | **GPU** kernel | a blocking stream sync per frame |
| detect | **GPU** | 300x6 output copied down, decoded on the host (cheap) |
| crop | **GPU** kernel | the box list is built on the host and uploaded |
| segment | **GPU** | **the whole prototype bank copied down; the mask fold is a host loop** |
| embed x2 | **GPU** | 512 floats per crop copied down (cheap) |
| track | **host** | bytetrack, per camera, in the `shipvision` lane |
| mtmc | **host** | gate, gram, agglomerative clustering, identity assignment |
| reassembly, records, JSON | **host** | one event per frame |

Four items came out of this profile and are filed with their numbers:
`MASK-FOLD-BELONGS-ON-THE-DEVICE`, `ENGINE-COPIES-EVERY-OUTPUT-HOME`,
`EXECUTE-BLOCKS-THE-INSTANCE-THREAD` and `THE-BUILD-NEVER-VECTORISES`.

## Eight open instants is a four-camera group's arithmetic

The design load associates nothing, and the cause is a constant. `max_instants` bounds how many
instants the barrier keeps open and evicts the oldest past it; the default is 8, documented as
"half a second at the default window". Every camera holds one instant open — the one its last
frame landed in — and seals more as it moves on, so the number legitimately open scales with the
**fleet**. At fifty cameras a bound of eight spends every eviction on a bucket the group is
still filling.

Same rig, same plan, one line different (`max_instants` in the chain): 50 cameras × 20 fps over
gstreamer RTSP from the pan fixture, GPUs 0/2/5/6, 92 workers, 40 s.

| bound | instants opened | evicted | offered | admitted | ids / tracks | frames accepted |
|---|---|---|---|---|---|---|
| **8** (the default) | 3 866 | **1 007 (26.0%)** | 56 050 | **170** | 0 / 0 | 29 565 |
| **8** | 3 570 | **720 (20.2%)** | 67 749 | **97** | 0 / 0 | 31 760 |
| **8** | 3 788 | **1 075 (28.4%)** | 53 637 | **545** | 0 / 0 | 30 725 |
| 16 | 1 828 | **0** | 77 305 | 2 140 | 14 / 19 | 34 729 |
| 32 | 1 817 | **0** | 74 820 | 2 258 | 0 / 0 | 34 095 |
| 64 | 1 912 | **0** | 84 407 | 2 154 | 10 / 10 | 34 908 |
| 64 | 1 838 | **0** | 68 482 | 1 359 | 12 / 19 | 32 292 |
| 64 | 1 729 | **0** | 60 721 | 527 | 10 / 19 | 31 729 |
| 128 | 1 964 | **0** | 79 781 | 782 | 9 / 18 | 34 598 |

**The knee is 16 and nothing above it buys anything.** Eviction is a step, not a slope: at 8 it
takes a fifth to a quarter of every instant opened, and from 16 up it is exactly zero in six
runs. Admission rises about tenfold across that step (97–545 → 527–2 258) and then stops rising,
so the fix is "large enough", not "larger".

**What the run does with the extra evidence is noisier than the eviction count.** Global ids are
0 / 0 in all three runs at 8 and non-zero in five of the six above it — the 32 arm admitted the
most observations of any run here and still resolved none. So the honest claim is that the bound
was **preventing** association and no longer is; what the clusterer then does with a 50-camera
instant is the next question, not this one's answer.

**It is not a throughput knob.** Frames accepted are 29 565 / 31 760 / 30 725 at 8 against
31 729 – 34 908 above it: the means differ by ~10% but the ranges overlap, and a single pair
(the first 8-arm against the first 64-arm) would have read as +18%. Three runs an arm is what
says otherwise.

**The fix is that the default follows the fleet** — `max(8, cameras seen or announced)`,
recomputed as cameras arrive, on both planes. A chain that names `max_instants` still gets
exactly that number, in both directions: an operator who has measured their own arrival spread
can say so, and eviction stays testable.

Seen *union* announced, and not the live set, because the live set is a roster decision. The
chain this repository ships declares four cameras and every fleet here is fifty, so a bound
derived from the roster would have been 8 again — the bug, on the shipped configuration. Both
arms, same build, same load:

| chain | announced | bound in force | evicted | admitted | ids / tracks |
|---|---|---|---|---|---|
| `ship_person_cpu.yaml`, **unchanged** | 4 | **54** | 0 | 1 683 | **25 / 68** |
| the same with a fifty-camera roster | 50 | **50** | 0 | 2 180 | 0 / 0 |

The first row is the headline: **the chain this repository ships resolves 25 global identities
at the design load**, where every run at the default bound resolved none. The second row is the
caveat, and it is the same one the sweep carries — identity is erratic at this load (0 to 25
across nine runs above the knee, and the arm that admitted the MOST observations resolved none),
so what the bound fixed is *eviction*, deterministically, and identity is downstream of that.
## The mask fold, moved to where the batch already is

`ship_segmenter` answers 300x38 detection rows **and** a `(32, 160, 160)` prototype bank per
crop. The bank exists only to be reduced to one number — the mask's area — and until now that
reduction ran on the host, after both outputs had been copied home: 3.1 MB down the bus and
1.44 ms of CPU per crop, which the profile above names as **81.3% of all device-to-host memory
time** at this load. The kernel for it landed earlier (10.0 us a crop, pinned against the
readable fold); this is the run with it wired into the engine, where it can read the network's
output before anything comes home.

One binary, one plan, one switch (`SHIPINFER_DEVICE_FOLD`), two arms twice each — 50 cameras ×
20 fps over gstreamer RTSP from the pan fixture, GPUs 0/2/5/6, 92 workers, 40 s:

| the fold runs | frames accepted | host CPU | **ms of host CPU a frame** | segmenter busy |
|---|---|---|---|---|
| **on the device** | 35 839 | 733.9 s | **20.48** | 116–118% |
| **on the device** | 35 693 | 728.9 s | **20.42** | 111–117% |
| on the host | 31 801 | 751.8 s | 23.64 | 82–94% |
| on the host | 31 519 | 741.8 s | 23.53 | 85–91% |

**+13.0% frames retired and −13.3% host CPU a frame**, and unlike the instant-bound sweep the
two arms' ranges do not overlap on either number: 35 693–35 839 against 31 519–31 801, and
20.42–20.48 ms against 23.53–23.64. The instance threads' own CPU is unchanged per frame
(15.1–15.3 ms against 15.3–15.4), which is what says the saving is the copy and the fold rather
than the scheduling: the same work per frame, minus a 3.1 MB trip home and a 1.44 ms reduction.

**What goes up is the segmenter's occupancy**, from ~87% to ~116%, and that is the trade being
made: `execute()`'s wall time now contains the fold kernel, so work that was the host's is the
GPU's. At this load the host was the wall — the profile said so — which is why the trade pays.

**The numbers are the same numbers.** `test_fold_wiring` runs one batch through two adapters,
one folding on the device and one not, and compares the areas against `graph/mask_area.cpp`'s
answer on the unfolded outputs — at two mask thresholds, because a fold that never read the
bank would agree at one.

## How much of the fleet is in one instant, and what the window costs

Every reason the barrier reports is about TIME — an instant ran out of window, or a camera moved
on. None of them says how many cameras were actually in the instant when it ended, and a
cross-camera association over one camera is not one. `mtmc_instant_cameras` reports that now,
tallied where every ended bucket passes.

**An instant holds about eleven cameras, whatever the fleet is.** Same rate, same fixture, same
60 ms window:

| fleet | cameras an instant held (mean / largest) | offered | admitted | ids / tracks |
|---|---|---|---|---|
| 50 × 20 fps, 92 workers | **11.8 / 47** of fifty (24%) | 90 265 | **1 987 (2.2%)** | 18 / 18 |
| 12 × 20 fps, 24 workers | **10.5 / 12** of twelve (88%) | 42 067 | **31 440 (74.7%)** | 13 / 72 |

So the fleet grew and the window did not. The arithmetic that fits this pair is the gate's:
`min_hits` counts *consecutive* qualifying instants and a track can only qualify in an instant
its camera is in, so 0.24³ = 1.4% against 0.88³ = 68% — close to the measured 2.2% and 74.7%.

**And the window sweep refutes that as the whole story**, which is why both tables are here.
Same fifty cameras, same load, only `sync_window_ms` moved:

| `sync_window_ms` | cameras / instant | instants | admitted | ids / tracks | frames accepted | frame p50 |
|---|---|---|---|---|---|---|
| **60** (the default) | 11.8 | 1 951 | 1 987 | 18 / 18 | 36 081 | 259 ms |
| 120 | 9.4 | 3 067 | 898 | 42 / 67 | 33 272 | 309 ms |
| 250 | 12.3 | 2 335 | 3 512 | **50 / 77** | 30 085 | 334 ms |

Cameras per instant barely moves while identities go 18 → 42 → 50, so cameras-per-instant is not
the variable identity tracks. What moves is the close **reason**: `advanced` is 21% of instants
at 60 ms and 85% at 250 ms, because a window wider than the frame period (50 ms at 20 fps) puts
a camera's next frame inside its own bucket's span — which `barrier.h`'s docstring predicted
from first principles and nothing had measured until now.

**The window is a real lever and it is not free.** 250 ms buys 18 → 50 identities and costs 17%
of the frames (36 081 → 30 085) and 29% of p50 latency (259 → 334 ms). Whether that trade is the
right one is a deployment's question, not a benchmark's — what this page can say is that it is a
trade, with both sides measured.

**And the gate is where the evidence actually goes.** One more arm, the default window and the
reference's `min_hits` dropped from 3 to 1:

| `min_hits` | offered | admitted | ids / tracks | frames accepted | frame p50 |
|---|---|---|---|---|---|
| **3** (production's default) | 90 265 | 1 987 (2.2%) | 18 / **18** | 36 081 | 259 ms |
| 1 | 75 376 | **75 322 (99.9%)** | 23 / **190** | 34 419 | 282 ms |

Read the last column of the ids pair, not the first. At the shipped defaults every identity
holds **exactly one track** — 18 identities over 18 tracks is not cross-camera association, it
is eighteen cameras each holding its own. With the gate open, 190 tracks resolve into 23
identities: about eight tracks an identity, which is what the mtmc stage exists to produce.

So `min_hits 3` is not merely conservative at this fleet size — it is **unreachable**. It counts
three *consecutive* qualifying instants, a track can only qualify in an instant its camera is
in, and a camera is in 24% of them. The gate's intent (three confirmations before a merge) needs
a counter over the instants a camera *was* in; until it has one, the design load's identities are
singletons whatever the window does.

## The gate at the design load, after it counted sightings

`MTMC-GATE-COUNTS-INSTANTS-NOT-SIGHTINGS` predicted this arithmetic and the tables above
measured the symptom: a camera is in 24% of instants, `min_hits` counted three *consecutive*
instants, so 0.24³ = 1.4% — and the gate admitted 2.2% with every identity holding exactly one
track. shipvision#16 and its port (#249) make a streak survive an instant its camera was not
in. Same load, same route, same fixture, two arms:

| | before (11 Sep) | arm 1 | arm 2 |
|---|---|---|---|
| observations offered | 90 265 | 135 496 | 134 304 |
| admitted | **1 987 (2.2%)** | **114 084 (84.2%)** | **112 538 (83.8%)** |
| identities / tracks | 18 / **18** | 16 / **167** | 22 / **193** |
| cameras an instant held (mean / largest) | 11.8 / 47 | 11.2 / 39 | 11.3 / 37 |
| instants | 1 951 | 3 296 | 3 318 |

**The second column of the identity pair is the result.** 18 identities over 18 tracks was
eighteen cameras each holding its own — not cross-camera association at all. It is now about
nine tracks an identity on both arms, which is what the stage exists to produce.

**Cameras per instant did not move**, which is the control: 11.8 → 11.2 and 11.3. The window
sweep above had already shown that cameras-per-instant is not the variable identity tracks; this
says the gate change did not move it either, so the admission jump is the counting rule and not
a differently-shaped instant.

**What this arm does not claim.** Frames accepted was 61 981 and 61 979 at 279 ms and 267 ms
p50, on 16.2–16.3 busy cores of 48. The 36 081 in the window table is *not* a control for that:
several unrelated things merged in between (the mask fold on the device, the instant bound
following the fleet, the drain fix), so the throughput difference is not attributable to the
gate. Admission and identities are, because that is what the gate decides.

**One thing worth filing that this run surfaced.** `mtmc_frames late` is 24 950 of 68 538 frames
read — better than a third of the fleet's frames reach the barrier after their instant has
closed. That is a different question from the gate's and it now has a number on it.

Run with `SHIPINFER_BENCH_SOURCE=nvdec scripts/run_cpp_bench.sh <label>`: 50 cameras × 20 fps
over GStreamer RTSP from the pan fixture, 4 GPUs, 70 s with the analysis's 10 s warm-up, the
binary built inside `shipinfer-gst:jammy-nvdec` because that image is the one with both the
compiler and the GStreamer headers.

## The profiler was the bug, and what it says once it runs

`deploy/rootless/profile.sh --cpp` printed `loading engines...`, died ~1.2 s later with an empty
`threads: {}`, and wrote a report holding only the driver's context calls. Eight probes had
narrowed it to one sentence — *a binary that loads a TensorRT plan segfaults under this Nsight
Systems; a binary that only uses CUDA does not* — and every one of them was a binary of ours, so
"something of ours is involved" was still open.

**`trtexec` settles it.** NVIDIA's own loader, none of our code, same container, same plan, same
flags (`scripts/probe_nsys_trtexec.sh`):

| Nsight Systems | `trtexec --loadEngine` |
|---|---|
| 2024.5.1 | **exit 0** |
| 2024.6.2 | **exit 0** |
| **2025.1.3** | **exit 139 (SIGSEGV)** |

Nothing of ours is involved and the profiler is the variable. `profile.sh` picked
`ls -d /opt/nvidia/nsight-systems/* | sort -V | tail -1` — the newest — which is exactly how it
selected the broken one and kept selecting it. It now skips a version measured to segfault on
engine load, says so on stderr, and `SHIPINFER_NSYS_DIR` still overrides.

With 2024.6.2 the design load profiles: 50 cameras × 20 fps over GStreamer RTSP, 4 GPUs, 40 s,
37 572 frames read and 32 445 accepted, a 584 MB report.

### And the sync it was blocking

`EXECUTE-BLOCKS-THE-INSTANCE-THREAD` asked for one number before anything is built: the sync's
share of an **instance thread's wall time**, not of CUDA API time. The plan runs 7 instances a
GPU (detector 2, segmenter 2, person embedder 2, ship embedder 1) on 4 GPUs — 28 instance
threads.

| | 12 cameras × 20 fps | 50 × 20 (design load) |
|---|---|---|
| `cudaStreamSynchronize` total | 21.31 s | **304.29 s** |
| calls | 23 880 | 150 629 |
| average | 0.89 ms | **2.02 ms** |
| share of CUDA API time | 44.2% | 47.1% |
| **share of 28 threads' wall** | **3.2%** | **~14%** |

**The load is the variable, which is what the item suspected and could not show.** The re-pricing
had estimated "under 2%" from the 12-camera profile; the 12-camera number is 3.2% and the design
load is about four times that. The denominator is stated rather than implied: 304.29 s against
28 threads over ~77 s of the process's 80.85 s. Over the 40 s measurement window alone it is
25%, and that is the wrong denominator — the threads are alive for the startup and the drain too.

So the item has the justification it said it lacked. Whether ~14% is worth ~1 GB of VRAM and the
buffer-ring restructuring is a judgement for whoever builds it; what is no longer true is that
the number is unknown, or that it is 2%.

## How late a frame reaches the barrier, and why the window is not the lever

The gate arm left one number unexplained: `mtmc_frames late` was 24 950 of 68 538 frames read —
better than a third of the fleet reaching the barrier after its instant had closed, carrying no
global id. `late` says a frame missed its instant; nothing said *by how much*, and that is the
difference between a window too narrow and a chain too slow to reach one. `mtmc_arrival_lag_us`
reports it now, measured at the mtmc stage because that is the only place holding both the
capture stamp (wall) and the arrival moment on one clock.

| 50 cameras × 20 fps | arm 1 | arm 2 | arm 3 |
|---|---|---|---|
| samples | 60 847 | 61 837 | 61 743 |
| p50 | **248.5 ms** | **246.8 ms** | **237.5 ms** |
| p95 | 582.9 ms | 580.7 ms | 566.8 ms |
| p99 | 785.9 ms | 792.9 ms | 754.8 ms |
| max | 1 205 ms | 1 216 ms | 1 223 ms |
| `late` | 23 924 | 23 762 | 23 030 |
| frame p50 end to end | 284.2 ms | 284.7 ms | 272.8 ms |

**The window is 60 ms.** A frame arrives a median of ~240 ms after it was captured — four
windows — and the lag at this one stage is most of the frame's whole ~280 ms.

**And it is the SPREAD that strands a frame, not the delay.** A fleet delayed uniformly by
240 ms would bucket together perfectly: every camera's frame would land in the same
late-opened instant. What strands a frame is arriving after its instant closed, so the variable
is the *dispersion* against the window — p50 240 ms to p99 760 ms, about 520 ms, nine times the
window.

**At twelve cameras the lag fits inside the window, and the window stops mattering.** The same
sweep the 50-camera fleet answered 18 → 42 → 50 identities to, run at twelve:

| `sync_window_ms` | lag p50 | lag p95 | cameras / instant | admitted | ids / tracks | frames | p50 |
|---|---|---|---|---|---|---|---|
| **60** | 19.4 ms | 38.2 ms | 10.9 / 12 | 66 205 (89.9%) | 17 / 96 | 16 722 | 69.3 ms |
| 120 | 20.7 ms | 44.1 ms | 11.4 / 12 | 66 209 (89.9%) | 18 / 95 | 16 722 | 62.7 ms |
| 250 | 19.2 ms | 39.0 ms | 11.4 / 12 | 66 336 (89.9%) | 17 / 96 | 16 730 | 62.1 ms |

Identities, admission and frames are **flat**. What still moves is the close *reason* —
`window` 1 169 → 4 → 2 and `advanced` 365 → 1 465 → 1 466 — so a wider window changes how an
instant ends and not what it holds, because at twelve cameras they are all there anyway.

**That answers the question the fleet-size contrast raised.** The rate is the same 20 fps in
both fleets; what differs is how long a frame takes to reach the barrier — 19 ms at twelve
cameras, inside one window, against 240 ms at fifty, past four of them. The window was a lever
at fifty only because the lag had outgrown it, and the earlier sweep priced that lever at 17%
of the frames and 29% of p50. What has to come down is the chain's latency to `mtmc` and its
variance, which is upstream of the barrier and nothing the barrier can do.

Measured with `SHIPINFER_BENCH_SOURCE=nvdec scripts/run_cpp_bench.sh <label>` over GStreamer
RTSP from the pan fixture, 4 GPUs, 70 s with the analysis's 10 s warm-up;
`SHIPINFER_BENCH_CHAIN` points at a chain whose `sync_window_ms` is the swept variable.

## Camera affinity: a saturation effect, not a design-rate one

## Camera affinity: a saturation effect, not a deployment-density one

`PIPELINE-WORKERS-NEED-CAMERA-AFFINITY` said the shared worker pool reorders a camera's frames,
the per-camera tracker refuses a frame that does not advance its stream, and the refusal rate
rises with the worker count — 2.8% → 16.2% → 37.3% untracked at 24/48/92 workers. It also said,
in its own words, what those rows could not settle: they were measured at **saturation**
(12 cameras × 200 fps), where 50 601–76 420 frames were refused at the pipeline queue, so
"affinity's worth has to be measured as admission at a load the queue does not decimate —
12 × 20 fps with workers swept — and not from these rows."

Here is that sweep. `queue_rejected` is **0** on every arm, so nothing is being decimated:

| workers | untracked | untracked % | admitted | ids / tracks | frames accepted |
|---|---|---|---|---|---|
| **24** | 12 | **0.07%** | 66 446 / 73 844 (90.0%) | 17 / 96 | 16 730 |
| 48 | 23 | **0.14%** | 66 505 / 73 924 (90.0%) | 17 / 96 | 16 746 |
| 92 | 32 | **0.19%** | 66 397 / 73 781 (90.0%) | 17 / 96 | 16 737 |
| 92 (repeat) | 40 | **0.24%** | 66 199 / 73 643 (89.9%) | 17 / 91 | 16 733 |

`frames_dropped 0` on every arm as well. Raw per-arm output is
`.artifacts/cpp/aff{24,48,92,92b}.log`; the counters above are `track_frames_untracked`,
`mtmc_observations offered`/`admitted` and `mtmc_identities`, which the run summary's own grep
does not print — they are in the log.

**The reordering is real and it is negligible.** Untracked rises about 3–4× from 24 workers to
92 — the effect the item describes — on a base of one frame in fifteen hundred. Admission and
frames accepted do not move at all; the identity count moves once, 96 tracks to 91 on the
repeat arm, which is run-to-run variation at a constant 17 identities rather than a trend —
the 92 arm itself gives 96.

**So the 37.3% was the rate, not the worker count**, which is what the item suspected of its own
rows and could not prove without this arm. Affinity trades load balance for ordering, and load
balance is what this project exists to get right; paying that for 0.2% is the wrong trade *at
this density*.

**WHICH DENSITY, and it is the whole claim** — an earlier draft said "at the design rate" and
this page has two rows with a claim on that phrase. The deployment is fifty cameras on **sixteen**
GPUs: **3.1 cameras per device**. This sweep is twelve on four: **3.0 per device** — the same
box, and that is why it is the right arm. The fifty-on-four row at "### The design load, on four
of the sixteen GPUs" is **12.5 per device**, four times the deployment's, and it refuses
`queue_rejected 8645` (23%) — so it fails *this section's own control* and cannot be compared
with these arms at all. On four devices fifty cameras cannot reach `queue_rejected 0`.

So the verdict is scoped to what was measured, and **the re-open condition is arithmetic rather
than a feeling**: re-read this table at any load where `queue_rejected` is non-zero — which on
four devices is fifty cameras, and on the deployment's sixteen is a load this box cannot yet
generate. The 3.7% untracked in that fifty-on-four row is 17× these numbers and is **not**
evidence against building the sequencer; it is a measurement at 4× the density with a fifth of
the frames refused upstream.

**Where the frames that carry no ids go is NOT settled here, and this row used to claim it
was.** An earlier draft carried a `lag p50` column and concluded from it that the cause is chain
latency rather than ordering. The bench emits no such metric: it reports `reassembly_us_*` (time
in reassembly) and `frame_us_*` (captured→emitted), and neither is arrival-versus-capture at the
barrier. The column was one of those two under a new name, no raw output here backed it, and the
fifty-camera figure beside it was extrapolated from nothing this run measured — all twelve-camera
arms. Removed rather than relabelled, because `MTMC-A-THIRD-OF-FRAMES-ARRIVE-LATE` says in its
own words *"do not touch the window until that histogram exists"*, and answering it with the
proxy it was written to forbid is how the histogram never gets built. The instrument that does
measure it is `MTMC-LAG-IS-NOT-INSTRUMENTED` (#255, open at the time of writing — the
pointer is pending until it lands); read that item's numbers, not these.

Measured with, verbatim:

```
SHIPINFER_BENCH_IMAGE=shipinfer-gst:jammy-nvdec SHIPINFER_BENCH_GPUS=2,3,4,5 \
SHIPINFER_BENCH_WORKERS=<n> SHIPINFER_BENCH_SOURCE=nvdec SHIPINFER_BENCH_CAMERAS=12 \
  scripts/run_cpp_bench.sh <label>
```

The IMAGE is named because `deploy/rootless/cpp.sh` defaults to `shipinfer-gst:jammy`, where an
nvdec-less binary refuses the source outright; the GPU IDS because the script's own default is
`2,3,4,5` while neighbouring sections use `0/2/5/6`, so "4 GPUs" alone names a different set.
GStreamer RTSP from the pan fixture, 70 s. The 10 s warm-up is the ANALYSIS window, not an
exclusion: 16 730 accepted ÷ (12 × 20) ≈ 69.7 s, so these counters span the whole run. That can
only inflate untracked, which is the direction that does not flatter the conclusion.

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
