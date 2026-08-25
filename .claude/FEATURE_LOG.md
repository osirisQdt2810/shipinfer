# Feature Log

One entry per large feature or seam change. Append-only, newest on top. Skip it for tiny
edits, typo fixes and pure docs.

---

## 2026-08-25 — The benchmark's other two tiers, and an RTSP source

**What it is.** R44 asks for three benchmark tiers — system, algo, kernel — and only the
system one existed. R55 makes RTSP mandatory for the benchmark, not only for the tests, and
every measurement so far replayed JPEGs off disk. Both closed.

**`benchmarks/stages.py` — the algo tier.** Where does one frame's time go, stage by stage.
It *reads* rather than instruments: `PipelineStage.run` already stamps `elapsed_us` on every
outcome and `_CollectorObserver` already feeds it into `shipinfer_pipeline_stage_latency_us`,
so a second timing path would be a second implementation that could disagree with the one
operators watch. Reports per-call p50/p95, calls per frame, per-frame cost and share.

`calls_per_frame` is the whole point: a stage costing 8 ms on one frame in three costs 2.7 ms
per frame, and the embedders run once per *object batch*. Assuming one call per frame would
overstate the cheap stages and understate the expensive ones by the same factor.

It runs **below saturation deliberately** and warns loudly when the run did not keep up.
Under saturation a stage's latency includes the time it waited behind other frames, so a
backlog reads as an expensive stage — the same 98% bar `check_offer` holds the system tier to.

**`benchmarks/kernels.py` — the kernel tier.** What one op costs, per implementation, at the
shapes this project runs. Two corrections on the way in, both of the same family — measuring
something adjacent to what production does:

- It called `IMAGE_OPS.create(name)` with no arguments. `TorchImageOps` falls back to the CPU
  without a `device_index` and `PipelineRunner._build_ops` always supplies one, so the first
  run timed torch on the **CPU** and reported it 7–13× slower than numpy. Bound correctly,
  torch on `cuda:0` is 3.27× numpy on letterbox, 1.84× on crop_batch, 2.47× on nms.
- `letterbox_batch` returns numpy by contract, so a device implementation pays a copy home
  that numpy never makes. Timing only that column charges the device implementations for the
  round trip; `letterbox_to_device` is the device-fair case and the one production calls.

Both tiers report what they could **not** measure rather than printing a shorter table — a
missing column with no explanation is how "we never measured it" becomes "it is not faster" —
record the host load, and mark a spread over 20% as noisy. The first kernel run was taken at
load 41 of 48 with spreads to 76%.

**`--source rtsp`.** The bench cameras point at `scripts/rtsp_serve.py` over a real socket,
with `benchmarks/harness/rtsp.py` owning the server's lifetime. It refuses rather than
tolerates: a server that never accepts, or that exits early, raises at start-up with its
output attached — a run whose cameras cannot connect produces a clean-looking zero and this
project has already published one of those. Readiness is a socket poll rather than a sleep;
teardown is terminate-then-kill, because a GLib loop holding the port makes the *next* run
fail with an address already in use, minutes later and nowhere near the cause.

**Replay and RTSP are different experiments, not a fast one and a slow one.** Replay measures
the inference plane with the decode path removed, so a replay number is an upper bound on the
RTSP one. The source is recorded in the run metadata, printed on the console, and explained
in the README, because the failure to avoid is quoting a replay figure as though NVDEC were
in it.

**Tests.** 31 offline tests over the two tiers and the RTSP wiring, pinning the arithmetic and
all four server failure paths. The arithmetic is where a benchmark lies: every defect review
found in `run_bench.py` was a formula producing a plausible number from a run that did not
support it, not a broken measurement loop. 116 tests in `benchmarks/tests`.

**Still owed.** Neither tier has completed a run against real engines — the algo tier's first
attempt hit CUDA OOM because parallel agents held 22 GiB on four GPUs, and the RTSP path has
not been run at all. Both are needed for C1a, which is the profile that has to come before any
more optimisation.

---

## 2026-08-25 — Five of Triton's features, taken (`docs/qa/triton.md` §3)

**What it is.** The five rows of that document's "features Triton has that we should take"
table that were still a plan, implemented and the table rewritten to describe the code rather
than the intention:

1. **`GET /v2/models/{name}/stats`** (and the `/versions/{v}/` spelling) — `server/statistics.py`
   holds `ModelStatistics`, one per model, shared by its instances and by the ensemble path.
2. **Explicit model control** — `model_control: explicit` plus
   `POST /v2/repository/{index, models/{n}/load, models/{n}/unload}`.
3. **A rate limiter** — `scheduling/limits/`, a registry with `off` (default) and
   `concurrency`, configured per model.
4. **Warm-up from declared samples** — Triton's `model_warmup` key, materialised by
   `repository/warmup.py` and run by `ModelBackend.warmup`.
5. **Request tracing** — `core/tracing/`, Triton's seven event names, `none` (default) and
   `jsonlines` sinks, `rate=N` sampling.

**Why each one, in one line.** A histogram has no per-model cumulative count, so an operator
debugging one camera's model had to read the fleet's numbers to find one. A repository that
grows cannot be loaded whole. The queue bounds what is *waiting*, and nothing bounded what was
*running* — eight instances whose windows close together all enter compute at once. A fixed
count of zero-filled batches decides how often a model is warmed but not *what with*, and the
data is what selects the kernels. And six stamps with no sink cannot answer "why was frame
8213 slow".

**Two things the wiring changed that the feature list does not show.**
`DurationStat.observe(ns, count)` now adds `count * ns` rather than `ns`: crediting a batch's
span once instead of once per request divides the reported latency by the batch size, which is
an error in the flattering direction and was caught by the first test written against it. And
`ModelInstance.wait_ready` now returns as soon as the worker has *settled* either way — before
that, a worker that failed on its first line held start-up for the whole 120 s timeout and then
reported "did not become ready", hiding the cause. A typo in `model_warmup` is enough to reach
that path, which is how it was found.

**Where Triton was deliberately not followed**, each recorded in the document: `poll` model
control (a timer can load a half-written config), reload-on-load (it must stop the running copy
first, so a half-failed reload takes a working model down), and the general named-resource rate
limiter (the only resource this pipeline has needed to bound is "an execution").

**Cost.** 90 new offline tests, all class-based; 892 pass with no GPU. Nothing new is on by
default: `off` limiter, `none` trace sink, `none` model control, and no `model_warmup` in any
shipped config, so a deployment that does not opt in pays one virtual call per completed
request and two per batch.

## 2026-08-24 — The C++ data plane (`csrc/`)

**What it is.** A standalone binary that owns everything running once per frame or once per
object: ingest and per-camera pacing, the fair bounded queue, letterbox and crop-resize kernels
writing straight into TensorRT bindings, device-affine instance pools, the perception graph,
per-frame reassembly with a timeout, and the occupancy log. Python keeps the control plane.
See ADR-014 for why this is not the optional-extension contract ADR-007 governs.

**Why.** The Python data plane capped at 77 img/s using five cores of forty-eight while every
GPU queue sat empty. Four other candidates were eliminated by measurement first (the GPUs, the
worker pool, the reassembly lock, the load generator), so the remaining explanation was the
pure-Python share of the per-frame path holding the GIL.

**The measurement, and the one design decision that makes it trustworthy.** The binary writes
the *same* `*_buffer_size` occupancy JSONL the Python driver and the baseline binary write, so
`benchmarks/harness/analysis.py` scores all three with one implementation and one set of
guards. A port that exists to look good must not be scored by a friendlier judge than the thing
it is compared against.

    50 cameras x 20 fps on 4x A5000, 70 s, 10 s warmup, scored by the shared analysis
    pipeline: offered 983, growth +592.9/s, sustained 390.5, SATURATED

390.5 against the Python plane's 77.5 — **5.0x**, with 98% of the offered load actually
delivered where the Python driver could never exceed ~87 img/s.

**Four bugs found by running it**, each a design error: static plans refuse any batch but their
own; cross-device execution (an instance on gpu0 executing pixels on gpu1, surfacing as an
illegal access somewhere else entirely); `gpuDeviceSynchronize` after every kernel, which is
device-wide; and a pageable host source for 2 GB/s of copies.

**And a hole in the measurement itself**, which is the part worth remembering. The occupancy
log first carried `busy()` — leases held. This design has no queue in front of a pool (a worker
blocks inside `lease`), so a *fully committed* pool reads as a flat `busy == size` and the
analysis scores it SUSTAINED. `ship_segmenter` sat at exactly 4 with exactly 4 instances:
pegged, and invisible. Logging `waiting()` instead showed 37 of 48 workers blocked on it, and
every bottleneck since has been a model pool rather than the interpreter — which is the
qualitative change the port bought and is worth more than the 5x.

**Review found four blocking defects in the first version**, all real: a per-frame
`gpuMalloc`/`gpuFree` on the dispatch path with the reusable buffer voided by `(void)`; skipped
branches indistinguishable from failed stages (every ship-only frame sealed Incomplete);
reassembly eviction destroying a frame with no event and no per-camera attribution; and no ADR
for a second data plane. Fixing the second took Complete events from a minority to 28656 of
28808. Fixing the first — the one predicted to be the throughput lever — moved 390 to 400,
about 2.5%, which is a reminder that a plausible mechanism is not a measured one.

---

## 2026-08-24 — The benchmark harness: what counts as a measurement

**What it is.** `benchmarks/` drives ShipInfer and `counting-simulation` under one load and
compares them by the baseline's own buffer-growth saturation methodology: a buffer whose
occupancy grows over the steady window is a module that cannot keep up, and
`sustained = offered - growth`. `benchmarks/baseline/` is the upstream repo as a submodule;
its `sim_pipeline_v2.cpp` is compiled unchanged and run as its own binary — nothing here
re-implements it.

**The seam it owns.** `run_bench.system_throughput` is the only place that decides *what
counts as an image, once*. The baseline runs two independent single-model pipelines over
disjoint image streams, so its system rate is `det + seg`. ShipInfer runs one DAG where every
frame enters the pipeline queue once and then fans out into crops, so its rate is the
pipeline queue alone — summing its modules the way the baseline's report does would count
each frame at the queue and again at the detector.

**The taxonomy, which is the part worth remembering.** A run yields one of three things and
conflating them is how a harness publishes a number it did not measure:

- **SATURATED** (not capped) is a **capacity** — the buffer grew, so `offered - growth` is
  exact. This is the whole methodology.
- **SUSTAINED / DRAINING** is a **floor** — nothing grew, so capacity is *at least* the
  offered rate and this run cannot say how much more.
- **UNMEASURED** is nothing — a capped buffer sheds instead of growing, so its slope means
  nothing.

`ratio_of` is the single place a pair combines: capacity/capacity is exact, floor/capacity is
`>= Nx` (can meet a target), capacity/floor is `<= Nx` (can only miss one), floor/floor is
nothing. The first version had this inverted — it refused SATURATED as "a bound" — which made
a speed-up structurally unreachable, because both systems are offered the same load by
construction. Six review rounds to find that.

**The guards, each of which caught a real lie.** Offered is what *entered* (a dropped frame
cannot grow a buffer, so counting it as offered turned a shedding system into a 3.3x
overstatement); a capped module forces UNMEASURED; `check_offer` refuses a run whose
generator delivered under 98% of target; `reconcile` cross-checks the buffer-log rate against
`events_emitted/elapsed`, which a scheduler that *refuses* work cannot fool; every counter is
rated over the same window the fit uses. `--sweep` climbs the offered rate until something
saturates, because one point cannot settle a comparison when both sides get the same load.

**First result.** baseline 868.2 img/s, shipinfer 81.4 img/s, 0.09x against a 5x target. The
binding module is the pipeline queue, not any GPU queue, and it is insensitive to
`--pipeline-workers` over an 8x range — the wall is one Python process. See JOURNAL.

---

## 2026-08-23 — The pipeline plane: the perception DAG, reassembly, and the event contract

**Why.** `src/shipinfer/pipeline/` was empty, so nothing connected the cameras to the models
to anything downstream. This is the application half of PLANE 2 in
`references/bitbucket-subfaceid/docs/new-system-architecture.md`: detect, crop, segment,
embed and recognise, then join a frame's results and publish them on the contract the
tracking tier already consumes.

**Seams introduced.**

| Seam | Where | Extension point |
|---|---|---|
| Frame -> request | `pipeline/sink.py` | `QueueFrameSink`, the production `FrameSink` of ADR-011 |
| Stages | `pipeline/graph/` | subclass `PipelineStage`; `ModelStage`/`ObjectStage` cover a model |
| Reassembly eviction | `pipeline/reassembly/policy.py` | `@EVICTION_POLICIES.register` |
| Result sinks | `pipeline/sinks/` | `@RESULT_SINKS.register` (null / jsonlines / kafka) |
| Pipeline settings | `core/settings/pipeline.py` | one field on `ServerSettings` |

**Decisions.** Reassembly keeps `BodyDataCollector`'s shape (camera -> frame -> results,
complete-or-timeout at its own 1500 ms) and fixes the three things it got wrong: eviction
charges the overflow to the camera holding the most incomplete frames rather than dropping the
globally oldest entry, every internal structure is bounded including the per-camera index, and
a timeout emits a partial event naming the missing stages instead of deleting the frame. The
inherited drop-oldest behaviour ships as a registered policy so the regression test runs the
two side by side. Emission happens when the worker **seals** a frame, not when its
currently-expected stage set is momentarily satisfied — the set grows as branches are decided.
The schema keeps every v1 `Det2MOT` key with its v1 meaning (people only) and adds ships in
the same parallel-array idiom, so a deployed `motservice` needs no rebuild.

**Notable.** Three defects found by running the end-to-end test on a host that has GPUs, all
of which report as something other than what they are: one `ImageOps` shared across worker
threads overwrites a pinned buffer mid-DMA and says `crop_kernel failed: invalid argument`;
preprocessing every frame on `cuda:0` re-creates this project's founding bug one layer up; and
a worker whose current device is 0 holding ops built for `cuda:1` says `invalid resource
handle`. `ThreadLocalImageOps` binds one instance per thread to one device, round-robin over
the visible GPUs (ADR-002). A fourth, caught by the tests: a `RequestQueue` and a `ResultSink`
both define `__len__`, so `self._queue = queue or default` silently discarded an injected empty
one — every default in the runner is now `if x is None`.

**Known cost.** Pre-processing returns to the host before the model stages it to its own
device. A GPU-resident path needs `letterbox_to_device` writing into the chosen instance's
binding buffer, which means knowing the instance — a dispatcher decision, and the "Phase 2
fast path" the architecture document files for when a measurement says the round trip is what
hurts (ADR-007).

**Layering.** `pipeline` has no `ingest` edge in `scripts/hooks/check_layers.py`, so the
adapter describes what it needs from a frame as a four-member `TaggedFrame` protocol and the
runner takes a `FrameProducer` protocol, in the same spirit as `MemoryHandle` in ADR-001. The
rule was left alone rather than widened.

**Evidence.** 113 offline tests, passing identically with GPUs hidden and with eight visible.
Reassembly fairness, at capacity 16 with one camera submitting 100 incomplete frames beside one
submitting 2: `greediest_camera` leaves `{quiet: 2, loud: 14}` with all 86 evictions charged to
`loud`; `oldest_frame` leaves `{loud: 16}` and the quiet camera loses both. End to end, the
`replay` source into the mock backend into the `jsonlines` sink: 6 frames in, 6 events out,
every tag accounted for, none duplicated.

---

## 2026-08-23 — The ingest plane: one stateful actor per camera

**Why.** `src/shipinfer/ingest/` was empty, so the server could not read a camera at all.
PLANE 1 of `references/bitbucket-subfaceid/docs/new-system-architecture.md`: 50 cameras in,
tagged frames out, no inference in the path.

**Seams introduced.**

| Seam | Where | Extension point |
|---|---|---|
| Video sources | `ingest/sources/` | `@SOURCES.register` (gstreamer / pyav / replay) |
| Frame consumers | `ingest/sink.py` | the `FrameSink` protocol — `pipeline` supplies the production one |
| Environment contract | `src/shipinfer/envs.py` | one `EnvVar` per variable, typed, with `describe()` |
| Ingest errors | `core/errors/ingest.py` | four types, one per operator action |

**Decisions recorded.** ADR-011 — ingest depends on a sink protocol it owns, not on the
scheduler.

**Notable.** Two bugs found by the tests, both in code that only runs when something is
already wrong: `ExponentialBackoff.peek()` overflowed a float at ~attempt 1000 (a camera at
the 30 s cap reaches that in under nine hours — a guaranteed actor-thread death on a
long-running deployment), and the `frame_id` counter had to live on the actor rather than the
source, or a reconnect reissues frame 0 and hands a tracker a duplicate `(camera_id,
frame_id)`. Reconnect is exponential + jittered + capped, and a *frame* resets it, not a
successful connect — an RTSP source that opens and delivers nothing is the common real
failure and must not read as healthy.

Two tightenings to `scripts/hooks/check_layers.py` fell out of the work: `from shipinfer
import x` is now checked identically to `import shipinfer.x` (the two spellings had different
rules and the lax one was winning by accident), and `core` may not import the non-layer
top-level modules that every other layer can.

**Evidence.** 163 offline tests, no GPU, no GStreamer, no PyAV, no camera — the `replay`
source over a generated frame directory is what makes that possible and is what the
50-camera stress test will use. Reconnect tests assert the *sequence* of delays
(`[0.1, 0.2, 0.4, 0.8, 0.8, 0.8]`), not that a retry happened. No throughput measurement was
taken; `shipinfer bench` against `CountingSink` is the next step and is not claimed here.

---

## 2026-08-22 — Initial system: scheduler, runtime, backends, server, native kernels

**Why.** The previous generation (`references/bitbucket-subfaceid`) ran every model on GPU 0
and starved quiet cameras through a shared evict-oldest buffer. The requirement is 50
cameras × 20 fps across 16 GPUs with balanced load and bounded tail latency.

**Seams introduced.**

| Seam | Where | Extension point |
|---|---|---|
| Registry primitive | `core/registry.py` | eager + lazy registration |
| Placement policies | `scheduling/policies/` | `@POLICIES.register` |
| Request queues | `scheduling/queues/` | `@QUEUES.register` |
| Batchers | `scheduling/batching/` | `@BATCHERS.register` |
| Backends | `backends/` | `@BACKENDS.register` / `register_lazy` |
| Allocators | `runtime/memory/` | `@ALLOCATORS.register` |
| Graph caches | `runtime/graphs/` | `@GRAPH_CACHES.register` |
| Image ops | `runtime/ops/` | `@IMAGE_OPS.register` |
| Log sinks | `core/logging/sinks/` | `@SINKS.register` |
| Metrics exporters | `core/metrics/exporters/` | `@EXPORTERS.register` |
| Response caches | `server/cache/` | `@RESPONSE_CACHES.register` |
| CUDA providers | `runtime/providers/` | `@PROVIDERS.register` (custom variants only) |

**Decisions recorded.** ADR-001 through ADR-009 — the pure core, one-thread-one-GPU, torch
as substrate, locality-aware spillover, fair queueing, the Triton repository layout, the
Python/C++ split, CUDA-graph buffer lifetime, and the opt-in response cache.

**Evidence.** 149 offline tests (no GPU) + 12 GPU tests; 998 req/s at p99 7.6 ms with
11.7–13.2% per-device share across 8 × A5000; fused letterbox 1.41× faster than torch with
bit-identical output.
