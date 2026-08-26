# Journal

Newest on top. One entry per working session: what changed, what it cost, what is next.

---

## 26 Aug (evening → night) — T3 shipped end to end; C44 answered; B4/B5 closed

- **#26 (`service` tier) merged 17:39 UTC after five review rounds** — every finding real:
  round 1 the owner threading (ONE sweeper, workers never touch a ring); round 2 `_DeviceSpan`
  retention + saturation held with the slot (not failed as text); round 3 probe-not-spin and
  the `Model` admission seam (`admit_local` / `try_dispatch_local` / `count_local_rejection`);
  round 4 `WireRefusedError` spills, per-lane stamping, typed status codes, D2H-before-claim,
  in-flight depth, reader snapshot, `pending_timeout_ms`; round 5 APPROVE.
- **#27 (bench/topology) merged 18:00 UTC, APPROVE round 1**: both sweeps re-run on the merged
  tree — every rung SUSTAINED to 72 img/s; B put every person crop on the crowded GPU
  (1572/0/0), C spread them (847/345/382) with ship_embedder shared back. T3 closed; T3b
  (crowd fan-out) opened as the honest residual.
- **#28 (ci: C++ offline tier + prompt rename) merged 18:06 UTC by hand under V109** after
  py3.10/py3.12 tests passed and the review job failed with exactly the documented
  workflow-validation error. B4 and B5 closed.
- **C44 answered by the Nsight timeline** (12×5, GPUs 3–5, 30 s, merged tree): GPUs ~14%
  kernel-busy; host threads blocked in `cudaMemcpyAsync` (22.9 s, max 116 ms) and ~400 k
  kernel launches (~13 k/s); the per-crop torch pointwise storm is 74 launches/frame; D2H
  tails hit 39 ms (pageable, segmenter-sized). Levers for C1: fused `crop_batch`, pinned
  D2H staging for big outputs. Evidence: `c44-nsys-stats.txt` (scratchpad).
- Operator: **V108** (T4 is a topology, not a competitor benchmark — four topologies under
  one backbone), **V109** (standing self-merge grant), **V110** (the tier must hold for the
  full DAG), **V111** (the stale repo copy of user.md — answered by the snapshot PR).

## 2026-08-26 — Topology C is built, and the first real runs find what the fakes could not

**What merged.** #21 (the docs snapshot) after three review rounds, each of which found something
real: the request log carried two assignments for V41–V51 (renumbered once, every citation shifted,
verified mechanically — Section 1 runs V1–V90 plus V91, the appendix V41–V51, no duplicate, no gap),
the design doc's pinned budget had three values for one quantity (derived once: 72 rings ≈ 0.9 GiB
on the box existing once, ≈ 0.44 GiB registered per process for 4 shards × 3 models), and its error
block declared `RingFullError` a sibling of the type the spill path catches (it is a
`QueueFullError`, as built). #22 (the submodule at the per-algorithm tree; the tracking stage imports
`shipvision.mot`). Open: #23, the RTSP path, with the baseline gate fix (C48) folded in.

**T3, steps 2–4, on three stacked branches.** The wire and the proxy (`RemoteInstance` is a
`Placeable` with `device = cpu`; `ResultReader`, `RingIngress`), the `service` topology
(`ServiceSettings`, `Topology.shard_environment`, `ServiceMesh`, `Model.attach_remote`,
`InferenceServer` joining the tier after the models load and leaving it first on stop), and the
harness driving the shards (`benchmarks/harness/shards.py`: one child per shard through the real
`Fleet`, an explicit `shard_cameras` split to model a crowd that moved after planning). ADR-015 and
the feature-log entry are on the topology branch. Also on the way: every fleet shard ran `serve`
with no HTTP, so none was addressable (`serve_command(http_port_base=)`); `Model.infer_local` so
work that crossed once is never re-routed; `slot_bytes` = 1.5 MiB + 64 KiB because the heads travel
in the slot; a ring `open` no longer registers with the resource tracker (bpo-38119).

**What the real runs found.** 37 fake-model tests were green while every real shard died at start:
the edit that inserted `attach_remote` had landed between `total_depth`'s `@property` and its `def`,
making `attach_remote` a property and `total_depth` a method. The two-process multigpu test (two
`serve` processes through the real `ServiceTopology`, GPUs 3 and 4) found the first; the harness's
occupancy probe — reading `total_depth` as main declares it and handing a bound method to
`json.dumps` — found the second, as three shards with a metadata line and no samples. Both fixed,
and `test_service_engine.py` (two `InferenceServer`s in one process as shards 0 and 1) now covers
the engine's own path. The rule this confirms: a seam is proven by the first run through the real
object on both sides, and the fakes are for the protocol, not for the wiring.

**Evidence so far.** The two-process test: 24 requests to shard 0, 19 executed there, 5 by shard 1
through the ring. The fleet under a crowded split (6/3/3 cameras on GPUs 3–5, swept from 12 × 2 fps
to ×3): every rung SUSTAINED up to 72 img/s — 36 on the crowded GPU — and the per-device table shows
the fleet's locality exactly (every person crop on GPU 3, every ship crop on 4 and 5). `person_2K`
yields about one person crop per frame, so the crop stage is nowhere near binding at these rungs
and C has nothing to balance there; the informative rungs are ×4–×5, where the first run failed on
the crowded shard's embedder time-outs. The `service` sweep then needed three more fixes the fakes
could not see (the wire refused the pipeline's device-resident tensors; `abandon` existed only in
the tests' imagination; no real batch fit a 1.5 MiB slot — slots are per model and direction now),
and with them, plus the bounded replies and the ring's closed-handle rule (#25 rounds 1–2), **C's
confirmation sweep is clean: x1–x3 all SUSTAINED to 72 img/s — B's own ceiling — zero errors, the
borrow visible at every rung (x3: person_embedder 742/377/389 across GPUs 3/4/5 against B's
1582/0/0)**. With ~1 crop per frame the tier has nothing to *win* here — C matches B and spreads
the crop work; its win case needs the crowd fan-out the sizing assumes, which this dataset cannot
produce.

## 2026-08-25/26 — The split lands piece by piece, and the review keeps finding the unwired guard

**What merged.** shipinfer #12–#17 in order — pipeline, the bench's other two tiers, the offline
tier's hidden devices, the C++ data plane (six review rounds), the infra/docs slice with the
`Stop` hook, and the three reasons the fused kernels were unreachable. shipvision #6–#10: reid,
mot, mtmc, eval/tune, and S1 — the `tracking` → `mot` rename with one package per algorithm.
Open at the end of the stretch: shipinfer #18 (the topology seam and the fleet, review round 1
answered) and shipvision #11 (S3: the native tree, bindings and build in the per-algorithm
layout — `342 passed` on the native tier inside the container, `2025 passed` for the whole tree).

**S2 does not exist.** The ledger promised "StrongSORT and BoostTrack and the Python tweaks" as
the second slice of the V79 restructure. `git ls-tree` over every remote ref finds no such file,
and the remaining Python diff between the source branch (08:39) and main is main being *newer*
(#3, #5, #9 landed 14:27–19:13). Struck with the evidence, not deferred — a ledger line that
describes work nobody can do is the kind that gets "continued" forever.

**The port is real now (P1a–d).** `FairPriorityQueue<T>`, the five policies behind a registrar,
the dispatcher, `ModelInstance` as thread + queue + window, `Model::infer` → `std::future`, the
graph as stages over it, and a bench that runs the Python plane's shape: ~390 img/s at 48
workers, ~470 at 96, balanced across GPUs 2–5 to 1%, 0 failed, 0 timeouts. Rebased onto main's
squash of #15 with an identical base tree (`git diff` empty where it matters); the PR is queued
behind #18.

**What the review found, and the shape it keeps having.** The `Stop` hook parsed only below a
`## Now` heading the ledger never had — a mechanism that reported nine open items as "clear",
worse than the reminder it replaced. The same hook would have fired in the CI review job, where
none of its escapes are reachable. `scripts/build_native.py` asserted the phantom `is_available`
that the whole PR was about, so a good build failed the container recipe. `Fleet.supervise()`
consulted only `dead()` over a fleet `stop()` had just emptied, so Ctrl-C spun until SIGKILL.
`ShardPlan.instances_for` — the guard whose docstring promised half the instances per shard on a
shared GPU — had no caller. Round two of the same PR added three more of the same shape: the plan balanced *shards* while two
of four GPUs carried twice the load of the other two — the founding bug rebuilt one level up, hidden
by a metric that measured processes rather than the resource; a signal handler that called a blocking
`stop()` under a lock and deadlocked on the second Ctrl-C; floor division that silently dropped a third
of a device's configured instances. And one of my own making: an append to the append-only feature
log that duplicated the file end to end. Every one of these was **a guard written and not wired**, a
name fixed in one place and not the other, or a metric measuring the wrong thing. The tests that now pin them were written to fail on the
previous commit first, and did.

**The RTSP path, run for the first time.** The ledger's C4 said "wired and tested offline; not yet
run", and six container runs later that sentence reads as a warning. Each run died one layer
deeper, each on a defect no offline test could see: NVDEC's decoder opening a GL display the
headless container does not have; fifty camera threads racing `Gst.init` so half of them probed
an empty plugin registry and reported "no decoder found" on an image with three; PyGObject's
lazy attribute loading racing the same way (`'GLib' object has no attribute 'Idle'`); GIO's
default proxy resolver, libproxy, throwing a C++ exception through the C boundary and
terminating the process (`Unable to read configuration`) with fifty cameras connected and zero
frames read; and the appsink's methods missing because nobody had loaded the GstApp typelib.
Five commits on `fix/rtsp-headless-decode`, each with a test that fails without it. The sixth
run decoded fifty streams through NVDEC and pushed frames through the whole graph — and the
harness refused the number, correctly: one interpreter generated 2% of the offered load. The
RTSP measurement is a 12 × 5 run; the 50 × 20 one is the fleet's to make.

**What it cost.** The org's monthly spend limit cut off both background agents mid-task (the
S3 coder had already built the extension in the container; the T3 planner had read nothing).
The rest of S3 was done in the main session; the T3 plan is still owed and will be written here
rather than delegated until the limit resets. One host-run refusal from the advisory hook, for
naming `tests/runtime/test_native.py` in a host `pytest` command — the offline tier is exempt,
but that file imports torch and the hook is right to stop and ask.

**Next.** The queue: #18 → `port/p1-scheduling` → the docs snapshot (this journal, the ledger,
V78–V90, the two CLAUDE.md rules) → C47's CUDA-free test binaries (every C++ test binary still
links four accelerator libraries, so the fairness invariants cannot run on a machine with no
driver). Then P4 ingest, P5 config-in/events-out, P6 the parity harness; T3 (`service`) and T4
(DeepStream) against the topology seam; after shipvision #11, the parent's
`pipeline/graph/tracking.py` switches to `shipvision.mot` with the submodule bump in its own
commit.

---

**The request log, numbered once.** Review of the docs snapshot (#21) found the request log
carrying two assignments for V41–V51: the verbatim appendix's C++-convention requests, cited
by ~18 rules-index rows and the ledger, and the 24-Aug shipvision requests added to Section 1
under the same numbers. Repaired in one commit: V40 is the `csrc` mirrors `src` request the
index already cited, the duplicate of V49 folded into V49 with its time, the 24–26 Aug entries
renumbered V52–V90 (+9) with every citation in TASKS, CLAUDE.md, this journal and the design
doc shifted the same way, and main's misplaced `V43 — 14:40` (the trackers-the-services-used
note) is V91. Verified mechanically: no duplicate, no gap, no citation that resolves to nothing.
The rule-value paraphrases (V86, V88–V89) moved from the verbatim appendix into the rules table.

## 2026-08-23/24 — The benchmark answers, and the answer is no

**The number exists now.** Baseline **868.2 img/s** measured capacity at 50x20; ShipInfer
**81.4 img/s** measured capacity, found by `--sweep` saturating at the x2 rung (120 img/s
offered from 12 cameras). Ratio **0.09x** against a 5x target — missed by ~53x.

**Where the wall is.** Not the GPUs. At 120 img/s offered, every model queue sustained its
full offered rate with a CI straddling zero — the detector took 120 and retired 120. The
queue that grew, at +38.7/s, is the *pipeline* queue: in front of the Python worker pool and
behind nothing. Not the worker count either, which was swept rather than assumed:

    workers=24    87.6 img/s
    workers=96    81.4 img/s
    workers=192   85.0 img/s

An 8x range for under 8% of movement, non-monotonic. What is left is the interpreter — 12
decode threads plus up to 192 worker threads doing per-frame Python work in one process.
Section 9 of `new-system-architecture.md` already puts decode in separate **processes**, and
this driver does not. **The next PR is the process split, not a kernel.**

**Six review rounds on PR #3, 8 blocking items in the last one.** The one that mattered:
the harness was structurally incapable of producing a speed-up. `is_rate` refused SATURATED
as "a bound", which has the buffer-growth methodology backwards — a saturated run is the
*only* regime in which `offered - growth` is exact. Both systems are offered the same load
by construction, so either neither saturated (each reported the offer back, 1.00x) or one
did (comparison refused). A number is now a CAPACITY, a FLOOR or NOTHING, `ratio_of` is the
single place a pair combines, and floor-over-capacity reads `>= Nx` while
capacity-over-floor reads `<= Nx`.

Three of the eight were holes in guards I had added in *earlier rounds of the same review*.
The pattern: the idea right, the detail wrong, always in the flattering direction.

**Other real bugs found by review this round.** `redact()` failed *open* on a `/` or `@`
inside a password — `urlsplit` follows RFC 3986 where `/` ends the authority, so
`parts.password` was None and the URI was echoed whole, into `SourceOpenError` ->
`_last_error` -> the ingest health endpoint, on every retry. The fair queue's drain skipped
`notify_all` after its row-budget exits became `return`, so a blocked producer slept the
whole 500 ms timeout instead of waking at 50. `pipeline_sink_failures_total` could never
increment, because `emit` is documented "never raises" and the runner counted failures from
an `except` around it — so a dead broker lost every event with a green dashboard.

**The constraint that shaped what could be measured.** The baseline binary only survives
while *saturated*: static-batch plans, `setInputShape` with whatever batch it assembled, so
a partial batch throws inside a worker and aborts the process. It dies at 60 img/s. The two
capacities therefore come from different offers, which is fine for a capacity comparison but
has to be said out loud.

**Still absent.** Plane 3 (MOT/MTMC) does not exist — `shipinfer` imports only
`shipvision.detection.engine_build`. The `std::memcpy` audit is deferred by the operator
until the system is complete.

---

## 2026-08-22 — Project reset: from Sale Hunter scaffolding to ShipInfer

**What this session did.** Replaced the inherited scaffolding (a Shopee desktop-app
template) with the real project: a Triton-shaped multi-GPU inference server for the ship /
person perception pipeline described in `references/bitbucket-subfaceid/docs/`.

**Built.**

- `core/` as packages, not modules — `types`, `request`, `errors`, `settings`, `logging`
  (with a bounded async sink), `metrics` (with pluggable exporters), plus the `Registry`
  primitive every extension point uses.
- `scheduling/` — the part this project exists to own. Fair per-camera queueing with
  priority lanes, dynamic batching windows, five placement policies, and a dispatcher that
  spills to the next-shortest queue rather than dropping a frame.
- `runtime/` — the accelerator seam, on torch. Device manager, streams, CUDA graph capture,
  a pinned staging pool, and three image-op implementations (numpy / torch / native).
- `backends/` — mock, TensorRT (engine + persistent bindings + graph replay), ONNX Runtime,
  TorchScript.
- `server/` — instances, models, a validated ensemble DAG, response cache, health, KServe v2.
- `native/` — C++17 + CUDA/HIP behind pybind11: fused letterbox, batched crop, device NMS.
- A demo `model_repository/` carrying the real DAG on the mock backend, so the whole thing
  is runnable and testable anywhere.

**The correction that shaped the result.** The first version of `runtime/` hand-rolled a
CUDA layer against `cuda-python` — its own caching allocator, pinned-memory management,
stream wrappers, raw graph capture. It worked and it was slower than torch, missing exactly
the parts that are hard. Rewritten onto torch (ADR-003). The hand-written code was kept as
registry-selectable `custom_*` variants: runnable documentation for what the library does,
and the other half of the parity tests.

**Evidence.**

- offline suite: 149 passed, 12 deselected, ~7 s, no GPU
- GPU tier: 12 passed on 8 × RTX A5000
- `shipinfer bench person_embedder --cameras 50 --fps 20 --seconds 3 --skew 8`:
  3000/3000 completed at 998 req/s, p99 7.6 ms, 0 rejected; per-device share 11.7–13.2%
  across all 8 GPUs; every quiet camera served despite 8× skew
- fused letterbox, 8 × 1080p → 640², writing into a torch CUDA tensor: **9.7 ms (822 img/s)
  vs torch's 13.7 ms (585 img/s)**, output bit-identical

**Notable bugs found and fixed on the way.**

- `np.clip(..., out=fancy_index)` in the numpy crop path neither wrote back nor cast — it
  raised, and only for some dtypes. Caught by the GPU parity test; host-tier tests added so
  it would have been caught without a GPU.
- The ensemble DAG validated tensor *names* but not shapes, so a 512×512 ship crop fed a
  256×128 embedder and failed on the first frame instead of at start-up. Validation now
  type-checks every edge.
- The bench discarded futures completed during intermediate waits and reported 496 of 3000.

**Next.**

- `pipeline/` and `ingest/` are declared in the layout but not yet implemented: NVDEC
  decode, per-camera actors with backpressure to the decoder, and the `(cam, frame)`
  reassembler.
- Kafka result publishing, to hand tracklets to `motservice`.
- A real TensorRT engine in the repository, so the GPU tier exercises the production path
  rather than the mock.
