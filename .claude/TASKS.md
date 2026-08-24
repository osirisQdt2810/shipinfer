# Open work

Order settled by the operator (V49): **Plane 3 and Triton first; the ≥5× whole-system
optimisation is the final goal.** C10 (tmux) was left to me — decided below.

The ledger `scripts/hooks/unfinished_work.py` reads. **While any line is `[ ]` or `[~]`, the
session may not end** — the `Stop` hook refuses and hands the list back. Written because
intention failed twice: a PR was opened, "I'll continue" was written, and the turn ended.
Ending a turn *is* stopping, so the fix had to be a mechanism rather than a promise.

Its one residual failure mode is **work that never gets written here** — a hook cannot see what
it is not told. So a request becomes a line the moment it arrives, not at the end of a batch.

| mark | meaning |
|---|---|
| `[ ]` | open — blocks stopping |
| `[~]` | in progress this session — blocks stopping |
| `[x]` | done, evidence on the line |
| `[!]` | blocked on the operator, question on the line — does not block |
| `[-]` | dropped by the operator, reason on the line |

A stopping point is exactly three things: everything is `[x]`/`[!]`/`[-]`; an action needs
confirmation before it is safe; or the operator interrupted. **Opening a PR is not one. Pushing
is not one. Writing a summary is not one.** `AWAITING-OPERATOR:` on its own line also stands the
hook down, for when the operator asked to see something before it is executed.

> ## Z · The final gate — never remove this line (V52)
>
> **When every item below is closed, re-read `docs/qa/user.md` end to end and check each
> request against the repository.** Not the standing-rules index — the *whole file*, including
> the verbatim sections, because a request made in passing does not always become a rule. Write
> the result into `docs/qa/verification.md` with evidence per line, and say plainly what is
> still not done rather than reporting "all complete".
>
> This exists because the ledger is my summary of what was asked, and a summary can lose
> something. `user.md` is the record.

---

## Phase 1 · C++ style and layout (blocks every later C++ edit)

- [x] **A1 · `csrc/` mirrors `src/`** — built and 41 checks green; (V40, V50). Directories moved, headers renamed;
      **includes not yet updated, so it does not build.** Applies to `3rdparty/shipvision/csrc`
      too, which still has a split `include/` + `src/`. Names reused from Python (V48):

  | python | c++ | names kept |
  |---|---|---|
  | `core/types/` | `core/types.{h,cpp}` | `Tensor`, `DeviceBuffer` |
  | `runtime/platform.py` | `core/platform.h` | `GPU_CHECK`, `gpu*` |
  | `scheduling/queues/fair.py` | `scheduling/queues/fair.h` | `FairPriorityQueue` |
  | `runtime/ops/` | `runtime/ops.{h,cu}` | `ImageOps`, `letterbox`, `crop_batch` |
  | `backends/tensorrt/engine.py` | `backends/tensorrt/engine.{h,cpp}` | `TrtEngine` |
  | `server/instance.py` | `server/instance.{h,cpp}` | `ModelInstance` |
  | `server/model.py` | `server/model.{h,cpp}` | `Model` |
  | `pipeline/graph/state.py` | `pipeline/graph/state.h` | `FrameState`, `ObjectBatch` |
  | `pipeline/graph/graph.py` | `pipeline/graph/graph.{h,cpp}` | `PipelineGraph` |
  | `pipeline/reassembly/collector.py` | `pipeline/reassembly/collector.{h,cpp}` | `FrameCollector` |
  | `ingest/camera/actor.py` | `ingest/camera/actor.{h,cpp}` | `CameraActor` |
  | `ingest/sources/` | `ingest/sources/replay.{h,cpp}` | `ReplaySource` |
  | `benchmarks/harness/sampler.py` | `obs/sampler.{h,cpp}` | `OccupancySampler` |
  | `cli/commands/bench.py` | `cli/bench.cpp` | `main` |

- [x] **A2 · `X.cpp` pairs with `X.h`** (V44) — ten headers renamed.
- [x] **A4a · `core/platform.h`** — `GPU_CHECK` plus `gpu*` aliases, matching shipvision's
      existing `core/platform.hpp`. I had invented a second convention; two spellings of one
      thing is worse than either.
- [x] **A3 · Indent inside `namespace X {`** (V45) — `.clang-format` with
      `NamespaceIndentation: All` plus a pre-commit hook, so it is enforced not remembered.
- [x] **A4b · Delete `SHIPINFER_CUDA`/`cuda_check`**, move every call site to `GPU_CHECK` and
      `gpu*`. Also answers PR #8's finding that `csrc/` dropped ADR-003's ROCm rationale.
- [x] **A5 · Fix every include, rebuild, re-run the 41 C++ checks.**

## Phase 2 · The two open PRs

- [ ] **B1 · PR #6** — round-2 blocking answered and pushed; awaiting re-review, then merge.
- [~] **B2 · PR #8, four blocking**, all real:
      1. per-frame `gpuMalloc`/`gpuFree` on the dispatch path, with the reusable buffer I
         declared voided by `(void)`. `gpuFree` is device-blocking, so this reintroduces on
         every frame the stall I claimed to have removed.
      2. a skipped branch is indistinguishable from a failed stage: every ship-only frame seals
         `Incomplete` with `missing=["person_embedder"]`, which is most of the fleet.
      3. reassembly eviction destroys a frame with no event and no per-camera attribution and
         counts it as reported — the inversion ADR-005 exists to prevent.
      4. no ADR and no FEATURE_LOG entry for a second data plane; three real contradictions
         with ADR-007, ADR-003, ADR-005.
- [ ] **B3 · PR #8, five should-fix** — `complete()` compares sizes not inclusion;
      `sharding.py` belongs with its launcher; publish the full counters beside 390.5; a CI job
      that compiles `csrc/`; the nits.

## Phase 3 · Plane 3 — tracking (first, per V49)

- [ ] **C2a · Repackage `shipvision/tracking/`** (V50) in the shape of roboflow/trackers
      `src/trackers/core`: a **package per algorithm**, so an algorithm carries its own
      supporting classes. Today `trackers/{sort,bytetrack,botsort,ocsort,deepsortv2}.py` are
      flat files beside `association/`, `motion/`, `pool.py`.
- [ ] **C2b · Repackage `shipvision/mtmc/`** the same way.
- [ ] **C2c · C++ implementations of every MOT/MTMC algorithm** in `shipvision/csrc`, reachable
      through the ops binding (V50). A Python version existing is fine; both must exist.
- [ ] **C2d · Wire Plane 3 into `shipinfer`** — the DAG ends at the embedders and tracklets go
      nowhere; `shipinfer` imports only `shipvision.detection.engine_build`.

## Phase 4 · Triton parity

- [ ] **C3 · `docs/qa/triton.md`** (V26) — eight features, none implemented: per-model
      statistics endpoint, request tracing with named timestamps, `graph_spec` from the batcher,
      rate limiter, model warm-up from real samples, explicit load/unload, ensemble scheduling
      as a first-class scheduler.

## Phase 5 · Everything else still owed

- [ ] **C4 · RTSP in the benchmark** (R55) — tests cover it; the benchmark replays JPEGs, so
      NVDEC has never been exercised by a measurement.
- [ ] **C5 · Benchmark tiers algo and kernel** (R44) — only the system tier exists.
- [ ] **C6 · PR #3 findings 2, 6, 7, 8** — Kafka `produce()` counted as success;
      `actor.stop(0.0)` abandoning every camera thread; the `pending_frames` gauge going stale
      per camera; an uninterruptible reconnect `time.sleep`.
- [ ] **C7 · `wheels.sh` does not stage the TensorRT wheel** — a fresh cache breaks every bench.
- [ ] **C8 · `conftest.py` calls `device_count()` during collection**, so an unhealthy driver
      reddens the *offline* tier, which ADR-001 says must need no driver.
- [ ] **C9 · `shipvision` NV12 work** — 1021 lines uncommitted, 26 tests passing, 14 skipped for
      want of a native build. Its own PR in that repo (ADR-010).
- [x] **C10 · tmux — decided: not retrofitting.** The property tmux was asked for is that a
      long run survives a dropped session. `docker run --rm` already gives the load-bearing
      half — the container is not a child of my shell, so it survives — and every run already
      writes its occupancy log and console capture under `.artifacts/`, so the evidence outlives
      the run whether or not anyone was attached. What tmux would add is reattach-and-watch, and
      runs are now 40–70 s. Adding it would put a second supervisor between me and a container
      that already has one. **Revisit if a run ever exceeds ~10 minutes** — an engine-build sweep
      would qualify.
- [ ] **C11 · The `std::memcpy` audit** (V28) — deferred by the operator until the system is
      complete. Now also covers `csrc/`, which added several.

## Phase 6 · The final goal (V49)

- [ ] **C1 · ≥5× counting-simulation, whole system.** Measured: baseline 868.2 img/s against the
      C++ plane's 390.5 → **0.45×**. The interpreter is no longer the wall.
- [ ] **C1a · Profile before optimising** (V54, and the operator is right that I was reasoning
      from symptoms). The buffer-growth log says *which queue* grows, not *where the time goes*.
      Needed: Nsight Systems over one saturated run, plus per-stage host timings, so the answer
      is a flame graph rather than an inference. Everything in C1 waits on this.

---

## Z · Final gate

- [ ] **Z1 · Re-read `docs/qa/user.md` end to end** and check every request — verbatim sections
      included, not just the standing-rules index — against the repository. Result into
      `docs/qa/verification.md` with per-line evidence, stating plainly what is still not done.
