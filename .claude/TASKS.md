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

> ## Z · The final gate — never remove this line (V61)
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
      existing `core/platform.h`. I had invented a second convention; two spellings of one
      thing is worse than either.
- [x] **A3 · Indent inside `namespace X {`** (V45) — `.clang-format` with
      `NamespaceIndentation: All` plus a pre-commit hook, so it is enforced not remembered.
- [x] **A4b · Delete `SHIPINFER_CUDA`/`cuda_check`**, move every call site to `GPU_CHECK` and
      `gpu*`. Also answers PR #8's finding that `csrc/` dropped ADR-003's ROCm rationale.
- [x] **A5 · Fix every include, rebuild, re-run the 41 C++ checks.**

## Phase 2 · The two open PRs

- [x] **B1 · PR #6 — MERGED.** Two review rounds; both blocking findings were vacuous tests of mine that a mutation test would have caught before I pushed.
- [x] **B2 · PR #8, four blocking** — all four fixed and pushed (df41e29); finding 1 moved 390.5 -> 400 img/s (2.5%), finding 2 took Complete events from a minority to 28656/28808:
      1. per-frame `gpuMalloc`/`gpuFree` on the dispatch path, with the reusable buffer I
         declared voided by `(void)`. `gpuFree` is device-blocking, so this reintroduces on
         every frame the stall I claimed to have removed.
      2. a skipped branch is indistinguishable from a failed stage: every ship-only frame seals
         `Incomplete` with `missing=["person_embedder"]`, which is most of the fleet.
      3. reassembly eviction destroys a frame with no event and no per-camera attribution and
         counts it as reported — the inversion ADR-005 exists to prevent.
      4. no ADR and no FEATURE_LOG entry for a second data plane; three real contradictions
         with ADR-007, ADR-003, ADR-005.
- [x] **B3 · PR #8, five should-fix** — all taken. `complete()` uses inclusion, the batch assert is in, the counters are attributed, `sharding.py` left the PR (B3a), and the nine nits are closed: `<algorithm>` included, `put`'s O(cameras) scan replaced by `try_emplace` (it ran on **every** frame — fifty strings walked a thousand times a second to answer a question the map had already answered), `drain`'s comment no longer claims empty means closed, the two unused hooks say why they exist, and the test harness prints `FAIL: expected: …` so a regression stops reading as good news. A `csrc` CI job is B5, separately, because it edits `.github/workflows/**`.
      `sharding.py` belongs with its launcher; publish the full counters beside 390.5; a CI job
      that compiles `csrc/`; the nits.

## Phase 3 · Plane 3 — tracking (first, per V49)

- [x] **C2a · Repackage `shipvision/tracking/`** — done, per-algorithm packages under `core/` plus a TRACKERS registry; adversarial verification found the builder had broken 5 leaf import paths and an efficiency short-circuit, both restored. (V50) in the shape of roboflow/trackers
      `src/trackers/core`: a **package per algorithm**, so an algorithm carries its own
      supporting classes. Today `trackers/{sort,bytetrack,botsort,ocsort,deepsortv2}.py` are
      flat files beside `association/`, `motion/`, `pool.py`.
- [x] **C2b · Repackage `shipvision/mtmc/`** — done; verification found `matrix/` had become a module (4 leaf paths gone) and `clustering.base.CLUSTERERS` dropped. Both restored as shims.
- [x] **C2c · C++ implementations of the MOT/MTMC algorithms the reference services use** (V91
      narrowed this from "every algorithm"). Established from the references rather than
      guessed:
      - **`motservice` uses `deepsortv2` and nothing else** — its README says "currently
        supports only deepsort", and `config/algo` exposes exactly that one name.
      - **`mtmcservice` uses** `aic_cluster` (RAC-based agglomerative clustering),
        `aic_matrixbuilder` (appearance, euclidean, threshold 0.55) and `spatial_gating`
        (appearance plus a spatial/aspect-ratio gate), driven by `AICTracker` and
        `VTXTracker` — read off `config/algorithms/mctracker.json`.
      C++ already has `sort` and `bytetrack`, which predate this narrowing and can stay.
      `botsort`/`ocsort` are **out of scope** unless something starts using them.
- [x] **C2d · Wire Plane 3 into `shipinfer`** — the DAG ends at the embedders and tracklets go
      nowhere; `shipinfer` imports only `shipvision.detection.engine_build`.

## Phase 4 · Triton parity

- [x] **C3 · `docs/qa/triton.md`** (V26) — eight features, none implemented: per-model
      statistics endpoint, request tracing with named timestamps, `graph_spec` from the batcher,
      rate limiter, model warm-up from real samples, explicit load/unload, ensemble scheduling
      as a first-class scheduler.

- [x] **B3a · `sharding.py` removed from PR #8** — review was right that it is the wrong PR:
      236 lines plus 175 of tests, referenced by nothing, for a process-split approach the C++
      data plane superseded. The operator's standing rule is to delete what is redundant rather
      than carry it. Recoverable from history at `75fef9d` if a multi-process launcher is ever
      wanted for a 16-GPU box, which is a different reason from the one it was written for.
- [x] **B3b · `sharding.py` restored, and the "superseded" claim corrected** (V76). The
      operator's vLLM observation is the reason: vLLM's `MultiprocExecutor` spawns
      `context.Process` per GPU worker and talks ZMQ, and none of its twenty-one
      `threading.Thread` uses is on the model-execution path. So process-split and the C++
      plane answer *different* questions — "is one process using the machine?" versus "is the
      interpreter in the hot path?" — and 390 img/s on five of forty-eight cores is evidence
      for the first one going unanswered. Restored from `75fef9d` with the docstring's
      supersession claim replaced by that reasoning, plus **40 tests it never had**. Two of my
      first tests were wrong and the run said so: `describe()`'s header is `2 shard(s),` so
      `"shard "` matches twice not three times, and — the interesting one — round-robin also
      splits the 4-busy/12-quiet fleet perfectly, because four and twelve both alternate
      cleanly into two. That fixture cannot tell greedy from round-robin, so a test using it to
      claim greedy wins asserts nothing. The suite now compares against **both** round-robins
      on an odd-count fleet, and states outright that greedy merely ties on the even one.

- [x] **C24 · shipinfer PR #9 (`core-subsystems`), review round 1 — three BLOCKING, all real,
      all fixed in `1b1263b`.** (1) `model_control='explicit'` validated on a branch with no
      control endpoints — reproduced verbatim; the guard is unconditional again and a second
      validator refuses `EXPLICIT` until the piece with the endpoints replaces it.
      (2) `release()` bumped the semaphore and decremented `_held` under two lock acquisitions,
      so a waiter waking between them over-reported `peak_in_flight` — **I could not reproduce
      it**: two hammers (8 threads/0.2 ms/60 trials; 16 threads/no hold/300 trials) never saw the
      peak exceed the limit, and the second never saw it *reach* the limit. Fixed from the code,
      test pins the property against the source, and both the PR comment and the docstring say
      so rather than let a green threaded test imply coverage. (3) `TRACE_EVENTS` duplicated
      and the guard test moved out — `main` already had the `profiling.py` copy, so this piece
      added the second and removed the test; now a re-export and the test is back. Every fix
      mutation-verified. **Owed at rebase time:** `split/server` must replace the refusing
      validator with the relaxation (its `test_model_control.py:77,185` use `explicit`).
- [x] **C25 · The "claimed fixed" correction.** My commit messages on `split/server` and
      `split/pipeline` said #8's findings 5 and 6 were fixed. They were not — I had only carried
      the files over. Both are now actually fixed and mutation-verified: (5) `is_ready`,
      `stats`, `_ensembles_depending_on` and `__iter__` go through one `_models_snapshot()`
      helper, because four copies of `with self._lock` is how two got fixed and two did not;
      (6) Kafka binds the frame's own tag at `produce` time and the runner drains late verdicts
      after settling the current frame, so a refusal for `cam03/100` is no longer charged to
      `cam07/412`. Two more things found on the way: `frames_emitted` was declared and never
      incremented (the `== 0` assertion under a broken sink was true for the wrong reason), and
      two race harnesses I wrote were green on the broken code — deleted, with the structural
      check and an explanation in their place.
- [x] **C26 · shipvision PR #3 (`imgproc`), review round 1 — one BLOCKING, real, fixed in
      `3d03cb2`.** `gauss` was gated on `iou > threshold` like the other methods; Eq. (4) has no
      threshold, and the gate made gauss indistinguishable from linear below N_t (two vessels at
      IoU 0.40: 0.85 returned, 0.617 published). Reproduced, fixed, five distinguishing tests,
      four red with the gate restored. All three non-blocking notes taken: the crop-equals-resize
      claim is now scoped honestly and a test through `crop_batch` pins the deliberate one-pixel
      difference (red if the clamp changes); 82 bare test functions wrapped into classes; the
      `extent - 1` prose corrected. 472 pass, whole submodule.
- [x] **GPU hygiene checked** after the container runs: no compute apps, every device at
      ~15 MiB, no containers alive.
- [ ] **C22 · shipinfer PR #8 is also over the limit** (V80) — 18 commits, 119 files, 14.3k
      lines, and on its **fourth** BLOCKING review round, which is the symptom. Its seven
      blocking findings map onto the seams, which is itself the argument for splitting:
      1-2 the C++ reassembly race and the batch-abort that seals seven frames Complete;
      3-4 the half-pixel `crop_resize` offset and the missing parity tests; 5 the unlocked
      model-table iteration in `server/engine.py`; 6 a Kafka delivery failure charged to the
      wrong `(camera_id, frame_id)`; 7 the missing `-m gpu` evidence and the submodule pointer
      riding along with 13k lines — which is the reviewer making the operator's point.
- [ ] **B4 · The `platform.hpp` -> `core/platform.h` rename in the review prompt** needs its
      own PR. Reverted on `feat/cpp-data-plane` because a branch whose
      `.github/workflows/**` differs from `main` cannot run the review job at all — "Workflow
      validation failed. The workflow file must exist and have identical content to the
      version on the repository's default branch." That is the documented permanent exception
      in CLAUDE.md, and it cost PR #8 a review round. A one-line prompt fix is not worth
      blocking a PR's automation; it goes in a workflow-only PR that is merged by hand.

- [ ] **B5 · A `csrc` compile job in CI** — ~2 000 lines of data plane are green on one box
      only. Needs its own PR with B4, since both edit `.github/workflows/**` and a branch that
      does cannot run the review job.

- [x] **D1 · The integration review's three blocking findings** — all real, all reproduced,
      all fixed with a test verified red against the unfixed code: `native_version()` called
      a `version()` the extension never defined (an ownerless bug three lanes saw and none
      owned); the ensemble scheduler deadlocked refine-in-place by waiting on producers
      declared *after* the reader; and it resolved concurrent writes to one name by arrival
      rather than by declaration order.
- [x] **D2 · The tracking tests ran in no tier.** 42 of them skipped in the container and CI
      does not check the submodule out — so the one part of Plane 3 that is a *threading*
      correctness argument was tested nowhere. `test.sh` puts the submodule on PYTHONPATH
      (pure Python, no build), the skip is now per class rather than per module, and the
      concurrency properties need no tracker at all. **1005 offline tests, 0 skipped.**
- [x] **D3 · The per-camera lock had no test that could fail.** `nullcontext()` left all 42
      green. The first replacement was vacuous too, for a subtler reason worth recording: two
      threads racing one camera never both reach the tracker, because the ordering guard
      refuses whichever checks second — so a re-entrancy detector reports zero overlap with or
      without the lock. The property is now asserted from *inside* the critical section (the
      tracker checks whether its shard's lock is held), which a `nullcontext` cannot satisfy.

- [x] **D4 · PR #8's second review round** — the blocking finding and eight of the nine
      should-fix items, each with a test verified red: batched `queue`/`success` were reported
      exactly `batch_size` times too large; both of my previous round's fixes were
      *incomplete* (the version probe and the ensemble ordering) and review caught both; a
      worker that died read as alive; rejections were missing from `record_failure`;
      `ConfigurationError` was a 500; the ensemble traced no span of its own; `index()` blocked
      a readiness probe behind an unload; `release()` could leave `in_flight` negative; and a
      warm-up file could escape its version directory. Plus `execution.cuda_graph_batch_sizes`
      is now a **filter** on a mixed repository rather than a per-model assertion — treating a
      deployment-wide setting as a claim about each model made it unusable at all.
- [ ] **D5 · The one-crossing MTMC matchers are unreachable** from shipping code:
      `MTMC_MATCHERS.build("gated", backend="native")` resolves to the older pass-by-pass
      classes, so `_C.MtmcGatedMatcher` is only exercised through adapters defined inside the
      test file. Belongs in the submodule's own PR with C9 (ADR-010). Either wire it and
      measure what the crossings cost, or delete it — shipping two implementations and using
      the slower one is the thing the operator's "delete what is redundant" rule is about.

## Phase 5 · Everything else still owed

- [~] **C4 · RTSP in the benchmark** (R55) — wired and tested offline; **run six times on 26 Aug, then measured** (see below).
      `--source rtsp` points the bench cameras at `scripts/rtsp_serve.py` over a real socket,
      `benchmarks/harness/rtsp.py` owns the server's lifetime and refuses a run whose server
      never accepts or exits early, and the source is recorded in the metadata and printed on
      the console. 11 offline tests. The two sources measure **different things** — replay
      removes the decode path — so the README says a replay number is an upper bound on the
      RTSP one. **Run 26 Aug 02:31 on a quiet box (GPUs 2–5, load 19/48), and it did not survive
      start-up:** inside `shipinfer-gst:jammy` the ingest died with exit 139 after
      `gst_gl_display_gbm_new: could not find or open DRM device` (and `XDG_RUNTIME_DIR not set`) —
      `decodebin` ranks NVDEC's `nvh264dec` first and that element opens a GL display the headless
      container does not have. Two things to fix in the C4 PR: the pipeline must ask the hardware
      decoder for system memory (or fall back to `avdec_h264` when no display exists), and
      `bench.sh` must pass `SHIPINFER_GST_DECODER` through so the operator can force the decoder.
      Also: `bench.sh` refuses to start without the host-built baseline binary even for
      `--systems shipinfer`; staged from the port worktree for this run.
      **Third attempt (02:50, with the `video/x-raw` filter after `decodebin` from `fix/rtsp-headless-decode`):**
      still exit 134 — the GL display is created regardless of downstream caps — and a second fact:
      cameras with an explicit `h264` codec fail the decoder probe (`no h264 decoder found (tried
      nvv4l2decoder, nvh264dec, avdec_h264)`), so the container image has no software decoder and its
      only H.264 decoder is nvcodec's, which wants a GL display. The image cannot be rebuilt on this box
      (no network in containers). Next: pass `GST_GL_PLATFORM=egl GST_GL_WINDOW=surfaceless` through
      `bench.sh` and see whether a surfaceless EGL display satisfies nvcodec; if not, RTSP-in-container
      needs an image with `gstreamer1.0-libav`, which is an operator/infra step.
      **Probed inside the image (03:0x):** `gi` 3.50 (conda) over GStreamer 1.20.3, and `avdec_h264`,
      `nvh264dec`, `decodebin`, `rtspsrc`, `appsink` are all present; single-threaded, `x264enc !
      nvh264dec ! video/x-raw ! videoconvert` and the same through `decodebin` both reach EOS with no GL
      display. So the image is fine and the crashes are *races*: fifty camera threads touching
      `gi.repository` and `Gst.init` at once — the probe failures ("no h264 decoder found" while three
      exist) and `'GLib' object has no attribute 'Idle'` are the same race seen twice. Fixed on
      `fix/rtsp-headless-decode` (3 commits): the `video/x-raw` filter after `decodebin`/nvcodec, and the
      whole import-and-init of GStreamer under one lock (test with a fake `gi`: 16 threads → 1 init, all
      see the registry; red without the lock). **Fourth attempt** (hardware decode, fixed tree): no probe
      failures, no GL crash — and a new one: `terminate called … std::runtime_error: Unable to read
      configuration` after 28 s with fifty cameras connected and zero frames. The string lives in
      `libproxy.so.1` in the image: `rtspsrc` asks GIO for a proxy resolver, GIO's headless default is
      libproxy, and libproxy throws a C++ exception through the C boundary when it finds no GSettings or
      D-Bus. Fix (4th commit): `GIO_USE_PROXY_RESOLVER=dummy` set by `bench.sh` and defaulted in-process
      (`setdefault`, an operator's real proxy stays), with a test. **Fifth run:** the process survived the
      whole 40 s — and read nothing: every camera failed with `'GstAppSink' object has no attribute
      'try_pull_sample'`, because the appsink's *methods* exist on the Python object only when the GstApp
      typelib is loaded. Fix (5th commit): `_load_gst` loads `GstApp`, and `_try_pull_sample` falls back to
      the `try-pull-sample` signal. Sixth run in progress. Five defects on one code path that had never
      executed in the container — the ledger's "wired and tested offline; not yet run" was exactly right
      about what offline tests cannot show.
      **Sixth run (03:07): the RTSP path works end to end in the container** — fifty cameras decoded through
      NVDEC, frames flowed through detect → crop → embed, embedder stages timed out under the overload —
      and the harness refused the number: the generator delivered 20.9 img/s of the 1000 offered (2%), the
      one-interpreter wall the harness already documents. So the RTSP measurement is a 12 × 5 fps run
      (60 img/s, what one interpreter can generate) against the same load on replay; both in progress.
      The 50 × 20 RTSP measurement needs the fleet (one process per shard) driving the cameras — T2's gate.
      **12 × 5 result (03:10):** RTSP — `offered: 101.8 img/s achieved of 60 target (170%)`, detector
      SUSTAINED 101.8, the pipeline queue SATURATED (+103/s growth, sustained 0.0); replay — 60.0 of 60,
      every module SUSTAINED. Not like-for-like yet: the RTSP cameras delivered 1.7× the target rate, so
      the RTSP side offered a different load. **Why:** `scripts/rtsp_serve.py` never paced — the docstring
      credited `do-timestamp` + caps framerate, but the launch line had neither, so `multifilesrc !
      h264parse ! rtph264pay` pushed at socket speed. Two commits on the C4 branch: `identity sync=true`
      after `h264parse` (170% → 127%), then `single-segment=true` so the loop's restarted timestamps stay
      paced (→ 118%). **The real cause, found with a probe client inside the container: the cached
      fixture was encoded at 20 fps and the stream's own SPS timing paced it at 20 regardless of
      `--fps`.** Fixture cache keyed by rate (3rd bench commit) → `59.9 of 60 (100%)`, every module
      SUSTAINED, zero warnings; replay at the same load 60.0 SUSTAINED. **C4's RTSP measurement exists:**
      at 12 × 5 one interpreter keeps up with NVDEC decode and the pipeline; the 50 × 20 wall is load
      generation, the fleet's job. `fix/rtsp-headless-decode` is 8 commits, body at the scratchpad
      `pr-c4.md`, queued behind #20 → docs → mot.
- [x] **C1a-partial · The kernel tier has now actually run** — in the container, on GPU 5,
      with `_C` built for the container's own ABI (the image is 3.11, the host 3.10, so it
      needs its own build tree). torch is 3.4-4.5x numpy on the three ops; native
      `crop_batch` is 3.87x. **Native `nms` is 0.22x — 15x slower than torch.** The tool
      printed its own load warning (`load 23.6/48 — BUSY`) without being asked, which is what
      `load_note()` is for. The algo tier and the Nsight timeline still want a quiet box.
- [x] **C5 · Benchmark tiers algo and kernel** (R44) — both exist and are tested offline.
      `benchmarks/kernels.py` times each `ImageOps` implementation per op, bound to a device
      the way `PipelineRunner._build_ops` binds it; `benchmarks/stages.py` reads the
      `stage_latency_us` histograms the pipeline already fills and reports per-frame cost and
      share. 20 new offline tests pin the arithmetic (105 in `benchmarks/tests` total).
      Two findings on the way: the first kernel run timed **torch on the CPU** because
      `TorchImageOps` defaults there without a `device_index` — a true number about a
      configuration nobody runs; and `letterbox` returns numpy by contract, so timing only
      that column charges device implementations for a copy home that numpy never makes.
      Both are fixed and written down in `benchmarks/README.md`.
      **Not yet run to completion on a quiet box** — the algo tier's first attempt hit CUDA
      OOM because parallel agents held 22 GiB on four GPUs. Re-run for C1a.
- [x] **C6 · PR #3 findings 2, 6, 7, 8** — all four fixed, each with a test verified red
      against the unfixed code; 802 offline tests green. Kafka now registers `on_delivery` and
      charges the broker's verdict back to an `emit()` (`SinkDeliveryError`); `IngestManager.stop`
      uses `request_stop()` for the signal pass instead of `stop(timeout_s=0.0)`;
      `FrameCollector` writes the `pending_frames` zero for a camera that has gone idle and
      tracks the written series in a bounded set exposed through `sizes()`; the reconnect delay
      is `self._stop.wait(delay)`, and the offline tier now runs that default rather than only
      an injected `sleep=lambda _: None`.
- [x] **C7 · `wheels.sh` stages the TensorRT wheel** — copied from the host install, matched to the container ABI; absence warns rather than fails, since the offline tier needs none.
- [x] **C8 · `conftest.py` only asks the driver when a device tier was selected**, and treats a driver that raises as a machine with none. 791 offline tests green.
- [x] **C16 · `bindings/` holds pybind and nothing else** (V79). Three steps, three homes: step 1
      (pybind declarations) in `csrc/bindings/`, step 2 (numpy → plain C++) and step 3 (the
      algorithm) under `csrc/shipvision/`. `bindings/` went 1440 → 364 lines; the conversion and
      the nine session classes moved to `shipvision/{mot,mtmc}/session.h` beside the algorithms
      whose contracts they enforce, with the shared dtype vocabulary in `shipvision/interop/
      numpy.h`. Six `mtmc_*` matrix passes were declared in `bindings/mot.cpp` — a cross-camera
      pass findable only by reading the single-camera table — and are now in `bindings/mtmc.cpp`.
      The five per-session mutexes collapsed into one `TrackerSession` base. Builds clean,
      `_C` exposes the same 22 names.
- [x] **C17 · No `gil_scoped_release` anywhere in shipvision** (V79) — 29 removed. It is an
      algorithm library; thread discipline belongs to the caller. Two things worth stating
      rather than burying:
      * **A real cost.** The GIL is now held for a whole native call, so a threaded caller gets
        no overlap. At ~2.3 us per tracker frame that is nothing; for the hundreds-of-microsecond
        image ops it is real, and it is affordable because the deployment is one process per
        shard (B3b) where there is no competing Python thread for a release to help.
      * **A test went vacuous, and I verified it rather than assuming.**
        `tests/mot/test_native_thread_safety.py` used to kill the interpreter 10/10 with the
        mutex removed. Rebuilt with all twenty `hold()` calls deleted, it now passes 6/6 —
        because CPython itself serialises the two threads. The mutex stays (it is the object's
        invariant, and it guards a free-threaded build), the finding is written into the file so
        a green run is not mistaken for evidence, and what remains checkable offline — the lock
        is present, every session derives from the base, twenty acquisitions — is asserted
        against the source and mutation-tested in both directions.
- [x] **C18 · Two ownerless bugs from the V57 merge, found by running the suite** — 203 red tests
      in the submodule, both invisible to a type checker because a name used only inside a
      method body is not resolved until it runs. (1) Five `tracker.py` files called
      `_C.XTracker(...)` on a name only `backends/native.py` imports; `require_extension` now
      **returns** the module it vouched for, so the working spelling is the short one and the
      handle cannot be held without the check. (2) `_as_arrays` and `_columns_above` went with
      DeepSORTv2 when the native classes moved, leaving `NativeTracker.update` and BoT-SORT
      calling them from nowhere; both are back in the shared module. Plus `native.py`'s
      `__all__` still named five classes that had left it.
- [x] **C19 · The submodule layout tests, honestly updated rather than loosened.**
      `test_layout.py` asserted *one* class per `tracker.py`, which is what V57 changed — it now
      asserts *exactly the registered implementations of this name*, a stronger claim, and both
      failure directions are mutation-verified. `test_registry.py`'s `PUBLISHED` stays
      hand-written on purpose (those strings are a config-file contract) and gained the two new
      trackers. `NATIVE_ONLY` gained `interop` with the reason: a Python counterpart would be a
      module converting numpy to numpy.
      **And it caught a mistake of mine.** In C15 I replaced `test_spaces.py`'s hardcoded list
      with a registry read, which left `sorted(TRACKERS.names()) == sorted(TRACKER_NAMES)`
      comparing the registry against itself. Deleted; the assertion that carries the property
      is `all_spaces()` against the registry.
- [x] **C20 · The parent's 42 tracking tests were skipping again** — they import
      `shipvision.tracking`, renamed to `shipvision.mot` in V69, so D2's fix silently came
      undone. Six files repointed. **1061 offline tests, 0 skipped** (was 1019 + 42 skipped).
- [x] **C21 · shipvision PR #1 — MERGED.** Three blocking findings, each reproduced by
      *running* it before fixing: `Embedding` documented an L2-normalisation invariant it did
      not enforce (norm 5.0 where the contract says 1.0, on all three carriers — `Track`
      validated its embedding nowhere at all); the lazy `__getattr__` was exercised by nothing
      because `_REGISTRY_HOMES` ships empty, so four tests collected as empty parameter sets;
      and an alias silently outranked a real algorithm name, so `build("sort")` returned
      BoT-SORT with `names()` listing both. Each fix mutation-verified. One test-pollution bug
      of my own caught by the suite and fixed at both ends. APPROVE, auto-merged.
- [x] **C40 · shipinfer PR #11, review round 2 — three BLOCKING, all true.** (1) `models()`
      was a fifth live reader of the model table (and `__repr__` a sixth through it), and my
      structural guard grepped for `_models.values()` so it was blind to `sorted(self._models)`
      — exactly the reader it missed. Fixed; the predicate is every `self._models` minus an
      explicit allow-list, comment lines skipped, readers enumerated by name; red-first output
      names `sorted(self._models)`. (2) Evidence: the whole `-m gpu` tier (14 passed, 1 skipped)
      and a bench with per-device breakdown. **The CLAUDE.md bench command does not run on this
      harness**: `--seconds 5` is shorter than the default 10 s warm-up, and `run_bench.py` has
      no `--skew`; the 50 x 20 run was refused by the harness because the in-process generator
      offered 502 of 1000 img/s. At 12 x 5 = 60 img/s: 59.7 achieved, every module SUSTAINED,
      four devices within ~10% (detector 277/299/302/310). Stated limit: this does not bound the
      instrumentation's cost at loads the host cannot generate in-process. (3) The review read
      the body as pushed; the seam-by-seam rewrite had landed via `gh pr edit` just after — but
      the Size line and the "Registries: unchanged" checklist line were wrong regardless and are
      fixed. A `VLLM::EngineCore` process (user tts26, 21 GB on GPU 0) was found during hygiene
      and left alone: not mine.
- [x] **C41 · Fix the documented bench command** (26 Aug, in the docs snapshot): `.claude/CLAUDE.md`'s
      evidence command is now the harness — `deploy/rootless/bench.sh --systems shipinfer --seconds 40`,
      the algo and kernel tiers beside it — with the harness's real flags, `shipinfer bench … --skew 8`
      named as the in-process scheduler demonstration it is, and the 50 × 20 load pointed at
      `shipinfer fleet`. `README.md`'s line is a valid CLI example and stays.
- [x] **C42 · shipvision PR #4 MERGED. PR #5 (`detection`, 27 files) open**, rebuilt on the
      merged main with CI's exact pre-commit command. One fix while rebuilding: an *empty* ONNX
      passed `is_file()` and hit `_require_tensorrt`, so it read as "install tensorrt" on a
      laptop; the size check now precedes the runtime and the test moved to the class whose
      fixture hides tensorrt, red-first proven (`BackendUnavailableError` without the check).
- [x] **C37 · shipinfer PR #11, review round 1 — three BLOCKING, all true.** (1) my isort
      slip made CI lint red, so "948 passed" had **no CI-backed tier** — the number was local
      only. (2) the `except ValueError` in `model.py` was dead for every spec rejection because
      *I* had changed the spec to raise `ConfigurationError` one round earlier — a bad
      `graph_spec` named no model; fixed, test asserts the name, red against the dead wrapper.
      (3) the body omitted the largest change (ensemble scheduler +645/−193), statistics, the
      limiter/tracing wiring, the stats routes, and said "Removed: none" while
      `_refuse_late_producers` refuses configs that used to run — the **fourth** body defect;
      rewritten from the diff seam by seam. The scheduler's throughput sentence is labelled as
      design reasoning: the repository ships no ensemble to bench, so the measured fact is
      `peak_parallel_steps >= 2` on the mock DAG, cited by node id. Reviewer suggests three PRs;
      answered plainly why not (hunk-coupled files) and offered to split if asked. `b3aa2cc`.
      **The gate worked once:** pre-commit rewrote a file after the amend and `git status`
      blocked the push; the previous rule's first real catch.
- [x] **C38 · A claim I made from inference, corrected by measurement.** The gpu warm-up test's
      docstring said the detector plan "reports dynamic shapes" because a batch-4 sample ran.
      Probed with TensorRT in the container: `(8, 3, 640, 640)`, static, one profile. Batch 4
      ran because the backend copies `batch_size` rows into a binding sized for the plan. The
      config's "static batch 8" was right; the test now states the measured fact.
- [x] **C39 · shipvision PR #4 — APPROVE, then lint red for a reason of mine:** a whole-tree
      `ruff format` had dragged two unrelated files in; CI's pinned black disagreed on one
      assertion each. Restored to main, `14af02d`, CI's exact command passes. Rule added.
- [x] **C35 · shipinfer PR #11 (`server`, 24 files) open.** Carries #8's finding 5 (verified,
      fixed, snapshot helper), the three deferred debts (EXPLICIT relaxed now that the endpoints
      exist; `spec.py` landed with its caller, `ConfigurationError`, corrected docstring,
      `source` typed as the prose it is; `warmup_captures_graphs` told the truth), #10's
      follow-ups (`add_note` delegation; fast-fail pinned by a timed test), and a real GPU test:
      two declared samples execute exactly twice on the real TensorRT detector; a wrong-sized
      data file is refused naming the sample. Its first negative case assumed a "static batch 8"
      plan would refuse batch 4 — the plan is dynamic and ran it. **Follow-up:** the config's
      "static batch 8" comment is stale; check `scripts/build_engines.py` and fix the prose.
- [x] **C36 · shipvision PR #4, review round 1 — two BLOCKING, both true, and the first was
      the third body this day that promised tests not in the diff.** The alias-guard class and
      the `sys.modules.pop` precondition lived on the V79 branch; this branch took the test file
      from `feat/library`, which lacked both and *reverted* main's precondition. Fixed in
      `b10edf0` with the mutation re-run on this branch (3 of 4 red), the empty test given
      assertions, and an admission in the PR comment. Plus the black-hook trap (a rewriting hook
      passes on its second run) — caught by `git status` before the second push. Three rules
      added to CLAUDE.md.
- [x] **C33 · Two process lessons from one afternoon, made rules.** (a) `exc.add_note` is
      Python 3.11+; the container is 3.11, the project promises 3.10, and CI on 3.10 was the
      only place it could fail — local green is not evidence for a version the container does
      not run. Replaced by a 3.10-safe `_annotate` writing `__notes__` directly, verified on
      the host's 3.10. (b) **Twice in one hour I pushed a branch whose own check had just
      failed**, because the push was on a separate line from the check. Rule: *every push is
      `&&`-chained to the verification that gates it* — suite, format, mutation — so a red
      check cannot be followed by a push. Also: shipvision's PR pipeline runs `pre-commit
      run --show-diff-on-failure` (pinned ruff-format), not black; run `pre-commit run
      --all-files` there before pushing.
- [x] **C34 · The two csrc fixes are ported to the V79 branch** (`refactor/per-algorithm-
      packages`, 88d9a15) — `BoundDevice` before the ring in `session.h`, and the NMS mask
      through pinned scratch. The structural tests came with them, pointed at `session.h`:
      red before the port, green after; the extension compiles. `benchmarks/tests` green (121)
      after the realistic NMS fixture.
- [x] **C30 · shipinfer PR #10 (`runtime-seam`), review round 1 — four BLOCKING, all true,
      and two of them were about my PR body describing a diff that did not exist.** (1)
      `runtime/graphs/spec.py` had no caller; (2) it raised bare `ValueError`; (3) no tests for
      it, and the body claimed tests that were not in the diff; (4) the body did not describe
      the only behaviour change present — `backends/base.py` running declared `model_warmup`
      samples at start-up. **Written from the plan, not the diff. Same mistake as the split
      body on #2.** Fixed by making every claim true: `spec.py` *withdrawn* from this piece
      (lands with the server piece that wires it, carrying `ConfigurationError`, real tests and
      the corrected docstring — the harm is the missing size, not the oversized one); the
      warm-up wiring described with both operator-facing consequences and the setting's own
      docs updated; typed errors pass through warm-up unchanged (non-blocking, taken); seven
      backend-level tests, the sharpest verified red. Body rewritten from `git diff --stat`.
      Pushed `7dd331e`.
      **The server piece now owes three things at rebuild time:** the `EXPLICIT` validator
      relaxation (#9), `spec.py` wired into `_build_instances` (#10), and the dead
      `execution.warmup_captures_graphs` setting (#10 non-blocking).
- [x] **C31 · shipvision PR #3 (`imgproc`) MERGED** on round 2 (APPROVE). **PR #4
      (`registry-fallback`, 2 files) open** — it turned out to be a feature, not only the four
      alias-guard tests: an unpinned `build()` falls back past `BackendUnavailableError` down
      the preference order; pinned never falls back; only an unavailable runtime is "try the
      next". Read the diff before writing the message this time. `detection` is next after it.
- [x] **C32 · Two csrc fixes committed on `feat/csrc-native`, both mutation-verified with a real
      rebuild, both to be ported to `session.h` when the bindings-restructure branch is rebuilt:**
      `bc70dde` — `StagingRing ring_` was a member, so its events were created *before* the
      ctor body set the device (device-0 events recorded on device-5 streams = `invalid
      resource handle`); now a `BoundDevice` member declared before the ring. The mutated build
      reproduced the production error verbatim. `7177e0d` — the NMS mask was downloaded into a
      **fresh pageable** `std::vector` every call: measured 30.8 ms for that copy vs 1.7 ms
      pinned. Through the pinned scratch, native NMS is 33.3 → 6.1 ms at 25k candidates and
      **2x faster than torchvision at every n** (0.14/0.61, 0.96/1.88, 6.06/12.06 ms). On the
      bench's own fixture it is still 4.8 vs 2.1 ms: the remainder is host sort+gather of 18k
      boxes before upload, a contract difference (native takes host boxes), not a defect.
- [x] **C27 · closed by C32.** The "16x slower" was one allocation.
- [x] **C29 · shipinfer PR #9 MERGED** (APPROVE on round 2). **PR #10 (`runtime-seam`, 6 files)
      open.** `profiling.py` deliberately not taken from the #8 branch — its copy there would
      have reverted #9's re-export.
- [~] **C12 · The submodule PR sequence** (V70 + V78 + **V80**). PR #2 was 45 commits and 290
      files; the operator had to ask twice. **Writing "this PR is too big" in the description
      is not splitting it** — that was the mistake, and the rule is now a hard limit in
      CLAUDE.md with the numbers and the measuring commands.
      #2 is a draft, and eight package-scoped branches are built and committed. The split
      needed no cherry-picking at all: packages are file-disjoint, so each branch is
      `git checkout feat/library -- <paths>` onto main. That is the thing I had called
      expensive.

      | branch | files | depends on |
      |---|---|---|
      | `feat/registry-alias-guard` | 4 | — |
      | `feat/imgproc` | 36 | — |
      | `feat/reid` | 31 | — |
      | `feat/mot` | 71 | — |
      | `feat/detection` | 29 | imgproc |
      | `feat/mtmc` | 47 | reid |
      | `feat/eval-tune` | 39 | mot |
      | `feat/csrc-native` | 45 | all |

      The import graph is a DAG with no cycles (checked against real `import` statements, not
      grep over prose — the first pass matched docstrings and showed a false cycle between
      `tune` and `eval`). **PR #3 = imgproc is open.** Each later branch is rebased onto the
      merged main when its turn comes.
      `feat/mot` stays 71 files deliberately: the five algorithms share one lifecycle, one
      pool and one `backends/` module that `tracking/__init__.py` imports for registration, so
      halving it means editing imports and tests to hide real dependencies — trading a
      review-size problem for a correctness risk on the one plane whose argument is threads. — the uncommitted shipvision work is
      four features, so it is four PRs, **pushed one at a time and carried to merged in
      order** because a BLOCKING review on the first has to be fixed before the second is
      stacked on it: (1) the `mot`/`mtmc` rename, the `trackers`/`matchers` layout and each
      native class merged into its algorithm's `tracker.py`; (2) the imgproc library lifted
      out of `bindings/module.cpp` (891 → 130 lines); (3) the new `strongsort`/`boosttrack`
      trackers with their Optuna spaces; (4) the native MTMC tracker (C13).
- [ ] **C13 · A native C++ MTMC tracker** (V64) — `mtmc/trackers/cluster/tracker.py` holds a
      Python `threading.Lock` around `track()`, and the operator's point is that if a lock is
      needed at all it should be a C++ one. `mtmcservice`'s `VTXTracker`/`AICTracker` are the
      reference.
- [ ] **C14 · McByte** — the one tracker roboflow has that we lack (arXiv 2506.01373,
      mask-conditioned association). Their source is `references/roboflow-trackers/src/
      trackers/core/mcbyte/`, whose layout is the one we adopted independently.
- [x] **C15 · Optuna search spaces for the two new trackers** — `test_spaces.py` failed because
      `strongsort`/`boosttrack` had none. Added, and the test's hardcoded `TRACKER_NAMES` list
      now reads the registry instead: a two-place edit to add one tracker is a list where the
      second edit gets forgotten. 77 tune tests green. StrongSORT excludes `nsa` from its space
      for a reason worth keeping — it scales the appearance EMA by detection confidence, and
      MOT17 public detections carry a *constant* score, so on that benchmark the flag has no
      effect and a study sampling it would report its own sampler's spread as a finding.
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

- [x] **C1a-kernel · The kernel tier has run to completion, all three ops, all three
      implementations** (GPU 5, load ~18/48 — better than before, still not quiet). The
      inherited "50x on preprocessing" is **measured false**: on the production path
      (`letterbox_to_device`) native is 657 us vs torch 735 us = **1.12x**; `crop_batch`
      1303 vs 1387 = 1.07x; `letterbox` with the copy home is slower than torch, as the README
      predicted. **Native `nms` is 33.3 ms vs torch 2.1 ms = 0.06x — 16x slower — twice in a
      row.** That is a defect, not noise, and it is C27. Two consequences for C1: the fused
      kernels are not where the 5x is, and a per-frame budget built on the 50x figure is wrong.
- [~] **C27 · Native `nms` is 16x slower than torch.** Reproducible (34.8 ms, then 33.3 ms).
      **Root-caused and fixed on `refactor/per-algorithm-packages` (lands with its PR):** the
      `(n, ceil(n/64))` mask came back through a *pageable* `std::vector` — 30.8 ms for a fresh
      pageable 44 MB D2H against 1.7 ms pinned. `NmsScratch.host_mask` now points into the
      session's pinned download buffer; native nms 33 -> 6 ms in the same fixture. The remaining
      gap to torchvision is re-measured by the kernel tier once the pointer moves. Original notes:
      Candidates, in order of likelihood: a `gpuMalloc`/`gpuFree` per call on the mask scratch,
      a device-wide sync, or the host sweep over the `(n, ceil(n/64))` mask walking the whole
      matrix. Nsight on that one op is the next step; the fix lives in the submodule.
- [x] **C23 · Two more faults behind the same door as C22 — both root-caused and fixed.**
      *`invalid resource handle`*: `StagingRing ring_` was a **member** of `ImageOps`, so its
      three `StagingSlot`s created their CUDA events during member initialisation — before the
      ctor body ran `gpuSetDevice`. Events belonged to device 0; recording one on device 5's
      stream is exactly that error. `crop_batch` and `nms` never record the slot event, which is
      why they worked and made it look like a letterbox bug. Fixed with a `BoundDevice` member
      declared *before* `ring_`, so the language guarantees the order. Confirmed by the failure
      moving: after the fix the same call fails at the Python unpack instead of the CUDA call.
      *The unpack*: the submodule returns `extents` (3 and 4 values) since its #1 review; the
      parent unpacked 2 and 3. Now unpacked, and `LetterboxResult` gained an optional `extents`
      so the batch path carries them; `letterbox_to_device` still returns `(scales, pads)`
      because nothing downstream re-derives `out_h` yet — widening that contract is C28.
      Two defects, both visible only once the native ops path finally loaded (C22):
      * `NativeImageOps.letterbox_batch` raises `GpuError: gpuEventRecord failed: invalid
        resource handle`. Reproducible standalone (`--implementation native` alone fails; torch
        alone succeeds), and once it fires the CUDA context is poisoned so torch's *next*
        letterbox fails too — which is how one broken implementation made the other look
        broken.
      * `letterbox_to_device` unpacks two values from `letterbox_into`, which has returned
        three since the submodule added `extents` — the number that decides the sampling ratio,
        added precisely because Python re-deriving `out_h` from the scale can disagree by a
        pixel while scale and pad both still match. The parent should *use* it, not drop it.
      Both went unnoticed because the path was unreachable (C22): a dead path is where a
      breaking change is invisible.
- [ ] **C28 · Carry `extents` through the ABC** for all three implementations and have
      `detect.py` read it rather than ever re-deriving `out_h` from `scale`. Small, its own PR.
- [ ] **C48 · `bench.sh` refuses to start without the baseline binary even for `--systems
      shipinfer`**, which measures this project alone — so the documented evidence command exits 1
      on a clean checkout. The gate now reads `--systems` and fires only when the list names
      `baseline`; fixed on `fix/rtsp-headless-decode` (the harness PR), lands with C4.

## Phase 6 · The final goal (V49)

- [ ] **C1 · ≥5× counting-simulation, whole system.** Measured: baseline 868.2 img/s against the
      C++ plane's 390.5 → **0.45×**. The interpreter is no longer the wall *inside a process*;
      the remaining gap is that 390 is one process on a forty-eight-core box. Two halves:
      C1a (profile first, V63) then C1b.
- [x] **C1b · The multi-process launcher** — `server/launcher.py` plus the settings side.
      `Fleet` spawns one `Popen` per shard with `CUDA_VISIBLE_DEVICES` and
      `SHIPINFER_SHARD_CAMERAS` in the child's environment *before the interpreter starts* —
      subprocesses rather than `multiprocessing` for exactly that reason, since a spawn-context
      child inherits the parent's environ and setting it inside the child races whatever
      imported torch first. A dead shard takes the fleet down rather than leaving a quarter of
      the cameras dark behind green dashboards; a shard that will not drain is killed on a
      *shared* deadline, so stopping sixteen takes one drain and not sixteen.
      38 offline tests, all against processes that are not servers — everything worth getting
      wrong here is supervision, and testing it through a real server would test CUDA.
      **One bug the tests did not catch, and why:** the first `serve_command` emitted
      `--cameras` and a positional repository. `serve` has neither. The argv read plausibly and
      the CLI would have rejected it — and my test asserted the shape *I had invented*. It now
      reads `serve`'s real signature, and goes red against that exact mistake.
      Measurement still gated on C1a and a quiet box.
- [x] **C1a-algo · The algo tier has run to completion** (12 cameras x 5 fps, GPUs 2-5, load
      22/48 — the tool printed the load itself). Delivered 60.1 of 60 img/s. Per-stage p50,
      submit-to-result, with the new 1.6x buckets: detect 40-63 ms, crop 40-63 ms, segmenter
      40-63 ms, both embedders 16-25 ms. **These are queue-and-batch-window spans, not kernel
      time**: a yolo26n detect is single-digit ms on an A5000, so ~50 ms of p50 at 5 fps is the
      batch window and the model queue, which is what "submit-to-result" measures by design.
      Serial per frame 183 ms against wall per frame 12.55 ms — the instance pools and the
      worker pool are buying ~15x, so at this load the lever is a cheaper stage or a shorter
      window, not more workers. Two lessons on the way: the first complete run reported every
      p50 as 100 000 / 50 000 / 25 000 us and I read it as histogram saturation — it was the
      registry default's 2-2.5x bucket steps with upper-edge quantiles, and "100 000" meant
      "50-100 ms", which is real; the stage histogram now has explicit ~1.6x buckets and the
      output says what a quantile is. And `settings.tracking.enabled` crashed the harness at
      the end of a real GPU run because the unit tests never went through the call site; now
      `tracking_enabled()` is reached through a real `ServerSettings`.
- [~] **C1a · Profile before optimising** (V63). Two of the three pieces are in place and
      neither has been *run*, because the GPUs are held by parallel agents:
      - per-stage host timings — `benchmarks/stages.py` (C5), reads the histograms the
        pipeline already fills;
      - per-op costs — `benchmarks/kernels.py` (C5);
      - the device timeline — `deploy/rootless/profile.sh`, Nsight Systems mounted from the
        host the way TensorRT is. `perf_event_paranoid` is 4 here so CPU **sampling** needs
        CAP_PERFMON and is off; CUDA and NVTX tracing need no perf events and are the useful
        half — they answer "is the GPU idle, and between what", which is the question standing
        behind the 390 img/s ceiling.
      Remaining: run all three on a quiet box and write the answer down. Everything in C1
      waits on that.

## Phase 7 · Topology — B, C and DeepStream behind one abstraction (V83–V85)

The operator's final target is **C**: stateful streaming (decode, detect-local, track) pinned per
GPU, stateless inference (crops) balanced across every GPU by a queue whichever instance is free
pulls from. The three plane seams already exist; what is missing is the *topology* — how planes
are placed into processes and how work crosses between them. Order is evidence-driven: B lands and
is measured under skew before C's transport is written, and the DeepStream tier is a parallel,
file-disjoint lane. Cross-process work keeps `scheduling/` untouched: a remote instance is a proxy
that exposes the four attributes a policy reads.

- [x] **T1 · The `Topology` abstraction** (built on the fleet branch, lands with its PR). `server/topology/` — `Topology` ABC (`plan(settings,
      devices, cameras) -> ProcessPlan`, `launch()`, `describe()`), `TOPOLOGIES = Registry("topology")`,
      a `core/settings/topology.py` section (`kind: fleet | service | deepstream` plus per-kind
      options; env-overridable like every other section — there is no `envs.py`, the settings tree
      *is* the switch), `shipinfer launch --topology <kind>` with `fleet` kept as the alias. Pure
      control plane, no torch. Tests in `tests/server/test_topology.py`. Depends on the fleet branch
      landing (its PR follows the `split/*` PRs).
- [~] **T2 · B = `fleet`** — registered and wired (`shipinfer fleet --topology`); the skew bench is still owed. `Fleet` + `plan_shards` registered as `@TOPOLOGIES.register("fleet")`.
      Behaviour unchanged, tests move; the operator asked "chỉ chỉnh 1 chút" and the answer is yes —
      this is registration and interface conformance only. Then the skew bench on B (needs C41's
      `--skew`): per-device queue depth and p99 end-to-end are the numbers that size T3.
- [ ] **T3 · C = `service`.** *(Plan written 26 Aug, main session — the planner agent was cut off by
      the spend limit — at the scratchpad `plan-t3.md`, to land as `docs/design/topology-service.md` with
      T3's first PR: pairwise single-writer pinned rings (vLLM `ShmRingBuffer` discipline), `RemoteInstance`
      as a `Placeable`, `Topology.attach()` as the one new seam method, four PRs, three questions for the
      operator — slot size per model, never share the detector, the pinned budget.)* **Step 1 built** on
      `/tmp/t3a` (`feat/shared-ring`): `runtime/memory/shared_ring.py` — `RingLayout`, the four-state
      single-writer protocol, the header as the load signal, `RingFullError` / `PeerLostError` /
      `RingProtocolError` — 19 offline tests with a thread as the peer; queued behind the docs snapshot,
      `chore/shipvision-mot`, `fix/rtsp-headless-decode` and `fix/letterbox-extents`.
      **Step 2 built** on `/tmp/t3b` (`feat/remote-instance`, stacked on the ring): the wire format
      (`server/remote_wire.py`: request and response heads, host tensors only, a failure form, 12 tests) and
      `server/remote_instance.py` — `RemoteInstance` as a `Placeable` (`device` is `cpu`: a proxy is *not
      here*), `ResultReader` (one per process, heartbeat watch → `PeerLostError` with the tags), `RingIngress`
      (owner side: through the model's own `infer`, slot held until the future settles); 7 tests run two
      "processes" as objects over real rings. Found and fixed on the way: a ring closed under a live
      zero-copy view raised `BufferError` and skipped its unlink, a closed handle raised from a released
      view instead of answering "closed", and the handle's finaliser failed noisily later — the ring now
      drops its own export on close and parks an unclosable handle for `reap_pending_closes` (19 ring tests).
      **Step 3 built** (26 Aug, same branch): `ServiceSettings` (`shared_models`, `slots_per_pair=8`,
      `slot_bytes=1.5 MiB`, timeouts, `shard`/`peers`/`run_id` set by the launcher through
      `SHIPINFER_TOPOLOGY__SERVICE__{SHARD,PEERS,RUN_ID}`), `Topology.shard_environment(shard)` as the one
      new seam method (the plan's `attach()` turned out to be this: what a *child* is told, not what the
      parent attaches) carried by `Fleet.shard_env`, `ServiceTopology` registered as `service`,
      `server/service_mesh.py` (create the rings this shard reads, open the peers' with retry, one ingress
      per (peer, model), one reader, `Model.attach_remote` rebuilding the dispatcher with the proxies),
      `InferenceServer` joining the tier after the models load and leaving it *first* on stop; 7 mesh tests
      (two shards' meshes in one process: a request leaves shard 0's dispatcher and returns from shard 1 as
      `cuda:1`; the deep shard borrows the quiet one; stop takes the rings down; a peer that never appears)
      + 9 contract tests; ADR-015 and the feature-log entry on the branch. **Then the first run through the
      real `Model` found what the fakes could not:** a stray `@property` had made `attach_remote` unbound and
      every shard died at start — fixed, and `test_service_engine.py` (two `InferenceServer`s in one process
      as shards 0 and 1) now covers the engine's own path. Also on the way: every fleet shard ran `serve`
      with no HTTP, so none was addressable — `serve_command(http_port_base=)` gives each shard `base +
      index`; `Model.infer_local` so work that crossed once is never re-routed (two deep shards on stale
      headers could bounce it); `slot_bytes` = 1.5 MiB + 64 KiB because the heads travel in the slot; a
      ring `open` no longer registers with the resource tracker (bpo-38119: an attacher would unlink the
      owner's ring at exit). **Evidence (26 Aug, container, GPUs 3,4):** `tests/server/test_service_multigpu.py`
      — two real `serve` processes through the real `ServiceTopology` + `Fleet`, 24 requests posted to
      shard 0 over HTTP → *19 executed there, 5 executed by shard 1 through the ring*, every tag back on
      its own response, both processes gone after `stop` (GPUs back to 15 MiB). **Not yet:** the bench-scale
      run — `--topology` on the harness with the fleet driving the shards (PR-cut item 4). Queue: docs
      snapshot #21 (review round 1 answered) → `chore/shipvision-mot` → `fix/rtsp-headless-decode` →
      `fix/letterbox-extents` → `feat/shared-ring` → `feat/remote-instance` → `ci/cpp-offline-and-prompt`.
      Original sketch: `fleet` plus a cross-process inference tier, symmetric — every shard
      process serves its own GPU's crop-stage instances to its peers, so a dead process loses its K
      cameras and its capacity, nothing else. Pieces: (a) `runtime/memory/shared_ring.py` — a
      `multiprocessing.shared_memory` ring pinned in each process via `torch.cuda.cudart()
      .cudaHostRegister`; producer D2H into a slot, consumer H2D out of it (~1.5 MB/frame,
      ~125 us a copy on PCIe 4). Each process keeps `CUDA_VISIBLE_DEVICES` = its GPU: opening CUDA
      IPC handles on other GPUs would create G contexts per process, G^2 x ~300 MiB on the box.
      (b) `server/remote_instance.py` — a proxy with the policy-visible attributes and a `submit()`
      that writes into the owner's ring; the dispatcher's candidate set = local instances + proxies,
      default policy `locality_spillover`. (c) Reassembly waits on remote results; camera id rides
      with the request so the fair queue stays per-camera across processes. (d) A closed ring drops
      its proxy from the candidate set. Gate: same skew bench as T2, B against C, on a quiet box.
- [ ] **T4 · DeepStream = the competitor tier.** `benchmarks/harness/deepstream.py`,
      `deploy/deepstream/Dockerfile` (DeepStream base image, `pyds` — a bench dependency, never in
      the wheel), Triton `config.pbtxt` per model with `instance_group` across the GPUs — the one
      setting that makes the sketch C rather than B — and the operator's `mtmc_deepstream.py`
      moved under `benchmarks/deepstream/` and completed: `output_tensor_meta: true` with the probe
      reading `NvDsInferTensorMeta` from `obj_user_meta_list`, a bus watch, the same RTSP source as
      the system tier, the same result table. Parallel lane; file-disjoint from T1–T3.

- [ ] **C44 · The algo tier's first exact profile says `crop` is the largest per-frame cost.**
      12 cameras x 5 fps on GPUs 2–5, kept up (60.0 of 60), steady window, host load 22/48 with
      another user's 21 GiB job on GPU 0: per-frame cost crop 149.6 ms (46%), detect 98.6 ms
      (30%), ship_segmenter 41.5 ms, the two embedders ~17 ms each; serial 324 ms against wall
      16.9 ms. Means, not bucket edges — the earlier "16–63 ms p50" was the bucket resolution.
      These are submit-to-result spans (queue + batch window + work), so the next step is the
      Nsight timeline (C1a) to split wait from work — but a *crop* stage costing 1.5x detect is
      the first thing to look at: a per-object host loop or a D2H on the crop path would do it.

- [x] **C45 · The csrc piece's four owed defects** (from #8's second review, promised in the
      plane's own commit body) are fixed on `split/csrc`: a mutex on the two `FrameState`
      containers the sweeper copies; `execute` in two passes with per-frame try/catch returning a
      failure count so a batch-mate is never sealed Complete for a stage that never ran;
      `crop_resize` on the `TorchImageOps` arithmetic (clip, truncate, patch, `align_corners=False`,
      clamp inside the patch) instead of half a pixel off; readable references and parity tests
      for `crop_resize` and `nv12_letterbox`. 47 C++ checks, 0 failures, in the container.

## Phase 8 · The real port — `csrc/` mirrors the Python data plane (V88, V89)

The operator looked at `csrc/` and saw a different program. They were right: the C++ plane
(2.8k lines) is a purpose-built throughput binary sharing the Python plane's layout and names,
not a port — a worker pool leasing instances instead of one thread per instance with a queue,
no placement policy, a fixed 50 ms drain instead of the batch window, newest-first eviction,
replay-only ingest, engines from CLI flags. It answered C1's question (the interpreter is the
wall: 77 vs ~450 img/s per process) and earns its place as the starting point. The decision
(V89) is **B**: port for real, seam by seam, with a cross-plane parity harness as the acceptance
test, in the order that removes the largest architectural difference first. Control plane stays
Python (ADR-014). From now on a Python data-plane change is not done until the C++ seam is synced
(CLAUDE.md, "Two planes, one architecture").

- [x] **P0 · Say what it is** (ADR-014 amended; PR #15's body). ADR-014 amended: the current binary is the starting point of the
      port, not the plane; the sync rule recorded. The csrc PR body describes it the same way.
- [x] **P1 · Instance = one thread + one bounded queue; dispatcher + placement policy.** Built on
      `port/p1-scheduling` (five commits, rebased on the csrc piece after its fifth review):
      P1a the queue seam mirroring `fair.py`/`lanes.py`/`fifo.py`; P1b the five policies with their
      registry and the `Dispatcher`; P1c `ModelInstance` (thread + queue), the `Engine` contract with
      the TensorRT adapter, request/response/`WorkItem`, `Model`; P1d the graph as stages over
      `Model::infer` (`Dag`, `DetectStage`/`CropStage`/`ObjectStage`, `WorkerScratch`) and the bench
      running that shape — the pool graph is gone. 62 + 24 + 9 scheduling/server/pipeline checks,
      46 data-plane checks. Measured: ~390 img/s at 48 workers, ~470 at 96, balanced across GPUs to
      1%, 0 failed, 0 timeouts. Its PR follows the csrc piece.
- [x] **P2 · `BatchWindow`.** Landed with P1a/P1d: `max_delay_us` and preferred sizes per model, the
      two-phase wait, applied in each instance's queue (`--batch-delay-us`). Parity trace: P6.
- [x] **P3 · Fair queue eviction order.** P1a: oldest of the greediest, Python's tie-break. The
      divergence ADR-014 recorded is closed; the parity harness (P6) lists it as a case to pin.
- [ ] **P4 · Ingest.** RTSP (GStreamer/NVDEC) and replay behind one source registry, camera
      actors with reconnect, the manager's stop semantics — the Python `ingest/` mirrored.
- [ ] **P5 · Resolved config in, same events out.** The binary takes the settings tree and the
      model repository (`config.yaml`) the Python plane reads, not CLI flags; emits the same
      event schema (`pipeline/schema.py`) so one sink serves both planes.
- [ ] **P6 · The parity harness.** `benchmarks/parity/`: drive both planes with one recorded
      trace and diff events, per-camera eviction counts and batch boundaries. This is the gate
      the sync rule refers to; CI runs its offline half.

- [x] **C22 progress (V80/V81).** shipvision: #2 split into #1, #3, #4, #5, #6, #7, #8, #9 — all
      merged, #2 closed; the native sessions in the restructured `csrc/` layout remain (from the
      V79 branch). shipinfer: #8 split into #9, #10, #11, #12, #13, #14, #15 (all merged; csrc took six review
      rounds), #16 infra-docs (open), then `fix/native-reachable`, `feat/fleet-topology`,
      `port/p1-scheduling` in that order, each built and green.

- [~] **C46 · The shipvision restructure (V79) is the last unsplit branch.** S1 opened as shipvision's
      next PR (`refactor/mot-per-algorithm`: the rename and the per-algorithm layout, 1694 passing). `refactor/per-algorithm-
      packages` is 216 files / +12.4k / −4.5k against main. Cut, in dependency order, each with the
      parent's sync in mind: **S1** the `tracking` → `mot` rename with the compatibility shim
      (`TestTheOldImportPathsStillResolve`), tests moved — a rename is many files by nature, and the
      body says so; ~~**S2** the Python additions and fixes~~ — *there are none.* Checked after S1 merged
      (25 Aug): no shipvision branch has a StrongSORT or BoostTrack file (`git ls-tree` over every
      remote ref), and the remaining Python diff between `refactor/per-algorithm-packages` (tip
      08:39) and main is main being *newer* — #3, #5 and #9 landed 14:27–19:13 with the
      `_as_unit_vector` invariant, the study.py `constants` fix, and a different isort width; the
      V79 side of each hunk is the older text. S2 is struck, not deferred; **S3** `csrc/` in
      the new layout (`csrc/shipvision/{interop,imgproc,mot,mtmc}`, `bindings/` declarations only,
      no GIL release anywhere, one mutex on the tracker session) with the native sessions, the
      native tests and `tests/test_architecture.py` — **built and green** on `/tmp/svs3`
      (`feat/csrc-per-algorithm`; the coder agent was cut off by the org spend limit and the rest was
      done in the main session, 26 Aug): rebuilt in `shipinfer-gst:jammy`, native+gpu tiers `342 passed`,
      whole tree in the container `2025 passed, 51 skipped`, offline in a clean container `1683 passed`;
      three commits — **shipvision #11 merged 26 Aug 01:47** (one review round, APPROVE). Next: the
      parent's `pipeline/graph/tracking.py` imports `shipvision.mot` with the submodule bump in its own
      commit — **prepared on `/tmp/mot` (`chore/shipvision-mot`, two commits, body at the scratchpad
      `pr-mot.md`)**, queued behind #18 → p1 → docs. `tracking`→`mot` in csrc paths/namespaces
      and `bindings/mot.cpp`, built in `shipinfer-gst:jammy`, main's Python sessions are the spec.
      After S1 the parent's `pipeline/graph/tracking.py`
      imports `shipvision.mot` (the two-planes rule applies to the library seam too).
- [~] **Queue order on shipinfer after #15 (csrc):** `split/infra-docs` (#16, **merged 25 Aug 21:11**
      after two review rounds: the Stop hook parsed a heading the ledger never had, and fired in CI;
      both fixed with tests) → `fix/native-reachable` (**#17 merged 26 Aug 01:19** after one review round — the
      build recipe's phantom `is_available`, a stdlib patch in a test, untyped failures — all real, all fixed; GPU evidence: parity `6 passed` where `5 passed,
      1 skipped` before; the three fixes are built on `/tmp/fx`, body at the scratchpad `pr-fx.md`)
      → `feat/fleet-topology` (**#18 open**; review round 1 answered 26 Aug: `supervise()` never
      returned after `stop()` (Ctrl-C spun until SIGKILL), `instances_for` had no caller so shards
      sharing a GPU loaded the full instance count, and the child inherited the parent's physical
      `SHIPINFER_DEVICES__VISIBLE_GPUS` — all three real, fixed with tests. **Round 2** (26 Aug): the plan
      balanced *shards* while two of four GPUs carried twice the load (the founding bug one level up —
      GPUs are now assigned first and the greedy pick is weighted by device share; `device_imbalance` is
      the figure), a second Ctrl-C during the drain deadlocked the handler on its own lock (the handler
      only records now), floor division lost the remainder of an instance count (`rank_for` →
      `SHARE_RANK`), my feature-log append had duplicated the whole file, and the GPU evidence exists:
      `tests/server/test_shared_device.py` (gpu tier) runs a real server with `shared_by=[2]` on a real
      device — `3 passed` in the container. **Merged 26 Aug 02:10** after round 3 approved.) → `port/p1-scheduling` (**#19 open** 26 Aug 02:15; review round 1 at 02:22 — two real blockers:
      the spilled row crossed GPUs with `gpuMemcpyPeerAsync` against ADR-002's host-memory rule, and a
      stage timeout let the next frame's crop overwrite a scratch slot a queued request still pointed
      at; plus `QueueFullError` without its numbers, an always-padded batch, and `requires` as an
      identifier. Fixed in `aa4d688` (26 Aug 02:5x): the spill stages through pinned host memory,
      payloads own their buffers (`WorkerScratch::acquire` → `shared_ptr`, the request keeps it),
      the error carries depth/capacity, only a static plan is padded; four binaries 62/24/16/46 checks in
      the container, bench 40 s 15420 accepted / 0 failed, per-device spread ~11% now that a spill costs
      two PCIe copies). **Round 2** (02:42): evidence predated the padding-decision commit — re-run on the
      final tree, all four plans static, 66/24/16/46 checks, bench 16534 accepted / 0 failed, ~2% spread;
      plus device-less pipeline tests, closed ≠ saturated, sized crops, the eviction callback outside the
      lock. **Round 3** (02:57): the lane's eviction tie-break scanned the rotation deque, not insertion
      order as Python's dict scan does — a real parity defect the old test could not see; fixed with
      `seen_` and the review's serve-then-evict trace as a test (red on the old lane). **Merged 26 Aug
      03:11** after round 4 approved — four rounds, every finding real.) → a docs snapshot (this
      ledger, `docs/qa/user.md` through V89, the CLAUDE.md rules) → P1c.

- [x] **C47 · Two follow-ups the csrc review named for later** — *both built on `/tmp/c47`
      (`chore/csrc-cuda-free-tests`, stacked on #19; body at the scratchpad `pr-c47.md`): the owning buffers
      leave `core/types.h` so `--offline` builds `test_scheduling` (62), `test_server` (24) and the new
      `test_containment` (15) with g++ alone, `ldd` showing no accelerator library; the bench binary calls
      a mirrored containment gate before opening a device and the hook knows the three device binaries.*
      **#20 open** (26 Aug 03:17, rebased onto main after #19: offline C++ tier 66/24/15 on the host with
      no accelerator library linked, CUDA binaries 16/46 in the container, Python 1207 passing).
      **Merged 26 Aug 03:35** after one review round — three real findings: `--offline` still asked for
      OpenCV, a CUDA-free binary linked only its own closure so its policy registry shrank, and the two
      device test binaries had a rule only the hook enforced; all fixed with evidence. A CUDA-free build target for the
      queue and collector tests — pure CPU logic that today lives in a binary that needs nvcc and
      TensorRT to link, so the fairness invariants cannot run on a machine with no driver; and the
      host-run gap: `csrc/build/bench` run directly on the host passes both enforcement points
      (`require_container.py`'s lists and `runtime/containment.py`), so the binary should consult
      the containment gate itself and the hook should know its name.

---

## Z · Final gate

- [ ] **Z1 · Re-read `docs/qa/user.md` end to end** and check every request — verbatim sections
      included, not just the standing-rules index — against the repository. Result into
      `docs/qa/verification.md` with per-line evidence, stating plainly what is still not done.
