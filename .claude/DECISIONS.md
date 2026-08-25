# Architecture Decision Records

Newest last. Read the relevant ADR before changing the pattern it describes; if you change
it anyway, supersede the ADR in the same PR.

---

## ADR-001 — The pure core: `core`, `scheduling` and `repository` import no accelerator runtime

**Status:** Accepted · 2026-08-22

**Context.** The behaviour most worth testing in this system is the scheduler: fairness
between cameras, load balance across GPUs, backpressure under saturation, batching windows.
The previous generation had none of it tested, because every code path went through
TensorRT and a test therefore needed a GPU.

**Decision.** `shipinfer.core`, `shipinfer.scheduling` and `shipinfer.repository` import
nothing heavier than numpy, pydantic and PyYAML. No torch, no tensorrt, no onnxruntime, no
fastapi. Device memory is referenced through the `MemoryHandle` *protocol* rather than an
import. The rule is enforced by `scripts/hooks/check_layers.py` at commit time and by
`tests/test_architecture.py` in CI.

**Consequences.** The default `pytest` run needs no GPU and finishes in seconds, so CI runs
on a cheap runner and every contributor can run it. The cost is one indirection: a device
tensor carries an opaque handle instead of a torch tensor, and the bridge lives in
`runtime/tensor.py`. That has been worth it every time.

---

## ADR-002 — One worker thread, one CUDA context, one GPU

**Status:** Accepted · 2026-08-22

**Context.** The obvious reading of "multi-GPU" is that a thread reaches across into
another GPU's memory. That requires peer-to-peer access, works only on some topologies, and
makes every ownership question hard.

**Decision.** A GPU is the unit of a *worker*. Each model instance owns one worker thread,
which calls `set_device` once at start-up and never again. Work moves between GPUs by being
**queued** somewhere else — a CPU-side decision made on queue-depth metadata alone. Nothing
in this codebase performs a cross-device memory access.

**Consequences.** Load balancing becomes a scheduling problem instead of a topology
problem, and it works identically on PCIe and NVLink. The price is that a payload which
must move between GPUs goes through host memory; ADR-004 is the policy that makes that
rare. The `(camera_id, frame_id)` tag is what makes reordering safe, so it is carried on
every request and never rewritten.

---

## ADR-003 — Torch is the runtime substrate; we do not reimplement it

**Status:** Accepted · 2026-08-22 · *the most consequential decision here*

**Context.** An early version of this server hand-rolled its own CUDA layer against
`cuda-python`: a bucketed caching allocator, pinned-memory management, stream wrappers, and
CUDA graph capture with raw `cudaStreamBeginCapture`. It worked. It was also slower than
torch, missing the parts that are genuinely hard (stream-aware block reuse, graph-aware
allocation, side-stream warm-up, shared graph memory pools), and it would have needed a
second implementation for ROCm.

**Decision.** **Torch is a hard dependency and the runtime substrate.** Allocation,
transfers, streams, events and CUDA graph capture all come from it. On ROCm, `torch.cuda`
*is* the HIP API, so AMD support costs nothing extra. Custom code is written in exactly two
places: the layer above (scheduling, batching, placement) and fused kernels torch has no
equivalent for.

vLLM, SGLang and TensorRT-LLM all make the same call, for the same reasons.

**Consequences.** Far less code, and it is faster. The hand-written implementations were
kept — as registry-selectable `custom_*` variants with docstrings explaining what the
library does that they do not. They serve as runnable documentation and as the other half
of the parity tests; they are not the production path. `runtime/providers/` exists only to
support them and as an escape hatch for a driver call torch does not expose.

---

## ADR-004 — Locality-aware spillover is the default placement policy

**Status:** Accepted · 2026-08-22

**Context.** Two policies are obviously wrong for this workload. Pinning a camera to a GPU
is what produced the imbalance in the previous system. Pure load balancing ignores that a
1080p frame is ~6 MB and moving it costs a D2H plus an H2D, while a person crop is a few KB
and can go anywhere.

**Decision.** The default is `locality_spillover`: run on the GPU where the data already
is, *until* that GPU's queue exceeds `spill_threshold` (default 4), then fall back to
power-of-two-choices. `spill_threshold: 0` degenerates to pure balancing and a large value
to pinning; neither extreme is right, which is why it is a knob.

Power-of-two-choices rather than join-shortest-queue as the fallback: two random probes
capture almost all the benefit, cost O(1) in pool size, and avoid the herding that makes
JSQ misbehave when many dispatchers read the same depths at once.

**Consequences.** The frame stays put through detect → segment → crop; only crops travel.
Measured on 8 × A5000 at 1000 req/s, the per-device share stays inside 11.7–13.2%.

---

## ADR-005 — Fair queueing and honest backpressure, not a shared eviction buffer

**Status:** Accepted · 2026-08-22

**Context.** The previous system pushed every camera's detections into one shared
1000-entry buffer that evicted the **oldest** entry when full. A crowded camera produced 30
detections per frame where a quiet one produced 2, so the quiet camera's work was evicted
before it finished the pipeline. Nothing logged it. `docs/flow.md` in `references/` records
the symptom: "camera đông người được nhận diện đầy đủ, camera vắng người thỉnh thoảng bị
miss".

**Decision.** Three changes, all in `scheduling/queues/`:

1. **Per-instance bounded queues**, never one shared buffer.
2. **Fair queueing**: requests are bucketed by `camera_id` and drained round-robin, so one
   camera cannot occupy consecutive batch slots. O(1) per operation.
3. **Honest overflow**: the default is to raise `QueueFullError` carrying depth and
   capacity. Where eviction is chosen instead, it drops the oldest request of the
   *greediest* camera in the lowest-priority lane — penalising the flood, not its victim.

Priority lanes sit above fairness, so a `TRACKING_CRITICAL` request from a camera about to
lose a target jumps the queue. That is the customisation a generic server cannot express
and the reason this one exists.

**Consequences.** Backpressure now reaches the producer as a signal it can act on. A test
pins the behaviour: with 8× skew, every quiet camera is served every time it asks.

---

## ADR-006 — The model repository is Triton's, unchanged

**Status:** Accepted · 2026-08-22

**Context.** We need an on-disk format for models, versions and per-model configuration.

**Decision.** Use Triton's layout (`<model>/config.*` plus numbered version directories)
and its config vocabulary — `max_batch_size`, `instance_group` with per-device `count`,
`dynamic_batching` with `max_queue_delay` and `preferred_batch_size`, `version_policy`,
ensembles. YAML instead of protobuf text, because the rest of the project already parses
YAML and `config.pbtxt` needs a protobuf dependency to read reliably; a `config.pbtxt`
found in a repository produces an explicit "convert it" error rather than being ignored.

**Consequences.** An existing Triton repository can be pointed at this server after a
config translation and nothing else moves. The semantics people already know — `count` is
per device, an empty `gpus` means all of them — apply unchanged, which is what lets one
config file run on a 2-GPU dev box and a 16-GPU node.

---

## ADR-007 — Python owns the control plane, C++/CUDA owns the data plane

**Status:** Accepted · 2026-08-22

**Context.** The control plane (configuration, lifecycle, repository scanning, HTTP) runs
once and benefits from being readable. The data plane runs 15 000 times a second.

**Decision.** Python for the control plane; `native/` (C++17 + CUDA/HIP, pybind11) for
fused kernels. The extension is **optional** — every native component has a Python
counterpart implementing the same contract, and `execution.provider` selects between them:
`auto` (native if importable), `native` (or refuse to start), `python` (always).

The boundary is numpy in, **device pointer out**. Every kernel exposes an `_into` entry
point that writes straight into a caller-owned torch CUDA tensor; the numpy-returning form
exists for parity tests. Preprocessing feeds an engine on the same device, so returning to
the host first would undo most of what the fusion saved.

**Consequences.** A machine with no build still runs, just slower, and `provider: native`
means a production deploy fails loudly rather than silently regressing. Measured on an
A5000: the fused letterbox writing into a torch tensor runs 8 × 1080p → 640² in 9.7 ms
against torch's 13.7 ms, bit-identical output.

---

## ADR-008 — CUDA graphs need persistent I/O buffers

**Status:** Accepted · 2026-08-22 · **superseded in part by ADR-013** (default flipped off)

**Context.** For the small, launch-bound models here — a 512-d embedder at batch 8 — kernel
launch overhead is a large share of wall time. Graph replay collapses an entire inference
into one `cudaGraphLaunch`. vLLM applies the same technique to its decode step.

**Decision.** Capture one graph per batch size, using `torch.cuda.CUDAGraph` (side-stream
warm-up, shared memory pool, allocator cooperation — see ADR-003). The precondition is ours
to honour: TensorRT binding buffers are allocated **once at load**, sized for
`max_batch_size`, and never reallocated; a smaller batch uses a prefix. Pinned staging comes
from `MemoryPool.staging`, which is keyed on shape so addresses are stable.

Capture failure is not an error: it means "take the ordinary launch path". After
`cuda_graph_max_capture_failures` the cache stops trying, so a model with dynamic control
flow does not pay a failed capture on every batch.

**Consequences.** Buffers are sized for the worst case, so memory is held that a small batch
does not use. That is the trade, and `max_batch_size` is the knob.

The default-on posture in this ADR no longer holds — see ADR-013. The mechanism, the
preconditions and the capture-failure policy above are unchanged.

---

## ADR-009 — The response cache is off by default

**Status:** Accepted · 2026-08-22

**Context.** A static camera watching an unchanged scene produces byte-identical crops frame
after frame. Re-embedding a parked ship sixty times a minute competes for the same GPU as
the moving one. Triton has a response cache; vLLM's prefix cache is the same idea.

**Decision.** Implemented (`server/cache/`, keyed on a BLAKE2b hash of every input byte),
**opt-in per model** via `parameters.response_cache`, and **off by default**.

**Consequences.** It is only sound for a deterministic, stateless model. Enabling it on a
stateful or stochastic one returns a stale answer with full confidence, which is far worse
than being slow. Defaulting it off means the failure mode requires someone to have decided.
No model in the demo repository enables it. `ship_recognizer` used to — a pure gallery
lookup, the one shape a cache is safe for — and it was removed when vessel identity
moved to the stateful tier, so the example went with it. The default is unchanged.

---

## ADR-010 — Libraries with their own lifecycle get their own repository

**Status:** Superseded by [ADR-012](#adr-012--one-algorithms-repository-not-four) · 2026-08-23

**Context.** The fused CUDA/HIP kernels started life as `native/` inside this repository.
They are not part of the server, though: a kernel that turns a batch of frames into a
normalised NCHW tensor is useful to anything that feeds a vision model, and the same is
true of the tracking, re-identification and multi-camera-association work that is coming.
Keeping them in-tree means one CI, one release, and one reviewer for two very different
kinds of code — a Python control plane and a set of GPU kernels, which need entirely
different things looked at in review.

**Decision.** Each such library is its own repository under the `shipinfer-` prefix,
vendored here as a git submodule under `3rdparty/`. The first is
`shipinfer-imgproc`; `shipinfer-mot`, `shipinfer-reid` and `shipinfer-mtmc` follow.

Three rules make it work rather than merely tidy:

1. **The dependency is one-way.** A module repository never imports `shipinfer`. It is a
   standalone library with its own types; the server adapts it. That is what keeps it
   independently useful and independently testable.
2. **Each repository carries its own reviewer.** `.github/reviewer-prompt.md` holds a
   domain specialist — a GPU kernel engineer for `imgproc`, a tracking engineer for `mot` —
   and the pipeline reads it. The workflow files are identical across the fleet and
   generated from `scripts/templates/module-repo/`, so the shape never drifts while the
   expertise stays specific.
3. **A submodule bump is its own commit.** A kernel change and a server change are never
   entangled in one revert.

The name is the function, not the implementation. `native` said how the code was written;
`imgproc` says what it does, and it is the term the reference services in this fleet
already use for exactly these kernels.

**Consequences.** A change spanning the server and a kernel is now two PRs and a pin bump —
that is the real cost, and it is the reason the seam between them is deliberately narrow
(numpy in, device pointer out). In exchange, the kernels get a reviewer who reads PTX, CI
that installs a CUDA toolkit only where one is needed, and a release cadence of their own.

The parent's CI deliberately does **not** check the submodule out for the offline tier.
That is not an oversight: the server must run without the kernels, and a test suite that
always has them would never prove it.

---

## ADR-011 — Ingest depends on a sink protocol it owns, not on the scheduler

**Status:** Accepted · 2026-08-23

**Context.** The ingest plane produces tagged frames; the inference pool consumes them. The
obvious wiring is for a camera actor to build the `WorkItem` and push it into
`scheduling.queues.RequestQueue` directly — one fewer indirection, and `scheduling` is a pure
layer, so the import breaks nothing ADR-001 was protecting. The first implementation of
`ingest/` did exactly that, and it required adding an `ingest → scheduling` edge to the
layering DAG.

**Decision.** `ingest` depends on a one-method `FrameSink` protocol that it defines
(`ingest/sink.py`), and `pipeline` supplies the `RequestQueue`-backed implementation.
`ingest` imports `core` and `runtime` and nothing else.

Deciding that a frame becomes a request for a particular model, in a particular queue, at a
particular priority, is **dispatch policy**. It belongs beside the DAG that consumes it,
because the code that performs that mapping is the same code that must undo it when it
reassembles per-frame results — and one decision split across two packages is one that
drifts. `ingest`'s job ends at "here is a tagged frame".

The protocol signals refusal by **raising**, not by returning a bool. An RTSP camera cannot
be backpressured: it sends the next frame whether or not anyone is ready. So something must
drop, and the camera actor is the only component in the system that knows which camera a
frame came from and can therefore charge the drop to the camera that caused it — which is the
whole substance of ADR-005. The two errors are `QueueFullError` (drop this frame, continue)
and `RequestCancelledError` (the consumer is gone, stop); both already live in `core.errors`
and are already what `RequestQueue.put` raises, so the adapter needs no translation layer.

**Consequences.** The layering DAG in ADR-001 is unchanged, which matters more than one edge:
the moment a layer rule is loosened to accommodate code, the next loosening is easier to
argue for. `pipeline` owns both halves of the frame-to-request mapping. And the component
that has to scale — 50 cameras at 20 fps, 1000 frames a second — is measurable against
`CountingSink` with no scheduler in the process at all, which is how ingest throughput gets a
number that is about ingest.

The cost is one indirection and two shipped sinks that are not the production path
(`CountingSink`, `BoundedSink`). Their docstrings say so, and `BoundedSink` says explicitly
that it is *not* camera-fair: fairness is the scheduler's job, and a second, subtly different
implementation of it inside `ingest` is exactly what ADR-005 warns against.

---

## ADR-012 — One algorithms repository, not four

**Status:** Accepted · 2026-08-23 · supersedes [ADR-010](#adr-010--libraries-with-their-own-lifecycle-get-their-own-repository)

**Context.** ADR-010 gave each library its own repository under the `shipinfer-` prefix:
`shipinfer-imgproc`, and `-mot`, `-reid`, `-mtmc` as they arrived. The reasoning was sound
— a Python control plane and a set of GPU kernels want different reviewers — and the
practice did not survive contact with the work.

Three things went wrong. The libraries were not independent: a tracker needs the
re-identification distance metrics, multi-camera association needs both, and the evaluation
harness needs all three, so a change to a shared type meant a coordinated bump across four
repositories in one afternoon. Their *lifecycle* turned out to be identical — they are
released together because they are used together. And the split cost correctness here: this
repository's kernel loader still imported `shipinfer_imgproc` after the merge, so
`provider: native` could never resolve and the fused-kernel seam was dead code that no test
noticed, because every test either uses the numpy backend or skips without a device.

**Decision.** One repository, `shipvision`, holding imgproc, detection, reid, tracking, mtmc,
eval and tune, vendored here as `3rdparty/shipvision`. The boundary ADR-010 actually cared
about — algorithms are not the server, and the dependency runs one way — is unchanged and is
now enforced by a test in that repository that fails if it imports `shipinfer`.

What is kept from ADR-010: it is still a separate repository with its own CI and its own
reviewer, the parent still pins a commit, and a submodule bump is still its own commit so a
kernel change and a server change are never entangled in one revert.

**Consequences.** One reviewer sees Python and CUDA in the same pull request, which ADR-010
wanted to avoid; the mitigation is that repository's own review configuration rather than a
repository boundary. Four remotes were deleted, so their history lives only in `shipvision`.
And the rename has to be complete to mean anything — `runtime/native.py`, `runtime/ops/`,
`pyproject.toml`, the CI workflow and the documentation all named the old module, and a
half-applied rename is what turned this from a packaging decision into a silent capability
loss.

---

## ADR-013 — CUDA graphs are opt-in, not the default

**Status:** Accepted · 2026-08-23 · supersedes the default in ADR-008

**Context.** ADR-008 turned graph capture on by default on the strength of the mechanism: for
launch-bound models, replay collapses an inference into one `cudaGraphLaunch`. That reasoning
is still right, and it was never measured against the cost of *getting* the graphs.

The first real 50-camera run measured it. Start-up went from 13.71 s to 97.82 s — seven
times — because capture happens per model, per batch size, per instance, and this
deployment has four of each. Nothing in the steady state paid that back on the runs we
can currently perform, and every one of those runs is a benchmark or a test whose whole
cost is dominated by start-up.

**Decision.** `execution.cuda_graphs` defaults to **False**. It stays a first-class setting
with the mechanism from ADR-008 intact behind it, and the docstring carries the two numbers
so the next person can weigh them rather than rediscover them.

**Consequences.** A long-lived production server that would amortise 84 s of capture over
days is now opting in rather than opting out, which is the wrong default *for that
deployment* — stated here so the flip is a decision and not an oversight. The knob is one
line of config, and ADR-008's preconditions (buffers allocated once at load, pinned staging
from a shape-keyed pool) remain load-bearing for anyone who sets it.

What this does not change: capture failure still means "take the ordinary launch path", and
`cuda_graph_max_capture_failures` still stops a model with dynamic control flow from paying
a failed capture on every batch.

---

## ADR-014 — The data plane is a C++ binary; `csrc/` is not an optional extension

**Status:** Accepted · 2026-08-24 · scopes ADR-007, restores ADR-003's portability rationale in
a second place

**Context.** `CLAUDE.md` has said from the first commit: Python for the control plane,
C++17/CUDA for the data plane. The control plane was Python and correct. The data plane was
*also* Python, and it capped throughput at 77 img/s — not through the GIL in the naive sense
(`docker stats` read 390-534% CPU, so the C extensions really were running in parallel) but
through the pure-Python share of the per-frame path, which held the GIL and pinned the whole
process at about five cores of forty-eight while every GPU queue sat empty.

Four candidates were eliminated by measurement before this was reached: the GPUs (every model
queue SUSTAINED at its offered rate), the worker pool (24/96/192 workers moved throughput under
8%), the reassembly lock (98.9% of its hold removed, no change), and the load generator (it
delivered 100% of what was offered).

**Decision.** `csrc/` is a **first-class data plane**, not an optional accelerator: a standalone
binary owning ingest, the fair queue, preprocessing, TensorRT execution, the perception graph
and reassembly. Python keeps the control plane — settings, the model repository, registries, the
CLI, the HTTP surface — and hands this plane a resolved configuration.

**How this differs from ADR-007, which is not superseded.** ADR-007 governs *fused kernels* in
`3rdparty/shipvision`: an optional pybind11 extension where every native component has a Python
counterpart selected by `execution.provider`, so a machine with no build still runs. That
promise is unchanged and still enforced by CI not checking the submodule out. ADR-014 is a
different artefact with a different contract: `csrc/` is a whole plane, it has no Python
counterpart, and a deployment either runs it or runs the Python plane — the choice is which
binary you start, not a provider string.

**Consequences.**

- **Two implementations of the same seams now exist** — the fair queue, reassembly, the graph.
  That is a real cost, and the mitigation is explicit: both are judged by *one* measurement
  (`benchmarks/harness/analysis.py` reads the same occupancy log from all three systems), and
  the C++ side names the Python file it mirrors. Where they have already diverged it is
  recorded: `fair.h` evicts the *newest* frame of the greediest camera while
  `scheduling/queues/lanes.py` evicts the *oldest* — a different latency profile under
  sustained overload, now stated in both.
- **ADR-003's portability argument had to be restored here.** The first version hard-coded the
  CUDA runtime throughout, which would have made a ROCm build a port rather than a flag.
  `csrc/shipinfer/core/platform.h` is the alias layer — mirroring shipvision's own — and is the
  only header in the tree permitted to name a vendor runtime.
- **`csrc/` mirrors `src/shipinfer/`'s package layout**, a thing's `.h` and `.cpp` next to each
  other and the Python names reused, so a reader who knows one tree can navigate the other.
- The Python data plane stays. It is the reference implementation, it is what the offline tier
  tests, and it is what makes a claim about the C++ side falsifiable.
