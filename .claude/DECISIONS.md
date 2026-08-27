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

**What it is not yet (V79).** The operator read `csrc/` against `src/shipinfer/` and saw a
different program. That was accurate. The first binary shares the Python plane's layout and the
names of its seams, but not their shape: a pool of workers leasing instances instead of one
thread per instance with its own bounded queue; a device-local `lease()` instead of a placement
policy chosen by name; a fixed 50 ms drain instead of the batch window; newest-first eviction
where the Python queue evicts oldest; replay-only ingest; engines named on the command line
instead of read from the model repository. It answered the question it was written for — the
interpreter is the wall, 77 against ~450 img/s per process — and that is what it is evidence of.

**The decision (V80) is to port for real.** Every per-frame seam exists in both planes and must
be the *same* seam, ported in the order that removes the largest architectural difference first
(ledger Phase 8: instance thread + queue + dispatcher/policy → batch window → eviction order →
ingest → resolved config in, same events out), with a cross-plane parity harness as the gate.
And a **sync rule**: a change to a Python data-plane seam is not finished until the C++ seam
carries it and the parity harness agrees (CLAUDE.md, "Two planes, one architecture").

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

**Amended by ADR-016 (2026-08-27).** The threading core of this decision — one worker
thread, one CUDA context of its own, one GPU, placement decided on CPU-side metadata — is
unchanged. The clauses *"nothing in this codebase performs a cross-device memory access"*
and *"a payload which must move between GPUs goes through host memory"* are superseded:
DataPool payloads move VRAM→VRAM over a per-pair timed route, and a shard holds a bounded
number of peer mappings. The "works only on some topologies" caveat that motivated the old
clause was right and is now measured rather than avoided (ADR-016 §context).

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
a second place · **scope amended 2026-08-25 (V79, V80):** the binary as it stands is the *starting
point* of the plane, not the plane — see "What it is not yet" below.

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

## ADR-015 — Inference crosses processes through pinned host memory, never CUDA IPC

**Status:** Superseded by ADR-016 (2026-08-27) for the *payload* transport — the rings it
specifies survive as the control channel and as the RAM-fallback payload path; its two
objections to CUDA IPC are answered there, not waved away. Originally: Accepted · 2026-08-26 · builds on ADR-002 (spills cross through the host), ADR-005
(backpressure is typed and carried), ADR-006 (one process per shard) and the topology seam (#18).

**Context.** The fleet gives every shard its own process, its own GPU and its own cameras, and
so cannot balance a crowded shard against a quiet one: the crop-stage models — by default the two
embedders; the segmenter's 39 MB batches are the operator's call (its rings would dwarf
theirs) — stateless, one crop batch per request, saturate on one GPU while idling on
the next. Sharing them across processes needs a transport for ~15 000 crops/s of small batches
with a result each, and a load signal the dispatcher can read without asking.

Three transports were weighed. **CUDA IPC** (`cudaIpcGetMemHandle` / `OpenMemHandle`) would keep
crops on the device, but opening a peer's device memory needs a CUDA context on that device in
the opening process — one per peer GPU per shard, which is the per-process context cost the
fleet exists to avoid — and the handle lifecycle (open once, never close under a live kernel)
is a second protocol on top of the first. **gRPC/HTTP** serialises every batch twice and puts
the kernel's network stack on the hot path for a transfer that never leaves the machine.
**Shared host memory**, pinned, is what the spill path already does (ADR-002): the owner's H2D is
a DMA from the slot, the submitter's D2H lands in it, nothing is serialised beyond a fixed-size
header per tensor.

**Decision.** Cross-process inference is a **pinned shared-memory ring per (submitter, owner,
model), single writer each**, requests one way and results the other, in the shape of vLLM's
`ShmRingBuffer`: a slot is FREE → CLAIMED (writer) → WRITTEN (published) → TAKEN (reader) →
FREE (released after the result is written), and the ring header — depth, EWMA latency,
heartbeat, closed — is the load signal, read lock-free like a local instance's depth. A peer's
model appears to the dispatcher as a `Placeable` (`RemoteInstance`) whose `device` is `cpu`, so
the scheduler is untouched and a proxy never equals a request's resident device. Only crops
cross; the detector is never shared; a request that would not fit a slot is refused, not
split. Failures are typed on the wire: a full ring is a `QueueFullError` the dispatcher spills
on; a silent peer fails its in-flight requests with `PeerLostError` carrying the (camera, frame)
tags; the owner's exception crosses as text.

**Consequences.**

- **A pinned budget, derived once.** A directed pair has two rings (requests one way, results
  the other), so the box holds `N × (N−1) × models × 2` rings, each existing once; every ring
  is mapped by its writer and its reader, and each registers its own mapping, so a process pins
  `4 × (N−1) × models` rings. Slots are sized **per model and per direction** from the model's
  own config (`wire_slot_bytes`: max_batch × the tensors' bytes + 64 KiB for the heads,
  page-rounded; `slot_bytes` only when an extent is dynamic) — both processes derive the same
  numbers from one repository, so nothing is negotiated. For 4 shards and the repository's two
  embedders at 8 slots: request slots are 6.36 MB (16 × 3 × 256 × 128 fp32) and response slots
  0.2 MB, so **48 rings ≈ 1.26 GB of shared memory on the box, existing once; ≈ 0.63 GB mapped
  and registered per process** — the request rings dominate. The rings stay *pairwise and
  small* rather than one big ring per owner: N writers on one ring would need a
  compare-and-swap Python does not have on shared memory.
- **Two copies per remote request** (D2H into the slot, H2D out of it) against zero for a local
  one. That is the price of not opening peer contexts; the policy only pays it when the local
  queue is past its spill threshold, and the per-device counters show how often.
- **Loss is bounded and named.** A dead shard loses its cameras and its capacity; the requests
  it held for peers fail with their tags within `lost_after_ms`; nothing hangs. The fleet
  supervisor still treats a dead shard as a fleet failure (ADR-006) — `service` changes *what is
  lost*, not whether a dead shard is a failure.
- **The rings live in `/dev/shm`, and a container's is 64 MiB by default.** A deployment that
  runs the `service` topology in a container sets `--shm-size` to at least the box budget above;
  the tests size their rings down instead of assuming the host's.
- **The protocol is versioned** (`RING_VERSION`, `WIRE_VERSION`); a mismatch at `open` is a
  `RingProtocolError` naming both, so two builds cannot talk past each other.
- **Not decided here:** slot size per model, and whether the tier should ever carry frames. Both
  are asked of the operator in the topology PR.

---

## ADR-016 — DataPool: VRAM-first sharing over CUDA IPC slabs, bounded to a link-probed neighbourhood

**Status:** Accepted · 2026-08-27 · operator decisions V137/V138/V139 · supersedes ADR-015 for
the payload transport; amends ADR-002's payload clauses; builds on ADR-003 (torch owns the
allocator and the IPC machinery), ADR-005 (backpressure typed and carried), ADR-014 (`gpu*`
aliases are the only vendor names in `csrc/`).

**Context.** ADR-002 sent every cross-GPU payload through host memory because peer access
"works only on some topologies"; ADR-015 then built the spill transport on pinned rings and
rejected CUDA IPC on two grounds: **(1)** opening a peer's memory needs a CUDA context on
that device in the opening process — one per peer GPU per shard, "the per-process context
cost the fleet exists to avoid" (quantified in the ledger as `G` contexts per process,
`G² × ~300 MiB` box-wide); **(2)** the handle lifecycle — open once, never close under a
live kernel — is a second protocol on top of the first. Both objections are correct as
stated. What changed is the operator's requirement (V137): a payload that already sits in
VRAM must not be staged through RAM to reach another GPU ("gây down performance cực kì
mạnh"); cross-GPU access is permitted, the only criteria being perf and accuracy. And the
topology caveat is no longer folklore — it was measured on this box (Appendix A of
`docs/arch.md`): NVLink pairs move a 12 MB frame in 261 µs; a PXB pair moves it in
**98.6 ms** over direct P2P (three orders of magnitude, three pairs reproduced) and in
996 µs when staged; `cudaDeviceCanAccessPeer` says "yes" to all of them.

**Decision.**

1. **Payloads live in a DataPool** — per-shard pre-allocated VRAM slabs (plus one pinned
   host slab as the fallback location) carved into buffers; a buffer is referenced by a
   **ticket** `(slab, offset, size, format, tag)`. The ADR-015 rings are kept **as the
   control channel** that carries tickets, and as the payload path only in RAM-fallback
   mode. This is Triton's CUDA-shared-memory pattern (register a region once, reference
   region/offset/size per request) applied inside one node.
2. **Slab handles are exchanged once per pair at mesh join and opened once.** Never per
   buffer, never per request.
3. **The neighbourhood is bounded — this answers objection (1).** A shard opens the slabs of
   at most **K peers** (default K = 3, configurable), chosen by measured link quality
   (NVLink partners first). Every other peer is reached over the pinned-staged path (the
   ADR-015 transport, unchanged). So a shard holds **K + 1 contexts**, not G, and the
   box-wide context cost is `G × (K + 1) × C_ctx` instead of `G² × C_ctx`. The per-context
   cost `C_ctx` is a **measured input** (REPLACE_CTX_MIB MiB per foreign context on this
   box's A5000s under CUDA 12.6 — measured 2026-08-27, appendix), and the resulting
   **per-shard VRAM budget is written down at start-up** in the shard's Health report:
   `slabs + (K+1) × C_ctx + engines ≤ device memory − reserve`, refused before any camera
   opens if it does not hold. At K = 3 and C_ctx ≈ 300 MiB the context line is ~1.2 GB per
   shard on a 16-GPU node, against ADR-015's feared 4.8 GB.
4. **Every pair is timed at handshake — this is the topology caveat, made a routine.** One
   12 MB and one 128 KB copy per pair (a few ms, once) fill a route table:
   `direct` (NVLink/PIX-class) or `staged`. "Capable" is never trusted as "fast"; NCCL's
   `NCCL_P2P_LEVEL` distances (NVL/PIX/PXB/PHB/SYS) are the precedent. On the all-NVLink
   production node (V139) every route resolves to `direct`; the probe stays because the
   dev box demonstrably has poison pairs and because hardware assumptions change silently.
5. **Handle lifecycle — this answers objection (2).** Slabs are opened at join and
   **closed only at drain**, after the shard's engine has quiesced (no kernel may be in
   flight on a mapping being torn down). Tickets carry the slab **generation**; a shard
   that restarts issues a new generation, and a ticket against a stale generation is
   refused typed (`StaleTicketError`, carrying its `(camera, frame)` tag) rather than
   dereferenced. Peer death is signalled by the launcher over gRPC (V140) and invalidates
   that peer's mappings; in-flight tickets on them fail with `PeerLostError` as in ADR-015.
   A buffer that is the I/O of a captured CUDA graph is **pinned for the graph's life and
   never recycled** (ADR-008 applies to slab carves exactly as to any captured buffer).
6. **Two spill tiers, one pool** (V138): frame tickets when a shard's detect queue is deep
   (camera/fps skew), crop tickets when embed/segment queues are deep (crowding);
   independent thresholds; locality follows the data (a spilled frame's crops are cut and
   embedded where it landed); results always return to the camera's home shard.
7. **Torch is normative for the implementation** (ADR-003): slabs are torch allocations,
   handles travel through `torch.multiprocessing`'s CUDA-IPC sharing, peer copies are torch
   copies. The `cudaIpc*` / `cudaMemcpyPeer` names in this ADR and in `docs/arch.md` state
   the *semantics*; where a raw call is unavoidable in `csrc/`, it goes through
   `core/platform.h`'s `gpu*` aliases (ADR-014) or the ROCm build breaks.

**Consequences.**

- **Zero or one copy per shared payload** instead of ADR-015's two, and no payload byte
  ever transits host memory between two GPUs that have a `direct` or `staged`-VRAM route.
- **A context budget exists and is enforced**, which ADR-015 rightly said was missing; it
  is a start-up refusal, not a runtime surprise.
- **Reach is unchanged, transport is tiered**: neighbours over VRAM, everyone else over the
  ADR-015 rings — the fleet never loses a spill target because of K.
- **Two protocols after all** — ADR-015's objection stands as a cost, now paid explicitly:
  the ticket/generation rules above are the lifecycle protocol, tested at the unit level
  (stale ticket, dead peer, drain under load) before any camera uses them.
- **The dev box needs the probe** (pairs 0-3, 1-3, 2-4 are PXB-poison); the production
  node does not, and pays a few ms once for it anyway.
- **Not decided here:** K's default beyond 3, and whether the deepstream runner's graphs
  publish their NVMM buffers into the same slabs (phase E of the migration plan).
