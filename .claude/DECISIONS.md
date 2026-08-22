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

**Status:** Accepted · 2026-08-22

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
In the demo repository only `ship_recognizer` enables it — a pure gallery lookup.
