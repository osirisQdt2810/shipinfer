# Coding Conventions

Read this in full before non-trivial work. Part 1 is ordinary Python discipline; Part 2 is
what makes *this* codebase work; Part 3 is how to approach a task here.

---

## Part 1 — Universal Python

- **Formatting** is not a discussion: `black` (line length 96), `isort` (profile=black),
  `ruff`. `pre-commit run --all-files` before pushing.
- **Type hints everywhere** in `src/`. `mypy --strict` is the target; it is not a commit
  gate yet, but new code should not add to the debt. `from __future__ import annotations`
  at the top of every module.
- **Google-style docstrings** on every public class and function, and **short** — see the
  cap below. Document *why*, and the failure modes; the signature already says what the
  arguments are.
- **Documentation is capped (V145).** Module docstring **≤ 15 lines**, class/function
  **≤ 10**, comment block **≤ 4**. A line earns its place by saying what the code cannot:
  a constraint, a failure mode, a decision *and* its reason. Longer reasoning goes in an
  ADR (`.claude/DECISIONS.md`) or the PR body — once, with a pointer. `# doc: long <reason>`
  above the symbol exempts it. Check: `python3 scripts/hooks/check_docs.py [paths]`.
  Markdown too: a `FEATURE_LOG.md` entry **≤ 15 lines**, an ADR **≤ 30**, a PR body section
  **≤ 20**. Cut history, apologetics and anything a test already proves.
- **Errors are typed.** Raise from `shipinfer.core.errors`. Never return `None`, `[]` or
  `{}` to mean "something went wrong"; a dropped frame, a full queue and a dead GPU are
  three different events and an empty list distinguishes none of them.
- **Dataclasses for values**, `slots=True` when they are created per request. `frozen=True`
  unless mutation is the point.
- **No module-level side effects** beyond registry registration. No configuration at import
  time, no logging handlers, no CUDA calls.
- **Names say what a thing is, not what it wraps.** `FairPriorityQueue`, not `QueueImpl2`.

---

## Part 2 — This project

### 2.1 The ponytail principle — reuse, do not reimplement

**If a mature, optimised library already does it, use it.** The standing example is torch:
its caching device allocator, caching pinned-host allocator, streams, events and CUDA graph
capture are all better than anything written here, and on ROCm the same API covers HIP for
free. vLLM, SGLang and TensorRT-LLM all build *on* torch rather than beside it, and so does
this project (ADR-003).

Custom code is justified in exactly two places:

1. **The layer above** — scheduling, batching, placement, the pipeline. No library does
   this for our constraints, and it is where the differentiation is.
2. **Fused kernels with no library equivalent** — resize + colour convert + normalise +
   NHWC→NCHW as one pass instead of four. That lives in `native/`.

Everything else — allocation, transfers, NMS, interpolation, matmul, collectives, config
parsing, HTTP — comes from torch, torchvision, TensorRT, NCCL, pydantic or FastAPI.

Where a hand-written implementation already exists and is worth keeping, it stays as a
registry-selectable `Custom*` variant next to the library-backed default: `custom_caching`,
`custom_device`, `custom_pinned`, the `custom` graph cache, the raw `providers/`. They are
**documentation you can run** — read them to see what the library is doing for you — and
the parity tests keep them honest. They are not the production path and their docstrings
say so.

### 2.2 The layering rule

```
core  ←  repository, scheduling          (pure: numpy + pydantic, nothing heavier)
core  ←  runtime                         (the accelerator seam: torch lives here)
core, repository, runtime  ←  backends
everything above  ←  server  ←  pipeline
```

`core/`, `scheduling/` and `repository/` must import **no torch, no tensorrt, no
onnxruntime, no fastapi**. This is not tidiness: it is what makes the scheduler's fairness
and balancing behaviour testable on a laptop, and tests that need sixteen GPUs get written
once and then never run.

Enforced twice, on purpose: `scripts/hooks/check_layers.py` at commit time and
`tests/test_architecture.py` in CI.

### 2.3 Package per extension point, registry per family

Anything that could gain a second implementation is a **package**, not a module:

```
policies/  queues/  batching/  backends/  providers/  memory/  graphs/  ops/
sinks/     exporters/  cache/  cli/commands/
```

with `base.py` (the ABC), `registry.py` (a `Registry` object), and one class per file
carrying `@REGISTRY.register("name", "alias")`. Use `register_lazy` when importing the
implementation is expensive — `shipinfer repo ls` must work on a laptop that has never had
TensorRT installed.

Adding an implementation is a new file and a decorator. If you find yourself editing an
`if/elif` to add a case, the design is wrong.

### 2.4 Threading and the GPU

- **One worker thread, one CUDA context, one GPU, for the thread's whole life.**
  `DeviceManager.bind_current_thread` is called once at start-up and never per request.
- **Nothing reaches into another GPU's memory.** Work moves between GPUs by being *queued*
  elsewhere — a CPU-side decision made on metadata alone. That is what makes multi-GPU
  balancing simple instead of a lesson in peer-to-peer topology (ADR-002).
- **The `(camera_id, frame_id)` tag rides every request untouched.** Batching, reordering
  and spillover are all safe *because* reassembly keys on the tag, not on arrival order.
- **Pinned memory or the copy is not async.** `cudaMemcpyAsync` from pageable memory
  silently degrades to a synchronous copy — no error, no obvious symptom beyond a GPU that
  is only 60% busy. Stage through `MemoryPool.staging`.
- **CUDA graphs need static shapes and stable addresses.** One graph per batch size; I/O
  buffers allocated once at load and reused. A buffer that is freed and reallocated turns a
  captured graph into a replay against freed memory.

### 2.5 Performance discipline

This is an inference server; the hot path is measured in microseconds.

- **Nothing allocates per request** on the dispatch path. Metric handles are resolved once,
  buffers come from a pool, `slots=True` on per-request dataclasses.
- **Queue and dispatch operations are O(1)** in the number of cameras and instances. The
  fair queue's round-robin is a deque rotation, not a scan.
- **Batch, then act.** Any per-image Python loop around a CUDA call is a bug in waiting;
  the ops interface is batched precisely so that is hard to write.
- **Measure before claiming.** A speedup claim needs a `shipinfer bench` table or a
  benchmark script in the PR, comparing like with like — a device-to-device path against
  another device-to-device path, not against one that also copies to the host.

### 2.6 Configuration

- Per-**deployment** settings live in `core/settings/` (env-overridable, `SHIPINFER_*`).
  Per-**model** settings live in that model's `config.yaml`. Keep the split sharp: it is
  what lets one repository run unchanged on a 2-GPU dev box and a 16-GPU node.
- The model config deliberately mirrors Triton's vocabulary (`max_batch_size`,
  `instance_group`, `dynamic_batching`, `preferred_batch_size`). Where we add something
  Triton has no word for (`condition` on an ensemble step), the key is **new**, never an
  overloaded reinterpretation of an existing one.
- Validate at start-up, not at first use. A mis-wired ensemble should stop a deploy.

### 2.7 Observability

- **One logger for the process** (V145): `get_logger()` returns the `shipinfer` logger;
  the module is a field, not a separate logger. Structured `extra=log_context(...)`.
  Library code configures nothing at import time.
- Production uses the **async** log sink: a synchronous handler does a blocking write while
  holding the handler lock, on the thread that is feeding a GPU.
- Anything an operator would page on gets a metric. `queue_depth`, `spills_total`,
  `requests_rejected_total` and the latency histograms are the ones that matter.

### 2.8 Native code (`3rdparty/shipvision`)

The kernels are a **separate repository**, vendored as a submodule (ADR-010). Change them
there, with their own tests and their own reviewer; bump the pointer here in its own commit.
The rules below are that repository's, restated so a reader of this file knows what to
expect of it.

- C++17, `clang-format` (enforced by that repository's pre-commit, not this one).
- Use the `gpu*` aliases from `core/platform.h`; never write `cudaMalloc` directly, or the
  ROCm build silently stops compiling.
- Every kernel gets a `_into` entry point taking a device pointer. The numpy-returning form
  is a convenience for parity tests, and its docstring must say so.
- **The GIL belongs at the boundary, never in the library.** `include/` and `src/` must not
  mention Python; `bindings/` crosses exactly once per entry point — prepare with the lock
  held, release around the compute, wrap the result. Scattered releases across nested
  helpers are how a `py::` access ends up on the wrong side of one, and that failure is an
  interpreter crash rather than an exception.
- **Rotate reused buffers behind an event.** A pinned buffer overwritten while its DMA is in
  flight produces plausible output and is invisible to a benchmark that submits the same
  image twice.
- Check the async error slot after every launch (`check_launch`). Without it an
  out-of-bounds write surfaces as an unrelated failure three calls later.
- Scratch buffers live on the object, not the call: `cudaMalloc` and `cudaFree` are
  synchronising.

### 2.9 Testing

Two tiers, and the split is load-bearing:

- **Offline** (`pytest`) — must pass with **no GPU**. Scheduling invariants, config
  validation, layering, numpy ops. This is where fairness and balancing are pinned.
- **GPU** (`pytest -m gpu`, `-m multigpu`) — real devices, parity between numpy/torch/native,
  allocations landing on the right device.

Rules that keep them useful:

- **Test the property, not the implementation.** "A loud camera takes 2 of 8 batch slots",
  not "the deque was rotated".
- **A fake is fine when the contract is narrow.** `FakeInstance` is four fields, which is
  itself evidence the `Placeable` protocol is the right size.
- **Every fused kernel needs a parity test** against the readable implementation.
- **A load test must bound its in-flight requests.** Firing everything at once at a bounded
  pool measures the test's lack of backpressure, not the server.

---

## Part 3 — Agent Working Principles

1. **Read before writing.** `.claude/JOURNAL.md`, `.claude/DECISIONS.md`, and the module
   you are about to change. The reference services under `references/` are the spec.
2. **Think, then code.** For anything non-trivial, state the plan — files, signatures, edge
   cases — before writing it. Use the `planner` agent when the change spans layers.
3. **Simplicity first.** The smallest design that satisfies the requirement. A registry with
   one implementation is fine; a plugin system with a config DSL is not.
4. **Surgical changes.** Touch what the task needs. Refactoring adjacent code because it
   offends you belongs in its own PR.
5. **Reuse before writing** — the ponytail principle applies to your own code too. Search
   for an existing helper before adding one.
6. **Finish the job.** Code, tests, docs, and the evidence that it works. A feature without
   a passing test is not done; a performance claim without a measurement is not done.
7. **Report honestly.** If a test fails, say so and paste it. If something was skipped, say
   what and why. Do not narrow the scope silently.
