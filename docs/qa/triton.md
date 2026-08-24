# Triton Inference Server — three questions, answered from the source

Everything below was checked against the real repositories, cloned at 2026-08-23:
`triton-inference-server/{server, core, backend, tensorrt_backend}`. File and line references
are to those checkouts. Where I could not verify something I say so rather than filling the
gap from memory.

---

## 1. What language is Triton written in?

**C++, with Python for the client libraries, the Python backend, and the test suite.**

Line counts across `server`, `core`, `backend` and `tensorrt_backend`:

| | total | excluding `qa/` and tests |
|---|---|---|
| `.cc` | 87 136 | **71 969** |
| `.h` | 27 403 | **27 403** |
| `.py` | 106 510 | 18 048 |
| `.cu` | 81 | 81 |

So the server itself is ~99 000 lines of C++ and the Python is overwhelmingly test code —
`server/src` holds 26 C++ files against 10 Python ones. The 106k Python figure is misleading
if quoted alone, which is why it is broken down here.

Note the `.cu` count: **81 lines of CUDA in the entire server.** Triton is not a kernel
project. It orchestrates backends (TensorRT, ONNX Runtime, PyTorch, OpenVINO, Python) and
owns scheduling, batching, the model repository, and the protocols. The compute belongs to
whatever backend runs it. That is worth internalising before deciding what to copy from it.

---

## 2. Does Triton have CUDA graphs, and does it use them the way we do?

**Yes — and the important finding is *where* it lives.**

CUDA graph support is **not** in `server` or `core`. Grepping those two repositories for
`cuda_graph` finds only a QA test (`server/qa/L0_cuda_graph/trt_cuda_graph_test.py`). The
implementation is entirely inside the **TensorRT backend**:
`tensorrt_backend/src/instance_state.cc` and `instance_state.h`.

That is a deliberate architectural statement: graph capture is a property of *one* execution
runtime, so it sits behind the backend interface rather than in the server. A backend that
cannot capture graphs simply does not offer the option.

### How it works

Configured per model, not globally:

```
optimization { cuda { graphs: true } }
```
parsed at `tensorrt_backend/src/tensorrt_model.cc:84` into `use_cuda_graphs_`.

**Graphs are keyed on shape, and there are many per instance.** The storage is

```cpp
std::vector<std::map<std::vector<int64_t>, CudaGraph>> cuda_graph_execs_;
//         ^ one per event set          ^ keyed on the dims vector
```
(`instance_state.h:128`). One `cudaGraphExec_t` per (event set, concrete shape) pair — a graph
records addresses and sizes, so a different batch size is a different graph.

**Which shapes get captured, by default** (`instance_state.cc:3683-3700`, comment verbatim):

> Graphs are most likely to help for small batch sizes so by default build for batch sizes
> 1, 2, 3, 4, 6, 8, 12, 16, `max_batch_size`. If preferred batch size is specified, then the
> batch sizes will be 1, preferred batch sizes, `max_batch_size`.

So the capture set is derived from the *dynamic batching configuration*. That coupling is the
single most transferable idea here: the batch sizes a graph is captured for should be exactly
the batch sizes the batcher is configured to produce, or most requests miss the graph.

**Explicit control** exists via `graph_spec`, including a `lower_bound` spec, and
`allow_inexact_match_` (`instance_state.cc:3628-3642`) then lets a request whose shape is not
an exact key use a captured graph whose lower bound it satisfies — matched with
`std::map::lower_bound` and a per-dimension check.

**Two refusals worth copying:**

- **Shape tensors disqualify a model.** `BuildCudaGraph` walks every IO tensor and, if
  `engine_->isShapeInferenceIO(...)`, logs a warning and returns `false`
  (`instance_state.cc:3874-3886`). It declines rather than capturing something wrong.
- **A real `Enqueue` runs before capture**, with the comment "Enqueue to TRT to setup
  resources properly BEFORE capturing CUDA graph" (`instance_state.cc:3905-3912`). TensorRT
  allocates lazily on first execution; capturing before that records the allocation.

### Can it be switched on and off? (yes, in both, at different granularities)

**Triton: per model, in the config, not by environment variable.**

```
optimization { cuda { graphs: true } }
```
`tensorrt_backend/src/tensorrt_model.cc:84` reads it into `use_cuda_graphs_`, and
`instance_state.h:49` exposes `UseCudaGraphs()`. There is no global env override — the
decision is per model because whether a model *can* be captured is a property of the model:
`BuildCudaGraph` refuses outright when the engine uses shape-inference IO
(`instance_state.cc:3874`). A server-wide switch would have to be overridden per model
anyway, so Triton does not offer one.

**Ours: per model in settings, plus a global operator override.**

`execution.cuda_graphs` (default `True`) in `core/settings/execution.py` is the per-model
equivalent. On top of it, `SHIPINFER_CUDA_GRAPHS=on|off` overrides every model in one
restart:

```bash
SHIPINFER_CUDA_GRAPHS=off shipinfer serve      # is the graph path what is hurting?
```

The override exists for one job — answering that question without editing six config files —
and it wins over the settings when set, because an operator who typed it is running an
experiment and a config file silently overruling them would make the experiment useless.
Unset, every model keeps its own setting. An unparseable value raises rather than falling
back, so a typo cannot quietly leave graphs on.

It is read in `server/model.py`, not in `core/settings`: the layering rule forbids `core`
from importing `shipinfer.envs`, and rightly — a settings object whose meaning changes with
the environment cannot be reasoned about from the config file alone. `_graphs_enabled()` is
where the two are combined, and that is the only place.

### Does our server apply it the same way?

Partly, and the differences are worth naming.

| | Triton | Us |
|---|---|---|
| Where it lives | inside the TensorRT backend | `runtime/graphs/` — a **registry** (`torch` / `custom`), above the backend |
| Keyed on | (event set, exact dims), with lower-bound inexact match | shape + dtype, per instance |
| Capture set | derived from `preferred_batch_size` / `max_batch_size` | whatever is asked for |
| Warm-up before capture | mandatory real `Enqueue` | side-stream warm-up (ADR-008) |
| Stable addresses | binding buffers owned by the instance | `PinnedStagingPool` exists for exactly this |
| Refuses shape tensors | yes, explicitly, with a warning | **not checked** |

**What we should adopt:**

1. **Derive the capture set from the batcher's configuration.** We capture on demand; Triton
   captures the batch sizes the batcher will actually emit. Ours risks a graph per observed
   shape (unbounded) or a miss on every request (useless). This is a small change with a real
   effect and it is the one I would do first.
2. **Refuse to capture when the model cannot be captured**, and log it. Silence here means a
   deployment believes it is replaying graphs and is not — and our
   `shipinfer_cuda_graph_replays_total` counter would happily read zero without anyone asking
   why.
3. **Inexact (lower-bound) matching**, so a batch of 5 can replay the graph captured for 6 by
   padding. Triton's `allow_inexact_match_` is opt-in; it should be for us too.

**What our arrangement gets right:** putting graph capture behind a registry above the
backend means the `torch` implementation serves the torch backend and the TensorRT one, and
the hand-written `custom` variant exists to be read. Triton pays for its placement by having
no graph support at all in backends that could support it.

---

## 3. How does our system differ from Triton, and what should change?

### Features Triton has that we should take

| Feature | Where Triton does it | Why it matters here |
|---|---|---|
| **Per-model statistics endpoint** | `/v2/models/{m}/stats` | We have Prometheus histograms but no per-model view. An operator debugging one camera's model cannot get its numbers alone. |
| **The three-way compute split** | `nv_inference_compute_{input,infer,output}_duration_us` | **Adopted** — `runtime/profiling.py` now uses exactly these names. |
| **Request tracing with named timestamps** | `--trace-config`, `REQUEST_START … REQUEST_END` | We stamp six points in `Timings` but have no trace sink. Names are already aligned. |
| **`graph_spec` derived from the batcher** | `instance_state.cc:3683` | See §2. |
| **Rate limiter** | `core/src/rate_limiter.*` | Bounds concurrent executions per model. We bound queue depth, which is not the same thing: a burst can still put every instance into compute at once. |
| **Model warm-up in the config** | `model_warmup` in `config.pbtxt` | We warm up with a fixed iteration count; Triton lets the operator supply real sample inputs, which is what actually populates the right kernels. |
| **Explicit model control** | load/unload API | We load everything at start-up. Fifty cameras and six models is fine; a growing repository is not. |
| **Ensemble scheduling as a first-class scheduler** | `core/src/ensemble_scheduler` | Ours executes the DAG on a thread per request. Triton's ensemble steps are scheduled independently, so a slow stage does not hold a thread. |

### Where we should NOT copy Triton

- **gRPC and shared-memory protocols.** Triton needs them because clients are separate
  processes. Our ingest is in-process by design (ADR-011); adding a protocol between the
  camera and the model would add a serialisation of the frame, which is the largest object in
  the system.
- **Its backend ABI (`TRITONBACKEND_*`).** A C ABI exists so third parties can ship binary
  backends. We control every backend; a Python ABC costs nothing and is far more readable.
- **81 lines of CUDA.** Triton delegates compute entirely. Our fused kernels are the
  measured 50× on preprocessing and 1.5× on letterbox — that is a real advantage of *not*
  being Triton, and it should stay.

### What of ours is arguably worse than Triton and should be reconsidered

Stated as questions to settle with measurements, not as conclusions:

1. **`instance._execute` is fully synchronous.** Assemble → execute → scatter, one batch at a
   time, so the device idles through both copies. Triton has `num_copy_streams` and overlaps
   them. `shipinfer_device_idle_ratio` was added to measure exactly this; the decision waits
   on the number.
2. **Our `custom` allocator and `custom` graph variants** exist to be read, not run. They are
   honest about that, but they are also code that must keep compiling. If they ever diverge
   from the `torch` variants they become misleading rather than educational.
3. **The response cache** is off by default and has one implementation. Triton's is
   shared-memory-backed and cross-instance. Ours is per-process, so at 16 GPUs it is 16
   caches — which may be the wrong shape rather than merely a smaller one.
4. **We have no rate limiter.** See above.

---

## What I could not verify

- Triton's *measured* CUDA graph speed-up. The comment says graphs help most at small batch
  sizes; I did not benchmark it and neither did the repository, in anything I found.
- Whether `allow_inexact_match_` is on by default in any shipped configuration. The code
  initialises it to `false` and sets it from the graph specs; I did not trace every path.
- The rate limiter's exact policy. I read its existence, not its algorithm.
