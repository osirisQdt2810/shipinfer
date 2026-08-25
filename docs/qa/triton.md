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

`execution.cuda_graphs` (default `False` — ADR-013) in `core/settings/execution.py` is the per-model
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
| Capture set | derived from `preferred_batch_size` / `max_batch_size` | **the same** — `runtime/graphs/spec.py`, from the model's own batching window |
| Warm-up before capture | mandatory real `Enqueue` | side-stream warm-up (ADR-008) |
| Stable addresses | binding buffers owned by the instance | `PinnedStagingPool` exists for exactly this |
| Refuses shape tensors | yes, explicitly, with a warning | **not checked** |

**What we should adopt:**

1. **Derive the capture set from the batcher's configuration.** *(Done.)* We captured
   whatever `execution.cuda_graph_batch_sizes` listed — a deployment-wide default of
   `[1, 2, 4, 8, 16, 32]` that a model with `max_batch_size: 8` could only half use, and
   that a model with `preferred_batch_sizes: [6]` missed entirely. `runtime/graphs/spec.py`
   now applies Triton's rule to the model's own `BatchWindow`, so the capture set and the
   batcher are one answer rather than two that happen to agree. See §3.
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

All eight are adopted. The compute split already was, five landed together, and the last
two — the derived `graph_spec` and the ensemble scheduler — closed after them. The table is
the current state, not a plan: each row names the file, and where a decision differs from
Triton's it says so rather than leaving the reader to guess.

| Feature | Where Triton does it | Status here |
|---|---|---|
| **Per-model statistics endpoint** | `/v2/models/{m}/stats` | **Adopted** — `server/statistics.py` holds `ModelStatistics`; `GET /v2/models/{name}/stats` and `/v2/models/{name}/versions/{v}/stats` return Triton's `model_stats` array. Both of Triton's arithmetic conventions are followed and stated: every request in a batch is charged the whole batch's compute span, and `batch_stats` is keyed on **rows** rather than requests. |
| **The three-way compute split** | `nv_inference_compute_{input,infer,output}_duration_us` | **Adopted** — `runtime/profiling.py` uses exactly these names, and the same three spans are what `inference_stats` reports. |
| **Request tracing with named timestamps** | `--trace-config`, `REQUEST_START … REQUEST_END` | **Adopted** — `core/tracing/` owns the seven event names and the sinks (`none` by default, `jsonlines`), selected by `observability.trace_sink`. `TRACE_EVENTS` now lives there and `runtime/profiling.py` re-exports it, so the vocabulary has one home. Sampling is `rate=N`, as Triton's is: at 1000 frames a second tracing every request makes the instrument the bottleneck. |
| **`graph_spec` derived from the batcher** | `instance_state.cc:3683` | **Adopted** — `runtime/graphs/spec.py` applies Triton's rule (1, the preferred sizes, `max_batch_size`; otherwise the 1/2/3/4/6/8/12/16 ladder clamped to it) to the model's own `BatchWindow`, and `server/model.py` resolves it once per model. Two escapes, most specific first: `parameters.graph_spec` in the model's `config.yaml`, then `execution.cuda_graph_batch_sizes` **when the operator actually set it** — an untouched default no longer out-votes the model. A size outside the window stops the deploy rather than being clamped. `stats()["cuda_graphs"]` reports the sizes, the source and the reason whether or not capture is on. |
| **Rate limiter** | `core/src/rate_limiter.*` | **Adopted** — `scheduling/limits/`, registry-selectable (`off` is the default), configured per model as `rate_limiter: {kind, max_concurrent_executions}`. Triton's general named-resource model is deliberately *not* copied: the only resource this pipeline has ever needed to bound is "an execution". Instances that miss a slot **wait**; shedding stays at the queue, where the caller learns about it. |
| **Model warm-up in the config** | `model_warmup` in `config.pbtxt` | **Adopted** — same key, same field names (`name`, `batch_size`, `count`, `inputs` with `zero_data` / `random_data` / `input_data_file`). `repository/warmup.py` materialises the tensors; `ModelBackend.warmup` runs them instead of the zero-filled count when they are declared. A declared sample that fails now stops the instance becoming ready, because a model that believes it is warm and is not gives a first p99 nobody can interpret. |
| **Explicit model control** | load/unload API | **Adopted** — `model_control: explicit` plus `POST /v2/repository/{index,models/{n}/load,models/{n}/unload}`. Two differences, both deliberate: a load of an already-loaded model is **refused** rather than treated as a reload (a reload must stop the running copy first, so a half-failed one would take a working model down), and unloading a model a loaded ensemble composes is refused rather than discovered inside the DAG on the next frame. |
| **Ensemble scheduling as a first-class scheduler** | `core/src/ensemble_scheduler` | **Adopted** — `server/ensemble.py` no longer walks the step list on a pool thread that blocks on `future.result()`. An execution is state advanced by the completion callback of whichever step just finished, so a thread is held while a step is *dispatched* and never while one is waited for, and steps whose producers have all finished run concurrently. The pool now bounds DAG bookkeeping; `max_pending` bounds frames in flight. Validation, the `(camera_id, frame_id)` tag on every path including failure, the typed errors and the zero-row output of a skipped branch are unchanged — a skipped branch is still distinguishable from a failed one, and a tensor no surviving step will produce resolves as absent instead of parking its readers forever. |

Triton's third model-control mode, `poll`, is **not** adopted and will not be: re-scanning the
repository on a timer means a half-written config can be loaded, and the failure surfaces
minutes later with nothing pointing at the edit that caused it. An explicit `load` re-scans
the repository *at the moment the operator asks*, which gets the "repository grew" case
without the failure mode.

#### What the adopted keys look like

Per **model**, in `model_repository/<name>/config.yaml` — because the answer differs per
model, which is the split in CONVENTIONS 2.6:

```yaml
rate_limiter:
  kind: concurrency            # a name in scheduling/limits/RATE_LIMITERS; `off` by default
  max_concurrent_executions: 4 # how many of *this* model's instances may compute at once

# Only when the derivation is wrong for this engine. Left out — the normal case — the
# capture set follows `max_batch_size` and `preferred_batch_sizes` above.
parameters:
  graph_spec: [1, 4, 8]

model_warmup:
  - name: "batch 8, real frame"
    batch_size: 8
    count: 2
    inputs:
      images: {input_data_file: warmup_frame.bin}   # relative to the version directory
  - name: "batch 1, zeros"
    batch_size: 1
    inputs: {images: {zero_data: true}}
```

Per **deployment**, in settings (`SHIPINFER_*` env or a settings file):

```bash
SHIPINFER_MODEL_CONTROL=explicit          # /v2/repository/... is honoured; start-up loads
SHIPINFER_STARTUP_MODELS='[]'             #   only what startup_models names
SHIPINFER_OBSERVABILITY__TRACE_SINK=jsonlines
SHIPINFER_OBSERVABILITY__TRACE_SINK_OPTIONS='{"path": "/var/log/shipinfer/traces.jsonl", "rate": 100}'
```

One command answers "what has this model done":

```bash
curl -s localhost:8000/v2/models/person_embedder/stats | jq .model_stats[0]
```

The server's *own* view — `Model.stats()` / `EnsembleModel.stats()`, which Triton has no
field for — answers the other half, "what did the server decide":

```jsonc
"cuda_graphs": {"enabled": false, "batch_sizes": [1, 4, 8], "source": "derived",
                "reason": "derived from the batching window (max_batch_size=8, preferred=[4])"}
"steps": [{"model": "ship_segmenter", "condition": "ship_crops", "depends_on": ["ship_crops"]}]
"peak_parallel_steps": 3
```

Both are reported whether or not the feature is switched on: `cuda_graphs.enabled` is
`false` by default (ADR-013), and the capture set that *would* be used is still the thing an
operator needs when the replay counter reads zero.

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
4. **The rate limiter is a single resource, not Triton's resource model.** It bounds
   concurrent executions per model and nothing else. That is the bound this pipeline needed,
   and the registry means a named-resource limiter is a new file rather than a rewrite — but
   if a model ever needs to be bounded on something other than "an execution" (host RAM
   during preprocessing is the plausible one), this will need extending rather than tuning.
5. **An ensemble's steps now run concurrently, and only `max_pending` bounds it.** The
   old one-step-at-a-time walk was an accidental bound on how much work one DAG could put
   on its step models at once; scheduling on the dependency relation removes it. That is
   the point — but it means a wide ensemble can now saturate a step model that used to be
   fed one request at a time, and the bound that remains is per ensemble rather than per
   step. The per-model rate limiter is the tool for the step end of that, and whether the
   default `max_workers * 4` is the right capacity is a measurement nobody has taken.
6. **Per-model statistics are cumulative counters, not a distribution.** `/v2/statistics`
   and `/v2/models/{m}/stats` are diffed between scrapes to get a rate, exactly as Triton's
   are; the distribution still only exists in the Prometheus histograms. Two instruments for
   one quantity is a cost, and it is Triton's cost too.

---

## What I could not verify

- Triton's *measured* CUDA graph speed-up. The comment says graphs help most at small batch
  sizes; I did not benchmark it and neither did the repository, in anything I found.
- Whether `allow_inexact_match_` is on by default in any shipped configuration. The code
  initialises it to `false` and sets it from the graph specs; I did not trace every path.
- The rate limiter's exact policy. I read its existence, not its algorithm.
