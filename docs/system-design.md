# ShipInfer — system design, top to bottom

**Read this file first, then read the code in the order below.** Every heading names the file
you open next, so you never have to guess which layer you are in.

`docs/arch.md` is the *contract* — what the system must do, and why each decision was taken.
This file is the *map* — where each of those decisions lives in the tree, what state it is in,
and what has not been built yet. When the two disagree, `arch.md` wins and this file is stale;
say so.

---

## 0. The one-paragraph version

Fifty cameras at 20 fps (1000 frames/s) are decoded, detected, segmented, embedded, tracked and
handed downstream, across 16 GPUs. The box is GPU-rich, so the problems are **load balance**
(cameras are uneven) and **end-to-end latency**, not raw throughput. Everything is arranged
around those two. A *chain* (YAML) says what happens to a frame; a *runner* says where it
happens; the *engine* owns the GPUs and the batching.

## 1. The three concepts, in the order a frame meets them

| # | Concept | Means | Lives in |
|---|---|---|---|
| 1 | **Chain** (topology) | the per-frame graph: decode → detect → segment/embed → track → output. Declared in YAML, validated at load. | `topology/`, `topology/ship_person.yaml` |
| 2 | **Runner** | *how* a chain executes: all in this process, or one shard process per GPU. | `runners/` |
| 3 | **Engine** (model pool) | who owns the GPUs: model instances, batching, placement. A chain element asks it for inference. | `engine/` |

Keep these apart while reading. Most confusion comes from expecting the chain to know about
GPUs (it does not) or the engine to know about cameras (it does not).

---

## 2. The reading order

### Step 1 — the entry point: `src/shipinfer/cli/commands/run.py`

`shipinfer run` is the deployment command. It is a **composition root**: it reads settings,
loads a chain, picks a runner by name, places cameras, and starts it. Nothing per-frame happens
here.

> **Known problem:** this file is 694 lines and does not read like a composition root. It is
> item `V149-cli` below.

Also here: `serve.py` (HTTP facade), `bench.py` (the in-process scheduler demonstration),
`repo.py` (inspect the model repository), `doctor.py` (devices, CUDA provider, native
extension status), `registries.py` (which provides the `backends`, `policies`, `queues` and
`runners` listing commands — each prints one of the tables in §4), `deepstream.py`.

`python -m shipinfer` → `__main__.py` → the same CLI.

### Step 2 — what a frame is put through: `src/shipinfer/topology/`

- `base.py` — the `Element` ABC, `ChainItem` (a frame plus its meta), `Caps` (pixel format and
  location, e.g. `bgr@cpu`), `ElementKind`, `RowIndexed` (per-detection-row scatter-back).
  **This is the vocabulary the whole data plane speaks.**
- `chain.py` — `Topology`: loads the YAML, validates the DAG, negotiates caps between elements,
  and refuses a bad chain **at load time** rather than per frame. `Topology.describe()` prints
  the resolved chain; run it before reading further.
- `elements/` — one implementation per file. See the table in §4.

A chain is a DAG of *slots*; each slot names a `kind` (what stage it is) and an `impl` (which
implementation). `when:` guards a whole **frame**; `params: {classes: [...]}` selects
**detection rows**. Those two are different mechanisms and the distinction matters (ADR-017).

### Step 3 — how it executes: `src/shipinfer/runners/`

- `base.py` — the `Runner` contract. Read this before either implementation.
- `inprocess.py` — `InprocessRunner`: the whole chain on a thread pool in this process. **The
  per-frame loop is `_walk`**, and the seam is stated in its own docstring: above it,
  everything deals in work items and queues; below it, chain items and elements.
- `fleet.py` — `FleetRunner`: one shard process per GPU, driven over gRPC. **The default for
  deployment.**
- `service.py` — the shard's own half: it holds a runner and answers the control RPCs.

> **Known problem:** `inprocess.py` is 2 121 lines with eight responsibilities in one class, and
> `_walk` sits at line 1487. This is item `V149-runners` below, and it is the single biggest
> reason the code does not read top-down today.

### Step 4 — spawning and supervising shards: `src/shipinfer/launch/`

- `control.py` — the control vocabulary, with no transport in it.
- `proto/` — `shard.proto` plus the committed generated stubs.
- `supervisor.py` — `Fleet`: one process per shard, all-or-nothing start, one drain.
- `client.py` — `ShardClient`: Ready / UpdateTopology / AddCamera / Health / Stop.
- `signals.py` — Ctrl-C / SIGTERM → the fleet. **Never imports torch**, because it sets
  `CUDA_VISIBLE_DEVICES` before the child's interpreter starts.
- `cli/shard.py` — the child process: `--shard-id N --control-port P` and nothing else. A shard
  is *sent* its chain; it does not go looking for one.

### Step 5 — frames arriving: `src/shipinfer/ingest/`

`IngestManager` plus one **actor thread per camera**, which holds the RTSP session and
reconnects with jittered exponential backoff. Every frame is stamped `(camera_id, frame_id)` —
the tag that rides to the end and survives every error path. Sources: `sources/gstreamer.py`,
`sources/pyav.py`, `sources/replay.py` (a video file or a directory of frames).

### Step 6 — fairness and admission: `src/shipinfer/scheduling/`

**Pure Python, no hardware — the part this project exists to own, and the part most worth
having tests for.**

- `queues/fair.py` — `FairPriorityQueue`: one bounded lane per camera. When room must be made,
  the victim is the **oldest item of the longest lane in the lowest-priority band** — the
  greediest camera pays, never a quiet one (`fair.py:163-172`). This is the fix for the original
  failure: the old
  system funnelled every camera into one shared 1000-slot buffer that evicted the oldest entry
  regardless of camera, so a crowded camera silently starved a quiet one.
- `batching/` — assemble N requests into one batch, scatter results back.
- `policies/` — which instance gets the next batch. See §4.

### Step 7 — the GPUs: `src/shipinfer/engine/`

- `pool.py` — `InferenceServer`: the repository, the devices, the memory pool, the loaded
  models. This is what a chain's `pool` element talks to.
- `model.py` — `Model`: instances + dispatcher + batch window + cache.
- `instance.py` — `ModelInstance`: **one backend copy + one queue + one worker thread, pinned to
  one GPU for its whole life.** The unit of parallelism.
- `ensemble.py` — a DAG of models addressed as one model, validated at load.
- `spill/` — the ring tier kept as a control channel (ADR-015/016).

### Step 8 — the accelerator seam: `src/shipinfer/runtime/`

**The only package that knows a GPU exists.** Everything above it works unchanged with no
driver installed — that is ADR-001, and it is what makes the offline test tier possible.

`platform.py` (CUDA vs ROCm vs CPU), `device.py` (thread binding), `stream.py`, `tensor.py`
(core ↔ torch bridge), `graphs/` (CUDA graph capture), `memory/` (staging pool, allocators),
`ops/` (numpy / torch / native fused kernels), `native.py`.

### Step 9 — the execution runtimes: `src/shipinfer/backends/`

One per module, registered. A backend receives an assembled batch and returns one. **It does not
decide what to batch, where to run, or when.** `tensorrt/` is the production path; `onnx.py` and
`torch_backend.py` are the fallback and the parity check.

### Step 10 — the edges

- `api/` — KServe v2 over FastAPI, the engine's side-door. The **only** layer that may import
  fastapi/uvicorn, and lazily.
- `repository/` — the on-disk model repository (Triton's layout, `config.yaml`).
- `core/` — pure: types, request/response, the typed error vocabulary, settings, logging,
  metrics, and `registry.py`. **`core/` imports nothing from the project but `core/`.**

### Step 11 — the C++ plane: `csrc/`

`csrc/` is a **port** of the Python data plane, not a second design (~14 900 lines). Every
per-frame seam exists twice and must be the *same* seam. The control plane — settings, the model
repository, registries, the CLI, HTTP — stays Python and hands C++ a resolved config (ADR-014).

The gate that keeps them honest is `benchmarks/parity/` + `csrc/tests/test_ingest_parity.cpp`:
one scenario, both planes, one committed golden.

---

## 3. The five shared seams — protect these

1. **The registry** (`core/registry.py`) — every pluggable family is a folder of
   one-class-per-file modules plus a `Registry`. Adding an implementation is a new file and a
   decorator, never an edit to a switch statement.
2. **The scheduler** (`scheduling/`) — queues, batching, placement. Pure, testable on a laptop.
3. **The accelerator seam** (`runtime/`).
4. **The backend contract** (`backends/base.py`).
5. **The kernel boundary** (`3rdparty/shipvision` ↔ `runtime/ops/`) — fused kernels only, numpy
   in and device-pointer out, pinned as a submodule (ADR-010).

**Coupling rule, enforced by `scripts/hooks/check_layers.py` and `tests/test_architecture.py`:**
`core`, `scheduling` and `repository` import no torch, no tensorrt, no onnxruntime, no fastapi.
`runtime` may not import `engine`. `import shipinfer` must not pull in a backend.

---

## 4. Every component that exists today

Read out of the live registries, not from memory. `shipinfer runners`, `shipinfer backends`,
`shipinfer policies` and `shipinfer queues` print their own rows on this host.

### Chain elements, by kind

| Kind | Purpose | Implementations |
|---|---|---|
| `decode` | frames in, from a camera or a file | `gstreamer`, `pyav`, `replay` |
| `detect` | boxes per frame | `pool` (asks the engine) |
| `segment` | masks for detected rows | `pool` |
| `embed` | embeddings for detected rows | `pool` |
| `recognize` | ship identity against a gallery | `pool`, `shipvision` |
| `track` | tracklets per camera | `shipvision` |
| `mtmc` | identity across cameras | `shipvision` |
| `output` | hand results downstream | `jsonlines`, `kafka`, `none` |

### The other families

| Family | Registered today | Purpose |
|---|---|---|
| **Runners** | `inprocess`, `fleet` | how a chain executes |
| **Backends** | `tensorrt`, `onnxruntime`, `pytorch` | what runs a batch |
| **Placement policies** | `round_robin`, `join_shortest_queue`, `power_of_two`, `locality_spillover`, `sequence_affinity` | which instance gets the next batch |
| **Ingest sources** | `gstreamer`, `pyav`, `replay` | where frames come from |
| **Image ops** | `numpy`, `torch`, `native` | letterbox / crop / NMS; numpy is the readable reference every fused kernel is asserted against |
| **Allocators** | `torch_device`, `torch_pinned`, `host`, `custom_*` | memory; `torch_*` are the default (ADR-003) |
| **Graph caches** | `torch`, `custom` | CUDA graph capture |
| **Request queues** | `fair` (default), `fifo` | admission order; `fair` is priority-ordered across lanes and camera-fair within one |
| **Eviction policies** | `greediest_camera`, `oldest_frame` | who pays when a lane is full |
| **Batchers** | `stacking` | request batch → tensor batch |
| **Metric exporters** | `prometheus`, `jsonl` | observability |

**There is no `mock` implementation anywhere in `src/`.** That is deliberate (V148): a system
verified only against mocks is not verified. `git grep -ri mock origin/main -- src/` returns
zero, and `tests/system/test_real_chain.py` runs the real chain on real footage instead.

---

## 5. Where the phases stand

`arch.md` §10 defines the phases. Status as of 31 Aug 2026:

| Phase | Delivers | Status |
|---|---|---|
| **0** | *dropped (V142)* | n/a |
| **A** | skeleton: `topology/`, `runners/inprocess`, `engine/`+`api/` split out of `server/`, gRPC proto + launch supervisor | **done** |
| **B** | `/streams` API + round-robin add/remove over gRPC | **done** |
| **C** | track + mtmc + recognize elements in-chain | **done** (C1–C8b merged) |
| **D** | **DataPool**: slabs, tickets, per-pair route table, VRAM decode default (`gstreamer-gpu` → NV12 into the pool), two-tier spill | **not started** |
| **E** | deepstream runner as a chain compiler; `nvinferserver` element impl | **not started** |

The C++ port has its own sequence: **P1** (instance = thread + bounded queue, dispatcher,
placement), **P2** (batch window), **P3** (fair-queue eviction order) and **P4** (ingest) are
**done**; **P5** (resolved config in, same events out) is scoped and unblocked; **P6** (the
parity harness) has **PR-A merged** with PR-B (scheduling-seam parity) and PR-C (csrc runners
re-baseline) outstanding.

---

## 6. What is not built yet, and what each planned component is for

### Phase D — the DataPool (the next big one)

The single shared-data abstraction, VRAM-first. Today a decoded frame lands in host memory and
is copied to the device per stage. Phase D makes the frame land in a **slab** in VRAM at decode
time and travel by **ticket**, so a chain of nine stages does not pay nine copies.

Components it introduces (`arch.md` §3):

- **Slabs and tickets** — the allocation unit and the handle. Who may hold a handle, and for how
  long, is ADR-016.
- **The per-pair route table** — the "1000× trap": the route between a producer and a consumer
  is resolved once per *pair*, not once per frame.
- **One API, two locations** — the same call works whether the payload is in host or device
  memory, which is what lets a chain stay written once.
- **Two-tier work sharing** (`arch.md` §4) — spill between shards when one GPU is hot.
- **NV12 decode into the pool** — `gstreamer-gpu`, the one element impl `topology/ship_person.yaml`
  names but cannot load today.

**Blocked on an operator step:** the container image `shipinfer-gst:jammy` must be rebuilt with
`libgstreamer-plugins-bad1.0-dev` (for gstcuda) before the NV12 path can be built. Ledger item
`PHASE-D-NV12`.

### Phase E — the DeepStream runner

A third runner that **compiles** a chain into a DeepStream pipeline rather than walking it in
Python, plus an `nvinferserver` element implementation. The value is that the same YAML chain
runs on either, so the comparison is like-for-like. Design notes already exist at
`docs/design/topology-deepstream.md`.

**Also gated:** the DeepStream container image has to be pulled first (ledger `T4`).

### The benchmark evidence that is still owed

- **C1** — the ≥5× against the counting-simulation baseline, whole system. Gated on phases C+D
  per `arch.md` §10. Baseline measured at 868.2 img/s.
- **C4** — RTSP in the benchmark harness, re-scoped 28 Aug.

Both are runbook items in `.claude/TASKS.md` (search `C1 ·` and `C4 ·`); the working runbook
itself lives in the session scratchpad, not in the repository, so treat the ledger lines as the
authority.

### Cross-plane divergences found by the parity harness, each owed a decision

- **P6-D1** — `CameraHealth.last_error`: Python prefixes the exception type, C++ does not. Pick
  one spelling.
- **P6-D2** — `consecutive_failures` after a FATAL open: 0 in Python, 1 in C++. Does a failure
  that is never retried count as one?
- **P6-D3** — `stop()` fate stickiness: C++ latches `thread_abandoned_` for ever, Python
  re-reads `thread.is_alive()`.

Each is registered in `benchmarks/parity/known.py`, and **deleting its entry there is part of
the fix** — so the gate starts failing the moment a divergence is resolved but not de-registered.

### Readability work (V149) — why this file exists

The operator could not read main top-down against `arch.md`. Measured, the complaint was
correct on every count:

- **39% of `src/shipinfer` is prose** — 20 572 docstring/comment lines against 23 853 code
  lines, i.e. 0.86 prose lines per line of code. `runners/` 1.77, `topology/` 1.40, `api/` 1.38.
  `docs/` is only ~2 800 lines, so the bloat is *inside the source*.
- **Nine files over 850 lines**, worst `runners/inprocess.py` at 2 121.
- **94 single-use private helpers** under 12 lines.

Agreed plan: all three fixes, **one package per PR** — split the oversized file, cut the prose
in the files touched, delete the superfluous helpers there.

| Item | Scope |
|---|---|
| `V149-runners` | `inprocess.py` (2 121) → runner / frame-walk / camera lifecycle / placement; `fleet.py` (1 060); `service.py` (859) |
| `V149-topology` | `elements/pool.py` (1 697), `chain.py` (1 077), `recognize.py` (1 031), `track.py` (945), `barrier.py` (900), `base.py` (852), `mtmc.py` (847) |
| `V149-engine` | `pool.py` (1 421), `ensemble.py` (828), `spill/remote_instance.py` (747) |
| `V149-cli` | `cli/commands/run.py` (694) → a thin composition root |

Note that V145 (the documentation cap, `scripts/hooks/check_docs.py`) is a *different* and
weaker instrument: its wave 1 deleted 349 lines of prose and moved the violation count only
1031 → 1012. Shortening docstrings is not the fix; fewer and smaller files is.

### Other known gaps

- **`CONTAINER-TIER-15-RED`** — the offline tier inside the container has failures the host does
  not: order- and interpreter-dependent (py3.11 in the container vs py3.10 on the host). CI is
  unaffected, because CI runs on host runners.
- **V145-W2 / W3 / ARM** — trim waves 2 and 3, then arming the doc cap in pre-commit. Must not
  be armed before the waves land, or every commit is refused.

---

## 7. How to run it

```bash
pytest                                        # offline tier: pure logic, no GPU, must stay green
deploy/rootless/test.sh -m gpu                # the device tier, in the container
deploy/rootless/bench.sh --systems shipinfer --seconds 40   # the per-device evidence table
shipinfer run --runner fleet                  # the deployment command
shipinfer runners; shipinfer backends; shipinfer policies; shipinfer queues   # what is registered
shipinfer repo ls                             # the model repository, and how many instances it expands to
```

**Anything that touches an accelerator runs inside a container** — the GPU tiers, every
benchmark, `shipinfer bench|serve`, any engine build. The offline tier is exempt by design
(ADR-001): a plain `pytest` must pass on a machine with no driver, which is what makes the pure
layers verifiable anywhere.

"The offline suite is green" is **not** evidence that the server balances load. A bench run with
a per-device breakdown is.
