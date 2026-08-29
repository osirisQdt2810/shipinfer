# ShipInfer — System Architecture

> The binding architecture, agreed with the operator on 27 Aug 2026 (decision trail
> V129–V140 in `docs/qa/user.md`; hardware evidence in the appendix). Everything in this
> document overrides older descriptions where they disagree. The target workload is fixed
> by `references/bitbucket-subfaceid/docs/new-system-architecture.md`: **50 cameras ×
> 20 fps = 1000 frames/s, 10–20 people per frame (~15 000 crops/s), 16 GPUs, 24/7.** The
> box is GPU-rich, so the design optimises the two things that are actually scarce:
> **balance under skew** and **end-to-end latency** — never raw throughput.

---

## 1. The three concepts

The whole system is built from exactly three ideas. Everything else is plumbing for them.

**Element** — one processing step with typed input and output: `decode`, `detect`,
`segment`, `embed`, `recognize`, `track`, `mtmc`, `output`. Every element is an
abstraction point with interchangeable implementations behind a registry, in the same
shape the ingest sources already use (`@SOURCES.register`):

| Element | Implementations (each a class, each a registry entry) |
|---|---|
| decode | `gstreamer-gpu` (NV12 stays in VRAM — **the default**), `opencv`, `pyav`, `replay` |
| detect / segment / embed / recognize | `pool` (submit to the engine — the default), `nvinfer` (TensorRT inside a DeepStream graph), `nvinferserver` (Triton) |
| track | `shipvision` (bytetrack / botsort, CPU, stateful per camera), `nvtracker` (in-graph) |
| mtmc | `shipvision` (global, shardable by zone) |
| output | `kafka`, `jsonlines`, `null` |

**Topology** — the *declarative chain* of elements: which steps run, in what order, with
which branch conditions. It is data (a YAML file), not code:

```yaml
# topology/ship_person.yaml — the production chain
elements:
  decode:       {impl: gstreamer-gpu}               # NV12 → VRAM, the default
  detect:       {impl: pool, model: ship_detector}
  segment:      {impl: pool, model: ship_segmenter}
  embed_ship:   {impl: pool, model: ship_embedder,   after: segment, params: {classes: [ship]}}
  embed_person: {impl: pool, model: person_embedder, after: detect,  params: {classes: [person]}}
  recognize:    {impl: shipvision, after: embed_ship, params: {classes: [ship]}}   # gallery query
  track:        {impl: shipvision, per: camera, after: [recognize, embed_person]}  # stateful — home shard
  mtmc:         {impl: shipvision, scope: global}
  output:       {impl: kafka}
```

Three rules make this file unambiguous: an element's predecessor is the **previously declared
slot** unless `after:` says otherwise, every element must reach `output`, and a `when:`
guards **only the element it is written on** — a rejected item is skipped past that element
and handed to its successors, so a class-specific *branch* carries its condition on every
element in it. The two `after:` lines on `embed_person` and `track` are the fork and the
rejoin — without them the person branch would end in `embed_person` and the loader refuses
the chain (`ChainStructureError`).

**`when:` guards a frame; `classes:` selects rows.** These are two different questions and
the file says which is which. `when:` is evaluated **once per frame** and its answer is
skip-or-run for that whole element: a rejected frame is handed to the element's successors
untouched. `params: {classes: [...]}` is evaluated **per detection row** and decides which of
the frame's objects this element pays for — which is what a crop element like `embed_ship`
actually wants, because a frame containing one ship and fourteen people is neither "a ship
frame" nor "a person frame". Writing `when: class == ship` on such an element asks a
per-frame question about a per-row fact, and the loader refuses it at start-up
(`ChainStructureError`, naming the `params: {classes: [ship]}` to write instead): the
condition would skip the *entire* element for any frame the expression rejected, taking the
ships with it. `segment` carries no guard above, and that is the honest state: `when:` is the
right tool for a frame-level short circuit -- skipping the ship branch on a frame with no ships
-- but the chain must guard on a fact something files, and no element files one today. The
runner files `meta["fps"]` (`runners/frames.py:144`) and that is the only frame-level key a
`when:` can read right now. `topology/ship_person_cpu.yaml:33-35` says the same thing: a
frame-level guard lands here when there is a bench number behind it, not before.
See ADR-017 (amended 2026-08-28).

**`recognize` is a gallery query, not a pool submission.** Identity is `impl: shipvision`: the
element embeds each selected row and asks a gallery who it is, filing one `(id, score)` per
detection row. `impl: pool` files the model's response verbatim under that same key, which
`output` reads per row, so the loader refuses the pairing at start-up rather than failing every
frame (`ChainStructureError`) -- the chain above is the runnable spelling.

`topology/ship_person.yaml` in the repository is this file. It still carries the older
`when: class == …` spelling on its crop slots and is not loadable until phase D registers a
real `decode`; `topology/ship_person_cpu.yaml` is the runnable sibling and is written the way
this snippet is.

**Runner** — *how* a topology executes. Three runners, one chain definition:

| Runner | Execution | Use |
|---|---|---|
| `inprocess` | whole chain in one process, elements as thread stages | dev, tests, few cameras |
| `fleet` | N shard processes, one GPU each, every shard runs the chain for its cameras | production default |
| `deepstream` | **compiles** the chain into a GStreamer graph (`decode→nvurisrcbin`+`nvstreammux`, `detect→nvinfer`/`nvinferserver`, `track→nvtracker`, …); metadata leaves through a pad probe as the same events | all-NVIDIA path per shard |

The word "topology" always means the chain. Process/GPU placement is the runner's and the
launcher's business. (This resolves the naming collision the previous code had, where
"topology" named the placement — V129/V132.)

---

## 2. Process model and the control plane (gRPC — no argv commands)

```
                 NODE
┌──────────────────────────────────────────────────────────────────┐
│ [P0] API SERVER + LAUNCHER                (1 process, CPU-only)  │
│   HTTP: POST/DELETE /streams   ← cameras and videos enter HERE   │
│         GET /health, /cameras                                    │
│         POST /v2/models/{m}/infer  ← the tensor side-door (kept) │
│   spawns shard processes, supervises, respawns on death          │
│   gRPC CLIENT to every shard                                     │
│                                                                  │
│ [P1..Pn] SHARDS                 (1 process = 1 GPU = M cameras)  │
│   gRPC SERVER: AddCamera, RemoveCamera, Health, Stats,           │
│                UpdateTopology, Drain, Stop                       │
│   runs the topology for its cameras (anatomy in §5)              │
│                                                                  │
│ [Pm] MTMC                            (1 process, CPU, global)    │
│   consumes per-camera events, emits global ids                   │
└──────────────────────────────────────────────────────────────────┘
```

**Camera entry, both modes (V131):**
- *Offline*: `shipinfer run --topology ship_person.yaml --inputs a.mp4 b.mp4 …` — the M
  inputs are sharded evenly at start.
- *API*: the system is up; `POST /streams {"url": "rtsp://…"}` → the launcher picks the
  shard with the fewest cameras (round-robin over the least-loaded) and calls its
  `AddCamera` RPC. Removal is the mirror image. Rebalancing a whole camera between shards
  is `RemoveCamera` + `AddCamera` — the coarse-grained lever (§4 has the fine-grained one).

**The control plane is gRPC (V140).** A process must still be *spawned* (there is no RPC
into a process that does not exist), but the child receives nothing in argv beyond its
identity (`--shard-id N --control-port P`); **everything else — camera set, GPU binding,
topology, config — arrives as RPCs after the child reports ready.** The old mechanism,
where a "topology" object rendered an argv command line and the parent's only interface to
the child was that string plus environment variables, **is deleted** (V140: *"xóa luôn
cách dùng gọi command giữa 2 tiến trình"*). What this buys, concretely:

- add/remove/drain a camera on a live shard without restarting it;
- health and per-camera stats as typed responses instead of scraped logs;
- config changes without process churn;
- the launcher knows *ready* vs *running* vs *draining* instead of inferring from exit codes.

vLLM's engine-core split is the pattern reference: processes talk RPC; nothing meaningful
rides argv.

**Which runner, and its knobs, are settings — `SHIPINFER_RUNNER__*`.** `runner.runner` picks
the implementation by registered name (`fleet` by default, `inprocess` on a laptop; `shipinfer
runners` lists them and `shipinfer run --runner` overrides it), `runner.shards` is how many
shard processes (default: one per visible GPU, ADR-006; `--shards`), `runner.drain_s` is what
a shard gets to finish before SIGKILL (`--drain-s`), and `runner.service.*` configures the
spill tier (§4). Spelled in the environment they are `SHIPINFER_RUNNER__RUNNER`,
`SHIPINFER_RUNNER__SHARDS`, `SHIPINFER_RUNNER__DRAIN_S`, `SHIPINFER_RUNNER__SERVICE__…`. The
section used to be `topology`, and the old `SHIPINFER_TOPOLOGY__*` spelling now **fails
loudly** rather than being ignored: the settings tree is `extra="forbid"`, so a stale export
refuses the process at start-up with the offending key named instead of silently leaving the
default in place — which for `SHIPINFER_TOPOLOGY__SHARDS` would have been a fleet quietly
running the wrong number of processes.

**The RPC surface wraps invariants that already exist — do not rediscover them.** The C++
ingest manager hardened exactly this lifecycle across #33–#41, and the RPCs are thin skins
over it: `Stop` wraps the fleet-deadline stop that charges ONE timeout to the whole camera
set and *returns the abandonment count* (a lifetime signal — the caller must not unwind
buffers a detached thread still references); `AddCamera` wraps the insert→start→re-check
sequence that refuses, typed, a camera the fleet forgot mid-add instead of leaving it
running untracked; `Health` wraps the per-camera snapshot that is safe against concurrent
removal. A re-implementation that reuses these keeps four review rounds of concurrency
fixes; one that rewrites them buys the same bugs back.

---

## 3. The DataPool — one shared-data abstraction, VRAM-first (V137/V138)

The single most important structure in the system. Every image, crop, tensor and result
lives in a **DataPool buffer**, and buffers are shared between shards **without ever
staging payloads through ordinary RAM**.

**This reverses two recorded decisions, by name.** ADR-002 said "nothing in this codebase
performs a cross-device memory access" and sent every cross-GPU payload through host
memory; ADR-015 built the spill transport on pinned rings and rejected CUDA IPC on two
grounds — a CUDA context per peer GPU per shard (`G²` contexts box-wide), and a handle
lifecycle nobody had specified. **ADR-016** records the reversal, supersedes ADR-015 for the
payload transport (its rings survive as the control channel), amends ADR-002's payload
clauses (its threading core — one worker, one context of its own, one GPU — stands), and
answers both objections; §3.4 below is the short form.

### 3.1 Slabs and tickets

- Each shard pre-allocates a few **large VRAM slabs** on its GPU (e.g. 2 × 512 MB) plus
  one pinned-RAM slab (the fallback location). Buffers are carved from slabs.
- **At mesh join, once per shard pair:** slab handles are exchanged —
  `cudaIpcGetMemHandle` → 64-byte handle → peer `cudaIpcOpenMemHandle`s it **once** and
  keeps the mapping forever. From then on the peer sees the slab like its own memory.
- **Sharing a buffer = sending a ticket**: `(slab id, offset, size, format, tag)` — a few
  dozen bytes — over the control ring. The payload does not move, or moves exactly once
  VRAM→VRAM. This is precisely Triton's CUDA-shared-memory extension
  (`CudaSharedMemoryRegister` + region/offset/size per request) applied internally.

- **A carve that is the I/O of a captured CUDA graph is pinned for the graph's life** and
  never recycled while the graph exists (ADR-008 applies to slab carves exactly as to any
  captured buffer); recycling happens only through the slab's generation (§3.4).

The pre-existing shared-memory rings (#26) are **kept but demoted**: they are the control
channel that carries tickets (and the payload path only in RAM fallback mode). They no
longer carry image bytes between GPUs (V137: the RAM round-trip is rejected).

### 3.2 The per-pair route table (the 1000× trap)

`cudaDeviceCanAccessPeer` answers *capable*, not *fast*. Measured on the dev box
(appendix): NVLink pairs move a 12 MB frame in **261 µs**; a PXB pair (across a PCIe
bridge) takes **98.6 ms** for the same copy — three orders of magnitude, reproduced on
three pairs, **0-3, 1-3 and 2-4**, none of which lies inside the dev trio 3-4-5 — while the
same pair staged through pinned memory takes 996 µs. Therefore:

- **At every pair handshake, the DataPool times the link itself** (one 12 MB + one 128 KB
  copy, a few ms, once) and records the route: `direct` (NVLink/PIX) or `staged`
  (everything else). NCCL does exactly this (its `NCCL_P2P_LEVEL` distances — NVL, PIX,
  PXB, PHB, SYS — exist because of this same trap).
- Production nodes are expected to be all-NVLink (V139), where every route resolves to
  `direct`; the probe stays because the dev box demonstrably has poison pairs and because
  hardware assumptions change silently.

### 3.3 One API, two locations

| | VRAM mode (default) | RAM mode (fallback) |
|---|---|---|
| slab | `cudaMalloc`, carved | pinned host memory (#31 machinery) |
| handshake | CUDA IPC handle | shm segment |
| share a buffer | ticket | ticket — identical |
| move payload | 0 copies (direct read) or 1 `cudaMemcpyPeer` | 1 memcpy |
| used for | frames, crops, tensors, vectors | no-NVDEC hosts, CI, tests |

**Torch is normative for the implementation (ADR-003)**: slabs are torch allocations,
handles travel through `torch.multiprocessing`'s CUDA-IPC tensor sharing, peer copies are
torch copies. The `cudaIpc*` / `cudaMemcpyPeer` / `cudaDeviceCanAccessPeer` names in this
document state *semantics*, not call sites; where a raw call is unavoidable in `csrc/`, it
goes through `core/platform.h`'s `gpu*` aliases (ADR-014) so the ROCm build stays whole.

### 3.4 What it costs, and who may hold a handle (ADR-016)

Opening a peer's slab needs a CUDA context on that peer's device in *this* process. That
is the cost ADR-015 refused to pay, and it is real; the answer is to **bound it, measure
it, and budget it** rather than to avoid it:

- **K-neighbourhood.** A shard opens the slabs of at most **K peers** (default K = 3),
  chosen by the route table — NVLink partners first. Everyone else is reached over the
  pinned-staged path (the ADR-015 transport, unchanged). Reach is fleet-wide; the VRAM
  transport is tiered. A shard therefore holds **K + 1 contexts, not G**; box-wide the
  context line is `G × (K + 1) × C_ctx` rather than `G² × C_ctx`.
- **C_ctx is a measured input**, not folklore: **208 MiB** per foreign context, and it
  lands on the **owner's** device, not the opener's (A5000, CUDA 12.6, torch 2.7.1, measured
  2026-08-27 in-container — `benchmarks/link/ipc_context_cost.py`, Appendix A). A shard's
  own context is 243 MiB. So a device is charged once by every peer that opens *its* slabs:
  with symmetric K-neighbourhoods that is `K × 208 MiB` — **≈ 0.6 GB at K = 3**, against
  the `(G−1) × 208 MiB ≈ 3.1 GB` an unbounded 16-GPU mesh would take (ADR-015 feared
  ~4.8 GB at an assumed 300 MiB). The 208 MiB is the cost of the opener having *any*
  context on the owner's device (mapping + context), measured at K = 1 on one pair;
  linearity in K is assumed and **must be probed at K = 2, 3 before the start-up refusal
  is enforced** (phase D).
- **The budget is enforced at start-up.** Each shard writes its VRAM budget into its
  Health report — `slabs + own_ctx + K × C_ctx + engines ≤ device memory − reserve`, the
  `K × C_ctx` term being what its K openers will cost it — and
  refuses to open a camera if the inequality fails; a runtime OOM is never how this is
  discovered.
- **Handle lifecycle.** Slabs are opened at mesh join and **closed only at drain**, after
  the shard's engine has quiesced (no kernel in flight on a mapping being torn down).
  Every ticket carries the slab's **generation**; a restarted shard issues a new one, and a
  ticket against a stale generation fails typed (`StaleTicketError`, tagged) rather than
  being dereferenced. Peer death arrives from the launcher over gRPC (§2) and invalidates
  that peer's mappings; tickets in flight on them fail with `PeerLostError` as in ADR-015.

---

## 4. Two-tier work sharing (V138)

The three imbalance-prone stages are **embedding, detect, segment** — but they skew for
two different reasons, so there are two spill granularities with independent thresholds.
Both are decisions taken *at enqueue time, per item, with no central coordinator*: the
producer checks its local queue depth and either keeps the work or tickets it away.

```
TIER 1 — FRAME tickets (big items, few, 261 µs NVLink / ~1 ms staged per move)
  fixes: camera/fps skew and deep detect queues
  [decode → NV12 in slab] → {local detect queue > frame_threshold?}
        no  → detect locally
        yes → frame ticket to the least-loaded peer → detect runs THERE
              LOCALITY FOLLOWS THE DATA: that frame's crops are cut and
              embedded on the peer too — never bounced back mid-chain.

TIER 2 — CROP tickets (small items, many, ~30 µs per move on every pair)
  fixes: crowding (one frame → 30 embed jobs, ship crops → segment jobs)
  [crops from detect] → {local embed/segment queue > crop_threshold?}
        no  → run locally
        yes → crop tickets to peers → vectors/masks ticket back

INVARIANT: results always return to the camera's HOME shard for reassembly
and tracking (track is stateful per camera and never migrates mid-stream).
```

`frame_threshold` is conservative (moving a frame is a big decision), `crop_threshold` is
liberal (crops are nearly free). Whole-camera reassignment (§2) remains the coarse lever
above both. Measured precedent for tier 2: under fleet-only, a crowded camera put
1572 crops/s on one GPU (others idle); with sharing, 847/345/382 — every rung sustained.

---

## 5. Anatomy of a shard

One shard process, GPU *g*, M cameras. Thread counts for a 4-camera shard in parentheses.

```
① DECODE — one actor thread per camera (4)
   holds the RTSP session, reconnects with jittered exponential backoff,
   default impl gstreamer-gpu: NV12 lands in the DataPool slab, VRAM, no CPU hop.
   Every frame is stamped (camera_id, frame_id, two clocks) — the tag that
   rides to the end and survives every error path.

② FAIR LANES — one bounded lane per camera (cap ~8 frames)
   Two drop doors, both per-camera, both counted, neither cross-camera:
   • the SINK door (ingest/sink.py, csrc sink.h): a frame that finds no
     room is refused with QueueFullError and the camera's own actor drops
     THAT frame — the newest — and continues;
   • the LANE door (scheduling/queues/fair.py DROP_OLDEST): when room must
     be made, the victim is the OLDEST item of the LONGEST lane in the
     lowest-priority band, i.e. the greediest camera pays, never a quiet one.
   Early, whole-frame drop beats a half-processed pipeline; the old system's
   shared evict-oldest buffer — which evicted regardless of camera — is the
   failure both doors replace.
   There is NO scheduler thread: the fairness is a round-robin cursor
   inside the queue's take(), executed by whichever worker asks next.

③ PIPELINE WORKERS (configurable, e.g. 32)
   each worker walks ONE frame through the chain: preprocess → submit to
   the pool → wait (sleeping) → branch on class → crop batch → submit
   crops → wait → hand results to reassembly. Many workers = frames
   overlap in flight; the workers are cheap because they sleep at every
   pool wait.

④ LOCAL ENGINE SLICE — the per-model instances ON THIS GPU (7)
   e.g. detector×2, segmenter×2, embedders×1 each, recognizer×1.
   Instance loop: collect a batch (wait ≤ max_delay e.g. 5 ms or until the
   preferred size), H2D via pinned ping-pong staging, one TensorRT run,
   D2H, scatter results to waiting workers by tag. Two instances of a hot
   model overlap copy-of-next with compute-of-current.
   Admission check at every enqueue = the spill decision of §4.

⑤ REASSEMBLY — one collector thread
   waits for all parts of (camera, frame): boxes + K vectors + masks.
   Timeout ⇒ emit anyway with missing_stages naming what is absent —
   never pretend completeness.

⑥ TRACK — one stateful tracker per camera (CPU, shipvision MOT)
   consumes the assembled frame; never leaves the home shard.

⑦ OUTPUT — events (camera, frame, boxes, vectors, track ids) → Kafka.
```

**What crosses which boundary** (the rule the whole design hangs on):

| Data | Size | Crosses process/GPU? | Path |
|---|---|---|---|
| raw frame | MBs | only as a **tier-1 ticket** under skew | VRAM→VRAM per route table |
| crop | KBs | **tier-2 ticket** under load | VRAM→VRAM |
| vector / box / mask meta | bytes–KB | ticket return path | control ring |
| event | ~KB | leaves the system | Kafka |

---

## 6. The engine and the tensor side-door

The **engine** is the model pool: repository-loaded models, per-model queues, dynamic
batching, instances pinned one-GPU-for-life, and the placement policies
(`round_robin`, `jsq`, `power_of_two`, `locality_spillover`) — the "mini-Triton" of the
reference design, already built and measured. Elements of kind `pool` are thin clients of
it. It keeps one **side-door**: the KServe-v2 HTTP endpoint (`/v2/models/{m}/infer`) for
callers who bring their own tensors (V132 kept it deliberately).

---

## 7. Threading and the GIL contract (V70, reaffirmed by V142)

The old contract said worker threads spend their time inside TensorRT or CUDA memcpy,
"both of which release the GIL". Measurement (C1b) proved this false for our own kernel
bindings: shipvision's pybind layer held the GIL across H2D, kernel, the blocking stream
sync and D2H — a **GIL convoy** that serialized every worker (52 ms/frame apparent crop
cost that is 8.5 ms real; true chain cost 51.6 ms/frame; extra workers bought queueing,
not throughput).

**The law (V70, reaffirmed by V142 — operator's decision):** shipvision contains **no GIL
code, ever**. It delivers algorithms; the most it may hold is a `std::mutex` around a
stateful `tracker.track()`, which it already does. A proposal to release the GIL inside
shipvision (V140 (i)) was revoked the same day it was made, before any code landed; its
architecture test — "nothing in `csrc/` names `gil_scoped_*`" — stands. **Slowness is
accepted over GIL code in the library.**

The convoy is therefore a *server-side* problem, and the server has three levers, none of
them in shipvision:

1. **Fewer, fatter native calls per frame** — an element submits one batch, not one call
   per crop; the fair-lane worker (§5③) already walks a whole frame at a time.
2. **The hot plane in C++ under shipinfer's own `csrc/` (V34, ADR-014)** — where the call
   into the native kernels is made from C++, no Python frame holds the lock across it. The
   ingest plane already lives there (#33); the decode→preprocess→submit path of a shard is
   the next candidate when the measurement says the convoy binds. Whether `csrc/` links
   shipvision's C++ core directly or crosses its Python surface is a phase-D decision, made
   on a measurement.
3. **Per-worker streams, passed from the server** — `NativeImageOps(stream=…)` already
   accepts a raw stream handle and forwards it to every op; shipinfer today passes nothing,
   so every worker shares the legacy stream. Passing the worker's `Stream.handle` is one
   line in `runtime/ops/__init__.py` and is not GIL code.

VRAM-first parallelism (§3–§4) rests on 2 and 3; it does not need the library to change.

---

## 8. Formats and caps

Every element declares input/output caps as `<format>@<location>`: `nv12@gpu` (default
end-to-end), `bgr@cpu` (fallback), `tensor@gpu`, and `meta@cpu` for the result plane that
`track`, `mtmc` and `output` consume. `location` is closed to `gpu | cpu`; `format` is an
open lowercase token. The chain loader validates adjacent caps at load time: an edge is
compatible when producer and consumer share a format and a location (a `*` wildcard matches
either half but **never bridges `gpu` and `cpu`**), and the first compatible pair in
declaration order wins — so declaring `nv12@gpu` first is how an element says "VRAM
preferred". A chain that would silently download to CPU refuses to load with a
`CapsMismatchError` that names both sides; converts are never inserted implicitly — they
are spelled as elements (A1 decision). The kernels for the default path
(`nv12_letterbox`, device-resident crop, `letterbox_into`) already exist in shipvision.

---

## 9. Package layout

The tree is the architecture; a reader should find every §-heading of this document as a
directory:

```
src/shipinfer/
├── api/            # §2  HTTP: /streams, /health, KServe side-door
├── launch/         # §2  spawn + supervise shards; gRPC client + proto/ (.proto + stubs); placement
├── topology/       # §1  Element ABC + caps; chain loader (YAML); element registries
│   └── elements/   #     decode/ detect/ segment/ embed/ recognize/ track/ mtmc/ output/
├── runners/        # §1  inprocess.py · fleet.py · service.py (the shard's servicer) ·
│                   #     deepstream/ (the chain→graph compiler, phase E)
├── cli/shard.py    # §2  the shard process: two flags in, everything else over gRPC
├── engine/         # §6  model pool: instances, scheduler, batching, policies (was server/)
│   ├── ensemble.py #     the KServe-visible model DAG — kept, and NOT the frame chain (§6)
│   ├── health.py   #     the pool's own report; statistics.py beside it
│   └── spill/      #     ADR-015's rings: remote instance, wire, mesh (§4's transport)
├── datapool/       # §3  slabs, tickets, IPC handshake, per-pair probe, route table
├── ingest/         # §5① camera actors + source implementations (used by decode elements)
└── core/           #     types, errors, settings, registry, logging, metrics (unchanged)
```

Three names in that tree are not §-headings and are worth placing explicitly.
**`engine/ensemble.py` stays**, and is not a duplicate of the chain: an ensemble is a
*model* composed of models, addressable over KServe as one name (§6's side-door), while a
topology is a chain of *elements* over frames. They answer different callers and neither
subsumes the other. **`engine/spill/`** is ADR-015's ring transport, demoted by ADR-016 to
the control channel and the RAM fallback, which is why it lives under the engine rather than
beside the DataPool. **`cli/shard.py`** is the child process's entry point: it composes an
engine, a topology and a runner, which is exactly what `launch` and `runners` may not do.

**`pipeline/` is not in the tree because it retires into it.** Its ~30 modules are the
previous generation of everything above, and each one lands somewhere named: `graph/` is
superseded by the chain loader and the runners (phase C); `reassembly/` becomes §5⑤ inside a
shard, under `runners/`; `sinks/{kafka,jsonlines,null}` become `output` element
implementations under `topology/elements/`, over the transports themselves which moved to
`topology/sinks/` — the element assembles the event, the sink carries it, and the split is
what keeps a broker client out of the element registry; `deepstream/` becomes the phase-E chain compiler
under `runners/deepstream/`; `runner.py` is the in-process runner's precedent and is
superseded by it. Until phase C and phase E have landed those, `pipeline/` remains the
working application and `csrc/shipinfer/pipeline/` follows it, module for module.

The `.proto` and its generated stubs live under `launch/proto/`, not `api/`: `api/` will
import `launch` in phase B so `POST /streams` can reach the shards, and putting the stubs in
`api/` would make `launch` import `api` for them — a cycle. The servicer that answers those
RPCs is `runners/service.py`, because it holds a runner and a launcher must not.

`server/` is gone (A2, PR-1…PR-6): its pool became `engine/`, its KServe surface moved under
`api/`, its topology-as-placement classes dissolved into `launch/` + `runners/`, and the
argv-command mechanism was deleted outright. Two `core/` modules were renamed with the
vocabulary in the same phase: `core/settings/topology.py` → `runner.py`
(`TopologySettings.kind` → `RunnerSettings.runner`, and the section is `settings.runner`), and
`core/errors/topology.py` → `core/errors/launch.py`, which also ends its collision with
`core/errors/chain.py`'s `TopologyError` — the chain's failures, which are a different thing.

**`csrc/` is the second plane and stays a mirror (ADR-014).** Nothing above changes the
two-planes rule: every Python package that has a native counterpart keeps it at the same
path under `csrc/shipinfer/` — today `ingest/` (actors, sink, sources), tomorrow
`datapool/` (slab carve, route probe, peer copy) — and every native component keeps a
Python implementation so the offline tier runs with no build. The renames apply to both
trees in the same PR; a plane is never allowed to drift from the other's layout.

---

## 10. Migration plan (top-down, reusing what is proven)

Nothing below rewrites an algorithm; the work is moving proven parts under the new seams.

| Phase | Delivers | Reuses |
|---|---|---|
| **0** | *dropped (V142)* — no shipvision change; the server passes its worker stream to `NativeImageOps(stream=)` (§7 lever 3) when phase B wires the runners | the existing `stream` parameter |
| **A** | skeleton: `topology/` (Element ABC, caps, YAML loader), `runners/inprocess`, `engine/` + `api/` split out of `server/`, gRPC proto + launch supervisor; argv-command deleted | pool, scheduler, stages, ingest as-is |
| **B** | `/streams` API + round-robin add/remove over gRPC | ingest add/remove_camera (built, both planes) |
| **C** | track + mtmc + recognize elements in-chain | shipvision mot/mtmc |
| **D** | DataPool: slabs, tickets, pair-probe, VRAM decode default (`gstreamer-gpu` → NV12 into pool), two-tier spill | rings (as control channel), pinned staging, nv12 kernels, C-sweep evidence |
| **E** | deepstream runner as chain compiler; `nvinferserver` element impl | #32's graph builder + probe |

Each phase is independently shippable and reviewed; the offline test tier keeps its
no-GPU/no-gst guarantees throughout (element ABCs and the chain loader are pure).

---

## Appendix A — measured hardware facts (dev box, 8× RTX A5000, 27 Aug 2026)

| Link (pair class) | 12 MB frame | 128 KB crop | Route decision |
|---|---|---|---|
| NVLink NV4 — **timed** 0-1, 3-4; 5-6 and 2-7 are the same class per `topo -m`, *not timed* | 261 µs (48 GB/s) | 29 µs | direct |
| SYS (cross-NUMA) — timed 3-5, 4-6 | 753 µs (16.7 GB/s) | 30 µs | staged (driver) |
| PXB direct P2P — timed 0-3, 1-3, 2-4 | **98.6 ms** | **49.3 ms** | **poison — never direct** |
| PXB staged via pinned — same pairs | 996 µs (12.6 GB/s) | 29 µs | staged |
| same device (baseline) | 47 µs (267 GB/s) | 17 µs | — |

The PXB "direct" cells are **a fixed ~49 ms cost per copy, not a bandwidth**: 98.6 ms for
12 MB against 49.3 ms for 128 KB is a 2× time ratio over a 96× size ratio — the signature
of a driver fallback to synchronous chunked staging, not of a slow wire. So a *small*
payload is relatively *worse* on a PXB direct route (1700× the NVLink crop floor), which is
why the crop column is in the table and why no payload size makes PXB direct acceptable.

`cudaDeviceCanAccessPeer` returns true for *every* pair above, including the poison ones.
The PXB rows were reproduced on pairs **0-3, 1-3 and 2-4**; the dev trio (GPUs 3,4,5)
contains no PXB pair *among its own members*, which is why no benchmark ever tripped it;
a 16-GPU deployment would. The rows above are the 2026-08-27 12:38 UTC re-run of the
probe, inside the `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime` container per CLAUDE.md
(GPUs verified idle before and after): script `benchmarks/link/link_probe.py`, runner
`benchmarks/link/run.sh`, raw JSON + `nvidia-smi topo -m` in
`benchmarks/link/results/2026-08-27/link_probe.log` (pairs probed: 0-1, 3-4 NV4; 0-3, 1-3,
2-4 PXB; 3-5, 4-6 SYS; same-device baseline 47 µs / 267 GB/s).

| Context cost (ADR-016 input, `benchmarks/link/ipc_context_cost.py`, same run) | measured |
|---|---|
| one foreign CUDA context — GPU 4's process opens a 64 MiB slab owned by GPU 3 | **+208 MiB on the owner's device (GPU 3)**, **+0 MiB on the opener's (GPU 4)** |
| a process's own context on its own device (child before opening anything) | 243 MiB |

Production nodes are expected all-NVLink (V139); the probe stays regardless.

Other load-bearing numbers: true serial chain cost **51.6 ms/frame** (C1b discriminator);
full-DAG floor **48 img/s per GPU** with segmenter×2 (#47: 96 → 143.8 img/s across 3 GPUs
— the single-instance segmenter was the fleet's binding stage); tier-2 spill measured
1572/0/0 → 847/345/382 crops/s.

## Appendix B — decision record

| Decision | Ref |
|---|---|
| Pause and restate; four architecture questions | V129 |
| track/mtmc in-chain (shardable later); keep tensor side-door; names topology/runner | V132 |
| Cross-GPU VRAM access allowed (perf+accuracy the only criteria); VRAM-first sharing via CUDA-IPC slab handles + tickets; RAM demoted to fallback; decode default = gstreamer→NV12 in VRAM | V137 |
| Two-tier spill (frame + crop); imbalance-prone stages: embedding, detect, segment | V138 |
| Production nodes all-NVLink; probe kept | V139 |
| ~~GIL fix inside shipvision with per-thread streams~~ (revoked by V142); this document; top-down re-implementation; gRPC control plane, argv-command deleted | V140 |
| **No GIL code in shipvision, ever**; it only delivers algorithms; at most a mutex around `tracker.track()`; slowness accepted — V70 stands, V140 (i) revoked | V142 |
| `topology/` package landed: Element ABC, caps, registry per kind, chain loader; default predecessor = declaration order, `after` overrides; no implicit converts | A1 |
| ADR-016 (this PR): supersedes ADR-015's payload transport, amends ADR-002's payload clauses, K-neighbourhood context budget, handle lifecycle | `.claude/DECISIONS.md` |
