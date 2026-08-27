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
  decode:    {impl: gstreamer-gpu}                  # NV12 → VRAM, the default
  detect:    {impl: pool, model: ship_detector}
  segment:   {impl: pool, model: ship_segmenter, when: class == ship}
  embed_ship:   {impl: pool, model: ship_embedder,   after: segment}
  embed_person: {impl: pool, model: person_embedder, when: class == person}
  recognize: {impl: pool, model: ship_recognizer, after: embed_ship}
  track:     {impl: shipvision, per: camera}        # stateful — pinned to the home shard
  mtmc:      {impl: shipvision, scope: global}
  output:    {impl: kafka}
```

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

---

## 3. The DataPool — one shared-data abstraction, VRAM-first (V137/V138)

The single most important structure in the system. Every image, crop, tensor and result
lives in a **DataPool buffer**, and buffers are shared between shards **without ever
staging payloads through ordinary RAM**.

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

The pre-existing shared-memory rings (#26) are **kept but demoted**: they are the control
channel that carries tickets (and the payload path only in RAM fallback mode). They no
longer carry image bytes between GPUs (V137: the RAM round-trip is rejected).

### 3.2 The per-pair route table (the 1000× trap)

`cudaDeviceCanAccessPeer` answers *capable*, not *fast*. Measured on the dev box
(appendix): NVLink pairs move a 12 MB frame in **261 µs**; a PXB pair (across a PCIe
bridge) takes **98.6 ms** for the same copy — three orders of magnitude, reproduced on
three pairs — while the same pair staged through pinned memory takes 996 µs. Therefore:

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

Implementation stands on torch (ADR-003): torch's CUDA caching allocator and
`torch.multiprocessing`'s CUDA-IPC tensor sharing are the base, not hand-rolled handles.

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
   A full lane drops THAT camera's newest frame, counted, at the door
   (early, whole-frame drop beats a half-processed pipeline; the old
   system's shared evict-oldest buffer is the failure this replaces).
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

## 7. Threading and the GIL contract (V70 revised → V140)

The old contract said worker threads spend their time inside TensorRT or CUDA memcpy,
"both of which release the GIL". Measurement (C1b) proved this false for our own kernel
bindings: shipvision's pybind layer held the GIL across H2D, kernel, the blocking stream
sync and D2H — a **GIL convoy** that serialized every worker (52 ms/frame apparent crop
cost that is 8.5 ms real; true chain cost 51.6 ms/frame; extra workers bought queueing,
not throughput).

**The revised law (V140, operator's decision):** shipvision releases the GIL around the
pure-native section of every binding, **and** moves to per-thread CUDA streams *in the
same change* — stream-0 co-tenancy was survivable only because the convoy hid it; release
the lock without splitting streams and the serializer just moves into CUDA. The
architecture test flips from "never touch the GIL" to "release around native, never
acquire". VRAM-first parallelism (§3–§4) is meaningless without this; it lands first.

---

## 8. Formats and caps

Every element declares input/output caps: `nv12@gpu` (default end-to-end), `bgr@cpu`
(fallback), `tensor@gpu`. The chain loader validates adjacent caps at load time and
inserts explicit converts only where declared caps disagree — a chain that would silently
download to CPU refuses to load instead. The kernels for the default path
(`nv12_letterbox`, device-resident crop, `letterbox_into`) already exist in shipvision.

---

## 9. Package layout

The tree is the architecture; a reader should find every §-heading of this document as a
directory:

```
src/shipinfer/
├── api/            # §2  HTTP: /streams, /health, KServe side-door; gRPC service defs (.proto)
├── launch/         # §2  spawn + supervise shards; gRPC clients; round-robin placement
├── topology/       # §1  Element ABC + caps; chain loader (YAML); element registries
│   └── elements/   #     decode/ detect/ segment/ embed/ recognize/ track/ mtmc/ output/
├── runners/        # §1  inprocess.py · fleet.py · deepstream/ (the chain→graph compiler)
├── engine/         # §6  model pool: instances, scheduler, batching, policies (from server/)
├── datapool/       # §3  slabs, tickets, IPC handshake, per-pair probe, route table
├── ingest/         # §5① camera actors + source implementations (used by decode elements)
└── core/           #     types, errors, settings, registry, logging, metrics (unchanged)
```

`server/` disappears: its pool becomes `engine/`, its KServe surface moves under `api/`,
its topology-as-placement classes dissolve into `launch/` + `runners/`, and the
argv-command mechanism is deleted outright.

---

## 10. Migration plan (top-down, reusing what is proven)

Nothing below rewrites an algorithm; the work is moving proven parts under the new seams.

| Phase | Delivers | Reuses |
|---|---|---|
| **0** | shipvision GIL release + per-thread streams (§7) — the prerequisite | C1b dossier, arch test update |
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
| NVLink (0-1, 3-4, 5-6, 2-7) | 261 µs (48 GB/s) | ~30 µs | direct |
| SYS (cross-NUMA) | 756 µs (16.7 GB/s) | 31 µs | staged (driver) |
| PXB direct P2P | **98.6 ms (0.1 GB/s)** | **49 ms** | **poison — never direct** |
| PXB staged via pinned | 996 µs (12.6 GB/s) | 29 µs | staged |

`cudaDeviceCanAccessPeer` returns true for *every* pair above, including the poison ones.
Today's dev trio (GPUs 3,4,5) happens to contain no PXB pair, which is why no benchmark
ever tripped it; a 16-GPU deployment would. Production nodes are expected all-NVLink
(V139); the probe stays regardless.

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
| GIL fix inside shipvision with per-thread streams (V70 revised); this document; top-down re-implementation; gRPC control plane, argv-command deleted | V140 |
