# T3 — topology C, `service`: a cross-process inference tier over the fleet

*Implementation plan. No code. Written in the main session after the planner agent was cut
off by the spend limit; references read: vLLM `shm_broadcast.py` (`ShmRingBuffer`,
`MessageQueue`), `multiproc_executor.py` (`MultiprocExecutor`, `WorkerProc`),
`core_client.py` (`DPLBAsyncMPClient`), Triton `rate_limiter.h`.*

## 1. The shape, in the Triton/vLLM analogy

`fleet` (T2) is vLLM's data-parallel deployment with no load balancer: N engine processes,
each owning its GPU and its cameras, nothing crossing. `service` adds what vLLM's
`DPLBAsyncMPClient` adds — a router that sees every engine's load and sends work to the least
loaded — but **symmetrically and per model instance**, the way Triton's `instance_group`
spreads one model over GPUs: every shard process keeps *serving* its own GPU's crop-stage
instances, and *also* offers them to its peers. The "tier" is not a new process; it is every
shard's crop-stage `Model` seeing, in its dispatcher's candidate set, the other shards'
instances as proxies. Who routes: the existing `Dispatcher` with the existing
`locality_spillover` policy — local while the local queue is shallow, spill to the shallowest
peer when it is not. Who publishes load: each proxy reads the owner's queue depth and EWMA
latency from a shared-memory header the owner writes on every enqueue/dequeue (vLLM publishes
`lb_engines` counts back through `process_engine_outputs`; ours are plain shared counters,
because the reader is in the same box). How work crosses: one pinned shared-memory **ring per (submitter, owner, model)** — vLLM's
`ShmRingBuffer` discipline (fixed slots, a state byte each, spin then `sched_yield`, timeout)
— with exactly one writer per ring, the submitter, which is why the rings are pairwise rather
than one per owner (§2: Python has no compare-and-swap on shared memory). How results come
back: the same pair has a **response ring** in the other direction (vLLM's
`worker_response_mq` per worker, made pairwise so it too has one writer — the owner); the
owner's instance writes the output rows into that ring's slot and publishes it; the
submitter's reader thread resolves the `ResponseFuture`. Nothing crosses a GPU boundary except through
host pinned memory: D2H on the owner side of the copy, H2D on the other — the ledger's
~125 µs per 1.5 MB on PCIe 4 — because opening CUDA IPC handles on other GPUs would cost G
contexts per process (G² × ~300 MiB on the box).

What we deliberately do **not** take: vLLM's ZMQ control plane (one box, no sockets needed —
the ring header *is* the control plane); Triton's rate limiter's resource accounting (our
policy already reasons in queue depth and latency, which is what the ledger says balances the
skewed fleet); a central router process (a dead router would be the shared-buffer failure the
project exists to fix, wearing a new coat).

## 2. Files, signatures, errors

### `core/errors/topology.py` (extend)
```python
class RingFullError(ShipInferError):        # ADR-005: carries depth and capacity
    def __init__(self, owner: str, model: str, depth: int, capacity: int) -> None: ...
class PeerLostError(ShipInferError):        # carries (camera_id, frame_id) of every in-flight request lost
    def __init__(self, owner: str, tags: Sequence[tuple[str, int]]) -> None: ...
class RingProtocolError(ShipInferError):    # header version / layout mismatch between processes
```

### `core/settings/topology.py` (extend)
```python
class ServiceSettings(BaseModel):
    """`service`: the fleet plus a cross-process tier for the crop-stage models."""
    shared_models: list[str] = ["person_embedder", "ship_embedder", "ship_segmenter"]  # crops, not frames
    slots_per_pair: int = 8       # per (submitter, owner, model) ring — single writer each; 8 × slot_bytes pinned per ring
    slot_bytes: int = 1_638_400   # 1.5 MiB + 64 KiB (400 pages): one crop batch (32 × 3 × 128 × 64 fp16 is exactly 1.5 MiB) plus the request head and the per-tensor heads, which travel in the slot ahead of the bytes; letterboxed full frames do not fit, by design
    submit_timeout_ms: int = 5    # a full ring refuses after this; the policy then picks another candidate
    heartbeat_ms: int = 200       # an owner that has not stamped its header in 5 × this is lost
    spill_threshold: int = 4      # forwarded to locality_spillover for the shared models
class TopologySettings(BaseModel):
    kind: str = "fleet"; shards: int | None; drain_s: float = 20.0
    service: ServiceSettings = ServiceSettings()
```
Environment the launcher adds for `service` children (beside the three from T2):
`SHIPINFER_TOPOLOGY__KIND=service`, `SHIPINFER_SERVICE__RING_DIR=<run dir>` (where the
`multiprocessing.shared_memory` names are published as small JSON files, one per (owner,
model) — the "handle" vLLM pickles across processes), `SHIPINFER_SERVICE__SHARD_INDEX`.

### `runtime/memory/shared_ring.py` (new; `runtime/` may use torch)
```python
class RingLayout:      # pure arithmetic, testable offline
    """header | slot metadata (written, claimed, reader flag) × slots | slot payloads × slots."""
    def __init__(self, slots: int, slot_bytes: int) -> None: ...
    header_bytes: int; total_bytes: int
    def slot_offset(self, index: int) -> int: ...
    def meta_offset(self, index: int) -> int: ...
class RingHeader:      # struct over the first page: version, owner shard, model name hash,
                       # depth (u32), ewma_latency_us (f32), heartbeat_ns (u64), closed (u8)
class SharedRing:
    """One pinned `multiprocessing.shared_memory` ring. Created by the owner, opened by peers."""
    @classmethod
    def create(cls, name: str, layout: RingLayout, *, pin: bool = True) -> SharedRing: ...
    @classmethod
    def open(cls, name: str, layout: RingLayout, *, pin: bool = True) -> SharedRing: ...
    def claim(self, timeout_s: float) -> int:            # writer side: atomically take a free slot or raise RingFullError
    def publish(self, index: int) -> None:                # writer: memory_fence, written=1
    def take(self, timeout_s: float | None) -> int | None:  # reader (owner): next written slot, or None on timeout
    def release(self, index: int) -> None:                # reader: written=0, claimed=0 → free
    def payload(self, index: int) -> memoryview: ...
    def pinned_tensor(self, index: int) -> torch.Tensor:  # torch.frombuffer over the slot, host-registered once per process
    def stamp(self, depth: int, ewma_latency_us: float) -> None:  # owner, on every enqueue/dequeue
    def header(self) -> RingHeader: ...
    def close(self) -> None: ...   # owner: closed=1, then unlink after peers drop (the "d" in the ledger)
```
Pinning: `torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 0)` once per process per ring, in
`create`/`open`; `cudaHostUnregister` in `close`. The registration is per process because a
page is pinned for *this* process's DMA engine; the ring exists once. Slot claim is a
compare-and-swap on the slot's `claimed` byte — Python has no CAS on shared memory, so the
claim is a per-slot `multiprocessing` lock-free trick vLLM avoids by having one writer; we
have N writers, so: **one submit ring per (submitter, owner, model)**, single writer each,
which restores vLLM's discipline exactly. **The budget, derived once and cited everywhere
else:** a directed pair (A → B) has two rings — requests (A writes, B reads) and results (B
writes, A reads) — so the box holds `N × (N−1) × models × 2` rings of `slots × slot_bytes`,
each existing once as one `shared_memory` block. Every ring is mapped by exactly two processes
(its writer and its reader) and each registers its own mapping, so a process pins
`4 × (N−1) × models` rings. For 4 shards and 3 models: 72 rings on the box; at 64 slots of
1.5 MiB that is 6.8 GiB of shared memory — too much. Hence the setting is `slots_per_pair`,
sized 8 by default, with 64 KiB of headroom per slot for the heads (`slot_bytes` above):
**72 × 12.5 MiB ≈ 0.9 GiB of shared memory on the box, existing once; 36 × 12.5 MiB ≈ 0.44 GiB
registered per process** (the same pages, pinned through each mapper's registration). **This
is the first design decision the coder must not silently re-make**: single writer per ring,
pairwise rings, small slot counts.

### `server/remote_instance.py` (new)
```python
class RemoteInstance:  # satisfies scheduling.policies.base.Placeable
    """A peer shard's instance of one model, seen through its ring header."""
    def __init__(self, owner: int, model: str, submit: SharedRing, results: SharedRing, staging: PinnedStagingPool) -> None: ...
    @property device -> Device      # `Device.cpu()`: a proxy is not here. Policies compare `device` to
                                    # `request.resident_device`, so a proxy never equals the local device and never
                                    # wins a locality tie; the response's `executed_on` says where it really ran
    @property depth -> int          # header.depth, read without a lock (same rule as the local instance)
    @property ewma_latency_us -> float
    @property is_ready -> bool      # header.heartbeat within 5 × heartbeat_ms and not closed
    def enqueue(self, item: WorkItem) -> None:
        """D2H the batch's input rows into a claimed slot (async on the caller's stream, event-recorded),
        write the request header (request_id, camera_id, frame_id, priority, deadline_ns, shapes), publish.
        Raises RingFullError after submit_timeout_ms — the dispatcher's retry loop then excludes this
        candidate and re-selects, which is exactly how `Dispatcher.dispatch` already handles a full local queue."""
class ResultReader(threading.Thread):
    """One per process: waits on this process's result rings, resolves futures, H2D into the
    caller's staged output, fails every pending future of a lost owner with PeerLostError."""
```
The tag `(camera_id, frame_id)` rides in the slot header, and `WorkItem.fairness_key` is
still the camera id on the *owner's* side: the owner's `ModelInstance` receives the remote
item through the same `FairPriorityQueue` as local work, so per-camera fairness holds across
processes (the ledger's (c)).

### `server/instance.py` (small change)
`ModelInstance` gains a `RingIngress` thread started only under `service`: `take()` from every
inbound ring for this model, wrap the slot as an `InferenceRequest` whose inputs are torch
tensors over the pinned slot (`from_torch`), `enqueue` as a normal `WorkItem` whose future's
callback writes outputs to the submitter's result ring and `release`s the slot. No change to
`execute_batch`: a remote item is a local item that happens to hold pinned memory.

### `server/model.py` (small change)
`Model.__init__` accepts `extra_instances: Sequence[Placeable] = ()` and passes
`instances=[*self._instances, *extra_instances]` to the `Dispatcher`. `Model` never knows what a
proxy is — the dispatcher and the policy see `Placeable`, which is the whole point of that
protocol. `policy` for a shared model defaults to `locality_spillover(spill_threshold)`.

### `server/topology/service.py` (new, `@TOPOLOGIES.register("service")`)
```python
class ServiceTopology(Topology):
    name = "service"
    def plan(...) -> ShardPlan: return plan_shards(...)           # identical to fleet
    def command(self, shard, *, repository) -> Sequence[str]: ...  # identical to fleet
    def environment(self, settings) -> Mapping[str, str]: ...       # + KIND, RING_DIR, SHARD_INDEX
    def describe(self) -> str: "fleet plus a cross-process tier for <shared models>"
```
The child side: `server/engine.py` (or wherever models are built) asks
`build_topology(settings.topology.kind).attach(model, settings, devices)` — a new optional
`Topology.attach(...) -> Sequence[Placeable]` hook returning the proxies for that model, `()`
for `fleet`. That keeps the contract small: **one** new optional method, with a default.

### `pipeline/reassembly/collector.py` (no change expected)
The collector waits on `ResponseFuture`s; a remote result resolves the same future. What
changes is only *when* it resolves — and the stage timeout already covers a lost peer, with
`PeerLostError` naming the tags instead of a bare timeout.

## 3. One crop batch spilling from shard A (GPU 2) to shard B (GPU 3)

1. A's crop stage builds the batch on GPU 2 (device tensor, 32 crops fp16, ~1.5 MB) and calls
   `model.infer(request)` with `resident_device = cuda:0` (A's logical view).
2. `Dispatcher.select`: local instance depth 9 > `spill_threshold` 4 → `locality_spillover`
   falls back to shortest queue among *ready* candidates; B's proxy reports depth 1 → chosen.
3. `RemoteInstance.enqueue`: `claim()` a slot on the A→B ring for this model (spin ≤ 5 ms;
   `RingFullError` → the dispatcher marks the candidate and re-selects, at most `attempts`).
   `cudaMemcpyAsync` D2H into the pinned slot on A's stream (~125 µs for 1.5 MB), record an
   event, **wait on the event** (the slot must be complete before `publish`; this is the one
   synchronous wait on the path, ~130 µs), write the slot header (request_id, camera_id,
   frame_id, priority, deadline_ns, dtype/shape), `memory_fence`, `publish`. A's future is
   registered in A's `ResultReader.pending[request_id]`.
4. B's `RingIngress` (blocked in `take()` with a 1 ms poll then `sched_yield` spin) sees the
   written flag, wraps the slot as tensors, `enqueue`s a `WorkItem` into B's instance
   `FairPriorityQueue` under fairness key = A's camera id. B `stamp`s depth+1.
5. B's instance batch window closes; `execute_batch` copies the pinned slot H2D on GPU 3
   (~125 µs, on B's stream), runs, and the completion callback D2H's the output rows into
   A's result ring slot (~50 µs for 32 × 512 fp16 = 32 KB... negligible), publishes, `release`s
   the inbound slot, `stamp`s depth−1.
6. A's `ResultReader` sees the result slot, H2D on GPU 2 into the staged output (if the
   consumer wants device memory; the collector takes host arrays today, so this copy may be
   skipped), resolves the future with an `InferenceResponse` carrying `timings.remote_ns`.
Cost: two PCIe copies (~250 µs) + one event wait + two ring hand-offs (spin latency ~10–50 µs
each) ≈ **0.4 ms added to a batch that takes ~3–8 ms to embed** — worth it whenever the local
queue's wait exceeds that, which is what `spill_threshold` encodes in queue depths.

## 4. Failure modes

| Failure | What happens | Where it is tested |
|---|---|---|
| Peer B dies mid-request | B's heartbeat stops; A's `ResultReader` marks B's rings lost after 5 × 200 ms, fails every pending future for B with `PeerLostError(tags)`, `RemoteInstance.is_ready` → False so the dispatcher stops selecting it (ledger (d)); the fleet supervisor still takes the fleet down (`ShardExitedError`) — C changes *what is lost*, not whether a dead shard is a fleet failure | offline: fake ring with a frozen heartbeat; multigpu: kill one shard |
| Ring full | `claim` times out → `RingFullError(depth, capacity)`; dispatcher re-selects (local or another peer); never silently dropped | offline |
| Consumer slower than producer | depth in the header rises → `locality_spillover` stops spilling to it (its depth exceeds the local one) — the ring is a queue whose depth the policy reads, so the back-pressure *is* the routing signal | offline, with a scripted depth |
| A process restarts | the new process re-`create`s its rings under the same names after `unlink`; peers `open` lazily on `is_ready` flipping back; pending futures from the old incarnation were already failed by the heartbeat rule | multigpu |
| `CUDA_VISIBLE_DEVICES` mismatch | a proxy has no device ordinal at all (`device` is `cpu`), so it can never compare equal to `resident_device` and a peer's numbering never reaches a local policy; the owner reports `executed_on` in *its* numbering, which the per-device counters label by shard; a header whose owner index disagrees with the ring name → `RingProtocolError` at open | offline |
| Header layout drift between versions | `RingHeader.version` checked at `open`; mismatch → `RingProtocolError` naming both versions | offline |

## 5. Tests by tier

- **offline** (`tests/runtime/test_shared_ring.py`, `tests/server/test_remote_instance.py`,
  `tests/server/test_service_topology.py`): `TestTheLayoutIsArithmetic` (offsets, page
  alignment, total bytes); `TestClaimPublishTakeRelease` over a real `shared_memory` block in one
  process with a thread as the peer (no torch: `pin=False`); `TestAFullRingRefusesWithNumbers`;
  `TestTheProxyIsAPlaceable` (a `FakeRing` header drives `depth`/`ewma`/`is_ready`;
  `locality_spillover` picks local under threshold, the proxy above it — reusing
  `tests/scheduling/test_policies.py`'s `FakeInstance` style); `TestALostOwnerFailsItsFutures`
  (frozen heartbeat → `PeerLostError` carrying the tags); `TestServiceRegistersAgainstTheSeam`
  (`TOPOLOGIES["service"]`, `attach()` returns `()` for fleet and proxies for service,
  `describe()` names the shared models).
- **gpu** (`-m gpu`): `TestPinnedSlotsRoundTrip` — D2H into a slot, H2D out on the same device,
  bit-exact; the registration happens once per process (`cudaHostRegister` counted through a
  wrapper).
- **multigpu** (`-m multigpu`): two real shard processes on GPUs 2 and 3 with the mock backend,
  one spills to the other, results resolve, per-device counters show the split; kill one, the
  other's pending futures fail with `PeerLostError`.
- **bench gate** (inside the container): `deploy/rootless/bench.sh --systems shipinfer --cameras 50
  --fps 20 --seconds 40 --gpus 2,3,4,5` with the fleet driving the shards, once under
  `--topology fleet` and once under `--topology service` (the harness gains the flag; `shipinfer
  bench` is the in-process demonstration and one interpreter generates ~2% of 50 × 20, so it
  cannot produce this number). "C beats B" must show: per-device *retired* counts within 10% of each other under `--skew 8` (B shows the
  busy shard's device at the offered skew), p99 end-to-end on the busy cameras lower, and no
  increase in `frames_failed`. Both numbers as CAPACITY per the harness's rules.

## 6. The PR cut

1. **`feat(runtime): the pinned shared ring`** — `runtime/memory/shared_ring.py`,
   `core/errors/topology.py` (+2 errors), `tests/runtime/test_shared_ring.py` (+gpu class).
   ~6 files.
2. **`feat(server): RemoteInstance and the result reader`** — `server/remote_instance.py`,
   `server/instance.py` (`RingIngress`), `server/model.py` (`extra_instances`), tests. ~8 files.
3. **`feat(topology): service`** — `server/topology/service.py`, `Topology.attach`,
   `core/settings/topology.py` (`ServiceSettings`), the launcher's environment, CLI wiring,
   `tests/server/test_service_topology.py`, the multigpu test, docs + feature log + ADR-015
   ("inference crosses processes through pinned host memory, never CUDA IPC"). ~10 files.
4. **`bench(topology): B against C`** — `--topology` on the harness (`bench.sh` → `run_bench.py`,
   the fleet driving the shards), the per-device table, the evidence run. ~4 files.

## 7. Open questions for the operator

- Slot size: 1.5 MiB fits an embedder crop batch; the **segmenter** takes letterboxed crops that
  may exceed it at `max_batch_size` — cap the batch for shared models, or size slots per model
  from `config.yaml` dims (my default: per model, from dims, rounded to 64 KiB)?
- Should the **detector** ever be shared? The ledger says crops, not frames; a 1080p frame is
  6 MB and would triple the pinned footprint. My default: never — `shared_models` excludes it
  and a config naming it is refused.
- Pinned budget (the derivation in §2): 4 shards × 3 models → 72 rings on the box, existing once,
  ≈ 0.9 GiB of shared memory at 8 slots × (1.5 MiB + 64 KiB); each process registers the 36 it
  maps, ≈ 0.44 GiB. Acceptable on this host (the operator's box has 500+ GB), but it must be
  stated in the settings docstring; confirm the ceiling.
