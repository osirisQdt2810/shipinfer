# Journal

Newest on top. One entry per working session: what changed, what it cost, what is next.

---

## 2026-08-22 — Project reset: from Sale Hunter scaffolding to ShipInfer

**What this session did.** Replaced the inherited scaffolding (a Shopee desktop-app
template) with the real project: a Triton-shaped multi-GPU inference server for the ship /
person perception pipeline described in `references/bitbucket-subfaceid/docs/`.

**Built.**

- `core/` as packages, not modules — `types`, `request`, `errors`, `settings`, `logging`
  (with a bounded async sink), `metrics` (with pluggable exporters), plus the `Registry`
  primitive every extension point uses.
- `scheduling/` — the part this project exists to own. Fair per-camera queueing with
  priority lanes, dynamic batching windows, five placement policies, and a dispatcher that
  spills to the next-shortest queue rather than dropping a frame.
- `runtime/` — the accelerator seam, on torch. Device manager, streams, CUDA graph capture,
  a pinned staging pool, and three image-op implementations (numpy / torch / native).
- `backends/` — mock, TensorRT (engine + persistent bindings + graph replay), ONNX Runtime,
  TorchScript.
- `server/` — instances, models, a validated ensemble DAG, response cache, health, KServe v2.
- `native/` — C++17 + CUDA/HIP behind pybind11: fused letterbox, batched crop, device NMS.
- A demo `model_repository/` carrying the real DAG on the mock backend, so the whole thing
  is runnable and testable anywhere.

**The correction that shaped the result.** The first version of `runtime/` hand-rolled a
CUDA layer against `cuda-python` — its own caching allocator, pinned-memory management,
stream wrappers, raw graph capture. It worked and it was slower than torch, missing exactly
the parts that are hard. Rewritten onto torch (ADR-003). The hand-written code was kept as
registry-selectable `custom_*` variants: runnable documentation for what the library does,
and the other half of the parity tests.

**Evidence.**

- offline suite: 149 passed, 12 deselected, ~7 s, no GPU
- GPU tier: 12 passed on 8 × RTX A5000
- `shipinfer bench person_embedder --cameras 50 --fps 20 --seconds 3 --skew 8`:
  3000/3000 completed at 998 req/s, p99 7.6 ms, 0 rejected; per-device share 11.7–13.2%
  across all 8 GPUs; every quiet camera served despite 8× skew
- fused letterbox, 8 × 1080p → 640², writing into a torch CUDA tensor: **9.7 ms (822 img/s)
  vs torch's 13.7 ms (585 img/s)**, output bit-identical

**Notable bugs found and fixed on the way.**

- `np.clip(..., out=fancy_index)` in the numpy crop path neither wrote back nor cast — it
  raised, and only for some dtypes. Caught by the GPU parity test; host-tier tests added so
  it would have been caught without a GPU.
- The ensemble DAG validated tensor *names* but not shapes, so a 512×512 ship crop fed a
  256×128 embedder and failed on the first frame instead of at start-up. Validation now
  type-checks every edge.
- The bench discarded futures completed during intermediate waits and reported 496 of 3000.

**Next.**

- `pipeline/` and `ingest/` are declared in the layout but not yet implemented: NVDEC
  decode, per-camera actors with backpressure to the decoder, and the `(cam, frame)`
  reassembler.
- Kafka result publishing, to hand tracklets to `motservice`.
- A real TensorRT engine in the repository, so the GPU tier exercises the production path
  rather than the mock.
