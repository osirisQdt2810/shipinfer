# Feature Log

One entry per large feature or seam change. Append-only, newest on top. Skip it for tiny
edits, typo fixes and pure docs.

---

## 2026-08-22 — Initial system: scheduler, runtime, backends, server, native kernels

**Why.** The previous generation (`references/bitbucket-subfaceid`) ran every model on GPU 0
and starved quiet cameras through a shared evict-oldest buffer. The requirement is 50
cameras × 20 fps across 16 GPUs with balanced load and bounded tail latency.

**Seams introduced.**

| Seam | Where | Extension point |
|---|---|---|
| Registry primitive | `core/registry.py` | eager + lazy registration |
| Placement policies | `scheduling/policies/` | `@POLICIES.register` |
| Request queues | `scheduling/queues/` | `@QUEUES.register` |
| Batchers | `scheduling/batching/` | `@BATCHERS.register` |
| Backends | `backends/` | `@BACKENDS.register` / `register_lazy` |
| Allocators | `runtime/memory/` | `@ALLOCATORS.register` |
| Graph caches | `runtime/graphs/` | `@GRAPH_CACHES.register` |
| Image ops | `runtime/ops/` | `@IMAGE_OPS.register` |
| Log sinks | `core/logging/sinks/` | `@SINKS.register` |
| Metrics exporters | `core/metrics/exporters/` | `@EXPORTERS.register` |
| Response caches | `server/cache/` | `@RESPONSE_CACHES.register` |
| CUDA providers | `runtime/providers/` | `@PROVIDERS.register` (custom variants only) |

**Decisions recorded.** ADR-001 through ADR-009 — the pure core, one-thread-one-GPU, torch
as substrate, locality-aware spillover, fair queueing, the Triton repository layout, the
Python/C++ split, CUDA-graph buffer lifetime, and the opt-in response cache.

**Evidence.** 149 offline tests (no GPU) + 12 GPU tests; 998 req/s at p99 7.6 ms with
11.7–13.2% per-device share across 8 × A5000; fused letterbox 1.41× faster than torch with
bit-identical output.
