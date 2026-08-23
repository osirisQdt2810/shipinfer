---
name: ponytail-principle
description: "Ponytail" is the user's term for: reuse mature, highly-optimised libraries instead of reimplementing them
metadata:
  type: feedback
---

The user calls this **"ponytail"**: basic, powerful, already-optimised libraries must be
used, not reimplemented. Torch is the standing example — its caching device allocator,
caching pinned-host allocator, `Stream`/`Event`, and `CUDAGraph` capture (with side-stream
warm-up and shared memory pools) are all things they explicitly do not want hand-rolled.

**Why:** a hand-written version runs but is "cực kì chậm chạp" compared to the native
package, and it is more code to maintain for a worse result. Their reference point is vLLM,
which builds *on* torch and writes custom CUDA only where torch has no equivalent.

**How to apply:** default to torch/TensorRT/NCCL/cuBLAS/numpy/pydantic/FastAPI for anything
they already do well. Write custom code only for (a) the layer above — scheduling, batching,
placement — and (b) fused kernels with no library equivalent. A hand-written version that
already exists may stay as a registry-selectable `Custom*` variant, which the user values as
a readable explanation of what the native library does; see
[[package-per-extension-point]].
