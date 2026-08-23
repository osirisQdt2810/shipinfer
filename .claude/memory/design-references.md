---
name: design-references
description: Triton Inference Server and vLLM are the two systems this design must follow
metadata:
  type: project
---

The user asked, in their words, to "follow Triton inference server và vllm để thiết kế sao
cho hệ thống tối ưu nhất". Concretely:

- **From Triton:** the model repository layout and `config.yaml` vocabulary
  (`max_batch_size`, `instance_group`, `dynamic_batching`, `preferred_batch_size`,
  `max_queue_delay`), ensembles, the sequence batcher (sticky per-sequence routing), the
  response cache, KServe v2 as the HTTP protocol.
- **From vLLM:** CUDA graph capture/replay for launch-bound models, continuous batching,
  keeping the scheduler off the critical path, and the general stance of building on torch
  rather than beside it.

This is an **optimised inference system** first: the user explicitly weighted performance
above other concerns, and expects the hot data plane in C++/CUDA behind pybind while the
control plane stays Python. See [[ponytail-principle]].
