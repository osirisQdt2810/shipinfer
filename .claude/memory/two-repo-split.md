---
name: two-repo-split
description: shipinfer owns the system, shipvision owns the algorithms; the dependency is one-way
metadata:
  type: project
---

As of 2026-08-23 the work is **two repositories**, and the boundary is not organisational:

- **`shipinfer`** — the system. RTSP/GStreamer ingest for ~50 cameras, scheduling across 16
  GPUs, the Triton-shaped serving core, the pipeline, Kafka output, observability.
- **`shipvision`** — the algorithms. imgproc kernels, detection, re-ID, MOT, MTMC, Optuna
  tuning, evaluation metrics.

`shipinfer` calls `shipvision`; **never the reverse**, and `shipvision` imports nothing from
`shipinfer`. The user's own analogy: "shipinfer sẽ gọi tới thư viện này, nó kiểu giống như
vllm gọi code aiter vậy."

**Why:** an algorithm is judged by HOTA/IDF1/rank-1/mAP on recorded footage, and that
measurement has to run in seconds with no GPU and no engine to load — otherwise it does not
get run, and whichever algorithm shipped first wins by default instead of by evidence.

**How to apply:** a new algorithm goes in `shipvision` behind its family's registry, with
both a `native` and a `python` backend and a parity test between them. Anything about
cameras, queues, GPUs, HTTP or Kafka goes in `shipinfer`. The four earlier per-module repos
(`shipinfer-imgproc`, `-mot`, `-reid`, `-mtmc`) are being folded into `shipvision` and then
deleted — the user chose "Gộp hết, xoá repo cũ".

Related: [[ponytail-principle]], [[package-per-extension-point]], [[reference-repos]]
