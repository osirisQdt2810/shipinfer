---
name: reference-repos
description: What each checkout under references/ is, and which parts are worth porting
metadata:
  type: reference
---

`references/` holds read-only SSH clones from `github.com/ShipControlPrj`. They are the
previous generation of this system and the source of the requirements:

- `bitbucket-subfaceid` — the C++/DeepStream perception service. `docs/flow.md`,
  `docs/new-system-architecture.md` and `docs/session-2026-06-05.md` are the actual spec:
  the pipeline, the single-GPU limitation, and the shared-`personBuffer` eviction bug that
  starved quiet cameras.
- `bitbucket-generic-object-detection-trt` — TensorRT detector. `src/tools/imgproc/*.cu`
  holds the resize/crop/normalise CUDA kernels worth porting as one fused kernel.
- `bitbucket-generic-feature-extractor-trt` — TensorRT ReID embedder, same kernel structure.
- `gitea-generic-multi-object-tracking-cpp` — DeepSORT-style MOT with Kalman, Hungarian and
  lapjv under `src/tracker/`.
- `bitbucket-motservice` / `bitbucket-mtmcservice` — single-camera and multi-camera tracking
  services; they consume perception results over Kafka.
- `bitbucket-countingservice` — the one Python service (FastAPI + pydantic-settings +
  confluent-kafka + torch), and the closest thing to a house style for Python here.
