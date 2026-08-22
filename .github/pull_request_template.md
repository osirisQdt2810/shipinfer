## Context
<!-- Why is this change needed? What problem does it solve? -->
<!-- Paraphrase the request in your own words — do not paste a chat message. -->
- Related issue:
- Background / motivation:
- Constraints / assumptions:

---

## Content / Changes
<!-- What exactly changed in this PR -->
-
-
-

<!-- Optional: call out non-obvious changes -->
- Refactors:
- New features:
- Removed / deprecated behavior:

---

## Test Plan
<!-- How was this change validated -->

### Test Details
<!-- Commands, configs, or steps used to test -->
-

### Test Output / Feature Demonstration
<!-- Paste real output: the pytest tail, the `shipinfer bench` table, a benchmark diff. -->
<!-- "Tests pass" is not a test plan. -->
-

---

## ShipInfer checklist
<!-- Delete a line only when it genuinely cannot apply. "N/A — why" is a valid answer. -->

- **Layers (ADR-001)**: `core/`, `scheduling/` and `repository/` still import no torch,
  tensorrt, onnxruntime or fastapi. `import shipinfer` still does not pull in a backend.
- **Reuse (ADR-003, the ponytail principle)**: nothing here reimplements what torch,
  torchvision or TensorRT already does. If it does, say which and why the library was
  insufficient.
- **Seams**: a new policy / queue / batcher / backend / sink / exporter is a new file plus a
  `@REGISTRY.register` decorator — not a new branch in an `if/elif`.
- **Threading (ADR-002)**: one thread, one context, one GPU; no cross-device memory access;
  the `(camera_id, frame_id)` tag survives every path including error paths.
- **Backpressure (ADR-005)**: nothing silently drops work. A refusal raises `QueueFullError`
  carrying depth and capacity; an eviction penalises the greediest camera, not its victim.
- **CUDA graphs (ADR-008)**: no captured I/O buffer is freed or reallocated.
- **Performance**: no per-request allocation on the dispatch path; no per-image Python loop
  around a device call; no host round trip that could have stayed on the device.
- **Measurements**: any speedup claim is backed by numbers in the Test Output, comparing
  like with like (device-to-device against device-to-device).
- **GPU evidence**: for a change under `runtime/`, `backends/` or `native/`, paste
  `pytest -m gpu` output and a `shipinfer bench` table with the per-device share.
- **Native (`native/`)**: uses the `gpu*` aliases so the ROCm build still compiles; releases
  the GIL around launches; has a `_into` device-pointer entry point; has a parity test.
- **Config**: new/renamed/removed settings keys, and what an older deployment does when it
  reads this config. Per-deployment settings in `core/settings/`, per-model in `config.yaml`.
- **Secrets**: no token, key or credential in the diff, including tests and fixtures.
- **Docs**: `.claude/FEATURE_LOG.md` for a large feature or seam change;
  `.claude/DECISIONS.md` (ADR) when an architectural pattern changes.
