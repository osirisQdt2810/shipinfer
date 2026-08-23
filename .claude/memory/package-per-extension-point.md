---
name: package-per-extension-point
description: Every extensible family is a folder of one-class-per-file modules plus a registry, never one big module
metadata:
  type: feedback
---

Anything that could gain a second implementation gets its own **package**: `policies/`,
`queues/`, `batching/`, `backends/`, `providers/`, `memory/`, `graphs/`, `ops/`, `sinks/`,
`exporters/`, `cli/commands/`. One class per file, a `base.py` with the ABC, a
`registry.py` holding a `Registry` object, and `@REGISTRY.register("name")` on each class.
This applies to `core/` too — logging, metrics, settings and errors are packages, not
modules.

**Why:** the user expects the project to grow large, and wants it reusable and easy to
extend without editing a switch statement. They asked for this specifically after seeing a
monolithic `policy.py`.

**How to apply:** when adding an implementation, add a file and a decorator — never an
`elif`. Reuse `shipinfer.core.registry.Registry`; use `register_lazy` for anything whose
import is expensive (TensorRT, torch backends). Hand-written alternates to a library-backed
default are named `Custom*` and registered alongside it — see [[ponytail-principle]].
