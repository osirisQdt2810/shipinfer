"""The benchmark harness: one load, two systems, one measurement.

Split by role rather than by system, because the parts that must be *identical* across the
two systems are exactly the parts worth isolating:

- :mod:`~benchmarks.harness.config` — the load, the hardware and the artefacts. One object,
  shared, so "same load" is a fact rather than a claim.
- :mod:`~benchmarks.harness.sampler` — the once-a-second occupancy writer, in the shape the
  baseline already writes, so the two logs are parsed by the same reader.
- :mod:`~benchmarks.harness.baseline` — drives ``benchmarks/baseline``'s own binary. It
  compiles ``sim_pipeline_v2.cpp`` unchanged; nothing here edits the submodule.
- :mod:`~benchmarks.harness.shipinfer` — drives our stack end to end, ingest through
  reassembly.
- :mod:`~benchmarks.harness.analysis` — the measurement itself: fit each buffer's growth
  over the steady window and turn it into a sustained rate.

The entry point is ``benchmarks/run_bench.py``, which owns the one decision none of these
modules can make alone: what counts as an image, once, for each system.
"""

from __future__ import annotations

__all__ = ["analysis", "baseline", "config", "sampler", "shipinfer"]
