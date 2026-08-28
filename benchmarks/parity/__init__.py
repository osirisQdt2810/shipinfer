"""The cross-plane ingest parity harness: one scenario, two planes, one committed golden.

The sync rule in ``CLAUDE.md`` says a change to a Python data-plane seam is not finished
until the C++ seam carries the same change. This package is the gate that says whether it
does, for ingest -- the one seam both planes fully implement today.

Pure and offline: it drives the real :class:`~shipinfer.ingest.IngestManager` against a
scripted source, so it imports ``shipinfer.ingest`` and nothing heavier. No torch, no
accelerator, no GPU tier -- deliberately, because none of what it compares needs a device.
"""

from __future__ import annotations

from benchmarks.parity.diff import Difference, ParityReport, by_camera, compare
from benchmarks.parity.known import KNOWN, KnownDivergence
from benchmarks.parity.scenario import CameraScript, Scenario, load_scenario
from benchmarks.parity.trace import SCHEMA_VERSION, Record, Trace, TraceWriter, read_trace

__all__ = [
    "KNOWN",
    "SCHEMA_VERSION",
    "CameraScript",
    "Difference",
    "KnownDivergence",
    "ParityReport",
    "Record",
    "Scenario",
    "Trace",
    "TraceWriter",
    "by_camera",
    "compare",
    "load_scenario",
    "read_trace",
]
