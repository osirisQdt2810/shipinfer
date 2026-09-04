"""The record seam: `build_records` on both planes, from one description of a frame.

The event seam compares two JSON writers on records a scenario states. This one compares the
two BUILDERS -- the translation units that run in production -- by describing what the graph
leaves behind and letting each plane turn it into records itself. P5-A-ALLOC's second half,
unblocked by the resolved chain plan, which is where a real field map now comes from.

Python's `build_records` and the C++ one are the same function twice, and the whole of what
they do is: resolve a class id to a label, scatter each batch's rows onto their detections,
REFUSE a row two batches cover (the chain plane's own decision -- `PoolEmbed._scatter`,
`ChainWalk.inbound`), and compose the `det_id`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shipinfer.core.events import PerceptionEvent
from shipinfer.pipeline.graph.objects import ObjectBatch
from shipinfer.pipeline.graph.state import build_records
from shipinfer.topology.elements.detections import Detections

from .record_scenario import RecordScenario, load_record_scenario

SCENARIOS = Path(__file__).resolve().parent / "scenarios" / "records"
GOLDEN = Path(__file__).resolve().parent / "golden" / "records"

__all__ = ["GOLDEN", "SCENARIOS", "batches_of", "detections_of", "load", "render"]


def detections_of(scenario: RecordScenario) -> Detections:
    """The scenario's rows as the detector would have left them.

    Labels resolved HERE from the scenario's table, because that is what the Python plane
    does (`Detections.labels` is the resolved name per row) -- the C++ plane resolves the
    same table inside `build_records`, so the two paths meet at the label and not before.
    """
    unknown = "unknown"
    return Detections(
        boxes=np.array([spec.box for spec in scenario.detections], dtype=np.float32).reshape(
            -1, 4
        ),
        scores=np.array([spec.score for spec in scenario.detections], dtype=np.float32),
        class_ids=np.array([spec.class_id for spec in scenario.detections], dtype=np.int32),
        labels=tuple(
            scenario.labels.get(spec.class_id, unknown) for spec in scenario.detections
        ),
    )


def batches_of(scenario: RecordScenario) -> dict[str, ObjectBatch]:
    """Each `batch`/`row` block as an `ObjectBatch`, keyed by name as the graph keys it."""
    built: dict[str, ObjectBatch] = {}
    for spec in scenario.batches:
        data = np.array(spec.rows, dtype=np.float32).reshape(len(spec.rows), spec.width)
        built[spec.name] = ObjectBatch(
            name=spec.name,
            class_name="",
            object_indices=spec.indices,
            data=data,
        )
    return built


def render(scenario: RecordScenario) -> str:
    """The event's one JSON line, built through the PRODUCTION record builder.

    `PerceptionEvent` directly rather than `build()`, for the event seam's reason: the
    classmethod stamps `emitted_unix_ns` from a wall clock and derives `latency_us` from a
    monotonic reading, and a byte gate cannot have either in it.
    """
    records = build_records(
        scenario.camera,
        scenario.frame,
        detections_of(scenario),
        batches_of(scenario),
        dict(scenario.fields),
    )
    event = PerceptionEvent(
        camera_id=scenario.camera,
        frame_id=scenario.frame,
        source_id=scenario.source,
        objects=records,
        img_width=scenario.width,
        img_height=scenario.height,
        img_fps=round(scenario.fps),
        captured_unix_ns=scenario.captured_unix_ns,
        emitted_unix_ns=scenario.emitted_unix_ns,
        latency_us=(
            max(0, (scenario.emitted_unix_ns - scenario.captured_ns) // 1000)
            if scenario.captured_ns
            else 0
        ),
        missing_stages=scenario.missing,
        reason=scenario.reason,
    )
    return event.to_json()


def load(name: str) -> RecordScenario:
    """A scenario by name under ``scenarios/records/``, or by path."""
    named = Path(name)
    return load_record_scenario(named if named.suffix == ".scn" else SCENARIOS / f"{name}.scn")
