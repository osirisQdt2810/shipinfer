"""Build one event scenario on the Python plane and write the line it serialises to.

The golden here is a single JSON line rather than a trace: an event is one value, and what
the two planes must agree on is its bytes -- key order included, because a deployed
`motservice` reads this and the C++ half writes JSON without ever parsing it.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.parity.drive_python import GOLDEN as INGEST_GOLDEN
from benchmarks.parity.drive_python import SCENARIOS as INGEST_SCENARIOS
from benchmarks.parity.event_scenario import EventScenario, ObjectSpec, load_event_scenario
from shipinfer.core.events import ObjectRecord, PerceptionEvent

__all__ = ["GOLDEN", "SCENARIOS", "load", "render"]

SCENARIOS = INGEST_SCENARIOS / "events"
GOLDEN = INGEST_GOLDEN / "events"


def _record(spec: ObjectSpec) -> ObjectRecord:
    return ObjectRecord(
        det_id=spec.det_id,
        class_name=spec.class_name,
        score=spec.score,
        bbox=spec.bbox,
        embedding=spec.embedding,
        ship_id=spec.ship_id,
        similarity=spec.similarity,
        mask_area_px=spec.mask_area_px,
        track_id=spec.track_id,
        track_state=spec.track_state,
        global_id=spec.global_id,
    )


def render(scenario: EventScenario) -> str:
    """The event's one JSON line.

    `PerceptionEvent` directly rather than `build()`: the classmethod stamps
    `emitted_unix_ns` from the wall clock and derives `latency_us` from a monotonic reading,
    and a gate that compares bytes cannot have either in it. The scenario states both, so
    what is compared is the SCHEMA -- key order, types, null handling -- and not two clocks.
    """
    event = PerceptionEvent(
        camera_id=scenario.camera,
        frame_id=scenario.frame,
        source_id=scenario.source,
        objects=tuple(_record(spec) for spec in scenario.objects),
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


def load(name: str) -> EventScenario:
    """A scenario by name under ``scenarios/events/``, or by path."""
    named = Path(name)
    return load_event_scenario(named if named.suffix == ".scn" else SCENARIOS / f"{name}.scn")
