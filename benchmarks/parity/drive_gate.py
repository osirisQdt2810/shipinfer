"""The observation gate through the REFERENCE, so the C++ port has something to agree with.

The same shape as :mod:`benchmarks.parity.drive_identity`: a scenario file both planes read,
and the reference's answer committed as the golden. `csrc/tests/test_gate_parity.cpp` replays
it. What is being pinned is the ORDER of the two gates -- height, then age over qualifying
frames only -- and the boundedness of the one piece of state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

SCENARIOS = Path(__file__).resolve().parent / "scenarios" / "gate"
GOLDEN = Path(__file__).resolve().parent / "golden" / "gate"

__all__ = ["GOLDEN", "SCENARIOS", "GateScenario", "load", "render_gate"]


class GateScenario(NamedTuple):
    name: str
    min_hits: int
    min_height_fraction: float
    #: One list of ``(camera, track, box_height_px)`` per instant.
    instants: list[list[tuple[str, int, float]]]


def load(name: str) -> list[GateScenario]:
    """Parse a scenario file. See its own header for the format."""
    path = Path(name)
    if path.suffix != ".txt":
        path = SCENARIOS / f"{name}.txt"
    scenarios: list[GateScenario] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        head, *rest = stripped.split()
        if head == "scenario":
            scenarios.append(GateScenario(rest[0], int(rest[1]), float(rest[2]), []))
            continue
        if head != "instant":
            raise ValueError(f"{path}: unknown line {stripped!r}")
        if not scenarios:
            raise ValueError(f"{path}: an `instant` before any `scenario`")
        instant: list[tuple[str, int, float]] = []
        for field in rest:
            key, height = field.split(":")
            camera, track = key.split("#")
            instant.append((camera, int(track), float(height)))
        scenarios[-1].instants.append(instant)
    return scenarios


def render_gate(scenarios: list[GateScenario]) -> str:
    """Run every scenario through the reference and render what it admitted.

    Imported here rather than at module scope so this module stays importable without the
    submodule checked out.
    """
    import numpy as np
    from shipvision.mtmc.frames import TrackKey, TrackObservation
    from shipvision.mtmc.gating import ObservationGate
    from shipvision.types import FrameTag, Track

    def observation(camera: str, track: int, height: float) -> Any:
        return TrackObservation(
            key=TrackKey(camera, track),
            track=Track(
                track_id=track,
                box=np.asarray([0.0, 0.0, 40.0, height], dtype=np.float32),
                tag=FrameTag(camera_id=camera, frame_id=0),
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            ),
            frame_height=1080,
            frame_width=1920,
        )

    lines = [
        "# The REFERENCE's answer to scenarios/gate/*.txt. Emitted, never hand-edited:",
        "#   python scripts/emit_parity_golden.py --kind gate --scenario basic --emit-golden",
    ]
    for scenario in scenarios:
        gate = ObservationGate(
            min_hits=scenario.min_hits, min_height_fraction=scenario.min_height_fraction
        )
        lines.append(f"scenario {scenario.name}")
        for instant in scenario.instants:
            admitted = gate.filter([observation(*entry) for entry in instant])
            names = " ".join(f"{o.key.camera_id}#{o.key.track_id}" for o in admitted)
            lines.append(f"admitted {names}".rstrip())
        lines.append(f"held {len(gate)}")
    return "\n".join(lines) + "\n"
