"""Cross-camera identity through the REFERENCE, so the C++ port has something to agree with.

The golden here is not a trace of a run: it is the reference implementation's answer to a
scenario file both planes read. `csrc/tests/test_identity_parity.cpp` replays the same
scenarios through `GlobalIdAssigner` and compares -- so a change to either side that alters
one id fails, which is the only thing that keeps a 500-line port a port.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

SCENARIOS = Path(__file__).resolve().parent / "scenarios" / "identity"
GOLDEN = Path(__file__).resolve().parent / "golden" / "identity"

__all__ = ["GOLDEN", "SCENARIOS", "Scenario", "load", "render_identity"]


class Instant(NamedTuple):
    """One synchronised instant: a label, a key and an embedding per observation."""

    labels: list[int]
    keys: list[tuple[str, int]]
    embeddings: list[tuple[float, float]]


class Scenario(NamedTuple):
    name: str
    max_age: int
    capacity: int
    max_tracks: int
    instants: list[Instant]


def load(name: str) -> list[Scenario]:
    """Parse a scenario file. See its own header for the format."""
    path = Path(name)
    if path.suffix != ".txt":
        path = SCENARIOS / f"{name}.txt"
    scenarios: list[Scenario] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        head, *rest = stripped.split()
        if head == "scenario":
            scenarios.append(Scenario(rest[0], int(rest[1]), int(rest[2]), int(rest[3]), []))
            continue
        if head != "instant":
            raise ValueError(f"{path}: unknown line {stripped!r}")
        if not scenarios:
            raise ValueError(f"{path}: an `instant` before any `scenario`")
        labels, keys, embeddings = [], [], []
        for field in rest:
            label, key, embedding = field.split(":")
            camera, track = key.split("#")
            x, y = embedding.split(",")
            labels.append(int(label))
            keys.append((camera, int(track)))
            embeddings.append((float(x), float(y)))
        scenarios[-1].instants.append(Instant(labels, keys, embeddings))
    return scenarios


def render_identity(scenarios: list[Scenario]) -> str:
    """Run every scenario through the reference and render its answers.

    Imported here rather than at module scope so this module stays importable without the
    submodule checked out -- the same reason `topology/elements/track.py` loads it lazily.
    """
    import numpy as np
    from shipvision.mtmc.frames import TrackKey, TrackObservation
    from shipvision.mtmc.identity import GlobalIdAssigner
    from shipvision.types import FrameTag, Track

    def observation(camera: str, track: int, embedding: tuple[float, float]) -> Any:
        return TrackObservation(
            key=TrackKey(camera, track),
            track=Track(
                track_id=track,
                box=np.asarray([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
                tag=FrameTag(camera_id=camera, frame_id=0),
                embedding=np.asarray(embedding, dtype=np.float32),
            ),
            frame_height=1080,
            frame_width=1920,
        )

    lines = [
        "# The REFERENCE's answer to scenarios/identity/*.txt. Emitted, never hand-edited:",
        "#   python scripts/emit_parity_golden.py --kind identity --scenario basic --emit-golden",
    ]
    for scenario in scenarios:
        assigner = GlobalIdAssigner(
            max_age=scenario.max_age,
            capacity=scenario.capacity,
            max_tracks=scenario.max_tracks,
            validate_every_step=True,
        )
        lines.append(f"scenario {scenario.name}")
        for instant in scenario.instants:
            observations = [
                observation(camera, track, embedding)
                for (camera, track), embedding in zip(
                    instant.keys, instant.embeddings, strict=True
                )
            ]
            result = assigner.assign(observations, instant.labels)
            answers = " ".join(
                f"{key.camera_id}#{key.track_id}={global_id}"
                for key, global_id in sorted(result.items())
            )
            lines.append(f"result {answers}".rstrip())
        lines.append(f"issued {assigner.issued}")
    return "\n".join(lines) + "\n"
