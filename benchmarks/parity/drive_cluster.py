"""The whole cross-camera composition through the REFERENCE.

`drive_gate` and `drive_identity` pin the two stateful halves; this pins that they compose the
same way -- gate, gram, gated matcher, clusterer, assigner -- which is the part a port can get
right in every piece and still get wrong between them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

SCENARIOS = Path(__file__).resolve().parent / "scenarios" / "cluster"
GOLDEN = Path(__file__).resolve().parent / "golden" / "cluster"

__all__ = ["GOLDEN", "SCENARIOS", "ClusterScenario", "load", "render_cluster"]


class ClusterScenario(NamedTuple):
    name: str
    repeats: int
    #: ``(camera, track, x, y, box_height_px)`` per observation, one instant repeated.
    instant: list[tuple[str, int, float, float, float]]


def load(name: str) -> list[ClusterScenario]:
    path = Path(name)
    if path.suffix != ".txt":
        path = SCENARIOS / f"{name}.txt"
    scenarios: list[ClusterScenario] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        head, *rest = stripped.split()
        if head == "scenario":
            scenarios.append(ClusterScenario(rest[0], int(rest[1]), []))
            continue
        if head != "instant":
            raise ValueError(f"{path}: unknown line {stripped!r}")
        if not scenarios:
            raise ValueError(f"{path}: an `instant` before any `scenario`")
        for field in rest:
            key, embedding, height = field.split(":")
            camera, track = key.split("#")
            x, y = embedding.split(",")
            scenarios[-1].instant.append(
                (camera, int(track), float(x), float(y), float(height))
            )
    return scenarios


def render_cluster(scenarios: list[ClusterScenario]) -> str:
    """Run every scenario through `ClusterMTMCTracker` and render the global ids."""
    import numpy as np
    from shipvision.mtmc.frames import CameraTracks, FrameTrackCluster
    from shipvision.mtmc.tracker import ClusterMTMCTracker
    from shipvision.types import FrameTag, Track

    def track(camera: str, tid: int, x: float, y: float, height: float) -> Any:
        return Track(
            track_id=tid,
            box=np.asarray([0.0, 0.0, 40.0, height], dtype=np.float32),
            tag=FrameTag(camera_id=camera, frame_id=0),
            embedding=np.asarray([x, y], dtype=np.float32),
        )

    lines = [
        "# The REFERENCE's answer to scenarios/cluster/*.txt. Emitted, never hand-edited:",
        "#   python scripts/emit_parity_golden.py --kind cluster --scenario basic --emit-golden",
    ]
    for scenario in scenarios:
        tracker = ClusterMTMCTracker(validate_every_step=True)
        lines.append(f"scenario {scenario.name}")
        for instant in range(scenario.repeats):
            by_camera: dict[str, list[Any]] = {}
            for camera, tid, x, y, height in scenario.instant:
                by_camera.setdefault(camera, []).append(track(camera, tid, x, y, height))
            cluster = FrameTrackCluster(
                tuple(
                    CameraTracks(
                        tag=FrameTag(camera_id=camera, frame_id=instant),
                        tracks=tuple(tracks),
                        height=1080,
                        width=1920,
                    )
                    for camera, tracks in by_camera.items()
                )
            )
            answers = sorted(
                f"{result.track.tag.camera_id}#{result.track.track_id}="
                f"{-1 if result.global_id is None else int(result.global_id)}"
                for result in tracker.track(cluster)
            )
            lines.append("ids " + " ".join(answers))
        lines.append(f"identities {len(tracker.assigner)}")
    return "\n".join(lines) + "\n"
