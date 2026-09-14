"""One camera's per-ROW track ids, as each plane maps a tracker's answer back onto detections.

The seam `tracker_options` in `known.py` names, and the case it never had. Both planes run the
same ByteTrack over the same boxes; what differs is the mapping:

* C++ `TrackerShard::update` reads `Track::last_match` -- the row the pool recorded when it
  applied the measurement.
* Python `elements/track.py::_attribute` re-derives it by IoU between the track's FILTERED box
  and the frame's boxes, cut at `attribution_iou`.

So the golden is ONE PLANE'S answer, and the C++ binary prints its own beside it: where they
differ, that difference IS the registered divergence, and the count is the number
`SHIPVISION-TRACK-LAST-MATCH` asks for before anyone builds the fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

SCENARIOS = Path(__file__).resolve().parent / "scenarios" / "tracking"
GOLDEN = Path(__file__).resolve().parent / "golden" / "tracking"

__all__ = ["GOLDEN", "SCENARIOS", "TrackingScenario", "load", "render_tracking"]


class TrackingScenario(NamedTuple):
    name: str
    #: One list of ``(x1, y1, x2, y2, score)`` per frame, in frame order.
    frames: list[list[tuple[float, float, float, float, float]]]


def load(name: str) -> list[TrackingScenario]:
    """Parse ``scenarios/tracking/<name>.scn``, or a path. Line-oriented like its siblings."""
    path = Path(name)
    if path.suffix != ".scn":
        path = SCENARIOS / f"{name}.scn"
    scenarios: list[TrackingScenario] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        head, *rest = stripped.split()
        if head == "scenario":
            scenarios.append(TrackingScenario(rest[0], []))
        elif head == "frame":
            if not scenarios:
                raise ValueError(f"{path}: a `frame` before any `scenario`")
            scenarios[-1].frames.append([])
        elif head == "box":
            if not scenarios or not scenarios[-1].frames:
                raise ValueError(f"{path}: a `box` before any `frame`")
            x1, y1, x2, y2, score = (float(value) for value in rest)
            scenarios[-1].frames[-1].append((x1, y1, x2, y2, score))
        else:
            raise ValueError(f"{path}: unknown line {stripped!r}")
    return scenarios


def _label(seen: dict[int, int], track_id: int) -> str:
    """This scenario's label for a track id, minted in first-seen order.

    A free function and not a closure over the loop's ``seen``: ruff's B023 is right that
    a closure over a loop variable is a trap, and the binding is the only thing it needed.
    """
    return str(seen.setdefault(int(track_id), len(seen) + 1))


def render_tracking(scenarios: list[TrackingScenario]) -> str:
    """Run every scenario through the real `track` ELEMENT and render the per-row ids.

    THE ELEMENT and not `TrackerShard` directly, because the attribution under test lives in
    the element: the shard answers with publishable TRACKS and only `_attribute` turns those
    back into rows. Driving the shard would compare the two planes' trackers, which agree.
    """
    import numpy as np

    from shipinfer.core.request import RequestContext
    from shipinfer.topology.base import Caps, ChainItem, ElementContext, ElementKind
    from shipinfer.topology.elements.detections import Detections
    from shipinfer.topology.registry import create_element

    lines = [
        "# One plane's per-row track ids for scenarios/tracking/*.scn. Emitted, never",
        "# hand-edited:",
        "#   python scripts/emit_parity_golden.py --kind tracking --scenario attribution"
        " --emit-golden",
        "# `-` is a row no track claimed. The C++ binary prints its own beside this one;",
        "# where they differ, that is `tracker_options` in known.py.",
    ]
    for scenario in scenarios:
        element = create_element(
            ElementKind.TRACK,
            "shipvision",
            "track",
            {"options": {"min_hits": 1, "max_age": 30}},
        )
        element.open(ElementContext(workers=1))
        # NORMALISED PER SCENARIO, never the literal id: `shipvision` mints track ids from one
        # PROCESS-WIDE counter, so the number depends on how many tests ran first and a golden
        # holding it would pass alone and fail in a suite. What the seam is about is WHICH ROW
        # carries an identity and whether it is the SAME one as before, and that survives.
        seen: dict[int, int] = {}
        try:
            lines.append(f"scenario {scenario.name}")
            for frame_id, boxes in enumerate(scenario.frames):
                detections = Detections(
                    boxes=np.asarray([list(box[:4]) for box in boxes], dtype=np.float32),
                    scores=np.asarray([box[4] for box in boxes], dtype=np.float32),
                    class_ids=np.zeros(len(boxes), dtype=np.int32),
                    labels=("person",) * len(boxes),
                )
                item = ChainItem(
                    context=RequestContext(
                        camera_id="cam0",
                        frame_id=frame_id,
                        captured_ns=1_000_000_000 * (frame_id + 1),
                        captured_unix_ns=1_700_000_000_000_000_000 + 50_000_000 * frame_id,
                    ),
                    caps=Caps.parse("bgr@cpu"),
                    payload=np.zeros((240, 320, 3), dtype=np.uint8),
                    meta={"detections": detections, "frame_hw": (240, 320)},
                )
                emitted = element.process(item)
                per_row: list[str] = ["-"] * len(boxes)
                tracks = emitted.meta.get("tracks") or ()
                rows = emitted.meta.get("track_rows") or ()
                for track, row in zip(tracks, rows, strict=True):
                    if 0 <= int(row) < len(per_row):
                        per_row[int(row)] = _label(seen, track.track_id)
                lines.append(f"frame {frame_id} ids " + " ".join(per_row))
        finally:
            element.close()
    return "\n".join(lines) + "\n"
