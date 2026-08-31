# doc: long the measured table IS the deliverable; deleting it leaves the claim unsourced
"""What a frame of the bench data actually yields — measured, because the premise was wrong.

T3b was opened on the belief that the stock ``person`` set gives ~1 person per frame, so C's
crop fan-out (10-20 people per frame) had no case to win and needed synthetic crowd frames.
**On the real engine that belief does not hold**, and the mosaic built to fix it makes matters
worse at the grid its own CLI first defaulted to.

Every number here counts COCO ``person`` rows only (column 5 == 0). An earlier revision of
this file counted rows of ANY class, which on this data over-reported by ~8% -- four ``class
33`` rows in one photo and one ``class 60`` -- and, worse, could not have supported the claim
it was cited for: a car is not a crop. Measured 4 frames each, yolo26n, score >= 0.35, through
``NumpyImageOps.letterbox_batch`` so the tensor is the one the server would have sent:

===================== ==================== =========================================
source                 people/frame         note
===================== ==================== =========================================
single photo           9-19 (mean 15.2)     already the 10-20 the sizing assumes
mosaic 2x2 (4 photos)  15-23                marginally better
mosaic 3x3 (9)         11-17                no better than a single photo
mosaic 4x4 (16)        4-6                  **3x worse** -- was the CLI default; now 2
===================== ==================== =========================================

The cliff is the detector's fixed 640x640 input: a 4x4 mosaic puts each source photo in a
~160px cell, and the people inside it fall under the model's minimum size. Composing more
people into a frame does not compose more *detectable* people into it.

Threshold sweep on the single photos, since a count means nothing without its score floor:
0.25 -> 12-20, 0.35 -> 9-19, 0.5 -> 7-10, 0.7 -> 2-5. At no sensible floor is it ~1.

So this file is the standing evidence for two claims T3b now rests on: the stock data already
represents the fan-out case, and a mosaic is not the way to get more of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Pillow is declared in the `dev` extra, which CI installs, so this skip is a safety net for a
# lean install rather than the normal path -- if it ever starts firing in CI the extra broke.
pytest.importorskip(
    "PIL", reason="Pillow is needed to read the bench photos; pip install '.[dev]'"
)

from benchmarks.harness.crowd import compose_crowd_frames

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "model_repository" / "ship_detector" / "1" / "model.plan"
PHOTOS = REPO / "benchmarks" / "baseline" / "data" / "person"

#: The sizing the architecture is designed against (`.claude/CLAUDE.md`): 10-20 PEOPLE per
#: frame -- people, which is why every count here filters on the class column.
FRAMES, SCORE = 4, 0.35

#: COCO ``person``. ``PipelineSettings.class_labels`` maps ``{0: person, 8: ship}``, and those
#: are COCO ids because the engine is yolo26n over all 80 classes -- so a count that ignores
#: column 5 counts cars and handbags, which are not crops the fan-out pays for.
PERSON = 0

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not PLAN.exists(), reason=f"no engine at {PLAN}; run scripts/build_engines.py"
    ),
    pytest.mark.skipif(not PHOTOS.is_dir(), reason=f"no person photos at {PHOTOS}"),
]


def _letterboxed(path: Path, size: int = 640) -> np.ndarray:
    """The detector's input, built by the code that builds it in production.

    `NumpyImageOps.letterbox_batch` is the readable reference every fused kernel is asserted
    against, and `pipeline/graph/detect.py` feeds the engine through it with BGR source pixels
    and the default `NormalizeParams` (std 255, swap_rb). Hand-rolling this drifted: the copy
    used `int()` where the seam uses `round()` for the resized extent, so the tensor measured
    was not the tensor the server would have sent. For a file whose only product is a number
    about what the detector sees, the detector's own preprocessing is the only defensible one.
    """
    from PIL import Image

    from shipinfer.runtime.ops.base import NormalizeParams
    from shipinfer.runtime.ops.numpy_ops import NumpyImageOps

    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    return NumpyImageOps().letterbox_batch([bgr], (size, size), NormalizeParams()).tensor


def _rows(server, path: Path) -> np.ndarray:
    """`output0` rows clearing the score bar. yolo26 is end-to-end, so NMS is done."""
    from shipinfer.core.request import InferenceRequest, RequestContext
    from shipinfer.core.types import Tensor

    request = InferenceRequest(
        model_name="ship_detector",
        inputs={"images": Tensor.from_numpy(_letterboxed(path))},
        context=RequestContext(camera_id="premise", frame_id=0),
    )
    boxes = server.infer(request).result(timeout=120).outputs["output0"].numpy()
    rows = boxes.reshape(-1, 6)
    return rows[rows[:, 4] >= SCORE]


def _count(server, path: Path, class_id: int | None = PERSON) -> int:
    """Detections of one class, or of every class when ``class_id`` is None.

    Column 4 is the score and column 5 the class id (`topology/elements/detections.py`
    473 and 507). Ignoring column 5 was the defect that made this file's first answer wrong.
    """
    rows = _rows(server, path)
    if class_id is None:
        return len(rows)
    return int((rows[:, 5].astype(np.int32) == class_id).sum())


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """A repository holding only the detector.

    The real one also holds the segmenter and two embedders, whose plans are gitignored and
    built per machine; loading them here would make this check fail for want of a model it
    never asks a question of.
    """
    from shipinfer.core.settings import ServerSettings
    from shipinfer.engine import InferenceServer

    root = tmp_path_factory.mktemp("detector_only")
    (root / "ship_detector").symlink_to(REPO / "model_repository" / "ship_detector")
    with InferenceServer(ServerSettings(model_repository=root)) as running:
        yield running


class TestTheBenchDataAlreadyRepresentsTheFanOutCase:
    """The premise check T3b asked for, kept as a test because the answer was surprising."""

    def test_a_single_photo_already_yields_the_sizing_s_ten_to_twenty(self, server) -> None:
        """The blocker T3b was opened to remove does not exist for this dataset.

        PEOPLE, not detections: the count filters column 5 on :data:`PERSON`, because the
        fan-out Phase C pays for is person crops and ship crops, not rows.
        """
        counts = [_count(server, path) for path in sorted(PHOTOS.glob("*.jpg"))[:FRAMES]]

        assert min(counts) >= 8, f"expected a crowd of people per photo, got {counts}"
        assert np.mean(counts) >= 10, (
            f"stock person photos yield {np.mean(counts):.1f} PEOPLE/frame (counts {counts}); "
            f"the ledger's premise was ~1, which would mean C's fan-out case needs synthetic "
            f"frames"
        )

    def test_the_cliff_is_cell_size_not_mosaicing(self, server, tmp_path: Path) -> None:
        """The 2x2 and 3x3 rows of the table, so three of its four rows are asserted.

        2x2 is modestly BETTER than a single photo, which is why the flat claim "composing
        more people never composes more detectable ones" is wrong as stated and the docstring
        says cell size instead. The assertion is the ordering, not the bands: exact counts are
        data-dependent and would rot into a flaky gate.
        """
        grids = {
            grid: np.mean(
                [
                    _count(server, path)
                    for path in compose_crowd_frames(
                        PHOTOS, tmp_path / f"g{grid}", grid=grid, frames=FRAMES
                    )
                ]
            )
            for grid in (2, 4)
        }

        assert grids[4] < grids[2], (
            f"means by grid: {grids}. The 4x4 cliff is the table's headline; if this inverts, "
            f"the detector's input size changed and the table needs re-measuring"
        )

    def test_a_four_by_four_mosaic_yields_fewer_not_more(self, server, tmp_path: Path) -> None:
        """The grid this tool used to default to, and it is the wrong direction.

        Not a bug in the composer — it does exactly what it says — but the detector's input
        is a fixed 640x640, so sixteen photos land in ~160px cells and their people fall under
        the model's minimum size. This is the measurement that stops a bench run from citing
        mosaics as the fan-out source.
        """
        composed = compose_crowd_frames(PHOTOS, tmp_path / "crowd", grid=4, frames=FRAMES)
        singles = sorted(PHOTOS.glob("*.jpg"))[:FRAMES]

        mosaic = [_count(server, path) for path in composed]
        single = [_count(server, path) for path in singles]

        assert np.mean(mosaic) < np.mean(single), (
            f"mosaic {mosaic} vs single {single}: if this ever passes the other way, the "
            "detector's input size changed and the docstring's table needs re-measuring"
        )
