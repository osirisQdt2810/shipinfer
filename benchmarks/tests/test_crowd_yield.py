"""What a frame of the bench data actually yields — measured, because the premise was wrong.

T3b was opened on the belief that the stock ``person`` set gives ~1 detection per frame, so
C's crop fan-out (10-20 people per frame) had no case to win and needed synthetic crowd
frames. **On the real engine that belief does not hold**, and the mosaic built to fix it makes
matters worse at the grid its own CLI defaults to. Measured here, 4 frames each, yolo26n:

===================== ==================== =========================================
source                 detections/frame     note
===================== ==================== =========================================
single photo           13-18 (score>=0.35)  already the 10-20 the sizing assumes
mosaic 2x2 (4 photos)  18-20                marginally better
mosaic 3x3 (9)         12-17                no better than a single photo
mosaic 4x4 (16)        3-6                  **3x worse** -- the CLI's default grid
mosaic 4x4 at 4K       3-7                  not an output-resolution problem
===================== ==================== =========================================

The cliff is the detector's fixed 640x640 input: a 4x4 mosaic puts each source photo in a
~160px cell, and the people inside it fall under the model's minimum size. Composing more
people into a frame does not compose more *detectable* people into it.

Threshold sweep on the single photos, since a count means nothing without its score floor:
0.25 -> 16-20, 0.35 -> 13-18, 0.5 -> 8-11, 0.7 -> 0-5. At no sensible floor is it ~1.

So this file is the standing evidence for two claims T3b now rests on: the stock data already
represents the fan-out case, and a mosaic is not the way to get more of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.harness.crowd import compose_crowd_frames

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "model_repository" / "ship_detector" / "1" / "model.plan"
PHOTOS = REPO / "benchmarks" / "baseline" / "data" / "person"

#: The sizing the architecture is designed against (`.claude/CLAUDE.md`): 10-20 people per
#: frame. A 4x4 mosaic holds 16 photos, so a detector that finds most of them lands in band.
GRID, FRAMES, SCORE = 4, 4, 0.35

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not PLAN.exists(), reason=f"no engine at {PLAN}; run scripts/build_engines.py"
    ),
    pytest.mark.skipif(not PHOTOS.is_dir(), reason=f"no person photos at {PHOTOS}"),
]


def _letterboxed(path: Path, size: int = 640) -> np.ndarray:
    """The detector's own input contract: NCHW RGB in [0, 1], padded with YOLO gray."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    )
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.ascontiguousarray(
        np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    )


def _count(server, path: Path) -> int:
    """Rows of `output0` clearing the score bar. yolo26 is end-to-end, so NMS is done."""
    from shipinfer.core.request import InferenceRequest, RequestContext
    from shipinfer.core.types import Tensor

    request = InferenceRequest(
        model_name="ship_detector",
        inputs={"images": Tensor.from_numpy(_letterboxed(path))},
        context=RequestContext(camera_id="premise", frame_id=0),
    )
    boxes = server.infer(request).result(timeout=120).outputs["output0"].numpy()
    return int((boxes.reshape(-1, 6)[:, 4] >= SCORE).sum())


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
        """The blocker T3b was opened to remove does not exist for this dataset."""
        counts = [_count(server, path) for path in sorted(PHOTOS.glob("*.jpg"))[:FRAMES]]

        assert min(counts) >= 8, f"expected a crowd per photo, got {counts}"
        assert np.mean(counts) >= 10, (
            f"stock person photos yield {np.mean(counts):.1f} detections/frame; the ledger's "
            f"premise was ~1, which would mean C's fan-out case needs synthetic frames"
        )

    def test_a_four_by_four_mosaic_yields_fewer_not_more(self, server, tmp_path: Path) -> None:
        """The tool's own default grid, and it is the wrong direction.

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
