"""The device fold against a real TensorRT segmenter, which is the one thing fakes cannot say.

`tests/backends/test_mask_fold_parity.py` proves the arithmetic against the readable
reference, and `test_mask_fold_wiring.py` proves the plumbing against a fake `BindingSet`.
Neither can answer whether `bindings.device_tensor(name)` hands back what TensorRT actually
wrote, on the stream the engine ran on — so this runs ONE batch through the repository's own
`ship_segmenter` twice, with and without the fold attached, and compares.

The same batch both ways is the point: a fold that read the wrong tensor, or read them before
the kernels retired, would answer *something*, and only the host path says what.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.request import InferenceRequest
from shipinfer.core.types import Tensor
from shipinfer.topology.elements.masks import InstanceMaskArea

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "model_repository" / "ship_segmenter" / "1" / "model.plan"
CROP = (640, 640)
BATCH = 4


@pytest.fixture(scope="module")
def segmenter():
    """The repository's `ship_segmenter`, loaded alone.

    Module-scoped: a TensorRT engine load per test would be one load per assertion, and this
    file's whole point is that both arms see the SAME engine.
    """
    if not PLAN.is_file():
        pytest.skip(f"no engine at {PLAN}; build one with `python scripts/build_engines.py`")
    from shipinfer.core.settings import ServerSettings
    from shipinfer.engine import InferenceServer
    from tests.support.devices import a_test_device

    settings = ServerSettings(
        model_repository=REPO / "model_repository",
        load_all_models=False,
        startup_models=["ship_segmenter"],
        devices={"visible_gpus": [a_test_device()]},
        pipeline={"workers": 1},
    )
    server = InferenceServer(settings)
    try:
        server.start()
        yield server.get("ship_segmenter")
    finally:
        server.stop()


def crops(seed: int = 11) -> Tensor:
    """One batch of ship-shaped noise. The numbers need not be ships: both arms see these."""
    rng = np.random.default_rng(seed)
    return Tensor.from_numpy(rng.random((BATCH, 3, *CROP), dtype=np.float32))


def run(handle, batch: Tensor) -> dict[str, np.ndarray]:
    request = InferenceRequest(model_name="ship_segmenter", inputs={"images": batch})
    response = handle.infer(request).result(60.0)
    return {name: value.numpy() for name, value in response.outputs.items()}


def spec(score_threshold: float = 0.25, mask_threshold: float = 0.5) -> InstanceMaskArea:
    return InstanceMaskArea(
        crop_hw=CROP,
        name="mask_area_px",
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
    )


class TestTheFoldOnTheEngineItself:
    @pytest.fixture(autouse=True)
    def _detached(self, segmenter):
        """The model is module-scoped, so each arm starts from no fold and leaves none.

        Without it the second arm meets the first arm's fold and is refused — correctly,
        which is `test_two_slots_that_fold_it_differently_are_refused` one layer down.
        """
        segmenter.attach_fold(None)
        yield
        segmenter.attach_fold(None)

    @pytest.mark.parametrize("mask_threshold", [0.5, 0.995])
    def test_the_device_area_is_the_host_area(self, segmenter, mask_threshold: float) -> None:
        """The claim, end to end: same engine, same batch, the bank reduced in two places.

        TWO MASK CUTS, which is how `test_fold_wiring` keeps the C++ side honest and why it
        is needed here too: this batch is noise, so at a permissive cut every mask fills its
        crop and every area is 409 600 — a fold that never read the bank at all would agree.
        The arm below moves the cut until the answers separate.
        """
        batch = crops()
        fold = spec(score_threshold=0.0, mask_threshold=mask_threshold)

        segmenter.attach_fold(None)
        host = run(segmenter, batch)
        assert sorted(host) == ["output0", "output1"], "the bank comes home without a fold"
        expected = fold(host)["mask_area_px"]

        assert segmenter.attach_fold(fold) is True, "a TensorRT backend takes the fold"
        device = run(segmenter, batch)

        assert sorted(device) == ["mask_area_px", "output0"], "the bank stayed on the device"
        np.testing.assert_allclose(device["mask_area_px"], expected, rtol=1e-5, atol=1e-3)

    def test_the_two_cuts_are_two_answers(self, segmenter) -> None:
        """What makes the pair above non-vacuous: the fold reads the planes, so moving the
        cut moves the area. Agreeing at one threshold proves nothing on its own."""
        batch = crops()
        areas = []
        for mask_threshold in (0.5, 0.995):
            segmenter.attach_fold(None)
            segmenter.attach_fold(spec(score_threshold=0.0, mask_threshold=mask_threshold))
            areas.append(run(segmenter, batch)["mask_area_px"].copy())

        assert not np.allclose(areas[0], areas[1]), (
            f"the same areas at both cuts ({areas[0].ravel()[:2]} and "
            f"{areas[1].ravel()[:2]}) means the fold never read the prototype bank"
        )

    def test_the_production_floor_reports_no_area_for_a_crop_with_no_ship(
        self, segmenter
    ) -> None:
        """The other half of the same fact, on the same engine: at `score_threshold=0.25`
        this noise clears nothing, so the answer is 0 — and a fold that ignored the floor
        would report the area of whatever the strongest row happened to project."""
        segmenter.attach_fold(spec())

        areas = run(segmenter, crops(seed=5))["mask_area_px"]

        assert areas.shape == (BATCH, 1)
        assert float(areas.max()) == 0.0, "no crop of noise is a ship"
