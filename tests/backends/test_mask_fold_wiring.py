"""Does the fold reach the engine, and does the bank stop coming home?

`test_mask_fold_parity.py` answers whether the arithmetic is right. This one answers the
question a correct fold can still fail: whether it is attached, whether the output it replaces
stops being fetched and advertised, and whether two chain slots can quietly disagree about it.

Offline: the bindings and the engine are fakes, and the fold runs on CPU torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from shipinfer.backends.tensorrt.backend import TensorRTBackend
from shipinfer.backends.tensorrt.engine import EngineIO, LoadedEngine
from shipinfer.core.errors import ConfigurationError
from shipinfer.core.types import DataType, Tensor
from shipinfer.topology.elements.masks import InstanceMaskArea

torch = pytest.importorskip("torch")

ROWS, COEFFS, PLANE = 4, 8, 10
DETECTIONS = EngineIO("output0", DataType.FP32, (8, ROWS, 6 + COEFFS), is_input=False)
PROTOTYPES = EngineIO("output1", DataType.FP32, (8, COEFFS, PLANE, PLANE), is_input=False)
IMAGES = EngineIO("images", DataType.FP32, (8, 3, 64, 64), is_input=True)


def spec(**overrides: Any) -> InstanceMaskArea:
    """The chain's own fold object — what travels down, since `topology` is a pure layer."""
    settings: dict[str, Any] = {
        "crop_hw": (640, 640),
        "detections": "output0",
        "prototypes": "output1",
        "name": "mask_area_px",
        "score_threshold": 0.25,
        "mask_threshold": 0.5,
    }
    settings.update(overrides)
    return InstanceMaskArea(**settings)


class FakeBindings:
    """Device tensors plus a record of which outputs were asked for."""

    def __init__(self) -> None:
        rng = np.random.default_rng(3)
        self.tensors = {
            "output0": torch.from_numpy(rng.normal(size=DETECTIONS.shape).astype(np.float32)),
            "output1": torch.from_numpy(rng.normal(size=PROTOTYPES.shape).astype(np.float32)),
        }
        self.tensors["output0"][:, :, 4] = 0.9  # every crop has an instance
        self.fetched: list[str] = []

    def device_tensor(self, name: str) -> Any:
        return self.tensors[name]

    def fetch_output(self, name: str, rows: int, stream: Any, *, async_copy: bool) -> Any:
        self.fetched.append(name)
        return self.tensors[name][:rows].numpy()

    def stage_input(self, name: str, array: Any, stream: Any, *, async_copy: bool) -> None:
        pass


class NullStream:
    def activate(self) -> Any:
        class _Ctx:
            def __enter__(self) -> None:
                return None

            def __exit__(self, *exc: object) -> bool:
                return False

        return _Ctx()


@dataclass
class _Config:
    name: str = "ship_segmenter"
    max_batch_size: int = 8
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Execution:
    async_transfers: bool = False


@dataclass
class _Context:
    #: No graph cache, so `_maybe_replay` takes the ordinary launch path.
    graphs: Any = None
    config: _Config = field(default_factory=_Config)
    execution: _Execution = field(default_factory=_Execution)
    instance_name: str = "ship_segmenter:0@cuda:0"


def backend_with_engine() -> tuple[TensorRTBackend, FakeBindings]:
    """A backend with its collaborators replaced but its own methods intact."""
    backend = object.__new__(TensorRTBackend)
    bindings = FakeBindings()
    backend._context = _Context()
    backend._loaded = LoadedEngine(
        engine=object(), io=(IMAGES, DETECTIONS, PROTOTYPES), path=None  # type: ignore[arg-type]
    )
    backend._bindings = bindings
    backend._ctx = object()
    backend._stream = NullStream()
    backend._fold = None
    # The enqueue is TensorRT's and has nothing to say about the fold; stubbed on the
    # INSTANCE so every other method under test is still the real one.
    backend._enqueue = lambda stream: None  # type: ignore[method-assign]
    backend._enqueues = 0
    return backend, bindings


class TestWhatTheBackendAdvertises:
    def test_without_a_fold_every_engine_output_is_advertised(self) -> None:
        backend, _ = backend_with_engine()

        assert [s.name for s in backend.output_specs] == ["output0", "output1"]

    def test_the_bank_is_replaced_by_the_area_at_the_end(self) -> None:
        """`TrtEngineAdapter`'s arrangement on the other plane: the reduced output stops
        being advertised and a width-1 one takes its place LAST, so every output is still
        `rows` long and the scatter above is unchanged."""
        backend, _ = backend_with_engine()
        backend.set_fold(spec())

        names = [s.name for s in backend.output_specs]

        assert names == ["output0", "mask_area_px"], "the bank is gone and the area is last"
        assert backend.output_specs[-1].shape == (1,), "one area a row"

    def test_a_fold_naming_an_output_the_engine_lacks_is_refused(self) -> None:
        """Unchecked it would fold whatever sat at that index — arithmetic on the wrong
        planes, which looks like an answer rather than an error."""
        backend, _ = backend_with_engine()

        with pytest.raises(ConfigurationError, match="does not have"):
            backend.set_fold(spec(prototypes="protos"))

    def test_a_fold_cannot_be_attached_before_the_engine_is_loaded(self) -> None:
        backend, _ = backend_with_engine()
        backend._loaded = None

        with pytest.raises(ConfigurationError, match="loaded engine"):
            backend.set_fold(spec())


class TestWhatExecuteDoesWithIt:
    def test_the_bank_is_not_copied_home_and_the_area_is(self) -> None:
        """The copy the fold exists to remove: a `(32, 160, 160)` bank is 3.1 MB a row and
        nothing above reads it once an area has been computed from it."""
        backend, bindings = backend_with_engine()
        backend.set_fold(spec())

        outputs = backend.execute({"images": Tensor.from_numpy(np.zeros((3,), np.float32))}, 3)

        assert bindings.fetched == ["output0"], "the prototype bank was never fetched"
        assert sorted(outputs) == ["mask_area_px", "output0"]
        assert outputs["mask_area_px"].numpy().shape == (3, 1)

    def test_without_a_fold_both_outputs_still_come_home(self) -> None:
        """The default has to be untouched, or attaching a fold somewhere would change what
        every other model returns."""
        backend, bindings = backend_with_engine()

        outputs = backend.execute({"images": Tensor.from_numpy(np.zeros((3,), np.float32))}, 3)

        assert bindings.fetched == ["output0", "output1"]
        assert sorted(outputs) == ["output0", "output1"]

    def test_the_area_is_the_number_the_readable_fold_computes(self) -> None:
        """Wiring and arithmetic checked together once, so a correctly attached fold that
        reads the wrong tensors cannot pass both files separately."""
        backend, bindings = backend_with_engine()
        backend.set_fold(spec())

        outputs = backend.execute({"images": Tensor.from_numpy(np.zeros((3,), np.float32))}, 3)

        expected = InstanceMaskArea(crop_hw=(640, 640))(
            {
                "output0": bindings.tensors["output0"][:3].numpy(),
                "output1": bindings.tensors["output1"][:3].numpy(),
            }
        )["mask_area_px"]
        np.testing.assert_allclose(outputs["mask_area_px"].numpy(), expected, rtol=0, atol=0)
