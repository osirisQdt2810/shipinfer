"""``PoolDetect`` letterboxes the frame and decodes the rows back into source pixels.

The gap this closes, in one sentence: the chain's detector filed ``response.outputs`` raw
under ``meta["boxes"]`` and nothing anywhere letterboxed anything — ``ChainFrameSink`` submits
the frame at whatever size it was decoded (``runners/frames.py``) — so the rows a detector
returned were in the pixels of an input nobody had produced, and there was no key a tracker
could read. ``meta["detections"]`` is that key, and this file is what says the numbers in it
are right.

**Every box here is hand-computed, and that is deliberate.** The arithmetic is four lines and
each of them has a plausible wrong version: applying the scale before subtracting the pad,
subtracting the pad twice, re-deriving the scale from the shapes instead of using the one the
letterbox reported. All four wrong versions produce boxes of a believable size in a believable
place, so a test that asserted shapes would pass against every one of them. The source frames
are sized so the scale is not 1 and the pad is not 0, because a 1:1 letterbox hides exactly the
mistakes this is for.

The geometry, once, for the 200x400 frame every test below uses against a 100x100 model:

    scale = min(100/200, 100/400) = 0.25      pad = (pad_x, pad_y) = (0, 25)

so a row at ``[x1, y1, x2, y2]`` in model pixels decodes to
``[(x1 - 0) * 4, (y1 - 25) * 4, (x2 - 0) * 4, (y2 - 25) * 4]``.

Offline throughout: :class:`~shipinfer.runtime.ops.NumpyImageOps` is the readable reference
implementation and touches no device, which is the whole reason the ops seam exists in the
shape it does.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError, InferenceError, ValidationError
from shipinfer.core.request import (
    InferenceRequest,
    InferenceResponse,
    RequestContext,
    ResponseFuture,
)
from shipinfer.core.settings import PipelineSettings
from shipinfer.core.types import DataType, Tensor, TensorSpec
from shipinfer.core.types.spec import DYNAMIC
from shipinfer.runtime.ops import NormalizeParams, NumpyImageOps
from shipinfer.topology import Caps, ChainItem, ElementContext
from shipinfer.topology.elements.detections import DecodeParams, Detections, Normalization
from shipinfer.topology.elements.pool import PoolDetect

#: The model input every test letterboxes to. Square and small, so the scale and the pad are
#: both non-trivial for a 200x400 frame and both easy to write down.
DST = (100, 100)

#: The source frame's ``(height, width)``. 2:1, so the letterbox pads on Y and not on X.
FRAME_HW = (200, 400)

#: What that combination produces, and what the decode has to undo. Written here rather than
#: recomputed in each test: a test that derived these from the same expression the code uses
#: would agree with the code and prove nothing.
SCALE = 0.25
PAD = (0.0, 25.0)


def frame(height: int = FRAME_HW[0], width: int = FRAME_HW[1]) -> Tensor:
    """One decoded frame as ``ChainFrameSink`` hands it over: batch-major ``(1, H, W, 3)``."""
    pixels = np.arange(height * width * 3, dtype=np.uint8).reshape(1, height, width, 3)
    return Tensor.from_numpy(pixels)


def rows(*values: tuple[float, float, float, float, float, int]) -> Tensor:
    """A detector's output for one frame: ``(1, R, 6)`` of ``[x1,y1,x2,y2,score,class]``."""
    return Tensor.from_numpy(np.asarray([list(v) for v in values], dtype=np.float32)[None])


def item(payload: Tensor | None = None, *, caps: str = "bgr@cpu") -> ChainItem:
    return ChainItem(
        RequestContext(camera_id="cam-1", frame_id=7),
        Caps.parse(caps),
        payload=frame() if payload is None else payload,
    )


@dataclass(frozen=True)
class FakeConfig:
    """The two lists ``PoolDetect`` reads off a model handle, and nothing else.

    Structural on purpose: the element reaches them with :func:`getattr` through the
    :class:`~shipinfer.topology.base.ModelResolver` it was handed, so ``topology`` never learns
    that ``shipinfer.repository`` exists. This class being three lines is the measure of how
    narrow that reach is.
    """

    input_specs: tuple[TensorSpec, ...] = ()
    output_specs: tuple[TensorSpec, ...] = ()


@dataclass(frozen=True)
class FakeArtifact:
    config: FakeConfig


class FakeDetector:
    """A model handle: answers with a fixed output, records what it was asked.

    ``release`` is a barrier a test can arm to hold one frame inside ``infer`` while another
    walks past it — the only way to catch per-frame geometry that was stored on the element.
    """

    def __init__(
        self,
        outputs: dict[str, Tensor] | None = None,
        *,
        artifact: FakeArtifact | None = None,
        release: threading.Barrier | None = None,
    ) -> None:
        self.outputs = outputs if outputs is not None else {}
        self.artifact = artifact
        self.release = release
        self.requests: list[InferenceRequest] = []
        self._lock = threading.Lock()

    def infer(self, request: InferenceRequest) -> ResponseFuture:
        with self._lock:
            self.requests.append(request)
        if self.release is not None:
            self.release.wait(timeout=5.0)
        future = ResponseFuture(request)
        future.set_result(
            InferenceResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                model_version=1,
                outputs=self.outputs,
                context=request.context,
            )
        )
        return future


class FakePool:
    """A :class:`~shipinfer.topology.base.ModelResolver` over one model."""

    def __init__(self, **models: FakeDetector) -> None:
        self.models = models

    def get(self, name: str) -> FakeDetector:
        return self.models[name]


def opened(
    detector: FakeDetector,
    *,
    decode: dict[str, Any] | None = None,
    ops: Any = "numpy",
    input_name: str = "images",
) -> PoolDetect:
    """A ``PoolDetect`` opened against ``detector``, with numpy ops unless told otherwise.

    ``dst_size`` is always supplied: ``FakeDetector`` declares no artefact unless a test gives
    it one, and an element that cannot read the extent off the model refuses on purpose. The
    tests that care about *where* the extent comes from build their element by hand.
    """
    params: dict[str, Any] = {
        "input": input_name,
        "decode": {"dst_size": list(DST), **(decode or {})},
    }
    element = PoolDetect("detect", params, model="ship_detector")
    element.open(
        ElementContext(
            models=FakePool(ship_detector=detector),
            ops=NumpyImageOps() if ops == "numpy" else ops,
        )
    )
    return element


class TestItRefusesWithoutTheOpsItWasPromised:
    """``needs_image_ops`` is a promise, and this is the half that keeps it honest.

    The declaration tells ``shipinfer run`` to resolve an implementation; ``_do_open`` has to
    refuse when the runner did not, or the two drift and the symptom is an ``AttributeError``
    on ``None`` at the first frame of a deploy instead of a message at start-up.

    **Why a refusal and not a numpy fallback.** ``topology`` may not import ``runtime``, so a
    fallback here would mean a second letterbox implementation living in the pure layer — a
    reimplementation of the thing the ops seam exists to own (CONVENTIONS 2.1), unfused, in
    Python, on the path of a thousand frames a second. A deployment that silently got it would
    read as a successful start-up and measure as a throughput cliff. The pool is handed in for
    the same reason and refuses the same way.
    """

    def test_it_declares_that_it_needs_them(self) -> None:
        assert PoolDetect.needs_image_ops is True

    def test_no_other_element_does(self) -> None:
        """The gating is only worth having if it is selective: a chain of mocks, or of `pool`
        embedders, must resolve no ops at all."""
        from shipinfer.topology.elements.mock import MockDetect
        from shipinfer.topology.elements.pool import PoolEmbed, PoolRecognize, PoolSegment

        assert [
            cls.needs_image_ops for cls in (PoolSegment, PoolEmbed, PoolRecognize, MockDetect)
        ] == [False, False, False, False]

    def test_a_context_with_no_ops_is_a_typed_refusal_at_open(self) -> None:
        element = PoolDetect(
            "detect", {"decode": {"dst_size": list(DST)}}, model="ship_detector"
        )

        with pytest.raises(ConfigurationError) as caught:
            element.open(ElementContext(models=FakePool(ship_detector=FakeDetector())))

        message = str(caught.value)
        assert "detect" in message
        assert "get_image_ops" in message, "the message must name what to pass"
        assert not element.is_open, "a failed open must not leave the element half-open"

    def test_closing_forgets_them_so_a_reopen_takes_the_current_ones(self) -> None:
        """A restarted shard resolves ops bound to its device; holding the previous cycle's
        would be a binding to a context that was released."""
        detector = FakeDetector()
        element = opened(detector, decode={"boxes_output": "boxes"})
        element.close()

        assert element._ops is None
        with pytest.raises(ConfigurationError, match="image ops"):
            element.open(ElementContext(models=FakePool(ship_detector=detector)))


class TestTheLetterboxIsRecordedNotRecomputed:
    def test_what_is_submitted_is_the_model_s_input_not_the_frame(self) -> None:
        detector = FakeDetector({"boxes": rows((0, 0, 1, 1, 0.9, 8))})
        element = opened(detector, decode={"boxes_output": "boxes"})

        element.process(item())

        (request,) = detector.requests
        submitted = request.inputs["images"]
        assert submitted.shape == (1, 3, *DST), "the frame was submitted unletterboxed"
        assert submitted.numpy().dtype == np.float32

    def test_the_pixels_are_the_ops_implementation_s_own_answer(self) -> None:
        """Not re-derived here: the element must call the seam, not reimplement it."""
        detector = FakeDetector({"boxes": rows((0, 0, 1, 1, 0.9, 8))})
        element = opened(detector, decode={"boxes_output": "boxes"})
        source = frame()

        element.process(item(source))

        expected = NumpyImageOps().letterbox_batch(
            [source.numpy()[0]], DST, NormalizeParams(), pad_value=114
        )
        np.testing.assert_array_equal(
            detector.requests[0].inputs["images"].numpy(), expected.tensor
        )
        assert float(expected.scales[0]) == SCALE
        assert tuple(float(p) for p in expected.pads[0]) == PAD

    def test_the_geometry_is_per_frame_and_not_kept_on_the_element(self) -> None:
        """Two frames of different sizes, in flight at once, decoded correctly each.

        One element instance is shared by every pipeline worker, so scale and pad stored as
        attributes would be overwritten by whichever frame reached ``_prepare`` last — and the
        result is boxes computed from the wrong letterbox, with no exception and no symptom
        short of a tracker that swaps identities. The barrier makes the interleaving certain
        rather than likely: neither frame can leave ``infer`` until both have entered it, so
        the second one's pre-processing has definitely happened before the first one decodes.
        """
        detector = FakeDetector(
            {"boxes": rows((10, 30, 50, 45, 0.9, 8))}, release=threading.Barrier(2)
        )
        element = opened(detector, decode={"boxes_output": "boxes"})
        results: dict[str, ChainItem] = {}

        def walk(name: str, payload: Tensor) -> None:
            results[name] = element.process(item(payload))

        threads = [
            threading.Thread(target=walk, args=("wide", frame(200, 400))),
            threading.Thread(target=walk, args=("square", frame(100, 100))),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)

        # scale 0.25, pad (0, 25)          -> ((10-0)*4, (30-25)*4, (50-0)*4, (45-25)*4)
        np.testing.assert_allclose(
            results["wide"].meta["detections"].boxes, [[40.0, 20.0, 200.0, 80.0]]
        )
        # scale 1.0, pad (0, 0)            -> unchanged
        np.testing.assert_allclose(
            results["square"].meta["detections"].boxes, [[10.0, 30.0, 50.0, 45.0]]
        )
        assert results["wide"].meta["frame_hw"] == (200, 400)
        assert results["square"].meta["frame_hw"] == (100, 100)


class TestTheGeometryCarriesEverythingTheOpsReported:
    """``_prepare`` returns what the seam *said*, and re-derives none of it.

    ``extents`` is the member with no reader yet and it is carried anyway, because the seam is
    being defined here and the first consumer -- the element that crops out of the letterboxed
    frame -- would otherwise compute ``out_h`` from ``scale`` and disagree by a pixel:
    ``pad = (T - r) / 2`` is the same for ``r`` and ``r + 1`` whenever ``T - r`` is even, which
    is exactly why ``LetterboxResult`` reports it separately.
    """

    def test_prepare_carries_scale_pad_extents_and_the_source_size(self) -> None:
        element = opened(FakeDetector(), decode={"boxes_output": "boxes"})
        source = frame()

        _, geometry = element._prepare(item(source))

        reported = NumpyImageOps().letterbox_batch(
            [source.numpy()[0]], DST, NormalizeParams(), pad_value=114
        )
        assert geometry.frame_hw == FRAME_HW
        assert geometry.scale == SCALE
        assert geometry.pad == PAD
        assert geometry.extents == tuple(int(v) for v in reported.extents[0])
        # 200x400 at scale 0.25 is 50x100 written inside a 100x100 canvas -- and 50 is not
        # recoverable from `scale` and `pad` alone, which is the whole argument for the field.
        assert geometry.extents == (50, 100)


class TestTheFrameHasToBeTheFrameItPromises:
    def test_a_float_frame_is_refused_rather_than_truncated_to_black(self) -> None:
        """The dtype was documented and not checked. Every ``ImageOps`` implementation writes
        source pixels into a uint8 canvas, so a frame already scaled to 0-1 is *truncated*
        into it: an all-black letterbox, a detector that answers with nothing on every camera,
        and an empty scene as the only symptom."""
        element = opened(FakeDetector(), decode={"boxes_output": "boxes"})
        floats = Tensor.from_numpy(np.zeros((1, *FRAME_HW, 3), dtype=np.float32))

        with pytest.raises(ValidationError) as caught:
            element.process(item(floats))

        message = str(caught.value)
        assert "uint8" in message, "the message must name the dtype it needs"
        assert "detect" in message

    def test_the_uint8_frame_it_does_want_is_accepted(self) -> None:
        """The other half, so the check above is a check and not a refusal of everything."""
        element = opened(
            FakeDetector({"boxes": rows((10, 30, 50, 45, 0.9, 8))}),
            decode={"boxes_output": "boxes"},
        )

        assert element.process(item()).meta["detections"] is not None


class TestTheRowsBecomeSourcePixels:
    def test_two_rows_decode_to_the_boxes_written_on_this_test(self) -> None:
        """The whole point of the file, with the arithmetic done by hand.

        Removing either half of the un-letterbox — the pad subtraction or the scale division —
        turns this red and nothing else, which is what makes it worth having.
        """
        detector = FakeDetector(
            {
                "boxes": rows(
                    (10, 30, 50, 45, 0.9, 8),
                    (0, 25, 20, 45, 0.5, 0),
                )
            }
        )
        element = opened(detector, decode={"boxes_output": "boxes"})

        result = element.process(item())

        detections = result.meta["detections"]
        assert isinstance(detections, Detections)
        np.testing.assert_allclose(
            detections.boxes,
            [
                [40.0, 20.0, 200.0, 80.0],  # (10-0)*4, (30-25)*4, (50-0)*4, (45-25)*4
                [0.0, 0.0, 80.0, 80.0],  # ( 0-0)*4, (25-25)*4, (20-0)*4, (45-25)*4
            ],
        )
        np.testing.assert_allclose(detections.scores, [0.9, 0.5])
        assert detections.labels == ("ship", "person")
        assert detections.class_ids.tolist() == [8, 0]

    def test_a_box_that_decodes_outside_the_frame_is_clamped_to_it(self) -> None:
        """The frame's extent is the one the letterbox was computed from, so the clamp is
        meaningful rather than a no-op: a detector firing on the grey bars decodes to negative
        y, and a negative crop box is a crash three stages later."""
        detector = FakeDetector({"boxes": rows((90, 20, 120, 90, 0.8, 8))})
        element = opened(detector, decode={"boxes_output": "boxes"})

        result = element.process(item())

        # x: (90-0)*4 = 360, (120-0)*4 = 480 -> clamped to width 400
        # y: (20-25)*4 = -20 -> 0,  (90-25)*4 = 260 -> clamped to height 200
        np.testing.assert_allclose(
            result.meta["detections"].boxes, [[360.0, 0.0, 400.0, 200.0]]
        )

    def test_the_score_threshold_is_applied_and_the_drop_is_counted(self) -> None:
        """Before any crop is made: a crop that will be discarded still costs a resize and an
        embedding, and at 15 000 crops a second that is the difference between fitting on the
        GPUs and not."""
        detector = FakeDetector(
            {
                "boxes": rows(
                    (10, 30, 50, 45, 0.9, 8),
                    (10, 30, 50, 45, 0.1, 8),
                )
            }
        )
        element = opened(detector, decode={"boxes_output": "boxes", "score_threshold": 0.25})

        detections = element.process(item()).meta["detections"]

        assert len(detections) == 1
        assert detections.discarded == 1, "a dropped row must be a number, not a guess"

    def test_a_reported_count_truncates_the_padded_rows(self) -> None:
        """The trailing rows of a fixed-size output are undefined, not zero: reading them
        produces plausible boxes out of nothing."""
        detector = FakeDetector(
            {
                "boxes": rows(
                    (10, 30, 50, 45, 0.9, 8),
                    (10, 30, 50, 45, 0.8, 8),
                    (10, 30, 50, 45, 0.7, 8),
                ),
                "num_detections": Tensor.from_numpy(np.asarray([1], dtype=np.int32)),
            }
        )
        element = opened(
            detector,
            decode={"boxes_output": "boxes", "count_output": "num_detections"},
        )

        assert len(element.process(item()).meta["detections"]) == 1

    def test_an_empty_frame_is_an_empty_Detections_and_not_an_error(self) -> None:
        """A quiet camera is the common case. "No ships" and "the detector is dead" are two
        events, and an exception here would make the first one look like the second."""
        detector = FakeDetector({"boxes": Tensor.from_numpy(np.zeros((1, 0, 6), np.float32))})
        element = opened(detector, decode={"boxes_output": "boxes"})

        detections = element.process(item()).meta["detections"]

        assert isinstance(detections, Detections)
        assert detections.is_empty


class TestWhatLandsOnTheItem:
    def test_the_frame_extent_is_filed_beside_the_detections(self) -> None:
        """``track`` needs the extent the boxes are measured in, and must not re-derive it
        from a payload that may already have been dropped."""
        detector = FakeDetector({"boxes": rows((10, 30, 50, 45, 0.9, 8))})
        element = opened(detector, decode={"boxes_output": "boxes"})

        assert element.process(item()).meta["frame_hw"] == FRAME_HW

    def test_the_raw_rows_are_not_carried_as_well(self) -> None:
        """``meta["boxes"]`` is gone rather than kept beside the decoded key.

        Nothing in ``src/`` read it — it was ``response.outputs`` filed under a name — and
        keeping it would pin the whole output tensor alive for the rest of the walk so a
        future consumer could redo arithmetic this element has already done correctly. The
        mocks still file their own ``boxes`` list; that key is theirs, not this element's.
        """
        detector = FakeDetector({"boxes": rows((10, 30, 50, 45, 0.9, 8))})
        element = opened(detector, decode={"boxes_output": "boxes"})

        assert "boxes" not in element.process(item()).meta

    def test_the_payload_and_the_cap_are_handed_on_untouched(self) -> None:
        """A `pool` element adds a key; it does not relabel what it was given. The letterboxed
        tensor is what was *submitted*, not what travels on — the next element still needs the
        frame."""
        source = frame()
        detector = FakeDetector({"boxes": rows((10, 30, 50, 45, 0.9, 8))})
        element = opened(detector, decode={"boxes_output": "boxes"})

        result = element.process(item(source, caps="bgr@cpu"))

        assert result.payload is source
        assert result.caps == Caps.parse("bgr@cpu")

    def test_the_tag_rides_untouched(self) -> None:
        detector = FakeDetector({"boxes": rows((10, 30, 50, 45, 0.9, 8))})
        element = opened(detector, decode={"boxes_output": "boxes"})
        walked = item()

        result = element.process(walked)

        assert result.context is walked.context
        assert detector.requests[0].context is walked.context


class TestWhereTheGeometryComesFrom:
    """The model's ``config.yaml`` first, the slot's ``params:`` over it, a refusal under both.

    Nothing is guessed. A letterbox to the wrong extent is accepted by a dynamic-shape engine
    and makes every box on every camera wrong, so "default to 640x640" is the one answer that
    cannot be allowed — it turns a deployment mistake into a silent accuracy loss.
    """

    @staticmethod
    def _declaring(inputs: tuple[TensorSpec, ...], outputs: tuple[TensorSpec, ...]):
        return FakeDetector(
            {"output0": rows((10, 30, 50, 45, 0.9, 8))},
            artifact=FakeArtifact(FakeConfig(input_specs=inputs, output_specs=outputs)),
        )

    def test_the_model_s_declared_input_is_the_letterbox_target(self) -> None:
        detector = self._declaring(
            (TensorSpec("images", DataType.FP32, (3, 100, 100)),),
            (TensorSpec("output0", DataType.FP32, (300, 6)),),
        )
        element = PoolDetect("detect", model="ship_detector")
        element.open(
            ElementContext(models=FakePool(ship_detector=detector), ops=NumpyImageOps())
        )

        assert element._dst_size == DST
        assert element._boxes_output == "output0", "the single declared output wins"
        # And it is really used: the decode below only lands on these numbers at scale 0.25.
        np.testing.assert_allclose(
            element.process(item()).meta["detections"].boxes, [[40.0, 20.0, 200.0, 80.0]]
        )

    def test_the_slot_overrides_a_dynamic_input(self) -> None:
        """A deployment whose engine declares a dynamic input has to say, and this is where.

        ``-1`` is what a dynamic-shape engine declares, and it is the *only* thing the
        override may disagree with: a static declaration is a statement about the artefact,
        and the slot contradicting it is checked below rather than obeyed.
        """
        detector = self._declaring(
            (TensorSpec("images", DataType.FP32, (3, DYNAMIC, DYNAMIC)),),
            (TensorSpec("output0", DataType.FP32, (300, 6)),),
        )
        element = PoolDetect(
            "detect", {"decode": {"dst_size": [100, 100]}}, model="ship_detector"
        )
        element.open(
            ElementContext(models=FakePool(ship_detector=detector), ops=NumpyImageOps())
        )

        assert element._dst_size == DST

    def test_a_slot_that_contradicts_a_static_input_stops_the_deploy(self) -> None:
        """The cross-check the code lost when it moved out of ``pipeline/graph/stage.py``.

        A static engine refuses the wrong extent loudly at the first submission, so this
        refusal looks redundant — until the same chain runs against a **dynamic** engine one
        deployment over, which accepts a 100x100 frame for a detector trained at 640 and
        answers with boxes that are wrong on every camera. That is verbatim the failure
        ``_resolve_dst_size``'s docstring gives as its reason for existing, and this is the
        half that catches it while an operator is still watching.
        """
        detector = self._declaring(
            (TensorSpec("images", DataType.FP32, (3, 640, 640)),),
            (TensorSpec("output0", DataType.FP32, (300, 6)),),
        )
        element = PoolDetect(
            "detect", {"decode": {"dst_size": [100, 100]}}, model="ship_detector"
        )

        with pytest.raises(ConfigurationError) as caught:
            element.open(
                ElementContext(models=FakePool(ship_detector=detector), ops=NumpyImageOps())
            )

        message = str(caught.value)
        assert "100" in message and "640" in message, "the message must name both ends"
        assert "ship_detector" in message
        assert not element.is_open

    def test_an_input_the_model_does_not_declare_stops_the_deploy(self) -> None:
        """``params: {input: pixels}`` on a single-input model used to open and then fail
        per frame, because ``_resolve_dst_size`` falls back to "the single declared spec"
        when the name misses -- so the typo resolved a perfectly good extent and hid itself
        until the first frame of the deploy (CONVENTIONS 2.6)."""
        detector = self._declaring(
            (TensorSpec("images", DataType.FP32, (3, 100, 100)),),
            (TensorSpec("output0", DataType.FP32, (300, 6)),),
        )
        element = PoolDetect("detect", {"input": "pixels"}, model="ship_detector")

        with pytest.raises(ConfigurationError) as caught:
            element.open(
                ElementContext(models=FakePool(ship_detector=detector), ops=NumpyImageOps())
            )

        message = str(caught.value)
        assert "pixels" in message, "the message must name the input that was asked for"
        assert "images" in message, "and the ones the model does declare"
        assert not element.is_open

    def test_a_model_that_declares_no_inputs_is_not_second_guessed(self) -> None:
        """A handle with no artefact -- a test's fake, a backend that declares nothing -- has
        nothing to disagree with, and the slot's ``params:`` are then the only truth there is.
        Refusing here would make every element in this file unopenable."""
        element = opened(
            FakeDetector({"boxes": rows((0, 0, 1, 1, 0.9, 8))}),
            decode={"boxes_output": "boxes"},
        )

        assert element.is_open
        assert element._dst_size == DST

    def test_a_model_that_declares_nothing_usable_stops_the_deploy(self) -> None:
        detector = self._declaring((TensorSpec("x", DataType.FP32, (4,)),), ())
        element = PoolDetect("detect", model="ship_detector")

        with pytest.raises(ConfigurationError) as caught:
            element.open(
                ElementContext(models=FakePool(ship_detector=detector), ops=NumpyImageOps())
            )

        assert "dst_size" in str(caught.value), "the message must name the override"

    def test_an_ambiguous_output_stops_the_deploy_rather_than_being_guessed(self) -> None:
        detector = self._declaring(
            (TensorSpec("images", DataType.FP32, (3, 100, 100)),),
            (
                TensorSpec("a", DataType.FP32, (300, 6)),
                TensorSpec("b", DataType.FP32, (300, 6)),
            ),
        )
        element = PoolDetect("detect", model="ship_detector")

        with pytest.raises(ConfigurationError, match="boxes_output"):
            element.open(
                ElementContext(models=FakePool(ship_detector=detector), ops=NumpyImageOps())
            )

    def test_a_count_output_is_trusted_only_when_the_model_declares_it(self) -> None:
        """Guessing the name and finding nothing is indistinguishable from a model that
        reports no count, and the two demand opposite handling of the trailing rows."""
        without = self._declaring(
            (TensorSpec("images", DataType.FP32, (3, 100, 100)),),
            (TensorSpec("output0", DataType.FP32, (300, 6)),),
        )
        with_count = self._declaring(
            (TensorSpec("images", DataType.FP32, (3, 100, 100)),),
            (
                TensorSpec("output0", DataType.FP32, (300, 6)),
                TensorSpec("num_detections", DataType.INT32, (1,)),
            ),
        )

        def context(model: FakeDetector) -> ElementContext:
            return ElementContext(models=FakePool(ship_detector=model), ops=NumpyImageOps())

        plain = PoolDetect("detect", model="ship_detector")
        plain.open(context(without))
        counting = PoolDetect("detect", model="ship_detector")
        counting.open(context(with_count))

        assert plain._count_output is None
        assert counting._count_output == "num_detections"

    def test_a_bad_extent_is_refused_by_name(self) -> None:
        element = PoolDetect(
            "detect", {"decode": {"dst_size": [0, 640]}}, model="ship_detector"
        )

        with pytest.raises(ConfigurationError, match="dst_size"):
            element.open(
                ElementContext(
                    models=FakePool(ship_detector=FakeDetector()), ops=NumpyImageOps()
                )
            )


class TestTheRefusalsAreTypedAndNamed:
    def test_a_missing_output_names_what_the_model_did_return(self) -> None:
        detector = FakeDetector({"something_else": rows((0, 0, 1, 1, 0.9, 8))})
        element = opened(detector, decode={"boxes_output": "boxes"})

        with pytest.raises(InferenceError) as caught:
            element.process(item())

        assert "boxes" in str(caught.value)
        assert "something_else" in str(caught.value)

    def test_rows_that_are_not_one_frame_are_refused_rather_than_reshaped(self) -> None:
        """A detector whose output layout changed is a deployment error, and guessing at the
        new one would attach scores to coordinates."""
        detector = FakeDetector(
            {"boxes": Tensor.from_numpy(np.zeros((4, 6), dtype=np.float32))}
        )
        element = opened(detector, decode={"boxes_output": "boxes"})

        with pytest.raises(InferenceError, match=r"\(1, rows, 6\)"):
            element.process(item())

    def test_a_row_of_the_wrong_width_is_the_decode_s_own_refusal(self) -> None:
        detector = FakeDetector(
            {"boxes": Tensor.from_numpy(np.zeros((1, 4, 5), dtype=np.float32))}
        )
        element = opened(detector, decode={"boxes_output": "boxes"})

        with pytest.raises(ValidationError, match=r"x1,y1,x2,y2,score,class"):
            element.process(item())

    def test_a_payload_that_is_not_a_tensor_says_so_with_its_type(self) -> None:
        element = opened(FakeDetector(), decode={"boxes_output": "boxes"})
        walked = ChainItem(
            RequestContext(camera_id="cam-1", frame_id=1),
            Caps.parse("nv12@gpu"),
            payload="frame:cam-1:1",
        )

        with pytest.raises(ValidationError) as caught:
            element.process(walked)

        assert "str" in str(caught.value)
        assert "DataPool" in str(caught.value)

    def test_a_batch_of_frames_in_one_item_is_refused(self) -> None:
        """One chain item is one frame (``topology/base.py``); an element that quietly
        letterboxed only the first would drop the rest with no counter anywhere."""
        element = opened(FakeDetector(), decode={"boxes_output": "boxes"})
        batched = Tensor.from_numpy(np.zeros((2, 200, 400, 3), dtype=np.uint8))

        with pytest.raises(ValidationError, match="one frame"):
            element.process(item(batched))

    def test_a_decode_block_that_is_not_a_mapping_is_refused_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="decode"):
            PoolDetect("detect", {"decode": [640, 640]}, model="ship_detector")


class TestTheDefaultsMirrorTheSettingsTheyCannotImport:
    """Two literals in two packages that may not import each other, tied by a test.

    ``topology`` is pure, so the decode's defaults are written out rather than read from
    :class:`~shipinfer.core.settings.PipelineSettings` — and a literal that drifts from the
    settings key is a chain quietly thresholding at a different number than the deployment was
    tuned for. Exactly the assertion ``tests/runners/test_pool_element.py`` already makes for
    the stage timeout.
    """

    def test_the_decode_defaults_are_the_pipeline_settings_defaults(self) -> None:
        settings = PipelineSettings()
        default = DecodeParams()

        assert default.score_threshold == settings.score_threshold
        assert default.max_detections == settings.max_detections
        assert default.class_labels == settings.class_labels

    def test_the_normalisation_default_is_the_ops_default(self) -> None:
        """``Normalization`` is the structural twin of ``NormalizeParams``; a drift would feed
        a detector unnormalised pixels and lose accuracy with nothing raising."""
        ours, theirs = Normalization(), NormalizeParams()

        assert (ours.mean, ours.std, ours.swap_rb) == (theirs.mean, theirs.std, theirs.swap_rb)

    def test_a_zero_std_is_refused_rather_than_divided_by(self) -> None:
        with pytest.raises(ValidationError, match="non-zero"):
            Normalization(std=(255.0, 0.0, 255.0))

    def test_the_slot_can_override_every_decode_knob(self) -> None:
        element = PoolDetect(
            "detect",
            {
                "decode": {
                    "dst_size": list(DST),
                    "boxes_output": "boxes",
                    "score_threshold": 0.75,
                    "max_detections": 3,
                    "class_labels": {8: "vessel"},
                    "pad_value": 0,
                    "normalize": {"mean": [1.0, 2.0, 3.0], "std": [2.0, 2.0, 2.0]},
                }
            },
            model="ship_detector",
        )
        element.open(
            ElementContext(
                models=FakePool(
                    ship_detector=FakeDetector({"boxes": rows((0, 0, 1, 1, 0.9, 8))})
                ),
                ops=NumpyImageOps(),
            )
        )

        assert element._decode.score_threshold == 0.75
        assert element._decode.max_detections == 3
        assert element._decode.class_labels == {8: "vessel"}
        assert element._pad_value == 0
        assert element._normalize == Normalization(
            mean=(1.0, 2.0, 3.0), std=(2.0, 2.0, 2.0), swap_rb=True
        )
