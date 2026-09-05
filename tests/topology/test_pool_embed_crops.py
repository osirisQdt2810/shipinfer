"""``PoolEmbed`` cuts one crop per detection and scatters the vectors back onto the rows.

The gap this closes, in one sentence: ``embed`` submitted the *whole payload* — the frame —
to a re-identification model whose input is ``3x256x128``, and filed the raw
``response.outputs`` under ``meta["vectors"]``. That is a ``{name: Tensor}`` dict, which is
exactly the form ``ShipvisionTrack._embeddings`` refuses by name ("an embedder's raw output
tensors are not an attribution"), so the chain could not reach ``track`` with appearance at
all. C4's build report said so: "C8 will need the embed→track scatter-back before the demo
chain runs". This file is that scatter-back, and what says it is aligned.

**Alignment is hand-checked, not shape-checked, and that is the whole point.** Every wrong
version of this code produces a mapping of the right size, with vectors of the right width,
covering rows that exist. An off-by-one, a mapping keyed by crop position instead of detection
index, a chunk boundary that re-based its indices — all four pass any assertion about shapes
and all four attach an appearance vector to the wrong object, a corruption with no exception
and no symptom short of a tracker that swaps identities. So the fake embedder here answers
with a vector that **encodes which crop it was given** (row ``i`` of the answer is
``[i, i, i, i]``), and every test says which detection row that vector must land on.

The class selection is deliberately non-contiguous — a ship, a person, a ship, a person — for
the same reason: a subset selected as ``rows[0:2]`` and a subset selected by label agree on a
sorted frame and disagree on a real one.

Offline throughout: :class:`~shipinfer.runtime.ops.NumpyImageOps` is the readable reference
implementation and touches no device, which is the whole reason the ops seam exists in the
shape it does. The end-to-end class at the bottom needs ``3rdparty/shipvision`` and skips
without it; everything above it runs on a checkout that has never seen the submodule.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError, InferenceError, ValidationError
from shipinfer.core.metrics import MetricsRegistry
from shipinfer.core.request import (
    InferenceRequest,
    InferenceResponse,
    RequestContext,
    ResponseFuture,
)
from shipinfer.core.types import DataType, Tensor, TensorSpec
from shipinfer.core.types.spec import DYNAMIC
from shipinfer.runners.inprocess import InprocessRunner
from shipinfer.runtime.ops import NumpyImageOps
from shipinfer.topology import (
    Caps,
    ChainItem,
    ChainSpec,
    ElementContext,
    RowIndexed,
    Topology,
)
from shipinfer.topology import bridge as bridge_module
from shipinfer.topology.elements.detections import Detections, Normalization
from shipinfer.topology.elements.pool import PoolEmbed, PoolSegment
from shipinfer.topology.elements.track import ShipvisionTrack
from tests.support.models import materialise

pytestmark = [pytest.mark.timeout(60)]

needs_shipvision = pytest.mark.skipif(
    not bridge_module.shipvision_available(),
    reason="shipvision.mot is not importable; the submodule is not checked out",
)

#: One crop's extent. Tall and not square, like the demo repository's two embedders, so a
#: transposed ``(height, width)`` is visible rather than symmetric.
CROP = (16, 8)

#: The source frame. Big enough that four boxes are genuinely different pixels.
FRAME_HW = (120, 200)

#: What the fake embedder answers with per row.
WIDTH = 4


# -- doubles ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeConfig:
    """The three things a crop element reads off a model handle, and nothing else.

    Structural on purpose: the element reaches them with :func:`getattr` through the
    :class:`~shipinfer.topology.base.ModelResolver` it was handed, so ``topology`` never learns
    that ``shipinfer.repository`` exists. ``max_batch_size`` is here and not in
    ``test_pool_detect_decode.py``'s copy because a detector submits one frame and an embedder
    submits N rows, so only one of the two can exceed a bound on its own.
    """

    input_specs: tuple[TensorSpec, ...] = ()
    output_specs: tuple[TensorSpec, ...] = ()
    #: Defaulted to ``0`` like ``ModelConfig``'s, so "the line is absent" is expressible.
    max_batch_size: int = 0

    @property
    def effective_max_batch_size(self) -> int:
        """``max_batch_size or 1`` — ``ModelConfig``'s rule, and what the assembler enforces."""
        return self.max_batch_size or 1


@dataclass(frozen=True)
class FakeArtifact:
    config: FakeConfig


#: What ``FakeEmbedder()`` declares unless a test says otherwise. A handle always carries a
#: config in the engine, and ``max_batch_size: 0`` would bound every test at one row.
DECLARED = FakeArtifact(FakeConfig(max_batch_size=8))


class FakeEmbedder:
    """A model handle whose answer says which crop each row came from.

    Row ``i`` of the response is ``[i, i, i, i]`` **within the request it belongs to**, so a
    chunked frame produces ``0,1,2,0,1`` and a test can tell a correct join from one that
    dropped or duplicated a chunk. ``requests`` keeps every request in submission order, which
    is what the chunking assertions read.

    **It enforces the row bound the engine enforces.** ``StackingBatcher.assemble`` refuses a
    request carrying more rows than ``effective_max_batch_size`` and the engine fails the
    future with it; a double that accepted any batch would pass a chunking bug that loses
    every crop of every crowded frame in production. ``artifact=None`` is a handle that
    declares nothing, which bounds nothing here.
    """

    def __init__(
        self,
        *,
        artifact: FakeArtifact | None = DECLARED,
        output: str = "embedding",
        answer: np.ndarray | None = None,
    ) -> None:
        self.artifact = artifact
        self.output = output
        self.answer = answer
        self.requests: list[InferenceRequest] = []

    @property
    def crops(self) -> list[np.ndarray]:
        """The crop batch of every request, in submission order."""
        return [next(iter(r.inputs.values())).numpy() for r in self.requests]

    def infer(self, request: InferenceRequest) -> ResponseFuture:
        self.requests.append(request)
        rows = next(iter(request.inputs.values())).shape[0]
        bound = getattr(
            getattr(self.artifact, "config", None), "effective_max_batch_size", None
        )
        if bound is not None and rows > bound:
            # Word for word `StackingBatcher.assemble`'s, delivered the way the engine
            # delivers it: on the future, from `_fail_batch`, not out of `infer`.
            future = ResponseFuture(request)
            future.set_exception(
                InferenceError(f"assembled batch of {rows} rows exceeds max_batch_size {bound}")
            )
            return future
        answer = (
            self.answer
            if self.answer is not None
            else np.tile(np.arange(1, rows + 1, dtype=np.float32)[:, None], (1, WIDTH))
        )
        future = ResponseFuture(request)
        future.set_result(
            InferenceResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                model_version=1,
                outputs={self.output: Tensor.from_numpy(answer)},
                context=request.context,
            )
        )
        return future


class FakePool:
    """A :class:`~shipinfer.topology.base.ModelResolver` over one model."""

    def __init__(self, **models: Any) -> None:
        self.models = models

    def get(self, name: str) -> Any:
        return self.models[name]


# -- helpers ----------------------------------------------------------------------------------


def frame(height: int = FRAME_HW[0], width: int = FRAME_HW[1]) -> Tensor:
    """One decoded frame as ``ChainFrameSink`` hands it over: batch-major ``(1, H, W, 3)``."""
    pixels = np.arange(height * width * 3, dtype=np.uint8).reshape(1, height, width, 3)
    return Tensor.from_numpy(pixels)


def mixed() -> Detections:
    """Four rows, ship/person/ship/person, in four visibly different places.

    Interleaved on purpose: the ship rows are ``(0, 2)``, which no slice expresses, so a
    subset selected by label and a subset selected by position cannot agree by accident.
    """
    return Detections(
        boxes=np.array(
            [[0, 0, 40, 40], [50, 10, 90, 70], [100, 20, 160, 90], [10, 60, 60, 110]],
            dtype=np.float32,
        ),
        scores=np.full(4, 0.9, dtype=np.float32),
        class_ids=np.array([8, 0, 8, 0], dtype=np.int32),
        labels=("ship", "person", "ship", "person"),
    )


def item(
    detections: Detections | None = None,
    *,
    payload: Tensor | None = None,
    caps: str = "bgr@cpu",
    frame_hw: tuple[int, int] | None = FRAME_HW,
    **meta: Any,
) -> ChainItem:
    """One item as ``PoolDetect`` hands it on: the source frame, plus what it filed."""
    filed: dict[str, Any] = dict(meta)
    if detections is not None:
        filed["detections"] = detections
    if frame_hw is not None:
        filed["frame_hw"] = frame_hw
    return ChainItem(
        RequestContext(camera_id="cam-1", frame_id=7),
        Caps.parse(caps),
        payload=frame() if payload is None else payload,
        meta=filed,
    )


def opened(
    embedder: FakeEmbedder,
    *,
    params: dict[str, Any] | None = None,
    ops: Any = "numpy",
    metrics: MetricsRegistry | None = None,
) -> PoolEmbed:
    """A ``PoolEmbed`` opened against ``embedder``, with numpy ops unless told otherwise.

    ``crop.size`` and ``output`` are always supplied: ``FakeEmbedder`` declares no artefact
    unless a test gives it one, and an element that cannot read either off the model refuses on
    purpose. The tests that care about *where* those come from build their element by hand.
    """
    declared: dict[str, Any] = {
        "input": "images",
        "output": "embedding",
        "crop": {"size": list(CROP)},
    }
    declared.update(params or {})
    element = PoolEmbed("embed", declared, model="person_embedder")
    element.open(
        ElementContext(
            models=FakePool(person_embedder=embedder),
            ops=NumpyImageOps() if ops == "numpy" else ops,
            metrics=metrics,
        )
    )
    return element


def which_crop(vector: np.ndarray) -> int:
    """The crop index :class:`FakeEmbedder` encoded into a row, so a test can name it.

    The fake writes ``i + 1`` and this subtracts the one back off. The offset is not
    decoration: ``shipvision`` refuses an all-zero embedding by design — it has no direction,
    so it sits at cosine 0 from every gallery entry, which is a plausible answer to every
    query rather than an obvious failure — so crop 0 cannot be encoded as a zero vector if the
    end-to-end class is to reach the real tracker with it.
    """
    return int(vector[0]) - 1


# -- the crop call ------------------------------------------------------------------------------


class TestItCropsOncePerFrameNotOncePerBox:
    """One ``crop_batch`` call for all N boxes.

    Not an optimisation detail: the ops interface is batched precisely so that a per-crop
    Python loop around a kernel launch is hard to write (CONVENTIONS 2.5), and at 10-20 people
    a frame across a thousand frames a second that loop is the difference between a shard that
    keeps up and one that does not. The counting spy is the only way to see the difference,
    because both versions produce the same crops.
    """

    def test_one_call_carries_every_box(self) -> None:
        calls: list[np.ndarray] = []
        ops = NumpyImageOps()
        real = ops.crop_batch

        def spy(image, boxes, dst_size, params):
            calls.append(boxes)
            return real(image, boxes, dst_size, params)

        ops.crop_batch = spy  # type: ignore[method-assign]
        element = opened(FakeEmbedder(), ops=ops)

        element.process(item(mixed()))

        assert len(calls) == 1, "four boxes must be four rows of one call, not four calls"
        assert calls[0].shape == (4, 4)

    def test_the_crops_are_the_model_s_input_extent(self) -> None:
        embedder = FakeEmbedder()
        element = opened(embedder)

        element.process(item(mixed()))

        assert embedder.crops[0].shape == (4, 3, *CROP)

    def test_they_are_cut_from_the_source_frame_not_the_letterbox(self) -> None:
        """``PoolDetect`` hands its payload on unchanged precisely so this element can crop the
        full-resolution frame: cropping a letterbox and resizing again is both slower and
        blurrier, and the boxes the detector filed are already in these pixels. The proof is
        that two different boxes give two different crops of the *frame* — a letterboxed
        payload would have been 16x8 already and every crop identical."""
        embedder = FakeEmbedder()
        element = opened(embedder)
        pixels = frame().numpy()[0]

        element.process(item(mixed()))

        crops = embedder.crops[0]
        assert not np.allclose(crops[0], crops[2]), "two boxes, two sets of pixels"
        # The top-left crop's first pixel is the frame's, normalised the way the element says.
        expected = NumpyImageOps().crop_batch(pixels, mixed().boxes[:1], CROP, Normalization())
        assert np.allclose(crops[:1], expected)


class TestThePixelScaleIsTheSlotsOwn:
    """``params: {crop: {normalize: ...}}`` is what the pixels were scaled by.

    The **second** silent-corruption axis of this element, and the one every other assertion in
    this file is blind to. Crops normalised with the wrong mean and std are the right rows, in
    the right order, at the right extent, keyed to the right detections; the engine answers
    without an error; and the only symptom is appearance matching that degrades weeks later.
    Before this class existed, replacing the resolved ``Normalization`` with the default at the
    ``crop_batch`` call left every other test in this file green — and, when review ran the
    same mutation, the entire offline tier with them. A knob the element parses, validates and
    documents was reaching the kernel unobserved.
    """

    #: Different from the default (mean 0, std 255, ``swap_rb`` True) in all three fields, so a
    #: crop cut with the default disagrees in every pixel rather than in a corner case.
    ODD = {"mean": [1, 2, 3], "std": [4, 5, 6], "swap_rb": False}

    #: The same three values as the element must resolve them.
    RESOLVED = Normalization(mean=(1.0, 2.0, 3.0), std=(4.0, 5.0, 6.0), swap_rb=False)

    def crop_params(self) -> dict[str, Any]:
        return {"crop": {"size": list(CROP), "normalize": self.ODD}}

    def test_the_declared_normalisation_reaches_the_crop(self) -> None:
        embedder = FakeEmbedder()
        element = opened(embedder, params=self.crop_params())
        pixels = frame().numpy()[0]

        element.process(item(mixed()))

        expected = NumpyImageOps().crop_batch(pixels, mixed().boxes, CROP, self.RESOLVED)
        assert np.allclose(embedder.crops[0], expected)

    def test_and_the_default_would_have_produced_other_pixels(self) -> None:
        """The other half of the test above. "The crops equal the reference" is also satisfied
        by an element that ignored the slot *and* a reference that did too, so this says the
        two normalisations are distinguishable in the first place."""
        pixels = frame().numpy()[0]
        boxes = mixed().boxes

        declared = NumpyImageOps().crop_batch(pixels, boxes, CROP, self.RESOLVED)
        default = NumpyImageOps().crop_batch(pixels, boxes, CROP, Normalization())

        assert not np.allclose(declared, default)

    def test_a_slot_that_declares_none_gets_the_documented_default(self) -> None:
        """``crop.normalize`` is optional and its default is
        :class:`~shipinfer.topology.elements.detections.Normalization`'s own — not the
        artefact's, because a model repository config has no normalisation section to read."""
        embedder = FakeEmbedder()
        element = opened(embedder)
        pixels = frame().numpy()[0]

        element.process(item(mixed()))

        expected = NumpyImageOps().crop_batch(pixels, mixed().boxes, CROP, Normalization())
        assert np.allclose(embedder.crops[0], expected)

    def test_a_zero_std_is_refused_at_open_not_divided_by(self) -> None:
        """The symptom otherwise is a crop batch of infinities that a re-ID engine accepts."""
        with pytest.raises(ConfigurationError, match="std"):
            PoolEmbed(
                "embed",
                {"crop": {"size": list(CROP), "normalize": {"std": [0, 1, 1]}}},
                model="person_embedder",
            ).open(
                ElementContext(
                    models=FakePool(person_embedder=FakeEmbedder()), ops=NumpyImageOps()
                )
            )


# -- the scatter-back ---------------------------------------------------------------------------


class TestTheVectorsLandOnTheRowsTheyCameFrom:
    """Hand-checked alignment. See the module docstring for why nothing here asserts a shape."""

    def test_every_row_is_covered_when_no_classes_are_declared(self) -> None:
        element = opened(FakeEmbedder())

        emitted = element.process(item(mixed()))

        vectors = emitted.meta["vectors"]
        assert sorted(vectors) == [0, 1, 2, 3]
        assert [which_crop(vectors[row]) for row in (0, 1, 2, 3)] == [0, 1, 2, 3]

    def test_a_class_subset_maps_crop_position_to_detection_index(self) -> None:
        """The ship rows are 0 and 2, and they are crops 0 and 1. Anything that keyed the
        mapping by crop position would answer ``{0: ..., 1: ...}`` and attach the second ship's
        appearance to the first person — which is the corruption this element exists to make
        impossible."""
        element = opened(FakeEmbedder(), params={"classes": ["ship"]})

        emitted = element.process(item(mixed()))

        vectors = emitted.meta["vectors"]
        assert sorted(vectors) == [0, 2], "the ship rows, not the first two rows"
        assert which_crop(vectors[0]) == 0
        assert which_crop(vectors[2]) == 1

    def test_the_other_subset_is_the_other_two_rows(self) -> None:
        """The mirror, so the test above is a check on selection and not on the ordering of a
        frame that happens to start with a ship."""
        element = opened(FakeEmbedder(), params={"classes": ["person"]})

        emitted = element.process(item(mixed()))

        vectors = emitted.meta["vectors"]
        assert sorted(vectors) == [1, 3]
        assert which_crop(vectors[1]) == 0
        assert which_crop(vectors[3]) == 1

    def test_only_the_selected_boxes_are_cropped(self) -> None:
        """The selection has to reach the *crop*, not just the mapping: cropping four boxes
        and filing two would pay the whole GPU bill for the half of it that is used."""
        embedder = FakeEmbedder()
        element = opened(embedder, params={"classes": ["ship"]})

        element.process(item(mixed()))

        assert embedder.crops[0].shape[0] == 2
        expected = NumpyImageOps().crop_batch(
            frame().numpy()[0],
            np.ascontiguousarray(mixed().boxes[[0, 2]]),
            CROP,
            Normalization(),
        )
        assert np.allclose(embedder.crops[0], expected)

    def test_two_embedders_in_series_merge_their_coverage_rather_than_replacing_it(
        self,
    ) -> None:
        """Two embedders on **one branch**, the second declared ``after:`` the first, so the
        second finds the first's mapping in ``item.meta``. ``derive`` merges metadata by key,
        so a plain assignment by the second one would replace the first one's mapping
        wholesale — every ship reaching ``track`` with no appearance, on a chain whose
        per-element counters both say they ran.

        This is one of the two ways two embedders meet. The other one — parallel branches
        rejoining at ``track`` — is :class:`TestTheShippedChainRunsThemInParallel`, whose
        merge happens in the runner and never reaches this method at all.
        """
        ships = opened(FakeEmbedder(), params={"classes": ["ship"]})
        people = opened(FakeEmbedder(), params={"classes": ["person"]})

        emitted = people.process(ships.process(item(mixed())))

        vectors = emitted.meta["vectors"]
        assert sorted(vectors) == [0, 1, 2, 3], "both embedders' rows, not the last one's"
        assert [which_crop(vectors[row]) for row in (0, 2)] == [0, 1], "the ship pair"
        assert [which_crop(vectors[row]) for row in (1, 3)] == [0, 1], "the person pair"
        assert isinstance(
            vectors, RowIndexed
        ), "the series merge keeps the declaration, or a later fan-in would not union it"

    def test_two_embedders_in_series_covering_one_row_is_a_typed_refusal(self) -> None:
        """The series half of the fan-in's rule, and it answers the same way.

        Two slots declared ``after:`` one another with overlapping ``classes:`` both cover the
        ship rows. At the rejoin that is an ``InferenceError`` naming both slots; here it used
        to be ``{**existing, **covered}`` — last-writer-wins, per row, no exception and no
        counter, so the frame reaches ``track`` with an appearance vector chosen by declaration
        order. The chain-file mistake is identical in both compositions, so the answer is too.
        """
        first = opened(FakeEmbedder(), params={"classes": ["ship"]})
        second = opened(FakeEmbedder(), params={"classes": ["ship", "person"]})

        with pytest.raises(InferenceError) as raised:
            second.process(first.process(item(mixed())))

        message = str(raised.value)
        assert "vectors" in message, "the key"
        assert "'embed'" in message, "this slot"
        assert "an earlier slot" in message, "and the one already holding the row"
        assert "classes" in message, "where the fix is"

    def test_a_non_mapping_already_under_the_key_is_refused(self) -> None:
        """Merging into a raw output dict is not possible and overwriting it would hide the
        producer that needs fixing."""
        element = opened(FakeEmbedder())

        with pytest.raises(ValidationError, match="mapping"):
            element.process(item(mixed(), vectors=np.zeros((4, WIDTH), dtype=np.float32)))


# -- the empty frame ------------------------------------------------------------------------


class TestAFrameWithNothingToCropCostsNothing:
    """A quiet camera is the common case at this sizing, not an edge one.

    An empty crop batch handed to a model is a request that costs a queue slot, an instance
    slot and a round trip to be told nothing — 50 cameras of empty water at 20 fps is a
    thousand of them a second.
    """

    def test_zero_detections_submits_nothing(self) -> None:
        embedder = FakeEmbedder()
        element = opened(embedder)

        emitted = element.process(item(Detections.empty()))

        assert embedder.requests == []
        assert emitted.meta["vectors"] == {}

    def test_a_frame_holding_none_of_this_element_s_classes_submits_nothing(self) -> None:
        embedder = FakeEmbedder()
        element = opened(embedder, params={"classes": ["ship"]})
        people = Detections(
            boxes=np.array([[0, 0, 40, 40]], dtype=np.float32),
            scores=np.full(1, 0.9, dtype=np.float32),
            class_ids=np.zeros(1, dtype=np.int32),
            labels=("person",),
        )

        emitted = element.process(item(people))

        assert embedder.requests == []
        assert emitted.meta["vectors"] == {}

    def test_it_does_not_clobber_a_peer_s_coverage(self) -> None:
        """``embed_person`` sees a frame of ships and covers none of them. Filing its empty
        mapping over ``embed_ship``'s would be the two-embedder failure arriving through the
        quiet door."""
        ships = opened(FakeEmbedder(), params={"classes": ["ship"]})
        people = opened(FakeEmbedder(), params={"classes": ["person"]})
        only_ships = Detections(
            boxes=np.array([[0, 0, 40, 40], [50, 10, 90, 70]], dtype=np.float32),
            scores=np.full(2, 0.9, dtype=np.float32),
            class_ids=np.full(2, 8, dtype=np.int32),
            labels=("ship", "ship"),
        )

        emitted = people.process(ships.process(item(only_ships)))

        assert sorted(emitted.meta["vectors"]) == [0, 1]

    def test_covering_nothing_over_a_peer_hands_the_item_on_untouched(self) -> None:
        """Nothing to add and the peer's mapping already correct: the item flows on *itself*
        rather than through a ``derive()`` that would build a second meta dict and a second
        ``ChainItem`` to say what this one already says. An unchanged item is a legal thing to
        flow — it is what a false ``when:`` hands on — and the quiet camera is the common case
        at this sizing, so the allocation the docstring promises not to make is worth one
        assertion."""
        ships = opened(FakeEmbedder(), params={"classes": ["ship"]})
        people = opened(FakeEmbedder(), params={"classes": ["person"]})
        only_ships = Detections(
            boxes=np.array([[0, 0, 40, 40]], dtype=np.float32),
            scores=np.full(1, 0.9, dtype=np.float32),
            class_ids=np.full(1, 8, dtype=np.int32),
            labels=("ship",),
        )
        embedded = ships.process(item(only_ships))

        assert people.process(embedded) is embedded

    def test_with_no_peer_the_empty_mapping_is_still_filed(self) -> None:
        """The other side of it: "this element ran and covered no rows" and "this element never
        ran" are different facts, and only the first is evidence the chain is wired the way the
        operator thinks. So the key appears even when nothing filled it."""
        element = opened(FakeEmbedder(), params={"classes": ["ship"]})
        people = Detections(
            boxes=np.array([[0, 0, 40, 40]], dtype=np.float32),
            scores=np.full(1, 0.9, dtype=np.float32),
            class_ids=np.zeros(1, dtype=np.int32),
            labels=("person",),
        )
        arrived = item(people)

        emitted = element.process(arrived)

        assert emitted is not arrived
        assert emitted.meta["vectors"] == {}

    def test_an_empty_frame_is_not_an_error(self) -> None:
        element = opened(FakeEmbedder())

        emitted = element.process(item(Detections.empty()))

        assert emitted.meta["detections"] is not None


# -- chunking ---------------------------------------------------------------------------------


class TestACrowdedFrameIsChunkedAtTheModelsBatch:
    """A frame holds however many objects the detector found; an engine's plan is built at a
    fixed batch. Without this, one crowded frame becomes a single oversized request and *every*
    crop in it is lost — the failure ``pipeline/graph/objects.py::_chunks`` exists because of,
    reproduced here on the chain."""

    @staticmethod
    def crowd(count: int) -> Detections:
        boxes = np.array(
            [[i * 2, i, i * 2 + 30, i + 30] for i in range(count)], dtype=np.float32
        )
        return Detections(
            boxes=boxes,
            scores=np.full(count, 0.9, dtype=np.float32),
            class_ids=np.full(count, 8, dtype=np.int32),
            labels=("ship",) * count,
        )

    @staticmethod
    def bounded(limit: int) -> FakeEmbedder:
        return FakeEmbedder(
            artifact=FakeArtifact(
                FakeConfig(
                    input_specs=(TensorSpec("images", DataType.FP32, (3, *CROP)),),
                    output_specs=(TensorSpec("embedding", DataType.FP32, (WIDTH,)),),
                    max_batch_size=limit,
                )
            )
        )

    def test_seven_rows_at_a_bound_of_three_is_three_requests(self) -> None:
        embedder = self.bounded(3)
        element = opened(embedder)

        element.process(item(self.crowd(7)))

        assert [crops.shape[0] for crops in embedder.crops] == [3, 3, 1]

    def test_the_join_keeps_every_row_in_order(self) -> None:
        """Three chunks of ``0,1,2 / 0,1,2 / 0`` must scatter onto rows 0..6 in that order. A
        join that dropped a chunk, duplicated one, or re-based the indices per chunk produces a
        mapping of a plausible size covering rows that exist."""
        element = opened(self.bounded(3))

        emitted = element.process(item(self.crowd(7)))

        vectors = emitted.meta["vectors"]
        assert sorted(vectors) == list(range(7))
        assert [which_crop(vectors[row]) for row in range(7)] == [0, 1, 2, 0, 1, 2, 0]

    def test_a_frame_inside_the_bound_is_still_one_request(self) -> None:
        embedder = self.bounded(8)
        element = opened(embedder)

        element.process(item(self.crowd(7)))

        assert len(embedder.requests) == 1

    def test_a_config_with_no_max_batch_size_line_is_chunked_to_one_row(self) -> None:
        """``0`` is what a ``config.yaml`` gets by *omission*, and this engine reads it as a
        bound of **one row** (``ModelConfig.effective_max_batch_size``), not as "no bound".
        Read the other way, a 25-person frame is one 25-row request the assembler refuses and
        every crop of every crowded frame is lost -- the reviewer's `person_embedder`
        scenario, and slow beats gone."""
        embedder = self.bounded(0)
        element = opened(embedder)

        emitted = element.process(item(self.crowd(25)))

        assert [crops.shape[0] for crops in embedder.crops] == [1] * 25
        assert sorted(emitted.meta["vectors"]) == list(range(25))

    def test_a_handle_that_declares_no_config_at_all_is_chunked_to_one_row_too(self) -> None:
        """The fallback is the same number: a resolver that answers with no artefact says
        nothing about the bound, and one row per request is the only size always accepted."""
        embedder = FakeEmbedder(artifact=None)
        element = opened(embedder)

        element.process(item(self.crowd(5)))

        assert [crops.shape[0] for crops in embedder.crops] == [1] * 5

    def test_the_chunks_carry_the_same_frame_tag(self) -> None:
        """Reassembly, tracing and every log line group on the tag (ADR-002), so two requests
        for one frame must not become two frames."""
        embedder = self.bounded(3)
        element = opened(embedder)

        element.process(item(self.crowd(7)))

        tags = {(r.context.camera_id, r.context.frame_id) for r in embedder.requests}
        assert tags == {("cam-1", 7)}


# -- the refusals ------------------------------------------------------------------------------


class TestItRefusesWhatItCannotCrop:
    def test_it_declares_that_it_needs_image_ops(self) -> None:
        assert PoolEmbed.needs_image_ops is True

    def test_a_context_with_no_ops_is_a_typed_refusal_at_open(self) -> None:
        element = PoolEmbed(
            "embed",
            {"output": "embedding", "crop": {"size": list(CROP)}},
            model="person_embedder",
        )

        with pytest.raises(ConfigurationError) as caught:
            element.open(ElementContext(models=FakePool(person_embedder=FakeEmbedder())))

        message = str(caught.value)
        assert "embed" in message
        assert "get_thread_local_image_ops" in message, (
            "the message must name what to pass -- and that is the *thread-local* call, "
            "because one instance across four workers is one staging ring across four threads"
        )
        assert not element.is_open, "a failed open must not leave the element half-open"

    def test_closing_forgets_the_ops_so_a_reopen_takes_the_current_ones(self) -> None:
        embedder = FakeEmbedder()
        element = opened(embedder)
        element.close()

        assert element._ops is None
        with pytest.raises(ConfigurationError, match="image ops"):
            element.open(ElementContext(models=FakePool(person_embedder=embedder)))

    def test_a_device_resident_payload_is_refused_rather_than_downloaded(self) -> None:
        """The cap declaration stays ``nv12@gpu, tensor@gpu, bgr@cpu`` because phase D makes
        that chain work; what is refused is the *frame*, by name, with the phase that fixes it.
        Downloading six megabytes per frame here is the cost arch.md section 8 makes the caps
        refuse at load time."""
        element = opened(FakeEmbedder())
        walked = ChainItem(
            RequestContext(camera_id="cam-1", frame_id=1),
            Caps.parse("nv12@gpu"),
            payload="frame:cam-1:1",
            meta={"detections": mixed(), "frame_hw": FRAME_HW},
        )

        with pytest.raises(ValidationError) as caught:
            element.process(walked)

        assert "str" in str(caught.value)
        assert "DataPool" in str(caught.value)

    def test_the_caps_it_accepts_are_unchanged(self) -> None:
        assert PoolEmbed.accepts == ("nv12@gpu", "tensor@gpu", "bgr@cpu")
        assert PoolEmbed.produces == ("*@*",)

    def test_a_float_frame_is_refused_rather_than_truncated_to_black(self) -> None:
        element = opened(FakeEmbedder())
        floats = Tensor.from_numpy(np.zeros((1, *FRAME_HW, 3), dtype=np.float32))

        with pytest.raises(ValidationError, match="uint8"):
            element.process(item(mixed(), payload=floats))

    def test_a_chain_that_filed_no_detections_is_refused_by_name(self) -> None:
        element = opened(FakeEmbedder())

        with pytest.raises(ValidationError, match="detections"):
            element.process(item())

    def test_raw_model_outputs_under_detections_are_refused_by_type(self) -> None:
        """The same refusal ``track`` makes one element further on, made here so the frame
        fails where the wrong thing was filed rather than two elements later."""
        element = opened(FakeEmbedder())
        walked = ChainItem(
            RequestContext(camera_id="cam-1", frame_id=7),
            Caps.parse("bgr@cpu"),
            payload=frame(),
            meta={"detections": {"output0": object()}},
        )

        with pytest.raises(ValidationError, match="detections"):
            element.process(walked)

    def test_a_frame_the_boxes_were_not_decoded_into_is_refused(self) -> None:
        """The boxes are in the detector's source pixels, so cropping a differently sized frame
        takes the wrong pixels for every object at once — a corruption with no other symptom."""
        element = opened(FakeEmbedder())

        with pytest.raises(ValidationError, match="pixels"):
            element.process(item(mixed(), payload=frame(60, 100)))

    def test_a_chain_that_filed_no_frame_hw_is_not_second_guessed(self) -> None:
        """An element that filed detections without the extent is saying nothing about it, and
        inventing a disagreement out of an absence would refuse a working chain."""
        element = opened(FakeEmbedder())

        emitted = element.process(item(mixed(), frame_hw=None))

        assert sorted(emitted.meta["vectors"]) == [0, 1, 2, 3]

    def test_a_missing_output_names_what_the_model_did_send(self) -> None:
        element = opened(FakeEmbedder(output="features"))

        with pytest.raises(InferenceError, match="features"):
            element.process(item(mixed()))

    def test_a_row_count_that_is_not_the_crop_count_is_refused(self) -> None:
        """Refused rather than zipped to the shorter of the two: a scatter-back that silently
        drops the last object attaches every remaining vector correctly and loses one identity
        per frame with no counter anywhere."""
        element = opened(FakeEmbedder(answer=np.zeros((3, WIDTH), dtype=np.float32)))

        with pytest.raises(InferenceError, match="one row per crop"):
            element.process(item(mixed()))

    def test_a_classes_value_that_is_not_a_list_is_refused_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="classes"):
            PoolEmbed("embed", {"classes": "ship"}, model="person_embedder")

    def test_a_crop_block_that_is_not_a_mapping_is_refused_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="crop"):
            PoolEmbed("embed", {"crop": [256, 128]}, model="person_embedder")


# -- what the artefact says --------------------------------------------------------------------


class TestTheModelIsAskedBeforeTheSlotIsGuessedAt:
    """The artefact knows its own input extent and output name; the slot overrides for an
    engine whose input is dynamic. Nothing is guessed, for the reason ``PoolDetect`` states: a
    crop resized to the wrong extent is accepted by a dynamic-shape engine and answers with a
    vector computed from the wrong pixels."""

    @staticmethod
    def declaring(*, shape=(3, *CROP), output: str = "embedding") -> FakeEmbedder:
        return FakeEmbedder(
            artifact=FakeArtifact(
                FakeConfig(
                    input_specs=(TensorSpec("images", DataType.FP32, shape),),
                    output_specs=(TensorSpec(output, DataType.FP32, (WIDTH,)),),
                )
            )
        )

    def test_the_crop_extent_comes_from_the_declared_input(self) -> None:
        element = PoolEmbed("embed", {"input": "images"}, model="person_embedder")
        element.open(
            ElementContext(
                models=FakePool(person_embedder=self.declaring()), ops=NumpyImageOps()
            )
        )

        assert element._crop_size == CROP

    def test_the_output_name_comes_from_the_single_declared_output(self) -> None:
        element = PoolEmbed("embed", {"input": "images"}, model="person_embedder")
        element.open(
            ElementContext(
                models=FakePool(person_embedder=self.declaring(output="feat")),
                ops=NumpyImageOps(),
            )
        )

        assert element._output == "feat"

    def test_a_model_declaring_nothing_and_a_slot_that_does_not_either_is_refused(
        self,
    ) -> None:
        element = PoolEmbed("embed", {"input": "images"}, model="person_embedder")

        with pytest.raises(ConfigurationError, match="how big a crop"):
            element.open(
                ElementContext(
                    models=FakePool(person_embedder=FakeEmbedder()), ops=NumpyImageOps()
                )
            )

    def test_a_multi_output_model_must_be_told_which_output_holds_the_rows(self) -> None:
        embedder = FakeEmbedder(
            artifact=FakeArtifact(
                FakeConfig(
                    input_specs=(TensorSpec("images", DataType.FP32, (3, *CROP)),),
                    output_specs=(
                        TensorSpec("embedding", DataType.FP32, (WIDTH,)),
                        TensorSpec("logits", DataType.FP32, (10,)),
                    ),
                )
            )
        )
        element = PoolEmbed("embed", {"input": "images"}, model="person_embedder")

        with pytest.raises(ConfigurationError, match="one row per crop"):
            element.open(
                ElementContext(models=FakePool(person_embedder=embedder), ops=NumpyImageOps())
            )

    def test_a_crop_size_the_static_input_contradicts_is_refused_at_open(self) -> None:
        element = PoolEmbed(
            "embed", {"input": "images", "crop": {"size": [64, 64]}}, model="person_embedder"
        )

        with pytest.raises(ConfigurationError) as caught:
            element.open(
                ElementContext(
                    models=FakePool(person_embedder=self.declaring()), ops=NumpyImageOps()
                )
            )

        assert "64" in str(caught.value)

    def test_a_dynamic_input_is_what_the_override_exists_for(self) -> None:
        element = PoolEmbed(
            "embed", {"input": "images", "crop": {"size": [64, 64]}}, model="person_embedder"
        )
        element.open(
            ElementContext(
                models=FakePool(person_embedder=self.declaring(shape=(3, DYNAMIC, DYNAMIC))),
                ops=NumpyImageOps(),
            )
        )

        assert element._crop_size == (64, 64)

    def test_an_input_the_model_does_not_declare_is_refused_at_open(self) -> None:
        element = PoolEmbed("embed", {"input": "pixels"}, model="person_embedder")

        with pytest.raises(ConfigurationError, match="declares no such input"):
            element.open(
                ElementContext(
                    models=FakePool(person_embedder=self.declaring()), ops=NumpyImageOps()
                )
            )


# -- the payload and the neighbours ------------------------------------------------------------


class TestThePayloadIsHandedOnUntouched:
    """``produces: *@*`` is the precise claim that this element adds a metadata key and passes
    the payload through, so the loader resolves its outbound cap from its negotiated inbound
    one. Stamping a concrete cap here would relabel a ``bgr@cpu`` frame."""

    def test_the_frame_is_the_same_object(self) -> None:
        element = opened(FakeEmbedder())
        walked = item(mixed())

        emitted = element.process(walked)

        assert emitted.payload is walked.payload
        assert emitted.caps == walked.caps

    def test_the_upstream_metadata_survives(self) -> None:
        element = opened(FakeEmbedder())

        emitted = element.process(item(mixed()))

        assert emitted.meta["frame_hw"] == FRAME_HW
        assert len(emitted.meta["detections"]) == 4

    def test_the_segmenter_crops_like_an_embedder_now(self) -> None:
        """P6-SEGMENT-CROP: what kept ``PoolSegment`` whole-frame was its ``_finish``, not its
        crop half -- a segmentation engine emits mask *prototypes*, and one crop's mask is a
        fold over two outputs that the per-row scatter cannot express. ``_reduced`` is that
        fold, so the element is an ordinary crop element and inherits every declaration
        this file's subject makes."""
        assert PoolSegment.needs_image_ops is True
        assert PoolSegment.selects_rows is True
        assert PoolSegment.files_raw_response is False
        assert PoolSegment.meta_key == "masks"


# -- the segmenter's fold ------------------------------------------------------------------


#: The prototype bank's extent, and the coefficient count that goes with it.
PROTO_HW, COEFFS = (4, 2), 2
#: One prototype cell in crop pixels: ``16 x 8`` crop over a ``4 x 2`` bank.
CELL_PX = (CROP[0] * CROP[1]) / (PROTO_HW[0] * PROTO_HW[1])


class FakeSegmenter:
    """A YOLO-seg engine whose mask area says which crop it was given.

    Crop ``i`` gets a prototype plane whose first ``i + 1`` cells are positive and the rest
    negative, with coefficients ``[1, 0]`` — so the fold counts ``i + 1`` cells and the area
    is ``(i + 1) * CELL_PX``. The same reasoning as :class:`FakeEmbedder`: every wrong version
    of the scatter produces areas of the right dtype on rows that exist.
    """

    #: What the real `ship_segmenter` declares, at toy extents: two outputs, so the element's
    #: check that `params: {segment: {detections: ...}}` names one of them has something to
    #: check against.
    DECLARES = FakeArtifact(
        FakeConfig(
            output_specs=(
                TensorSpec("output0", DataType.FP32, (2, 6 + COEFFS)),
                TensorSpec("output1", DataType.FP32, (COEFFS, *PROTO_HW)),
            ),
            max_batch_size=8,
        )
    )

    def __init__(self, *, score: float = 0.9, bound: int | None = None) -> None:
        self.artifact = self.DECLARES if bound is None else replace_bound(self.DECLARES, bound)
        self.score = score
        self.requests: list[InferenceRequest] = []

    def infer(self, request: InferenceRequest) -> ResponseFuture:
        self.requests.append(request)
        count = next(iter(request.inputs.values())).shape[0]
        cells = PROTO_HW[0] * PROTO_HW[1]
        rows = np.zeros((count, 2, 6 + COEFFS), dtype=np.float32)
        protos = np.full((count, COEFFS, *PROTO_HW), -1.0, dtype=np.float32)
        for i in range(count):
            rows[i, 0, 4], rows[i, 0, 6] = self.score, 1.0  # score, then coefficient 0
            rows[i, 1, 4] = self.score / 2.0  # a weaker row, so the argmax is checkable
            protos[i, 0].reshape(-1)[: min(i + 1, cells)] = 1.0
        future = ResponseFuture(request)
        future.set_result(
            InferenceResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                model_version=1,
                outputs={
                    "output0": Tensor.from_numpy(rows),
                    "output1": Tensor.from_numpy(protos),
                },
                context=request.context,
            )
        )
        return future


def replace_bound(artifact: FakeArtifact, bound: int) -> FakeArtifact:
    """The same declaration at a different `max_batch_size`, so a chunk boundary is reachable."""
    config = artifact.config
    return FakeArtifact(
        FakeConfig(
            input_specs=config.input_specs,
            output_specs=config.output_specs,
            max_batch_size=bound,
        )
    )


def segmenter(model: FakeSegmenter, *, params: dict[str, Any] | None = None) -> PoolSegment:
    """A ``PoolSegment`` opened against ``model``, selecting ships unless told otherwise."""
    declared: dict[str, Any] = {
        "input": "images",
        "classes": ["ship"],
        "crop": {"size": list(CROP)},
    }
    declared.update(params or {})
    element = PoolSegment("segment", declared, model="ship_segmenter")
    element.open(ElementContext(models=FakePool(ship_segmenter=model), ops=NumpyImageOps()))
    return element


class TestTheSegmenterFoldsTwoOutputsIntoOneAreaPerRow:
    """P6-SEGMENT-CROP: the crop half is the embedder's, and ``_reduced`` is the rest.

    A YOLO-seg engine emits detection rows and a bank of mask prototypes, never a mask, so the
    quantity a record carries is one the model has no output for. The fold runs once per
    chunk, before the scatter, and what reaches ``meta["masks"]`` is one area per detection.
    """

    def test_each_area_lands_on_the_crop_it_was_computed_from(self) -> None:
        """Ships are rows 0 and 2, so crop 0 is row 0 and crop 1 is row 2 — not rows 0 and 1."""
        element = segmenter(FakeSegmenter())

        result = element.process(item(mixed()))

        assert result is not None
        assert dict(result.meta["masks"]).keys() == {0, 2}
        assert result.meta["masks"][0] == pytest.approx([1 * CELL_PX])
        assert result.meta["masks"][2] == pytest.approx([2 * CELL_PX])

    def test_the_areas_are_filed_as_row_indexed_so_a_fan_in_may_union_them(self) -> None:
        element = segmenter(FakeSegmenter())

        result = element.process(item(mixed()))

        assert result is not None
        assert isinstance(result.meta["masks"], RowIndexed)

    def test_a_crop_the_engine_found_nothing_in_reports_no_area(self) -> None:
        """The refusal :class:`InstanceMaskArea` exists for: below the floor the argmax row is
        noise, and its mask covers the plane."""
        element = segmenter(FakeSegmenter(score=0.01))

        result = element.process(item(mixed()))

        assert result is not None
        assert [tuple(value) for value in result.meta["masks"].values()] == [(0.0,), (0.0,)]

    def test_the_engine_is_asked_for_crops_and_not_for_the_frame(self) -> None:
        model = FakeSegmenter()
        element = segmenter(model)

        element.process(item(mixed()))

        (crops,) = [next(iter(r.inputs.values())).numpy() for r in model.requests]
        assert crops.shape == (2, 3, *CROP), "two ships, cut to the model's extent"

    def test_a_frame_with_no_selected_row_submits_nothing(self) -> None:
        """The quiet camera, which is the common case at this sizing."""
        model = FakeSegmenter()
        element = segmenter(model, params={"classes": ["vehicle"]})

        result = element.process(item(mixed()))

        assert model.requests == [], "no crop is no request"
        assert result is not None and result.meta["masks"] == {}, "and no area is claimed"

    def test_the_default_output_name_is_the_folds_own(self) -> None:
        """The key `_reduced` files under, pinned.

        Every assertion above reads `meta["masks"]`, which is keyed by detection row — so the
        output name round-trips through `_rows_of` and nothing sees it. It IS seen in the
        scatter-back refusal, and it reached review as
        `"<member 'name' of 'InstanceMaskArea' objects>"`: reading a `slots=True` dataclass's
        default off the class gives a descriptor, and `str()` of one does not raise.
        """
        assert segmenter(FakeSegmenter())._output == "mask_area_px"
        assert segmenter(FakeSegmenter(), params={"output": "hull_px"})._output == "hull_px"

    def test_a_misspelt_engine_output_is_refused_at_open(self) -> None:
        """`outpu0` would otherwise open, report the shard ready, and then fail every frame of
        every camera — the outage `_check_one_filler_per_row` argues load-time refusal beats."""
        with pytest.raises(ConfigurationError, match="names output 'outpu0'"):
            segmenter(FakeSegmenter(), params={"segment": {"detections": "outpu0"}})

    def test_both_cuts_are_settable_from_the_slot(self) -> None:
        """`score_threshold` and `mask_threshold` are deployment knobs, so both are reachable
        and an unknown third is refused rather than silently ignored."""
        element = segmenter(
            FakeSegmenter(), params={"segment": {"score_threshold": 0.4, "mask_threshold": 0.6}}
        )

        assert (element._fold.score_threshold, element._fold.mask_threshold) == (0.4, 0.6)
        with pytest.raises(ConfigurationError, match="does not know"):
            segmenter(FakeSegmenter(), params={"segment": {"scoer_threshold": 0.4}})

    def test_a_value_the_fold_refuses_is_a_typed_refusal(self) -> None:
        """A chain-file typo has to raise this project's own type.

        `float("high")` and `InstanceMaskArea.__post_init__`'s range check both raise a bare
        `ValueError`, which is not a `ShipInferError` — so a caller catching the project's
        vocabulary would miss a malformed slot and see it as an unhandled exception instead.
        """
        with pytest.raises(ConfigurationError, match="not a valid fold"):
            segmenter(FakeSegmenter(), params={"segment": {"mask_threshold": 1.5}})
        with pytest.raises(ConfigurationError, match="not a valid fold"):
            segmenter(FakeSegmenter(), params={"segment": {"score_threshold": "high"}})

    def test_a_frame_chunked_at_the_models_batch_folds_once_per_chunk(self) -> None:
        """The fold is per RESPONSE, so it has to run before the chunks are joined.

        Five ships at a bound of two is three requests, and the areas still have to land on
        rows 0..4 in order. Folding after the join would read three separate `(n, R, 6+M)`
        answers as one and attribute every area to the wrong ship.
        """
        model = FakeSegmenter(bound=2)
        element = segmenter(model, params={"classes": ["ship"]})
        crowd = Detections(
            boxes=np.array(
                [[i * 8, i, i * 8 + 30, i + 30] for i in range(5)], dtype=np.float32
            ),
            scores=np.full(5, 0.9, dtype=np.float32),
            class_ids=np.full(5, 8, dtype=np.int32),
            labels=("ship",) * 5,
        )

        result = element.process(item(crowd))

        assert [len(r.inputs["images"].numpy()) for r in model.requests] == [2, 2, 1]
        assert result is not None
        areas = {row: float(value[0]) for row, value in result.meta["masks"].items()}
        assert areas == {i: (i % 2 + 1) * CELL_PX for i in range(5)}, "per chunk, in order"

    def test_an_engine_with_one_output_is_refused_by_name(self) -> None:
        """A detection-only engine in a ``segment`` slot: the fold says which output is
        missing rather than filing an area computed from nothing."""
        element = segmenter(FakeSegmenter())
        response = InferenceResponse(
            request_id="r",
            model_name="ship_segmenter",
            model_version=1,
            outputs={"output0": Tensor.from_numpy(np.zeros((1, 2, 8), np.float32))},
            context=RequestContext(camera_id="cam-1", frame_id=7),
        )

        with pytest.raises(InferenceError, match="'output1' is missing"):
            element._reduced(response)


# -- metrics -------------------------------------------------------------------------------------


class TestTheFanOutIsCounted:
    """The fan-out is what says whether the embedder's queue depth is a scheduling problem or a
    crowded scene, and it is the one number nothing else in the chain reports."""

    @staticmethod
    def value(registry: MetricsRegistry, name: str) -> Any:
        return next(m for m in registry.collect() if m.name == name)

    def test_crops_per_frame_and_the_running_total_are_recorded(self) -> None:
        registry = MetricsRegistry()
        element = opened(FakeEmbedder(), metrics=registry)

        element.process(item(mixed()))
        element.process(item(mixed()))

        total = self.value(registry, "shipinfer_element_crops_total")
        assert total.value(element="embed") == 8

    def test_an_empty_frame_is_recorded_as_a_zero_not_skipped(self) -> None:
        """ "this camera has nobody in it" and "the embedder is not running" are the two things
        this histogram has to tell apart, and one that only moves when there is work cannot."""
        registry = MetricsRegistry()
        element = opened(FakeEmbedder(), metrics=registry)

        element.process(item(Detections.empty()))

        histogram = self.value(registry, "shipinfer_element_crops_per_frame")
        assert histogram.snapshot(element="embed") == (1, 0.0), "one frame, zero crops"

    def test_a_runner_that_offered_no_registry_counts_nothing_rather_than_minting_one(
        self,
    ) -> None:
        """A metric on a registry no exporter reads is worse than an absent one, because it
        reads as evidence (``ElementContext``)."""
        element = opened(FakeEmbedder())

        element.process(item(mixed()))

        assert element._metrics.per_frame is None
        assert element._metrics.rows is None


# -- the two branches, rejoining -----------------------------------------------------------------

#: The shape ``topology/ship_person.yaml`` now has: two embedders forked off ``detect``
#: (``embed_person`` carries ``after: detect`` in the shipped file too, precisely so it does
#: *not* follow ``embed_ship``) and rejoined at ``track``, each picking its rows with
#: ``params: classes:``. The shipped file used to guard the two with ``when: class == ...``
#: against a field nothing in the chain sets, so neither embedder ran and there was no rejoin
#: to merge; V148 part 1 converted it, and this fixture is what it arrived at.
#:
#: Written as a chain file rather than as two hand-built elements because the wiring is the
#: thing under test: the two branches never see each other's item, so ``_scatter``'s merge
#: cannot run and the union has to be taken at the rejoin.
TWO_BRANCHES = """
name: two_embedders
elements:
  decode: {impl: replay}
  detect: {impl: pool, model: ship_detector}
  embed_ship:
    impl: pool
    model: ship_embedder
    after: detect
    params: {input: images, output: embedding, crop: {size: [16, 8]}, classes: [ship]}
  embed_person:
    impl: pool
    model: person_embedder
    after: detect
    params: {input: images, output: embedding, crop: {size: [16, 8]}, classes: [person]}
  track:  {impl: shipvision, kind: track, after: [embed_ship, embed_person]}
  output: {impl: none}
"""


def two_branches() -> tuple[InprocessRunner, PoolEmbed, PoolEmbed]:
    """The loaded chain, with its two real ``PoolEmbed`` elements opened on fakes."""
    runner = InprocessRunner(Topology.from_spec(ChainSpec.from_yaml(TWO_BRANCHES)))
    opened_branches = []
    for slot, model in (("embed_ship", "ship_embedder"), ("embed_person", "person_embedder")):
        element = runner.topology.node(slot).element
        element.open(
            ElementContext(models=FakePool(**{model: FakeEmbedder()}), ops=NumpyImageOps())
        )
        opened_branches.append(element)
    return runner, opened_branches[0], opened_branches[1]


class TestTheShippedChainRunsThemInParallel:
    """Both embedders' vectors reach ``track`` on the wiring ``TWO_BRANCHES`` declares.

    The composition matters and this is why it is loaded from a spec rather than composed by
    hand: with ``embed_ship`` and ``embed_person`` both ``after: detect``, neither element ever
    sees the other's item, so :meth:`PoolEmbed._scatter`'s additive merge is never reached and
    the whole property rests on the runner's fan-in. A fan-in that took one branch's
    ``meta["vectors"]`` wholesale would drop the other's entirely — at the sizing in
    ``CLAUDE.md``, ~15 000 person crops a second cut, embedded on a GPU and discarded, every
    person arriving at the tracker with ``embedding=None``, with no exception and no counter
    to say so. Both elements ran; the loss is at the seam between them.

    The class name is aspirational and stays that way until C8b: the shipped
    ``topology/ship_person.yaml`` guards both embedders with ``when: class == ...`` and
    declares no ``classes:``, so on it neither runs. ``TWO_BRANCHES`` is the wiring the
    conversion has to produce, and it is what is under test here.
    """

    def test_what_an_embedder_files_is_declared_row_indexed(self) -> None:
        """The element says what shape it filed, and the fan-in reads that and nothing else.

        This is the whole reason two rejoining embedders union while two rejoining segmenters
        do not: ``PoolSegment`` files a model's raw ``{output name: Tensor}`` response under
        ``meta["masks"]``, which is a ``Mapping`` too. Sniffing the shape cannot tell them
        apart; declaring it can. Asserting the *type* rather than the contents is deliberate —
        the contents are pinned below, and it is the type the merge dispatches on.
        """
        _, ships, _ = two_branches()

        emitted = ships.process(item(mixed()))

        assert isinstance(emitted.meta["vectors"], RowIndexed)

    def test_the_tracker_receives_both_branches_vectors(self) -> None:
        """The four rows of the frame, from the two elements that covered two each."""
        runner, ships, people = two_branches()
        detected = item(mixed())

        # What the walk does at a fork: the *same* item to each branch, neither seeing the
        # other, and then one merge at the node they rejoin on.
        merged = runner._walker.inbound(
            runner.topology.node("track"),
            {"embed_ship": ships.process(detected), "embed_person": people.process(detected)},
        )

        assert merged is not None
        vectors = merged.meta["vectors"]
        assert sorted(vectors) == [0, 1, 2, 3], "both branches' rows, not one branch's"
        assert [which_crop(vectors[row]) for row in (0, 2)] == [0, 1], "the ship pair"
        assert [which_crop(vectors[row]) for row in (1, 3)] == [0, 1], "the person pair"

    def test_neither_branch_covers_the_other_s_rows(self) -> None:
        """The vacuity guard: the union is two disjoint halves and not one branch twice.

        Without this, a fan-in that returned the first contributor unchanged would still pass
        the test above on any chain where one embedder happened to cover everything.
        """
        _, ships, people = two_branches()
        detected = item(mixed())

        assert sorted(ships.process(detected).meta["vectors"]) == [0, 2]
        assert sorted(people.process(detected).meta["vectors"]) == [1, 3]


# -- end to end ---------------------------------------------------------------------------------


@needs_shipvision
class TestTheChainReachesTrackWithAppearance:
    """detect (already proven) -> embed (this element) -> track (C4), on the real tracker.

    The one assertion that matters: ``track`` *accepts* the mapping this element files and the
    appearance reaches ``shipvision``. C4's ``_embeddings`` refuses a raw ``{name: Tensor}``
    dict by design, so before this element the chain could not have produced a tracked object
    with an embedding at all — which is what C4's build report meant by "C8 will need the
    embed→track scatter-back before the demo chain runs".
    """

    @pytest.fixture
    def tracker(self) -> Iterator[ShipvisionTrack]:
        element = ShipvisionTrack(
            "track", {"algorithm": "bytetrack", "options": {"min_hits": 1, "max_age": 3}}
        )
        element.open(ElementContext())
        try:
            yield element
        finally:
            element.close()

    def test_every_embedded_row_reaches_the_tracker_with_its_own_vector(self, tracker) -> None:
        embedder = opened(FakeEmbedder())

        emitted = tracker.process(embedder.process(item(mixed())))

        tracks = emitted.meta["tracks"]
        assert len(tracks) == 4
        assert all(track.embedding is not None for track in tracks)

    def test_a_subset_embedder_leaves_the_other_rows_on_motion_alone(self, tracker) -> None:
        """Partial coverage is this chain's normal case, not its exception: ``embed_ship`` runs
        on the ship rows and the person rows track on geometry."""
        embedder = opened(FakeEmbedder(), params={"classes": ["ship"]})

        emitted = tracker.process(embedder.process(item(mixed())))

        covered = sorted(track.embedding is not None for track in emitted.meta["tracks"])
        assert covered == [False, False, True, True]

    def test_a_frame_this_embedder_covered_none_of_is_tracked_rather_than_refused(
        self, tracker
    ) -> None:
        """The empty mapping. ``embed_person`` seeing a frame of ships is an ordinary frame,
        and refusing it for being unremarkable would fail a whole camera."""
        embedder = opened(FakeEmbedder(), params={"classes": ["person"]})
        only_ships = Detections(
            boxes=np.array([[0, 0, 40, 40], [50, 10, 90, 70]], dtype=np.float32),
            scores=np.full(2, 0.9, dtype=np.float32),
            class_ids=np.full(2, 8, dtype=np.int32),
            labels=("ship", "ship"),
        )

        emitted = tracker.process(embedder.process(item(only_ships)))

        assert len(emitted.meta["tracks"]) == 2
        assert all(track.embedding is None for track in emitted.meta["tracks"])

    def test_both_branches_of_the_shipped_chain_reach_it_with_appearance(self, tracker) -> None:
        """The whole path, on the wiring the chain file declares: two embedders forked off
        ``detect``, the runner's fan-in, and the real tracker at the end of it.

        The assertion is per row and not a count: it is the person rows that the fan-in used
        to drop, and a tracker that reported four tracks with two embeddings would have looked
        exactly like the partial-coverage case above, which is legitimate.
        """
        runner, ships, people = two_branches()
        detected = item(mixed())

        merged = runner._walker.inbound(
            runner.topology.node("track"),
            {"embed_ship": ships.process(detected), "embed_person": people.process(detected)},
        )
        assert merged is not None

        tracks = tracker.process(merged).meta["tracks"]
        assert len(tracks) == 4
        assert [track.embedding is not None for track in tracks] == [True] * 4


class TestTheCropHalfWorksWithoutTheSubmoduleAtAll:
    """CI does not check ``3rdparty/shipvision`` out, and everything up to ``track`` must still
    run — which is the promise that makes the pure layers verifiable anywhere (CLAUDE.md).

    The absence is *arranged* rather than assumed, exactly as ``tests/topology/test_bridge.py``
    does it: ``None`` in :data:`sys.modules` is what CPython's import machinery reads as "known
    not importable", the same ``ImportError`` a missing submodule raises.
    """

    @pytest.fixture
    def masked(self) -> Iterator[None]:
        saved = {
            name: sys.modules.get(name) for name in list(sys.modules) if "shipvision" in name
        }
        for name in saved:
            sys.modules[name] = None  # type: ignore[assignment]
        sys.modules["shipvision"] = None  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.modules.pop("shipvision", None)
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_detect_to_embed_still_produces_an_aligned_mapping(self, masked) -> None:
        element = opened(FakeEmbedder())

        emitted = element.process(item(mixed()))

        assert [which_crop(emitted.meta["vectors"][row]) for row in range(4)] == [0, 1, 2, 3]


class TestOverARealEngine:
    """The reviewer's `person_embedder` scenario on the real assembler, not a double.

    `FakeEmbedder` now enforces the engine's row bound, but that bound is still a fixture's
    claim. This drives a real :class:`~shipinfer.engine.InferenceServer` (real TorchScript fixtures,
    `KIND_CPU`) over a config whose `max_batch_size:` line is absent — the omission that
    used to lose every crop of every multi-detection frame.
    """

    @pytest.fixture()
    def embedder_repository(self, tmp_path):
        # No `max_batch_size:` line: omission means 0, and 0 means the assembler bounds every
        # request at one row. A bare omission is refused at load (dynamic_batching defaults
        # enabled and demands a bound), so the reachable spelling turns batching off too.
        root = tmp_path / "embedder_repository"
        (root / "embedder" / "1").mkdir(parents=True)
        (root / "embedder" / "config.yaml").write_text(
            "name: embedder\n"
            "platform: pytorch\n"
            "dynamic_batching: {enabled: false}\n"
            "inputs:\n"
            "  - {name: images, data_type: FP32, dims: [3, 8, 8]}\n"
            "outputs:\n"
            "  - {name: embedding, data_type: FP32, dims: [16]}\n"
            "instance_groups:\n"
            "  - {kind: KIND_CPU, count: 1}\n"
        )
        materialise(root)
        return root

    def test_an_omitted_max_batch_size_loses_no_crop(self, embedder_repository) -> None:
        from shipinfer.core.settings import ServerSettings
        from shipinfer.engine import InferenceServer

        settings = ServerSettings(model_repository=embedder_repository)
        with InferenceServer(settings) as engine:
            element = PoolEmbed(
                "embed", {"input": "images", "output": "embedding"}, model="embedder"
            )
            element.open(ElementContext(models=engine, ops=NumpyImageOps()))
            try:
                emitted = element.process(
                    item(TestACrowdedFrameIsChunkedAtTheModelsBatch.crowd(3))
                )
            finally:
                element.close()

        vectors = emitted.meta["vectors"]
        assert sorted(vectors) == [0, 1, 2], "a crop was lost to the one-row bound"
        assert all(v.shape == (16,) for v in vectors.values())
