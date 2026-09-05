"""The ``pool`` element: the chain's client of the model pool.

Four kinds share one implementation (``detect``, ``segment``, ``embed``, ``recognize``), and
all four hold the same four promises:

* the model name is resolved **once, at open**, so a chain that names a model the pool does
  not have stops the deploy instead of failing on the first frame at 3 a.m.;
* the request carries the item's :class:`RequestContext` **by identity** — the tag reassembly
  and tracing group on (ADR-002);
* the outputs land under the kind's own metadata key, which is the vocabulary the downstream
  elements read;
* backpressure propagates. A full pool raises
  :class:`~shipinfer.core.errors.QueueFullError`; it never becomes an empty result, because
  "no ships in this frame" and "the detector is saturated" demand opposite responses.

The element under test is :class:`PoolRecognize` wherever one kind has to stand for the four:
it is the plainest of them, and the only one left that forwards its payload untouched. The
other three each replace both
of ``_do_process``'s hooks and need image ops to open at all: ``detect`` letterboxes its frame
and decodes the boxes back into source pixels, and ``embed`` and ``segment`` cut one crop per
detection (C8, P6-SEGMENT-CROP). Testing the shared behaviour through a subclass that overrides
it would be testing the override. ``tests/topology/test_pool_detect_decode.py`` and
``tests/topology/test_pool_embed_crops.py`` are where those three contracts live.

The pool is a **fake with one method**, which is the honest measure of how narrow
:class:`~shipinfer.topology.base.ModelResolver` is: ``get(name)``. That is why this file
needs no server, no engine and no driver.

It lives under ``tests/runners/`` rather than ``tests/topology/`` because the element is only
meaningful with a runner's :class:`ElementContext` behind it, and because it arrived with the
runner package.
"""

from __future__ import annotations

import textwrap
import time
from typing import ClassVar

import numpy as np
import pytest

from shipinfer.core.errors import (
    CapsMismatchError,
    ConfigurationError,
    ModelNotFoundError,
    QueueFullError,
    RequestTimeoutError,
    ValidationError,
)
from shipinfer.core.request import (
    InferenceRequest,
    InferenceResponse,
    RequestContext,
    ResponseFuture,
)
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor
from shipinfer.topology import (
    Caps,
    ChainItem,
    ChainSpec,
    Element,
    ElementContext,
    ElementKind,
    Topology,
    create_element,
    registry_for,
)
from shipinfer.topology.elements.pool import (
    _DEFAULT_INPUT,
    _DEFAULT_TIMEOUT_S,
    PoolDetect,
    PoolEmbed,
    PoolRecognize,
    PoolSegment,
)

#: kind -> (class, the metadata key its results are filed under)
KINDS = {
    ElementKind.DETECT: (PoolDetect, "detections"),
    ElementKind.SEGMENT: (PoolSegment, "masks"),
    ElementKind.EMBED: (PoolEmbed, "vectors"),
    ElementKind.RECOGNIZE: (PoolRecognize, "identities"),
}

#: The one that files the model's outputs raw and hands the payload on untouched — the shared
#: behaviour this file is about. The other three are deliberately absent: each replaces
#: ``_do_process``'s two hooks and each needs image ops, so none is a fair sample of the
#: shared code and none is openable without a runner that resolved an implementation.
#: ``detect`` letterboxes the frame and decodes the boxes back
#: (``tests/topology/test_pool_detect_decode.py``); ``embed`` and ``segment`` cut one crop per
#: detection and scatter one row back onto each (``tests/topology/test_pool_embed_crops.py``).
#: What is asserted *here* is that the untouched kind stayed untouched.
PASSTHROUGH_KINDS = {
    kind: value
    for kind, value in KINDS.items()
    if kind not in (ElementKind.DETECT, ElementKind.EMBED, ElementKind.SEGMENT)
}


class FakeModel:
    """One model handle: records what it was asked, answers with what it was given.

    ``infer`` returns a resolved future by default, which is what makes the element's
    ``result()`` wait a no-op in a test. ``error`` makes it raise — the pool refusing work —
    and ``answer=False`` makes it return a future nobody resolves, which is how the timeout
    path is reached without waiting for a real one.
    """

    def __init__(
        self,
        outputs: dict[str, Tensor] | None = None,
        *,
        error: BaseException | None = None,
        answer: bool = True,
    ) -> None:
        self.outputs = outputs if outputs is not None else {}
        self.error = error
        self.answer = answer
        self.requests: list[InferenceRequest] = []

    def infer(self, request: InferenceRequest) -> ResponseFuture:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        future = ResponseFuture(request)
        if self.answer:
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
    """A :class:`ModelResolver` over a dict, counting lookups.

    The count is the point of one of the tests: a lookup per frame would put a lock
    acquisition on the path of every one of a thousand frames a second.
    """

    def __init__(self, **models: FakeModel) -> None:
        self.models = models
        self.lookups: list[str] = []

    def get(self, name: str):
        self.lookups.append(name)
        try:
            return self.models[name]
        except KeyError:
            raise ModelNotFoundError(name, sorted(self.models)) from None


def tensor(rows: int = 1) -> Tensor:
    return Tensor.from_numpy(np.zeros((rows, 4), dtype=np.float32))


def item(payload: object = None, *, caps: str = "tensor@gpu", **meta: object) -> ChainItem:
    return ChainItem(
        RequestContext(camera_id="cam-1", frame_id=7),
        Caps.parse(caps),
        payload=tensor() if payload is None else payload,
        meta=dict(meta),
    )


class TestTheRegistrations:
    @pytest.mark.parametrize("kind", list(KINDS))
    def test_pool_is_the_registered_default_of_every_model_kind(
        self, kind: ElementKind
    ) -> None:
        """arch.md §1: ``pool`` is the default implementation of all four model kinds."""
        expected, _ = KINDS[kind]

        assert registry_for(kind).get("pool") is expected
        assert expected.impl == "pool"
        assert expected.kind is kind

    def test_the_loader_builds_one_with_its_model_name(self) -> None:
        """``create_element`` is the one door, and it threads ``model=`` through."""
        element = create_element(ElementKind.DETECT, "pool", "detect", model="ship_detector")

        assert isinstance(element, PoolDetect)
        assert element.model == "ship_detector"
        assert not element.is_open


class TestResolvingTheModel:
    def test_the_model_is_resolved_at_open(self) -> None:
        pool = FakePool(ship_recognizer=FakeModel())
        element = PoolRecognize("recognize", model="ship_recognizer")

        element.open(ElementContext(models=pool))

        assert pool.lookups == ["ship_recognizer"]

    def test_an_unknown_model_stops_the_deploy_rather_than_the_first_frame(self) -> None:
        """The whole reason the lookup is at ``open``: §2.6, validate at start-up."""
        pool = FakePool(person_embedder=FakeModel())
        element = PoolRecognize("recognize", model="ship_recognizer")

        with pytest.raises(ModelNotFoundError) as caught:
            element.open(ElementContext(models=pool))

        assert "ship_recognizer" in str(caught.value)
        assert not element.is_open, "a failed open must not leave the element half-open"

    def test_it_is_resolved_once_and_not_per_frame(self) -> None:
        pool = FakePool(ship_recognizer=FakeModel())
        element = PoolRecognize("recognize", model="ship_recognizer")
        element.open(ElementContext(models=pool))

        for frame in range(5):
            element.process(item(**{"frame": frame}))

        assert pool.lookups == ["ship_recognizer"], "one lookup, five frames"

    def test_a_runner_that_passes_no_pool_is_a_wiring_mistake(self) -> None:
        element = PoolRecognize("recognize", model="ship_recognizer")

        with pytest.raises(ConfigurationError, match="needs a model pool"):
            element.open(ElementContext())

    def test_an_element_with_no_model_name_says_so(self) -> None:
        """Unreachable through the loader, which refuses a model kind with no ``model:``."""
        element = PoolRecognize("recognize")

        with pytest.raises(ConfigurationError, match="has no model"):
            element.open(ElementContext(models=FakePool()))


class TestSubmittingOneItem:
    @pytest.mark.parametrize("kind", list(PASSTHROUGH_KINDS))
    def test_the_outputs_land_under_the_kind_s_metadata_key(self, kind: ElementKind) -> None:
        cls, meta_key = PASSTHROUGH_KINDS[kind]
        outputs = {"out": tensor()}
        pool = FakePool(some_model=FakeModel(outputs))
        element = cls("slot", model="some_model")
        element.open(ElementContext(models=pool))

        result = element.process(item(**{"class": "ship"}))

        assert result is not None
        assert result.meta[meta_key] is outputs
        assert result.meta["class"] == "ship", "it adds a key, it removes none"

    def test_the_request_carries_the_item_s_context_by_identity(self) -> None:
        """A copy would let the two tags drift the moment anything stamped one (ADR-002)."""
        model = FakeModel()
        element = PoolRecognize("recognize", model="ship_recognizer")
        element.open(ElementContext(models=FakePool(ship_recognizer=model)))
        walked = item()

        element.process(walked)

        assert model.requests[0].context is walked.context

    def test_the_payload_is_submitted_under_the_model_s_input_name(self) -> None:
        model = FakeModel()
        element = PoolRecognize("recognize", {"input": "pixels"}, model="ship_recognizer")
        element.open(ElementContext(models=FakePool(ship_recognizer=model)))
        payload = tensor(rows=2)

        element.process(item(payload=payload))

        assert model.requests[0].inputs == {"pixels": payload}
        assert model.requests[0].model_name == "ship_recognizer"

    @pytest.mark.parametrize("caps", ["nv12@gpu", "tensor@gpu", "bgr@cpu"])
    def test_the_successor_carries_the_cap_it_arrived_with(self, caps: str) -> None:
        """The payload is handed on untouched, so its label is handed on untouched too.

        This element used to stamp its own first ``produces`` on every successor, which
        **relabelled** the payload: a ``bgr@cpu`` frame left it claiming to be ``nv12@gpu``,
        and the device-to-host download arch.md §8 exists to refuse became invisible to every
        element downstream. The cap on an item is the cap of the edge it is travelling, and
        that is the loader's answer, not this element's.
        """
        element = PoolRecognize("recognize", model="ship_recognizer")
        element.open(ElementContext(models=FakePool(ship_recognizer=FakeModel())))

        result = element.process(item(caps=caps))

        assert result is not None
        assert result.caps == Caps.parse(caps), "carried, not relabelled"
        assert result.context is not None

    def test_the_wait_defaults_to_the_stage_timeout_the_settings_declare(self) -> None:
        """Two defaults that have to agree, in two packages that cannot import each other.

        ``topology`` is pure and may not read ``core.settings`` at import time, so the
        element's default is a literal — and a literal that drifts from
        ``pipeline.stage_timeout_ms`` is a chain quietly waiting a different length of time
        than the deployment was tuned for. This is the assertion that ties them.
        """
        assert ServerSettings().pipeline.stage_timeout_ms / 1000.0 == _DEFAULT_TIMEOUT_S
        assert PoolRecognize("recognize", model="m")._timeout_s == _DEFAULT_TIMEOUT_S


class TestWhereTheTwoKnobsComeFrom:
    """Params, then the runner's context, then the module default — and nothing else.

    The middle step is the one that used not to exist: the two literals above mirrored the
    settings *defaults*, so an operator who lowered ``stage_timeout_ms`` to 500 ms got 5 s
    waits from every ``pool`` element and the key applied to nothing but the docstring.
    """

    def test_the_context_supplies_both_when_the_slot_declares_neither(self) -> None:
        element = PoolRecognize("recognize", model="ship_recognizer")

        element.open(
            ElementContext(
                models=FakePool(ship_recognizer=FakeModel()),
                stage_timeout_s=0.5,
                input_name="pixels",
            )
        )

        assert element._timeout_s == 0.5
        assert element._input == "pixels"

    def test_the_slot_s_params_win_over_the_deployment_s_settings(self) -> None:
        """A tensor name belongs to the model and one slot may need a longer wait."""
        element = PoolRecognize(
            "recognize", {"timeout_s": 2.0, "input": "frames"}, model="ship_recognizer"
        )

        element.open(
            ElementContext(
                models=FakePool(ship_recognizer=FakeModel()),
                stage_timeout_s=0.5,
                input_name="pixels",
            )
        )

        assert element._timeout_s == 2.0
        assert element._input == "frames"

    def test_a_context_that_says_nothing_leaves_the_module_defaults(self) -> None:
        """What a chain-validation test and a hand-built context get."""
        element = PoolRecognize("recognize", model="ship_recognizer")

        element.open(ElementContext(models=FakePool(ship_recognizer=FakeModel())))

        assert element._timeout_s == _DEFAULT_TIMEOUT_S
        assert element._input == _DEFAULT_INPUT

    def test_reopening_under_a_different_context_takes_the_new_numbers(self) -> None:
        """A restarted shard must not keep the previous run's timeout."""
        pool = FakePool(ship_recognizer=FakeModel())
        element = PoolRecognize("recognize", model="ship_recognizer")

        element.open(ElementContext(models=pool, stage_timeout_s=0.25))
        element.close()
        element.open(ElementContext(models=pool, stage_timeout_s=1.5))

        assert element._timeout_s == 1.5

    def test_the_context_s_timeout_is_the_one_actually_waited_on(self) -> None:
        """Asserting the attribute is not enough: this is the value ``result()`` is given.

        The fake model hands back a future nobody resolves, so the only thing that can end
        this call is the bound — and the bound has to be the context's 10 ms rather than the
        module's five seconds, or the test would hang for five seconds before passing.
        """
        element = PoolRecognize("recognize", model="ship_recognizer")
        element.open(
            ElementContext(
                models=FakePool(ship_recognizer=FakeModel(answer=False)), stage_timeout_s=0.01
            )
        )

        started = time.monotonic()
        with pytest.raises(RequestTimeoutError, match=r"0\.01s"):
            element.process(item())
        elapsed = time.monotonic() - started
        assert elapsed < 1.0, "it waited the module default rather than the context's bound"


class TestFailuresAreCarriedNotSwallowed:
    def test_a_full_pool_propagates_and_never_becomes_an_empty_result(self) -> None:
        """Backpressure reaches the runner, which counts it against the right camera (ADR-005)."""
        refused = QueueFullError("ship_recognizer", 256, 256)
        element = PoolRecognize("recognize", model="ship_recognizer")
        element.open(ElementContext(models=FakePool(ship_recognizer=FakeModel(error=refused))))

        with pytest.raises(QueueFullError) as caught:
            element.process(item())

        assert caught.value is refused, "propagated untouched, not rewrapped"

    def test_a_pool_that_does_not_answer_in_time_is_a_typed_timeout(self) -> None:
        """A worker blocked forever on one model takes a whole shard's throughput with it."""
        element = PoolRecognize("recognize", {"timeout_s": 0.01}, model="ship_recognizer")
        element.open(ElementContext(models=FakePool(ship_recognizer=FakeModel(answer=False))))

        with pytest.raises(RequestTimeoutError, match="ship_recognizer"):
            element.process(item())

    def test_a_payload_that_is_not_a_tensor_is_refused_with_its_type(self) -> None:
        """Until phase D a device frame handle is not submittable, and saying so is the fix."""
        element = PoolRecognize("recognize", model="ship_recognizer")
        element.open(ElementContext(models=FakePool(ship_recognizer=FakeModel())))

        with pytest.raises(ValidationError) as caught:
            element.process(item(payload="frame:cam-1:7"))

        assert "str" in str(caught.value)
        assert "DataPool" in str(caught.value), "the message says where the fix lives"


class TestTheLifecycle:
    def test_close_forgets_the_handle_and_open_resolves_it_again(self) -> None:
        """The handle belongs to the pool, whose life is longer than the element's."""
        pool = FakePool(ship_recognizer=FakeModel())
        element = PoolRecognize("recognize", model="ship_recognizer")

        element.open(ElementContext(models=pool))
        element.close()
        element.open(ElementContext(models=pool))

        assert pool.lookups == ["ship_recognizer", "ship_recognizer"]
        assert element.is_open


# -- the cap the loader resolves for a pool element ---------------------------------------
#
# A `pool` element declares `produces: *@*`, so its outbound cap is whatever the loader
# negotiated on its inbound edge. That is a *chain* property, not an element property, and
# these are the two halves of it: the host-memory chain is refused, and the device chain
# still loads.


@registry_for(ElementKind.DECODE).register("pool-test-gpu")
class DeviceDecode(Element):
    """A decoder that leaves the frame in VRAM — phase D's ``gstreamer-gpu``, in advance.

    Declared here because every shipped decode implementation delivers ``bgr@cpu`` today
    (``elements/decode.py``), so there is nothing in the registry to put a device cap on the
    head of a chain, and the propagation this section is about has two halves.
    """

    kind: ClassVar[ElementKind] = ElementKind.DECODE
    produces: ClassVar[tuple[str, ...]] = ("nv12@gpu",)

    def _do_open(self, context: ElementContext) -> None:
        return None

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        return item.derive(caps=self.output_caps[0])


#: ``decode -> segment{impl: pool} -> track -> output``. The segmenter's outbound cap is
#: whatever it was handed, and the tracker takes every plane, so this chain loads either way
#: and is where the *propagation* is read off the edges.
CHAIN = """
name: pool_caps
elements:
  decode:  {impl: __DECODE__}
  segment: {impl: pool, model: ship_segmenter}
  track:   {impl: shipvision}
  output:  {impl: none}
"""

#: The same chain with the tracker taken out, so the segmenter hands straight to the sink. A
#: sink serialises, so it accepts ``meta@cpu`` and ``bgr@cpu`` and nothing device-resident —
#: which makes this the wiring where a relabelled cap is caught.
DIRECT = """
name: pool_caps_direct
elements:
  decode:  {impl: __DECODE__}
  segment: {impl: pool, model: ship_segmenter}
  output:  {impl: none}
"""


def chain(decode: str, text: str = CHAIN) -> Topology:
    return Topology.from_spec(
        ChainSpec.from_yaml(textwrap.dedent(text).replace("__DECODE__", decode))
    )


class TestWhatTheLoaderResolvesForAPoolElement:
    def test_a_device_frame_handed_to_a_serialising_sink_is_refused_at_load(self) -> None:
        """The relabelling this element used to do, caught where it has to be caught.

        ``nv12@gpu`` reaches the segmenter, which accepts it — and hands it on as
        ``nv12@gpu``, which a sink cannot take. Refused at start-up, naming both sides. With
        a concrete ``produces: bgr@cpu`` on the element this chain *loaded*: the segmenter
        told the loader it had brought the frame back to host memory, and the per-frame
        download arch.md §8 exists to refuse would have shown up as a mysteriously slow sink.
        """
        with pytest.raises(CapsMismatchError) as caught:
            chain("pool-test-gpu", DIRECT)

        assert "segment" in str(caught.value)
        assert "output" in str(caught.value)

    def test_a_device_producer_behind_it_loads_with_the_device_cap_on_both_edges(
        self,
    ) -> None:
        """The other half: the wildcard must not refuse the chain the deployment runs.

        The negotiated inbound cap is propagated to the outbound edge, so the ``*@*`` never
        reaches an ``Edge`` — a chain whose edges read ``*@*`` would be a chain nobody
        checked.
        """
        loaded = chain("pool-test-gpu")

        negotiated = {(edge.producer, edge.consumer): str(edge.caps) for edge in loaded.edges}
        assert negotiated[("decode", "segment")] == "nv12@gpu"
        assert negotiated[("segment", "track")] == "nv12@gpu"
        assert loaded.node("segment").element.output_caps == (
            Caps.parse("*@*"),
        ), "the element still declares the wildcard; only the edge is resolved"
