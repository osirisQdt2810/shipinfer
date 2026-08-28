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

The element under test is :class:`PoolSegment` wherever one kind has to stand for the four:
it is the plainest of them, and since C3 the *detector* is not — ``PoolDetect`` replaces both
of ``_do_process``'s hooks to letterbox its frame and decode the boxes back into source pixels,
so it needs image ops to open at all and files a decoded ``Detections`` rather than the raw
outputs. Testing the shared behaviour through the one subclass that overrides it would be
testing the override. ``tests/topology/test_pool_detect_decode.py`` is where the detector's own
contract lives.

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
    ElementContext,
    ElementKind,
    Topology,
    create_element,
    registry_for,
)
from shipinfer.topology.elements.mock import MockDecode
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

#: The two that file the model's outputs raw and hand the payload on untouched — the shared
#: behaviour this file is about. ``detect`` and ``embed`` are deliberately absent: both replace
#: ``_do_process``'s two hooks and both need image ops, so neither is a fair sample of the
#: shared code and neither is openable without a runner that resolved an implementation.
#: ``detect`` letterboxes the frame and decodes the boxes back
#: (``tests/topology/test_pool_detect_decode.py``); ``embed`` cuts one crop per detection and
#: scatters the vectors back onto the rows (``tests/topology/test_pool_embed_crops.py``). What
#: is asserted *here* is that the untouched kinds stayed untouched.
PASSTHROUGH_KINDS = {
    kind: value
    for kind, value in KINDS.items()
    if kind not in (ElementKind.DETECT, ElementKind.EMBED)
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
        pool = FakePool(ship_segmenter=FakeModel())
        element = PoolSegment("segment", model="ship_segmenter")

        element.open(ElementContext(models=pool))

        assert pool.lookups == ["ship_segmenter"]

    def test_an_unknown_model_stops_the_deploy_rather_than_the_first_frame(self) -> None:
        """The whole reason the lookup is at ``open``: §2.6, validate at start-up."""
        pool = FakePool(person_embedder=FakeModel())
        element = PoolSegment("segment", model="ship_segmenter")

        with pytest.raises(ModelNotFoundError) as caught:
            element.open(ElementContext(models=pool))

        assert "ship_segmenter" in str(caught.value)
        assert not element.is_open, "a failed open must not leave the element half-open"

    def test_it_is_resolved_once_and_not_per_frame(self) -> None:
        pool = FakePool(ship_segmenter=FakeModel())
        element = PoolSegment("segment", model="ship_segmenter")
        element.open(ElementContext(models=pool))

        for frame in range(5):
            element.process(item(**{"frame": frame}))

        assert pool.lookups == ["ship_segmenter"], "one lookup, five frames"

    def test_a_runner_that_passes_no_pool_is_a_wiring_mistake(self) -> None:
        element = PoolSegment("segment", model="ship_segmenter")

        with pytest.raises(ConfigurationError, match="needs a model pool"):
            element.open(ElementContext())

    def test_an_element_with_no_model_name_says_so(self) -> None:
        """Unreachable through the loader, which refuses a model kind with no ``model:``."""
        element = PoolSegment("segment")

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
        element = PoolSegment("segment", model="ship_segmenter")
        element.open(ElementContext(models=FakePool(ship_segmenter=model)))
        walked = item()

        element.process(walked)

        assert model.requests[0].context is walked.context

    def test_the_payload_is_submitted_under_the_model_s_input_name(self) -> None:
        model = FakeModel()
        element = PoolSegment("segment", {"input": "pixels"}, model="ship_segmenter")
        element.open(ElementContext(models=FakePool(ship_segmenter=model)))
        payload = tensor(rows=2)

        element.process(item(payload=payload))

        assert model.requests[0].inputs == {"pixels": payload}
        assert model.requests[0].model_name == "ship_segmenter"

    @pytest.mark.parametrize("caps", ["nv12@gpu", "tensor@gpu", "bgr@cpu"])
    def test_the_successor_carries_the_cap_it_arrived_with(self, caps: str) -> None:
        """The payload is handed on untouched, so its label is handed on untouched too.

        This element used to stamp its own first ``produces`` on every successor, which
        **relabelled** the payload: a ``bgr@cpu`` frame left it claiming to be ``nv12@gpu``,
        and the device-to-host download arch.md §8 exists to refuse became invisible to every
        element downstream. The cap on an item is the cap of the edge it is travelling, and
        that is the loader's answer, not this element's.
        """
        element = PoolSegment("segment", model="ship_segmenter")
        element.open(ElementContext(models=FakePool(ship_segmenter=FakeModel())))

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
        assert PoolSegment("segment", model="m")._timeout_s == _DEFAULT_TIMEOUT_S


class TestWhereTheTwoKnobsComeFrom:
    """Params, then the runner's context, then the module default — and nothing else.

    The middle step is the one that used not to exist: the two literals above mirrored the
    settings *defaults*, so an operator who lowered ``stage_timeout_ms`` to 500 ms got 5 s
    waits from every ``pool`` element and the key applied to nothing but the docstring.
    """

    def test_the_context_supplies_both_when_the_slot_declares_neither(self) -> None:
        element = PoolSegment("segment", model="ship_segmenter")

        element.open(
            ElementContext(
                models=FakePool(ship_segmenter=FakeModel()),
                stage_timeout_s=0.5,
                input_name="pixels",
            )
        )

        assert element._timeout_s == 0.5
        assert element._input == "pixels"

    def test_the_slot_s_params_win_over_the_deployment_s_settings(self) -> None:
        """A tensor name belongs to the model and one slot may need a longer wait."""
        element = PoolSegment(
            "segment", {"timeout_s": 2.0, "input": "frames"}, model="ship_segmenter"
        )

        element.open(
            ElementContext(
                models=FakePool(ship_segmenter=FakeModel()),
                stage_timeout_s=0.5,
                input_name="pixels",
            )
        )

        assert element._timeout_s == 2.0
        assert element._input == "frames"

    def test_a_context_that_says_nothing_leaves_the_module_defaults(self) -> None:
        """What a chain-validation test and a hand-built context get."""
        element = PoolSegment("segment", model="ship_segmenter")

        element.open(ElementContext(models=FakePool(ship_segmenter=FakeModel())))

        assert element._timeout_s == _DEFAULT_TIMEOUT_S
        assert element._input == _DEFAULT_INPUT

    def test_reopening_under_a_different_context_takes_the_new_numbers(self) -> None:
        """A restarted shard must not keep the previous run's timeout."""
        pool = FakePool(ship_segmenter=FakeModel())
        element = PoolSegment("segment", model="ship_segmenter")

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
        element = PoolSegment("segment", model="ship_segmenter")
        element.open(
            ElementContext(
                models=FakePool(ship_segmenter=FakeModel(answer=False)), stage_timeout_s=0.01
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
        refused = QueueFullError("ship_segmenter", 256, 256)
        element = PoolSegment("segment", model="ship_segmenter")
        element.open(ElementContext(models=FakePool(ship_segmenter=FakeModel(error=refused))))

        with pytest.raises(QueueFullError) as caught:
            element.process(item())

        assert caught.value is refused, "propagated untouched, not rewrapped"

    def test_a_pool_that_does_not_answer_in_time_is_a_typed_timeout(self) -> None:
        """A worker blocked forever on one model takes a whole shard's throughput with it."""
        element = PoolSegment("segment", {"timeout_s": 0.01}, model="ship_segmenter")
        element.open(ElementContext(models=FakePool(ship_segmenter=FakeModel(answer=False))))

        with pytest.raises(RequestTimeoutError, match="ship_segmenter"):
            element.process(item())

    def test_a_payload_that_is_not_a_tensor_is_refused_with_its_type(self) -> None:
        """Until phase D a device frame handle is not submittable, and saying so is the fix."""
        element = PoolSegment("segment", model="ship_segmenter")
        element.open(ElementContext(models=FakePool(ship_segmenter=FakeModel())))

        with pytest.raises(ValidationError) as caught:
            element.process(item(payload="frame:cam-1:7"))

        assert "str" in str(caught.value)
        assert "DataPool" in str(caught.value), "the message says where the fix lives"


class TestTheLifecycle:
    def test_close_forgets_the_handle_and_open_resolves_it_again(self) -> None:
        """The handle belongs to the pool, whose life is longer than the element's."""
        pool = FakePool(ship_segmenter=FakeModel())
        element = PoolSegment("segment", model="ship_segmenter")

        element.open(ElementContext(models=pool))
        element.close()
        element.open(ElementContext(models=pool))

        assert pool.lookups == ["ship_segmenter", "ship_segmenter"]
        assert element.is_open


# -- the cap the loader resolves for a pool element ---------------------------------------
#
# A `pool` element declares `produces: *@*`, so its outbound cap is whatever the loader
# negotiated on its inbound edge. That is a *chain* property, not an element property, and
# these are the two halves of it: the host-memory chain is refused, and the device chain
# still loads.


@registry_for(ElementKind.DECODE).register("pool-test-cpu")
class CpuDecode(MockDecode):
    """A decoder that hands out host memory. The producer §8 refuses to see downloaded from."""

    produces: ClassVar[tuple[str, ...]] = ("bgr@cpu",)


#: ``decode -> segment{impl: pool} -> track -> output``. The tracker takes ``nv12@gpu`` or
#: ``meta@cpu`` and nothing else, so what the segmenter hands on decides whether this chain
#: loads at all.
CHAIN = """
name: pool_caps
elements:
  decode:  {impl: __DECODE__}
  segment: {impl: pool, model: ship_segmenter}
  track:   {impl: mock}
  output:  {impl: mock}
"""


def chain(decode: str) -> Topology:
    text = textwrap.dedent(CHAIN).replace("__DECODE__", decode)
    return Topology.from_spec(ChainSpec.from_yaml(text))


class TestWhatTheLoaderResolvesForAPoolElement:
    def test_a_host_memory_producer_behind_it_is_refused_at_load(self) -> None:
        """The relabelling this element used to do, caught where it has to be caught.

        ``bgr@cpu`` reaches the segmenter, which accepts it — and hands it on as ``bgr@cpu``,
        which the tracker cannot take. Refused at start-up, naming both sides. With a
        concrete ``produces: nv12@gpu`` on the element this chain *loaded*: the segmenter
        told the loader it had turned host memory into device memory, and the download
        arch.md §8 exists to refuse would have shown up as a mysteriously slow tracker.
        """
        with pytest.raises(CapsMismatchError) as caught:
            chain("pool-test-cpu")

        assert "segment" in str(caught.value)
        assert "track" in str(caught.value)

    def test_a_device_producer_behind_it_loads_with_the_device_cap_on_both_edges(
        self,
    ) -> None:
        """The other half: the wildcard must not refuse the chain the deployment runs.

        The negotiated inbound cap is propagated to the outbound edge, so the ``*@*`` never
        reaches an ``Edge`` — a chain whose edges read ``*@*`` would be a chain nobody
        checked.
        """
        loaded = chain("mock")

        negotiated = {(edge.producer, edge.consumer): str(edge.caps) for edge in loaded.edges}
        assert negotiated[("decode", "segment")] == "nv12@gpu"
        assert negotiated[("segment", "track")] == "nv12@gpu"
        assert loaded.node("segment").element.output_caps == (
            Caps.parse("*@*"),
        ), "the element still declares the wildcard; only the edge is resolved"
