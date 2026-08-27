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

The pool is a **fake with one method**, which is the honest measure of how narrow
:class:`~shipinfer.topology.base.ModelResolver` is: ``get(name)``. That is why this file
needs no server, no engine and no driver.

It lives under ``tests/runners/`` rather than ``tests/topology/`` because the element is only
meaningful with a runner's :class:`ElementContext` behind it, and because it arrived with the
runner package.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipinfer.core.errors import (
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
from shipinfer.core.types import Tensor
from shipinfer.topology import (
    Caps,
    ChainItem,
    ElementContext,
    ElementKind,
    create_element,
    registry_for,
)
from shipinfer.topology.elements.pool import (
    PoolDetect,
    PoolEmbed,
    PoolRecognize,
    PoolSegment,
)

#: kind -> (class, the metadata key its results are filed under)
KINDS = {
    ElementKind.DETECT: (PoolDetect, "boxes"),
    ElementKind.SEGMENT: (PoolSegment, "masks"),
    ElementKind.EMBED: (PoolEmbed, "vectors"),
    ElementKind.RECOGNIZE: (PoolRecognize, "identities"),
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


def item(payload: object = None, **meta: object) -> ChainItem:
    return ChainItem(
        RequestContext(camera_id="cam-1", frame_id=7),
        Caps.parse("tensor@gpu"),
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
        pool = FakePool(ship_detector=FakeModel())
        element = PoolDetect("detect", model="ship_detector")

        element.open(ElementContext(models=pool))

        assert pool.lookups == ["ship_detector"]

    def test_an_unknown_model_stops_the_deploy_rather_than_the_first_frame(self) -> None:
        """The whole reason the lookup is at ``open``: §2.6, validate at start-up."""
        pool = FakePool(person_embedder=FakeModel())
        element = PoolDetect("detect", model="ship_detector")

        with pytest.raises(ModelNotFoundError) as caught:
            element.open(ElementContext(models=pool))

        assert "ship_detector" in str(caught.value)
        assert not element.is_open, "a failed open must not leave the element half-open"

    def test_it_is_resolved_once_and_not_per_frame(self) -> None:
        pool = FakePool(ship_detector=FakeModel())
        element = PoolDetect("detect", model="ship_detector")
        element.open(ElementContext(models=pool))

        for frame in range(5):
            element.process(item(**{"frame": frame}))

        assert pool.lookups == ["ship_detector"], "one lookup, five frames"

    def test_a_runner_that_passes_no_pool_is_a_wiring_mistake(self) -> None:
        element = PoolDetect("detect", model="ship_detector")

        with pytest.raises(ConfigurationError, match="needs a model pool"):
            element.open(ElementContext())

    def test_an_element_with_no_model_name_says_so(self) -> None:
        """Unreachable through the loader, which refuses a model kind with no ``model:``."""
        element = PoolDetect("detect")

        with pytest.raises(ConfigurationError, match="has no model"):
            element.open(ElementContext(models=FakePool()))


class TestSubmittingOneItem:
    @pytest.mark.parametrize("kind", list(KINDS))
    def test_the_outputs_land_under_the_kind_s_metadata_key(self, kind: ElementKind) -> None:
        cls, meta_key = KINDS[kind]
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
        element = PoolDetect("detect", model="ship_detector")
        element.open(ElementContext(models=FakePool(ship_detector=model)))
        walked = item()

        element.process(walked)

        assert model.requests[0].context is walked.context

    def test_the_payload_is_submitted_under_the_model_s_input_name(self) -> None:
        model = FakeModel()
        element = PoolDetect("detect", {"input": "pixels"}, model="ship_detector")
        element.open(ElementContext(models=FakePool(ship_detector=model)))
        payload = tensor(rows=2)

        element.process(item(payload=payload))

        assert model.requests[0].inputs == {"pixels": payload}
        assert model.requests[0].model_name == "ship_detector"

    def test_the_successor_carries_the_declared_output_cap(self) -> None:
        element = PoolDetect("detect", model="ship_detector")
        element.open(ElementContext(models=FakePool(ship_detector=FakeModel())))

        result = element.process(item())

        assert result is not None
        assert result.caps == Caps.parse("nv12@gpu")
        assert result.context is not None


class TestFailuresAreCarriedNotSwallowed:
    def test_a_full_pool_propagates_and_never_becomes_an_empty_result(self) -> None:
        """Backpressure reaches the runner, which counts it against the right camera (ADR-005)."""
        refused = QueueFullError("ship_detector", 256, 256)
        element = PoolDetect("detect", model="ship_detector")
        element.open(ElementContext(models=FakePool(ship_detector=FakeModel(error=refused))))

        with pytest.raises(QueueFullError) as caught:
            element.process(item())

        assert caught.value is refused, "propagated untouched, not rewrapped"

    def test_a_pool_that_does_not_answer_in_time_is_a_typed_timeout(self) -> None:
        """A worker blocked forever on one model takes a whole shard's throughput with it."""
        element = PoolDetect("detect", {"timeout_s": 0.01}, model="ship_detector")
        element.open(ElementContext(models=FakePool(ship_detector=FakeModel(answer=False))))

        with pytest.raises(RequestTimeoutError, match="ship_detector"):
            element.process(item())

    def test_a_payload_that_is_not_a_tensor_is_refused_with_its_type(self) -> None:
        """Until phase D a device frame handle is not submittable, and saying so is the fix."""
        element = PoolDetect("detect", model="ship_detector")
        element.open(ElementContext(models=FakePool(ship_detector=FakeModel())))

        with pytest.raises(ValidationError) as caught:
            element.process(item(payload="frame:cam-1:7"))

        assert "str" in str(caught.value)
        assert "DataPool" in str(caught.value), "the message says where the fix lives"


class TestTheLifecycle:
    def test_close_forgets_the_handle_and_open_resolves_it_again(self) -> None:
        """The handle belongs to the pool, whose life is longer than the element's."""
        pool = FakePool(ship_detector=FakeModel())
        element = PoolDetect("detect", model="ship_detector")

        element.open(ElementContext(models=pool))
        element.close()
        element.open(ElementContext(models=pool))

        assert pool.lookups == ["ship_detector", "ship_detector"]
        assert element.is_open
