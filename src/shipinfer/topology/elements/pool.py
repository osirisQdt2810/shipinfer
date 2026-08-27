"""The ``pool`` elements: a chain step that submits to the model pool and waits.

Four kinds run a model from the repository — ``detect``, ``segment``, ``embed``,
``recognize`` — and for all four the default implementation is the same one: build one
request, hand it to the engine, sleep until the answer arrives, add the outputs to the item's
metadata. That is what arch.md §1 calls "a thin client of the engine" and §5③ calls "submit
to the pool → wait (sleeping)". The alternatives behind the same four kinds (``nvinfer``,
``nvinferserver``) run the model somewhere else entirely, which is exactly why this is a
registry and not a base class with a flag.

**Why this file can live in a pure layer.** It never imports the engine. The model pool
arrives as :attr:`~shipinfer.topology.base.ElementContext.models`, a
:class:`~shipinfer.topology.base.ModelResolver` — one method, ``get(name)`` — which the
engine satisfies structurally and a test satisfies with a dict. So
``import shipinfer.topology`` still costs no torch and a chain naming ``impl: pool`` is still
*validatable* on a laptop: the name is data, and only ``open()`` needs the pool to exist.

**The name is resolved once, at open, not per frame.** ``get(name)`` on the engine takes a
lock and can raise; doing it per frame would put a lock acquisition and a dictionary lookup
on the path of every one of a thousand frames a second, and — worse — would turn a
misconfigured chain into a failure that appears on the first frame of a deploy instead of at
start-up. A model that is not in the pool must stop the deploy (§2.6: validate at start-up,
not at first use).

**The caps, and one promise this element cannot yet keep.** It declares ``nv12@gpu`` first
because the device path is the default end to end (§8), ``tensor@gpu`` for a producer that
already cropped, and ``bgr@cpu`` for the host fallback. It *produces* ``nv12@gpu``, because
it hands the payload on unchanged for the next element to read. Those two are consistent for
the two device caps and inconsistent for ``bgr@cpu``: an element that is handed host memory
and claims to hand on ``nv12@gpu`` is lying to the loader. Nothing is wired to the host path
today — no registered decode implementation produces ``bgr@cpu`` outside the tests — and the
fix when something is is to declare ``produces: *@*`` and let the loader resolve it from the
inbound cap, exactly as ``MockPassthrough`` does. That is a change to what §8 refuses, so it
is not smuggled in here; it is named, and it is the first thing to do before a host-memory
decoder ships.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, ClassVar

from shipinfer.core.errors import ConfigurationError, RequestTimeoutError, ValidationError
from shipinfer.core.request import InferenceRequest
from shipinfer.core.types import Tensor
from shipinfer.topology.base import ChainItem, Element, ElementContext, ElementKind
from shipinfer.topology.registry import registry_for

__all__ = [
    "PoolDetect",
    "PoolEmbed",
    "PoolRecognize",
    "PoolSegment",
]

#: Where the model's outputs are read from in the request. Mirrors the default of
#: ``ingest.input_name``, and is overridable per slot with ``params: {input: ...}`` because
#: the tensor name belongs to the *model*, not to the deployment.
_DEFAULT_INPUT = "images"

#: Seconds to wait for the pool. Mirrors the default of ``pipeline.stage_timeout_ms``, and is
#: a bound rather than ``None`` on purpose: a worker blocked forever on one model takes a
#: whole shard's throughput with it, and the queue's own expiry cannot help a request that
#: has already been dispatched.
_DEFAULT_TIMEOUT_S = 5.0


class _PoolElement(Element):
    """Shared behaviour for the four model kinds: submit one request, store its outputs.

    Deliberately not registered and deliberately private: it is not an implementation, it is
    the half of one that four kinds share. Each subclass adds a ``kind`` and the metadata key
    its results are filed under, and that is the whole difference between a detector and a
    recogniser *at this layer* — which is the point, because the difference that matters is in
    the model, and the model is named by the chain file.

    Args:
        name: the chain slot.
        params: ``input`` (the model's input tensor name) and ``timeout_s``.
        model: the repository model to run. Required — the loader already refuses a model kind
            that names none, so a missing one here is a programming error and says so.
    """

    accepts: ClassVar[tuple[str, ...]] = ("nv12@gpu", "tensor@gpu", "bgr@cpu")
    produces: ClassVar[tuple[str, ...]] = ("nv12@gpu",)

    #: Where this kind's results are filed in :attr:`ChainItem.meta`. The one thing a
    #: subclass must declare, and the vocabulary the downstream elements read: ``track``
    #: wants ``boxes``, ``mtmc`` wants ``vectors``.
    meta_key: ClassVar[str] = ""

    def __init__(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__(name, params, model=model)
        self._input = str(self.params.get("input", _DEFAULT_INPUT))
        self._timeout_s = float(self.params.get("timeout_s", _DEFAULT_TIMEOUT_S))
        self._handle: Any = None

    def _do_open(self, context: ElementContext) -> None:
        """Resolve the model **now**, so a bad name stops the deploy.

        Raises:
            ConfigurationError: the runner handed no model pool, or this element carries no
                model name. Both are wiring mistakes in the process that built the runner,
                not in the operator's chain file, and the message says which.
            ModelNotFoundError: the pool has no such model — the resolver's own typed error,
                raised here rather than caught, because it already names the model and lists
                the ones that exist.
        """
        if not self.model:
            raise ConfigurationError(
                f"pool element {self.name!r} has no model; a {type(self).__name__} runs a "
                "repository model and the chain must name it with `model: <name>`"
            )
        if context.models is None:
            raise ConfigurationError(
                f"pool element {self.name!r} needs a model pool, and the runner passed none. "
                "An ElementContext with `models=` is what a `pool` element is opened with; "
                "without one it could only fail on the first frame"
            )
        self._handle = context.models.get(self.model)

    def _do_close(self) -> None:
        # The handle belongs to the pool, whose lifetime is longer than this element's; the
        # element only forgets it, so a reopened chain resolves it again.
        self._handle = None

    def _do_process(self, item: ChainItem) -> ChainItem:
        """Submit one request for this item and file the outputs under :attr:`meta_key`.

        The request carries the item's :class:`~shipinfer.core.request.RequestContext`
        **by identity**, not a copy: that tag is what reassembly, tracing and every log line
        group on (ADR-002), and a copy would let the two drift the moment anything stamped a
        timestamp on one of them.

        Returns:
            The successor item — same tag, same payload, one metadata key richer.

        Raises:
            ValidationError: the item's payload is not a tensor this element can submit. A
                device-resident frame handle becomes a submittable tensor with the DataPool
                (arch.md §3, phase D); until then the chain in front of a pool element has to
                hand over a :class:`~shipinfer.core.types.Tensor`.
            QueueFullError: the pool is saturated. Propagated untouched — it is backpressure,
                and the runner turns it into a counted, per-camera drop (ADR-005). It must
                never become a ``None``: "no ships in this frame" and "the detector is full"
                demand opposite responses.
            RequestTimeoutError: the pool did not answer within ``timeout_s``.
            ServerStateError: called before :meth:`Element.open` — the base class's refusal.
        """
        payload = item.payload
        if not isinstance(payload, Tensor):
            raise ValidationError(
                f"pool element {self.name!r} was handed a payload of type "
                f"{type(payload).__name__} and needs a core Tensor to submit; the caps say "
                f"{item.caps}. A device-resident frame handle becomes a tensor in phase D "
                "(the DataPool)"
            )
        request = InferenceRequest(
            model_name=self.model or "",
            inputs={self._input: payload},
            context=item.context,
        )
        future = self._handle.infer(request)
        try:
            response = future.result(self._timeout_s)
        except FutureTimeoutError as exc:
            raise RequestTimeoutError(
                f"model {self.model!r} did not answer element {self.name!r} for "
                f"{item.key} within {self._timeout_s}s"
            ) from exc
        # `output_caps[0]` is the only cap this element declares, which is what makes it the
        # right answer here — an element with two `produces` and two consumers hands a
        # different cap to each and would have to be told which edge it is on.
        return item.derive(caps=self.output_caps[0], **{self.meta_key: response.outputs})


@registry_for(ElementKind.DETECT).register("pool")
class PoolDetect(_PoolElement):
    """Detection through the model pool — the default detector (arch.md §1)."""

    kind: ClassVar[ElementKind] = ElementKind.DETECT
    meta_key: ClassVar[str] = "boxes"


@registry_for(ElementKind.SEGMENT).register("pool")
class PoolSegment(_PoolElement):
    """Segmentation through the model pool."""

    kind: ClassVar[ElementKind] = ElementKind.SEGMENT
    meta_key: ClassVar[str] = "masks"


@registry_for(ElementKind.EMBED).register("pool")
class PoolEmbed(_PoolElement):
    """Embedding through the model pool — one instance per embedder slot in the chain."""

    kind: ClassVar[ElementKind] = ElementKind.EMBED
    meta_key: ClassVar[str] = "vectors"


@registry_for(ElementKind.RECOGNIZE).register("pool")
class PoolRecognize(_PoolElement):
    """Identity through the model pool."""

    kind: ClassVar[ElementKind] = ElementKind.RECOGNIZE
    meta_key: ClassVar[str] = "identities"
