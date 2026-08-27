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

**The caps: it hands on what it was handed.** It *accepts* ``nv12@gpu`` first because the
device path is the default end to end (§8), then ``tensor@gpu`` for a producer that already
cropped, then ``bgr@cpu`` for the host fallback. It *produces* ``*@*``, which is not vagueness
but the precise claim: this element reads the payload, adds a metadata key and hands the
payload on **unchanged**, so its outbound cap is its negotiated inbound cap and the loader
resolves it as such (:func:`~shipinfer.topology.chain._resolve_produced`, exactly as
``MockPassthrough`` does).

Declaring a concrete ``produces: nv12@gpu`` instead — which this file did until the wildcard
went in — is a **relabelling**: fed ``bgr@cpu`` it told the loader host memory was device
memory, so the device-to-host download §8 exists to refuse became invisible one element
further down, with every edge reported valid. The wildcard is why a ``bgr@cpu`` decoder in
front of a ``pool`` element and an ``nv12@gpu``-only tracker behind it is now refused at load.
The corollary is that :meth:`_PoolElement._do_process` must not stamp a cap on the item it
derives: the cap on an item is the cap of the edge it is travelling, and the edge is the
loader's (:class:`~shipinfer.topology.chain.Edge`).

**Two knobs, and where each one comes from.** The wait and the input tensor name are resolved
at :meth:`~shipinfer.topology.base.Element.open`, in a fixed precedence: this slot's
``params:`` first, then the runner's :class:`~shipinfer.topology.base.ElementContext`
(``stage_timeout_s`` from ``pipeline.stage_timeout_ms``, ``input_name`` from
``ingest.input_name``), then the module default below. Params win because a tensor name
belongs to the *model* and one slot may need a longer wait than the rest of the chain; the
context wins over the default because an operator who lowers ``stage_timeout_ms`` to 500 ms
means it — before the context carried the value, every ``pool`` element still waited 5 s and
the settings key applied to nothing. The default is the floor for an element opened outside a
runner, which is what a chain-validation test does.
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

#: Where the model's outputs are read from in the request. The **last** resort: mirrors the
#: default of ``ingest.input_name`` for an element opened with a context that carries no
#: resolved value at all.
_DEFAULT_INPUT = "images"

#: Seconds to wait for the pool. The last resort in the same way, mirroring the default of
#: ``pipeline.stage_timeout_ms``, and a bound rather than ``None`` on purpose: a worker
#: blocked forever on one model takes a whole shard's throughput with it, and the queue's own
#: expiry cannot help a request that has already been dispatched.
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
        params: ``input`` (the model's input tensor name) and ``timeout_s``. Both override
            whatever the runner resolved from the settings — see the module docstring for the
            precedence.
        model: the repository model to run. Required — the loader already refuses a model kind
            that names none, so a missing one here is a programming error and says so.
    """

    accepts: ClassVar[tuple[str, ...]] = ("nv12@gpu", "tensor@gpu", "bgr@cpu")
    #: ``*@*``: the payload is handed on untouched, so the outbound cap *is* the negotiated
    #: inbound one and the loader fills it in. See the module docstring for what a concrete
    #: cap here would relabel.
    produces: ClassVar[tuple[str, ...]] = ("*@*",)

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
        # The params-or-default answer, so an element that is never opened still has both.
        # `_do_open` re-resolves them against the runner's context, which is the only place
        # the deployment's settings exist.
        self._input = str(self.params.get("input", _DEFAULT_INPUT))
        self._timeout_s = float(self.params.get("timeout_s", _DEFAULT_TIMEOUT_S))
        self._handle: Any = None

    def _resolve_settings(self, context: ElementContext) -> None:
        """Fix the wait and the input name for this run: params, then context, then default.

        Recomputed from scratch on every open rather than filled in where it is still unset,
        so that a chain reopened under a different context does not keep the previous one's
        numbers.
        """
        fallback_timeout = (
            _DEFAULT_TIMEOUT_S if context.stage_timeout_s is None else context.stage_timeout_s
        )
        fallback_input = _DEFAULT_INPUT if context.input_name is None else context.input_name
        self._timeout_s = float(self.params.get("timeout_s", fallback_timeout))
        self._input = str(self.params.get("input", fallback_input))

    def _do_open(self, context: ElementContext) -> None:
        """Resolve the model **now**, so a bad name stops the deploy.

        Also fixes the wait and the input tensor name from the context the runner built — the
        one moment the deployment's settings are visible to a pure element.

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
        self._resolve_settings(context)
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

        **A timeout abandons the request; it does not cancel it.** There is no cancellation
        path to call: :class:`~shipinfer.core.request.ResponseFuture` is a plain
        :class:`concurrent.futures.Future` and neither
        :class:`~shipinfer.scheduling.queues.base.RequestQueue` nor the engine's model exposes
        a "remove this one" — an item that has been queued is dequeued, assembled and executed
        whatever its future says, and only the *result* is discarded
        (``engine/instance.py::_complete``'s ``set_running_or_notify_cancel``). So a frame that
        times out here still costs the instance slot that made it late, and under sustained
        overload that compounds: every timed-out frame keeps consuming the capacity that caused
        the timeout, which is a queue that never drains rather than one that sheds. The bound
        is still worth having — it frees the *worker* — and phase B adds the real cancellation
        path (a queue-side removal plus a pre-dispatch check) once its bench has measured how
        much of the overload this accounts for. It is deliberately not guessed at here.

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
        # No `caps=`: the payload is handed on unchanged, so the cap it carries is the cap it
        # arrived with. Stamping `output_caps[0]` here would relabel a `bgr@cpu` frame as
        # whatever this class declares first, and resolving the real outbound cap means
        # knowing which edge the item is travelling — which is the loader's answer
        # (`Edge.caps`), not this element's.
        return item.derive(**{self.meta_key: response.outputs})


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
