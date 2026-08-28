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

**Three of the four hand the payload on untouched; ``detect`` does not.** A detector is the
one element in the chain that has to *transform* the frame — letterbox it to the model's input
and then undo exactly that transform to put the boxes back in source pixels — so
:class:`PoolDetect` replaces the two hooks ``_do_process`` calls, ``_prepare`` and ``_finish``,
and is the only element that declares ``needs_image_ops``. That split, rather than a flag or an
``isinstance`` inside the shared method, is what keeps the per-frame geometry on the walking
worker's stack: one element instance is shared by every worker, so an attribute would let two
frames overwrite each other's scale.

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
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from shipinfer.core.errors import (
    ConfigurationError,
    InferenceError,
    RequestTimeoutError,
    ValidationError,
)
from shipinfer.core.request import InferenceRequest, InferenceResponse
from shipinfer.core.types import Tensor
from shipinfer.topology.base import ChainItem, Element, ElementContext, ElementKind
from shipinfer.topology.elements.detections import (
    DecodeParams,
    Normalization,
    decode_detections,
)
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

#: The YOLO letterbox grey. The default because a detector config that omits it was trained
#: with it, and a different bar colour is a silent accuracy loss rather than a failure.
_DEFAULT_PAD_VALUE = 114

#: Fallback output names for a detector, used only when the artefact declares nothing usable
#: and the slot names nothing. A loaded model's own declared outputs always win: a tensor's
#: name belongs to the artefact, and hard-coding one refuses a valid engine over a naming
#: preference — ultralytics calls its single output ``output0``, other exports call it
#: ``boxes``.
_DEFAULT_BOXES_OUTPUT = "boxes"
_DEFAULT_COUNT_OUTPUT = "num_detections"


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
            precedence. :class:`PoolDetect` reads a third, ``decode``, which is a mapping and
            is documented on that class.
        model: the repository model to run. Required — the loader refuses an element whose
            :attr:`~shipinfer.topology.base.Element.requires_model_name` is set and names
            none, so a missing one here is a programming error and says so.
    """

    accepts: ClassVar[tuple[str, ...]] = ("nv12@gpu", "tensor@gpu", "bgr@cpu")
    #: ``*@*``: the payload is handed on untouched, so the outbound cap *is* the negotiated
    #: inbound one and the loader fills it in. See the module docstring for what a concrete
    #: cap here would relabel.
    produces: ClassVar[tuple[str, ...]] = ("*@*",)

    #: Both declarations, and this is the implementation for which they coincide: the chain
    #: file has to name the model (the loader's question) *and* `_do_open` resolves that name
    #: against `ElementContext.models` (the pool's). An `nvinfer` element will answer the
    #: first `True` and the second `False`, which is why they are two attributes.
    requires_model_name: ClassVar[bool] = True
    #: The one thing in this file the process that builds the runner reads *before* the
    #: chain is opened: a chain carrying one of these is a chain whose runner has to be handed
    #: a model pool, and `shipinfer run` builds an `InferenceServer` because of it.
    needs_model: ClassVar[bool] = True

    #: Where this kind's results are filed in :attr:`ChainItem.meta`. The one thing a
    #: subclass must declare, and the vocabulary the downstream elements read: ``track`` wants
    #: ``detections``, ``mtmc`` wants ``vectors``.
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
        """Submit one request for this item and file what the model said on the successor.

        The three steps a subclass may not change are here — build the request, wait on the
        bound, propagate every refusal as itself. The two it may are :meth:`_prepare` and
        :meth:`_finish`, and a detector replaces both: it letterboxes the frame on the way in
        and un-letterboxes the boxes on the way out.

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
            The successor item — same tag, same payload, and whatever :meth:`_finish` added.

        **Every one of these is raised as itself and reaches the submitter as itself.** The
        runner that walks this element re-raises a
        :class:`~shipinfer.core.errors.ShipInferError` unchanged into the item's future
        and wraps only a foreign exception, because "the detector's queue is full", "the
        detector never answered" and "the detector has a bug" are three events a caller
        responds to in three different ways. It charges them to three different counters too,
        so an overloaded shard does not read as a shard full of bugs.

        Raises:
            ValidationError: :meth:`_prepare` could not make a submittable tensor out of the
                payload. A device-resident frame handle becomes one with the DataPool
                (arch.md §3, phase D); until then the chain in front of a pool element has to
                hand over a host-resident :class:`~shipinfer.core.types.Tensor`.
            InferenceError: :meth:`_finish` could not read the model's answer — a detector
                whose output name or layout is not what the chain was told.
            QueueFullError: the pool is saturated. Propagated untouched — it is backpressure,
                and the runner turns it into a counted, per-camera drop (ADR-005) whose depth
                and capacity name the queue that refused. It must never become a ``None``:
                "no ships in this frame" and "the detector is full" demand opposite responses.
            RequestTimeoutError: the pool did not answer within ``timeout_s``. A saturation
                signal, not a fault: counted apart from ``items_failed`` for that reason.
            ServerStateError: called before :meth:`Element.open` — the base class's refusal.
        """
        tensor, carried = self._prepare(item)
        request = InferenceRequest(
            model_name=self.model or "",
            inputs={self._input: tensor},
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
        return self._finish(item, response, carried)

    # -- the two halves a subclass may replace -----------------------------------------
    #
    # Split out of `_do_process` rather than left inline, and the seam is where it is for one
    # reason: `PoolDetect` has to *undo* in the second half exactly the transform it applied
    # in the first, and the numbers that describe that transform belong to one frame. An
    # element attribute would be wrong -- one element instance is shared by every pipeline
    # worker (`runners/inprocess.py`), so two frames in flight would overwrite each other's
    # scale and pad and publish boxes computed from the wrong letterbox. So the first half
    # *returns* what the second half needs, the walk carries it on its own stack, and no
    # subclass can accidentally make it shared state.
    #
    # The alternative -- `if isinstance(self, PoolDetect)` inside `_do_process`, or a
    # `decodes: bool` flag -- is the switch statement CONVENTIONS 2.3 exists to refuse.

    def _prepare(self, item: ChainItem) -> tuple[Tensor, Any]:
        """What to submit for this item, and what :meth:`_finish` needs to interpret it.

        The default submits the payload untouched: three of the four kinds are handed a
        tensor somebody upstream already shaped, and re-shaping it here would be a second,
        invisible pre-processing step.

        Returns:
            ``(tensor, carried)``. ``carried`` is opaque to this class and ``None`` here.

        Raises:
            ValidationError: the item's payload is not a tensor this element can submit.
        """
        payload = item.payload
        if not isinstance(payload, Tensor):
            raise ValidationError(
                f"pool element {self.name!r} was handed a payload of type "
                f"{type(payload).__name__} and needs a core Tensor to submit; the caps say "
                f"{item.caps}. A device-resident frame handle becomes a tensor in phase D "
                "(the DataPool)"
            )
        return payload, None

    def _finish(self, item: ChainItem, response: InferenceResponse, carried: Any) -> ChainItem:
        """File the model's answer on the successor item.

        The default files the raw outputs under :attr:`meta_key` — everything the model said,
        under this kind's name, for a consumer that knows the model.

        No ``caps=``: the payload is handed on unchanged, so the cap it carries is the cap it
        arrived with. Stamping ``output_caps[0]`` here would relabel a ``bgr@cpu`` frame as
        whatever this class declares first, and resolving the real outbound cap means knowing
        which edge the item is travelling — which is the loader's answer (``Edge.caps``), not
        this element's.
        """
        return item.derive(**{self.meta_key: response.outputs})


@dataclass(frozen=True, slots=True)
class _Letterbox:
    """The geometry of one frame's pre-processing, carried to its own decode.

    Per frame and on the walking worker's stack, never on the element: one ``PoolDetect``
    instance is shared by every pipeline worker, so an attribute here would let two frames in
    flight overwrite each other's scale and publish boxes computed from the wrong letterbox --
    a corruption with no exception and no visible symptom short of a tracker that swaps
    identities.

    The numbers are the ones the ops implementation *reported*, never recomputed from the
    shapes. Re-deriving them is where off-by-one box drift comes from, which is the whole
    reason :class:`~shipinfer.topology.base.LetterboxLike` returns them at all.
    """

    #: The source frame's ``(height, width)``, for clamping and for ``meta["frame_hw"]``.
    frame_hw: tuple[int, int]
    scale: float
    #: ``(pad_x, pad_y)`` in destination pixels.
    pad: tuple[float, float]
    #: ``(out_h, out_w)`` — the extent actually written inside the letterbox, before padding.
    #: Carried rather than left to the first consumer to re-derive, for the reason
    #: :class:`~shipinfer.topology.base.LetterboxLike` returns it at all: ``pad = (T - r) / 2``
    #: is the same for ``r`` and ``r + 1`` whenever ``T - r`` is even, so a consumer computing
    #: ``out_h`` from ``scale`` can disagree by a pixel while scale and pad both still match.
    #: The decode does not need it — it works in source pixels — and the first element that
    #: crops does, which is why it is here rather than plumbed in later.
    extents: tuple[int, int]


@registry_for(ElementKind.DETECT).register("pool")
class PoolDetect(_PoolElement):
    """Detection through the model pool: letterbox, submit, decode into source pixels.

    The only one of the four ``pool`` elements that transforms its payload, and it does the
    two halves of that transform in one place because they have to agree exactly. Pre-process
    scales and pads the frame to the model's input; the decode subtracts *those* pads and
    divides by *that* scale. Split across two elements, or recomputed from the shapes, and
    every published box drifts.

    This is the proven arithmetic from ``pipeline/graph/detect.py`` moved onto the chain, not
    a second implementation of it: the same
    :class:`~shipinfer.topology.elements.detections.decode_detections`, the same
    ``ImageOps.letterbox_batch``, the same order of operations. What is new is where the
    configuration comes from.

    **Why this element exists at all.** Before it, ``detect`` filed ``response.outputs`` raw
    under ``meta["boxes"]`` and nothing in the chain letterboxed anything — ``ChainFrameSink``
    submits the full-size frame as it was decoded (``runners/frames.py``). So the rows a
    detector returned were in the pixels of an input nobody had produced, and there was no key
    a tracker could read. ``meta["boxes"]`` is **gone** rather than kept beside
    ``meta["detections"]``: nothing in ``src/`` read it, and keeping it would pin the raw
    output tensor alive for the rest of the walk so that a future consumer could re-do
    arithmetic this element has already done correctly.

    **Where each knob comes from, in this order.** The model's ``config.yaml`` is the source
    of truth for anything the artefact knows — the input extent and the output tensor names —
    and this slot's ``params: {decode: {...}}`` overrides it for a deployment that knows
    better. Nothing is guessed: a model that declares neither a usable input shape nor a
    ``dst_size`` stops the deploy, because the alternative is a letterbox to the wrong size
    and a whole shard's boxes silently wrong.

    ``params: {decode: {...}}`` takes:

    * ``dst_size: [height, width]`` — the model input. Default: the declared input spec.
    * ``pad_value`` — the letterbox bar fill. Default 114, the YOLO grey; a detector trained
      with it and served without it is a silent accuracy loss.
    * ``normalize: {mean: [...], std: [...], swap_rb: bool}`` — see
      :class:`~shipinfer.topology.elements.detections.Normalization`.
    * ``score_threshold``, ``max_detections``, ``class_labels: {id: name}`` — see
      :class:`~shipinfer.topology.elements.detections.DecodeParams`.
    * ``boxes_output`` / ``count_output`` — the detector's output tensor names. Default: the
      model's single declared output, and ``num_detections`` only when the model *declares*
      it. Guessing a count name and finding nothing is indistinguishable from a model that
      reports no count, and the difference matters — the trailing rows of a padded output are
      undefined, not zero.
    """

    kind: ClassVar[ElementKind] = ElementKind.DETECT
    #: The decoded, source-pixel :class:`~shipinfer.topology.elements.detections.Detections`
    #: — the key ``track`` reads. Not the raw rows: see the class docstring.
    meta_key: ClassVar[str] = "detections"
    #: This element letterboxes, so it is opened with :attr:`ElementContext.ops` and refuses
    #: without one. The declaration is what makes ``shipinfer run`` resolve an implementation
    #: out of ``runtime.ops`` for a chain that contains one, and only for such a chain.
    needs_image_ops: ClassVar[bool] = True

    def __init__(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__(name, params, model=model)
        decode = self.params.get("decode") or {}
        if not isinstance(decode, Mapping):
            raise ConfigurationError(
                f"detect element {name!r}: `params: decode:` must be a mapping of decode "
                f"settings, got {type(decode).__name__}"
            )
        self._decode_params: Mapping[str, Any] = dict(decode)
        # Everything below is *resolved at open*, against the model this element was given:
        # the artefact knows its own input extent and output names, and asking it is what
        # keeps one repository running unchanged on two deployments (CONVENTIONS 2.6).
        self._ops: Any = None
        self._dst_size: tuple[int, int] = (0, 0)
        self._pad_value: int = _DEFAULT_PAD_VALUE
        self._normalize = Normalization()
        self._decode = DecodeParams()
        self._boxes_output = ""
        self._count_output: str | None = None

    # -- opening -----------------------------------------------------------------------

    def _do_open(self, context: ElementContext) -> None:
        """Resolve the pool, the ops and the geometry — all three now, none per frame.

        Raises:
            ConfigurationError: no model pool (the base class's refusal), no image ops, a
                model whose declared input does not say how big its input is and a slot that
                does not either, or a resolved letterbox the artefact contradicts
                (:meth:`_refuse_a_letterbox_the_model_disagrees_with`). Every one of them
                names what to pass.
            ModelNotFoundError: the pool has no such model.
        """
        super()._do_open(context)
        if context.ops is None:
            raise ConfigurationError(
                f"detect element {self.name!r} needs image ops and the runner passed none. "
                "A `pool` detector letterboxes the frame before it submits and undoes exactly "
                "that transform to put the boxes back in source pixels, so it is opened with "
                "an `ElementContext` carrying `ops=` — "
                "`shipinfer.runtime.ops.get_thread_local_image_ops(...)`, resolved by the "
                "process that builds the runner (`cli/commands/run.py`, `cli/shard.py`), the "
                "same way `models=` is. One per *thread*, which is why it is that call and not "
                "`get_image_ops()`: this element is shared by every worker walking the chain, "
                "and every implementation the latter returns is per-thread by contract. It "
                "is not defaulted here: `topology` may not import `runtime`, so a fallback "
                "would mean a second, "
                "unfused letterbox living in a pure layer — at 1000 frames a second that is a "
                "throughput cliff reported as a successful start-up"
            )
        self._ops = context.ops
        self._dst_size = self._resolve_dst_size()
        self._refuse_a_letterbox_the_model_disagrees_with()
        # Count first: it is identified by *name*, and knowing which output it is removes it
        # from the candidates for the rows. Resolved the other way round, a perfectly ordinary
        # detector that declares `output0` plus `num_detections` looks like two candidate row
        # tensors and gets refused.
        self._count_output = self._resolve_count_output()
        self._boxes_output = self._resolve_boxes_output()
        self._pad_value = int(self._decode_params.get("pad_value", _DEFAULT_PAD_VALUE))
        self._normalize = self._resolve_normalization()
        self._decode = self._resolve_decode_params()

    def _do_close(self) -> None:
        super()._do_close()
        # The ops belong to the runner, which resolved one bound to this shard's device; the
        # element only forgets it, so a reopened chain is handed the current one.
        self._ops = None

    def _declared(self, attribute: str) -> dict[str, Any]:
        """The model's declared ``input_specs``/``output_specs`` by name, ``{}`` if it says none.

        Read off the handle with :func:`getattr` rather than imported: the pool arrives as the
        structural :class:`~shipinfer.topology.base.ModelResolver`, so this layer never learns
        that ``shipinfer.repository`` exists — the same inversion that keeps ``topology``
        importable with no accelerator. A handle that carries no artefact (a test's fake, a
        backend that declares nothing) answers ``{}`` and the ``params:`` have to say.
        """
        config = getattr(getattr(self._handle, "artifact", None), "config", None)
        specs = getattr(config, attribute, None) or ()
        return {spec.name: spec for spec in specs}

    def _resolve_dst_size(self) -> tuple[int, int]:
        """The letterbox target: this slot's ``dst_size``, else the model's declared input.

        Raises:
            ConfigurationError: neither says, or either says something that is not two
                positive integers. Refused rather than defaulted to 640x640, because a
                letterbox to the wrong extent produces a frame the backend accepts on a
                dynamic-shape engine and boxes that are wrong on every camera.
        """
        declared = self._decode_params.get("dst_size")
        if declared is not None:
            return _extent(declared, f"detect element {self.name!r}: `decode.dst_size`")
        specs = self._declared("input_specs")
        spec = specs.get(self._input) or (
            next(iter(specs.values())) if len(specs) == 1 else None
        )
        extent = _static_extent(spec)
        if extent is not None:
            return extent
        shape = tuple(getattr(spec, "shape", ()) or ())
        raise ConfigurationError(
            f"detect element {self.name!r} cannot tell how big model {self.model!r} wants its "
            f"input: its declared inputs are {sorted(specs) or 'none'} and the one named "
            f"{self._input!r} is {shape or 'absent'}, which is not a static (3, H, W). Give the "
            "model a `config.yaml` that declares it, or say so on the slot: "
            "`params: {decode: {dst_size: [640, 640]}}`"
        )

    def _refuse_a_letterbox_the_model_disagrees_with(self) -> None:
        """Cross-check the resolved input name and extent against what the artefact declares.

        Two refusals the proven path made at start-up and this element had lost when the code
        moved out of ``pipeline/graph/stage.py`` (``validate``, which named both ends).

        **An ``input`` the model does not declare.** :meth:`_resolve_dst_size` falls back to
        "the single declared spec" when ``specs.get(self._input)`` misses, so a typo'd
        ``params: {input: pixels}`` on a single-input model resolved a perfectly good extent,
        opened successfully, and then failed inside the backend on every frame of the deploy.
        CONVENTIONS 2.6 is validate at start-up, not at first use.

        **A declared input the resolved letterbox cannot go into.** Asked of the spec itself,
        with :meth:`~shipinfer.core.types.TensorSpec.matches` — the same question
        ``stage.py`` asked of ``expected_row_shape`` — rather than of a hand-rolled predicate
        over the extent. It is one call and it covers three families at once: a
        ``decode.dst_size`` that contradicts a *static* ``(3, H, W)``; an input that is not
        image-shaped at all (``x[4]``, a rank-4 ``1x3x512x512``, an NHWC ``512x512x3``); and,
        by answering ``True``, the dynamic ``3x?x?`` that the override exists for. An earlier
        version of this guard read the extent with :func:`_static_extent` and so let the middle
        family through — a model that can never receive a letterboxed frame opened cleanly and
        failed inside the backend on every frame of the deploy.

        Left unchecked, the first family is the failure :meth:`_resolve_dst_size`'s own
        docstring gives as its reason for existing, one deployment further on: a static engine
        refuses the wrong extent loudly, a dynamic one accepts it and makes every box on every
        camera wrong.

        A handle that declares no inputs at all — a test's fake, a backend that declares
        nothing — is not second-guessed: there is nothing to disagree with, and the slot's
        ``params:`` are then the only statement of the truth.

        Raises:
            ConfigurationError: either disagreement, naming both ends.
        """
        specs = self._declared("input_specs")
        if not specs:
            return
        spec = specs.get(self._input)
        if spec is None:
            raise ConfigurationError(
                f"detect element {self.name!r} submits its frame as input {self._input!r} and "
                f"model {self.model!r} declares no such input (it declares {sorted(specs)}). "
                "Name the model's own input on the slot: `params: {input: <name>}` — or fix "
                "`ingest.input_name`, which is where an element that does not say gets it"
            )
        if not spec.matches((3, *self._dst_size)):
            raise ConfigurationError(
                f"detect element {self.name!r} would submit a letterboxed "
                f"(3, {self._dst_size[0]}, {self._dst_size[1]}) frame to model "
                f"{self.model!r}, which declares that input as {spec.describe()} — a shape "
                "that cannot receive one. `decode.dst_size` is the override for an engine "
                "whose input is *dynamic* (`3x?x?`); against a declared static extent it is a "
                "mistake a dynamic-shape engine would accept silently and answer with boxes "
                "that are wrong on every camera, and against an input that is not image-shaped "
                "at all it says this model is not a detector. Drop the override, name the "
                "model's own input (`params: {input: <name>}`), or fix its `config.yaml`"
            )

    def _resolve_boxes_output(self) -> str:
        """Which response output holds the rows.

        This slot's ``boxes_output``, else the model's single remaining declared output — an
        end-to-end detector has one, and ``yolo26n`` calls it ``output0`` — else ``boxes`` if a
        multi-output model declares that name, else a refusal that lists what the model does
        declare. "Remaining" means with the count output already taken out, which is why
        :meth:`_resolve_count_output` runs first: a detector declaring ``output0`` plus
        ``num_detections`` is the ordinary case, not an ambiguous one.

        A name is a property of the artefact, so guessing one refuses a valid engine over a
        naming preference.
        """
        declared = self._decode_params.get("boxes_output")
        if declared:
            return str(declared)
        specs = {
            name: spec
            for name, spec in self._declared("output_specs").items()
            if name != self._count_output
        }
        if len(specs) == 1:
            return next(iter(specs))
        if _DEFAULT_BOXES_OUTPUT in specs:
            return _DEFAULT_BOXES_OUTPUT
        raise ConfigurationError(
            f"detect element {self.name!r} cannot tell which output of model {self.model!r} "
            f"holds the detection rows: it declares {sorted(specs) or 'none'}. Name it on the "
            "slot: `params: {decode: {boxes_output: output0}}`"
        )

    def _resolve_count_output(self) -> str | None:
        """The output reporting how many rows were filled, or ``None`` if there is not one.

        Only trusted when this slot names it or the model *declares* it. Guessing the name and
        finding nothing is indistinguishable from a model that reports no count, and the
        difference matters: the trailing rows of a padded output are undefined, not zero, so
        reading them produces plausible boxes out of nothing.
        """
        declared = self._decode_params.get("count_output")
        if declared:
            return str(declared)
        if _DEFAULT_COUNT_OUTPUT in self._declared("output_specs"):
            return _DEFAULT_COUNT_OUTPUT
        return None

    def _resolve_normalization(self) -> Normalization:
        """The pre-processing normalisation this slot declares, or the module default."""
        declared = self._decode_params.get("normalize") or {}
        if not isinstance(declared, Mapping):
            raise ConfigurationError(
                f"detect element {self.name!r}: `decode.normalize` must be a mapping of "
                f"`mean`/`std`/`swap_rb`, got {type(declared).__name__}"
            )
        default = Normalization()
        try:
            return Normalization(
                mean=_triple(declared.get("mean", default.mean), "decode.normalize.mean"),
                std=_triple(declared.get("std", default.std), "decode.normalize.std"),
                swap_rb=bool(declared.get("swap_rb", default.swap_rb)),
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise ConfigurationError(f"detect element {self.name!r}: {exc}") from exc

    def _resolve_decode_params(self) -> DecodeParams:
        """The score threshold, the per-frame cap and the class-id table.

        Defaults mirror ``pipeline.score_threshold`` / ``max_detections`` / ``class_labels``,
        which is a literal in a pure layer for the reason the wait is: ``topology`` may not
        read the settings tree. What it must *not* do is invent a different number, so a test
        ties the two (``tests/topology/test_pool_detect_decode.py``).
        """
        default = DecodeParams()
        labels = self._decode_params.get("class_labels")
        if labels is not None and not isinstance(labels, Mapping):
            raise ConfigurationError(
                f"detect element {self.name!r}: `decode.class_labels` must be a mapping of "
                f"class id to label, got {type(labels).__name__}"
            )
        try:
            return DecodeParams(
                class_labels=(
                    default.class_labels
                    if labels is None
                    else {int(key): str(value) for key, value in labels.items()}
                ),
                score_threshold=float(
                    self._decode_params.get("score_threshold", default.score_threshold)
                ),
                max_detections=int(
                    self._decode_params.get("max_detections", default.max_detections)
                ),
            )
        except (ValueError, TypeError) as exc:
            raise ConfigurationError(f"detect element {self.name!r}: {exc}") from exc

    # -- one frame ---------------------------------------------------------------------

    def _prepare(self, item: ChainItem) -> tuple[Tensor, Any]:
        """Letterbox the frame to the model's input, and remember how.

        One call for resize-with-pad, colour convert, normalise and NHWC->NCHW, because those
        four steps read and write the same pixels and doing them separately is four passes
        over a 1080p frame (``runtime/ops/base.py``).

        Raises:
            ValidationError: the payload is not a host-resident ``(1, H, W, 3)`` or
                ``(H, W, 3)`` **uint8** frame. A device-resident handle is refused by name
                rather than downloaded: an implicit device-to-host copy per frame is exactly the cost
                arch.md section 8 makes the caps refuse at load time, and it becomes a real
                submission with the DataPool (phase D).
        """
        image = self._frame_of(item)
        letterboxed = self._ops.letterbox_batch(
            [image], self._dst_size, self._normalize, pad_value=self._pad_value
        )
        geometry = _Letterbox(
            frame_hw=(int(image.shape[0]), int(image.shape[1])),
            scale=float(letterboxed.scales[0]),
            pad=(float(letterboxed.pads[0][0]), float(letterboxed.pads[0][1])),
            extents=(int(letterboxed.extents[0][0]), int(letterboxed.extents[0][1])),
        )
        return Tensor.from_numpy(letterboxed.tensor), geometry

    def _frame_of(self, item: ChainItem) -> np.ndarray:
        """One ``(H, W, 3)`` uint8 frame out of the item's payload.

        The dtype is checked and not merely documented: every ``ImageOps`` implementation
        writes source pixels into a uint8 canvas, so a float32 payload is *truncated* into it
        rather than refused — a frame in the 0-1 scale becomes an all-black letterbox and the
        detector answers with nothing on every camera, which reads as an empty scene.
        """
        payload = item.payload
        if not isinstance(payload, Tensor):
            raise ValidationError(
                f"detect element {self.name!r} was handed a payload of type "
                f"{type(payload).__name__} and needs a core Tensor to letterbox; the caps say "
                f"{item.caps}. A device-resident frame handle becomes a tensor in phase D "
                "(the DataPool)"
            )
        if payload.host is None:
            raise ValidationError(
                f"detect element {self.name!r} was handed a device-resident payload "
                f"({payload.describe()}) and letterboxes on the host; the caps say "
                f"{item.caps}. Downloading it here would be a per-frame device-to-host copy "
                "nothing asked for — the device path arrives with the DataPool (phase D)"
            )
        array = payload.numpy()
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValidationError(
                f"detect element {self.name!r} needs one (H, W, 3) frame to letterbox and the "
                f"payload is {payload.describe()}. One chain item is one frame, so a batch "
                "dimension above 1 is a producer that assembled a batch this element does not "
                "scatter"
            )
        if array.dtype != np.uint8:
            raise ValidationError(
                f"detect element {self.name!r} needs one (H, W, 3) uint8 frame to letterbox "
                f"and the payload is {payload.describe()}. Source pixels are letterboxed into "
                "a uint8 canvas and normalised from the 0-255 scale, so an already-scaled "
                "float frame is truncated to black rather than refused — and a detector that "
                "sees black answers with nothing on every camera"
            )
        return array

    def _finish(self, item: ChainItem, response: InferenceResponse, carried: Any) -> ChainItem:
        """Decode the rows into source-frame pixels and file them under ``detections``.

        Also files ``meta["frame_hw"]``: the boxes are in the source frame's pixels, and the
        elements behind this one — ``track`` first — need the extent those pixels are measured
        in without re-deriving it from a payload that may already be gone.

        Raises:
            InferenceError: the model returned no such output, or returned rows whose layout is
                not one frame of ``(rows, 6)``. Raised rather than reshaped: a detector whose
                output layout changed is a deployment error, and guessing at the new one would
                attach scores to coordinates.
            ValidationError: the rows are the right rank and the wrong width — the decode's own
                refusal, which names the layout it expects.
        """
        geometry: _Letterbox = carried
        rows = response.outputs.get(self._boxes_output)
        if rows is None:
            raise InferenceError(
                f"detect element {self.name!r}: model {self.model!r} returned no output "
                f"{self._boxes_output!r} (got: {sorted(response.outputs)})"
            )
        array = rows.numpy()
        if array.ndim != 3 or array.shape[0] != 1:
            raise InferenceError(
                f"detect element {self.name!r}: model {self.model!r} returned "
                f"{self._boxes_output!r} as {rows.describe()} and one frame's rows are "
                "(1, rows, 6)"
            )
        count: int | None = None
        if self._count_output is not None:
            reported = response.outputs.get(self._count_output)
            if reported is not None:
                count = int(reported.numpy().reshape(-1)[0])
        detections = decode_detections(
            array[0],
            params=self._decode,
            scale=geometry.scale,
            pad=geometry.pad,
            frame_hw=geometry.frame_hw,
            count=count,
        )
        return item.derive(
            **{self.meta_key: detections},
            frame_hw=geometry.frame_hw,
        )


def _static_extent(spec: Any) -> tuple[int, int] | None:
    """The ``(height, width)`` a spec pins for a ``(3, H, W)`` input, or ``None`` for none.

    ``None`` covers "no such spec", "not an image-shaped input" and "a dynamic dimension"
    alike, and collapsing the three is the point: all of them mean *the artefact does not say*,
    which is the condition under which the slot's ``decode.dst_size`` is the only statement of
    the truth.

    This *extracts*, which is why it exists at all and why it has exactly one caller,
    :meth:`PoolDetect._resolve_dst_size`. The cross-check that follows the resolution
    (:meth:`PoolDetect._refuse_a_letterbox_the_model_disagrees_with`) *validates*, and asks
    :meth:`~shipinfer.core.types.TensorSpec.matches` instead — a predicate written here would
    have to agree with the library's, and the version that tried was narrower than it in a way
    that let a non-image-shaped input open.
    """
    shape = tuple(getattr(spec, "shape", ()) or ())
    if len(shape) == 3 and shape[0] == 3 and shape[1] > 0 and shape[2] > 0:
        return int(shape[1]), int(shape[2])
    return None


def _extent(value: Any, what: str) -> tuple[int, int]:
    """``value`` as a positive ``(height, width)``, or a ``ConfigurationError`` naming it."""
    try:
        height, width = (int(part) for part in value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{what} must be [height, width], got {value!r}") from exc
    if height < 1 or width < 1:
        raise ConfigurationError(f"{what} must be positive, got [{height}, {width}]")
    return height, width


def _triple(value: Any, what: str) -> tuple[float, float, float]:
    """``value`` as three floats — one per channel."""
    try:
        first, second, third = (float(part) for part in value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{what} must be three numbers, got {value!r}") from exc
    return first, second, third


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
