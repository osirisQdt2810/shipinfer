# doc: long the wildcard-cap rule and the two-hooks split are both load-bearing and unobvious
"""The ``pool`` elements: a chain step that submits to the model pool and waits.

Four kinds run a model from the repository — ``detect``, ``segment``, ``embed``,
``recognize`` — and for all four the default implementation is this one: build a request,
hand it to the engine, sleep until the answer arrives, add the outputs to the item's meta.
The alternatives behind the same kinds (``nvinfer``, ``nvinferserver``) run the model
elsewhere entirely, which is why this is a registry and not a base class with a flag.

**Why this lives in a pure layer.** It never imports the engine: the pool arrives as
:class:`~shipinfer.topology.base.ModelResolver` — one method, ``get(name)`` — which the
engine satisfies structurally and a test satisfies with a dict. The name is resolved once at
``open``, not per frame: ``get`` takes a lock, and a missing model must stop the deploy rather
than fail on the first frame (CONVENTIONS 2.6).

**It hands on the caps it was handed.** Accepts ``nv12@gpu``, then ``tensor@gpu``, then
``bgr@cpu``; *produces* ``*@*``, which is a precise claim rather than vagueness: this element
adds a meta key and passes the payload through **unchanged**, so its outbound cap is its
negotiated inbound one. Declaring a concrete ``produces: nv12@gpu`` — which this file did —
was a **relabelling**: fed ``bgr@cpu`` it told the loader host memory was device memory, and
the download that arch.md section 8 exists to refuse went invisible one element later with
every edge reported valid. The corollary: :meth:`_PoolElement._do_process` must not stamp a
cap on the item it derives, because a cap belongs to the edge and the edge is the loader's.

**Two of the four read pixels; two forward.** A detector must letterbox and then undo exactly
that transform to put boxes back in source pixels; an embedder must cut the frame into one
crop per detection and put the vectors back on the rows they came from. Both replace
``_prepare`` and ``_finish`` and declare ``needs_image_ops``. That split — rather than a flag
or an ``isinstance`` inside one method — is what keeps the per-frame geometry on the walking
worker's stack: one element instance is shared by every worker, so an attribute would let two
frames overwrite each other's scale, or each other's row indices, which is an appearance
vector attached to the wrong object.

**The embedder is where the chain's cardinality changes**: one frame in, N crops at the
model, N vectors back. It answers two questions no forwarding element has to — *which rows*
(its own ``params: classes:``, because ``when:`` decides frames, not rows) and *how many
requests* (``max_batch_size``, because a crowded frame outgrows a fixed plan).

**Two knobs**, resolved at ``open`` in a fixed precedence: this slot's ``params:``, then the
runner's context (``stage_timeout_s``, ``input_name``), then the module default. Params win
because a tensor name belongs to the model; the context wins over the default because an
operator who lowers ``stage_timeout_ms`` means it — before the context carried it, every
``pool`` element still waited 5 s and the settings key applied to nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from shipinfer.core.errors import (
    ConfigurationError,
    InferenceError,
    RequestTimeoutError,
    ValidationError,
)
from shipinfer.core.request import InferenceRequest, InferenceResponse
from shipinfer.core.types import Tensor
from shipinfer.topology.base import (
    ChainItem,
    Element,
    ElementContext,
    ElementKind,
    RowIndexed,
)
from shipinfer.topology.elements.detections import (
    DecodeParams,
    Detections,
    Normalization,
    decode_detections,
    parse_classes,
)
from shipinfer.topology.registry import registry_for

if TYPE_CHECKING:  # pragma: no cover - typing only
    from shipinfer.core.metrics import Counter, Histogram

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

#: The crop batch of a *never opened* crop element, and nothing else: :meth:`_do_open`
#: replaces it per instance with one of the extent that instance resolved, and
#: :meth:`~shipinfer.topology.base.Element.process` refuses before ``open``, so no frame ever
#: sees this one. It exists only so the attribute has a value and a type from ``__init__``
#: rather than an ``Optional`` :meth:`_PoolCropElement._prepare` would have to unwrap.
_EMPTY_CROPS = Tensor.from_numpy(np.empty((0, 3, 1, 1), dtype=np.float32))


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
            precedence. :class:`PoolDetect` reads a third, ``decode``, which is a mapping and is
            documented on that class.
        model: the repository model to run. Required — the loader refuses an element whose
            :attr:`~shipinfer.topology.base.Element.requires_model_name` is set and names none,
            so a missing one here is a programming error and says so.
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
    #: The default ``_finish`` files the response verbatim, so a consumer that reads this
    #: element's key per detection row cannot be satisfied by it — the loader refuses that
    #: pairing rather than letting it fail on frame 1.
    files_raw_response: ClassVar[bool] = True

    meta_key: ClassVar[str] = ""

    #: What this element does with the frame's pixels, for :meth:`_frame_of`'s refusals: a
    #: detector ``letterbox``es and an embedder ``crop``s. Only the two subclasses that read
    #: pixels ever reach that method; the default is the neutral word, so a third one that
    #: forgets to declare its own still produces a sentence rather than a blank.
    _frame_verb: ClassVar[str] = "submit"

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
                model name. Both are wiring mistakes in the process that built the runner, not
                in the operator's chain file, and the message says which.
            ModelNotFoundError: the pool has no such model — the resolver's own typed error,
                raised here rather than caught, because it already names the model and lists the
                ones that exist.
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

        Build the request, wait on the bound, propagate every refusal as itself. The two hooks a
        subclass replaces are :meth:`_prepare` and :meth:`_finish`; a detector replaces both.

        Exactly one subclass replaces this method: :class:`_PoolCropElement` submits *no*
        request on a frame with nothing to crop and more than one on a frame past
        ``max_batch_size``. Both still go through :meth:`_submit`, so the wait has one
        definition however many requests a frame costs.

        The request carries the item's ``RequestContext`` **by identity**, not a copy: that tag
        is what reassembly, tracing and every log line group on (ADR-002).

        **A timeout abandons the request; it does not cancel it.** There is no cancellation path
        to call — an item that has been queued is dequeued, assembled and executed whatever its
        future says, and only the *result* is discarded. So a timed-out frame still costs the
        instance slot that made it late, and under sustained overload that compounds into a
        queue that never drains rather than one that sheds. The bound is still worth having,
        because it frees the *worker*; real cancellation needs a queue-side removal and is
        deliberately not guessed here.

        Returns:
            The successor item — same tag, same payload, and whatever :meth:`_finish` added.

        Raises:
            QueueFullError: the pool is saturated — backpressure, propagated untouched.
            RequestTimeoutError: the pool did not answer within ``timeout_s``.
            ValidationError: from :meth:`_prepare` or :meth:`_finish`; a mis-wired chain.
            InferenceError: the model answered and the answer cannot be attributed to rows.

        Note:
            All four propagate **as themselves**, and that is the contract:
            :meth:`~shipinfer.runners.walk.ChainWalk.count_failure` sorts them into three
            counters — backpressure, timed-out, failed — so collapsing them into one type here
            would collapse the numbers an operator acts on.
        """
        tensor, carried = self._prepare(item)
        return self._finish(item, self._submit(item, tensor), carried)

    def _submit(self, item: ChainItem, tensor: Tensor) -> InferenceResponse:
        """One request for ``tensor``, waited on under this element's bound.

        Extracted from :meth:`_do_process` rather than left inline because a *crop* element
        submits this same request more than once for one frame — a frame holding 25 people
        against a plan built at batch 16 is two requests, not one that the backend refuses
        (:class:`_PoolCropElement`). The wait, the tag and the timeout message are identical
        for all of them, and a second copy of them is a second place for the bound to drift.

        Raises:
            RequestTimeoutError: the pool did not answer within ``timeout_s``.
            QueueFullError: the pool is saturated. Propagated untouched.
        """
        request = InferenceRequest(
            model_name=self.model or "",
            inputs={self._input: tensor},
            context=item.context,
        )
        future = self._handle.infer(request)
        try:
            return future.result(self._timeout_s)
        except FutureTimeoutError as exc:
            raise RequestTimeoutError(
                f"model {self.model!r} did not answer element {self.name!r} for "
                f"{item.key} within {self._timeout_s}s"
            ) from exc

    # -- what the artefact says, and what the payload holds ----------------------------

    def _declared(self, attribute: str) -> dict[str, Any]:
        """The model's declared ``input_specs``/``output_specs`` by name.

        ``{}`` when the artefact declares none.

        Read off the handle with :func:`getattr` rather than imported: the pool arrives as the
        structural :class:`~shipinfer.topology.base.ModelResolver`, so this layer never learns
        that ``shipinfer.repository`` exists — the same inversion that keeps ``topology``
        importable with no accelerator. A handle that carries no artefact (a test's fake, a
        backend that declares nothing) answers ``{}`` and the ``params:`` have to say.
        """
        config = getattr(getattr(self._handle, "artifact", None), "config", None)
        specs = getattr(config, attribute, None) or ()
        return {spec.name: spec for spec in specs}

    def _max_batch_rows(self) -> int:
        """The rows one request may carry: the engine's bound, never ``None``.

        Read off the artefact for the reason :meth:`_declared` is — the number belongs to the
        engine, not to the chain file — and it is ``effective_max_batch_size``, because that
        is what the assembler is built with (``engine/model.py``) and therefore what a request
        is refused against.

        ``max_batch_size: 0`` is **not** "no bound" here. Triton reads it as server-side
        batching off; this engine reads it as ``max_batch_size or 1``
        (``repository/model_config.py``), and ``0`` is what a ``config.yaml`` gets by
        *omission* — so an unchunked frame of 15 people is refused whole, every frame. The
        fallback is ``1``: one crop per request beats every crop lost.
        """
        config = getattr(getattr(self._handle, "artifact", None), "config", None)
        limit = getattr(config, "effective_max_batch_size", None) or getattr(
            config, "max_batch_size", None
        )
        return int(limit) if limit else 1

    def _frame_of(self, item: ChainItem) -> np.ndarray:
        """One ``(H, W, 3)`` uint8 frame out of the item's payload.

        Shared by the two elements that read pixels rather than hand them on — the detector
        letterboxes, the embedder crops — because the four refusals below are the same four
        for both and a second copy is a second place for them to drift. :attr:`_frame_verb`
        is what each one calls its use of the frame, so the message names the act that failed.

        This is also the cap negotiation's *second* half. ``accepts`` lists ``nv12@gpu`` and
        ``tensor@gpu`` ahead of ``bgr@cpu`` because the device path is the default end to end
        (arch.md §8) and a ``pool`` element that only forwards its payload genuinely accepts
        all three. An element that reads the pixels on the host does not, and the honest place
        to say so is here rather than in a narrowed ``accepts``: narrowing it would refuse the
        chain phase D makes work, and staying silent would download six megabytes per frame
        that nobody asked for. So the declaration stays and the *frame* is refused, by name,
        with the phase that fixes it.

        The dtype is checked and not merely documented: every ``ImageOps`` implementation
        writes source pixels into a uint8 canvas, so a float32 payload is *truncated* into it
        rather than refused — a frame in the 0-1 scale becomes all black and the model answers
        with nothing on every camera, which reads as an empty scene.
        """
        payload = item.payload
        verb = self._frame_verb
        if not isinstance(payload, Tensor):
            raise ValidationError(
                f"{self.kind.value} element {self.name!r} was handed a payload of type "
                f"{type(payload).__name__} and needs a core Tensor to {verb}; the caps say "
                f"{item.caps}. A device-resident frame handle becomes a tensor in phase D "
                "(the DataPool)"
            )
        if payload.host is None:
            raise ValidationError(
                f"{self.kind.value} element {self.name!r} was handed a device-resident payload "
                f"({payload.describe()}) and {verb}s on the host; the caps say "
                f"{item.caps}. Downloading it here would be a per-frame device-to-host copy "
                "nothing asked for — the device path arrives with the DataPool (phase D)"
            )
        array = payload.numpy()
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValidationError(
                f"{self.kind.value} element {self.name!r} needs one (H, W, 3) frame to {verb} "
                f"and the payload is {payload.describe()}. One chain item is one frame, so a "
                "batch dimension above 1 is a producer that assembled a batch this element "
                "does not scatter"
            )
        if array.dtype != np.uint8:
            raise ValidationError(
                f"{self.kind.value} element {self.name!r} needs one (H, W, 3) uint8 frame to "
                f"{verb} and the payload is {payload.describe()}. Source pixels are written "
                "into a uint8 canvas and normalised from the 0-255 scale, so an already-scaled "
                "float frame is truncated to black rather than refused — and a model that "
                "sees black answers with nothing on every camera"
            )
        return array

    # -- the two halves a subclass may replace -----------------------------------------
    #
    # Split out of `_do_process` rather than left inline, and the seam is where it is for one
    # reason: `PoolDetect` has to *undo* in the second half exactly the transform it applied
    # in the first, and the numbers that describe that transform belong to one frame. An
    # element attribute would be wrong — one element instance is shared by every pipeline
    # worker (`runners/inprocess.py`), so two frames in flight would overwrite each other's
    # scale and pad and publish boxes computed from the wrong letterbox. So the first half
    # *returns* what the second half needs, the walk carries it on its own stack, and no
    # subclass can accidentally make it shared state.
    #
    # The alternative — `if isinstance(self, PoolDetect)` inside `_do_process`, or a
    # `decodes: bool` flag — is the switch statement CONVENTIONS 2.3 exists to refuse.

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
    flight overwrite each other's scale and publish boxes computed from the wrong letterbox —
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

    The only one of the four that transforms its payload, and it does both halves of the
    transform in one place because they must agree exactly: pre-process scales and pads the
    frame to the model's input, and the decode subtracts *those* pads and divides by *that*
    scale. Split across two elements, or recomputed from shapes, and every published box drifts.

    This is the proven arithmetic from ``pipeline/graph/detect.py`` moved onto the chain rather
    than reimplemented — the same ``decode_detections``, the same ``letterbox_batch``, the same
    order. What is new is where the configuration comes from.

    **Nothing is guessed.** The model's ``config.yaml`` is the source of truth for what the
    artefact knows — input extent, output tensor names — and this slot's
    ``params: {decode: {...}}`` overrides it for a deployment that knows better. A model that
    declares neither a usable input shape nor a ``dst_size`` stops the deploy, because the
    alternative is a letterbox to the wrong size and a whole shard's boxes silently wrong.

    ``params: {decode: {...}}`` takes:

    * ``dst_size: [height, width]`` — the model input. Default: the declared input spec.
    * ``pad_value`` — letterbox fill, default 114 (the YOLO grey; trained with it and served
      without it is a silent accuracy loss).
    * ``normalize``, ``score_threshold``, ``max_detections``, ``class_labels`` — see
      :mod:`~shipinfer.topology.elements.detections`.
    * ``boxes_output`` / ``count_output`` — default to the model's declared output, and
      ``num_detections`` only when the model *declares* it. Guessing a count name and finding
      nothing is indistinguishable from a model that reports no count, and the trailing rows of
      a padded output are undefined rather than zero.
    """

    kind: ClassVar[ElementKind] = ElementKind.DETECT
    #: The decoded, source-pixel :class:`~shipinfer.topology.elements.detections.Detections`
    #: — the key ``track`` reads. Not the raw rows: see the class docstring.
    #: Decoded and filed as this frame's detections, not as the raw response.
    files_raw_response: ClassVar[bool] = False

    meta_key: ClassVar[str] = "detections"
    #: This element letterboxes, so it is opened with :attr:`ElementContext.ops` and refuses
    #: without one. The declaration is what makes ``shipinfer run`` resolve an implementation
    #: out of ``runtime.ops`` for a chain that contains one, and only for such a chain.
    needs_image_ops: ClassVar[bool] = True
    _frame_verb: ClassVar[str] = "letterbox"

    def detection_labels(self) -> tuple[str, ...] | None:
        """The label vocabulary this detector was configured with, or ``None`` if it said none.

        Answered from ``params:`` alone and before ``open()``, because the loader asks it while
        validating a chain on a host with no pool — which is also why a malformed table is
        ``None`` here rather than a refusal: :meth:`_resolve_decode_params` raises for that at
        ``open()``, with the message that names the key, and two refusals for one typo is one
        too many.
        """
        labels = self._decode_params.get("class_labels")
        if not isinstance(labels, Mapping):
            return None
        return tuple(str(value) for value in labels.values())

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
            ConfigurationError: no model pool (the base class's refusal), no image ops, a model
                whose declared input does not say how big its input is and a slot that does not
                either, or a resolved letterbox the artefact contradicts
                (:meth:`_refuse_a_letterbox_the_model_disagrees_with`). Every one of them names
                what to pass.
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

    def _resolve_dst_size(self) -> tuple[int, int]:
        """The letterbox target: this slot's ``dst_size``, else the model's declared input.

        Raises:
            ConfigurationError: neither says, or either says something that is not two positive
                integers. Refused rather than defaulted to 640x640, because a letterbox to the
                wrong extent produces a frame the backend accepts on a dynamic-shape engine and
                boxes that are wrong on every camera.
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
            f"{self._input!r} is {shape or 'absent'}, which is not a static (3, H, W). Give "
            "the model a `config.yaml` that declares it, or say so on the slot: "
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
        return _normalization(
            self._decode_params.get("normalize"),
            f"detect element {self.name!r}: `decode.normalize`",
        )

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
            ValidationError: the payload is not a host-resident ``(1, H, W, 3)`` or ``(H, W,
                3)`` **uint8** frame. A device-resident handle is refused by name rather than
                downloaded: an implicit device-to-host copy per frame is exactly the cost
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


def _normalization(declared: Any, what: str) -> Normalization:
    """A slot's ``normalize:`` block as a :class:`Normalization`, or the module default.

    One function rather than one method per element: the detector's ``decode.normalize`` and
    the embedder's ``crop.normalize`` are the same three keys with the same defaults and the
    same refusals, and the only thing that differs is which key path the message names — which
    is what ``what`` carries. A per-element copy would be a third place for the default mean
    to drift from :class:`Normalization`'s.

    Raises:
        ConfigurationError: not a mapping, or a value that is not three numbers, or a zero in
            ``std`` (:class:`Normalization`'s own refusal, re-raised with the key path).
    """
    block = declared or {}
    if not isinstance(block, Mapping):
        raise ConfigurationError(
            f"{what} must be a mapping of `mean`/`std`/`swap_rb`, got {type(block).__name__}"
        )
    default = Normalization()
    try:
        return Normalization(
            mean=_triple(block.get("mean", default.mean), f"{what}.mean"),
            std=_triple(block.get("std", default.std), f"{what}.std"),
            swap_rb=bool(block.get("swap_rb", default.swap_rb)),
        )
    except (ValidationError, ValueError, TypeError) as exc:
        raise ConfigurationError(f"{what}: {exc}") from exc


def _triple(value: Any, what: str) -> tuple[float, float, float]:
    """``value`` as three floats — one per channel."""
    try:
        first, second, third = (float(part) for part in value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{what} must be three numbers, got {value!r}") from exc
    return first, second, third


class _CropMetrics:
    """The crop element's metric handles, resolved once at ``open``.

    The same null-object shape as :class:`~shipinfer.topology.elements.track._TrackMetrics`,
    and for the same reason: at 15 000 crops a second a metric looked up by string per frame
    is a hash and a dict probe nobody needs to pay for, and ``context.metrics is None`` should
    be answered once rather than by an ``if`` on the per-frame path.

    ``None`` means the runner offered no registry, and then nothing is counted rather than a
    private registry being minted — a metric no exporter scrapes reads as evidence and is
    worse than an absent one (:class:`~shipinfer.topology.base.ElementContext`).
    """

    __slots__ = ("element", "per_frame", "rows")

    #: Object counts, not microseconds, so the latency buckets are the wrong ruler. The
    #: sizing this project targets is 10-20 objects a frame (CLAUDE.md), so the interesting
    #: resolution is around there and the tail is what says a frame ran into ``max_batch_size``.
    BUCKETS: ClassVar[tuple[float, ...]] = (0, 1, 2, 4, 8, 16, 32, 64, 128)

    def __init__(self, registry: Any, element: str) -> None:
        self.element = element
        histogram = getattr(registry, "histogram", None)
        if histogram is None:
            self.per_frame: Histogram | None = None
            self.rows: Counter | None = None
            return
        self.per_frame = registry.histogram(
            "shipinfer_element_crops_per_frame",
            "Crops one element cut from one frame, per element. The fan-out arch.md section 5 "
            "branches on: a frame is one row here and N rows at the model, so this is what "
            "says whether the embedder's queue depth is a scheduling problem or simply a "
            "crowded scene. The tail above `max_batch_size` is the frames that cost two "
            "requests instead of one.",
            self.BUCKETS,
        )
        self.rows = registry.counter(
            "shipinfer_element_crops_total",
            "Crops submitted by one element since start-up, per element. Divided by the "
            "frames that element saw it is the mean fan-out; against the model's own request "
            "counter it is the check that no row was dropped between the crop and the submit.",
        )

    def frame(self, crops: int) -> None:
        """One frame's fan-out. Called once per frame, including the zero-crop frames.

        The zeros are recorded rather than skipped: "this camera has nobody in it" and "the
        embedder is not running" are the two things this histogram has to tell apart, and a
        gauge that only moves when there is work cannot.
        """
        if self.per_frame is not None:
            self.per_frame.observe(crops, element=self.element)
        if self.rows is not None and crops:
            self.rows.inc(crops, element=self.element)


class _PoolCropElement(_PoolElement):
    """A ``pool`` element that runs its model on **one crop per detection**, not the frame.

    The element that changes cardinality: one frame in, N rows at the model, N vectors back,
    scattered onto the rows they came from — arch.md section 5's "branch on class -> crop
    batch -> submit crops". The same single ``crop_batch`` call for all N boxes, the same
    chunking at ``max_batch_size``, the same rule that every row knows its detection.

    **Why not the whole frame.** ``embed`` used to hand the detector's letterboxed frame to a
    re-identification model whose input is 3x256x128, and file the raw ``response.outputs``
    under ``meta['vectors']`` — which ``track`` refuses, because an embedder's raw output
    tensors are not an attribution. So the chain could not reach ``track`` with appearance.

    **The scatter-back is a mapping, and that is load-bearing.** ``track`` accepts one row per
    detection or ``{detection index: vector}``. A per-row array would need a filler for the rows
    this slot did not embed — and the sizing puts two embedders side by side, so *partial
    coverage is the normal case*. The mapping says exactly which rows were covered and stays
    legal when two of these merge into one frame's meta; a NaN row would have to be recognised
    as absence by every consumer, and the first one that forgot would match a track against a
    vector of NaNs and never say so.

    **Which rows is its own ``params: classes:``, not the chain's ``when:``.** A ``when:`` guard
    is evaluated once per *item*, and an item is a whole frame — so it can only decide whether
    the element runs at all. A frame holds ships and people at once, which is why row selection
    cannot be a frame guard.

    ``params:`` takes, on top of the base's ``input`` and ``timeout_s``:

    * ``classes: [ship]`` — which detection labels to crop. Default: every row.
    * ``crop: {size: [h, w]}`` — the extent each crop is resized to. Default: the model's own
      declared input. **The one an operator has to know about**: a model with a dynamic input
      declares no extent, so without this the deploy is refused at ``open()`` — and a wrong
      extent is the silent failure, accepted by the backend and answered with a vector
      computed from the wrong pixels.
    * ``crop: {normalize: ...}`` — the normalisation applied to each crop, as ``PoolDetect``'s
      ``decode: {normalize: ...}``.
    * ``output: <name>`` — which response output holds one row per crop. Default: the model's
      single output, and required when it has more than one.
    """

    #: A crop element scatters its response back onto the rows it cropped, so its key is
    #: row-indexed and readable by ``output`` — the opposite of the base's default.
    files_raw_response: ClassVar[bool] = False

    #: This element crops, so it is opened with :attr:`ElementContext.ops` and refuses without
    #: one — the same declaration, and the same refusal, as :class:`PoolDetect`.
    needs_image_ops: ClassVar[bool] = True
    #: It selects which rows to crop with ``params: classes:``, so the loader refuses a
    #: ``when: class == …`` on this slot and names the key that does the job — see
    #: :attr:`~shipinfer.topology.base.Element.selects_rows`.
    selects_rows: ClassVar[bool] = True
    _frame_verb: ClassVar[str] = "crop"

    def __init__(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__(name, params, model=model)
        crop = self.params.get("crop") or {}
        if not isinstance(crop, Mapping):
            raise ConfigurationError(
                f"{self.kind.value} element {name!r}: `params: crop:` must be a mapping of "
                f"crop settings, got {type(crop).__name__}"
            )
        self._crop_params: Mapping[str, Any] = dict(crop)
        self._classes = parse_classes(
            self.params.get("classes"), f"{self.kind.value} element {self.name!r}"
        )
        # Resolved at open against the model this element was given, for the reason
        # `PoolDetect`'s are: the artefact knows its own input extent and output names.
        self._ops: Any = None
        self._crop_size: tuple[int, int] = (0, 0)
        self._normalize = Normalization()
        self._max_rows: int | None = None
        self._output = ""
        self._metrics = _CropMetrics(None, name)
        # Bound once, and re-bound at `open` when the real handles arrive: `self._metrics.frame`
        # written at the call site mints a bound-method object on every frame, and this one is
        # on the per-frame path at a thousand frames a second (CONVENTIONS 2.5). The same
        # binding, for the same reason, as `track`'s `_on_implicit_reset`.
        self._on_frame: Callable[[int], None] = self._metrics.frame
        self._nothing_to_crop = _EMPTY_CROPS

    # -- opening -----------------------------------------------------------------------

    def _do_open(self, context: ElementContext) -> None:
        """Resolve the pool, the ops, the crop extent, the output name and the row bound.

        All five now, none per frame — the rule :class:`PoolDetect` states and the reason
        ``max_batch_size`` is read here rather than at the first crowded frame.

        Raises:
            ConfigurationError: no model pool (the base class's refusal), no image ops, a model
                whose declared input does not say how big a crop it wants and a slot that does
                not either, a crop extent the artefact contradicts, or an output this element
                cannot identify. Every one of them names what to pass.
            ModelNotFoundError: the pool has no such model.
        """
        super()._do_open(context)
        if context.ops is None:
            raise ConfigurationError(
                f"{self.kind.value} element {self.name!r} needs image ops and the runner "
                "passed none. A `pool` crop element cuts one crop per detection out of the "
                "source frame before it submits, so it is opened with an `ElementContext` "
                "carrying `ops=` — "
                "`shipinfer.runtime.ops.get_thread_local_image_ops(...)`, resolved by the "
                "process that builds the runner (`cli/commands/run.py`, `cli/shard.py`), the "
                "same way `models=` is. One per *thread*, which is why it is that call and "
                "not `get_image_ops()`: this element is shared by every worker walking the "
                "chain, and every implementation the latter returns is per-thread by "
                "contract. It is not defaulted here: `topology` may not import `runtime`, so "
                "a fallback would mean a per-crop resize loop in Python living in a pure "
                "layer — at 15 000 crops a second that is a throughput cliff reported as a "
                "successful start-up"
            )
        self._ops = context.ops
        self._crop_size = self._resolve_crop_size()
        self._refuse_a_crop_the_model_disagrees_with()
        self._normalize = _normalization(
            self._crop_params.get("normalize"),
            f"{self.kind.value} element {self.name!r}: `crop.normalize`",
        )
        self._max_rows = self._max_batch_rows()
        self._output = self._resolve_output()
        self._metrics = _CropMetrics(context.metrics, self.name)
        self._on_frame = self._metrics.frame
        # Built once, here, so a quiet camera's frame does not build one: `_prepare` has to
        # answer with a tensor and this is the one it answers with when there is nothing to
        # crop. It is never submitted and never written to, so one instance shared by every
        # worker is safe — which an array that a kernel wrote into would not be.
        self._nothing_to_crop = Tensor.from_numpy(
            np.empty((0, 3, *self._crop_size), dtype=np.float32)
        )

    def _do_close(self) -> None:
        super()._do_close()
        # The ops belong to the runner, which resolved one bound to this shard's device; the
        # element only forgets it, so a reopened chain is handed the current one.
        self._ops = None

    def _resolve_crop_size(self) -> tuple[int, int]:
        """One crop's ``(height, width)``: this slot's ``crop.size``, else the model's input.

        Raises:
            ConfigurationError: neither says, or either says something that is not two positive
                integers. Refused rather than defaulted, for the reason
                :meth:`PoolDetect._resolve_dst_size` is: a crop resized to the wrong extent is a
                frame a dynamic-shape engine accepts and answers with vectors that are wrong for
                every object on every camera.
        """
        declared = self._crop_params.get("size")
        if declared is not None:
            return _extent(declared, f"{self.kind.value} element {self.name!r}: `crop.size`")
        specs = self._declared("input_specs")
        spec = specs.get(self._input) or (
            next(iter(specs.values())) if len(specs) == 1 else None
        )
        extent = _static_extent(spec)
        if extent is not None:
            return extent
        shape = tuple(getattr(spec, "shape", ()) or ())
        raise ConfigurationError(
            f"{self.kind.value} element {self.name!r} cannot tell how big a crop model "
            f"{self.model!r} wants: its declared inputs are {sorted(specs) or 'none'} and the "
            f"one named {self._input!r} is {shape or 'absent'}, which is not a static "
            "(3, H, W). Give the model a `config.yaml` that declares it, or say so on the "
            "slot: `params: {crop: {size: [256, 128]}}`"
        )

    def _refuse_a_crop_the_model_disagrees_with(self) -> None:
        """Cross-check the resolved input name and crop extent against the artefact.

        The same two refusals :meth:`PoolDetect._refuse_a_letterbox_the_model_disagrees_with`
        makes, asked of the spec itself with
        :meth:`~shipinfer.core.types.TensorSpec.matches` rather than of a hand-rolled
        predicate over the extent — one call covers a ``crop.size`` that contradicts a static
        ``(3, H, W)``, an input that is not image-shaped at all, and the dynamic ``3x?x?`` the
        override exists for.

        Raises:
            ConfigurationError: either disagreement, naming both ends.
        """
        specs = self._declared("input_specs")
        if not specs:
            return
        spec = specs.get(self._input)
        if spec is None:
            raise ConfigurationError(
                f"{self.kind.value} element {self.name!r} submits its crops as input "
                f"{self._input!r} and model {self.model!r} declares no such input (it declares "
                f"{sorted(specs)}). Name the model's own input on the slot: "
                "`params: {input: <name>}` — or fix `ingest.input_name`, which is where an "
                "element that does not say gets it"
            )
        if not spec.matches((3, *self._crop_size)):
            raise ConfigurationError(
                f"{self.kind.value} element {self.name!r} would submit "
                f"(3, {self._crop_size[0]}, {self._crop_size[1]}) crops to model "
                f"{self.model!r}, which declares that input as {spec.describe()} — a shape "
                "that cannot receive one. `crop.size` is the override for an engine whose "
                "input is *dynamic* (`3x?x?`); against a declared static extent it is a "
                "mistake a dynamic-shape engine would accept silently and answer with an "
                "appearance vector computed from the wrong pixels. Drop the override, name "
                "the model's own input (`params: {input: <name>}`), or fix its `config.yaml`"
            )

    def _resolve_output(self) -> str:
        """Which response output holds one row per crop.

        This slot's ``output``, else the model's single declared output — a re-identification
        engine has one, and both embedders in the demo repository call it ``embedding``. A name
        is a property of the artefact, so guessing one refuses a valid engine over a naming
        preference; that is why a multi-output model must say.
        """
        declared = self.params.get("output")
        if declared:
            return str(declared)
        specs = self._declared("output_specs")
        if len(specs) == 1:
            return next(iter(specs))
        raise ConfigurationError(
            f"{self.kind.value} element {self.name!r} cannot tell which output of model "
            f"{self.model!r} holds one row per crop: it declares {sorted(specs) or 'none'}. "
            "Name it on the slot: `params: {output: embedding}`"
        )

    # -- one frame ---------------------------------------------------------------------

    def _do_process(self, item: ChainItem) -> ChainItem:
        """Crop, submit, scatter — and on a frame with nothing to crop, do none of it.

        Overrides the base walk for one reason, and it is the first line: **zero rows means no
        request**. The base submits unconditionally because the payload it forwards is always
        submittable; here the payload is a fan-out that can be empty, and an empty crop batch
        handed to a model is a request that costs a queue slot, an instance slot and a
        round-trip to be told nothing. A quiet camera is the common case at this sizing, not
        an edge one.

        The three halves the base names are still the three halves here —
        :meth:`_prepare` cuts the crops, :meth:`_PoolElement._submit` runs them,
        :meth:`_finish` files them — so a second crop element (a segmenter fed crops) needs
        only its own :meth:`_finish`.

        Returns:
            The successor item, payload untouched, carrying this element's ``meta_key``.
        """
        crops, rows = self._prepare(item)
        self._on_frame(len(rows))
        if not rows:
            # An empty mapping, not an absent key: "this element ran and had nothing to crop"
            # and "this element never ran" are different facts, and only the first one is
            # evidence that the chain is wired the way the operator thinks. `track` reads an
            # empty mapping as coverage of no rows and tracks the frame on motion alone.
            return self._scatter(item, {})
        return self._finish(item, self._submit_crops(item, crops), rows)

    def _prepare(self, item: ChainItem) -> tuple[Tensor, Any]:
        """One ``(N, 3, h, w)`` crop batch for the rows this slot embeds, and those rows.

        **One ``crop_batch`` call for all N boxes**, never a Python loop around a kernel
        launch: the ops interface is batched precisely so that the loop is hard to write
        (CONVENTIONS 2.5), and the rows come back in the order the boxes went in, which is the
        whole basis of the scatter-back in :meth:`_finish`.

        The pixels are scaled by the normalisation this slot resolved at ``open`` (``params:
        {crop: {normalize: ...}}``, defaulting to
        :class:`~shipinfer.topology.elements.detections.Normalization`) and never by a
        re-derived one: a crop fed to a re-identification engine in the wrong scale is
        answered without an error and shows up only as appearance matching that degrades.

        The crops are cut from the **source** frame the item still carries, not from the
        letterboxed one the detector submitted — which is why ``PoolDetect`` hands its payload
        on unchanged. Cropping the full-resolution frame is both cheaper and sharper than
        cropping a letterbox and resizing again (``pipeline/graph/crop.py``), and the boxes
        ``PoolDetect`` filed are already in exactly those pixels.

        Returns:
            ``(crops, rows)``, where ``rows`` is the detection index of every crop, in order.
            ``rows`` is empty when there is nothing to embed and the crops are then a shared
            empty batch that is never submitted.

        Raises:
            ValidationError: the chain filed no ``detections`` (no detector ran in front of this
                element, or its ``when:`` skipped one), the payload is not a host ``(H, W, 3)``
                uint8 frame (:meth:`_PoolElement._frame_of`), or the frame this element holds is
                not the frame the boxes were measured in.
        """
        detections = item.meta.get("detections")
        if not isinstance(detections, Detections):
            raise ValidationError(
                f"{self.kind.value} element {self.name!r} crops one box per detection and the "
                f"chain filed meta['detections'] as {type(detections).__name__}; it needs the "
                "decoded, source-pixel `Detections` a `pool` detector files. Put a `detect` "
                "element in front of this one, and check that its `when:` does not skip the "
                "frames this element runs on"
            )
        rows = self._selected(detections)
        if not rows:
            return self._nothing_to_crop, ()
        image = self._frame_of(item)
        self._refuse_a_frame_the_boxes_do_not_belong_to(item, image)
        boxes = detections.boxes_at(rows)
        crops = self._ops.crop_batch(image, boxes, self._crop_size, self._normalize)
        return Tensor.from_numpy(crops), rows

    def declared_classes(self) -> tuple[str, ...] | None:
        """See :meth:`~shipinfer.topology.base.Element.declared_classes`."""
        return self._classes

    def _selected(self, detections: Detections) -> range | tuple[int, ...]:
        """The detection rows this slot embeds — every one, or the declared classes.

        A ``range`` in the "no ``classes:``" case, exactly as ``track._selected`` returns one
        and for the same reason: nothing here materialises it. :meth:`_finish` iterates it
        once to build the mapping, :meth:`_prepare` reads its length, and
        :meth:`~shipinfer.topology.elements.detections.Detections.boxes_at` recognises it as
        the whole frame and copies the boxes flat instead of gathering them — so the common
        case allocates no index list. The indices are in the detector's own order, which
        is descending score, and the crops follow that order into the model.
        """
        if self._classes is None:
            return range(len(detections))
        return detections.indices_of_any(self._classes)

    def _refuse_a_frame_the_boxes_do_not_belong_to(
        self, item: ChainItem, image: np.ndarray
    ) -> None:
        """Refuse a frame whose extent is not the one the boxes were decoded into.

        ``PoolDetect`` files ``meta["frame_hw"]`` beside the detections precisely so that the
        elements behind it need not re-derive it, and this is the one place it is *checked*
        rather than read: the boxes are in those pixels, so cropping a frame of a different
        extent takes the wrong pixels for every object at once. It is two integer compares on
        the per-frame path and it catches the corruption that has no other symptom — a
        resized payload, a second decoder, a chain where two frames were confused — which
        would otherwise surface as a tracker that swaps identities weeks later.

        A chain that filed no ``frame_hw`` is not second-guessed: an element that filed
        detections without it is saying nothing about the extent, and inventing a
        disagreement out of an absence would refuse a working chain.
        """
        declared = item.meta.get("frame_hw")
        if declared is None:
            return
        actual = (int(image.shape[0]), int(image.shape[1]))
        if tuple(int(value) for value in declared) != actual:
            raise ValidationError(
                f"{self.kind.value} element {self.name!r} was handed a {actual[0]}x{actual[1]} "
                f"frame and the detections it must crop were decoded into "
                f"{tuple(declared)} pixels. The boxes are in the detector's source frame, so "
                "cropping a differently sized one takes the wrong pixels for every object; "
                "the element between the two that resized the payload is the bug"
            )

    def _submit_crops(self, item: ChainItem, crops: Tensor) -> InferenceResponse:
        """Run the crop batch, in as many requests as the model's ``max_batch_size`` allows.

        A frame holds however many objects the detector found — 10 to 20 people is the
        sizing this project targets and 25 was observed — while an engine's plan is built at
        a fixed batch. Without this, one crowded frame becomes a single request larger than
        the model can ever accept:

            InferenceError: assembled batch of 25 rows exceeds max_batch_size 16

        and *every* crop in that frame is lost. Batching across requests is already bounded
        correctly by the queue; what no amount of scheduling fixes is a single request that
        exceeds the bound on its own. This is ``pipeline/graph/objects.py::_chunks``, which
        exists because that failure was real.

        The chunks are one logical batch, so they are rejoined into one response before
        :meth:`_finish` sees them and the scatter-back never learns that chunking happened.
        Only the output this element declares is carried across the join, for the reason
        ``objects.py::_quantities`` materialises only what it forwards: a segmentation engine
        emits megabytes of prototypes per row, and concatenating an output nobody reads would
        pay for it on every crowded frame.

        The slices are :meth:`~shipinfer.core.types.Tensor.slice_batch` views, so chunking
        copies no pixels.

        **The chunks are submitted one at a time, and that is the latency budget to know
        about.** Each :meth:`_submit` blocks on its own future, so a frame split into K
        requests costs K sequential round trips and can hold this worker for up to
        ``K * timeout_s`` — with the default four workers, a crowded frame is a worker
        unavailable to any camera for that long. It is not a regression (``pipeline/graph/
        objects.py`` serialises the same way) and K is small on a *batching* model: 25 objects
        against a plan built at batch 16 is K=2. On ``max_batch_size: 0`` it is not — the
        engine's bound is then one row, K == N, and a 15-person frame is 15 round trips, so
        declare a ``max_batch_size`` on a model a crop element feeds. Submitting all K first and
        then collecting the futures would cost one round trip instead of K, at the price of K
        requests in flight per worker against a bound the scheduler sizes for one; that trade
        belongs with the asynchronous walk (arch.md section 5, item 5) and not here.

        Raises:
            InferenceError: a chunk came back without the declared output.
            RequestTimeoutError, QueueFullError: the base's, per chunk.
        """
        total = crops.shape[0]
        limit = self._max_rows
        if total <= limit:
            return self._submit(item, crops)
        chunks = [
            self._submit(item, crops.slice_batch(start, min(start + limit, total)))
            for start in range(0, total, limit)
        ]
        joined = np.concatenate([self._rows_of(chunk).numpy() for chunk in chunks], axis=0)
        # Everything except `outputs` is the **first chunk's**: `request_id`, `timings` and
        # `executed_on` describe one of the K requests this frame cost, not their sum, and
        # `executed_on` names one of up to K instances. `_finish` reads only `outputs`, so
        # nothing is wrong today; a consumer that attributes device load off a crop element's
        # response would be reading chunk 0 and must aggregate here instead.
        return replace(chunks[0], outputs={self._output: Tensor.from_numpy(joined)})

    def _rows_of(self, response: InferenceResponse) -> Tensor:
        """The declared output, or an :class:`InferenceError` naming what the model did send.

        Raised rather than returned empty: "the embedder answered with nothing" and "the
        embedder's output is called something else" demand opposite responses, and an absent
        vector that reached ``track`` would simply turn appearance matching off for the frame
        with nothing said (CONVENTIONS: never return an empty result to mean failure).
        """
        rows = response.outputs.get(self._output)
        if rows is None:
            raise InferenceError(
                f"{self.kind.value} element {self.name!r}: model {self.model!r} returned no "
                f"output {self._output!r} (got: {sorted(response.outputs)})"
            )
        return rows

    def _finish(self, item: ChainItem, response: InferenceResponse, carried: Any) -> ChainItem:
        """Scatter the model's rows back onto the detections they were cut from.

        ``{detection index: vector}`` — the mapping form
        :meth:`~shipinfer.topology.elements.track.ShipvisionTrack._embeddings` accepts, chosen
        because partial coverage is this chain's normal case and not its exception; the class
        docstring has the argument. The pairing is positional and that is the whole contract:
        ``crop_batch`` returned the rows in the order the boxes went in, so row ``i`` of the
        response belongs to ``rows[i]``.

        No ``caps=``: the payload — the source frame — is handed on unchanged, so the cap it
        carries is the cap it arrived with. This element adds a metadata key and nothing else,
        which is why its ``produces`` stays ``*@*``.

        Raises:
            InferenceError: no such output, or a row count that is not the crop count. Refused
                rather than zipped to the shorter of the two, because a scatter-back that
                silently drops the last object attaches every remaining vector correctly and
                loses one identity per frame with no counter anywhere.
        """
        rows: range | tuple[int, ...] = carried
        vectors = self._rows_of(response).numpy()
        if vectors.ndim < 2 or vectors.shape[0] != len(rows):
            raise InferenceError(
                f"{self.kind.value} element {self.name!r}: model {self.model!r} answered "
                f"{len(rows)} crops with {vectors.shape} under {self._output!r}, and a "
                "scatter-back needs one row per crop. A mismatch here would attach an "
                "appearance vector to the wrong object, which has no symptom until a tracker "
                "starts swapping identities"
            )
        return self._scatter(item, {row: vectors[i] for i, row in enumerate(rows)})

    def _scatter(self, item: ChainItem, covered: dict[int, Any]) -> ChainItem:
        """File ``covered`` under this element's key, **merged** with what is already there.

        Additive, and not a nicety. The sizing runs two of these over disjoint rows —
        ``embed_ship`` and ``embed_person``, both filing ``meta["vectors"]`` — and ``derive``
        merges meta *by key*, so a plain assignment by the second would replace the first
        wholesale. Every ship would then reach ``track`` with ``embedding=None`` on a chain
        whose logs and counters all say both embedders ran.

        Two of them meet in one of two ways. In **series** the second finds the first's mapping
        here and merges into it. In **parallel** neither sees the other's item and the union is
        taken at the rejoin (:meth:`~shipinfer.runners.walk.ChainWalk.inbound`). Both are legal
        and neither may lose half the frame.

        What is filed is a :class:`~shipinfer.topology.base.RowIndexed`, which is what tells the
        fan-in the value is keyed by detection row and may be unioned. A bare dict still reaches
        ``track`` and simply does not union — the right default for a value nobody declared, and
        why the ``segment`` slots filing raw outputs are left alone. Merging into a peer keeps
        the peer's declaration: a plain dict stays plain.

        **Both halves refuse an overlap.** Disjoint rows union; a row two elements both cover is
        an :class:`~shipinfer.core.errors.InferenceError` here exactly as at the fan-in, because
        two elements covering one detection is a property of the chain file, equally true in
        series and in parallel. ``{**existing, **covered}`` would turn that into a silently
        wrong vector.

        Raises:
            ValidationError: something upstream filed a non-mapping under this key, so there is
                nothing to merge into — a mis-wired chain, and refusing beats replacing.
            InferenceError: two elements cover the same detection row.
        """
        existing = item.meta.get(self.meta_key)
        if existing is None:
            return item.derive(**{self.meta_key: RowIndexed(covered)})
        if not isinstance(existing, Mapping):
            raise ValidationError(
                f"{self.kind.value} element {self.name!r} scatters its rows into "
                f"meta[{self.meta_key!r}] and something upstream filed a "
                f"{type(existing).__name__} there. Two elements writing this key must both "
                "write a {detection index: row} mapping, or the second one silently replaces "
                "the first one's coverage"
            )
        if not covered:
            # Nothing to add and the peer's mapping is already correct, so hand the item on
            # **itself** rather than deriving a copy of it: `derive()` would build a second
            # meta dict and a second `ChainItem` to say exactly what this one says, and an
            # unchanged item is already a legal thing to flow down the chain (that is what a
            # false `ElementNode.admits` hands on). The quiet camera is the common case.
            return item
        for row in covered:
            if row in existing:
                raise InferenceError(
                    f"{self.kind.value} element {self.name!r} scatters detection row {row!r} "
                    f"into meta[{self.meta_key!r}] and an earlier slot on this branch already "
                    "filed that row. Elements sharing this key merge their coverage rather "
                    "than one replacing the other, and two of them covering one detection "
                    "means the chain file asked both of them for it -- check their "
                    "`params: classes:` do not overlap"
                )
        merged = {**existing, **covered}
        # `RowIndexed` only if the peer declared one: the writer says what it wrote, and
        # promoting a plain dict here would declare on its behalf.
        declared = RowIndexed(merged) if isinstance(existing, RowIndexed) else merged
        return item.derive(**{self.meta_key: declared})


@registry_for(ElementKind.SEGMENT).register("pool")
class PoolSegment(_PoolElement):
    """Segmentation through the model pool.

    Still the forwarding element, and deliberately so *for now*. The demo repository's
    ``ship_segmenter`` is fed crops in the proven pipeline (``pipeline/graph/graph.py`` cuts a
    ``ship_mask_crops`` set at 640x640 and hands it to an ``ObjectStage``), so the *first*
    half of this element is exactly :class:`_PoolCropElement`'s and adopting it is a one-line
    change of base class. The *second* half is not: a YOLO segmentation engine emits detection
    rows and a bank of mask prototypes, never a mask, and the mask for one crop is the two
    multiplied together and then reduced to an area
    (``pipeline/graph/masks.py::InstanceMaskArea``) — a fold over two outputs at once, which
    a per-row scatter-back cannot express.

    Filing the raw rows per detection instead would be worse than leaving this alone: it pins
    a ``(32, 160, 160)`` prototype tensor per frame alive for the rest of the walk so that a
    later consumer can redo arithmetic that already exists, which is 3 MB a frame of
    reassembly memory spent on pixels nobody reads. So the crop half lands with the
    segmenter's own ``_finish``, in its own slice, and this class keeps the base's behaviour
    until then rather than gaining half of a feature.
    """

    kind: ClassVar[ElementKind] = ElementKind.SEGMENT
    meta_key: ClassVar[str] = "masks"


@registry_for(ElementKind.EMBED).register("pool")
class PoolEmbed(_PoolCropElement):
    """Embedding through the model pool: one crop per detection, one vector per row.

    One instance per embedder slot in the chain, and the sizing expects two of them side by
    side — ``embed_ship`` with ``classes: [ship]`` and ``embed_person`` with
    ``classes: [person]`` — each filing the rows it covered into the same
    ``meta["vectors"]`` key. That is the case the mapping scatter-back is for; see
    :class:`_PoolCropElement`.

    A re-identification engine L2-normalises its own output (both embedders in the demo
    repository are ResNet-50 backbones "global-pooled and L2-normalised to a 2048-d vector",
    and their ``config.yaml`` says so), and the proven path normalised nothing in Python
    either — ``ObjectStage`` forwards ``embedding`` untouched. So this element forwards it
    untouched too. Re-normalising here would be a second normalisation of an already-unit
    vector: harmless on a well-behaved model, and a silent divide-by-a-tiny-number on the row
    where the engine answered with zeros.
    """

    kind: ClassVar[ElementKind] = ElementKind.EMBED
    meta_key: ClassVar[str] = "vectors"


@registry_for(ElementKind.RECOGNIZE).register("pool")
class PoolRecognize(_PoolElement):
    """Identity through the model pool."""

    kind: ClassVar[ElementKind] = ElementKind.RECOGNIZE
    meta_key: ClassVar[str] = "identities"
