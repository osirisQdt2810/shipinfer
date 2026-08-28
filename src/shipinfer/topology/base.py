"""The element contract: one processing step with declared caps and a lifecycle.

An **element** is the first of arch.md's three concepts (§1): one step of the perception
chain — ``decode``, ``detect``, ``segment``, ``embed``, ``recognize``, ``track``, ``mtmc``,
``output`` — with interchangeable implementations behind a registry, in the same shape the
ingest sources already use.

Three deliberate absences, because each one is what keeps this package pure and testable
with no driver installed:

1. **No engine import.** A ``pool`` element submits to the model pool, but it receives that
   pool through the :class:`ElementContext` handed to :meth:`Element.open`. Dependencies
   arrive; they are never imported. That inversion is the only reason ``topology`` can sit
   directly on ``core``.
2. **No batching.** :meth:`Element.process` takes **one** :class:`ChainItem`. Batching is
   the engine's job (arch.md §5④), where a batch is assembled across cameras from the
   per-model queue; a pipeline worker walks a single frame through the chain (§5③). An
   element that batched internally would be a second, invisible scheduler.
3. **No placement.** Which GPU, which shard and which process is the runner's business
   (§1). An element is told, via :class:`ElementContext`, and does not choose.

``open`` / ``process`` / ``close`` are **template methods**, exactly as
:class:`~shipinfer.ingest.base.FrameSource` does it, and for the same reason: the
invariants below would otherwise be re-implemented — and eventually mis-implemented — once
per implementation.

* ``open`` is idempotent and unwinds a partial failure before re-raising;
* ``close`` is idempotent, so a restart path and a shutdown path can both call it;
* ``process`` before ``open`` is a typed refusal, not undefined behaviour.
"""

from __future__ import annotations

import abc
import contextlib
import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Protocol

from shipinfer.core.errors import ServerStateError, UnknownElementKindError
from shipinfer.core.metrics import MetricsRegistry
from shipinfer.core.request import RequestContext
from shipinfer.core.types import Device
from shipinfer.topology.barrier import WaiterBudget
from shipinfer.topology.caps import Caps, parse_caps

__all__ = [
    "CameraGroup",
    "ChainItem",
    "Element",
    "ElementContext",
    "ElementKind",
    "ImageOpsLike",
    "LetterboxLike",
    "ModelResolver",
    "RowIndexed",
]


class ElementKind(str, enum.Enum):
    """The eight kinds of step a chain is built from (arch.md §1).

    A closed vocabulary, unlike the implementations behind each kind. The kind decides what
    an element *means* to the chain — whether it may be a root, whether it may be followed,
    what its slot name resolves to — and a ninth meaning is an architecture change, not a
    configuration one.

    What a kind deliberately does **not** decide is whether an element runs a repository
    model. That was a kind-level rule once (``detect``/``segment``/``embed``/``recognize``
    needed ``model:``, the other four did not) and it was wrong in exactly one place that
    matters: ``recognize`` is a *gallery query* over an embedding for the shipvision
    implementation and a network for the pool one, so the same kind is both. The requirement
    is :attr:`Element.requires_model_name`, declared per implementation -- and whether the
    element resolves that name against *this process's* model pool is a second declaration,
    :attr:`Element.needs_model`.

    A ``str`` enum so a kind can be written into YAML, a log line or a metric label without
    a conversion at each site.
    """

    DECODE = "decode"
    DETECT = "detect"
    SEGMENT = "segment"
    EMBED = "embed"
    RECOGNIZE = "recognize"
    TRACK = "track"
    MTMC = "mtmc"
    OUTPUT = "output"

    @classmethod
    def names(cls) -> list[str]:
        return [kind.value for kind in cls]

    @classmethod
    def parse(cls, text: str) -> ElementKind:
        """Parse an explicit ``kind:`` value.

        Raises:
            UnknownElementKindError: naming every kind, because a one-word typo should not
                need the source to diagnose.
        """
        try:
            return cls(text.strip().lower())
        except ValueError:
            raise UnknownElementKindError(text, cls.names()) from None

    @classmethod
    def infer(cls, slot: str) -> ElementKind:
        """The kind a chain slot name implies: the whole name, else its first word.

        ``detect:`` is a detector and ``embed_ship:`` is an embedder, which is what lets a
        chain declare two embedders without either one repeating ``kind: embed``. The split
        is on the *first* underscore only, so ``embed_ship_v2`` still resolves.

        Raises:
            UnknownElementKindError: neither reading resolves. ``sement:`` is a typo, and a
                chain that quietly ran without its segmentation step would be far worse
                than one that refuses to load.
        """
        head = slot.strip().lower()
        for candidate in (head, head.split("_", 1)[0]):
            try:
                return cls(candidate)
            except ValueError:
                continue
        raise UnknownElementKindError(slot, cls.names())


#: Sentinel for "leave this alone", so ``derive(payload=None)`` can mean *clear it*.
_KEEP: Final[Any] = object()


class RowIndexed(dict):  # type: ignore[type-arg]
    """A ``meta`` value that is a **scatter-back**: ``{detection row: that row's result}``.

    A plain ``dict`` in every respect that matters at runtime — the type *is* the payload of
    this class, and it carries no behaviour. What it carries is a **declaration**: the value
    under this metadata key is keyed by *detection row index* and is therefore partial by
    design, so two elements that both filed one over disjoint rows describe one frame between
    them and their answers compose. ``PoolEmbed`` files one per frame
    (:meth:`~shipinfer.topology.elements.pool._PoolCropElement._scatter`) and ``track`` reads
    it (:meth:`~shipinfer.topology.elements.track.ShipvisionTrack._embeddings`).

    **Why a type and not a sniff.** The fan-in
    (:meth:`~shipinfer.runners.inprocess.InprocessRunner._merge_meta`) has to tell "two
    branches each covered half the rows, union them" from "two branches disagree about one
    value, take the first". It used to answer that with ``isinstance(value, Mapping)``, and a
    mapping is exactly what it cannot distinguish on: ``_PoolElement._finish`` files a
    model's raw ``response.outputs`` — a ``{output name: Tensor}`` dict — under its own
    ``meta_key``, and ``segment`` and ``recognize`` both keep that default. Sniffed, two
    rejoining segmenters either refuse every frame (engines that name their output the same
    collide on the *name*) or silently fabricate a composite ``{'ship_masks': …,
    'person_masks': …}`` that no engine emitted. Neither mapping says what shape it is, so
    the merge guessed — and this class is the shape saying so itself. Declaring it costs the
    writer one constructor call and buys the reader a total answer.

    So: **only a** ``RowIndexed`` **unions at a fan-in.** Any other mapping keeps the
    first-writer-wins rule that predates the union, which is what an undeclared value should
    get — it is a value the runner has no attribution rule for, not one it may invent one for.

    Being a ``dict`` subclass is the other half of the design. Every consumer that tests
    ``isinstance(..., Mapping)``, iterates it, or does ``dict(...)`` on it keeps working
    unchanged, so nothing downstream had to learn the name; and a chain that files a bare
    ``{row: vector}`` still reaches ``track`` correctly, it merely does not get the union it
    never asked for.

    It lives here, next to :class:`ChainItem`, because it is a statement about ``ChainItem.
    meta`` — the only place these values exist — and because ``topology`` is the deepest
    layer both sides of the contract may import: the elements that write it are
    ``topology.elements``, the fan-in that reads it is ``runners``, and ``runners`` imports
    ``topology`` (``scripts/hooks/check_layers.py``). ``core`` is the other layer both may
    reach, and it is the wrong one: ``core`` has no word for a chain item and must not gain a
    vocabulary for one metadata dict's conventions.
    """

    __slots__ = ()


@dataclass(slots=True)
class ChainItem:
    """One unit of work travelling along the chain, and the tag that identifies it.

    The invariant this type exists to hold is ADR-002's: **the ``(camera_id, frame_id)``
    tag rides untouched from ingest to the last element**. Every element derives its output
    from its input with :meth:`derive` rather than constructing a fresh item, so a new
    implementation cannot forget to carry the tag forward — the mistake that makes a
    reassembled frame mix two cameras.

    ``slots=True`` and not frozen: one of these exists per frame per element — at 1000 fps
    through a nine-element chain, nine thousand a second — and the mutable ``meta`` dict is
    the point, not an oversight.

    Args:
        context: the tag. Never replaced, only carried.
        caps: what ``payload`` currently is. Set by the producing element to the cap the
            loader negotiated for the edge, so a consumer never has to guess.
        payload: the frame, the crops, the tensor — whatever this element's caps describe.
            Typed ``object`` because ``core`` has no word for a device buffer and topology
            must not learn one; the caps are the contract, not the Python type.
        meta: accumulated results — boxes, classes, vectors, track ids. Additive: an
            element adds keys and does not remove another's.
    """

    context: RequestContext
    caps: Caps
    payload: object | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int]:
        """The reassembly key — the same one every other plane groups on."""
        return self.context.key

    def derive(
        self, *, caps: Caps | None = None, payload: Any = _KEEP, **meta: Any
    ) -> ChainItem:
        """A successor item: same tag, new caps/payload, ``meta`` merged.

        Keyword-only on purpose. A positional form would make ``derive(new_payload)`` read
        as a caps change at the call site of every element ever written.
        """
        return ChainItem(
            context=self.context,
            caps=caps if caps is not None else self.caps,
            payload=self.payload if payload is _KEEP else payload,
            meta={**self.meta, **meta},
        )


class ModelResolver(Protocol):
    """What an element needs from the model pool: a handle, by name.

    A :class:`typing.Protocol` rather than an import. The engine (arch.md §6) satisfies it,
    a test satisfies it with a dict, and ``topology`` never learns that either exists — the
    inversion that keeps this package importable with no accelerator on the host.
    """

    def get(self, name: str) -> Any:
        """The model registered under ``name``.

        Raises:
            ModelNotFoundError: the implementor's promise, not this protocol's.
        """
        ...


class LetterboxLike(Protocol):
    """What a letterbox call answers with: the tensor and the geometry that undoes it.

    The structural half of :class:`shipinfer.runtime.ops.base.LetterboxResult`. Every member
    is a numpy array, and they are typed ``Any`` for one reason: the scales and pads must be
    *carried*, never recomputed. Postprocess has to invert exactly the transform preprocess
    applied, and re-deriving the numbers from the shapes is where off-by-one box drift comes
    from — so an element takes these four and does no arithmetic of its own.
    """

    #: ``(N, C, H, W)`` float32, already normalised.
    tensor: Any
    #: Per-image resize scale.
    scales: Any
    #: Per-image ``(pad_x, pad_y)`` in destination pixels.
    pads: Any
    #: Per-image ``(out_h, out_w)`` — the extent actually written, before padding.
    extents: Any


class ImageOpsLike(Protocol):
    """The preprocessing an element needs, as a shape rather than an import.

    :class:`shipinfer.runtime.ops.base.ImageOps` satisfies this, and ``topology`` may not say
    so: ``runtime`` is the accelerator seam and a pure layer that named it would put torch
    behind ``import shipinfer.topology``. The same inversion as :class:`ModelResolver` — the
    runner is handed an ops implementation, resolves it once and puts it on
    :attr:`ElementContext.ops`; the element calls it and never learns where it came from.

    Deliberately **narrower** than ``ImageOps``. Two members are here, because two are what
    the elements call: ``letterbox_batch`` for the detector (``pipeline/graph/detect.py`` is
    the proven path it follows: one letterbox per frame, its ``scales``/``pads``/``extents``
    stored and handed to the decode) and ``crop_batch`` for the embedder
    (``pipeline/graph/crop.py``: one call for all N boxes of a frame, never a Python loop
    around a kernel launch). ``nms`` and ``letterbox_to_device`` are real and are absent on
    purpose — a protocol member nobody calls is a coupling nobody needs, and the element that
    needs one adds it with a test, which is exactly how ``crop_batch`` arrived.

    ``params`` and ``dst_size`` are the caller's business: an element is *told* what the model
    wants, through its own ``params:`` or the runner's context. Typing ``params`` as ``Any``
    rather than importing ``NormalizeParams`` is the same purity argument one member down.
    """

    def letterbox_batch(
        self,
        images: Sequence[Any],
        dst_size: tuple[int, int],
        params: Any,
        *,
        pad_value: int = 114,
    ) -> LetterboxLike:
        """Resize-with-pad, colour-convert, normalise and transpose, in one pass.

        Args:
            images: ``(H, W, 3)`` uint8 frames, possibly of differing sizes.
            dst_size: ``(height, width)`` of the model input.
            params: normalisation and channel order — a
                :class:`shipinfer.runtime.ops.base.NormalizeParams`.
            pad_value: fill for the letterbox bars.
        """
        ...

    def crop_batch(
        self,
        image: Any,
        boxes: Any,
        dst_size: tuple[int, int],
        params: Any,
    ) -> Any:
        """Cut N boxes out of one frame and resize them into one ``(N, C, h, w)`` tensor.

        The mirror of :meth:`letterbox_batch` for the fan-out: one frame in, one row per
        detection out. **One call for all N boxes** — the signature is batched precisely so
        that a per-crop Python loop around a kernel launch is hard to write (CONVENTIONS 2.5),
        and at 10-20 people a frame across a thousand frames a second that loop is the
        difference between a shard that keeps up and one that does not.

        The rows come back in the order the boxes went in, which is the whole basis of the
        scatter-back: the element that called this holds the detection index of every box and
        pairs it with the row at the same position. Losing that order is how an embedding ends
        up attached to the wrong object — a corruption with no exception and no symptom short
        of a tracker that swaps identities.

        Args:
            image: the ``(H, W, 3)`` uint8 source frame, in **source** pixels — not the
                letterboxed one the detector submitted. Cropping from the full-resolution
                frame is both cheaper and sharper than cropping the letterbox and resizing
                again (``pipeline/graph/crop.py``).
            boxes: ``(N, 4)`` float32 ``[x1, y1, x2, y2]`` in ``image`` pixel coordinates —
                the layout :class:`~shipinfer.topology.elements.detections.Detections` stores.
            dst_size: ``(height, width)`` of one crop, matching the consuming model's declared
                input.
            params: normalisation and channel order, as for :meth:`letterbox_batch`.
        """
        ...


@dataclass(frozen=True, slots=True)
class CameraGroup:
    """Cameras that have to be placed on one runner, and the name they are grouped under.

    What :meth:`Element.camera_group` answers. A cross-camera element associates one group
    against one identity space, and that state lives in one process: split the group across
    two shards and each half runs its own tracker, so one object is given two ids, both of
    them plausible, and nothing in the metrics disagrees (``docs/arch.md`` §4).

    The element declares the membership; **the runner decides the placement**, because only
    the runner knows where any camera is. This object is the whole of what crosses between
    them, which is what keeps the fleet from having to know that ``mtmc`` exists.

    Args:
        name: what to call the group in a refusal — the element's ``params: group:``, or its
            slot name when the chain did not say.
        cameras: the declared roster. Never empty: an element with nothing to declare answers
            ``None`` rather than an empty group, so "this chain does not group cameras" and
            "this chain groups them into nothing" cannot be the same value.
    """

    name: str
    cameras: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ElementContext:
    """Everything the surrounding runner tells an element at :meth:`Element.open`.

    Frozen: this is what the runner *decided*, and an element that could edit it would be
    choosing its own placement. Every field is optional so that a chain can be loaded,
    validated and walked with mock elements before any of the three exist — which is what
    the offline tier does.

    Args:
        shard_id: which shard process this element belongs to (arch.md §2). Stateful
            elements key their per-camera state on the camera id, never on this; it is for
            logs, metrics and the one decision ``scope: global`` needs.
        device: the GPU this shard owns, or ``None`` on a host with no accelerator.
        models: the model pool, for elements of kind ``pool``.
        stage_timeout_s: how long an element may wait for one model call, resolved by the
            runner from ``pipeline.stage_timeout_ms``. ``None`` means the runner did not say,
            and an element falls back to its own module default.
        input_name: the input tensor name a decoded frame is submitted under, resolved by the
            runner from ``ingest.input_name``. ``None`` means the runner did not say.
        metrics: the registry an element records its own counters on, so that one exporter
            carries the runner's numbers and the chain's rather than two halves of a dropped
            frame living in different places. ``None`` means the runner offered none, and an
            element must then count nothing rather than mint a private registry nobody
            scrapes — a metric on a registry no exporter reads is worse than an absent one,
            because it reads as evidence.
        workers: how many pipeline workers will walk this chain concurrently. The number an
            element needs to know it must not block *all* of them: the walk is synchronous,
            so an element that waits for other cameras' frames — an MTMC barrier is the
            standing case — can close its instant only by timeout once every worker is parked
            inside it. ``None`` means the runner did not say, and an element that needs the
            number must then refuse to wait at all rather than guess one.
        waiter_budget: the permits those waits are drawn from, **shared by every element in
            this process**. ``workers`` alone is not enough, because each element counts only
            its own waiters: a chain with two ``mtmc`` slots would have each barrier admit
            ``workers - 1`` and park every worker between them. One
            :class:`~shipinfer.topology.barrier.WaiterBudget` with ``workers - 1`` permits,
            built by the runner, makes the invariant hold however many waiting elements there
            are. ``None`` means the runner did not say, and an element that waits then falls
            back to a private budget sized from ``workers`` — correct when it is the only
            such element, which is what the offline tier builds.
        ops: batched image preprocessing, bound to this shard's device, in the same shape
            ``models=`` already has -- ``shipinfer run`` (``cli/commands/run.py``) or the
            shard process (``cli/shard.py``) resolves one implementation from ``runtime.ops``,
            the layer that may import torch, and hands it to the runner, which puts it here.
            What arrives is a ``ThreadLocalImageOps``, because a chain is walked by
            ``pipeline.workers`` threads over one shared element and an ``ImageOps`` belongs
            to one thread. ``None`` means the runner resolved none, which is the normal answer
            for a chain no element of which declares
            :attr:`~shipinfer.topology.base.Element.needs_image_ops`; an element that needs it
            and finds ``None`` must raise, rather than falling back to a per-image loop in
            Python.

    The last five are **resolved settings, not settings**. ``topology`` is a pure package and
    must not import :mod:`shipinfer.core.settings`, :mod:`shipinfer.runners` or
    :mod:`shipinfer.runtime` — an element that read the settings tree itself would also be
    choosing its own configuration, which is the thing this frozen object exists to prevent,
    and one that imported the ops registry would put torch behind ``import
    shipinfer.topology``. So the runner resolves each of them once and hands over what an
    element cannot otherwise know, and an element resolves the two it can also be told twice
    with a fixed precedence: its own ``params:``, then this context, then its module default.
    Without them the two settings keys mirrored in ``topology/elements/pool.py`` would apply
    to nothing.

    ``metrics`` and ``ops`` are the two that arrive as **objects rather than numbers**, and
    both are typed structurally for the reason above: ``MetricsRegistry`` lives in ``core``,
    which this layer may import, and an ops implementation does not, which is why
    :class:`ImageOpsLike` exists.
    """

    shard_id: int = 0
    device: Device | None = None
    models: ModelResolver | None = None
    stage_timeout_s: float | None = None
    input_name: str | None = None
    metrics: MetricsRegistry | None = None
    workers: int | None = None
    ops: ImageOpsLike | None = None
    waiter_budget: WaiterBudget | None = None


class Element(abc.ABC):
    """One step of the chain. Subclass, declare caps, register, done.

    Subclasses declare four class attributes and implement two hooks. The class attributes
    are read by the loader *before* anything is instantiated, which is what lets a chain be
    validated end to end without opening a camera or a CUDA context.

    Args:
        name: the slot this element fills in the chain (``embed_ship``), not its
            implementation name. Two embedders differ by this and by their params.
        params: implementation-specific settings, straight from the chain's ``params:``.
        model: the repository model this element runs, from the chain's ``model:``, or
            ``None`` for the four kinds that have no model (``decode``, ``track``, ``mtmc``,
            ``output``). Keyword-only, and a plain string: an element resolves it to a
            handle through :attr:`ElementContext.models` at ``open``, because a *name* is
            data the loader can validate on a laptop and a *handle* is not.

    An implementation that overrides ``__init__`` must accept ``model`` and forward it --
    :func:`~shipinfer.topology.registry.create_element` always passes it, so a two-argument
    constructor is a ``TypeError`` at load time.

    ``__init__`` must be **cheap and hardware-free**. The loader instantiates every element
    in the chain to read its caps, so a constructor that opened a stream or a CUDA context
    would make ``shipinfer`` unable to *validate* a topology on a laptop. Acquire resources
    in ``_do_open``; that is what it is for.
    """

    #: Which kind of step this is. Left ``None`` on the ABC so that the registry can refuse
    #: a class that forgot to declare one instead of registering it under a lie.
    kind: ClassVar[ElementKind | None] = None
    #: The registered implementation name, set by ``@registry.register`` so that the name in
    #: the chain file and the name in a log line cannot drift.
    impl: ClassVar[str] = "abstract"
    #: Caps this element will take. Empty means "nothing upstream" — only legal on a root.
    accepts: ClassVar[tuple[str, ...]] = ()
    #: Caps this element hands on. Empty means it is a sink.
    produces: ClassVar[tuple[str, ...]] = ()
    #: Whether the chain file must name a ``model:`` for this slot -- the **loader's**
    #: question, and the only one it asks. Read off the built element by
    #: :meth:`~shipinfer.topology.chain.Topology.from_spec`, never inferred from :attr:`kind`:
    #: ``detect`` is a model *kind* and only some of its implementations run a repository
    #: model, ``recognize: {impl: shipvision}`` is a gallery query with nothing to name, and
    #: ``impl: mock`` invents a box and reads nothing.
    #:
    #: Deliberately **not** the same declaration as :attr:`needs_model`, which asks whether
    #: ``open()`` will reach into this process's model pool. The two agree for every
    #: implementation phase C ships and come apart at the first one that runs its model
    #: somewhere else: an ``nvinfer`` detect *names* a ``model:`` artefact and executes it
    #: inside GStreamer, so it declares ``requires_model_name = True`` with
    #: ``needs_model = False``. Folded into one attribute, that element would have to choose
    #: between refusing a correct chain and making ``shipinfer run`` build an
    #: ``InferenceServer`` nothing in the chain would ever submit to.
    requires_model_name: ClassVar[bool] = False

    #: Whether :meth:`open` resolves this element's model against
    #: :attr:`ElementContext.models` -- the **pool's** question, about this process rather
    #: than about the YAML. A class that answers ``True`` must raise from :meth:`_do_open`
    #: when the context carries no pool, so that the declaration and the requirement cannot
    #: drift.
    #:
    #: Two callers read it, and neither is the chain loader (that one asks
    #: :attr:`requires_model_name`):
    #:
    #: * the process that builds a runner, to decide whether to build an ``InferenceServer``
    #:   at all (``cli/commands/run.py``) -- asking the *kind* there would load engines for a
    #:   chain of mocks, and asking :attr:`requires_model_name` would load them for a
    #:   deepstream chain whose models run inside GStreamer;
    #: * the walk, to re-check a frame's deadline in front of the elements that can *wait*
    #:   (``runners/inprocess.py``) -- an element that submits to the pool and sleeps on the
    #:   answer is exactly the one a frame nobody can act on must not be given another GPU
    #:   for, and a local one is not.
    needs_model: ClassVar[bool] = False

    #: Whether :meth:`open` reads :attr:`ElementContext.ops` -- the third of these
    #: declarations and the newest, asked by the same caller as :attr:`needs_model` and for
    #: the same reason: the process that builds a runner has to know, *before* it builds one,
    #: whether to resolve an image-ops implementation out of ``shipinfer.runtime.ops``. That
    #: resolution is not free and it is not portable -- ``get_image_ops`` may construct a
    #: torch context bound to this shard's device -- so a chain of mocks must not pay for it,
    #: which is exactly the mistake ``node.kind in MODEL_KINDS`` was for the model pool.
    #:
    #: Separate from :attr:`needs_model` rather than folded into it, because the two come
    #: apart in both directions: ``PoolSegment`` submits a crop somebody else prepared
    #: (``needs_model``, no ops), and a future ``crop`` element would preprocess without
    #: running a repository model at all (ops, no ``needs_model``). Today only
    #: :class:`~shipinfer.topology.elements.pool.PoolDetect` answers ``True`` -- it is the one
    #: element that letterboxes a whole frame and then has to undo exactly that transform to
    #: put the boxes back in source pixels.
    #:
    #: A class that answers ``True`` must raise from :meth:`_do_open` when the context carries
    #: no ops, so that the declaration and the requirement cannot drift.
    needs_image_ops: ClassVar[bool] = False

    #: Whether this implementation selects **detection rows** with ``params: classes:`` -- the
    #: loader's question, asked so that the *other* filter cannot be pointed at a row.
    #:
    #: ``when:`` and ``classes:`` look interchangeable in a chain file and are not, and the
    #: mistake is silent in both directions. ``when:`` guards ONE ELEMENT for a whole FRAME:
    #: :meth:`~shipinfer.topology.chain.Condition.matches` reads ``item.meta``, an item is a
    #: frame, and a frame holds ships *and* people at once -- so ``when: class == ship`` on a
    #: crop element is false on every frame nobody set ``meta["class"]`` on, and would be the
    #: wrong tool even if something did: the whole ship branch would run on a frame containing
    #: one person. ``classes:`` selects rows within the frame, which is the question that was
    #: being asked.
    #:
    #: So :meth:`~shipinfer.topology.chain.Topology.from_spec` refuses ``when: class == …`` on
    #: an element that answers ``True`` here, naming ``classes:`` as the fix. It is a
    #: declaration rather than a check on ``impl``, for the reason :meth:`camera_group` is a
    #: hook: the loader must not carry a list of implementation names that read a param.
    selects_rows: ClassVar[bool] = False

    def __init__(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        model: str | None = None,
    ) -> None:
        self.name = name
        self.params: dict[str, Any] = dict(params or {})
        self.model = model
        self._is_open = False
        self._context: ElementContext | None = None
        # Parsed once, here, rather than per edge in the loader: a cap typo in a *class*
        # should fail as soon as that class is instantiated, and the loader instantiates
        # every element in the chain.
        self._input_caps = parse_caps(self.accepts)
        self._output_caps = parse_caps(self.produces)

    # -- declared contract -------------------------------------------------------------

    @property
    def input_caps(self) -> tuple[Caps, ...]:
        """Parsed :attr:`accepts`, in declaration order (= preference order)."""
        return self._input_caps

    @property
    def output_caps(self) -> tuple[Caps, ...]:
        """Parsed :attr:`produces`, in declaration order (= preference order)."""
        return self._output_caps

    @property
    def is_sink(self) -> bool:
        """Whether nothing can follow this element."""
        return not self._output_caps

    # -- lifecycle ---------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def context(self) -> ElementContext | None:
        """What the runner said at :meth:`open`; ``None`` before it was called."""
        return self._context

    def open(self, context: ElementContext | None = None) -> None:
        """Acquire whatever this element needs to run. Idempotent.

        A partially-opened element leaks whatever it did manage to acquire — a socket, a
        decoder thread, a CUDA context on a shared box. The subclass often cannot tell how
        far it got, so unwind unconditionally and best-effort, then re-raise the *original*
        failure, which is the one worth reading.

        Raises:
            ShipInferError: whatever the implementation needs to say. Nothing is swallowed.
        """
        if self._is_open:
            return
        self._context = context if context is not None else ElementContext()
        try:
            self._do_open(self._context)
        except BaseException:
            with contextlib.suppress(Exception):
                self._do_close()
            self._context = None
            raise
        self._is_open = True

    def process(self, item: ChainItem) -> ChainItem | None:
        """Run this element on **one** item.

        Returns:
            The successor item, or ``None`` when this element *filtered* the item — a
            ``when:`` branch that did not fire, or a sink that consumed it. ``None`` never
            means "it failed": a failure raises, because "no ships in this frame" and "the
            segmenter is dead" demand opposite operator responses and an empty result
            distinguishes neither.

        Raises:
            ServerStateError: called before :meth:`open`. A refusal rather than an implicit
                open, because opening on the pipeline worker's thread is how a CUDA context
                ends up on the wrong thread.
        """
        if not self._is_open:
            raise ServerStateError(
                f"element {self.name!r} ({type(self).__name__}) was asked to process "
                "before open(); the runner opens every element at start-up"
            )
        return self._do_process(item)

    def close(self) -> None:
        """Release everything :meth:`open` acquired. Idempotent.

        The element is marked closed *before* the hook runs, so a hook that fails cannot
        leave behind an element that still claims to be open — a reconnect path would then
        skip re-opening it and process nothing forever.
        """
        if not self._is_open:
            return
        self._is_open = False
        try:
            self._do_close()
        finally:
            self._context = None

    def __enter__(self) -> Element:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- per-camera lifecycle ----------------------------------------------------------
    #
    # Two announcements, both no-ops here, both **best-effort**. They exist because a
    # stateful element keys its state on the camera id and nothing was ever telling it when
    # a camera id stopped meaning anything: a removed camera's tracker shard leaked for the
    # process's life, and — worse — a *re-added* camera was refused forever, because its
    # ingest actor mints a fresh `FrameCounter` starting at 0 while the tracker still held
    # the previous run's high-water mark. ADR-018 names remove + add as the one recovery for
    # a lost camera, so "a re-added camera restarts at frame_id = 0" has to be a state the
    # chain can be in.
    #
    # Best-effort is a promise about the runner, not an excuse: the runner calls these for
    # every element, catches whatever one of them raises, logs it with the element's name and
    # carries on, because a tracker that fails to drop a shard must not be able to keep a
    # camera from being removed. An implementation should still raise rather than swallow —
    # the log line is the evidence, and an element that hid the failure would leave the
    # operator with a leak and no record of it.

    def camera_added(self, camera_id: str) -> None:
        """A camera is now placed on this element's runner. Reset any state held for its id.

        Called **after** the ingest actor exists, so a refused placement — a duplicate id, a
        stopped runner, a source nobody registered — announces nothing. That order costs a
        window: the actor is started before this returns, so on a camera that opens instantly
        a frame can reach ``process`` before this hook does. The window is worth it, because
        the alternative is telling every element about a camera that was refused, and an
        element cannot tell that apart from one that was placed and never sent a frame. What
        it costs is bounded and local: an element that meets an unexpected camera id builds
        its state lazily, and at worst the first frame of a *re-added* camera is refused by
        state this call was about to clear.

        **This is not serialised against the walk. Guard your own state.** The hook runs on
        the thread that called ``add_camera``, holding the runner's ``_lifecycle`` lock -- the
        ``RLock`` that serialises ``start``, ``stop``, ``add_camera``, ``remove_camera`` and
        ``drain``, and nothing else. The walk takes no lock at all (``runners/base.py``:
        ``_lifecycle`` is "deliberately *not* taken by ``submit``"), so up to
        :attr:`ElementContext.workers` worker threads can be inside *this element's*
        :meth:`process` -- including for *this camera* -- while this call is running. An
        implementation must therefore hold its own lock around its per-camera table;
        ``pipeline/graph/tracking.py``'s ``_admit`` is the shape (one lock around insertion
        into the map, never held across the work itself, so one slow camera cannot stall the
        other forty-nine).

        **Return promptly.** Every lifecycle operation on this runner queues behind this
        call, so a hook that waits on a model, a socket or another camera's frame stalls the
        shard's whole control plane -- including the ``stop`` that would end the wait.

        Called only between :meth:`open` and :meth:`close`, so an implementation may assume
        its resources exist.
        """

    def camera_removed(self, camera_id: str) -> None:
        """A camera is gone from this element's runner. Drop everything held for its id.

        Called **after** the ingest actor is stopped, which is the safe order and the reason
        it is worth stating: dropping per-camera state while a decoder is still publishing
        would let the very next frame rebuild it, and the element would end up holding state
        for a camera nobody is reading — the leak this hook exists to close, reintroduced by
        the order it was closed in.

        **"After the actor is stopped" is not "after the last frame", and this hook is not
        serialised against the walk.** It runs on the thread that called ``remove_camera``
        (or ``drain``), holding the runner's ``_lifecycle`` lock -- which serialises the
        lifecycle operations against each other and nothing else. The walk takes no lock
        (``runners/base.py``: ``_lifecycle`` is "deliberately *not* taken by ``submit``"), so
        a worker can be inside *this element's* :meth:`process` for *this very camera* at the
        moment this is called, and up to :attr:`ElementContext.workers` of them can be at
        once. A removal racing an in-flight frame is ordinary rather than a bug to assert
        against, and it has two consequences an implementation has to be written for: state
        this call drops can be rebuilt by a frame that was already in the lane, and
        ``remove_camera`` answering ``False`` means the decoder thread was abandoned at its
        deadline rather than joined, so a late item can arrive well after that. So: hold the
        element's own lock around its per-camera table (``pipeline/graph/tracking.py``'s
        ``_admit`` is the shape), make the drop idempotent, and handle a late item as a first
        frame rather than as an impossibility.

        **Return promptly.** ``_lifecycle`` serialises ``start``, ``stop``, ``add_camera``,
        ``remove_camera`` and ``drain``, so a hook that waits holds up every one of them --
        including the ``stop`` that would end the wait.

        Called only between :meth:`open` and :meth:`close`. Shutdown does **not** call it for
        every camera — :meth:`close` releases everything, per camera or not, and announcing
        fifty removals on the way out would only be a slower spelling of that.
        """

    # -- what a runner may ask of any element -------------------------------------------

    def camera_group(self) -> CameraGroup | None:
        """The cameras this element must see **together**, or ``None``. Default: ``None``.

        Asked by a runner that places cameras across processes, once, when it is built. An
        element that associates across cameras — ``mtmc`` is the standing case — holds one
        identity space for a group, and that state lives in one process; a runner that split
        the group would give each half its own tracker and one object two ids, both plausible
        (``docs/arch.md`` §4). So the element declares the membership and the runner enforces
        the placement, which is the only split that works: the element is told
        :attr:`ElementContext.shard_id` and nothing about where any *camera* is, and it opens
        before a single camera has been placed, so it cannot see a split; the runner owns
        ``{camera_id: shard_id}`` and cannot know which cameras belong together.

        **A hook and not a kind test**, deliberately. A runner asking every node
        ``node.element.camera_group()`` needs no ``ElementKind`` switch, no import of an
        element implementation module, and no second parse of a ``params:`` key the element
        has already read — which is the ``if/elif`` per implementation that ADR-017 §2's
        registry exists to delete.

        Called on an element that may not be open, so answer from ``params:`` alone.

        Returns:
            The group, or ``None`` for the ordinary element that does not constrain
            placement — including one that *is* cross-camera but whose chain declared no
            roster, because "the chain did not say" and "the chain grouped nothing" are
            different facts and only the first one lets the runner balance by load.
        """
        return None

    def declared_classes(self) -> tuple[str, ...] | None:
        """The detection labels this slot's ``params: classes:`` names, or ``None``.

        Asked by the loader so it can check them against the labels the chain's detector will
        actually emit — a ``classes: [vessel]`` in front of a detector whose table says
        ``ship`` selects no rows, runs no model and reports nothing wrong, which is the most
        expensive kind of silence in this file.

        A hook, not a second parse of ``params:``, for :meth:`camera_group`'s reason: the
        element has already parsed the key and applied its own refusals, and a loader that
        re-read it would be a second interpretation of one setting.

        Returns:
            The labels, or ``None`` for "this slot named none" — which means *every* row, and
            is not the same as an empty tuple ("select nothing", which is strange but
            unambiguous). Only :attr:`selects_rows` implementations answer anything else.
        """
        return None

    def detection_labels(self) -> tuple[str, ...] | None:
        """The labels this element will *emit* on its detections, or ``None`` if it cannot say.

        The other half of :meth:`declared_classes`. A detector that was configured with a class
        table knows the vocabulary the whole chain downstream of it will speak; the loader asks
        for it so that a ``classes:`` naming a label outside it is refused at load rather than
        discovered as a branch that never fires.

        Returns:
            The declared labels, or ``None`` when this element declares no table — in which
            case the loader checks nothing, because a default table is a fallback and not a
            statement about this deployment's model.
        """
        return None

    # -- subclass hooks ----------------------------------------------------------------

    @abc.abstractmethod
    def _do_open(self, context: ElementContext) -> None:
        """Acquire resources. Called at most once per ``open``/``close`` cycle."""

    @abc.abstractmethod
    def _do_process(self, item: ChainItem) -> ChainItem | None:
        """Do the work on one item. See :meth:`process` for the return contract."""

    def _do_close(self) -> None:
        """Release resources. Optional: a stateless element has nothing to do here."""

    def __repr__(self) -> str:
        state = "open" if self._is_open else "closed"
        kind = self.kind.value if self.kind is not None else "unkinded"
        return f"<{type(self).__name__} {kind}/{self.impl} name={self.name!r} {state}>"
