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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Protocol

from shipinfer.core.errors import ServerStateError, UnknownElementKindError
from shipinfer.core.request import RequestContext
from shipinfer.core.types import Device
from shipinfer.topology.caps import Caps, parse_caps

__all__ = [
    "MODEL_KINDS",
    "ChainItem",
    "Element",
    "ElementContext",
    "ElementKind",
    "ModelResolver",
]


class ElementKind(str, enum.Enum):
    """The eight kinds of step a chain is built from (arch.md §1).

    A closed vocabulary, unlike the implementations behind each kind. The kind decides what
    an element *means* to the chain — whether it needs a model, whether it may be a root,
    whether it is stateful per camera — and a ninth meaning is an architecture change, not
    a configuration one.

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


#: The kinds that run a model from the repository, and therefore need ``model:`` in the
#: chain. ``track``/``mtmc`` are algorithms with their own state, ``decode``/``output`` are
#: I/O; none of the four has a model to name.
MODEL_KINDS: Final = frozenset(
    {ElementKind.DETECT, ElementKind.SEGMENT, ElementKind.EMBED, ElementKind.RECOGNIZE}
)

#: Sentinel for "leave this alone", so ``derive(payload=None)`` can mean *clear it*.
_KEEP: Final[Any] = object()


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

    The last two are **resolved settings, not settings**. ``topology`` is a pure package and
    must not import :mod:`shipinfer.core.settings` — an element that read the settings tree
    itself would also be choosing its own configuration, which is the thing this frozen object
    exists to prevent. So the runner reads the tree once and hands over the two numbers an
    element cannot otherwise know, and an element resolves them with a fixed precedence:
    its own ``params:``, then this context, then its module default. Without them the two
    settings keys mirrored in ``topology/elements/pool.py`` would apply to nothing.
    """

    shard_id: int = 0
    device: Device | None = None
    models: ModelResolver | None = None
    stage_timeout_s: float | None = None
    input_name: str | None = None


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
    #: Whether :meth:`open` resolves this element against :attr:`ElementContext.models` --
    #: **the pool question, and only that one**. ``pool`` answers ``True``; ``mock`` answers
    #: ``False`` because it invents a box.
    #:
    #: Two readers, and they ask it together rather than each inventing their own test: the
    #: in-process runner's expiry gate (which elements submit and wait, and so can be late)
    #: and ``run``'s pool predicate, which builds an
    #: :class:`~shipinfer.engine.InferenceServer` only when
    #: :attr:`~shipinfer.runners.base.Runner.needs_model_pool` is ``True`` as well
    #: (``cli/commands/run.py``) -- a ``fleet`` launcher runs the same chain and builds none,
    #: because its shards each build their own.
    #:
    #: **"Must this slot name a ``model:``?" is a different question and a different
    #: ClassVar**, owned by the seam slice that replaces the kind-level check. The two are
    #: not the same predicate and will visibly diverge: an ``nvinfer`` element names a
    #: ``model:`` -- GStreamer needs the artefact -- and never touches this process's pool,
    #: so it answers yes there and ``False`` here. Neither can be inferred from :attr:`kind`,
    #: because a ``detect`` slot is a model kind whichever ``impl`` fills it.
    #:
    #: A class that answers ``True`` must raise from :meth:`_do_open` when the context carries
    #: no pool, so that the declaration and the requirement cannot drift; ``tests/topology``
    #: walks every registered implementation and checks exactly that.
    needs_model: ClassVar[bool] = False

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
