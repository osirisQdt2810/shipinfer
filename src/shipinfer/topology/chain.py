"""The chain: a YAML declaration, validated into a runnable DAG of elements.

The second of arch.md's three concepts (§1). A **topology** is the declarative chain — which
steps run, in what order, with which branch conditions — and it is *data*, not code:

.. code-block:: yaml

    elements:
      decode:    {impl: gstreamer-gpu}
      detect:    {impl: pool, model: ship_detector}
      segment:   {impl: pool, model: ship_segmenter, when: class == ship}
      embed_ship: {impl: pool, model: ship_embedder, after: segment}

Two ideas do all the work here.

**Order is the declaration's order.** An element with no ``after:`` follows the one declared
before it, which is how the common case — a straight line of nine steps — needs no wiring at
all. ``after:`` overrides it, and is what expresses a branch or a fan-in. The cost of this
convenience is that reordering two lines in the file reorders the chain, so
:meth:`Topology.describe` prints the resolved edges and the loader refuses anything
ambiguous.

**Everything is checked at load time.** Kinds, implementations, model names, cycles,
structure and — the one that matters most — the caps of every pair of elements that can
hand data to each other. Validation lives in exactly one place,
:meth:`Topology.from_spec`, so there is no second door into a half-checked chain. A
mis-wired chain stops a deploy; it does not become a camera that quietly produces no
detections at 3 a.m.

"Every pair that can hand data to each other" is wider than "every declared edge", and both
extensions exist because a per-edge check alone is evadable:

* a wildcard half of an element's ``produces`` is **propagated** — resolved from what
  actually arrives at that element — before its outbound edges are negotiated, so a
  cap-transparent passthrough cannot launder a ``nv12@gpu`` producer into a ``bgr@cpu``
  consumer (:func:`_resolve_produced`);
* a ``when:`` element is skipped for items its condition rejects, so its predecessor hands
  straight to its successor; that **bypass** pair is negotiated as well
  (:meth:`ElementNode.admits`).

The loader is deliberately **not** clever: where two adjacent elements disagree it refuses
rather than inserting a conversion. See :mod:`shipinfer.topology.caps` for why an implicit
device-to-host copy is the worst thing this file could do.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from shipinfer.core.errors import (
    CapsMismatchError,
    ChainCycleError,
    ChainSpecError,
    ChainStructureError,
    ConditionSyntaxError,
    UnknownElementError,
)
from shipinfer.topology.base import MODEL_KINDS, ChainItem, Element, ElementKind
from shipinfer.topology.caps import ANY, Caps, negotiate
from shipinfer.topology.registry import create_element

__all__ = [
    "ChainSpec",
    "Condition",
    "Edge",
    "ElementNode",
    "ElementSpec",
    "Topology",
    "load_topology",
]

#: ``<field> <op> <value>``, with the value's surrounding whitespace and quotes stripped.
_CONDITION = re.compile(
    r"^\s*(?P<field>[A-Za-z_][A-Za-z0-9_.]*)\s*(?P<op>==|!=)\s*(?P<value>\S.*?)\s*$"
)


@dataclass(frozen=True, slots=True)
class Condition:
    """A ``when:`` expression: run this element only for items whose metadata matches.

    Two operators and one comparison, on purpose. ``when: class == ship`` is the whole
    requirement (arch.md §1) — the segmenter runs on ships, the person embedder on people —
    and a chain file is not the place to grow an expression language. When one genuinely
    needs more, that is a filter element with its own implementation, testable on its own.

    The comparison is on the **string form** of the metadata value. A detector may put a
    class name, an enum or a label id in ``meta["class"]``, and a chain file has only text
    to compare against; ``str(value) == "ship"`` is the one rule that reads the same for all
    three.
    """

    field: str
    op: Literal["==", "!="]
    value: str

    @classmethod
    def parse(cls, text: str) -> Condition:
        """Parse ``"class == ship"``.

        Raises:
            ConditionSyntaxError: naming the accepted shape. Refusing beats guessing: a
                ``when: class = ship`` silently read as "always true" would run the heaviest
                model in the chain on every frame.
        """
        match = _CONDITION.match(text)
        if match is None:
            raise ConditionSyntaxError(
                f"condition {text!r} must be '<field> == <value>' or '<field> != <value>'"
            )
        value = match["value"].strip().strip("'\"")
        return cls(match["field"], match["op"], value)  # type: ignore[arg-type]

    def matches(self, meta: Mapping[str, Any]) -> bool:
        """Whether an item's metadata satisfies this condition.

        A **missing** field satisfies neither operator. "The frame has no class yet" is not
        evidence that the class differs, and letting ``!=`` fire on absence would run the
        person branch on a frame nobody had classified — the kind of bug that shows up as a
        mysterious extra 15% of GPU load.
        """
        if self.field not in meta:
            return False
        actual = str(meta[self.field])
        return actual == self.value if self.op == "==" else actual != self.value

    def __str__(self) -> str:
        return f"{self.field} {self.op} {self.value}"


class _Strict(BaseModel):
    """Base for the chain schema: unknown keys are errors, and a spec never mutates.

    ``extra="forbid"`` is the point of having a schema at all. A tolerated ``mdoel:
    ship_detector`` would load a chain whose detector has no model and fail much later, far
    from the typo.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class ElementSpec(_Strict):
    """One slot in the chain file, exactly as written.

    A dumb record: it says what the YAML said and validates only its own shape. Whether the
    implementation exists, whether the model is needed and whether the caps line up are
    *chain* questions, and they live in :meth:`Topology.from_spec` so that there is one
    place to read the rules.
    """

    #: Registered implementation name, e.g. ``pool`` or ``gstreamer-gpu``.
    impl: str
    #: Explicit kind, when the slot name does not imply it. Usually omitted. A **string**,
    #: not the enum, on purpose: pydantic would answer ``kind: decdoe`` with a schema error
    #: while ``decdoe:`` as a slot name gets an
    #: :class:`~shipinfer.core.errors.UnknownElementKindError`, and one typo deserves one
    #: error type. :meth:`Topology.from_spec` resolves it through
    #: :meth:`~shipinfer.topology.base.ElementKind.parse`, which raises that one.
    kind: Optional[str] = None
    #: Repository model name. Required for the four model kinds, meaningless for the rest.
    model: Optional[str] = None
    #: Branch condition, e.g. ``class == ship``.
    when: Optional[str] = None
    #: Predecessors. **Absent** means "the element declared before me"; an explicit empty
    #: list means "none", which makes this a root and is legal only for ``decode``.
    after: Optional[list[str]] = None
    #: Statefulness hint for the runner: ``camera`` pins this element to the camera's home
    #: shard (arch.md §5⑥). Not interpreted here; the runner reads it.
    #:
    #: **Not honoured by any runner yet.** ``InprocessRunner`` shares one element instance
    #: across every worker, so with ``workers > 1`` two frames of one camera can be inside a
    #: ``per: camera`` element at the same time and its per-camera order can invert. Nothing
    #: stateful ships today (the mocks are stateless and a ``pool`` element holds only a model
    #: handle), so this is a promise not yet kept rather than a live bug. Resolved in phase C,
    #: either as a per-camera element instance or as a camera-keyed lock around a
    #: ``per: camera`` element.
    per: Optional[Literal["camera", "frame", "global"]] = None
    #: ``global`` marks an element with one instance for the whole fleet (mtmc). Carried, not
    #: yet interpreted — same phase C caveat as :attr:`per`.
    scope: Optional[Literal["shard", "global"]] = None
    #: Implementation-specific settings, handed to the element untouched.
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("after", mode="before")
    @classmethod
    def _one_or_many(cls, value: object) -> object:
        """``after: segment`` and ``after: [segment, detect]`` are both ordinary usage."""
        return [value] if isinstance(value, str) else value

    @property
    def predecessors(self) -> tuple[str, ...]:
        """The explicitly declared predecessors; empty when ``after`` was omitted too."""
        return tuple(self.after or ())

    @property
    def declares_predecessors(self) -> bool:
        """Whether the file said anything about this element's predecessors."""
        return self.after is not None

    @property
    def condition(self) -> Condition | None:
        """The parsed ``when:``, or ``None``.

        Raises:
            ConditionSyntaxError: the expression is malformed.
        """
        return None if self.when is None else Condition.parse(self.when)


class ChainSpec(_Strict):
    """A whole chain file, parsed but not yet validated as a chain.

    ``elements`` keeps the file's order — Python dicts and ``yaml.safe_load`` both preserve
    insertion order — and that order is load-bearing: it is the default predecessor rule.
    """

    #: Human-readable chain name. Defaults to the file stem, as a model config defaults its
    #: name to its directory.
    name: str = ""
    elements: dict[str, ElementSpec] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, text: str, *, name: str = "", source: str = "<string>") -> ChainSpec:
        """Parse chain YAML.

        Args:
            text: the file's contents.
            name: chain name to use when the document omits one.
            source: what to name in an error message.

        Raises:
            ChainSpecError: the document is not YAML, is not a mapping, or does not match
                the schema. The source is always in the message — a pydantic report with no
                file name is nearly useless once a deployment has more than one topology.
                Only schema failures are wrapped: anything else escaping this call is a bug
                here, not a bad file, and swallowing it into a ChainSpecError would send an
                operator to edit a chain that is fine.
        """
        try:
            raw: Any = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise ChainSpecError(f"invalid YAML in {source}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ChainSpecError(f"{source}: expected a mapping at the top level")
        data = dict(raw)
        data.setdefault("name", name)
        try:
            # `model_validate`, not `cls(**data)`: keyword expansion turns a non-string YAML
            # key (`1: foo` at the top level) into a TypeError from CPython rather than a
            # validation error, which would need a broader `except` here to catch.
            return cls.model_validate(data)
        except (ValidationError, ValueError) as exc:
            raise ChainSpecError(f"{source}: {exc}") from exc

    @classmethod
    def from_file(cls, path: str | Path) -> ChainSpec:
        """Read and parse one chain file.

        Raises:
            ChainSpecError: unreadable, or not a valid chain document.
        """
        file = Path(path)
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ChainSpecError(f"cannot read topology {file}: {exc}") from exc
        return cls.from_yaml(text, name=file.stem, source=str(file))


@dataclass(frozen=True, slots=True)
class ElementNode:
    """One validated element, with its resolved place in the chain.

    Args:
        name: the slot name from the file — the element's identity in logs and metrics.
        kind: resolved from ``kind:`` or inferred from the slot name.
        element: the instantiated implementation. Instantiated at load time, which is why
            :class:`~shipinfer.topology.base.Element` constructors must be hardware-free.
        spec: what the file said, kept for the runner (``per``, ``scope``, ``params``).
        inputs: predecessor slot names, resolved.
        outputs: successor slot names, resolved.
        condition: the parsed ``when:``, or ``None``.
        donor: which predecessor donates payload and caps at a fan-in. Filled in by
            :class:`Topology`, which is the only object that knows the negotiated edge caps
            this is resolved from; ``None`` on a root.
    """

    name: str
    kind: ElementKind
    element: Element
    spec: ElementSpec
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    condition: Condition | None = None
    donor: str | None = None

    @property
    def is_root(self) -> bool:
        return not self.inputs

    @property
    def is_sink(self) -> bool:
        return not self.outputs

    def admits(self, item: ChainItem) -> bool:
        """Whether this element should see ``item`` at all.

        Only the ``when:`` branch — the caps were settled at load time. Named ``admits``
        rather than ``accepts`` because :attr:`Element.accepts` is the cap declaration, and
        two different questions should not share a word.

        Lives on the node rather than in a runner so that all three runners (arch.md §1)
        answer it identically; a per-runner copy of this is how ``inprocess`` and ``fleet``
        would come to disagree about which frames the segmenter sees.

        **The contract for a ``False``: skip and continue.** The item is *not* dropped and
        the walk does *not* stop. A runner hands it, unchanged, to this element's successors
        — ``when: class == ship`` means "the segmenter does not run on people", not "people
        leave the chain", and a person still has to reach the tracker and the output. Two
        consequences follow, and both are load-time rules rather than runner conventions:

        * the hand-over from this element's predecessor to its successor is real, so the
          loader negotiates that **bypass** pair too and refuses a chain where it cannot
          work (see :func:`_negotiate_edges`);
        * dropping an item is a different act, spelled by an element returning ``None`` from
          :meth:`~shipinfer.topology.base.Element.process`, not by a condition.
        """
        return self.condition is None or self.condition.matches(item.meta)


@dataclass(frozen=True, slots=True)
class Edge:
    """A validated hand-over between two elements, and the cap it carries.

    The negotiated cap is stored because it is a *decision*, not a derivation: it depends on
    both sides' declaration order, and the runner and every log line should read the same
    answer the loader reached.

    **The cap belongs to the edge, not to the element.** Each edge is negotiated on its own
    pair, so two edges out of one producer, or two edges into one consumer, may carry
    different caps — a fan-in where a detector hands ``nv12@gpu`` and a tracker hands
    ``meta@cpu`` into the same element is legal and does happen. Read the cap from the edge
    you are about to traverse; asking "what cap does ``track`` take?" has no single answer.
    The one exception is an element that wildcards a half of its ``produces``: there the
    loader must pick one cap for the outbound edge, so it requires the inbound ones to agree
    and refuses when they do not.
    """

    producer: str
    consumer: str
    caps: Caps

    def __str__(self) -> str:
        return f"{self.producer} -> {self.consumer} [{self.caps}]"


class Topology:
    """A validated chain: elements in topological order, with typed edges.

    Build one with :meth:`from_spec`, :meth:`from_file` or :func:`load_topology`. The
    constructor takes already-validated parts on purpose — it is not a second door into a
    chain nobody checked.

    A topology owns **stateful** element instances, so loading the same file twice gives two
    independent chains. That is what a runner needs (one chain per shard, sometimes one per
    worker) and it is why nothing here is cached.
    """

    def __init__(self, name: str, nodes: Sequence[ElementNode], edges: Sequence[Edge]) -> None:
        self._name = name
        self._edges = tuple(edges)
        # Donors are resolved *here* rather than in a runner, for the reason
        # `ElementNode.admits` gives for itself: it is topology data, and three runners that
        # each resolved it would eventually disagree about which branch donated a frame. It
        # cannot be resolved by `from_spec` before this point, because it reads the caps the
        # loader negotiated per edge, and it cannot live on the node's constructor, because a
        # node is built before its edges exist.
        self._nodes = _with_donors(nodes, self._edges)
        self._by_name = {node.name: node for node in self._nodes}

    # -- construction ------------------------------------------------------------------

    @classmethod
    def from_spec(cls, spec: ChainSpec) -> Topology:
        """Validate a parsed chain and instantiate its elements.

        The single place the chain rules live, in the order a reader would want them: what
        each element *is*, then how they are wired, then whether the wiring can carry data.
        Each step raises before the next runs, so the first message an operator sees names
        the first thing wrong rather than a consequence of it.

        Raises:
            ChainSpecError: the chain declares no elements.
            UnknownElementKindError: a slot, or an explicit ``kind:``, names no kind.
            ChainStructureError: a model kind with no ``model:``, a root that is not a
                decode element, does not say what it produces or carries a ``when:``, no
                output element, an output element with a successor, a branch that reaches no
                output, or a wildcard ``produces`` whose inbound edges disagree.
            UnknownElementImplError: an ``impl:`` nobody registered.
            ConfigurationError: a registered class whose kind is not its registry's.
            UnknownElementError: an ``after:`` naming an element that is not declared.
            ChainCycleError: the ``after:`` edges form a cycle.
            CapsMismatchError: two elements that can hand data to each other agree on no
                format/location — including the *bypass* pair created when a ``when:``
                element is skipped.
        """
        if not spec.elements:
            raise ChainSpecError(
                f"chain {spec.name or '<unnamed>'} declares no elements; "
                "a topology is a chain of at least a decode and an output"
            )

        kinds = {
            slot: (
                ElementKind.parse(declared.kind)
                if declared.kind is not None
                else ElementKind.infer(slot)
            )
            for slot, declared in spec.elements.items()
        }

        for slot, declared in spec.elements.items():
            if kinds[slot] in MODEL_KINDS and not declared.model:
                raise ChainStructureError(
                    f"element {slot!r} is a {kinds[slot].value} element and needs "
                    "`model: <repository model name>`"
                )

        elements = {
            slot: create_element(
                kinds[slot], declared.impl, slot, declared.params, model=declared.model
            )
            for slot, declared in spec.elements.items()
        }

        inputs = _resolve_predecessors(spec)
        order = _topological_order(inputs)
        outputs = _consumers(inputs)

        nodes = [
            ElementNode(
                name=slot,
                kind=kinds[slot],
                element=elements[slot],
                spec=spec.elements[slot],
                inputs=inputs[slot],
                outputs=tuple(outputs[slot]),
                condition=spec.elements[slot].condition,
            )
            for slot in order
        ]
        _check_structure(nodes)
        edges = _negotiate_edges(nodes)
        return cls(spec.name, nodes, edges)

    @classmethod
    def from_file(cls, path: str | Path) -> Topology:
        """Read, parse and validate one chain file."""
        return cls.from_spec(ChainSpec.from_file(path))

    # -- the validated chain -----------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> tuple[ElementNode, ...]:
        """Every element, in topological order — a legal execution order."""
        return self._nodes

    @property
    def edges(self) -> tuple[Edge, ...]:
        return self._edges

    @property
    def roots(self) -> tuple[ElementNode, ...]:
        """The decode elements: where frames enter."""
        return tuple(node for node in self._nodes if node.is_root)

    @property
    def sinks(self) -> tuple[ElementNode, ...]:
        """The elements nothing follows: where results leave."""
        return tuple(node for node in self._nodes if node.is_sink)

    def node(self, name: str) -> ElementNode:
        """One element by slot name.

        Raises:
            KeyError: no such slot; the message lists the ones there are. A plain
                ``KeyError`` and *not* the typed
                :class:`~shipinfer.core.errors.UnknownElementError`, which is deliberate: by
                the time a ``Topology`` exists every name in the file has been resolved, so a
                miss here comes from a mistyped literal in a runner or a CLI, not from a
                chain an operator can fix. Raising a ``ConfigurationError`` would send them
                to edit a file that is correct, and would let a caller's bug be swallowed by
                the same ``except TopologyError`` that handles bad configuration.
        """
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"topology {self._name or '<unnamed>'} has no element {name!r}; "
                f"declared: {sorted(self._by_name)}"
            ) from None

    def successors(self, name: str) -> tuple[ElementNode, ...]:
        """The elements that consume this one's output, in declaration order."""
        return tuple(self.node(successor) for successor in self.node(name).outputs)

    def predecessors(self, name: str) -> tuple[ElementNode, ...]:
        """The elements whose output this one consumes."""
        return tuple(self.node(predecessor) for predecessor in self.node(name).inputs)

    def describe(self) -> str:
        """The resolved chain as text — what a ``shipinfer topology show`` would print.

        Worth having because the default-predecessor rule means the file does not state most
        of the wiring. This is where an operator checks that "declared before me" resolved to
        what they meant.
        """
        lines = [f"topology {self._name or '<unnamed>'} ({len(self._nodes)} elements)"]
        for node in self._nodes:
            marks = []
            if node.spec.model:
                marks.append(f"model={node.spec.model}")
            if node.condition is not None:
                marks.append(f"when={node.condition}")
            if node.is_root:
                marks.append("root")
            if node.is_sink:
                marks.append("sink")
            suffix = f"  [{', '.join(marks)}]" if marks else ""
            lines.append(f"  {node.name}: {node.kind.value}/{node.element.impl}{suffix}")
        lines.extend(f"  {edge}" for edge in self._edges)
        return "\n".join(lines)

    def __iter__(self) -> Iterator[ElementNode]:
        return iter(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return (
            f"<Topology {self._name or '<unnamed>'}: "
            f"{len(self._nodes)} elements, {len(self._edges)} edges>"
        )


def load_topology(path: str | Path) -> Topology:
    """Read and validate one topology file — the counterpart of ``load_model_config``.

    What it will **not** do, and the docstring says so because the temptation is real: it
    never inserts a conversion to make two disagreeing caps fit. ``nv12@gpu`` feeding
    ``bgr@cpu`` is refused with a
    :class:`~shipinfer.core.errors.CapsMismatchError` naming both sides, because the
    alternative is a 1000 fps chain that silently downloads every frame to host memory and
    looks healthy while doing it (arch.md §8). Declare matching caps, or spell the convert
    element out in the file.

    Raises:
        ChainSpecError: unreadable, or not a valid chain document.
        TopologyError: any of the chain rules — see :meth:`Topology.from_spec`.
    """
    return Topology.from_file(path)


# -- validation steps -------------------------------------------------------------------


def _resolve_predecessors(spec: ChainSpec) -> dict[str, tuple[str, ...]]:
    """Apply the default-predecessor rule, then check every name resolves.

    The default is the previously declared element, which is what makes a nine-step straight
    chain need no wiring. An explicit ``after:`` — including an explicit empty list — always
    wins.
    """
    known = list(spec.elements)
    inputs: dict[str, tuple[str, ...]] = {}
    previous: str | None = None
    for slot, declared in spec.elements.items():
        if declared.declares_predecessors:
            resolved = declared.predecessors
        else:
            resolved = () if previous is None else (previous,)
        for target in resolved:
            if target == slot:
                raise ChainCycleError([slot])
            if target not in spec.elements:
                raise UnknownElementError(slot, target, known)
        # Duplicates would produce two edges between the same pair, and a fan-in with a
        # repeated producer is a typo rather than a topology.
        inputs[slot] = tuple(dict.fromkeys(resolved))
        previous = slot
    return inputs


def _consumers(inputs: Mapping[str, tuple[str, ...]]) -> dict[str, list[str]]:
    """Invert ``consumer -> producers`` into ``producer -> consumers``, order preserved."""
    consumers: dict[str, list[str]] = {slot: [] for slot in inputs}
    for consumer, producers in inputs.items():
        for producer in producers:
            consumers[producer].append(consumer)
    return consumers


def _topological_order(inputs: Mapping[str, tuple[str, ...]]) -> list[str]:
    """Kahn's algorithm, keeping declaration order among ready elements.

    **Why not** :class:`graphlib.TopologicalSorter`, which is in the standard library and
    would be the ponytail-principle answer: its ``static_order`` iterates the ready set as a
    ``set`` of ``str``, and the iteration order of a string set depends on
    ``PYTHONHASHSEED``. Two runs of the same chain file would then print different
    :meth:`Topology.describe` output, and this chain's default-predecessor rule already makes
    declaration order semantic — a nine-element straight line must come out in the order it
    was written. Kahn over a ``deque`` seeded in declaration order gives exactly that in
    twelve lines. (``graphlib`` is still the right tool for a cycle *check* on its own; the
    tie-break is the whole reason it is not used here.)

    Raises:
        ChainCycleError: with one real cycle as a path, found by :func:`_find_cycle`.
    """
    pending = {slot: len(producers) for slot, producers in inputs.items()}
    consumers = _consumers(inputs)

    ready = deque(slot for slot in inputs if pending[slot] == 0)
    order: list[str] = []
    while ready:
        slot = ready.popleft()
        order.append(slot)
        for consumer in consumers[slot]:
            pending[consumer] -= 1
            if pending[consumer] == 0:
                ready.append(consumer)
    if len(order) != len(inputs):
        # Not `[slot for slot, count in pending.items() if count > 0]`: that set is the cycle
        # *plus every element downstream of it*, so a two-element cycle in a nine-element
        # chain gets reported as seven names and the reader has to find the loop themselves.
        raise ChainCycleError(_find_cycle(inputs))
    return order


def _find_cycle(inputs: Mapping[str, tuple[str, ...]]) -> list[str]:
    """One cycle, as the path data would try to travel: a depth-first back edge.

    Walks forward (producer to consumer) and starts from the slots in declaration order, so
    the cycle reported for a given file is always the same one — an error message that moves
    between runs is one nobody trusts. Recursive, which is fine: a chain is a dozen elements
    written by hand, not a graph.

    Returns:
        The cycle in flow order, e.g. ``["detect", "track"]`` for
        ``detect -> track -> detect``. Empty only if there is no cycle, which the one caller
        has already ruled out by counting.
    """
    consumers = _consumers(inputs)
    grey: set[str] = set()
    done: set[str] = set()
    path: list[str] = []

    def walk(slot: str) -> list[str]:
        grey.add(slot)
        path.append(slot)
        for consumer in consumers[slot]:
            if consumer in grey:
                return path[path.index(consumer) :]
            if consumer not in done:
                found = walk(consumer)
                if found:
                    return found
        path.pop()
        grey.discard(slot)
        done.add(slot)
        return []

    for slot in inputs:
        if slot not in done:
            cycle = walk(slot)
            if cycle:
                return cycle
    return []


def _check_structure(nodes: Sequence[ElementNode]) -> None:
    """The rules that separate a DAG from a runnable chain.

    A chain must *start* somewhere frames come from, and *end* somewhere results go. All of
    these are start-up refusals because the failure they prevent is silent: a chain with no
    output element runs every model at full cost and emits nothing.

    1. every root is a ``decode`` element;
    2. a root needs no input caps, because nothing precedes it;
    3. a root **states** what it produces — no wildcard half, since there is no inbound edge
       to resolve one from (:func:`_resolve_produced` is the other half of that rule);
    4. a root carries no ``when:``, because there is no metadata for one to read yet;
    5. the chain has at least one ``output`` element;
    6. every ``output`` element is a sink;
    7. every element reaches an ``output``.

    No "the chain has no root" rule: it cannot happen. This runs after
    :func:`_topological_order`, and a non-empty finite DAG always has a node of in-degree
    zero — a graph where every node has a predecessor has a cycle, which is the error the
    sort has already raised.
    """
    for root in (node for node in nodes if node.is_root):
        if root.kind is not ElementKind.DECODE:
            raise ChainStructureError(
                f"element {root.name!r} has no predecessor, so it is a root, but a root "
                f"must be a decode element (it is a {root.kind.value} element); add "
                "`after: <element>` to place it in the chain"
            )
        if any(not cap.is_wildcard for cap in root.element.input_caps):
            raise ChainStructureError(
                f"root element {root.name!r} requires input caps "
                f"{[str(cap) for cap in root.element.input_caps]} but nothing precedes it"
            )
        if any(cap.is_wildcard for cap in root.element.output_caps):
            raise ChainStructureError(
                f"root element {root.name!r} declares output caps "
                f"{[str(cap) for cap in root.element.output_caps]}, but a root has no "
                "inbound edge to resolve a `*` from; a root must say what it produces "
                "(for example `nv12@gpu`), or every cap downstream of it is unknown"
            )
        if root.condition is not None:
            # The one condition rule the loader can decide statically. Everything a `when:`
            # can test is written into `item.meta` by an element, and at ingest the meta is
            # empty -- `Condition.matches` is False for a missing field (absence is not
            # evidence), so this decoder would be skipped for every frame of every camera
            # and the chain would run, healthy-looking, producing nothing at all.
            raise ChainStructureError(
                f"root element {root.name!r} carries `when: {root.condition}`, but a root "
                "has no predecessor to write that metadata: the condition would be false "
                "for every frame and the chain would ingest nothing. Move the condition "
                "onto an element downstream of the decoder"
            )

    outputs = [node for node in nodes if node.kind is ElementKind.OUTPUT]
    if not outputs:
        raise ChainStructureError(
            "the chain has no output element; every chain ends in an element of kind "
            "`output`, or its results go nowhere"
        )
    # Said separately from the rule above, because "no output sink" is the wrong diagnosis
    # for `mtmc: {after: output}`: the chain has an output element, it just is not the end.
    for emitter in outputs:
        if not emitter.is_sink:
            raise ChainStructureError(
                f"output element {emitter.name!r} must be a sink, but the chain gives it "
                f"successors {list(emitter.outputs)}; an output element emits results and "
                "hands nothing on, so move those elements before it"
            )

    # Reverse reachability, not forward: in a DAG every element is reachable from some
    # root by construction, so the check with teeth is the other direction — a branch
    # whose results no output emits is a configuration mistake that costs GPU time and
    # produces nothing.
    by_name = {node.name: node for node in nodes}
    reaching = {node.name for node in outputs}
    frontier = deque(reaching)
    while frontier:
        node = by_name[frontier.popleft()]
        for producer in node.inputs:
            if producer not in reaching:
                reaching.add(producer)
                frontier.append(producer)
    stranded = [node.name for node in nodes if node.name not in reaching]
    if stranded:
        raise ChainStructureError(
            f"element(s) {sorted(stranded)} reach no output element; their results would "
            "be computed and dropped"
        )


@dataclass(frozen=True, slots=True)
class _Arrival:
    """One cap that can be handed to an element, and where it came from.

    ``skipped`` is set when this cap arrives only because a ``when:`` element in between did
    not fire, which is the difference between an edge and a bypass in an error message.
    """

    origin: str
    caps: Caps
    skipped: str | None = None


def _agreed_half(node: str, arrivals: Sequence[_Arrival], half: str) -> str:
    """The one value every arriving cap has for ``format`` or ``location``.

    Raises:
        ChainStructureError: the arriving caps disagree. Only reached for a half the element
            wildcarded: a fan-in whose edges genuinely carry different caps is legal (see
            :class:`Topology`), but an element that says "I hand on whatever I was given"
            cannot be handing on two different things at once — the loader would have to
            pick one and stamp it on the outbound edge.
    """
    values = {getattr(arrival.caps, half) for arrival in arrivals}
    concrete = sorted(values - {ANY})
    if len(concrete) > 1:
        raise ChainStructureError(
            f"element {node!r} declares a wildcard {half} in its `produces`, so the loader "
            f"resolves it from what arrives; but its inputs disagree: "
            f"{[f'{a.origin} -> {a.caps}' for a in arrivals]}. Declare the {half} "
            f"explicitly on {node!r}, or stop feeding it two different {half}s"
        )
    return concrete[0] if concrete else ANY


def _resolve_produced(node: ElementNode, arrivals: Sequence[_Arrival]) -> tuple[Caps, ...]:
    """The caps an element really hands on, with any wildcard half filled in.

    This is cap *propagation*, and it is what makes arch.md §8's promise hold for a whole
    chain rather than for one edge at a time. An element declaring ``accepts: *@*`` and
    ``produces: *@*`` — a passthrough, a filter, a tee — would otherwise satisfy an
    ``nv12@gpu`` producer on its input and a ``bgr@cpu`` consumer on its output, and the
    device-to-host download the loader exists to refuse would reappear in the middle of the
    chain with both edges reported as valid. GStreamer resolves this the same way: a
    passthrough's output caps are its *negotiated* input caps, not its template.

    Only the wildcard halves are resolved. A concrete ``produces`` is what the element said
    it makes, and nothing upstream gets to overrule it — that is how the tracker turns
    ``nv12@gpu`` into ``meta@cpu``.

    Raises:
        ChainStructureError: the element wildcards a half and its inputs disagree on it.
    """
    declared = node.element.output_caps
    if not any(cap.is_wildcard for cap in declared):
        return declared
    resolved = [
        Caps(
            cap.format if cap.format != ANY else _agreed_half(node.name, arrivals, "format"),
            (
                cap.location
                if cap.location != ANY
                else _agreed_half(node.name, arrivals, "location")
            ),
        )
        for cap in declared
    ]
    # Two declared caps can resolve to the same thing (`*@gpu, nv12@*` behind an nv12@gpu
    # producer). Dedupe, preserving preference order, so the error messages stay readable.
    return tuple(dict.fromkeys(resolved))


def _negotiate_edges(nodes: Sequence[ElementNode]) -> tuple[Edge, ...]:
    """One :class:`Edge` per hand-over, or a refusal naming both sides.

    Walks ``nodes`` in **topological order**, which is what lets an element's outbound caps
    be resolved from its inbound ones: every predecessor has already been negotiated by the
    time an element is reached. That ordering is the whole mechanism; a loop over edges in
    any other order could not propagate anything.

    Two hand-overs are checked per element, and only the first becomes an :class:`Edge`:

    * the **edge** itself, producer to consumer;
    * for a ``when:`` element, the **bypass** — its predecessor straight to its successor.
      An item the condition rejects is skipped past the element and continues down the
      chain (see :meth:`ElementNode.admits`), so that pair really does exchange data at
      runtime and deserves the same refusal. It is not an ``Edge`` because no element hands
      anything over there; it is the *absence* of one element on an existing route.
    """
    by_name = {node.name: node for node in nodes}
    arrivals: dict[str, list[_Arrival]] = {node.name: [] for node in nodes}
    edges: list[Edge] = []
    for node in nodes:
        produced = _resolve_produced(node, arrivals[node.name])
        for consumer_name in node.outputs:
            consumer = by_name[consumer_name]
            accepted = consumer.element.input_caps
            caps = negotiate(produced, accepted)
            if caps is None:
                raise CapsMismatchError(
                    node.name,
                    [str(cap) for cap in produced],
                    consumer.name,
                    [str(cap) for cap in accepted],
                )
            edges.append(Edge(node.name, consumer_name, caps))
            arrivals[consumer_name].append(_Arrival(node.name, caps))
            if node.condition is None:
                continue
            for arrival in arrivals[node.name]:
                if negotiate([arrival.caps], accepted) is None:
                    raise CapsMismatchError(
                        arrival.origin,
                        [str(arrival.caps)],
                        consumer.name,
                        [str(cap) for cap in accepted],
                        skipped=node.name,
                    )
                arrivals[consumer_name].append(
                    _Arrival(arrival.origin, arrival.caps, skipped=node.name)
                )
    return tuple(edges)


def _with_donors(
    nodes: Sequence[ElementNode], edges: Sequence[Edge]
) -> tuple[ElementNode, ...]:
    """Fill in every node's :attr:`ElementNode.donor` from the negotiated edge caps.

    The donor is the first predecessor, **in the consumer's ``accepts`` order**, whose edge
    carries a cap this element prefers. ``Element.accepts`` is a preference list and
    :func:`~shipinfer.topology.caps.negotiate` already treats it as one, so this reads the
    same order the loader read when it negotiated. A node whose predecessors all carry the
    same cap therefore adopts the first one *declared*, which is what a reader of the chain
    file would expect.

    Why it matters at all: at a fan-in the metadata is the union of every branch, but the
    payload and caps have to come from exactly one of them — half a frame handle plus half a
    metadata dict is not a payload — and picking by the negotiated cap makes that choice the
    loader's rather than a runner's.

    Resolved once per topology, so a runner's per-frame merge is an attribute read.
    """
    caps_by_edge = {(edge.producer, edge.consumer): edge.caps for edge in edges}
    return tuple(
        node if node.is_root else replace(node, donor=_donor_for(node, caps_by_edge))
        for node in nodes
    )


def _donor_for(node: ElementNode, caps_by_edge: Mapping[tuple[str, str], Caps]) -> str:
    """One node's donor. Falls back to the first predecessor when no edge cap is known."""
    for declared in node.element.input_caps:
        for name in node.inputs:
            caps = caps_by_edge.get((name, node.name))
            if caps is not None and caps.matches(declared):
                return name
    return node.inputs[0]
