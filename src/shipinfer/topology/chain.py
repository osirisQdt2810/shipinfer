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
structure and — the one that matters most — the caps of every adjacent pair. Validation
lives in exactly one place, :meth:`Topology.from_spec`, so there is no second door into a
half-checked chain. A mis-wired chain stops a deploy; it does not become a camera that
quietly produces no detections at 3 a.m.

The loader is deliberately **not** clever: where two adjacent elements disagree it refuses
rather than inserting a conversion. See :mod:`shipinfer.topology.caps` for why an implicit
device-to-host copy is the worst thing this file could do.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from shipinfer.core.errors import (
    CapsMismatchError,
    ChainCycleError,
    ChainSpecError,
    ChainStructureError,
    ConditionSyntaxError,
    UnknownElementError,
)
from shipinfer.topology.base import MODEL_KINDS, ChainItem, Element, ElementKind
from shipinfer.topology.caps import Caps, negotiate
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
    #: Explicit kind, when the slot name does not imply it. Usually omitted.
    kind: Optional[ElementKind] = None
    #: Repository model name. Required for the four model kinds, meaningless for the rest.
    model: Optional[str] = None
    #: Branch condition, e.g. ``class == ship``.
    when: Optional[str] = None
    #: Predecessors. **Absent** means "the element declared before me"; an explicit empty
    #: list means "none", which makes this a root and is legal only for ``decode``.
    after: Optional[list[str]] = None
    #: Statefulness hint for the runner: ``camera`` pins this element to the camera's home
    #: shard (arch.md §5⑥). Not interpreted here; the runner reads it.
    per: Optional[Literal["camera", "frame", "global"]] = None
    #: ``global`` marks an element with one instance for the whole fleet (mtmc).
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
            return cls(**data)
        except Exception as exc:  # pydantic ValidationError, or our own ValueError
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
    """

    name: str
    kind: ElementKind
    element: Element
    spec: ElementSpec
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    condition: Condition | None = None

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
        """
        return self.condition is None or self.condition.matches(item.meta)


@dataclass(frozen=True, slots=True)
class Edge:
    """A validated hand-over between two elements, and the cap it carries.

    The negotiated cap is stored because it is a *decision*, not a derivation: it depends on
    both sides' declaration order, and the runner and every log line should read the same
    answer the loader reached.
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
        self._nodes = tuple(nodes)
        self._edges = tuple(edges)
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
            UnknownElementKindError: a slot names no kind.
            ChainStructureError: a model kind with no ``model:``, no decode root, no output
                sink, or a branch that reaches no output.
            UnknownElementImplError: an ``impl:`` nobody registered.
            UnknownElementError: an ``after:`` naming an element that is not declared.
            ChainCycleError: the ``after:`` edges form a cycle.
            CapsMismatchError: two adjacent elements agree on no format/location.
        """
        if not spec.elements:
            raise ChainSpecError(
                f"chain {spec.name or '<unnamed>'} declares no elements; "
                "a topology is a chain of at least a decode and an output"
            )

        kinds = {
            slot: declared.kind if declared.kind is not None else ElementKind.infer(slot)
            for slot, declared in spec.elements.items()
        }

        for slot, declared in spec.elements.items():
            if kinds[slot] in MODEL_KINDS and not declared.model:
                raise ChainStructureError(
                    f"element {slot!r} is a {kinds[slot].value} element and needs "
                    "`model: <repository model name>`"
                )

        elements = {
            slot: create_element(kinds[slot], declared.impl, slot, declared.params)
            for slot, declared in spec.elements.items()
        }

        inputs = _resolve_predecessors(spec)
        order = _topological_order(inputs)
        outputs: dict[str, list[str]] = {slot: [] for slot in spec.elements}
        for consumer, producers in inputs.items():
            for producer in producers:
                outputs[producer].append(consumer)

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
            UnknownElementError: no such slot; the message lists the ones there are.
        """
        try:
            return self._by_name[name]
        except KeyError:
            raise UnknownElementError(
                self._name or "<chain>", name, list(self._by_name)
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


def _topological_order(inputs: Mapping[str, tuple[str, ...]]) -> list[str]:
    """Kahn's algorithm, keeping declaration order among ready elements.

    Declaration order as the tie-break so that two chains that differ only in an irrelevant
    ordering still print and execute identically — a stable order is what makes
    :meth:`Topology.describe` reviewable in a diff.

    Raises:
        ChainCycleError: naming the elements still waiting, which are exactly the cycle and
            whatever hangs off it.
    """
    pending = {slot: len(producers) for slot, producers in inputs.items()}
    consumers: dict[str, list[str]] = {slot: [] for slot in inputs}
    for consumer, producers in inputs.items():
        for producer in producers:
            consumers[producer].append(consumer)

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
        raise ChainCycleError([slot for slot, count in pending.items() if count > 0])
    return order


def _check_structure(nodes: Sequence[ElementNode]) -> None:
    """The four rules that separate a DAG from a runnable chain.

    A chain must *start* somewhere frames come from, and *end* somewhere results go. Both
    are start-up refusals because the failure they prevent is silent: a chain with no output
    sink runs every model at full cost and emits nothing.
    """
    roots = [node for node in nodes if node.is_root]
    if not roots:
        raise ChainStructureError("the chain has no root element")
    for root in roots:
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

    outputs = [node for node in nodes if node.kind is ElementKind.OUTPUT and node.is_sink]
    if not outputs:
        raise ChainStructureError(
            "the chain has no output sink; every chain ends in an element of kind "
            "`output`, or its results go nowhere"
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


def _negotiate_edges(nodes: Sequence[ElementNode]) -> tuple[Edge, ...]:
    """One :class:`Edge` per hand-over, or a refusal naming both sides."""
    by_name = {node.name: node for node in nodes}
    edges: list[Edge] = []
    for node in nodes:
        for consumer_name in node.outputs:
            consumer = by_name[consumer_name]
            caps = negotiate(node.element.output_caps, consumer.element.input_caps)
            if caps is None:
                raise CapsMismatchError(
                    node.name,
                    [str(cap) for cap in node.element.output_caps],
                    consumer.name,
                    [str(cap) for cap in consumer.element.input_caps],
                )
            edges.append(Edge(node.name, consumer_name, caps))
    return tuple(edges)
