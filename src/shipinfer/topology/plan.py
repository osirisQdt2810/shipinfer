"""The resolved chain, as text the C++ data plane can read.

`Topology.from_spec` is the one door a chain becomes trustworthy through (ADR-017), and
ADR-014 says the control plane "hands this plane a resolved configuration". This is that
artefact: the validated chain flattened to a line-oriented plan, so the C++ plane needs no
YAML parser, no element registry and no second copy of the chain rules.

Line-oriented and not JSON for three reasons: `csrc` has a JSON *writer* only, the
cross-plane convention is already lines (`csrc/tests/parity_files.h`), and a plan is a flat
ordered list rather than a tree. One format then serves the transport and the harness.

Pure, like the rest of `topology/`: geometry a chain file does not declare is resolved from
the model repository by the CALLER and handed in as ``dims``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from shipinfer.core.errors import ConfigurationError

from .base import ElementKind
from .chain import ROW_FIELD_KINDS, Topology

__all__ = [
    "PLAN_VERSION",
    "PlanNode",
    "PlanSyntaxError",
    "ResolvedPlan",
    "RuntimeLike",
    "parse_plan",
    "plan_text",
    "resolve_plan",
]

#: Bumped when a verb changes meaning. A reader refuses a version it does not know, because
#: a plan silently half-understood is a chain running something other than what was declared.
PLAN_VERSION = 1

#: Which event field a kind's outputs fill -- `chain.py`'s table, imported rather than
#: repeated: the loader refuses a chain whose two slots could fill one of these for one row,
#: and a second copy would let the two disagree about which kinds those are.


@dataclass(frozen=True, slots=True)
class PlanNode:
    """One resolved slot: what it is, what it runs on, and the geometry it needs."""

    slot: str
    kind: str
    impl: str
    model: str | None = None
    # doc: long the two empty selections mean opposite things, and used to be one spelling
    #: ``None`` is "no selection declared" -- every row. ``()`` is a DECLARED empty selection
    #: -- no rows. `elements/detections.py` names the failure: "conflating the two would make
    #: a typo silently select everything -- at `track` a wrong answer, at an embedder a
    #: doubled GPU bill." So the format spells both: no line, or `classes -`.
    classes: tuple[str, ...] | None = None
    crop: tuple[int, int] | None = None
    letterbox: tuple[int, int] | None = None
    score_threshold: float | None = None
    max_detections: int | None = None
    #: Instances PER DEVICE, the batch window in microseconds, and the artefact relative to
    #: the repository root -- what the C++ bench restated on its command line, where its
    #: numbers disagreed with the repository's on three of four models.
    instances: int | None = None
    queue_delay_us: int | None = None
    artefact: str | None = None
    #: A segmentation fold's two cuts: the score below which a crop reports no area, and the
    #: mask probability at which a cell counts as inside. Both defaulted on both planes, which
    #: is exactly why the plan has to carry them -- an omission agrees by luck today and
    #: diverges the moment a chain file states one.
    fold_score: float | None = None
    fold_mask: float | None = None
    when: str | None = None
    per: str | None = None
    scope: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedPlan:
    """A whole chain, resolved. :func:`plan_text` writes it, :func:`parse_plan` reads it."""

    name: str
    nodes: tuple[PlanNode, ...]
    edges: tuple[tuple[str, str, str], ...] = ()
    labels: Mapping[int, str] = field(default_factory=dict)
    fields: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    version: int = PLAN_VERSION

    def node(self, slot: str) -> PlanNode:
        for node in self.nodes:
            if node.slot == slot:
                return node
        raise KeyError(slot)


# doc: long the delimiters, and why a byte-compare golden cannot catch a value holding one
def _speakable(value: str, where: str, *, spaces: bool = False, commas: bool = True) -> str:
    """Refuse a value the format would reinterpret, naming the slot it came from.

    ``#`` starts a comment on both readers and a comma delimits ``classes``, so a value
    holding one comes back as different text -- which the byte-compare golden cannot catch,
    because the text is stable and only its MEANING changed. Multi-word labels ARE carried
    (COCO's ``traffic light``, a ``cargo ship``): ``plan`` and ``label`` take the rest of the
    line and ``classes`` is comma-delimited.

    In ``resolve_plan`` rather than the writer, because here the slot is still known: the
    operator reads ``embed element 'embed_ship': class 'a,b' ...`` and not a parse error
    from the other plane at deploy time.
    """
    refused = []
    if not value:
        refused.append("no characters")
    elif value != value.strip():
        # A padded value re-reads as its stripped form, so the two planes would publish
        # `"ship "` and `"ship"` for one label -- the `unknown` defect, one step milder.
        refused.append("leading or trailing space")
    # `spaces=True` means MULTI-WORD, not "unvalidated": both readers tokenise with
    # `line.split()`, which collapses runs of whitespace, so a tab, a newline or a double
    # space comes back as different text -- and a newline comes back as an extra LINE, which
    # in scenario 2 of #131's round 3 injected a `node` the chain never declared.
    elif re.search(r"\s\s|[^\S ]" if spaces else r"\s", value):
        refused.append("a tab, newline or repeated space" if spaces else "whitespace")
    if "#" in value:
        refused.append("`#`")
    if not commas and "," in value:
        refused.append("`,`")
    if not refused:
        return value
    raise ConfigurationError(
        f"{where}: {value!r} cannot be written to a plan -- it holds "
        f"{' and '.join(refused)}, which comes back as different text on the other plane "
        f"(or as no text at all). Rename it, or spell it without them"
    )


class RuntimeLike(Protocol):
    """What :func:`resolve_plan` reads off a ``repository.resolved.ModelRuntime``.

    Structural, like ``DecodeParamsLike`` in ``base.py`` and for the same reason: ``topology``
    is pure and may not import ``repository``, so the caller passes the objects and this
    layer states only which attributes it needs.
    """

    #: ``None`` where the model asks for no DEVICE instance at all -- a CPU-only model,
    #: which `topology/ship_person_cpu.yaml`'s own header documents as a supported target.
    #: The plan then carries no `instances` line and the consumer refuses by name.
    instances: int | None
    queue_delay_us: int
    artefact: str


def _runtime_int(
    runtime: RuntimeLike | None, field_name: str, where: str, *, zero_ok: bool = False
) -> int | None:
    """One runtime number, refused here rather than written into a plan the other plane
    refuses. ``instances`` is a count and must be positive; a ``queue_delay_us`` of 0 is
    "no window", which is what dynamic batching being off resolves to."""
    if runtime is None or (raw := getattr(runtime, field_name)) is None:
        return None
    value = int(raw)
    if value < 0 or (value == 0 and not zero_ok):
        raise ConfigurationError(
            f"{where}: {field_name} is {value}, and a plan carries a "
            f"{'non-negative' if zero_ok else 'positive'} one"
        )
    return value


def _extent(value: object, where: str) -> tuple[int, int]:
    """``[h, w]``, both positive. Refused rather than defaulted, as ``pool.py`` refuses it."""
    pair = tuple(value) if isinstance(value, (list, tuple)) else ()
    if len(pair) != 2 or not all(isinstance(n, int) and n > 0 for n in pair):
        raise ConfigurationError(f"{where} must be two positive integers, got {value!r}")
    return int(pair[0]), int(pair[1])


def _geometry(
    node_params: Mapping[str, object], key: str, where: str
) -> tuple[int, int] | None:
    """A declared ``crop.size`` / ``decode.dst_size``, or ``None`` when the file is silent."""
    section = node_params.get(key)
    if not isinstance(section, Mapping):
        return None
    declared = section.get("size" if key == "crop" else "dst_size")
    return None if declared is None else _extent(declared, f"{where}: `{key}`")


def resolve_plan(
    topology: Topology,
    *,
    dims: Mapping[str, tuple[int, int]] | None = None,
    runtimes: Mapping[str, RuntimeLike] | None = None,
) -> ResolvedPlan:
    """Flatten a validated chain into a plan.

    Geometry follows ``pool.py``'s rule: the slot's declaration wins, else the model's input,
    handed in as ``dims`` because a ``config.yaml`` is the repository's to read. ``runtimes``
    is the rest of that reading (``repository/resolved.py``) and is optional.

    Raises:
        ConfigurationError: an extent neither the slot nor ``dims`` states -- refused, since
            a crop at the wrong extent answers with vectors wrong for every object silently.
    """
    dims = dims or {}
    runtimes = runtimes or {}
    nodes: list[PlanNode] = []
    fields: dict[str, list[str]] = {}
    for node in topology.nodes:
        where = f"{node.kind.value} element {node.name!r}"
        params = node.spec.params
        crop = _geometry(params, "crop", where)
        # `decode.dst_size` is the DETECTOR's letterbox target. Read for every kind, an
        # `embed` slot carrying the key would get one -- harmless today, and the plan is
        # where it stops being obvious.
        letterbox = (
            _geometry(params, "decode", where) if node.kind is ElementKind.DETECT else None
        )
        # EFFECTIVE, not declared: a detect slot that writes no `score_threshold` still
        # decodes at one, and a plan that said nothing made the other plane default -- which
        # is the literal this whole seam exists to remove.
        decode = node.element.decode_parameters()
        runtime = runtimes.get(node.spec.model or "") if node.spec.model else None
        fold = node.element.fold_parameters()
        if node.kind in (ElementKind.EMBED, ElementKind.SEGMENT) and crop is None:
            crop = _model_extent(node.spec.model, dims, where, "crop")
        if node.kind is ElementKind.DETECT and letterbox is None:
            letterbox = _model_extent(node.spec.model, dims, where, "letterbox")
        nodes.append(
            PlanNode(
                slot=_speakable(node.name, where),
                kind=node.kind.value,
                impl=_speakable(node.element.impl, where),
                model=_speakable(node.spec.model, where) if node.spec.model else None,
                classes=_classes_of(node.element.declared_classes(), where),
                crop=crop,
                letterbox=letterbox,
                score_threshold=(
                    None if decode is None else _finite(decode.score_threshold, where)
                ),
                max_detections=(
                    None if decode is None else _positive(decode.max_detections, where)
                ),
                fold_score=(
                    None
                    if fold is None
                    else _finite(fold.score_threshold, where, "segment.score_threshold")
                ),
                fold_mask=(None if fold is None else _probability(fold.mask_threshold, where)),
                instances=_runtime_int(runtime, "instances", where),
                queue_delay_us=_runtime_int(runtime, "queue_delay_us", where, zero_ok=True),
                artefact=(
                    None
                    if runtime is None
                    else _speakable(str(runtime.artefact), f"{where}: artefact")
                ),
                when=(
                    None
                    if node.condition is None
                    else _speakable(str(node.condition), f"{where}: `when`", spaces=True)
                ),
                per=node.spec.per,
                scope=node.spec.scope,
            )
        )
        if (event_field := ROW_FIELD_KINDS.get(node.kind)) is not None:
            fields.setdefault(event_field, []).append(node.name)

    return ResolvedPlan(
        name=_chain_name(topology.name),
        nodes=tuple(nodes),
        edges=tuple(
            (edge.producer, edge.consumer, _speakable(str(edge.caps), "edge cap"))
            for edge in topology.edges
        ),
        labels=_labels(topology),
        fields={name: tuple(slots) for name, slots in fields.items()},
    )


def _model_extent(
    model: str | None, dims: Mapping[str, tuple[int, int]], where: str, what: str
) -> tuple[int, int]:
    """The model's declared input extent, or the refusal ``pool.py`` would raise at open."""
    extent = dims.get(model or "")
    if extent is None:
        raise ConfigurationError(
            f"{where} cannot tell how big a {what} model {model!r} wants: the slot declares "
            f"no extent and the repository says nothing usable. Give the model a "
            f"`config.yaml` that declares its input, or say so on the slot"
        )
    return _extent(list(extent), f"{where}: repository dims for {model!r}")


# doc: long two detectors are a supported chain shape, and the ids are the checkpoint's
def _labels(topology: Topology) -> dict[int, str]:
    """Every detector's EFFECTIVE id-to-label table, merged.

    Asked of the element (``decode_parameters``) and not re-read from ``params``, which is
    the difference between what a chain file wrote and what the chain will do: a slot that
    declares no table still decodes with one, and a plan that omitted it made the other plane
    invent its own -- the hard-coded table ADR-020 exists to delete.

    EVERY detector, not the first: `_check_declared_classes` unions the tables of all of
    them, so a two-detector chain loads, and returning the first would drop the second's ids.

    Raises:
        ConfigurationError: two detectors give one id different labels. Refused rather than
            resolved to whichever slot the loader ordered first.
    """
    merged: dict[int, str] = {}
    owner: dict[int, str] = {}
    for node in topology.nodes:
        decode = node.element.decode_parameters()
        if decode is None:
            continue
        for index, label in decode.class_labels.items():
            speakable = _speakable(
                str(label),
                f"{node.kind.value} element {node.name!r}: label {index}",
                spaces=True,
                commas=False,
            )
            if index in merged and merged[index] != speakable:
                raise ConfigurationError(
                    f"class id {index} is {merged[index]!r} in {owner[index]!r} and "
                    f"{speakable!r} in {node.name!r}; one plan carries one table, and picking "
                    f"one of two silently is how a class is published under the wrong name"
                )
            merged[int(index)] = speakable
            owner[int(index)] = node.name
    return merged


def _chain_name(name: str) -> str:
    """The chain's name, or ``""``. `-` is the sentinel the writer uses for an empty name, so
    it cannot also BE a name -- a chain called `-` would come back unnamed."""
    if name == "-":
        raise ConfigurationError(
            "chain name: `-` is the plan's spelling for an unnamed chain, so a chain cannot "
            "be called that; it would come back with no name at all"
        )
    return _speakable(name, "chain name", spaces=True) if name else ""


def _classes_of(declared: tuple[str, ...] | None, where: str) -> tuple[str, ...] | None:
    """The declared row selection, kept distinct from "no selection at all"."""
    if declared is None:
        return None
    for label in declared:
        if label == "-":
            raise ConfigurationError(
                f"{where}: `-` is the plan's spelling for a DECLARED EMPTY selection, so a "
                f"class cannot be called that; it would come back as 'no rows at all'"
            )
    return tuple(
        _speakable(label, f"{where}: class", spaces=True, commas=False) for label in declared
    )


def _probability(value: float, where: str) -> float:
    """Strictly inside (0, 1) -- the range `InstanceMaskArea` refuses outside, refused here so
    a plan is never written holding a value the other plane's `log(m / (1 - m))` cannot take."""
    number = float(value)
    if not 0.0 < number < 1.0:
        raise ConfigurationError(
            f"{where}: a mask probability is strictly inside (0, 1), got {number!r}"
        )
    return number


def _positive(value: int, where: str) -> int:
    """A cap is a positive count. Nothing between a chain file and the decode loop refused a
    `-1` -- not `_resolve_decode_params`'s bare `int()`, not the unvalidated `DecodeParams`
    (the `ge=1` lives on the settings tree, which a pure element does not read) -- and the two
    planes then disagreed about how many rows to keep."""
    count = int(value)
    if count <= 0:
        raise ConfigurationError(
            f"{where}: `decode.max_detections` is {count}; a cap is a positive count, and "
            f"`-1` for `no limit` is not a spelling either plane has"
        )
    return count


def _finite(value: float, where: str, key: str = "decode.score_threshold") -> float:
    """A finite threshold. YAML spells `.inf`, and the writer would then emit `score inf` --
    which its own reader refuses, so the plan would be unreadable by both planes.

    `key` because two blocks now reach this: a segment slot has no `decode:` block at all, and
    a message naming one sends the operator to a key they never wrote.
    """
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(
            f"{where}: `{key}` is {value!r}; a plan carries finite numbers only, because "
            f"neither `inf` nor `nan` has a spelling both planes read back"
        )
    return number


def plan_text(plan: ResolvedPlan) -> str:
    """The plan as the text both planes read. Deterministic, and the golden's own format."""
    lines = [
        "# A RESOLVED chain, written by `shipinfer plan` -- do not hand-edit.",
        f"plan {plan.version} {plan.name or '-'}",
    ]
    lines += [f"label {index} {name}" for index, name in sorted(plan.labels.items())]
    for node in plan.nodes:
        lines += ["", f"node {node.slot} {node.kind} {node.impl}"]
        if node.model:
            lines.append(f"model {node.model}")
        if node.classes is not None:
            # `-` for a declared-empty selection, which is a slot that selects NO rows. An
            # absent line is the other thing: no selection declared, so every row.
            lines.append("classes " + (",".join(node.classes) or "-"))
        if node.crop:
            lines.append(f"crop {node.crop[0]} {node.crop[1]}")
        if node.letterbox:
            lines.append(f"letterbox {node.letterbox[0]} {node.letterbox[1]}")
        if node.score_threshold is not None:
            lines.append(f"score {node.score_threshold!r}")
        if node.max_detections is not None:
            lines.append(f"max_detections {node.max_detections}")
        if node.instances is not None:
            lines.append(f"instances {node.instances}")
        if node.queue_delay_us is not None:
            lines.append(f"queue_delay_us {node.queue_delay_us}")
        if node.artefact:
            lines.append(f"artefact {node.artefact}")
        if node.fold_score is not None:
            lines.append(f"fold_score {node.fold_score!r}")
        if node.fold_mask is not None:
            lines.append(f"fold_mask {node.fold_mask!r}")
        if node.when:
            lines.append(f"when {node.when}")
        if node.per:
            lines.append(f"per {node.per}")
        if node.scope:
            lines.append(f"scope {node.scope}")
    lines.append("")
    lines += [f"edge {producer} {consumer} {caps}" for producer, consumer, caps in plan.edges]
    lines += [f"field {name} " + " ".join(slots) for name, slots in sorted(plan.fields.items())]
    return "\n".join(lines) + "\n"


class PlanSyntaxError(ConfigurationError):
    """A plan line the reader cannot honour, named by line number.

    Its own type because the failure is a *file*, not a chain: the C++ half raises the same
    way, and a plan one plane reads and the other refuses is the worst outcome here.
    """


def parse_plan(text: str, *, source: str = "<string>") -> ResolvedPlan:
    """Read a plan back. The inverse of :func:`plan_text`, byte for byte.

    Raises:
        PlanSyntaxError: an unknown verb, a bad argument count, an attribute before any
            ``node``, a missing or repeated ``plan`` header, or an unknown version.
    """
    name: str | None = None
    version = 0
    labels: dict[int, str] = {}
    nodes: list[dict[str, object]] = []
    edges: list[tuple[str, str, str]] = []
    fields: dict[str, tuple[str, ...]] = {}

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        verb, *args = line.split()
        where = f"{source}:{number}"
        if verb == "plan":
            if name is not None:
                raise PlanSyntaxError(f"{where}: a second `plan` header")
            version, name = _header(line, args, where)
        elif name is None:
            raise PlanSyntaxError(f"{where}: `{verb}` before the `plan` header")
        elif verb == "label":
            if len(args) < 2:
                raise PlanSyntaxError(f"{where}: expected `label <id> <name>`")
            index = _int(args[0], where)
            if index in labels:
                raise PlanSyntaxError(f"{where}: a second `label {index}`")
            labels[index] = line.split(maxsplit=2)[2]
        elif verb == "node":
            _want(args, 3, where, "node <slot> <kind> <impl>")
            # A repeat is refused rather than appended or last-wins: two blocks for one slot
            # is a plan whose meaning depends on which reader you ask, and the C++ table
            # carries the same row.
            if any(node["slot"] == args[0] for node in nodes):
                raise PlanSyntaxError(f"{where}: a second `node {args[0]}`")
            nodes.append({"slot": args[0], "kind": args[1], "impl": args[2]})
        elif verb == "edge":
            _want(args, 3, where, "edge <producer> <consumer> <format@location>")
            edges.append((args[0], args[1], args[2]))
        elif verb == "field":
            if len(args) < 2:
                raise PlanSyntaxError(f"{where}: expected `field <name> <slot>...`")
            if args[0] in fields:
                raise PlanSyntaxError(f"{where}: a second `field {args[0]}`")
            fields[args[0]] = tuple(args[1:])
        elif verb in _ATTRIBUTES:
            if not nodes:
                # Worded as the C++ reader words it: over there one branch answers both
                # "unknown verb" and "attribute before any node", so a message that named
                # only one of them drifted from its twin.
                raise PlanSyntaxError(
                    f"{where}: unknown verb {verb!r}, or an attribute before any `node`"
                )
            _ATTRIBUTES[verb](nodes[-1], args, where)
        else:
            raise PlanSyntaxError(
                f"{where}: unknown verb {verb!r}; expected one of "
                f"{sorted({'plan', 'label', 'node', 'edge', 'field', *_ATTRIBUTES})}"
            )

    if name is None:
        raise PlanSyntaxError(f"{source}: no `plan <version> <name>` header")
    # A `field` may only name a slot some `node` declared. Left unchecked, the C++ half
    # dereferenced a null `PlanNode*` before any worker started and this half raised a bare
    # `KeyError` from `ResolvedPlan.node` -- two different failures for one malformed plan.
    declared = {node["slot"] for node in nodes}
    for field_name, slots in fields.items():
        for slot in slots:
            if slot not in declared:
                raise PlanSyntaxError(
                    f"{source}: field {field_name!r} names slot {slot!r}, which no `node` "
                    f"declares"
                )
    return ResolvedPlan(
        name="" if name == "-" else name,
        nodes=tuple(PlanNode(**node) for node in nodes),  # type: ignore[arg-type]
        edges=tuple(edges),
        labels=labels,
        fields=fields,
        version=version,
    )


def _header(line: str, args: Sequence[str], where: str) -> tuple[int, str]:
    """``plan <version> <name>``, where the name is the REST of the line.

    Free text, because a chain may be called `ship person cpu` and a fixed arity would then
    refuse a plan `shipinfer plan` had just written -- one of three such holes the review of
    this format found. `label` reads its name the same way.
    """
    if len(args) < 2:
        raise PlanSyntaxError(f"{where}: expected `plan <version> <name>`")
    version = _int(args[0], where)
    if version != PLAN_VERSION:
        raise PlanSyntaxError(
            f"{where}: plan version {version}, and this reader knows {PLAN_VERSION}. "
            "A plan half understood is a chain running something other than what was declared"
        )
    return version, line.split(maxsplit=2)[2]


def _want(args: Sequence[str], count: int, where: str, form: str) -> None:
    if len(args) != count:
        raise PlanSyntaxError(f"{where}: expected `{form}`, got {len(args)} argument(s)")


#: Digits and an optional sign, and nothing else. `int()` alone accepts `1_000` and
#: surrounding space, which `std::stoi` refuses -- a reader divergence on a shared format is
#: the one failure this seam exists to prevent, however odd the input that triggers it.
_INTEGER = re.compile(r"^[+-]?[0-9]+$")
#: Likewise `float()` accepts `nan` and `inf`, and a plan carrying either cannot be written
#: back out: `json_number` refuses a non-finite double on both planes.
_NUMBER = re.compile(r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")


#: What an attribute verb does: fold one line's arguments into the node being built.
_Attribute = Callable[[dict, Sequence[str], str], None]


#: What a C++ `int` holds. Python's `int` is unbounded, so `label 99999999999999999999`
#: loaded here and was refused by `std::stoi` on the other plane -- and a plan carries class
#: ids and pixel extents, not arbitrary integers.
_INT_MAX = 2**31 - 1


def _int(text: str, where: str) -> int:
    if not _INTEGER.match(text):
        raise PlanSyntaxError(f"{where}: {text!r} is not an integer")
    value = int(text)
    if not -_INT_MAX - 1 <= value <= _INT_MAX:
        raise PlanSyntaxError(
            f"{where}: {text!r} does not fit in a 32-bit int, and a plan carries class ids "
            f"and pixel extents rather than arbitrary integers"
        )
    return value


def _extent_attr(key: str) -> _Attribute:
    def apply(node: dict[str, object], args: Sequence[str], where: str) -> None:
        _want(args, 2, where, f"{key} <height> <width>")
        extent = (_int(args[0], where), _int(args[1], where))
        # Positivity here and not only in `resolve_plan`: the C++ reader refuses it, and a
        # `crop 0 128` that loads on one plane and not the other is exactly the divergence
        # this format's two readers exist to avoid. Found by the shared refusal table.
        if extent[0] <= 0 or extent[1] <= 0:
            raise PlanSyntaxError(f"{where}: {key} must be two positive integers")
        node[key] = extent

    return apply


def _word_attr(key: str) -> _Attribute:
    def apply(node: dict[str, object], args: Sequence[str], where: str) -> None:
        _want(args, 1, where, f"{key} <value>")
        node[key] = args[0]

    return apply


def _score(node: dict[str, object], args: Sequence[str], where: str) -> None:
    _want(args, 1, where, "score <threshold>")
    # `_NUMBER` matches `1e400`, which `float()` reads as `inf` -- and an infinite threshold
    # passes no detection and reports nothing wrong, while the C++ reader refuses it on
    # `isfinite`. The regex says the SHAPE; this says the value.
    if not _NUMBER.match(args[0]) or not math.isfinite(float(args[0])):
        raise PlanSyntaxError(f"{where}: {args[0]!r} is not a finite number")
    node["score_threshold"] = float(args[0])


def _instances(node: dict[str, object], args: Sequence[str], where: str) -> None:
    """Instances per device: a positive count, because zero would load the engine and run
    nothing while every stage reported ready."""
    _want(args, 1, where, "instances <count>")
    count = _int(args[0], where)
    if count <= 0:
        raise PlanSyntaxError(f"{where}: instances is {count}; a slot runs at least one")
    node["instances"] = count


def _queue_delay(node: dict[str, object], args: Sequence[str], where: str) -> None:
    """The batch window in microseconds. `0` is legal and means no window at all, which is
    what `dynamic_batching: {enabled: false}` resolves to."""
    _want(args, 1, where, "queue_delay_us <microseconds>")
    delay = _int(args[0], where)
    if delay < 0:
        raise PlanSyntaxError(f"{where}: queue_delay_us is {delay}; a window is not negative")
    node["queue_delay_us"] = delay


def _fold_score(node: dict[str, object], args: Sequence[str], where: str) -> None:
    """The score floor. Finite, like `score` -- an infinite one passes no crop and reports
    nothing wrong."""
    _want(args, 1, where, "fold_score <threshold>")
    if not _NUMBER.match(args[0]) or not math.isfinite(float(args[0])):
        raise PlanSyntaxError(f"{where}: {args[0]!r} is not a finite number")
    node["fold_score"] = float(args[0])


def _fold_mask(node: dict[str, object], args: Sequence[str], where: str) -> None:
    """The mask probability, strictly inside (0, 1).

    `InstanceMaskArea.__post_init__` refuses the endpoints and the C++ fold takes
    `log(m / (1 - m))`, which is `-inf` at 0 and a division by zero at 1 -- so a plan carrying
    either is a chain that loads on neither plane, and the reader says so rather than the
    arithmetic.
    """
    _want(args, 1, where, "fold_mask <probability>")
    if not _NUMBER.match(args[0]):
        raise PlanSyntaxError(f"{where}: {args[0]!r} is not a number")
    value = float(args[0])
    if not 0.0 < value < 1.0:
        raise PlanSyntaxError(
            f"{where}: fold_mask is {value}; a mask probability is strictly inside (0, 1), "
            f"because the cut is log(m / (1 - m))"
        )
    node["fold_mask"] = value


def _max_detections(node: dict[str, object], args: Sequence[str], where: str) -> None:
    """A positive count. `-1` for "no limit" is a widespread convention and neither plane has
    it: here `keep[: -1]` would drop one row, and on the C++ side
    `static_cast<size_t>(-1)` is no bound at all -- one plan, two detection sets."""
    _want(args, 1, where, "max_detections <count>")
    count = _int(args[0], where)
    if count <= 0:
        raise PlanSyntaxError(
            f"{where}: max_detections is {count}; a cap is a positive count on both planes"
        )
    node["max_detections"] = count


def _classes(node: dict[str, object], args: Sequence[str], where: str) -> None:
    """Comma-delimited, so `classes cargo ship,fishing vessel` is two labels and not four.

    Space-delimited was the first spelling and it re-read `[cargo ship]` as two labels that
    no detector emits -- selecting no rows, running no model and reporting nothing wrong,
    which is the silence `_check_declared_classes` exists to prevent.
    """
    if not args:
        raise PlanSyntaxError(
            f"{where}: expected `classes <label>[,<label>...]`, or `classes -` for none"
        )
    joined = " ".join(args)
    if joined == "-":
        node["classes"] = ()
        return
    labels = tuple(part.strip() for part in joined.split(","))
    if not all(labels):
        raise PlanSyntaxError(f"{where}: an empty label in `classes`")
    node["classes"] = labels


def _when(node: dict[str, object], args: Sequence[str], where: str) -> None:
    if not args:
        raise PlanSyntaxError(f"{where}: expected `when <expression>`")
    node["when"] = " ".join(args)


#: Verbs that attach to the `node` block above them, the way `capacity` attaches to `queue`.
_ATTRIBUTES = {
    "model": _word_attr("model"),
    "classes": _classes,
    "crop": _extent_attr("crop"),
    "letterbox": _extent_attr("letterbox"),
    "score": _score,
    "max_detections": _max_detections,
    "fold_score": _fold_score,
    "fold_mask": _fold_mask,
    "instances": _instances,
    "queue_delay_us": _queue_delay,
    "artefact": _word_attr("artefact"),
    "when": _when,
    "per": _word_attr("per"),
    "scope": _word_attr("scope"),
}
