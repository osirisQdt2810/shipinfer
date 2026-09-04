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

from shipinfer.core.errors import ConfigurationError

from .base import ElementKind
from .chain import Topology

__all__ = [
    "PLAN_VERSION",
    "PlanNode",
    "PlanSyntaxError",
    "ResolvedPlan",
    "parse_plan",
    "plan_text",
    "resolve_plan",
]

#: Bumped when a verb changes meaning. A reader refuses a version it does not know, because
#: a plan silently half-understood is a chain running something other than what was declared.
PLAN_VERSION = 1

#: Which event field a kind's outputs fill. The event schema's business, stated here so the
#: C++ builder needs no copy of the rule (`pipeline/events/records.h` reads it as a FieldMap).
_EVENT_FIELD = {ElementKind.EMBED: "embedding", ElementKind.SEGMENT: "mask_area_px"}


@dataclass(frozen=True, slots=True)
class PlanNode:
    """One resolved slot: what it is, what it runs on, and the geometry it needs."""

    slot: str
    kind: str
    impl: str
    model: str | None = None
    classes: tuple[str, ...] = ()
    crop: tuple[int, int] | None = None
    letterbox: tuple[int, int] | None = None
    score_threshold: float | None = None
    max_detections: int | None = None
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
    if not spaces and re.search(r"\s", value):
        refused.append("whitespace")
    if "#" in value:
        refused.append("`#`")
    if not commas and "," in value:
        refused.append("`,`")
    if not refused:
        return value
    raise ConfigurationError(
        f"{where}: {value!r} cannot be written to a plan -- {' and '.join(refused)} would "
        f"come back as different text on the other plane. Rename it, or spell it without them"
    )


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
    topology: Topology, *, dims: Mapping[str, tuple[int, int]] | None = None
) -> ResolvedPlan:
    """Flatten a validated chain into a plan.

    Geometry follows ``pool.py``'s rule: the slot's declaration wins, else the model's
    input -- handed in as ``dims`` because reading a ``config.yaml`` is the repository's job.

    Raises:
        ConfigurationError: an extent neither the slot nor ``dims`` states. Refused, not
            defaulted: a crop at the wrong extent is answered with vectors wrong for every
            object on every camera, and nothing reports it.
    """
    dims = dims or {}
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
        decode = params.get("decode") if isinstance(params.get("decode"), Mapping) else {}
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
                classes=tuple(
                    _speakable(label, f"{where}: class", spaces=True, commas=False)
                    for label in node.element.declared_classes() or ()
                ),
                crop=crop,
                letterbox=letterbox,
                score_threshold=_as_float(decode.get("score_threshold"), where),
                max_detections=_as_int(decode.get("max_detections")),
                when=(
                    None
                    if node.condition is None
                    else _speakable(str(node.condition), f"{where}: `when`", spaces=True)
                ),
                per=node.spec.per,
                scope=node.spec.scope,
            )
        )
        if (event_field := _EVENT_FIELD.get(node.kind)) is not None:
            fields.setdefault(event_field, []).append(node.name)

    return ResolvedPlan(
        name=_speakable(topology.name, "chain name", spaces=True),
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
    """Every detector's id-to-label map, merged.

    Read off the chain rather than defaulted, because a raw class index is a property of the
    checkpoint: the demo detector calls a ship **8**, and a plane that assumed 1 labelled
    every ship `unknown` in its events while cropping the right rows.

    EVERY detector, not the first: `_check_declared_classes` unions the tables of all of
    them, so a two-detector chain loads -- and returning the first table here would drop the
    second one's ids, which is the same `unknown` defect one level along.

    Raises:
        ConfigurationError: two detectors give one id different labels, or a table disagrees
            with what the element itself parsed. Refused rather than resolved to whichever
            slot the loader ordered first.
    """
    merged: dict[int, str] = {}
    owner: dict[int, str] = {}
    for node in topology.nodes:
        if node.kind is not ElementKind.DETECT:
            continue
        declared = node.spec.params.get("decode")
        table = declared.get("class_labels") if isinstance(declared, Mapping) else None
        if not isinstance(table, Mapping):
            continue
        labels = {int(key): str(value) for key, value in table.items()}
        # The element already parsed this key and applied its own refusals, so it is the
        # authority on the label VALUES; the ids come from here because nothing exposes them.
        # A disagreement means the two readings have drifted, which is worth a refusal.
        parsed = node.element.detection_labels()
        if parsed is not None and sorted(parsed) != sorted(labels.values()):
            raise ConfigurationError(
                f"detect element {node.name!r} declares class_labels {sorted(labels.values())} "
                f"and its implementation parsed {sorted(parsed)}; the plan cannot say which"
            )
        for index, label in labels.items():
            if index in merged and merged[index] != label:
                raise ConfigurationError(
                    f"class id {index} is {merged[index]!r} in {owner[index]!r} and "
                    f"{label!r} in {node.name!r}; one plan carries one table, and picking "
                    f"one of two silently is how a class is published under the wrong name"
                )
            merged[index] = _speakable(
                label,
                f"detect element {node.name!r}: label {index}",
                spaces=True,
                commas=False,
            )
            owner[index] = node.name
    return merged


def _as_float(value: object, where: str) -> float | None:
    """A finite float, or nothing. YAML spells `.inf`, and the writer would then emit
    `score inf` -- which its own reader refuses, so the plan would be unreadable by both
    planes. Refused here, where the slot can still be named."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(
            f"{where}: `decode.score_threshold` is {value!r}; a plan carries finite numbers "
            f"only, because neither `inf` nor `nan` has a spelling both planes read back"
        )
    return number


def _as_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


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
        if node.classes:
            lines.append("classes " + ",".join(node.classes))
        if node.crop:
            lines.append(f"crop {node.crop[0]} {node.crop[1]}")
        if node.letterbox:
            lines.append(f"letterbox {node.letterbox[0]} {node.letterbox[1]}")
        if node.score_threshold is not None:
            lines.append(f"score {node.score_threshold!r}")
        if node.max_detections is not None:
            lines.append(f"max_detections {node.max_detections}")
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
            fields[args[0]] = tuple(args[1:])
        elif verb in _ATTRIBUTES:
            if not nodes:
                raise PlanSyntaxError(f"{where}: `{verb}` before any `node`")
            _ATTRIBUTES[verb](nodes[-1], args, where)
        else:
            raise PlanSyntaxError(
                f"{where}: unknown verb {verb!r}; expected one of "
                f"{sorted({'plan', 'label', 'node', 'edge', 'field', *_ATTRIBUTES})}"
            )

    if name is None:
        raise PlanSyntaxError(f"{source}: no `plan <version> <name>` header")
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


def _int(text: str, where: str) -> int:
    if not _INTEGER.match(text):
        raise PlanSyntaxError(f"{where}: {text!r} is not an integer")
    return int(text)


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
    if not _NUMBER.match(args[0]):
        raise PlanSyntaxError(f"{where}: {args[0]!r} is not a finite number")
    node["score_threshold"] = float(args[0])


def _max_detections(node: dict[str, object], args: Sequence[str], where: str) -> None:
    _want(args, 1, where, "max_detections <count>")
    node["max_detections"] = _int(args[0], where)


def _classes(node: dict[str, object], args: Sequence[str], where: str) -> None:
    """Comma-delimited, so `classes cargo ship,fishing vessel` is two labels and not four.

    Space-delimited was the first spelling and it re-read `[cargo ship]` as two labels that
    no detector emits -- selecting no rows, running no model and reporting nothing wrong,
    which is the silence `_check_declared_classes` exists to prevent.
    """
    if not args:
        raise PlanSyntaxError(f"{where}: expected `classes <label>[,<label>...]`")
    labels = tuple(part.strip() for part in " ".join(args).split(","))
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
    "when": _when,
    "per": _word_attr("per"),
    "scope": _word_attr("scope"),
}
