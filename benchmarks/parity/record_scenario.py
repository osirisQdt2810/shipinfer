"""One frame's STAGE OUTPUTS, so both planes' record builders run on the same inputs.

The event seam (`event_scenario.py`) states finished `ObjectRecord`s, so it compares the two
JSON writers and never the two builders -- which is P5-A-ALLOC's second half.

So a scenario here describes what the graph LEAVES BEHIND: the detections, the per-object
batches with their row indices, the label table, and the field map. Each plane builds its own
records and writes its own event, and the gate compares bytes -- except for a scenario
describing a frame both planes must REFUSE, which has no golden at all.

Numbers are restricted to :data:`~benchmarks.parity.event_scenario.FLOATS` for that seam's
reason: the comparison is a byte compare, so a value the two planes spell differently would
fail on formatting rather than on the builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shipinfer.core.errors import ConfigurationError

from .event_scenario import FLOATS, reason_of

__all__ = ["BatchSpec", "DetectionSpec", "RecordScenario", "load_record_scenario"]


@dataclass(frozen=True, slots=True)
class DetectionSpec:
    """One detection row, by CLASS ID -- so each plane resolves the label itself."""

    index: int
    class_id: int
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class BatchSpec:
    """One per-object batch: ``width`` floats per row, each row owning a detection index."""

    name: str
    width: int
    indices: tuple[int, ...] = ()
    rows: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class RecordScenario:
    """A frame's emission inputs, plus the two tables the builder is given."""

    name: str
    camera: str = "cam0"
    frame: int = 0
    source: str = "shard-0"
    width: int = 640
    height: int = 480
    fps: float = 1.0
    captured_ns: int = 0
    captured_unix_ns: int = 0
    emitted_unix_ns: int = 0
    missing: tuple[str, ...] = ()
    reason: str = "complete"
    labels: dict[int, str] = field(default_factory=dict)
    detections: tuple[DetectionSpec, ...] = ()
    batches: tuple[BatchSpec, ...] = ()
    # doc: long the order is declaration order, and what it does NOT decide
    #: ``ObjectRecord`` field -> the batch names that can fill it, in declared order. A
    #: COVERAGE UNION: a row two candidates cover is REFUSED on both planes, so the order
    #: does not pick a winner -- do not write a scenario that leans on one.
    #:
    #: It does decide which field a *message* names when two fields collide in one frame:
    #: Python iterates insertion order, the C++ plane a `std::map` sorted by field name. Both
    #: refuse, so the operational outcome agrees and no gate can see the difference.
    fields: tuple[tuple[str, tuple[str, ...]], ...] = ()


def _number(word: str, where: str) -> float:
    try:
        value = float(word)
    except ValueError:
        raise ConfigurationError(f"{where}: {word!r} is not a number") from None
    if value not in FLOATS:
        raise ConfigurationError(
            f"{where}: {word!r} is not one of the values both planes spell identically; "
            f"add it to `event_scenario.FLOATS` only after checking `repr` against the C++ "
            f"writer, because this gate is a byte compare"
        )
    return value


def _int(word: str, where: str) -> int:
    try:
        return int(word)
    except ValueError:
        raise ConfigurationError(f"{where}: {word!r} is not an integer") from None


def load_record_scenario(path: Path) -> RecordScenario:
    """Parse one scenario, naming the line of any refusal."""
    values: dict[str, object] = {}
    labels: dict[int, str] = {}
    detections: list[DetectionSpec] = []
    batches: list[BatchSpec] = []
    indices: list[list[int]] = []
    rows: list[list[tuple[float, ...]]] = []
    fields: list[tuple[str, tuple[str, ...]]] = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        where = f"{path}:{number}"
        directive, *words = line.split()
        if directive == "scenario":
            values["name"] = words[0]
        elif directive in ("camera", "source", "reason"):
            values[directive] = words[0]
        elif directive == "finished":
            values["reason"] = reason_of(words[0], where)
        elif directive == "frame":
            values["frame"] = _int(words[0], where)
        elif directive == "size":
            values["width"], values["height"] = _int(words[0], where), _int(words[1], where)
        elif directive == "fps":
            values["fps"] = _number(words[0], where)
        elif directive in ("captured_ns", "captured_unix_ns", "emitted_unix_ns"):
            values[directive] = _int(words[0], where)
        elif directive == "missing":
            values["missing"] = tuple(words)
        elif directive == "label":
            labels[_int(words[0], where)] = words[1]
        elif directive == "det":
            if len(words) != 7:
                raise ConfigurationError(
                    f"{where}: expected `det <index> <class_id> <score> <x1> <y1> <x2> <y2>`"
                )
            # The index must BE the position: `Detections` numbers its rows by iteration
            # order here, while the C++ loader stores what the line says -- so `det 5` first
            # would give `<cam>_<frame>_5` there and `..._0` here, and the gate would call
            # that a builder divergence.
            index = _int(words[0], where)
            if index != len(detections):
                raise ConfigurationError(
                    f"{where}: `det {index}` is the {len(detections)}th detection; an index "
                    f"is its position, because the Python plane cannot express anything else"
                )
            detections.append(
                DetectionSpec(
                    index=index,
                    class_id=_int(words[1], where),
                    score=_number(words[2], where),
                    box=(
                        _number(words[3], where),
                        _number(words[4], where),
                        _number(words[5], where),
                        _number(words[6], where),
                    ),
                )
            )
        elif directive == "batch":
            if len(words) != 2:
                raise ConfigurationError(f"{where}: expected `batch <name> <width>`")
            batches.append(BatchSpec(name=words[0], width=_int(words[1], where)))
            indices.append([])
            rows.append([])
        elif directive == "row":
            if not batches:
                raise ConfigurationError(f"{where}: `row` before any `batch`")
            width = batches[-1].width
            if len(words) != width + 1:
                raise ConfigurationError(
                    f"{where}: batch {batches[-1].name!r} is {width} wide, so a row is "
                    f"`row <index>` plus {width} value(s); got {len(words) - 1}"
                )
            indices[-1].append(_int(words[0], where))
            rows[-1].append(tuple(_number(word, where) for word in words[1:]))
        elif directive == "field":
            if len(words) < 2:
                raise ConfigurationError(f"{where}: expected `field <name> <batch>...`")
            fields.append((words[0], tuple(words[1:])))
        else:
            raise ConfigurationError(f"{where}: unknown directive {directive!r}")

    if "name" not in values:
        raise ConfigurationError(f"{path}: no `scenario <name>` line")
    return RecordScenario(
        labels=labels,
        detections=tuple(detections),
        batches=tuple(
            BatchSpec(spec.name, spec.width, tuple(index), tuple(row))
            for spec, index, row in zip(batches, indices, rows, strict=True)
        ),
        fields=tuple(fields),
        **values,  # type: ignore[arg-type]
    )
