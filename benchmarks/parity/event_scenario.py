"""One perception event's inputs, as a file both planes parse.

Line-oriented so the C++ half needs no JSON parser -- the rule `scenario.py` and
`queue_scenario.py` follow. An event is a *value*, so a scenario is its arguments::

    scenario mixed_frame   # then: camera, frame, source, size, fps, captured_ns,
    camera cam0            #   captured_unix_ns, emitted_unix_ns, missing, reason
    person <det_id> <score> <x1> <y1> <x2> <y2> [track <id> <state>] [global <id>] [emb v...]
    ship   <det_id> <score> <x1> <y1> <x2> <y2> [ship <id> <similarity>] [mask <area>]

Only floats both planes spell identically are allowed (:data:`FLOATS`): the gate is a byte
compare, so any other value would fail on formatting rather than on the schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shipinfer.core.errors import ConfigurationError

__all__ = ["EventScenario", "ObjectSpec", "load_event_scenario"]

#: Every numeric literal a scenario may use, integers included: the gate compares bytes, so a
#: value the two planes spell differently would fail on formatting and not on the schema. An
#: earlier version let any INTEGRAL float through, so `mask 100000` loaded and then failed the
#: byte gate at a column -- exactly what this list exists to prevent.
FLOATS = frozenset(
    {
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        0.25,
        0.5,
        0.75,
        0.875,
        10.0,
        12.5,
        19.6,
        20.0,
        30.0,
        40.0,
        100.0,
        200.0,
        300.0,
        400.0,
        480.0,
        640.0,
    }
)


@dataclass(frozen=True, slots=True)
class ObjectSpec:
    """One object's fields, before it becomes an ``ObjectRecord``."""

    class_name: str
    det_id: str
    score: float
    bbox: tuple[float, float, float, float]
    embedding: tuple[float, ...] = ()
    ship_id: int | None = None
    similarity: float | None = None
    mask_area_px: float | None = None
    track_id: int | None = None
    track_state: str | None = None
    global_id: int | None = None


@dataclass(frozen=True, slots=True)
class EventScenario:
    """A whole event's inputs, and the objects it carries."""

    name: str
    camera: str = "cam0"
    frame: int = 0
    source: str = "shard-0"
    width: int = 0
    height: int = 0
    fps: float = 0.0
    captured_ns: int = 0
    captured_unix_ns: int = 0
    emitted_unix_ns: int = 0
    missing: tuple[str, ...] = ()
    reason: str = "complete"
    objects: tuple[ObjectSpec, ...] = field(default_factory=tuple)


def _number(word: str, where: str) -> float:
    """One numeric literal, refused unless :data:`FLOATS` lists it.

    No escape hatch for integral values: `100000.0` is integral and the two planes spell it
    differently (`1e+05` against `100000.0` before the writer was fixed), so the hatch let
    through exactly what the list exists to catch. `float()` also accepts `nan` and `inf`,
    which have no valid JSON spelling at all -- both land here rather than in an untyped
    `ValueError` from `int()`.
    """
    try:
        value = float(word)
    except ValueError as exc:
        raise ConfigurationError(f"{where}: {word!r} is not a number") from exc
    if value not in FLOATS:
        raise ConfigurationError(
            f"{where}: {value!r} is not in FLOATS, and the two planes may spell it "
            f"differently -- the gate compares bytes. Add it to FLOATS with a test first"
        )
    return value


#: The collector's `FinishReason` vocabulary, which IS the wire vocabulary: `runner.py`
#: passes `result.reason` through verbatim. Spelled here so a scenario can name the outcome
#: and let each plane derive the word -- the only way this gate can catch a plane that
#: invents a different one.
FINISH_REASONS = ("complete", "incomplete", "timeout", "shutdown", "evicted")


def _reason_of(word: str, where: str) -> str:
    if word not in FINISH_REASONS:
        raise ConfigurationError(
            f"{where}: {word!r} is not a FinishReason; known: {list(FINISH_REASONS)}"
        )
    return word


def _object(class_name: str, words: list[str], where: str) -> ObjectSpec:
    """``<det_id> <score> <x1> <y1> <x2> <y2>`` then any of the keyword groups."""
    if len(words) < 6:
        raise ConfigurationError(f"{where}: {class_name} needs a det_id, a score and 4 bounds")
    fields: dict[str, object] = {
        "class_name": class_name,
        "det_id": words[0],
        "score": _number(words[1], where),
        "bbox": tuple(_number(w, where) for w in words[2:6]),
    }
    rest = words[6:]
    while rest:
        key, rest = rest[0], rest[1:]
        if key == "emb":
            fields["embedding"] = tuple(_number(w, where) for w in rest)
            rest = []
        elif key == "track":
            fields["track_id"], fields["track_state"] = int(rest[0]), rest[1]
            rest = rest[2:]
        elif key == "global":
            fields["global_id"], rest = int(rest[0]), rest[1:]
        elif key == "ship":
            fields["ship_id"] = int(rest[0])
            fields["similarity"] = _number(rest[1], where)
            rest = rest[2:]
        elif key == "mask":
            fields["mask_area_px"] = _number(rest[0], where)
            rest = rest[1:]
        else:
            raise ConfigurationError(f"{where}: unknown object keyword {key!r}")
    return ObjectSpec(**fields)  # type: ignore[arg-type]


def load_event_scenario(path: Path) -> EventScenario:
    """Parse one scenario, naming the line of any refusal."""
    fields: dict[str, object] = {}
    objects: list[ObjectSpec] = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        where = f"{path}:{number}"
        directive, *words = line.split()
        if directive in ("person", "ship"):
            objects.append(_object(directive, words, where))
        elif directive == "scenario":
            fields["name"] = words[0]
        elif directive in ("camera", "source"):
            fields[directive] = words[0]
        elif directive == "reason":
            # The word, for a scenario that wants one directly.
            fields["reason"] = words[0]
        elif directive == "finished":
            # The ENUM, and each plane derives its own word from it. `reason` states the
            # string and is therefore echoed by both planes unchanged, which is why the gate
            # could not see the C++ plane writing `failed` where Python writes `evicted`.
            fields["reason"] = _reason_of(words[0], where)
        elif directive == "frame":
            fields["frame"] = int(words[0])
        elif directive == "size":
            fields["width"], fields["height"] = int(words[0]), int(words[1])
        elif directive == "fps":
            fields["fps"] = _number(words[0], where)
        elif directive in ("captured_ns", "captured_unix_ns", "emitted_unix_ns"):
            fields[directive] = int(words[0])
        elif directive == "missing":
            fields["missing"] = tuple(words)
        else:
            raise ConfigurationError(f"{where}: unknown directive {directive!r}")
    if "name" not in fields:
        raise ConfigurationError(f"{path}: no scenario line")
    return EventScenario(objects=tuple(objects), **fields)  # type: ignore[arg-type]
