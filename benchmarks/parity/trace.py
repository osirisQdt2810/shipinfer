"""The parity trace: one canonical JSONL line per observed ingest event.

Byte-exact by construction rather than by convention -- fixed key order, integers only,
printable ASCII with no quote or backslash -- because the C++ half writes JSON and never
parses it (vendoring a parser for forty lines is refused by the ponytail principle), so its
diff against the golden is a line compare and only canonicalisation makes that sound.

Records are grouped by camera on output, fleet-level records last. That grouping is what
makes the line compare valid at all: cross-camera interleaving is scheduler nondeterminism,
never a parity property, and a gate that flaps gets disabled. See ``README.md`` for the
record-kind table.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from shipinfer.core.errors import ConfigurationError

__all__ = [
    "FIELDS",
    "KINDS",
    "SCHEMA_VERSION",
    "Record",
    "Trace",
    "TraceWriter",
    "differing_fields",
    "parse_lines",
    "read_trace",
]

#: Bumped whenever a record kind changes meaning. A golden written under an older schema is
#: refused rather than silently reinterpreted.
SCHEMA_VERSION = 1

#: Every record kind, and the names of the numbers and words it carries. This table IS the
#: one in ``README.md``: arity is checked against it on write, so a plane that emits the
#: wrong shape fails where it wrote rather than as an unreadable diff.
FIELDS: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "source_open": (("attempt",), ("outcome",)),
    "source_read": (("index",), ("outcome",)),
    "source_close": (("index",), ()),
    "frame": (("frame_id",), ()),
    "drop": ((), ("reason",)),
    "retry": (("attempt", "peek_us"), ()),
    "state": ((), ("from", "to")),
    "health": (
        (
            "frames_read",
            "frames_published",
            "frames_dropped",
            "empty_reads",
            "connects",
            "connect_failures",
            "consecutive_failures",
        ),
        ("state", "last_error"),
    ),
    "stop": (("abandoned",), ()),
    "end": (("cameras", "frames_read", "frames_published", "frames_dropped"), ()),
    # The queue seam. Fleet-level, every one of them, and the item's camera travels in the
    # words: a scheduling run is single-threaded with no clock in it, so WHICH camera comes
    # out next is the invariant rather than the nondeterminism, and only the ungrouped
    # sequence can compare that.
    "qput": (("rows", "depth"), ("camera", "status")),
    "qbatch": (("items", "rows"), ()),
    "qserved": (("rows",), ("camera",)),
    "qdrop": ((), ("camera", "reason")),
    "qstats": (("accepted", "rejected", "evicted", "expired", "depth", "capacity"), ()),
    "qcam": (("depth", "rejected", "evicted", "expired"), ("camera",)),
}

#: Unknown kinds are refused by name on read: a typo that read as "no records of that kind"
#: would make the gate pass by having compared nothing.
KINDS = frozenset(FIELDS)

#: Fleet-level records carry no camera and are compared as their own sequence.
FLEET_KINDS = frozenset({"stop", "end", "qput", "qbatch", "qserved", "qdrop", "qstats", "qcam"})


@dataclass(frozen=True, slots=True)
class Record:
    """One observed event: what happened, to which camera, with which numbers and words.

    Four fields and no more, because the C++ writer has to produce the same bytes and every
    field it has to special-case is a place the two writers can drift.
    """

    kind: str
    camera: str
    numbers: tuple[int, ...]
    text: tuple[str, ...]

    def to_line(self) -> str:
        """The canonical JSONL line. Key order is this order, always."""
        return json.dumps(
            {
                "kind": self.kind,
                "camera": self.camera,
                "n": list(self.numbers),
                "t": list(self.text),
            },
            separators=(",", ":"),
        )

    def fields(self) -> dict[str, int | str]:
        """This record by field name, so a difference can say *what* differs, not "index 3"."""
        numbers, text = FIELDS[self.kind]
        named: dict[str, int | str] = dict(zip(numbers, self.numbers, strict=False))
        named.update(zip(text, self.text, strict=False))
        return named

    def render(self) -> str:
        """One human line: ``kind camera field=value …``."""
        named = " ".join(f"{name}={value!r}" for name, value in self.fields().items())
        return f"{self.kind} {self.camera or '<fleet>'} {named}".rstrip()


@dataclass(frozen=True, slots=True)
class Trace:
    """A parsed trace file: which scenario produced it, on which plane, and its records."""

    scenario: str
    plane: str
    schema_version: int
    records: tuple[Record, ...]


def _checked_word(value: str, field: str) -> str:
    """Refuse anything the two writers would have to escape differently.

    Printable ASCII without ``"`` or ``\\`` is the whole alphabet, so neither writer needs an
    escaper and neither can grow one that disagrees with the other's.
    """
    if not isinstance(value, str):
        raise ConfigurationError(f"parity trace {field} must be a string, got {value!r}")
    for char in value:
        if not 0x20 <= ord(char) <= 0x7E or char in '"\\':
            raise ConfigurationError(
                f"parity trace {field} {value!r} carries {char!r}, which the canonical "
                f"writer refuses: records are printable ASCII without a quote or backslash, "
                f"so neither plane needs an escaper"
            )
    return value


class TraceWriter:
    """Collects records from every actor thread and emits one deterministic file.

    Thread-safe because the actors that feed it are one thread per camera; ordering within a
    camera is that thread's own program order, which is the only ordering compared.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._header: dict[str, object] | None = None
        self._records: list[Record] = []

    def header(self, scenario: str, plane: str, schema_version: int = SCHEMA_VERSION) -> None:
        """Declare what produced this trace. Exactly once, before any record."""
        with self._lock:
            if self._header is not None:
                raise ConfigurationError("parity trace header written twice")
            self._header = {
                "schema": int(schema_version),
                "scenario": _checked_word(scenario, "scenario"),
                "plane": _checked_word(plane, "plane"),
            }

    def record(
        self,
        kind: str,
        camera: str = "",
        numbers: tuple[int, ...] = (),
        text: tuple[str, ...] = (),
    ) -> None:
        """Append one record.

        Raises:
            ConfigurationError: unknown kind, a non-integer number, a camera on a fleet-level
                record (or none on a per-camera one), or a string the canonical writer
                cannot spell.
        """
        if kind not in KINDS:
            raise ConfigurationError(
                f"unknown parity record kind {kind!r}; known: {sorted(KINDS)}"
            )
        if (kind in FLEET_KINDS) != (camera == ""):
            raise ConfigurationError(
                f"record {kind!r}: fleet-level kinds {sorted(FLEET_KINDS)} carry no camera "
                f"and every other kind must carry one; got camera {camera!r}"
            )
        for number in numbers:
            if not isinstance(number, int) or isinstance(number, bool):
                raise ConfigurationError(
                    f"record {kind!r}: {number!r} is not an int. A trace carries no floats "
                    f"-- a rounding that differs in the last bit is a gate that flaps"
                )
        expected_numbers, expected_text = FIELDS[kind]
        if (len(numbers), len(text)) != (len(expected_numbers), len(expected_text)):
            raise ConfigurationError(
                f"record {kind!r} carries {list(expected_numbers)} and {list(expected_text)}; "
                f"got {len(numbers)} number(s) and {len(text)} word(s)"
            )
        entry = Record(
            kind=kind,
            camera=_checked_word(camera, "camera"),
            numbers=tuple(int(n) for n in numbers),
            text=tuple(_checked_word(t, f"{kind}.text") for t in text),
        )
        with self._lock:
            self._records.append(entry)

    def lines(self) -> list[str]:
        """The file, as lines: header, each camera's records in id order, then the fleet's."""
        with self._lock:
            if self._header is None:
                raise ConfigurationError("parity trace has no header; call header() first")
            header = json.dumps(self._header, separators=(",", ":"))
            records = list(self._records)
        cameras = sorted({r.camera for r in records if r.camera})
        ordered = [r for camera in cameras for r in records if r.camera == camera]
        ordered += [r for r in records if not r.camera]
        return [header, *(r.to_line() for r in ordered)]

    def trace(self) -> Trace:
        """This trace, parsed back -- what a fresh run hands the differ without a file."""
        return parse_lines(self.lines(), where="<in-process>")

    def write(self, path: Path) -> None:
        """Write the trace, creating the parent directory if it does not exist."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines()) + "\n", encoding="ascii")


def read_trace(path: Path) -> Trace:
    """Parse a trace file.

    Raises:
        ConfigurationError: the file is missing, the header is absent or of another schema,
            or a record names a kind this version does not have. Never a skip and never a
            lenient parse: a gate that fails open is not a gate.
    """
    if not path.is_file():
        raise ConfigurationError(
            f"no parity trace at {path}. Goldens are committed, not generated on demand: "
            f"regenerate one only when the change to the plane is itself the decision"
        )
    return parse_lines(path.read_text(encoding="ascii").splitlines(), where=str(path))


def parse_lines(lines: Sequence[str], *, where: str = "<lines>") -> Trace:
    """The reader, over lines rather than a path -- what a fresh in-process run comes back as.

    Raises:
        ConfigurationError: an empty trace, a header of another schema, an unknown record
            kind, or a record of the wrong arity -- named with ``where:line`` so a bad golden
            says which line is bad.
    """
    kept = [line for line in lines if line.strip()]
    if not kept:
        raise ConfigurationError(f"parity trace {where} is empty")
    head = json.loads(kept[0])
    if head.get("schema") != SCHEMA_VERSION:
        raise ConfigurationError(
            f"parity trace {where} is schema {head.get('schema')!r}, this build reads "
            f"{SCHEMA_VERSION}"
        )
    records = []
    for number, line in enumerate(kept[1:], start=2):
        parsed = json.loads(line)
        if parsed.get("kind") not in KINDS:
            raise ConfigurationError(
                f"{where}:{number}: unknown parity record kind {parsed.get('kind')!r}; "
                f"known: {sorted(KINDS)}"
            )
        numbers, text = list(parsed.get("n", ())), list(parsed.get("t", ()))
        expected_numbers, expected_text = FIELDS[parsed["kind"]]
        # Checked on READ and not only on write, because a truncated `n[]` otherwise reaches
        # `fields()` -- which zips names against values -- and dies later as a bare KeyError
        # against a field name, in whichever test happened to look. The C++ reader's
        # `differing_fields` already treats a short record as differing on the missing field.
        if (len(numbers), len(text)) != (len(expected_numbers), len(expected_text)):
            raise ConfigurationError(
                f"{where}:{number}: record {parsed['kind']!r} carries "
                f"{list(expected_numbers)} and {list(expected_text)}; got {len(numbers)} "
                f"number(s) and {len(text)} word(s)"
            )
        records.append(
            Record(
                kind=parsed["kind"],
                camera=parsed["camera"],
                numbers=tuple(int(n) for n in numbers),
                text=tuple(str(t) for t in text),
            )
        )
    return Trace(
        scenario=str(head["scenario"]),
        plane=str(head["plane"]),
        schema_version=int(head["schema"]),
        records=tuple(records),
    )


def differing_fields(left: Record, right: Record) -> tuple[str, ...]:
    """Which named fields two records disagree on -- ``("kind",)`` when they are not even
    the same event.

    Field-level rather than record-level because the known-divergence register is
    field-level: one health record can carry two independent, separately decided
    differences, and an entry that had to explain the whole record would have to know about
    the other entry.
    """
    if left.kind != right.kind:
        return ("kind",)
    if left.camera != right.camera:
        return ("camera",)
    mine, theirs = left.fields(), right.fields()
    return tuple(name for name in mine if mine[name] != theirs[name])
