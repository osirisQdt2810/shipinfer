"""The differ: two traces in, one report out. Pure, and the only thing that decides pass.

Per-camera sequences are compared in order and fleet-level records as their own sequence.
Cross-camera interleaving is **never** compared: which camera's actor thread reached its
first read first is scheduler nondeterminism, and a gate that flaps is a gate somebody
turns off.

At most one difference is reported per camera -- the first, with the field names that
differ. A deleted record otherwise cascades into a difference at every later index, and a
report nobody can read is a report nobody acts on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from benchmarks.parity.known import KNOWN, KnownDivergence
from benchmarks.parity.trace import Record, Trace, differing_fields

__all__ = ["Accepted", "Difference", "ParityReport", "by_camera", "compare"]

#: Where fleet-level records (``stop``, ``end``) are filed, and how a report names them.
FLEET = ""


@dataclass(frozen=True, slots=True)
class Difference:
    """One unexplained disagreement, at one index of one camera's sequence."""

    camera: str
    index: int
    left: Record | None
    right: Record | None
    why: str

    def render(self) -> str:
        return f"{self.camera or '<fleet>'}[{self.index}]: {self.why}"


@dataclass(frozen=True, slots=True)
class Accepted:
    """A disagreement a :class:`~benchmarks.parity.known.KnownDivergence` accounts for."""

    known_id: str
    camera: str
    index: int
    fields: tuple[str, ...]

    def render(self) -> str:
        return (
            f"{self.camera or '<fleet>'}[{self.index}]: {', '.join(self.fields)} -- known "
            f"divergence {self.known_id}"
        )


@dataclass(frozen=True, slots=True)
class ParityReport:
    """What the comparison found: what differs, and what was already decided to differ."""

    differences: tuple[Difference, ...]
    accepted: tuple[Accepted, ...]

    @property
    def ok(self) -> bool:
        return not self.differences

    def render(self) -> str:
        """Camera-first, one line each -- the form a failing gate has to be read in."""
        lines = [d.render() for d in self.differences]
        lines += [f"accepted: {a.render()}" for a in self.accepted]
        return "\n".join(lines) or "identical"


def by_camera(records: tuple[Record, ...]) -> dict[str, tuple[Record, ...]]:
    """Split a record stream into one sequence per camera, plus the fleet's under ``""``."""
    grouped: dict[str, list[Record]] = {}
    for record in records:
        grouped.setdefault(record.camera, []).append(record)
    return {camera: tuple(entries) for camera, entries in grouped.items()}


def _explained(
    python_record: Record,
    cpp_record: Record,
    fields: tuple[str, ...],
    known: Mapping[str, KnownDivergence],
) -> str | None:
    """The entry id that accounts for every differing field, or ``None`` if any is unexplained."""
    ids = []
    for field in fields:
        seam = f"{python_record.kind}.{field}"
        entry = next(
            (
                e
                for e in known.values()
                if e.seam == seam and e.matches(python_record, cpp_record)
            ),
            None,
        )
        if entry is None:
            return None
        ids.append(entry.id)
    return "+".join(sorted(set(ids))) if ids else None


def compare(
    left: Trace, right: Trace, *, known: Mapping[str, KnownDivergence] = KNOWN
) -> ParityReport:
    """Compare two traces of the same scenario.

    The register is consulted **only across planes**. Two traces from the same plane -- a
    fresh Python run against a Python-emitted golden -- must be identical, because any
    difference there is drift within one plane and no cross-plane decision can excuse it.

    Raises:
        ValueError: the traces are of different scenarios, which is a mistake in the caller
            rather than a parity finding.
    """
    if left.scenario != right.scenario:
        raise ValueError(
            f"refusing to compare scenario {left.scenario!r} against {right.scenario!r}"
        )
    cross_plane = left.plane != right.plane
    python_first = left.plane == "python"
    mine, theirs = by_camera(left.records), by_camera(right.records)
    differences: list[Difference] = []
    accepted: list[Accepted] = []
    for camera in sorted(set(mine) | set(theirs)):
        ours, yours = mine.get(camera, ()), theirs.get(camera, ())
        found = False
        for index, (one, other) in enumerate(zip(ours, yours, strict=False)):
            fields = differing_fields(one, other)
            if not fields:
                continue
            pair = (one, other) if python_first else (other, one)
            known_id = _explained(*pair, fields, known) if cross_plane else None
            if known_id is not None:
                accepted.append(Accepted(known_id, camera, index, fields))
                continue
            differences.append(Difference(camera, index, one, other, _why(fields, one, other)))
            found = True
            break
        if not found and len(ours) != len(yours):
            index = min(len(ours), len(yours))
            differences.append(
                Difference(
                    camera,
                    index,
                    ours[index] if index < len(ours) else None,
                    yours[index] if index < len(yours) else None,
                    f"{len(ours)} record(s) here against {len(yours)}; first extra is "
                    f"{(ours[index] if index < len(ours) else yours[index]).render()}",
                )
            )
    return ParityReport(tuple(differences), tuple(accepted))


def _why(fields: tuple[str, ...], one: Record, other: Record) -> str:
    """One sentence naming the fields and both values -- what a failing gate has to say."""
    if fields == ("kind",):
        return f"{one.kind} here, {other.kind} there ({one.render()} | {other.render()})"
    mine, theirs = one.fields(), other.fields()
    return f"{one.kind}: " + ", ".join(
        f"{field} {mine[field]!r} against {theirs[field]!r}" for field in fields
    )
