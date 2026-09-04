"""The queue-scenario format: what the request queue is told to do, call by call.

Line-oriented and not JSON, so the C++ half parses it with no vendored library -- the same
rule, and the same shape, as ``scenario.py``::

    scenario fair_eviction   # then: records_min, queue, capacity, overflow,
    queue fair               #   max_batch_size, max_delay_us
    put cam_loud [rows] [priority] [expired]
    take
    close

``take`` is refused on an open empty queue: ``get_batch`` blocks there by contract, and a
scenario that hangs one plane and not the other is not a gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shipinfer.core.errors import ConfigurationError

__all__ = [
    "OVERFLOWS",
    "PRIORITIES",
    "QUEUE_NAMES",
    "Op",
    "QueueScenario",
    "load_queue_scenario",
]

#: Queue implementations both planes register under the same name.
QUEUE_NAMES = ("fair", "fifo")
#: ``core.settings.OverflowPolicy`` / ``shipinfer::Overflow``, spelled as the config spells it.
OVERFLOWS = ("reject", "block", "drop_oldest")
#: ``core.request.Priority`` / ``shipinfer::Priority``, lower served first.
PRIORITIES = ("tracking_critical", "high", "normal", "background")

INT_SETTINGS = frozenset({"capacity", "max_batch_size", "max_delay_us", "records_min"})


@dataclass(frozen=True, slots=True)
class Op:
    """One operation: ``put`` with its item, or a bare ``take`` / ``close``."""

    verb: str
    camera: str = ""
    rows: int = 1
    priority: str = "normal"
    expired: bool = False


@dataclass(frozen=True, slots=True)
class QueueScenario:
    """A parsed scenario: the queue to build, and the operations to drive it with."""

    name: str
    queue: str
    capacity: int
    overflow: str
    max_batch_size: int
    max_delay_us: int
    records_min: int
    ops: tuple[Op, ...]


def _int(value: str, directive: str, path: Path, number: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{path}:{number}: {directive} takes an integer") from exc


def load_queue_scenario(path: Path) -> QueueScenario:
    """Parse one scenario, naming the line of any refusal.

    Raises:
        ConfigurationError: an unknown directive, a bad value, a missing ``scenario`` or
            ``queue`` line, or no operations at all -- each with the line number, because a
            scenario that fails to load in one plane and not the other is the worst outcome
            available here.
    """
    fields: dict[str, object] = {"max_delay_us": 0, "records_min": 1}
    ops: list[Op] = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        directive, *rest = line.split()
        if directive in INT_SETTINGS:
            fields[directive] = _int(rest[0] if rest else "", directive, path, number)
        elif directive == "scenario":
            fields["name"] = rest[0] if rest else ""
        elif directive == "queue":
            if not rest or rest[0] not in QUEUE_NAMES:
                raise ConfigurationError(f"{path}:{number}: queue is one of {QUEUE_NAMES}")
            fields["queue"] = rest[0]
        elif directive == "overflow":
            if not rest or rest[0] not in OVERFLOWS:
                raise ConfigurationError(f"{path}:{number}: overflow is one of {OVERFLOWS}")
            fields["overflow"] = rest[0]
        elif directive == "put":
            ops.append(_put(rest, path, number))
        elif directive in ("take", "close"):
            if rest:
                raise ConfigurationError(f"{path}:{number}: {directive} takes no argument")
            ops.append(Op(directive))
        else:
            raise ConfigurationError(f"{path}:{number}: unknown directive {directive!r}")
    return _built(fields, tuple(ops), path)


def _put(rest: list[str], path: Path, number: int) -> Op:
    """``put <camera> [rows] [priority] [expired]``, positional and all but the first optional."""
    if not rest:
        raise ConfigurationError(f"{path}:{number}: put names a camera")
    camera, *tail = rest
    rows = _int(tail[0], "put rows", path, number) if tail else 1
    if rows < 1:
        raise ConfigurationError(f"{path}:{number}: put rows must be >= 1")
    priority = tail[1] if len(tail) > 1 else "normal"
    if priority not in PRIORITIES:
        raise ConfigurationError(f"{path}:{number}: priority is one of {PRIORITIES}")
    if len(tail) > 2 and tail[2] != "expired":
        raise ConfigurationError(f"{path}:{number}: the fourth word is 'expired' or nothing")
    return Op("put", camera, rows, priority, len(tail) > 2)


def _built(fields: dict[str, object], ops: tuple[Op, ...], path: Path) -> QueueScenario:
    """Every required directive present, and at least one operation to run."""
    for required in ("name", "queue", "capacity", "overflow", "max_batch_size"):
        if required not in fields:
            raise ConfigurationError(f"{path}: no {required} line")
    if not ops:
        raise ConfigurationError(f"{path}: no operations, so it would compare nothing")
    return QueueScenario(ops=ops, **fields)  # type: ignore[arg-type]
