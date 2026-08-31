"""The register of decided divergences: differences the two planes are *known* to have.

A parity gate with no register has two settings, and both are wrong: fail on a difference
somebody already decided to keep, or loosen the comparison until it stops noticing. So a
difference is either a bug or an entry here -- with a citation on both sides, an OPEN ledger
line naming the fix, and a case that reproduces it. ``xfail`` is banned: an entry whose
divergence has been fixed must fail, or the register rots into a suppression list.

Entries are **field-level** (``seam`` is ``"<kind>.<field>"``): one health record can carry
two independently decided differences, and an entry that had to explain a whole record would
have to know about the other entry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from benchmarks.parity.trace import Record

__all__ = ["KNOWN", "KnownDivergence"]


@dataclass(frozen=True, slots=True)
class KnownDivergence:
    """One decided difference between the planes, with everything needed to unmake it.

    ``explains`` is ``None`` for a divergence that shows up in no trace field at all -- a
    return value, a lifetime rule -- which is documented here and reproduced by ``case``
    rather than by the differ.
    """

    id: str
    seam: str
    python: str
    cpp: str
    decided_in: str
    ledger: str
    case: str
    explains: Callable[[Record, Record], bool] | None = None

    def matches(self, python_record: Record, cpp_record: Record) -> bool:
        """Whether this entry explains this particular pair. Documentary entries never do."""
        return self.explains is not None and self.explains(python_record, cpp_record)


def _only_the_exception_type_prefix(python_record: Record, cpp_record: Record) -> bool:
    """Python's ``last_error`` is the C++ one with ``"<ExceptionType>: "`` in front of it."""
    mine = str(python_record.fields()["last_error"])
    theirs = str(cpp_record.fields()["last_error"])
    head, separator, tail = mine.partition(": ")
    return bool(separator) and head.isidentifier() and tail == theirs


def _fatal_open_charges_one_consecutive_failure(
    python_record: Record, cpp_record: Record
) -> bool:
    """A fatal open leaves the count at 0 on one plane and 1 on the other, never elsewhere.

    Keyed on the error's own words as well as on the counts. 0-against-1 on an unhealthy
    camera is a *shape*, and a future divergence of the same shape on some other failure
    would be a new decision rather than this one -- an entry that absorbed it would be
    suppressing a difference nobody had looked at.

    The key is ``"is unavailable"``, the message both planes build a
    ``SourceUnavailableError`` from, and not the type name: keying on the type name would
    tie this entry to ``last_error_type_prefix`` staying unfixed.
    """
    return (
        int(python_record.fields()["consecutive_failures"]) == 0
        and int(cpp_record.fields()["consecutive_failures"]) == 1
        and python_record.fields()["state"] == "unhealthy"
        and "is unavailable" in str(python_record.fields()["last_error"])
    )


KNOWN: Mapping[str, KnownDivergence] = {
    entry.id: entry
    for entry in (
        KnownDivergence(
            id="last_error_type_prefix",
            seam="health.last_error",
            python=(
                "src/shipinfer/ingest/camera/actor.py::CameraActor._record_failure stores "
                'f"{type(error).__name__}: {error}"'
            ),
            cpp=(
                "csrc/shipinfer/ingest/camera/actor.cpp::CameraActor::record_failure stores "
                "redact_in(reason), which is what() with no type in front of it"
            ),
            decided_in="P6 PR-A: found by this harness; neither spelling has been chosen yet",
            ledger="[ ] P6-D1 CameraHealth.last_error: pick one spelling across the planes",
            case="test_the_python_plane_still_prefixes_the_exception_type",
            explains=_only_the_exception_type_prefix,
        ),
        KnownDivergence(
            id="fatal_consecutive_failures",
            seam="health.consecutive_failures",
            python=(
                "src/shipinfer/ingest/camera/actor.py::CameraActor.health reports "
                "self._backoff.attempts, and the fatal SourceUnavailableError path never "
                "calls next_delay(), so the count stays 0"
            ),
            cpp=(
                "csrc/shipinfer/ingest/camera/actor.cpp::CameraActor::record_failure "
                "increments consecutive_failures_ itself, so the same fatal open leaves 1"
            ),
            decided_in="P6 PR-A: found by this harness",
            ledger=(
                "[ ] P6-D2 consecutive_failures after a fatal open: 0 (py) vs 1 (cpp) -- "
                "decide whether a failure that is never retried counts as one"
            ),
            case="test_a_fatal_open_leaves_the_python_failure_count_at_zero",
            explains=_fatal_open_charges_one_consecutive_failure,
        ),
        KnownDivergence(
            id="stop_fate_stickiness",
            seam="actor.stop",
            python=(
                "src/shipinfer/ingest/camera/actor.py::CameraActor.stop re-reads "
                "thread.is_alive() on every call, so a second stop() on a thread that has "
                "since exited answers True"
            ),
            cpp=(
                "csrc/shipinfer/ingest/camera/actor.h:139-145 -- thread_abandoned_ is "
                "'STICKY, deliberately, where the Python plane's stop() re-reads live "
                "is_alive()', because a detach is irreversible and its lifetime debt is "
                "permanent"
            ),
            decided_in=(
                "#39 round 4 (P4-NB5), stated in the header: the planes diverge only on a "
                "repeat stop() after an abandoned thread later exits"
            ),
            ledger=(
                "[ ] P6-D3 stop() fate stickiness: decide whether Python should latch the "
                "abandonment too, or the C++ header stays the single statement of it"
            ),
            case="test_a_second_stop_still_answers_live_on_the_python_plane",
        ),
    )
}
