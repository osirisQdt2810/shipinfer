"""The register of decided divergences: differences the two planes are *known* to have.

A parity gate with no register has two settings, and both are wrong: fail on a difference
somebody already decided to keep, or loosen the comparison until it stops noticing. So a
difference is either a bug or an entry here -- with a citation on both sides, an OPEN ledger
line naming the fix, and a case that reproduces it. ``xfail`` is banned: an entry whose
divergence has been fixed must fail, or the register rots into a suppression list.

It is **empty today**, which is the register working: the three entries it opened with were
closed by making the planes agree. Entries are field-level (``seam`` is ``"<kind>.<field>"``)
because one health record can carry two independently decided differences.
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


#: One entry, documentary (``explains=None``): the difference is in what each plane READS
#: from a chain, so it reaches no trace field -- it reaches two different ``track_id``
#: streams for one chain file, which no golden here holds yet. P6-D1/D2/D3 were closed by
#: converging the planes, and this one is open work with the same intent.
KNOWN: Mapping[str, KnownDivergence] = {
    "tracker_options": KnownDivergence(
        id="tracker_options",
        seam="track.track_id",
        python=(
            "src/shipinfer/topology/elements/track.py:526-541 reads params `algorithm`, "
            "`options`, `regression_reset` and `attribution_iou`, and TrackerShard refuses an "
            "unknown option key at open()"
        ),
        cpp=(
            "csrc/shipinfer/pipeline/tracking/bytetrack.cpp holds a default-constructed "
            "TrackerShard, and PlanNode carries no options, algorithm or regression_reset, so "
            "a chain that states them runs the defaults with nothing saying so"
        ),
        decided_in="PR #215 review round 3, finding 2",
        ledger="[ ] CSRC-TRACKER-OPTIONS carry the tracker's params on the plan",
        case="test_the_cpp_plane_reads_no_tracker_params",
    )
}
