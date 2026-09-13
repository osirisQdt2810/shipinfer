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


#: One entry, documentary (``explains=None``): the difference is in what each plane
#: READS from a chain, so it reaches no trace field -- it reaches different ``track_id``
#: streams for one chain file, which no golden here holds yet. P6-D1/D2/D3 were closed by
#: converging the planes; #259 narrowed `tracker_options`, #258 opened `mtmc_group_routing`.
KNOWN: Mapping[str, KnownDivergence] = {
    "tracker_options": KnownDivergence(
        id="tracker_options",
        seam="track.track_id",
        python=(
            "src/shipinfer/topology/elements/track.py reads `params: attribution_iou:`, which "
            "drops a detection row whose IoU with every published track is below it -- so the "
            "row serialises with a null track_id rather than somebody else's"
        ),
        cpp=(
            "csrc/shipinfer/pipeline/tracking/bytetrack.cpp has no attribution step at all: "
            "TrackerShard::update returns an id per detection, so no row is ever dropped for a "
            "poor overlap and a chain stating attribution_iou gets it on one plane only. "
            "`options`, `regression_reset` and `algorithm` no longer diverge -- the first two "
            "cross on the plan, and an algorithm this lane has not is now REFUSED by name "
            "rather than run as bytetrack"
        ),
        decided_in="PR #215 review round 3, finding 2; narrowed in #259 and again in #261",
        ledger="[ ] CSRC-TRACKER-ATTRIBUTION one plane drops a row for a poor overlap",
        case="test_the_cpp_plane_has_no_attribution_step",
    ),
}
