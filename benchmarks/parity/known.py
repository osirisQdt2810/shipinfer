"""The register of decided divergences: differences the two planes are *known* to have.

A parity gate with no register has two settings, and both are wrong: fail on a difference
somebody already decided to keep, or loosen the comparison until it stops noticing. So a
difference is either a bug or an entry here -- with a citation on both sides, an OPEN ledger
line naming the fix, and a case that reproduces it. ``xfail`` is banned: an entry whose
divergence has been fixed must fail, or the register rots into a suppression list.

It holds **one** entry. The three it opened with were closed by making the planes agree, and
the survivor is not a missing step: it is two different RULES for which rows go null. Entries
are field-level (``seam`` is ``"<kind>.<field>"``) because one health record can carry two
independently decided differences.
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


# doc: long the register's one surviving entry, and why it is a rule and not a missing step
#: One entry, documentary (``explains=None``): it reaches different ``track_id`` streams for
#: one chain file, which no golden here holds yet -- there is no tracking driver in
#: `benchmarks/parity/` at all. P6-D1/D2/D3 and `mtmc_group_routing` (#258, closed by #263)
#: were closed by converging; #259 and #261 narrowed `tracker_options` to its last knob.
#:
#: WHAT SURVIVES IS NOT A MISSING STEP but a different RULE for which rows go null. C++
#: decides inside the association, on the PREDICTED box, fused with detector score
#: (`IoU*score >= match_threshold`, 0.2; stage two 0.5). Python decides after the correction,
#: on the POSTERIOR box, on raw IoU at `attribution_iou` 0.3. Those are not comparable
#: numbers, so the nulled sets can differ -- at the DEFAULT, since no chain in this tree
#: states the knob and it cannot reach the plan anyway (`plan.py::_tracker_options`
#: serialises only `params: options:`). Refusing the key on a C++-routed chain would change
#: nothing. The fix that removes this is in shipvision: see `SHIPVISION-TRACK-LAST-MATCH`.
KNOWN: Mapping[str, KnownDivergence] = {
    "tracker_options": KnownDivergence(
        id="tracker_options",
        seam="track.track_id",
        python=(
            "src/shipinfer/topology/elements/track.py::_attribute RE-DERIVES which detection "
            "row each published track came from, by IoU between the track's FILTERED box and "
            "the frame's detection boxes, cut at `params: attribution_iou:` (default 0.3). It "
            "must: `shipvision.types.Track` has no field naming the detection that corrected "
            "the track, so the mapping dies at the library's Python boundary -- even on the "
            "native backend, where the exact column crosses the binding, is spent on the "
            "appearance EMA (3rdparty/shipvision/shipvision/mot/backends/native.py) and is "
            "then dropped by `decode`. A track matching no row above the cut claims none, and "
            "that row serialises with a null track_id"
        ),
        cpp=(
            "csrc/shipinfer/pipeline/tracking/shard.cpp does the same mapping EXACTLY and "
            "needs no threshold: it reads `Track::last_match`, the index of the detection "
            "that corrected the track, recorded by the pool when the measurement is applied "
            "(3rdparty/shipvision/shipvision/csrc/shipvision/mot/pool.cpp) and guaranteed to "
            "index the list `update` was handed. A row no confirmed track claims keeps -1 and "
            "is skipped in csrc/shipinfer/pipeline/graph/stages.cpp, which is how "
            "events/records.cpp leaves its track_id null -- so BOTH planes drop a row for a "
            "poor overlap and both write the same `null`. SETTLED 13 Sep: this lane is not "
            "missing a step and must not grow one. `options`, `regression_reset` and "
            "`algorithm` no longer diverge (#259, #261)"
        ),
        decided_in=(
            "PR #215 review round 3, finding 2; narrowed in #259 and #261; the premise "
            "corrected and the direction settled 13 Sep under CSRC-TRACKER-ATTRIBUTION"
        ),
        ledger="[ ] SHIPVISION-TRACK-LAST-MATCH carry the matched row onto the Python Track",
        case="test_each_plane_attributes_a_row_its_own_way",
    ),
}
