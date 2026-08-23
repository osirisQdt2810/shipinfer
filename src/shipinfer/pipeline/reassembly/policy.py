"""Who loses a slot when reassembly is full — the decision the old system got wrong.

The previous generation held per-person results in one shared buffer of 1000 entries and,
when full, dropped the **globally oldest** entry (``m_buffer`` in
``BodyDataCollector.h``, ``personBuffer`` in ``docs/flow.md``). A camera watching a busy
quay produced 30 detections a frame where a corridor camera produced 2, so the busy camera's
own inflow evicted the quiet camera's frames before they finished the pipeline. The operator
saw it exactly as you would expect: *"camera đông người được nhận diện đầy đủ, camera vắng
người thỉnh thoảng bị miss"* — the crowded cameras complete, the quiet ones intermittently
lose frames. Nothing logged it, because dropping the oldest entry is not an error.

The fix is one line of policy: **penalise the camera holding the most incomplete frames**,
never the frame that has waited longest. A flood then pays for itself, and a camera that
contributes two frames cannot be evicted by a camera that contributes thirty.

This is also why it is a registry rather than an ``if``: the inherited behaviour is shipped
here as :class:`OldestFrameEviction` so the regression test and the benchmark can run the
two side by side and *show* the difference, instead of asserting that the new one is fine.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import ClassVar, Protocol

from shipinfer.core.registry import Registry

__all__ = [
    "EVICTION_POLICIES",
    "EvictionPolicy",
    "GreediestCameraEviction",
    "OldestFrameEviction",
    "PendingIndex",
    "PendingKey",
]

#: ``(camera_id, frame_id)`` — the reassembly key, as it is everywhere else in the system.
PendingKey = tuple[str, int]


class PendingIndex(Protocol):
    """The three questions an eviction policy may ask about what is pending.

    Narrow on purpose. A policy handed the whole buffer could sort it, scan it, or keep a
    reference to it, and any of those turns an O(1) decision into an O(N) one on the path
    that is only reached when the system is already under pressure.
    """

    def camera_counts(self) -> Mapping[str, int]:
        """How many incomplete frames each camera is holding."""

    def oldest(self, camera_id: str) -> PendingKey | None:
        """The longest-waiting key for one camera, or ``None`` if it holds none."""

    def oldest_overall(self) -> PendingKey | None:
        """The longest-waiting key across every camera."""


class EvictionPolicy(abc.ABC):
    """Chooses which pending frame is dropped to make room for a new one."""

    name: ClassVar[str] = "abstract"

    @abc.abstractmethod
    def choose(self, pending: PendingIndex) -> PendingKey | None:
        """The key to evict, or ``None`` when nothing can be evicted.

        ``None`` means the collector refuses the new frame instead — an honest rejection, in
        the same spirit as :class:`~shipinfer.core.errors.QueueFullError`.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


EVICTION_POLICIES: Registry[EvictionPolicy] = Registry(
    "reassembly eviction policy", EvictionPolicy
)


@EVICTION_POLICIES.register("greediest_camera", "fair", "greedy")
class GreediestCameraEviction(EvictionPolicy):
    """Drop the oldest frame of the camera holding the most — the default, and the fix.

    Two properties make it the right default:

    * **a flood pays for itself.** The camera whose inflow filled the buffer is the one that
      loses a slot, so a quiet camera's frames are never collateral.
    * **it is a dynamic quota.** A static per-camera cap wastes capacity when only three
      cameras are busy; this achieves the same protection and adapts to the fleet's actual
      shape, which changes through the day.

    Ties break on ``camera_id`` so the choice is deterministic and therefore testable — an
    arbitrary tie-break would make a fairness assertion flaky at exactly the moment two
    cameras are equally loud.
    """

    name: ClassVar[str] = "greediest_camera"

    def choose(self, pending: PendingIndex) -> PendingKey | None:
        counts = pending.camera_counts()
        if not counts:
            return None
        # O(cameras), and only on overflow. 50 comparisons on a path that is already the
        # unhappy one; `open` and `deliver` stay O(1), which is where the frames are.
        camera = max(counts, key=lambda cam: (counts[cam], cam))
        return pending.oldest(camera)


@EVICTION_POLICIES.register("oldest_frame", "inherited", "drop_oldest")
class OldestFrameEviction(EvictionPolicy):
    """Drop the longest-waiting frame, whatever camera it belongs to.

    **This is the inherited bug.** It is shipped so that the regression test and the
    benchmark can demonstrate the difference rather than assert the absence of a problem —
    the same role the ``custom_*`` allocators play against torch's. Configuring it in
    production re-creates the silent starvation ADR-005 exists to prevent, and its own
    settings field says so.
    """

    name: ClassVar[str] = "oldest_frame"

    def choose(self, pending: PendingIndex) -> PendingKey | None:
        return pending.oldest_overall()
