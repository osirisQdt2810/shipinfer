"""Which lane band a camera's items are admitted into, and who said so.

Two tables with two lifetimes, deliberately, and that is the whole reason this is an object
rather than a dict on the runner: a launcher's placement can be *un-said* by a DELETE and a
re-POST, while a band this process's own config names is a fact about the settings and
outlives any placement. One dict meant nothing could be un-said, and a camera removed and
re-added inherited the dead placement's ``tracking_critical`` -- the wrong direction to get
wrong (ADR-005).
"""

from __future__ import annotations

import threading
from collections.abc import Mapping

from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.request import Priority

__all__ = ["PriorityBands"]

_LOG = get_logger(__name__)


class PriorityBands:
    """The runner's band bookkeeping: configured bands, placed bands, and the precedence.

    Every method is safe to call from any thread. The lock is taken on every write and on
    the one read that memoises; :meth:`for_camera` reads both dicts without it, because a
    dict lookup is atomic under the GIL and this runs once per submitted item.
    """

    def __init__(self) -> None:
        #: ``camera_id -> band`` for the cameras **this process's own configuration** names,
        #: plus the default learned once for a camera it does not. Never emptied: a camera
        #: removed and posted again is the same camera, and the operator's band for it did
        #: not change.
        self._configured: dict[str, Priority] = {}
        #: ``camera_id -> band`` a **launcher** named on the spec of a camera held right now.
        #: Tracks the live camera set: written when a spec carries a band, dropped when it
        #: does not and when the placement ends.
        self._placed: dict[str, Priority] = {}
        self._lock = threading.Lock()

    def configure(self, bands: Mapping[str, Priority]) -> None:
        """Take this process's configured bands, from the full settings.

        Empty on a fleet shard -- its ``ingest.cameras`` is stripped before the manager is
        built, which is why the launcher sends the band on the ``CameraSpec`` instead.
        """
        with self._lock:
            self._configured.update(bands)

    def record_placement(self, camera_id: str, band: Priority | None) -> None:
        """Record what a placement said. ``None`` **erases** what a previous one left."""
        with self._lock:
            if band is None:
                self._placed.pop(camera_id, None)
            else:
                self._placed[camera_id] = band

    def placed(self, camera_id: str) -> Priority | None:
        """What a placement has recorded for this camera, or ``None``.

        Not :meth:`for_camera`: this is the single value :meth:`restore` has to put back,
        without the fallback that would make ``NORMAL`` indistinguishable from "unset".
        """
        with self._lock:
            return self._placed.get(camera_id)

    def restore(self, camera_id: str, previous: Priority | None) -> None:
        """Put the placed band back the way :meth:`placed` found it.

        The undo half of an admission. ``IngestManager.add_camera`` refuses a duplicate id
        *after* the band was written, so without this a rejected ``POST /streams`` answers
        400 and still moves that camera's lane for the rest of its life. ``previous is None``
        pops rather than writes -- never ``if previous:``, since ``TRACKING_CRITICAL`` is 0.
        """
        self.record_placement(camera_id, previous)

    def forget_placement(self, camera_id: str) -> None:
        """The placement is over, so the band that came with it is over too."""
        with self._lock:
            self._placed.pop(camera_id, None)

    def clear_placements(self) -> None:
        """Every placement ends at once: a drain, or a new cycle re-reading its settings."""
        with self._lock:
            self._placed.clear()

    def for_camera(self, camera_id: str) -> Priority:
        """The band this camera's items are admitted into.

        Three sources in precedence: the placement, then this process's config, then the
        learned default. Expressed as *lookups* and not as write order into one dict, which
        is what lets a placement's band be un-said.

        ``is not None`` and never ``or``: ``TRACKING_CRITICAL`` is ``0`` and therefore falsy,
        so the shorter spelling demotes the one camera priorities exist for (ADR-005).
        """
        placed = self._placed.get(camera_id)
        if placed is not None:
            return placed
        configured = self._configured.get(camera_id)
        if configured is not None:
            return configured
        return self._learn(camera_id)

    def _learn(self, camera_id: str) -> Priority:
        """Resolve, log and memoise the band for a camera the config does not name.

        A camera added over the control plane at runtime is a normal event, so it gets the
        fleet default rather than an error. Logged **once** -- "my new camera is not being
        prioritised" is otherwise an invisible configuration gap -- and memoised, because
        paying for that discovery per frame would make it a performance bug too.

        Memoised into the *configured* table: what is recorded is that this process's
        configuration has nothing to say about this camera, which stays true after the
        camera is removed. A band a launcher named is undone by exactly that.
        """
        with self._lock:
            existing = self._configured.get(camera_id)
            if existing is not None:
                return existing
            _LOG.info(
                "camera %s is not in the ingest config; admitting it at priority %s",
                camera_id,
                Priority.NORMAL.name,
                extra=log_context(camera_id=camera_id),
            )
            self._configured[camera_id] = Priority.NORMAL
            return Priority.NORMAL
