"""The per-camera tracker table: one tracker per camera, and the ordering invariant.

Moved here from ``pipeline/graph/tracking.py`` in phase C4, unchanged in behaviour, because
the chain's ``track`` element and the counting-simulation pipeline's ``TrackStage`` both need
it and two copies of an invariant that has **no symptom when it breaks** is how one of them
stops being correct. That module imports it back, so the older graph keeps working — the
coexistence ``docs/arch.md`` section 9 describes, and the reason ``pipeline`` may import
``topology`` one-way (``scripts/hooks/check_layers.py`` states the direction).

**Why the sharding is a correctness constraint and not a scaling one.** Kalman state, track
ids and ageing all belong to one camera's view. Two cameras on one tracker associate one
camera's objects with the other's, and the output is not degraded tracking — it is a real
identity reported somewhere nothing happened. ``shipvision``'s ``BaseTracker.begin`` refuses
a camera change as a second line of defence, and that belt-and-braces is deliberate because
the failure is invisible.

**Why the ordering guard is here.** The fair lane preserves a camera's frame order (one FIFO
deque per fairness key), but the *workers* do not: with more than one pipeline worker, two of
a camera's frames are in flight at once and the later one can reach the tracker first. Feeding
a tracker a frame it has already passed double-ages every track and double-counts the hit that
promotes one, so it is refused — see :meth:`TrackerShard.update` for what the caller does with
that refusal.

**Where shipvision is named.** Nowhere at module scope. Every symbol comes from
:mod:`shipinfer.topology.bridge` inside a function, so ``import shipinfer.topology`` stays
free of the submodule and a chain that names it is still *validatable* on a host that never
checked it out.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from shipinfer.core.errors import ConfigurationError, TrackingError
from shipinfer.topology.bridge import load_mot

__all__ = [
    "DEFAULT_REGRESSION_RESET",
    "TrackerShard",
]

#: How far a camera's ``frame_id`` may go *backwards* before this element stops calling it a
#: reordering and starts calling it a restarted stream.
#:
#: Two populations have to be told apart with no lifecycle event to help. A **reorder** is
#: bounded by the number of pipeline workers — at most that many of a camera's frames are in
#: flight at once (arch.md §5③ sizes it at ~32), so a handful of frames' worth of regression is
#: ordinary and must be refused, not absorbed. A **restart** is an ingest actor that minted a
#: fresh ``FrameCounter`` without anybody calling ``remove_camera`` + ``add_camera``: the ids
#: drop from tens of thousands to zero, and refusing those means refusing every frame of that
#: camera for the process's life — the failure ADR-018 names remove+add as the recovery for,
#: with nobody left to make the two calls.
#:
#: 64 is comfortably above the first and far below the second. ``0`` disables the recovery and
#: refuses every regression, which is the old behaviour and what a deployment that would
#: rather see the frames stop than see the ids restart should set.
DEFAULT_REGRESSION_RESET = 64


class _CameraShard:
    """One camera's tracker, its lock, and how far through the stream it has been fed.

    ``__slots__`` and a plain class: one of these exists per camera for the process's life,
    so the shape matters less than it does per frame, but the lock and the high-water mark
    have to sit together — reading one without the other is the race this module prevents.
    """

    __slots__ = ("implicit_resets", "last_frame_id", "lock", "out_of_order", "tracker")

    def __init__(self, tracker: Any) -> None:
        self.lock = threading.Lock()
        self.tracker = tracker
        #: The highest ``frame_id`` fed to this tracker. ``-1`` because ``FrameTag`` allows
        #: frame 0 and a camera's first frame must not look like a replay.
        self.last_frame_id = -1
        self.out_of_order = 0
        self.implicit_resets = 0


class TrackerShard:
    """One tracker per camera, built lazily and never shared.

    The sharding is a correctness constraint, not a scaling one — see this module's
    docstring. Everything else here exists to make that constraint hold under the pipeline's
    worker threads.

    Args:
        algorithm: a name registered in ``shipvision.mot.TRACKERS`` (``sort``,
            ``bytetrack``, ``ocsort``, ``botsort``, ``deepsortv2``). Resolved through that
            registry, so adding a tracker there needs no edit here — an ``if/elif`` on this
            string is exactly what the registry exists to replace.
        options: constructor keyword arguments for that tracker.
        backend: pin ``python`` or ``native``. ``None`` takes the fastest one this host can
            actually build, which is the registry's documented behaviour and the numpy floor
            every caller in this tree wants.

    Raises:
        ConfigurationError: the library is absent, the algorithm is unknown, or ``options``
            holds a key the tracker's constructor does not accept. Raised **at construction**
            — one throwaway tracker is built here for no other reason — because a typo in a
            deployment's tracking options must stop the deploy rather than surface identically
            on every frame from inside a worker thread.
    """

    def __init__(
        self,
        algorithm: str,
        *,
        options: Mapping[str, Any] | None = None,
        backend: str | None = None,
    ) -> None:
        # Through the bridge, inside the function: one message for a missing submodule, and
        # `import shipinfer.topology` still costs nothing. It is called *first* so that "no
        # shipvision" is answered by the bridge's sentence rather than by whatever the build
        # below would say about a registry that does not exist.
        load_mot()
        self._algorithm = algorithm
        self._options = dict(options or {})
        self._backend = backend
        # Guards insertion into and removal from `_cameras` only. Never held across
        # `tracker.update`, so a slow camera cannot stall the other forty-nine, and never held
        # across a per-camera lock, so a lifecycle call cannot queue behind one camera's frame.
        self._admit = threading.Lock()
        self._cameras: dict[str, _CameraShard] = {}
        self._build()

    def _build(self) -> Any:
        try:
            return load_mot().TRACKERS.build(
                self._algorithm, backend=self._backend, **self._options
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(
                f"tracking: cannot build tracker {self._algorithm!r}"
                f"{'' if self._backend is None else f' (backend {self._backend!r})'} "
                f"with options {sorted(self._options)}: {exc}"
            ) from exc

    # -- properties ----------------------------------------------------------------------

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @property
    def cameras(self) -> tuple[str, ...]:
        """Cameras that have a tracker, in the order they first appeared."""
        return tuple(self._cameras)

    @property
    def camera_count(self) -> int:
        """How many cameras hold a tracker.

        Separate from :attr:`cameras` because it is read on the per-frame path — a gauge is
        only written when the number *changes* — and materialising a tuple of fifty strings to
        take its length would be an allocation per frame to answer a question ``len`` answers
        in one bytecode.
        """
        return len(self._cameras)

    def tracker_for(self, camera_id: str) -> Any:
        """One camera's tracker, for a test that wants to look at the state directly."""
        return self._shard(camera_id).tracker

    def stats(self) -> dict[str, int]:
        """What an operator reads: how many cameras are tracked, and how many frames lost
        the ordering race. ``stages_failed{stage="track"}`` counts the same refusals; this
        is the per-shard view of the same fact."""
        return {
            "cameras": len(self._cameras),
            "tracks": sum(s.tracker.pool_size for s in self._cameras.values()),
            "out_of_order": sum(s.out_of_order for s in self._cameras.values()),
            "implicit_resets": sum(s.implicit_resets for s in self._cameras.values()),
        }

    # -- the contract --------------------------------------------------------------------

    def update(
        self,
        detections: Any,
        *,
        image: np.ndarray | None = None,
        regression_reset: int | None = None,
        on_implicit_reset: Callable[[str], None] | None = None,
    ) -> list[Any]:
        """Advance one camera by one frame and return its publishable tracks.

        The first argument is a ``shipvision.types.Detections``, tag included, so the camera
        and the frame cannot be passed separately and cannot disagree with the boxes.

        **The invariant.** A camera's tracker sees that camera's frames one at a time and in
        strictly increasing ``frame_id`` order. The per-camera lock gives the first half; the
        per-camera high-water mark gives the second, and it has to, because reassembly does
        not order anything: two of one camera's frames can be in flight at once and the later
        one can reach this method first.

        **What a violation costs, and why it is refused rather than absorbed.** Feeding a
        tracker a frame it has already passed advances every filter a second time, ages every
        track a second time, and double-counts the hit that promotes a tentative track — so a
        replayed or reordered frame silently changes *which identities exist* downstream. The
        frame that lost the race is therefore refused: the caller emits it with its detections,
        its embeddings and its masks intact and with no track ids, naming ``track`` in
        ``missing_stages``. A frame with an honest gap is worth more than a fleet-wide identity
        built on a double-counted hit.

        **Not silently reordered, either.** A reorder buffer would have to wait for a frame
        that may never arrive — the ingest queue drops on overflow and on an expired deadline
        — so it needs a timeout, and that timeout is latency paid by every frame of every
        camera to rescue the few that raced. If the refusal rate is high enough to matter,
        the fix is upstream (fewer of one camera's frames in flight at once), and
        ``stats()["out_of_order"]`` is the number that says so.

        Args:
            detections: this frame's detections and its tag.
            image: the decoded frame, for the one tracker that compensates for camera motion.
            regression_reset: how far ``frame_id`` may go backwards before this is read as a
                restarted stream rather than a reordering — see
                :data:`DEFAULT_REGRESSION_RESET`. ``None`` (the default, and what the
                counting-simulation pipeline passes) refuses every regression.

                The decision is taken **here**, under the camera's lock, rather than by the
                caller reading :attr:`last_frame_id` and calling :meth:`reset`: those are three
                steps, and between any two of them another worker can advance the same camera.
                Read-decide-reset-update has to be one step for the same reason
                check-set-update does.
            on_implicit_reset: called with the camera id when that recovery fires, from inside
                the camera's lock. A callback rather than a return value or a counter the
                caller polls, because the event is rare and the alternatives both put work on
                the frames where nothing happened. It must not block — a counter increment is
                what it is for — since it is holding one camera's tracker while it runs.

        Raises:
            TrackingError: the frame does not advance this camera's stream, and the regression
                is small enough to be a reordering.
        """
        tag = detections.tag
        shard = self._shard(tag.camera_id)
        with shard.lock:
            if tag.frame_id <= shard.last_frame_id:
                behind = shard.last_frame_id - tag.frame_id
                if regression_reset and behind >= regression_reset:
                    # A restarted stream nobody announced. Forget the tracks and take this
                    # frame as the new stream's first: the ids restart, which is what a
                    # reconnect means, and the alternative is refusing every frame of this
                    # camera until the process ends.
                    shard.tracker.reset()
                    shard.last_frame_id = -1
                    shard.implicit_resets += 1
                    if on_implicit_reset is not None:
                        on_implicit_reset(tag.camera_id)
                else:
                    shard.out_of_order += 1
                    raise TrackingError(
                        f"camera {tag.camera_id!r}: frame {tag.frame_id} reached the tracker "
                        f"after frame {shard.last_frame_id}. Tracking is stateful and ordered; "
                        f"replaying a frame double-ages every track and double-counts the hit "
                        f"that promotes one, so this frame is published without track ids "
                        f"rather than with wrong ones"
                    )
            shard.last_frame_id = tag.frame_id
            return shard.tracker.update(detections, image=image)

    def reset(self, camera_id: str) -> None:
        """Forget one camera's tracks and its stream position, building a tracker if needed.

        For a camera that reconnected: continuity is broken, so the ids must not continue,
        and the next frame's id may be lower than the last one seen.
        """
        self._reset(self._shard(camera_id))

    def reset_if_present(self, camera_id: str) -> bool:
        """:meth:`reset`, but never *builds* a tracker for a camera that has none.

        What a lifecycle hook wants. ``camera_added`` fires for every camera on the shard,
        including the forty-nine that were never on this element, and :meth:`reset` would mint
        a tracker for each of them — a Kalman filter, a track pool and a registry build for a
        camera that may never send a frame, done on the thread holding the runner's lifecycle
        lock.

        The table is read under :attr:`_admit` and the lock is **released before** the camera's
        own lock is taken. Holding both would make every other camera's first frame queue
        behind this one camera's in-flight update; a shard dropped in the window is reset for
        nothing, which costs nothing.

        Returns:
            Whether there was a tracker to reset.
        """
        with self._admit:
            shard = self._cameras.get(camera_id)
        if shard is None:
            return False
        self._reset(shard)
        return True

    def drop(self, camera_id: str) -> bool:
        """Forget a camera entirely: its tracker, its lock and its stream position.

        **Takes the table lock and nothing else.** A worker can be inside ``tracker.update``
        for this very camera at this very moment — the walk takes no lock against the
        lifecycle — and waiting for it would hold the runner's ``_lifecycle`` for as long as
        that frame takes, stalling every other camera's add, remove and the ``stop`` that would
        end the wait. So the entry is unlinked and the in-flight frame finishes against a shard
        nobody can reach any more; its result is discarded by the caller that is being removed.

        Idempotent, and a frame that arrives after the drop rebuilds the shard as a first
        frame. Both are ordinary: ``remove_camera`` answering ``False`` means the decoder was
        abandoned at its deadline rather than joined.

        Returns:
            Whether there was anything to drop.
        """
        with self._admit:
            return self._cameras.pop(camera_id, None) is not None

    @staticmethod
    def _reset(shard: _CameraShard) -> None:
        with shard.lock:
            shard.tracker.reset()
            shard.last_frame_id = -1

    def _shard(self, camera_id: str) -> _CameraShard:
        """This camera's shard, creating it on first sight.

        The unlocked ``get`` first is not a micro-optimisation: it is what keeps admitting a
        frame off the one lock every camera shares, in the steady state where the shard
        already exists. A ``dict`` lookup is a single bytecode under the GIL, and the
        re-check inside the lock is what makes the create path safe.
        """
        shard = self._cameras.get(camera_id)
        if shard is not None:
            return shard
        with self._admit:
            shard = self._cameras.get(camera_id)
            if shard is None:
                shard = _CameraShard(self._build())
                self._cameras[camera_id] = shard
            return shard

    def __repr__(self) -> str:
        return f"<TrackerShard {self._algorithm} cameras={len(self._cameras)}>"
