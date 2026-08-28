"""Runs the fleet: start, stop, add and remove cameras while the server is live.

Adding a camera at runtime is not a luxury. The reference service exposed it over REST for a
reason — a fifty-camera site gains and loses cameras during commissioning, and restarting
the whole perception tier to onboard one of them means restarting the trackers too, which
loses every tracklet on every other camera. So the manager owns actor lifecycle, and an
actor is cheap: one thread, one decoder, no GPU state.

There is exactly one actor per camera id and one thread per actor, and neither is ever
recycled. A stopped camera that comes back gets a *new* actor, because reusing one would
mean deciding what its frame counter should now say, and the only safe answer to that is
"keep counting" — which is what a new actor with a preserved ``first_frame_id`` does
explicitly rather than by accident.
"""

from __future__ import annotations

import threading
import time

from shipinfer.core.errors import (
    CameraUnavailableError,
    ConfigurationError,
    DuplicateCameraError,
    ServerStateError,
)
from shipinfer.core.logging import LOG, log_context
from shipinfer.core.settings.ingest import CameraConfig, IngestSettings
from shipinfer.ingest.camera.actor import CameraActor, SourceFactory
from shipinfer.ingest.camera.db import load_camera_db
from shipinfer.ingest.camera.health import CameraHealth, CameraState, IngestSummary
from shipinfer.ingest.metrics import IngestMetrics
from shipinfer.ingest.sink import FrameSink

__all__ = ["IngestManager", "configured_cameras"]


def configured_cameras(settings: IngestSettings) -> list[CameraConfig]:
    """The enabled cameras a settings tree declares: the inline list plus ``camera_db``.

    A module function rather than only a method because the camera set is a *question about
    configuration*, and two callers have to ask it without owning a manager: the CLI, which
    places the configured fleet on whichever runner the operator chose
    (``cli/commands/run.py``), and :class:`~shipinfer.runners.inprocess.InprocessRunner`,
    which reads each camera's priority band out of it. Both used to reach for
    :meth:`IngestManager.configured_cameras`, which meant building the very object whose
    ``start()`` would run the fleet.

    Resolved eagerly and validated before a single thread is started, so a mistyped database
    is a start-up failure rather than fifty actors failing one at a time.

    Raises:
        ConfigurationError: a camera id is declared both inline and in the database. Silently
            preferring one would give the fleet a camera nobody configured.
    """
    cameras = list(settings.cameras)
    if settings.camera_db is not None:
        cameras.extend(load_camera_db(settings.camera_db))
    seen: set[str] = set()
    for camera in cameras:
        if camera.camera_id in seen:
            raise ConfigurationError(
                f"camera {camera.camera_id!r} is declared both inline and in "
                f"{settings.camera_db}"
            )
        seen.add(camera.camera_id)
    return [camera for camera in cameras if camera.enabled]


class IngestManager:
    """Owns every camera actor, and the fleet's health.

    Args:
        sink: where every camera publishes. One sink for the fleet, because the fairness
            that matters is *between* cameras and can only be arbitrated somewhere they
            meet — but see :mod:`shipinfer.ingest.sink` for why that arbitration is the
            consumer's job and not this package's (ADR-005).
        settings: the fleet configuration, including the camera list.
        metrics: shared metric handles, labelled by camera.
        source_factory: overrides source construction for every actor. The seam a test uses
            to run the whole manager against a fake camera.
    """

    def __init__(
        self,
        sink: FrameSink,
        *,
        settings: IngestSettings | None = None,
        metrics: IngestMetrics | None = None,
        source_factory: SourceFactory | None = None,
    ) -> None:
        self.settings = settings or IngestSettings()
        self.metrics = metrics or IngestMetrics()
        self._sink = sink
        self._source_factory = source_factory
        self._actors: dict[str, CameraActor] = {}
        self._lock = threading.Lock()
        self._started = False

    # -- fleet lifecycle ---------------------------------------------------------------

    def configured_cameras(self) -> list[CameraConfig]:
        """The cameras :meth:`start` will run: the settings list plus ``camera_db``.

        :func:`configured_cameras` over this manager's own settings; see it for why the
        resolution is a function first and a method second.
        """
        return configured_cameras(self.settings)

    def start(self) -> None:
        """Start an actor for every enabled camera. Idempotent."""
        if self._started:
            return
        cameras = self.configured_cameras()
        self._started = True
        for camera in cameras:
            self.add_camera(camera)
        LOG.info("ingest started with %d camera(s)", len(cameras))

    def stop(self, timeout_s: float = 5.0) -> int:
        """Stop every actor. Idempotent, and safe before :meth:`start`.

        Stop requests are issued to *all* actors first and only then joined, so shutting
        down fifty cameras costs one read timeout rather than fifty. The first pass is
        :meth:`~shipinfer.ingest.camera.actor.CameraActor.request_stop`, which is what that
        method exists for: a ``stop(timeout_s=0.0)`` joins for zero seconds, and
        ``Thread.join(0.0)`` returns immediately with the thread still alive — so every
        clean shutdown logged "did not stop within 0.0s; abandoning the thread" once per
        camera and marked each one STOPPED while it was still reading and publishing. Fifty
        false alarms per shutdown is how a real abandoned thread stops being noticed.

        ``timeout_s`` is one deadline for the whole fleet, not one per actor — synced from
        the C++ plane (#33): because the first pass signals everyone at t0, an actor still
        unfinished at t0+timeout is genuinely stuck, and charging the budget per actor would
        turn one stuck decoder into fifty consecutive waits. Returns how many actors had to
        be abandoned; 0 is the clean shutdown.
        """
        with self._lock:
            actors = list(self._actors.values())
            self._actors.clear()
        for actor in actors:
            actor.request_stop()
        deadline = time.monotonic() + timeout_s
        abandoned = 0
        for actor in actors:
            remaining = max(0.0, deadline - time.monotonic())
            if not actor.stop(timeout_s=remaining):
                abandoned += 1
        self._started = False
        self._refresh_gauges([])
        if actors:
            LOG.info(
                "ingest stopped %d camera(s)%s",
                len(actors),
                f", {abandoned} abandoned" if abandoned else "",
            )
        return abandoned

    def __enter__(self) -> IngestManager:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- per-camera control ------------------------------------------------------------

    #: The re-check's stop grace, mirroring the C++ plane's ``kRecheckStopGrace``: a
    #: freshly started actor that got its stop signal before its first wait is gone in
    #: milliseconds; one that is not is inside a blocked open, and the error path of an API
    #: call must not stall for the full shutdown grace.
    _RECHECK_STOP_GRACE_S = 0.25

    def add_camera(self, config: CameraConfig) -> CameraActor:
        """Start one camera. Returns its actor.

        Safe against a concurrent :meth:`stop`/:meth:`remove_camera` — the C++ plane's
        re-check (#35/#39), mirrored (#35 review, P4-NB2-py): the actor is started outside
        the lock, and a fleet that forgot the camera in that window aimed its stop request
        at a thread that did not exist yet (``CameraActor.start`` clears the signal). The
        map is re-checked after the start; a forgotten camera is stopped here and refused,
        rather than left reading and publishing behind a manager whose ``size()`` says 0
        and whose every later ``stop()`` misses it.

        Raises:
            DuplicateCameraError: a camera with this id is already running. Silently
                replacing it would leave two threads pulling one stream and two frame
                counters producing duplicate tags. A ``ConfigurationError`` subclass, so a
                caller that wants "the add was refused" still catches the base -- and one
                that has to tell a taken name from every other refusal (``api/streams.py``
                mints ids, so it does) can name this one.
            ServerStateError: the fleet forgot this camera while it was starting — a
                concurrent stop or removal landed between the insert and the start.
        """
        with self._lock:
            if config.camera_id in self._actors:
                raise DuplicateCameraError(
                    f"camera {config.camera_id!r} is already running; "
                    "remove it before adding it again"
                )
            actor = CameraActor(
                config,
                self._sink,
                settings=self.settings,
                metrics=self.metrics,
                source_factory=self._source_factory,
            )
            self._actors[config.camera_id] = actor
        actor.start()
        with self._lock:
            forgotten = self._actors.get(config.camera_id) is not actor
        if forgotten:
            actor.stop(timeout_s=self._RECHECK_STOP_GRACE_S)
            raise ServerStateError(
                f"camera {config.camera_id!r} was removed while it was starting; the fleet "
                "is stopping or the camera was removed — add it again once the manager is "
                "running"
            )
        LOG.info(
            "camera %s added",
            config.camera_id,
            extra=log_context(camera_id=config.camera_id),
        )
        self._refresh_gauges(self._snapshot())
        return actor

    def remove_camera(self, camera_id: str, *, timeout_s: float = 5.0) -> bool:
        """Stop and forget one camera.

        Returns whether the stop was clean: ``False`` means the thread had to be abandoned
        (#35 review, P4-NB4) — the C++ counterpart parks the actor on that answer; Python
        has nothing to park (the thread's bound method keeps the actor alive), but the
        abandonment is the caller's to know, not the log's to bury.

        Raises:
            ConfigurationError: no such camera. Naming what is running turns a typo in an
                operator's API call into an answer instead of a silent no-op.
        """
        with self._lock:
            actor = self._actors.pop(camera_id, None)
            if actor is None:
                raise ConfigurationError(
                    f"camera {camera_id!r} is not running; running: {sorted(self._actors)}"
                )
        clean = actor.stop(timeout_s=timeout_s)
        LOG.info("camera %s removed", camera_id, extra=log_context(camera_id=camera_id))
        self._refresh_gauges(self._snapshot())
        return clean

    def actor(self, camera_id: str) -> CameraActor:
        """The actor for one camera.

        Raises:
            ConfigurationError: no such camera.
        """
        with self._lock:
            actor = self._actors.get(camera_id)
        if actor is None:
            raise ConfigurationError(
                f"camera {camera_id!r} is not running; running: {self.camera_ids}"
            )
        return actor

    @property
    def camera_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._actors)

    def __len__(self) -> int:
        with self._lock:
            return len(self._actors)

    def __contains__(self, camera_id: object) -> bool:
        with self._lock:
            return camera_id in self._actors

    # -- health ------------------------------------------------------------------------

    def health(self) -> dict[str, CameraHealth]:
        """One snapshot per camera, and refresh the exported gauges while we are here."""
        snapshot = self._snapshot()
        self._refresh_gauges(snapshot)
        return {health.camera_id: health for health in snapshot}

    def summary(self) -> IngestSummary:
        """The fleet in one object: how many are streaming, and what is being lost."""
        snapshot = self._snapshot()
        self._refresh_gauges(snapshot)
        return IngestSummary(
            cameras=len(snapshot),
            streaming=sum(1 for h in snapshot if h.state is CameraState.STREAMING),
            unhealthy=sum(1 for h in snapshot if h.state is CameraState.UNHEALTHY),
            total_fps=sum(h.fps for h in snapshot),
            frames_read=sum(h.frames_read for h in snapshot),
            frames_published=sum(h.frames_published for h in snapshot),
            frames_dropped=sum(h.frames_dropped for h in snapshot),
        )

    def wait_ready(self, timeout_s: float = 30.0, *, poll_s: float = 0.05) -> None:
        """Block until every camera has delivered a frame.

        What turns a mistyped camera database into a failed deploy rather than a server that
        looks healthy and produces no detections.

        Raises:
            CameraUnavailableError: naming every camera that produced nothing in time.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            pending = [h.camera_id for h in self._snapshot() if h.frames_read == 0]
            if not pending:
                return
            if time.monotonic() >= deadline:
                raise CameraUnavailableError(pending, timeout_s)
            time.sleep(poll_s)

    def _snapshot(self) -> list[CameraHealth]:
        with self._lock:
            actors = list(self._actors.values())
        return [actor.health for actor in actors]

    def _refresh_gauges(self, snapshot: list[CameraHealth]) -> None:
        self.metrics.cameras_total.set(len(snapshot))
        self.metrics.cameras_streaming.set(
            sum(1 for h in snapshot if h.state is CameraState.STREAMING)
        )
        self.metrics.cameras_unhealthy.set(
            sum(1 for h in snapshot if h.state is CameraState.UNHEALTHY)
        )

    def __repr__(self) -> str:
        return f"<IngestManager cameras={len(self)} started={self._started}>"
