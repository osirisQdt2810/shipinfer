"""``/streams`` — where cameras and videos enter a running deployment (arch.md section 2).

The other half of this package's job, and the half that is not KServe. ``routes.py`` is the
*tensor* side-door: a caller who already has pixels posts them to
``/v2/models/{model}/infer``. This is the *camera* door: ``POST /streams {"url": "rtsp://..."}``
and the runner starts reading it, ``DELETE /streams/{id}`` and it stops.

**It is served by ``shipinfer run --http``, not by ``shipinfer serve``.** ``run`` is the
composition root that owns a runner, and a runner is the only thing in this system that owns
cameras (``runners/base.py``); ``serve`` builds an engine and no runner, so on that command
there would be nothing behind these routes to talk to.

**What this module may not do is build a runner.** It is handed one, through
:class:`CameraController` — a six-member ``Protocol`` that :class:`shipinfer.runners.base.Runner`
happens to satisfy structurally. ``api`` may name ``launch`` (``CameraSpec`` is the launcher's
vocabulary and is what ``add_camera`` takes) and may **not** name ``runners``, which is what
keeps an HTTP handler from constructing an executor, choosing a placement or deciding what to
batch — that would be the dispatch layer wearing a router (arch.md section 6). The Protocol
is the whole seam, and it is deliberately small enough to write a fake for in ten lines,
which is what ``tests/api/test_streams.py`` does for every status code below.

**No authentication, and the default bind is loopback.** These routes start and stop video
decoding on a GPU box; exposing them on ``0.0.0.0`` puts that behind no credential at all.
``shipinfer run --http`` therefore defaults ``--host`` to ``127.0.0.1``, the same choice the
gRPC control plane makes for a shard, and a deployment that needs it reachable puts an
authenticating proxy in front rather than changing the default here (phase B has no auth).
"""

from __future__ import annotations

from collections.abc import Container, Mapping
from functools import partial
from typing import Any, Protocol

from shipinfer.api.errors import http_error
from shipinfer.api.schemas import (
    DrainResult,
    StreamInfo,
    StreamList,
    StreamRemoved,
    StreamRequest,
)
from shipinfer.core.errors import ConfigurationError, ShipInferError
from shipinfer.core.logging import get_logger, log_context
from shipinfer.launch.control import CameraSpec, mint_camera_id

__all__ = ["CameraController", "build_streams_router"]

_LOG = get_logger("api")

#: Ceiling on how long ``POST /streams`` may hold a request open waiting for a runner to take
#: the camera. Mirrors ``routes.py``'s ``_INFER_TIMEOUT_S`` and for the same reason: an
#: unbounded wait hands a wedged runner — a fleet whose shard is paging in an engine, an
#: ingest manager blocked behind a stop — the power to hold HTTP workers forever, and enough
#: of those and the server stops answering its own health check. The two constants are equal
#: today and are deliberately separate: one bounds an inference, this bounds a placement, and
#: the day one of them needs tuning it is not the other.
_ADD_TIMEOUT_S = 120.0

#: What a camera's decoder gets to stop in on ``DELETE``. The same default
#: :meth:`shipinfer.runners.base.Runner.remove_camera` documents; a thread still running at it
#: is abandoned, and the answer says so in the body rather than in the status.
_REMOVE_TIMEOUT_S = 5.0


class CameraController(Protocol):
    """What ``/streams`` needs from whatever is behind it. Six members, no more.

    Structural rather than an import of :class:`shipinfer.runners.base.Runner`, and that is
    the point rather than a convenience: ``api`` must be *unable* to build a runner, so the
    only way one gets behind these routes is that ``cli/commands/run.py`` — the composition
    root — hands it over. Both shipped runners satisfy this by having been written to the
    control plane's own vocabulary (arch.md section 2), so there is no adapter to keep in step.

    :attr:`manages_cameras` is part of the contract because "no" is a real answer here and it
    is not an error: a runner that executes a chain but owns no ingest plane (``deepstream``,
    phase E) is correctly configured and simply cannot take a camera. That is a 501, which
    tells an operator to change ``--runner`` rather than to retry.
    """

    @property
    def manages_cameras(self) -> bool:
        """Whether :meth:`add_camera`, :meth:`remove_camera` and :meth:`drain` do anything."""

    def add_camera(self, camera: CameraSpec) -> None: ...

    def remove_camera(self, camera_id: str, *, timeout_s: float = 5.0) -> bool: ...

    def drain(self, timeout_s: float = 20.0) -> int: ...

    def health(self) -> dict[str, Any]: ...

    def stats(self) -> dict[str, Any]: ...


def build_streams_router(cameras: CameraController) -> Any:
    """Build the ``/streams`` router over one camera controller.

    A factory rather than a module-level ``APIRouter`` with a global, for the reason
    :func:`shipinfer.api.routes.build_router` gives: the controller is an argument, so two of
    them can be mounted in one process and a test can build one without touching module state.
    """
    import anyio
    import anyio.to_thread
    from fastapi import APIRouter, HTTPException, Query

    router = APIRouter()

    def _refuse_if_it_manages_no_cameras() -> None:
        """501, not 503 and not 500: this runner will never take a camera.

        ``Runner.add_camera``'s own refusal is a ``ServerStateError``, which the shared
        mapping turns into a retryable 503 — and a control plane that retries this one retries
        it forever, because no amount of waiting gives a ``deepstream`` runner an ingest
        manager. "Not implemented by this deployment" is exactly what 501 means.
        """
        if not cameras.manages_cameras:
            raise HTTPException(
                501,
                "this deployment's runner manages no cameras, so it can take none over HTTP; "
                "start it with a runner that does (`shipinfer runners` lists them, "
                "`shipinfer run --runner ...`)",
            )

    def _health() -> Mapping[str, Any]:
        """The controller's health report, or an honest stand-in.

        Never raises. Everything below reads the camera map out of this, and a listing that
        500s because one shard is unreachable is a listing that is useless exactly when it is
        needed — the fleet's own report already carries an ``unreachable`` entry per shard for
        that case rather than omitting it (``runners/fleet.py``).
        """
        try:
            return cameras.health()
        except Exception as exc:  # a control-plane failure, reported rather than raised
            _LOG.warning("the camera controller could not report health: %s", exc)
            return {"state": "unknown", "detail": f"{type(exc).__name__}: {exc}"}

    # -- reading -------------------------------------------------------------------------

    @router.get("/streams", response_model=StreamList)
    def list_streams() -> StreamList:
        """Every camera this deployment believes it is reading.

        Built from the runner's health report rather than from anything this router
        remembers, which is what makes it right for cameras added by ``--inputs`` or over
        gRPC as well as by ``POST`` — and what keeps it from confidently listing a camera that
        a shard restart took away.
        """
        return StreamList(streams=_streams(_health()))

    @router.get("/cameras", response_model=StreamList)
    def list_cameras() -> StreamList:
        """arch.md section 2 spells this door ``GET /cameras``; both names answer.

        One alias rather than a redirect: an operator reading the diagram and an operator
        reading the OpenAPI page should not have to discover that only one of them was right.
        """
        return StreamList(streams=_streams(_health()))

    @router.get("/health")
    def health() -> dict[str, Any]:
        """200 whatever the state, with the state in the body.

        Deliberately not 503 on a stopped runner: this is the endpoint that answers *what is
        wrong*, and a status code that hides the body is how "the deployment is down" becomes
        indistinguishable from "the deployment is unreachable". ``/v2/health/live`` and
        ``/v2/health/ready`` are the ones a load balancer reads, and they are the engine's
        (``routes.py``).
        """
        return dict(_health())

    # -- writing -------------------------------------------------------------------------

    @router.post("/streams", response_model=StreamInfo, status_code=201)
    async def add_stream(body: StreamRequest) -> StreamInfo:
        """Start reading one camera. 201 with where it landed.

        ``async`` for one reason: the deadline. ``add_camera`` is blocking work — a lock, a
        thread start, an ``AddCamera`` RPC per shard — so it goes to a worker thread either
        way, and a plain ``def`` handler would go to the same pool with *no* bound on how long
        it holds it. ``abandon_on_cancel=True`` is what makes the bound real: without it the
        cancel scope waits for the thread it is cancelling and the timeout is decorative.
        The abandoned call still finishes on its thread — a placement half-made is worse than
        one that completes late — and the caller gets 504 rather than an open socket.

        Raises:
            HTTPException: 501 if this runner manages no cameras; 400 for a duplicate id
                (``ConfigurationError``); 503 for a runner that is not running or a fleet with
                no room (``ServerStateError``, and ``NoShardAvailableError`` is one — see
                ``api/errors.py``); 504 if the placement outran ``_ADD_TIMEOUT_S``.
        """
        _refuse_if_it_manages_no_cameras()
        camera_id = body.camera_id or _mint(_camera_ids(_health()))
        camera = CameraSpec(camera_id=camera_id, url=body.url, fps=body.fps)
        try:
            with anyio.fail_after(_ADD_TIMEOUT_S):
                await anyio.to_thread.run_sync(
                    partial(cameras.add_camera, camera), abandon_on_cancel=True
                )
        except TimeoutError as exc:
            raise HTTPException(
                504,
                f"camera {camera_id!r} was not placed within {_ADD_TIMEOUT_S:.0f}s; "
                "it may still be being placed - read GET /streams before retrying",
            ) from exc
        except ShipInferError as exc:
            raise http_error(exc) from exc
        _LOG.info(
            "camera %s added over HTTP", camera_id, extra=log_context(camera_id=camera_id)
        )
        return _placed(camera)

    @router.delete("/streams/{camera_id}", response_model=StreamRemoved)
    def remove_stream(camera_id: str) -> StreamRemoved:
        """Stop reading one camera. 200 even when its thread had to be abandoned.

        ``clean=False`` is a body signal and never a 5xx (``StreamRemoved``): the camera is
        gone from the deployment either way, so a status that invites a retry would earn the
        caller a 404 and the impression that something worse happened.

        Raises:
            HTTPException: 501 if this runner manages no cameras; 404 for a camera nobody
                holds — the shared mapping's 400 is deliberately overridden here, because the
                resource is named in the URL and "there is no such camera" is what 404 says;
                503 for a fleet that is not running.
        """
        _refuse_if_it_manages_no_cameras()
        try:
            clean = cameras.remove_camera(camera_id, timeout_s=_REMOVE_TIMEOUT_S)
        except ConfigurationError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ShipInferError as exc:
            raise http_error(exc) from exc
        if not clean:
            _LOG.warning(
                "camera %s was removed but its thread was abandoned",
                camera_id,
                extra=log_context(camera_id=camera_id),
            )
        return StreamRemoved(clean=clean)

    @router.post("/streams/drain", response_model=DrainResult)
    def drain_streams(
        timeout_s: float = Query(20.0, ge=0.0, description="One deadline for every camera.")
    ) -> DrainResult:
        """Stop reading every camera and let what is in flight finish.

        One deadline for the whole camera set, not one per camera — everyone is signalled at
        once, so a camera still unfinished at it is genuinely stuck and charging the budget
        per camera would turn one stuck decoder into fifty consecutive waits
        (``ingest/manager.py``). The chain, its workers and the queue stay up: a drain is how
        a deployment is emptied without being torn down.

        Raises:
            HTTPException: 501 if this runner manages no cameras; 503 if it is not running.
        """
        _refuse_if_it_manages_no_cameras()
        try:
            abandoned = cameras.drain(timeout_s)
        except ShipInferError as exc:
            raise http_error(exc) from exc
        return DrainResult(abandoned=abandoned)

    # -- reading a health report -----------------------------------------------------------

    def _placed(camera: CameraSpec) -> StreamInfo:
        """What to say about a camera that was just accepted.

        The health report is consulted so the answer carries where it landed — the interesting
        fact on a fleet — and the posted ``url`` is filled in because the caller supplied it
        and no runner's health carries one (:class:`StreamInfo`). A camera missing from the
        report is still a 201: the add returned success, and reporting a failure would earn a
        retry and a duplicate.
        """
        for info in _streams(_health()):
            if info.camera_id == camera.camera_id:
                return info.model_copy(update={"url": camera.url})
        return StreamInfo(camera_id=camera.camera_id, url=camera.url)

    return router


def _camera_ids(health: Mapping[str, Any]) -> frozenset[str]:
    entries = health.get("cameras")
    return frozenset(entries) if isinstance(entries, Mapping) else frozenset()


def _mint(taken: Container[str]) -> str:
    """The lowest free ``cam-<n>``, so a POST with no id cannot collide with ``--inputs``.

    :func:`~shipinfer.launch.control.mint_camera_id` is the CLI's helper too, and "lowest
    free" rather than "one past the highest" is what makes a deployment that has removed
    ``cam-001`` reuse the name instead of drifting to ``cam-137`` — the ids are labels on
    every metric and log line, and unbounded drift makes them unreadable.
    """
    index = 0
    while mint_camera_id(index) in taken:
        index += 1
    return mint_camera_id(index)


def _streams(health: Mapping[str, Any]) -> list[StreamInfo]:
    """A runner's health report as a list of cameras, whichever runner wrote it.

    The two shapes are both honest and neither is a superset of the other
    (``runners/inprocess.py`` and ``runners/fleet.py``):

    * in-process — ``cameras: {id: CameraHealth.as_dict()}``, so the ingest ``state`` is right
      there and the shard is the runner's own ``shard_id``;
    * fleet — ``cameras: {id: {"shard": n, "pending": True}}`` with the per-camera detail one
      level down in ``shards[n]["cameras"][id]``, because that is the shard's own report and
      the launcher passes it through rather than re-deriving it.

    Read defensively — ``.get`` and an ``isinstance`` per level — because a health report is a
    ``dict`` by design (``launch/control.py`` explains why it is not a wire message) and a
    listing that raises on an unreachable shard's entry is a listing that fails exactly when
    it is being read to find out what is wrong.
    """
    entries = health.get("cameras")
    if not isinstance(entries, Mapping):
        return []
    default_shard = health.get("shard_id")
    streams: list[StreamInfo] = []
    for camera_id, entry in sorted(entries.items()):
        detail = entry if isinstance(entry, Mapping) else {}
        shard = detail.get("shard", default_shard)
        state = detail.get("state") or _state_on_shard(health, shard, str(camera_id))
        streams.append(
            StreamInfo(
                camera_id=str(camera_id),
                url=str(detail.get("url") or ""),
                shard=shard if isinstance(shard, int) else None,
                pending=bool(detail.get("pending", False)),
                state=str(state or ""),
            )
        )
    return streams


def _state_on_shard(health: Mapping[str, Any], shard: object, camera_id: str) -> str:
    """The ingest state a shard reported for one of its cameras, or ``""``.

    A launcher's camera map says *where* a camera is and the shard's own report says *how it
    is doing*; joining them here is what makes ``GET /streams`` answer the question an
    operator actually has ("which camera is dark") on a fleet as well as in process.
    """
    shards = health.get("shards")
    if not isinstance(shards, Mapping):
        return ""
    entry = shards.get(str(shard))
    reported = entry.get("cameras") if isinstance(entry, Mapping) else None
    camera = reported.get(camera_id) if isinstance(reported, Mapping) else None
    return str(camera.get("state", "")) if isinstance(camera, Mapping) else ""
