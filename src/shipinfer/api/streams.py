"""``/streams`` — where cameras and videos enter a running deployment (arch.md section 2).

The other half of this package's job, and the half that is not KServe. ``routes.py`` is the
*tensor* side-door: a caller who already has pixels posts them to
``/v2/models/{model}/infer``. This is the *camera* door: ``POST /streams`` with a url and
the runner starts reading it, ``DELETE /streams/{id}`` and it stops.

**Served by ``shipinfer run --http``, not by ``shipinfer serve``.** ``run`` is the
composition root that owns a runner, and a runner is the only thing here that owns cameras;
``serve`` builds an engine and no runner, so there would be nothing behind these routes.

**What this module may not do is build a runner.** It is handed one, through
:class:`CameraController` — a five-member ``Protocol`` that
:class:`shipinfer.runners.base.Runner` happens to satisfy structurally. ``api`` may name
``launch`` (``CameraSpec`` is the launcher's vocabulary) and may **not** name ``runners``,
which is what keeps an HTTP handler from constructing an executor, choosing a placement or
deciding what to batch — the dispatch layer wearing a router (arch.md section 6). The
Protocol is the whole seam, small enough to write a fake for in ten lines, which is what
``tests/api/test_streams.py`` does for every status code below.

**No authentication, and the default bind is loopback.** These routes start and stop video
decoding on a GPU box. ``shipinfer run --http`` defaults ``--host`` to ``127.0.0.1``, the
same choice the gRPC control plane makes for a shard, and a deployment that needs it
reachable puts an authenticating proxy in front rather than changing the default (phase B
has no auth).
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
from shipinfer.core.errors import (
    ConfigurationError,
    DuplicateCameraError,
    ServerStateError,
    ShipInferError,
)
from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.request import Priority
from shipinfer.launch.control import CameraSpec, mint_camera_id

__all__ = ["CameraController", "build_streams_router"]

_LOG = get_logger("api")

#: Ceiling on how long ``POST /streams`` may hold a request open waiting for a runner to
#: take the camera. Mirrors ``routes.py``'s ``_INFER_TIMEOUT_S`` and for the same reason: an
#: unbounded wait hands a wedged runner the power to hold HTTP workers forever, and enough
#: of those and the server stops answering its own health check. The two are equal today and
#: deliberately separate: one bounds an inference, this bounds a placement.
_ADD_TIMEOUT_S = 120.0

#: Ceiling on ``POST /streams/drain?timeout_s=``. Five minutes is far past any honest drain,
#: and the number bounds the *thread*, not the drain: the handler is synchronous, so the
#: caller's deadline is how long it holds one of anyio's forty shared workers. Without a
#: ceiling, forty requests stop the server answering anything at all.
_MAX_DRAIN_S = 300.0

#: What a camera's decoder gets to stop in on ``DELETE``. The same default
#: :meth:`shipinfer.runners.base.Runner.remove_camera` documents; a thread still running at it
#: is abandoned, and the answer says so in the body rather than in the status.
_REMOVE_TIMEOUT_S = 5.0


class CameraController(Protocol):
    """What ``/streams`` needs from whatever is behind it. Five members, no more.

    ``stats()`` was a sixth and is gone: no route called it, and a protocol member nothing uses
    is a requirement placed on every future controller for nothing. Per-camera counters are the
    metrics exporter's job (``core/metrics``); if a ``GET /stats`` is ever wanted, the member
    comes back with the route that needs it.

    Structural rather than an import of :class:`shipinfer.runners.base.Runner`, and that is the
    point rather than a convenience: ``api`` must be *unable* to build a runner, so the only way
    one gets behind these routes is that ``cli/commands/run.py`` hands it over. Both shipped
    runners satisfy this by having been written to the control plane's own vocabulary (arch.md
    section 2), so there is no adapter to keep in step.

    :attr:`manages_cameras` is part of the contract because "no" is a real answer and not an
    error: a runner that executes a chain but owns no ingest plane (``deepstream``, phase E) is
    correctly configured and simply cannot take a camera. That is a 501, which tells an operator
    to change ``--runner`` rather than to retry.
    """

    @property
    def manages_cameras(self) -> bool:
        """Whether :meth:`add_camera`, :meth:`remove_camera` and :meth:`drain` do anything.

        A ``@property`` and not a plain ``manages_cameras: bool``, even though this router
        only ever reads it, and the difference is not stylistic. A plain annotation declares a
        *settable instance* variable, and mypy then rejects both of the shapes that actually
        implement this protocol: every runner declares ``manages_cameras: ClassVar[bool]``
        (``runners/base.py``) -- "expected instance variable, got class variable" -- and a
        test double that computes it is a read-only attribute. Declared read-only here, all
        three satisfy it: a ``ClassVar``, a ``property``, and a plain ``self.x = True``.
        """

    def add_camera(self, camera: CameraSpec) -> None: ...

    def remove_camera(self, camera_id: str, *, timeout_s: float = 5.0) -> bool: ...

    def drain(self, timeout_s: float = 20.0) -> int: ...

    def health(self) -> dict[str, Any]: ...


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

    def _health(*, needed: bool = False) -> Mapping[str, Any]:
        """The controller's health report, or an honest stand-in — unless it is *needed*.

        Lenient by default, and that is right for every read: a listing that 500s because one
        shard is unreachable is useless exactly when it is needed, and the fleet's own report
        already carries an ``unreachable`` entry per shard for that case.

        ``needed=True`` is the one place the same failure is not cosmetic. :func:`_mint` reads
        the taken ids *out of this report*, so a stand-in with no ``cameras`` key does not mean
        "no cameras are running" — it means nothing is known — and minting from it hands out
        ``cam-000`` on a deployment that has run fifty cameras all morning. The caller then gets
        a **400 naming an id they never supplied**, which reports a control-plane fault as their
        mistake and is terminal. Raised, it is a ``ServerStateError`` -> 503, which says what is
        true: the server could not find out, ask again.

        The distinction is *which read*, not which route. ``add_stream``'s read **after** a
        successful placement is deliberately lenient: the camera is placed by then, and a
        failure there would earn a retry and a duplicate (:func:`_placed_from`).

        Raises:
            ServerStateError: only when ``needed`` and the controller could not report.
        """
        try:
            return cameras.health()
        except Exception as exc:  # a control-plane failure, reported rather than raised
            _LOG.warning("the camera controller could not report health: %s", exc)
            if needed:
                raise ServerStateError(
                    "the camera controller could not report which ids are in use, so the "
                    f"server cannot safely name this camera ({type(exc).__name__}: {exc}); "
                    "retry, or POST a camera_id of your own"
                ) from exc
            return {"state": "unknown", "detail": f"{type(exc).__name__}: {exc}"}

    async def _report(*, needed: bool = False) -> Mapping[str, Any]:
        """The same report, fetched off the event loop. For the ``async`` handler only.

        ``health()`` is blocking work on every runner and *serial* blocking work on a fleet —
        one gRPC ``Health`` per shard, each with its own deadline — so an ``async`` handler
        calling it directly would park the event loop for the sum of them, and one wedged shard
        would freeze every request this process is answering. So the read goes to a worker
        thread; the plain ``def`` handlers below get the same treatment for free, because that
        is where FastAPI runs them.

        ``abandon_on_cancel`` for :func:`add_stream`'s reason: without it the cancel scope waits
        for the very thread it is cancelling. Letting this one finish alone is safe —
        :func:`_health` changes nothing, whichever way it answers.

        ``needed`` is passed straight through; see :func:`_health` for which read sets it.

        Raises:
            ServerStateError: ``needed`` and the controller could not report.
        """
        return await anyio.to_thread.run_sync(
            partial(_health, needed=needed), abandon_on_cancel=True
        )

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

        The body is the runner's own report, passed through. On a fleet it carries
        ``lost: {camera_id: shard_id}`` — the cameras of shards whose process has exited,
        which are not being read and will not be re-placed (ADR-018). **That view lags**: it
        is a poll of the child processes taken as the report is built, so a shard that died
        a moment ago shows up on the next probe, within the launcher's supervision interval
        (``poll_s``, one second by default). An empty ``lost`` a fraction of a second after a
        crash is not a promise that nothing died.
        """
        return dict(_health())

    # -- writing -------------------------------------------------------------------------

    def _spec(body: StreamRequest, camera_id: str) -> CameraSpec:
        """The posted camera as the launcher's vocabulary, under the id it will be placed at.

        One function for the two constructions in :func:`add_stream` — the provisional spec the
        timeout message names and the named one that is actually placed. They were two argument
        lists, which is how ``priority`` would have reached the runner on one path and not the
        other, and the path it would have missed is the one that places cameras.

        The band arrives as a *name* (:data:`~shipinfer.api.schemas.BandName`) and becomes a
        :class:`~shipinfer.core.request.Priority` here, at the one place the wire vocabulary
        meets the launcher's — by :meth:`~shipinfer.core.request.Priority.parse`, the same call
        the schema's validator and :class:`~shipinfer.core.settings.ingest.CameraConfig` make.
        One rule for what a band is, owned in ``core``: a band added, renamed or aliased there
        reaches this door with no edit, and cannot reach it *differently*.

        The call cannot fail: the schema already refused every string that is not a member name,
        having lower-cased it first, which is what makes an unknown band a 422 rather than a
        500. That the validator has already narrowed the input is also why ``parse`` taking the
        numbers too is not a widening here: ``{"priority": 0}`` never reaches this function.
        """
        return CameraSpec(
            camera_id=camera_id,
            url=body.url,
            fps=body.fps,
            loop=body.loop,
            priority=None if body.priority is None else Priority.parse(body.priority),
        )

    async def _named(body: StreamRequest) -> CameraSpec:
        """The posted camera, named by the server when the caller did not name it.

        The report is read with ``needed=True`` because this is the read :func:`_mint` acts
        on, and a stand-in report mints a name that is already taken (:func:`_health` says
        what that costs the caller). It is only read when there is a name to mint: a POST that
        supplied its own ``camera_id`` needs no report and is therefore still placed on a
        deployment whose health is unreportable, which is the right answer -- nothing about
        that request depends on what the controller could not say.

        Raises:
            ServerStateError: an id had to be minted and the controller could not report which
                ones are in use.
        """
        camera_id = body.camera_id or _mint(_camera_ids(await _report(needed=True)))
        return _spec(body, camera_id)

    async def _hand_over(camera: CameraSpec) -> None:
        """Give one camera to the controller, on a worker thread it may hold past the scope."""
        await anyio.to_thread.run_sync(
            partial(cameras.add_camera, camera), abandon_on_cancel=True
        )

    @router.post("/streams", response_model=StreamInfo, status_code=201)
    async def add_stream(body: StreamRequest) -> StreamInfo:
        """Start reading one camera. 201 with where it landed.

        ``async`` for one reason: the deadline. Everything this handler does is blocking work —
        a health report per name, a lock, a thread start, an ``AddCamera`` RPC per shard — so
        **every one of those calls goes to a worker thread**. A plain ``def`` handler would
        reach the same pool with *no* bound on how long it holds a thread; an ``async`` one
        calling a controller method directly would park the event loop for the whole call and
        freeze every other request, ``GET /health`` first among them. ``abandon_on_cancel=True``
        is what makes the bound real: without it the cancel scope waits for the very thread it
        is cancelling.

        **What the deadline buys is the socket, not the thread.** A cancelled ``run_sync``
        returns to the caller and leaves the worker where it was, and the workers come from
        anyio's default limiter — 40 of them, shared with every plain ``def`` route in this app.
        So forty simultaneously wedged adds still stop the server answering, and the limit on
        that is the controller's own per-call deadline (``launch/client.py``), not this one.
        What this bound does guarantee is that no *caller* waits forever.

        The retry is :func:`_mint`'s: a minted id is read from one report and acted on in a
        later call, so two concurrent id-less POSTs can pick the same name and the loser is
        refused — with a message about an id the caller never supplied, so the name is minted
        once more against a fresh report. An id the *caller* chose is never retried: that
        duplicate is their 400 and it will be a duplicate on the next try too.

        The retry is keyed on :class:`~shipinfer.core.errors.DuplicateCameraError` and nothing
        wider. ``add_camera`` refuses for other configuration reasons too — an in-process runner
        whose chain names an unregistered source raises a plain ``ConfigurationError`` — and
        re-minting on those did the entire add a second time for a request that was a 400 either
        way.

        Raises:
            HTTPException: 501 if this runner manages no cameras; 400 for a duplicate id
                (``DuplicateCameraError``), any other configuration refusal, or a value the
                schema did not constrain that a layer below rejected (a ``ValueError``, which is
                what a pydantic ``ValidationError`` is); 503 for a runner that is not running, a
                fleet with no room, or a controller that could not report which ids are taken
                when one had to be minted (all ``ServerStateError``, and
                ``NoShardAvailableError`` is one — see ``api/errors.py``); 504 if the placement
                outran ``_ADD_TIMEOUT_S``.
        """
        _refuse_if_it_manages_no_cameras()
        # The request as posted, so a timeout taken before the server has named anything still
        # has something to name in its answer. Replaced by the named spec below.
        camera = _spec(body, body.camera_id)
        try:
            with anyio.fail_after(_ADD_TIMEOUT_S):
                camera = await _named(body)
                try:
                    await _hand_over(camera)
                except DuplicateCameraError:
                    if body.camera_id:
                        raise
                    camera = await _named(body)
                    await _hand_over(camera)
                report = await _report()
        except TimeoutError as exc:
            named = (
                f"camera {camera.camera_id!r}"
                if camera.camera_id
                else f"the camera at {camera.url!r}"
            )
            raise HTTPException(
                504,
                f"{named} was not placed within {_ADD_TIMEOUT_S:.0f}s; "
                "it may still be being placed - read GET /streams before retrying",
            ) from exc
        except ShipInferError as exc:
            raise http_error(exc) from exc
        except ValueError as exc:
            # A value this router did not think to constrain, refused a layer down.
            # Pydantic's own `ValidationError` is a `ValueError` and is NOT a
            # `ShipInferError`, so `CameraConfig`'s refusal used to fall past the clause
            # above into a 500 — and over gRPC into a refusal from every shard, a retryable
            # 503 for a request that can never succeed. `StreamRequest` now rejects the
            # three values a client actually gets wrong before this handler runs; this is
            # the net under everything else the settings tree validates.
            #
            # The net is wider than that, and the trade is accepted knowingly: a genuine
            # internal `ValueError` out of a runner is relabelled as the caller's mistake.
            # The alternative is to catch pydantic's `ValidationError` by name, which would
            # put pydantic's exception type in an HTTP handler and *still* 500 on the
            # settings tree's own plain `ValueError`s, which are the common case. The
            # traceback that says which it really was is written here — an `HTTPException`
            # is handled by starlette and would leave no trace on the server at all.
            _LOG.exception(
                "POST /streams refused a value the schema did not constrain",
                extra=log_context(camera_id=camera.camera_id),
            )
            raise HTTPException(400, str(exc)) from exc
        _LOG.info(
            "camera %s added over HTTP",
            camera.camera_id,
            extra=log_context(camera_id=camera.camera_id),
        )
        return _placed_from(report, camera)

    @router.delete("/streams/{camera_id}", response_model=StreamRemoved)
    def remove_stream(camera_id: str) -> StreamRemoved:
        """Stop reading one camera. 200 even when its thread had to be abandoned.

        ``clean=False`` is a body signal and never a 5xx (``StreamRemoved``): the camera is
        gone from the deployment either way, so a status that invites a retry would earn the
        caller a 404 and the impression that something worse happened.

        Raises:
            HTTPException: 501 if this runner manages no cameras; 404 for a camera nobody holds
                — the shared mapping's 400 is deliberately overridden here, because the resource
                is named in the URL and "there is no such camera" is what 404 says; 503 for a
                fleet that is not running.
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
        timeout_s: float = Query(
            20.0,
            ge=0.0,
            le=_MAX_DRAIN_S,
            description="One deadline for every camera.",
        )
    ) -> DrainResult:
        """Stop reading every camera and let what is in flight finish.

        One deadline for the whole camera set, not one per camera — everyone is signalled at
        once, so a camera still unfinished at it is genuinely stuck, and charging the budget per
        camera would turn one stuck decoder into fifty consecutive waits. The chain, its workers
        and the queue stay up: a drain is how a deployment is emptied without being torn down.

        The deadline is bounded at both ends. This is a plain ``def`` handler, so it holds one
        of anyio's forty shared worker threads for as long as the drain takes, and a caller who
        asks for ``timeout_s=1e9`` parks that thread for thirty-one years — forty such requests
        and the app answers nothing, health check included.

        Raises:
            HTTPException: 501 if this runner manages no cameras; 503 if it is not running; 422
                for a deadline outside ``[0, _MAX_DRAIN_S]``, which FastAPI answers before this
                function is entered.
        """
        _refuse_if_it_manages_no_cameras()
        try:
            abandoned = cameras.drain(timeout_s)
        except ShipInferError as exc:
            raise http_error(exc) from exc
        return DrainResult(abandoned=abandoned)

    return router


def _camera_ids(health: Mapping[str, Any]) -> frozenset[str]:
    entries = health.get("cameras")
    return frozenset(entries) if isinstance(entries, Mapping) else frozenset()


def _placed_from(health: Mapping[str, Any], camera: CameraSpec) -> StreamInfo:
    """What to say about a camera that was just accepted, given one health report.

    A pure function of a report rather than a call that fetches its own, because the caller is
    an ``async`` handler and ``health()`` blocks: the report is read once, on a worker thread,
    inside the deadline that handler owns. Taking it here would put a serial per-shard RPC back
    on the event loop, the one thing ``POST /streams`` is arranged to avoid.

    The report is consulted so the answer carries where the camera landed — the interesting fact
    on a fleet — and the posted ``url`` is filled in because no runner's health carries one. A
    camera missing from the report is still a 201: the add returned success, and reporting a
    failure would earn a retry and a duplicate.

    ``priority`` is deliberately *not* filled in from ``camera`` the way ``url`` is. The band on
    the spec is what was asked for; the field reports what the controller resolved, and on a
    camera the report does not mention nothing has resolved anything yet — so a 201 echoing the
    request would confirm a placement in a lane no runner has agreed to. It reads ``null``, and
    the ``GET`` that follows says what actually happened.
    """
    for info in _streams(health):
        if info.camera_id == camera.camera_id:
            return info.model_copy(update={"url": camera.url})
    return StreamInfo(camera_id=camera.camera_id, url=camera.url)


def _mint(taken: Container[str]) -> str:
    """The lowest free ``cam-<n>``, so a POST with no id cannot collide with ``--inputs``.

    :func:`~shipinfer.launch.control.mint_camera_id` is the CLI's helper too, and "lowest free"
    rather than "one past the highest" is what makes a deployment that has removed ``cam-001``
    reuse the name instead of drifting to ``cam-137`` — the ids are labels on every metric and
    log line, and unbounded drift makes them unreadable.

    **This is read-then-act and it is not atomic.** ``taken`` comes from a health report that
    was true when it was fetched; the only thing that decides a name is unique is the controller
    (an ingest manager's lock, or a shard's). Two concurrent id-less POSTs therefore mint the
    same ``cam-000`` and one is refused, with a message about an id its caller never supplied.
    There is nothing to lock here that would help, because the winner is decided a layer down;
    the answer is that ``add_stream`` re-mints against a fresh report once when the id was the
    server's own, and never when it was the caller's.
    """
    index = 0
    while mint_camera_id(index) in taken:
        index += 1
    return mint_camera_id(index)


def _streams(health: Mapping[str, Any]) -> list[StreamInfo]:
    """A runner's health report as a list of cameras, whichever runner wrote it.

    The two shapes are both honest and neither is a superset of the other:

    * in-process — ``cameras: {id: CameraHealth.as_dict()}``, so the ingest ``state`` is right
      there and the shard is the runner's own ``shard_id``;
    * fleet — ``cameras: {id: {"shard": n, "pending": True}}`` with the per-camera detail one
      level down in ``shards[n]["cameras"][id]``, because that is the shard's own report and
      the launcher passes it through, plus a top-level ``lost: {id: shard}`` naming the cameras
      of shards that have exited (ADR-018). ``lost`` is read from *there* rather than from the
      per-camera entry because it is a fact about the shard, not the camera: one dead process
      makes a dozen cameras lost at once.

    ``state`` and ``priority`` are read from the per-camera entry first and the shard's own
    entry second, because that is where each exists on each runner. The two are the same join
    and are taken together (:func:`_on_shard`), so a listing cannot say a camera is streaming on
    one shard and banded on another.

    Read defensively — ``.get`` and an ``isinstance`` per level — because a health report is a
    ``dict`` by design (``launch/control.py`` explains why it is not a wire message) and a
    listing that raises on an unreachable shard's entry is one that fails exactly when it is
    being read to find out what is wrong.
    """
    entries = health.get("cameras")
    if not isinstance(entries, Mapping):
        return []
    default_shard = health.get("shard_id")
    lost = health.get("lost")
    lost_ids = frozenset(lost) if isinstance(lost, Mapping) else frozenset()
    streams: list[StreamInfo] = []
    for camera_id, entry in sorted(entries.items()):
        detail = entry if isinstance(entry, Mapping) else {}
        shard = detail.get("shard", default_shard)
        on_shard = _on_shard(health, shard, str(camera_id))
        streams.append(
            StreamInfo(
                camera_id=str(camera_id),
                url=str(detail.get("url") or ""),
                shard=shard if isinstance(shard, int) else None,
                pending=bool(detail.get("pending", False)),
                state=str(detail.get("state") or on_shard.get("state") or ""),
                lost=str(camera_id) in lost_ids,
                # `or` is safe on these two and on nothing else in this file: a band name is
                # a non-empty string or `None`, so there is no falsy band to demote the way
                # `Priority.TRACKING_CRITICAL or ...` would (ADR-005).
                priority=_band_of(detail) or _band_of(on_shard),
            )
        )
    return streams


def _on_shard(health: Mapping[str, Any], shard: object, camera_id: str) -> Mapping[str, Any]:
    """What a shard's own report says about one of its cameras, or an empty mapping.

    A launcher's camera map says *where* a camera is and the shard's report says *how it is
    doing* and *which lane it landed in*; joining them here is what makes ``GET /streams``
    answer the questions an operator actually has ("which camera is dark", "did my
    ``tracking_critical`` take effect") on a fleet as well as in process.

    Returns the entry rather than one field of it because both callers want the same entry:
    two lookups down the same four levels could disagree with each other about which shard
    they read, and there is nothing to be gained by allowing that.
    """
    shards = health.get("shards")
    if not isinstance(shards, Mapping):
        return {}
    entry = shards.get(str(shard))
    reported = entry.get("cameras") if isinstance(entry, Mapping) else None
    camera = reported.get(camera_id) if isinstance(reported, Mapping) else None
    return camera if isinstance(camera, Mapping) else {}


def _band_of(detail: Mapping[str, Any]) -> str | None:
    """The scheduler band a per-camera health entry names, lower-cased, or ``None``.

    ``None`` for an entry that names no band, and for one whose band this deployment does not
    have. The second is not defensiveness for its own sake: a launcher's report is assembled
    from shard replies that crossed a ``google.protobuf.Struct``
    (``launch/proto/shard.proto``) and were written by a process this one did not build, so a
    shard from an older release is a real source of a name that is not in
    :class:`~shipinfer.core.request.Priority`. Answering ``None`` degrades one field of one
    camera; raising would fail the whole listing, and a listing fails when it is being read
    to find out what is wrong.

    :meth:`~shipinfer.core.request.Priority.parse` rather than a membership test against
    :data:`~shipinfer.api.schemas.BAND_NAMES`, for the reason :func:`build_streams_router`'s
    ``_spec`` gives: one rule decides what a band is. It accepts the numbers as well, which
    is right *here* and wrong at the request door — this is a report being read, so a runner
    that wrote ``0`` meant the band and not "unset", and there is no client to mislead.
    """
    band = detail.get("priority")
    if band is None:
        return None
    try:
        return Priority.parse(band).name.lower()
    except ValueError:
        return None
