"""``/streams`` over a fake controller: every status code, and why each one is that code.

The router is handed a :class:`~shipinfer.api.streams.CameraController` — six members — so a
fake is ten lines and every branch is reachable without a runner, a camera or a GPU. That the
fake is this small is itself the evidence the protocol is the right size (CONVENTIONS 2.9);
``tests/api/test_streams_over_a_runner.py`` is the other half, where a real
:class:`~shipinfer.runners.inprocess.InprocessRunner` sits behind the same routes.

What is under test is mostly the **mapping**, because that is what a client acts on:

* 400 vs 503 — a duplicate camera id will still be a duplicate on the next try; a fleet with
  no room will not still be full, so one is terminal and the other is retryable;
* 501 — a runner that owns no ingest plane is correctly configured and will never take a
  camera, so retrying is pointless and changing ``--runner`` is the fix;
* 404 on ``DELETE`` — the URL names a resource that does not exist;
* **200 with ``clean=false``** — the camera *was* removed and its decoder outlived the
  deadline. A 5xx there says the removal failed, and a control plane that retries gets a 404
  and concludes something worse happened.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Container
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from shipinfer.api import create_app
from shipinfer.api import streams as streams_module
from shipinfer.api.streams import CameraController
from shipinfer.core.errors import (
    ConfigurationError,
    NoShardAvailableError,
    ServerStateError,
)
from shipinfer.launch.control import CameraSpec

#: A launcher's report: where each camera was placed, with the shard's own per-camera detail
#: one level down. `cam-001` is *pending* — placed, not yet accepted (`runners/fleet.py`).
FLEET_HEALTH: dict[str, Any] = {
    "runner": "fleet",
    "state": "running",
    "shards": {
        "0": {
            "placed": ["cam-000"],
            "state": "running",
            "cameras": {"cam-000": {"camera_id": "cam-000", "state": "streaming"}},
        },
        "1": {"placed": [], "state": "unreachable", "detail": "TimeoutError: deadline"},
    },
    "cameras": {"cam-000": {"shard": 0}, "cam-001": {"shard": 1, "pending": True}},
}

#: An in-process runner's report: no shard map, the ingest state inline, and the runner's own
#: `shard_id` is the only shard there is.
INPROCESS_HEALTH: dict[str, Any] = {
    "runner": "inprocess",
    "state": "running",
    "shard_id": 0,
    "cameras": {"cam-000": {"camera_id": "cam-000", "state": "streaming", "fps": 19.9}},
}


class FakeCameras:
    """A controller that records what it was asked and answers what the test told it to."""

    def __init__(
        self,
        *,
        manages_cameras: bool = True,
        health: dict[str, Any] | None = None,
        refuse: Exception | None = None,
        refuse_ids: Container[str] = frozenset(),
        clean: bool = True,
        abandoned: int = 0,
    ) -> None:
        self.manages_cameras = manages_cameras
        self._health = health if health is not None else {"state": "running", "cameras": {}}
        self.refuse = refuse
        #: Ids this controller already holds, refused the way an ingest manager refuses a
        #: duplicate -- and remembered, because whoever took the name is in the report now.
        self.refuse_ids = refuse_ids
        self.clean = clean
        self.abandoned = abandoned
        self.added: list[CameraSpec] = []
        self.removed: list[tuple[str, float]] = []
        self.drained: list[float] = []
        #: Held for the duration of `add_camera`, so a test can make one take forever.
        self.block = threading.Event()
        self.block.set()

    def add_camera(self, camera: CameraSpec) -> None:
        self.block.wait(10.0)
        if self.refuse is not None:
            raise self.refuse
        if camera.camera_id in self.refuse_ids:
            self._remember(camera.camera_id)
            raise ConfigurationError(f"camera {camera.camera_id!r} is already running")
        self.added.append(camera)
        self._remember(camera.camera_id)

    def _remember(self, camera_id: str) -> None:
        """Put a camera in the health report, whoever it was that took the name.

        `setdefault`, so a health report the test wrote (a fleet's placement map) is not
        overwritten by this stand-in. What the mutation is for is minting: the *next* POST
        with no id must see this one as taken -- including when "the next POST" is this
        request's own retry after a refusal.
        """
        cameras = dict(self._health.get("cameras") or {})
        cameras.setdefault(camera_id, {"camera_id": camera_id, "state": "connecting"})
        self._health = {**self._health, "cameras": cameras}

    def remove_camera(self, camera_id: str, *, timeout_s: float = 5.0) -> bool:
        self.removed.append((camera_id, timeout_s))
        if self.refuse is not None:
            raise self.refuse
        return self.clean

    def drain(self, timeout_s: float = 20.0) -> int:
        self.drained.append(timeout_s)
        if self.refuse is not None:
            raise self.refuse
        return self.abandoned

    def health(self) -> dict[str, Any]:
        if isinstance(self._health, Exception):  # pragma: no cover - set by one test
            raise self._health
        return dict(self._health)

    def stats(self) -> dict[str, Any]:
        return {"cameras": len(self.added)}


class ThreadWatchingCameras:
    """A controller that records which thread each of its calls arrived on.

    ``manages_cameras`` is the load-bearing member. ``add_stream`` is ``async`` and reads it
    directly, so the thread that reads it *is* the event loop's for that request -- which is
    what lets this fake name the forbidden thread rather than assert against a string like
    "AnyIO worker thread" and pin an anyio implementation detail instead of the property.

    ``wedge_first_health`` is the failure the deadline exists for: one controller call that
    never comes back. On a fleet that is one shard paging in an engine, answered by a serial
    ``Health`` per shard.
    """

    def __init__(self, *, wedge_first_health: bool = False) -> None:
        self.calls: dict[str, list[str]] = {}
        self.added: list[CameraSpec] = []
        #: Set once the wedged ``health()`` has been entered, so a test knows it is safe to
        #: ask a second question and expect an answer.
        self.entered = threading.Event()
        self.release = threading.Event()
        self._wedge = wedge_first_health
        self._lock = threading.Lock()

    @property
    def loop_thread(self) -> str:
        """The thread the ``async`` handler itself ran on. Nothing blocking may touch it."""
        assert self.calls.get("manages_cameras"), "no POST or DELETE reached this controller"
        return self.calls["manages_cameras"][0]

    def _record(self, call: str) -> int:
        with self._lock:
            seen = self.calls.setdefault(call, [])
            seen.append(threading.current_thread().name)
            return len(seen)

    @property
    def manages_cameras(self) -> bool:
        self._record("manages_cameras")
        return True

    def add_camera(self, camera: CameraSpec) -> None:
        self._record("add_camera")
        with self._lock:
            self.added.append(camera)

    def remove_camera(self, camera_id: str, *, timeout_s: float = 5.0) -> bool:
        self._record("remove_camera")
        return True

    def drain(self, timeout_s: float = 20.0) -> int:
        self._record("drain")
        return 0

    def health(self) -> dict[str, Any]:
        if self._record("health") == 1 and self._wedge:
            self.entered.set()
            self.release.wait(10.0)
        with self._lock:
            placed = {camera.camera_id: {} for camera in self.added}
        return {"state": "running", "cameras": placed}

    def stats(self) -> dict[str, Any]:
        return {"cameras": len(self.added)}


def client_over(cameras: CameraController) -> TestClient:
    return TestClient(create_app(cameras=cameras))


@pytest.fixture()
def cameras() -> FakeCameras:
    return FakeCameras()


@pytest.fixture()
def client(cameras: FakeCameras):
    with client_over(cameras) as test_client:
        yield test_client


class TestAddingACamera:
    def test_a_posted_url_reaches_the_controller_and_answers_201(self, client, cameras) -> None:
        """201 Created, because a camera is a resource this call brought into existence."""
        response = client.post("/streams", json={"camera_id": "quay-1", "url": "rtsp://host"})

        assert response.status_code == 201, response.text
        assert cameras.added == [CameraSpec("quay-1", "rtsp://host", 0.0)]
        assert response.json()["camera_id"] == "quay-1"
        assert response.json()["url"] == "rtsp://host"

    def test_the_fps_is_carried_through_and_defaults_to_the_sources_own(
        self, client, cameras
    ) -> None:
        client.post("/streams", json={"url": "rtsp://host", "fps": 12.5})

        assert cameras.added[0].fps == 12.5
        client.post("/streams", json={"url": "rtsp://other"})
        assert cameras.added[1].fps == 0.0, "0.0 means whatever the source delivers"

    def test_a_request_with_no_id_is_named_by_the_same_helper_the_cli_uses(
        self, client, cameras
    ) -> None:
        """`cam-000`, `cam-001`, ... — `--inputs` mints the same names on the same box.

        Two spellings of "the next camera" would collide on a deployment that uses both
        doors, and two cameras under one id is a tracker keyed on nothing.
        """
        first = client.post("/streams", json={"url": "rtsp://a"}).json()
        second = client.post("/streams", json={"url": "rtsp://b"}).json()

        assert [first["camera_id"], second["camera_id"]] == ["cam-000", "cam-001"]

    def test_minting_skips_the_ids_that_are_already_taken(self) -> None:
        """Including ones this router never placed — `--inputs` names cameras too."""
        cameras = FakeCameras(
            health={"state": "running", "cameras": {"cam-000": {}, "cam-001": {}}}
        )
        with client_over(cameras) as client:
            body = client.post("/streams", json={"url": "rtsp://a"}).json()

        assert body["camera_id"] == "cam-002"

    def test_the_reply_says_where_the_camera_landed(self) -> None:
        """The interesting answer on a fleet: which shard is now reading it."""
        cameras = FakeCameras(health=FLEET_HEALTH)
        with client_over(cameras) as client:
            body = client.post("/streams", json={"camera_id": "cam-000", "url": "x"}).json()

        assert body["shard"] == 0
        assert body["state"] == "streaming"

    def test_an_unknown_field_is_refused_rather_than_silently_dropped(self, client) -> None:
        """A client that posts `{"uri": ...}` must not get a 201 and a camera reading nothing."""
        response = client.post("/streams", json={"url": "rtsp://host", "priority": "high"})

        assert response.status_code == 422
        assert "priority" in response.text


class TestWhyACameraIsRefused:
    def test_a_duplicate_id_is_400_because_a_retry_would_be_a_duplicate_too(self) -> None:
        cameras = FakeCameras(refuse=ConfigurationError("camera 'cam-000' is already running"))
        with client_over(cameras) as client:
            response = client.post("/streams", json={"camera_id": "cam-000", "url": "x"})

        assert response.status_code == 400
        assert "already running" in response.json()["detail"]

    def test_a_fleet_with_no_room_is_503_because_a_retry_might_land(self) -> None:
        """The whole reason `NoShardAvailableError` exists (`core/errors/launch.py`).

        As a `ConfigurationError` this was a 400, and a control plane that reads 400 as "my
        request is malformed" stops asking — for a condition that clears as soon as one shard
        finishes draining.
        """
        cameras = FakeCameras(
            refuse=NoShardAvailableError("cam-000", ["shard 0: draining", "shard 1: draining"])
        )
        with client_over(cameras) as client:
            response = client.post("/streams", json={"url": "rtsp://host"})

        assert response.status_code == 503
        assert "shard 1: draining" in response.json()["detail"]

    def test_a_runner_that_is_not_running_is_503(self) -> None:
        cameras = FakeCameras(refuse=ServerStateError("the fleet is not running"))
        with client_over(cameras) as client:
            response = client.post("/streams", json={"url": "rtsp://host"})

        assert response.status_code == 503

    def test_a_runner_that_manages_no_cameras_is_501_and_names_the_flag(self) -> None:
        """501, not 503: no amount of retrying gives this runner an ingest plane."""
        cameras = FakeCameras(manages_cameras=False)
        with client_over(cameras) as client:
            response = client.post("/streams", json={"url": "rtsp://host"})

        assert response.status_code == 501
        assert "--runner" in response.json()["detail"]
        assert cameras.added == [], "the camera reached a runner that cannot read it"

    def test_a_placement_that_outruns_the_deadline_is_504_not_a_held_worker(
        self, monkeypatch: pytest.MonkeyPatch, cameras
    ) -> None:
        """The bound on how long one request may hold a thread, and that it is real.

        `add_camera` runs in a worker thread with `abandon_on_cancel=True`; without that the
        cancel scope would wait for the very thread it is cancelling and this request would
        return 201 ten seconds later. The blocked call is released in the `finally` so the
        worker is not left parked for the rest of the session.
        """
        monkeypatch.setattr(streams_module, "_ADD_TIMEOUT_S", 0.05)
        cameras.block.clear()
        try:
            with client_over(cameras) as client:
                response = client.post("/streams", json={"url": "rtsp://host"})
            assert response.status_code == 504
            assert "GET /streams" in response.json()["detail"]
        finally:
            cameras.block.set()


class TestTwoPostsThatMintTheSameName:
    """The mint is read-then-act, so the report it read can be stale by the time it acts.

    Nothing here locks: the only thing that decides a name is unique is the controller, a
    layer down (an ingest manager's lock, or a shard's). So the race is not prevented, it is
    *answered* -- the loser of a mint it never asked for is re-minted against a fresh report
    rather than handed a 400 about an id its caller never supplied.
    """

    def test_a_minted_id_that_was_taken_in_the_race_is_minted_again(self) -> None:
        cameras = FakeCameras(refuse_ids={"cam-000"})
        with client_over(cameras) as client:
            response = client.post("/streams", json={"url": "rtsp://a"})

        assert response.status_code == 201, response.text
        assert response.json()["camera_id"] == "cam-001"
        assert [camera.camera_id for camera in cameras.added] == ["cam-001"]

    def test_an_id_the_caller_supplied_is_never_retried_under_another_name(self) -> None:
        """Their 400, and it will be a duplicate on the next try too.

        Retrying this one would place a camera the caller did not ask for, under a name they
        will not recognise, and answer 201 to a request that was wrong.
        """
        cameras = FakeCameras(refuse_ids={"cam-000"})
        with client_over(cameras) as client:
            response = client.post("/streams", json={"camera_id": "cam-000", "url": "x"})

        assert response.status_code == 400
        assert cameras.added == []

    def test_the_retry_happens_once_and_the_second_refusal_is_the_answer(self) -> None:
        """A loop here would be a POST that races forever against a busy deployment."""
        cameras = FakeCameras(refuse_ids={"cam-000", "cam-001"})
        with client_over(cameras) as client:
            response = client.post("/streams", json={"url": "rtsp://a"})

        assert response.status_code == 400
        assert "cam-001" in response.json()["detail"]
        assert cameras.added == []


class TestNothingBlockingRunsOnTheEventLoop:
    """The property that makes ``_ADD_TIMEOUT_S`` mean anything.

    ``add_stream`` is ``async``, so every blocking call it makes has to be pushed to a worker
    thread -- and ``health()`` is blocking work on every runner and *serial* blocking work on
    a fleet, one gRPC ``Health`` per shard with its own deadline. Called from the handler it
    would park the event loop for the sum of them, which stops every other request in the
    process, ``GET /health`` first among them: exactly the failure the deadline was added to
    prevent, reached by the route that enforces it.
    """

    def test_a_post_touches_the_controller_only_from_worker_threads(self) -> None:
        watcher = ThreadWatchingCameras()
        with client_over(watcher) as client:
            assert client.post("/streams", json={"url": "rtsp://a"}).status_code == 201

        assert watcher.calls["health"], "the report was never read"
        assert watcher.loop_thread not in watcher.calls["health"]
        assert watcher.loop_thread not in watcher.calls["add_camera"]

    def test_a_get_is_off_it_too_and_that_is_where_the_bar_comes_from(self) -> None:
        """A plain ``def`` route is the reference: FastAPI already runs those in the pool."""
        watcher = ThreadWatchingCameras()
        with client_over(watcher) as client:
            client.post("/streams", json={"url": "rtsp://a"})
            assert client.get("/health").status_code == 200

        assert watcher.loop_thread not in watcher.calls["health"]

    def test_a_wedged_report_is_a_504_and_the_next_request_still_answers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One controller call that never returns must cost one request, not the server.

        With ``health()`` back on the handler's own thread this fails twice over: the POST
        cannot time out at all -- ``fail_after`` cannot interrupt a blocking call on the
        thread the loop is running on -- and the ``GET /health`` beside it is not even
        dispatched until the wedge lets go.
        """
        monkeypatch.setattr(streams_module, "_ADD_TIMEOUT_S", 0.05)
        watcher = ThreadWatchingCameras(wedge_first_health=True)
        posted: dict[str, int] = {}

        with client_over(watcher) as client:

            def post() -> None:
                posted["status"] = client.post("/streams", json={"url": "rtsp://a"}).status_code

            caller = threading.Thread(target=post, name="posting")
            caller.start()
            try:
                assert watcher.entered.wait(5.0), "the POST never asked for a report"
                started = time.monotonic()
                assert client.get("/health").status_code == 200
                # Still wedged: the POST has to give up on its own deadline, not because the
                # test let the report go. That is the whole claim.
                caller.join(5.0)
                elapsed = time.monotonic() - started
                assert not caller.is_alive(), "the POST outlived its own deadline"
                # Both answers came back while the first report was still held. The bound is
                # loose on purpose -- what it has to tell apart is milliseconds from the ten
                # seconds the wedge holds for, and a shared box is allowed to be slow.
                assert elapsed < 2.0, "a request waited on the wedged report"
            finally:
                watcher.release.set()
                caller.join(10.0)

        assert posted == {"status": 504}


class TestRemovingACamera:
    def test_a_removed_camera_answers_200_and_clean(self, client, cameras) -> None:
        response = client.delete("/streams/cam-000")

        assert response.status_code == 200
        assert response.json() == {"clean": True}
        assert cameras.removed == [("cam-000", 5.0)]

    def test_an_abandoned_decoder_is_a_body_signal_and_not_a_5xx(self) -> None:
        """The camera is gone either way; the thread outliving its deadline is the news.

        A 500 would tell a control plane the removal failed. It would retry, get a 404, and
        conclude something worse than "one decoder is taking its time".
        """
        cameras = FakeCameras(clean=False)
        with client_over(cameras) as client:
            response = client.delete("/streams/cam-000")

        assert response.status_code == 200
        assert response.json() == {"clean": False}

    def test_an_unknown_camera_is_404_and_says_what_is_running(self) -> None:
        cameras = FakeCameras(
            refuse=ConfigurationError("no shard holds camera 'cam-009'; this fleet has ['a']")
        )
        with client_over(cameras) as client:
            response = client.delete("/streams/cam-009")

        assert response.status_code == 404
        assert "this fleet has" in response.json()["detail"]

    def test_a_stopped_fleet_is_503_rather_than_404(self) -> None:
        """ "I cannot ask" is not "there is no such camera"; only one of them is retryable."""
        cameras = FakeCameras(refuse=ServerStateError("the fleet is not running"))
        with client_over(cameras) as client:
            assert client.delete("/streams/cam-000").status_code == 503

    def test_a_runner_that_manages_no_cameras_is_501_here_too(self) -> None:
        cameras = FakeCameras(manages_cameras=False)
        with client_over(cameras) as client:
            assert client.delete("/streams/cam-000").status_code == 501
        assert cameras.removed == []


class TestListingWhatIsRunning:
    def test_a_fleet_listing_carries_the_shard_and_the_shards_own_state(self) -> None:
        """The launcher says *where*, the shard says *how it is doing*; the listing joins them.

        Without the join, `GET /streams` on a fleet answers a list of ids and the question an
        operator actually has — which camera is dark — goes unanswered.
        """
        cameras = FakeCameras(health=FLEET_HEALTH)
        with client_over(cameras) as client:
            body = client.get("/streams").json()

        assert body["streams"] == [
            {
                "camera_id": "cam-000",
                "url": "",
                "shard": 0,
                "pending": False,
                "state": "streaming",
            },
            {"camera_id": "cam-001", "url": "", "shard": 1, "pending": True, "state": ""},
        ]

    def test_a_pending_camera_is_reported_as_pending_and_not_as_running(self) -> None:
        """A reservation is not a camera being read, and reporting it as one hides a dark feed."""
        cameras = FakeCameras(health=FLEET_HEALTH)
        with client_over(cameras) as client:
            streams = client.get("/streams").json()["streams"]

        assert [s["pending"] for s in streams] == [False, True]

    def test_an_in_process_listing_uses_the_runners_own_shard_id(self) -> None:
        cameras = FakeCameras(health=INPROCESS_HEALTH)
        with client_over(cameras) as client:
            streams = client.get("/streams").json()["streams"]

        assert streams == [
            {
                "camera_id": "cam-000",
                "url": "",
                "shard": 0,
                "pending": False,
                "state": "streaming",
            }
        ]

    def test_cameras_is_an_alias_for_streams(self) -> None:
        """arch.md §2 draws this door as `GET /cameras`; both names answer."""
        cameras = FakeCameras(health=FLEET_HEALTH)
        with client_over(cameras) as client:
            assert client.get("/cameras").json() == client.get("/streams").json()

    def test_a_deployment_with_no_cameras_lists_nothing_and_is_not_an_error(
        self, client
    ) -> None:
        assert client.get("/streams").json() == {"streams": []}

    def test_a_controller_that_cannot_report_still_answers_a_listing(self) -> None:
        """A listing that raises is a listing that fails exactly when it is being read.

        The fleet's own report already names an unreachable shard rather than omitting it;
        this is the same decision one layer up.
        """
        cameras = FakeCameras()
        cameras._health = RuntimeError("the control channel is gone")  # type: ignore[assignment]
        with client_over(cameras) as client:
            assert client.get("/streams").json() == {"streams": []}


class TestDraining:
    def test_a_drain_reports_how_many_threads_were_abandoned(self) -> None:
        cameras = FakeCameras(abandoned=2)
        with client_over(cameras) as client:
            response = client.post("/streams/drain")

        assert response.status_code == 200
        assert response.json() == {"abandoned": 2}
        assert cameras.drained == [20.0]

    def test_a_clean_drain_is_zero(self, client, cameras) -> None:
        assert client.post("/streams/drain").json() == {"abandoned": 0}

    def test_the_timeout_is_the_callers_and_is_one_deadline_for_every_camera(
        self, client, cameras
    ) -> None:
        client.post("/streams/drain", params={"timeout_s": 1.5})

        assert cameras.drained == [1.5]

    def test_a_negative_timeout_is_refused_by_the_schema(self, client) -> None:
        assert client.post("/streams/drain", params={"timeout_s": -1}).status_code == 422

    def test_a_timeout_past_the_ceiling_is_refused_too(self, client, cameras) -> None:
        """This handler is synchronous, so the caller's deadline is how long it holds one of
        anyio's forty shared workers. Forty requests asking for a year each and the app stops
        answering anything, health check included -- a 422 is the cheaper failure."""
        assert client.post("/streams/drain", params={"timeout_s": 301}).status_code == 422
        assert client.post("/streams/drain", params={"timeout_s": 300}).status_code == 200
        assert cameras.drained == [300.0], "the ceiling itself is still a legal deadline"

    def test_a_stopped_runner_is_503(self) -> None:
        cameras = FakeCameras(refuse=ServerStateError("the fleet is not running"))
        with client_over(cameras) as client:
            assert client.post("/streams/drain").status_code == 503

    def test_a_runner_that_manages_no_cameras_is_501(self) -> None:
        cameras = FakeCameras(manages_cameras=False)
        with client_over(cameras) as client:
            assert client.post("/streams/drain").status_code == 501


class TestHealth:
    def test_a_stopped_controller_still_answers_200_with_the_state_in_the_body(self) -> None:
        """The endpoint that says *what is wrong* must not hide the body behind a 503.

        `/v2/health/live` and `/v2/health/ready` are the load balancer's, and they are the
        engine's (`routes.py`). This one is the operator's.
        """
        cameras = FakeCameras(health={"runner": "inprocess", "state": "stopped", "cameras": {}})
        with client_over(cameras) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["state"] == "stopped"

    def test_a_controller_that_cannot_report_answers_200_and_says_so(self) -> None:
        cameras = FakeCameras()
        cameras._health = RuntimeError("the control channel is gone")  # type: ignore[assignment]
        with client_over(cameras) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["state"] == "unknown"
        assert "control channel" in response.json()["detail"]


class TestWhatTheAppMounts:
    def test_a_camera_only_app_has_no_kserve_routes(self, client) -> None:
        """`shipinfer run --runner fleet` builds no engine in this process, so there is
        nothing behind `/v2/...` — and 404 is a better answer than a 500 per request."""
        assert client.post("/v2/models/ship_detector/infer", json={}).status_code == 404

    def test_an_app_with_nothing_behind_it_is_refused_at_construction(self) -> None:
        from shipinfer.core.errors import ConfigurationError as ConfigError

        with pytest.raises(ConfigError, match="neither an engine nor a camera controller"):
            create_app()
