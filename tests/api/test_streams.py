"""``/streams`` over a fake controller: every status code, and why each one is that code.

The router is handed a :class:`~shipinfer.api.streams.CameraController` — five members — so a
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

import logging
import threading
import time
from collections.abc import Container
from typing import Any, get_args

import pytest
from pydantic import ValidationError as SchemaValidationError

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from shipinfer.api import create_app
from shipinfer.api import streams as streams_module
from shipinfer.api.schemas import BAND_NAMES, BandName, StreamRequest
from shipinfer.api.streams import CameraController
from shipinfer.core.errors import (
    ConfigurationError,
    DuplicateCameraError,
    NoShardAvailableError,
    ServerStateError,
)
from shipinfer.core.request import Priority
from shipinfer.core.settings.ingest import CameraConfig
from shipinfer.launch.control import CameraSpec

#: A launcher's report: where each camera was placed, with the shard's own per-camera detail
#: one level down. `cam-001` is *pending* — placed, not yet accepted (`runners/fleet.py`).
#:
#: The band is one level down with the state and for the same reason: the launcher's own map
#: knows where a camera went and nothing about how it is being served, while the shard
#: resolved the lane and reports it (`runners/inprocess.py::_do_health`). It survives the
#: `google.protobuf.Struct` in `HealthReply.cameras` untouched, which is why no field had to
#: be added to `shard.proto` for a fleet listing to answer this.
FLEET_HEALTH: dict[str, Any] = {
    "runner": "fleet",
    "state": "running",
    "shards": {
        "0": {
            "placed": ["cam-000"],
            "state": "running",
            "cameras": {
                "cam-000": {
                    "camera_id": "cam-000",
                    "state": "streaming",
                    "priority": "high",
                }
            },
        },
        "1": {"placed": [], "state": "unreachable", "detail": "TimeoutError: deadline"},
    },
    "cameras": {"cam-000": {"shard": 0}, "cam-001": {"shard": 1, "pending": True}},
    "lost": {},
}

#: The same launcher after shard 1's process exited. `cam-002` was placed there and is not
#: being read; nothing will re-place it (ADR-018), and it stays in the camera map because a
#: missing entry would say it was never placed. Shard 1's own `placed` list excludes it, so
#: the report does not contradict itself.
FLEET_HEALTH_WITH_A_DEAD_SHARD: dict[str, Any] = {
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
    "cameras": {"cam-000": {"shard": 0}, "cam-002": {"shard": 1}},
    "lost": {"cam-002": 1},
}

#: An in-process runner's report: no shard map, the ingest state inline, the band inline too,
#: and the runner's own `shard_id` is the only shard there is.
INPROCESS_HEALTH: dict[str, Any] = {
    "runner": "inprocess",
    "state": "running",
    "shard_id": 0,
    "cameras": {
        "cam-000": {
            "camera_id": "cam-000",
            "state": "streaming",
            "fps": 19.9,
            "priority": "tracking_critical",
        }
    },
}

#: A controller that reports cameras and says nothing about their bands: a runner from before
#: the field existed, or one that resolves no lanes of its own. The listing must answer `null`
#: rather than a default -- `normal` here would be the server inventing the one fact an
#: operator came to check.
HEALTH_WITHOUT_BANDS: dict[str, Any] = {
    "runner": "inprocess",
    "state": "running",
    "shard_id": 0,
    "cameras": {"cam-000": {"camera_id": "cam-000", "state": "streaming"}},
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
        #: Every spec `add_camera` was *entered* with, refused ones included. `added` counts
        #: placements; this counts attempts, which is what the re-mint is bounded on.
        self.attempts: list[CameraSpec] = []
        self.removed: list[tuple[str, float]] = []
        self.drained: list[float] = []
        #: Held for the duration of `add_camera`, so a test can make one take forever.
        self.block = threading.Event()
        self.block.set()

    def add_camera(self, camera: CameraSpec) -> None:
        self.block.wait(10.0)
        self.attempts.append(camera)
        if self.refuse is not None:
            raise self.refuse
        if camera.camera_id in self.refuse_ids:
            self._remember(camera.camera_id)
            # The manager's own type (`ingest/manager.py`), not the base `ConfigurationError`:
            # a taken id is the one refusal a server-minted name may be re-minted after.
            raise DuplicateCameraError(f"camera {camera.camera_id!r} is already running")
        self.added.append(camera)
        self._remember(camera.camera_id)

    def _remember(self, camera_id: str) -> None:
        """Put a camera in the health report, whoever it was that took the name.

        `setdefault`, so a health report the test wrote (a fleet's placement map) is not
        overwritten by this stand-in. What the mutation is for is minting: the *next* POST
        with no id must see this one as taken -- including when "the next POST" is this
        request's own retry after a refusal.

        A controller whose `health()` raises has no report to put it in, so this is a no-op
        there -- the same thing the real failure means, and what lets a test set that up and
        still place a camera whose id the caller supplied.
        """
        if isinstance(self._health, Exception):
            return
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
        response = client.post("/streams", json={"url": "rtsp://host", "uri": "rtsp://host"})

        assert response.status_code == 422
        assert "uri" in response.text

    def test_the_posted_band_reaches_the_spec_the_controller_is_given(self) -> None:
        """A band is what the caller came for: the shard cannot work it out for itself.

        A fleet shard's ingest config is stripped (`runners/inprocess.py::_ingest`), so a
        camera posted here and placed by RPC has no configured table to resolve a lane from.
        If the name does not travel on the spec, it does not travel at all.
        """
        cameras = FakeCameras()
        with client_over(cameras) as client:
            response = client.post(
                "/streams", json={"url": "rtsp://host", "priority": "tracking_critical"}
            )

        assert response.status_code == 201, response.text
        assert cameras.added[0].priority is Priority.TRACKING_CRITICAL

    def test_a_camera_posted_without_a_band_leaves_the_choice_to_the_deployment(self) -> None:
        """`None`, not `normal`: "I have no opinion" and "put it in the middle lane" differ."""
        cameras = FakeCameras()
        with client_over(cameras) as client:
            assert client.post("/streams", json={"url": "rtsp://host"}).status_code == 201

        assert cameras.added[0].priority is None

    @pytest.mark.parametrize(
        ("spelling", "band"),
        [
            ("TRACKING_CRITICAL", Priority.TRACKING_CRITICAL),
            ("Tracking_Critical", Priority.TRACKING_CRITICAL),
            ("BackGround", Priority.BACKGROUND),
        ],
    )
    def test_the_band_name_is_matched_whatever_case_it_arrives_in(
        self, spelling: str, band: Priority
    ) -> None:
        """The name is a *member* name, and it is upper-case everywhere it is written.

        `Priority.TRACKING_CRITICAL` in Python, `TRACKING_CRITICAL` in a generated gRPC
        stub's enum, `tracking_critical` in the published schema -- one lane spelled three
        ways by the same deployment. Refusing the upper-cased one is a 422 for a client that
        named the right lane, so the model lower-cases a string before matching it.

        Asserted on the **resolved** `CameraSpec.priority` and over two different bands,
        because that is the property and the resolution has now moved: the router looks the
        member up with `Priority.parse`, the one rule `core` owns, rather than spelling
        `Priority[name.upper()]` a third time in the tree. A case that only covered
        `TRACKING_CRITICAL` would also pass against `Priority(0)`, which is not the same
        function of the input at all.
        """
        cameras = FakeCameras()
        with client_over(cameras) as client:
            response = client.post(
                "/streams", json={"url": "rtsp://host", "priority": spelling}
            )

        assert response.status_code == 201, response.text
        assert cameras.added[0].priority is band

    @pytest.mark.parametrize("band", ["urgent", "TRACKING-CRITICAL", "", 0, 2])
    def test_a_band_the_server_does_not_know_is_422_rather_than_a_default(
        self, client, band: object
    ) -> None:
        """Including the *numbers*, and that is the case worth having.

        `Priority.TRACKING_CRITICAL` is 0, so `{"priority": 0}` is a client that meant "unset"
        asking for the highest lane on the deployment. Names only, so the question cannot be
        asked ambiguously.

        Every one of these survives the case normalisation above, which is the point of
        normalising only a `str` and only its case: `0` is an `int` and reaches the literal
        untouched, and `tracking-critical` is not a member name in any case.
        """
        response = client.post("/streams", json={"url": "rtsp://host", "priority": band})

        assert response.status_code == 422
        assert "priority" in response.text

    def test_both_doors_refuse_an_unknown_band_in_the_same_words(self, client) -> None:
        """A camera arrives here or in `ingest.cameras`, and one lane vocabulary owns both.

        The two doors used to disagree about more than wording: this one had always matched
        band *names*, while `CameraConfig.priority` was a bare `Priority` annotation whose
        `IntEnum` coercion took the numbers only — so `priority: tracking_critical`, the
        spelling three docstrings in `src/` tell an operator to write, was a start-up
        failure. Both now call `Priority.parse`, and the cheapest way to keep them calling
        the *same* rule is to assert that a band neither knows is refused in one set of
        words. Reimplementing the check here would pass this file's own tests forever while
        the config door drifted.
        """
        with pytest.raises(SchemaValidationError) as excinfo:
            CameraConfig(camera_id="gate", uri="rtsp://x", priority="urgent")
        configured = excinfo.value.errors()[0]["msg"]

        response = client.post("/streams", json={"url": "rtsp://host", "priority": "urgent"})

        assert response.status_code == 422
        assert response.json()["detail"][0]["msg"] == configured
        assert "tracking_critical" in configured, configured

    def test_the_published_schema_offers_the_names_the_validator_accepts(self, client) -> None:
        """`/openapi.json` and the 422 above must describe the same field.

        `Priority` is an `IntEnum`, so typing the wire field as `Priority | None` had FastAPI
        publish `{"enum": [0, 1, 2, 3], "type": "integer"}` -- while every one of those four
        integers was refused. A generated client did exactly what the document told it to and
        got a 422 it could not read its way out of, which is worse than no document. The wire
        type is the band *names*, and the schema says so.
        """
        schema = client.get("/openapi.json").json()
        field = schema["components"]["schemas"]["StreamRequest"]["properties"]["priority"]
        offered = [option for option in field["anyOf"] if option.get("type") != "null"]

        assert offered == [
            {"enum": ["tracking_critical", "high", "normal", "background"], "type": "string"}
        ], field

    def test_the_offered_names_are_every_band_and_only_bands(self) -> None:
        """The schema is derived from `Priority`, so a fifth band cannot be silently unposted.

        Spelled-out strings would have to be edited a second time, and the failure of not
        doing so is invisible: the new lane simply cannot be asked for over HTTP.
        """
        assert tuple(band.name.lower() for band in Priority) == BAND_NAMES
        assert get_args(BandName) == BAND_NAMES

    def test_a_finite_video_can_be_asked_to_stop_at_its_end(self, client, cameras) -> None:
        """`loop: false` is `--no-loop` over HTTP, and it has to reach the spec to mean it.

        `CameraSpec.loop` decides whether *this* camera ever ends, so a client that posts a
        finite file and cannot say "once" gets it replayed forever.
        """
        assert (
            client.post("/streams", json={"url": "clip.mp4", "loop": False}).status_code == 201
        )

        assert cameras.added[0] == CameraSpec("cam-000", "clip.mp4", 0.0, loop=False)

    def test_a_camera_loops_by_default_because_a_live_source_never_ends(
        self, client, cameras
    ) -> None:
        client.post("/streams", json={"url": "rtsp://host"})

        assert cameras.added[0].loop is True


class TestARequestTheSchemaCanRefuseOnItsOwn:
    """422 before the handler runs, which is the only way both runners answer the same thing.

    `CameraSpec` validates nothing, so an unconstrained field was first inspected by
    `CameraConfig` a layer *below* the router -- and its refusal is a pydantic
    `ValidationError`, a `ValueError` and not a `ShipInferError`, so `add_stream`'s typed
    mapping never saw it. In process that was a 500; on a fleet the shard refused, the
    launcher collected a refusal from every shard and raised `NoShardAvailableError`, and the
    caller got a **retryable 503** for a request that can never succeed. Declaring the
    constraint on `StreamRequest` moves the answer ahead of the handler, where FastAPI names
    the field and neither runner is involved.
    """

    def test_an_empty_url_names_the_field_instead_of_500ing(self, client, cameras) -> None:
        response = client.post("/streams", json={"url": ""})

        assert response.status_code == 422, response.text
        assert "url" in response.text
        assert cameras.attempts == [], "a camera with no source reached the controller"

    def test_a_whitespace_only_url_is_the_same_mistake_and_the_same_answer(
        self, client, cameras
    ) -> None:
        """What a shell renders when the variable it interpolated was unset."""
        response = client.post("/streams", json={"url": "   "})

        assert response.status_code == 422, response.text
        assert "url must not be empty" in response.text
        assert cameras.attempts == []

    def test_a_negative_fps_is_refused_at_the_door_like_camera_config_refuses_it(
        self, client, cameras
    ) -> None:
        response = client.post("/streams", json={"url": "x.mp4", "fps": -1})

        assert response.status_code == 422, response.text
        assert "fps" in response.text
        assert cameras.attempts == []

    def test_a_camera_id_with_a_space_in_it_is_refused_here_and_not_by_a_shard(
        self, client, cameras
    ) -> None:
        """The third field, and the one this class's whole argument was written for.

        `CameraConfig` rejects a whitespace-carrying id (`core/settings/ingest.py`), so before
        the schema said so this body was a 400 in process -- the router's `ValueError` net --
        and a **retryable 503** on a fleet, because every shard refused separately and
        `NoShardAvailableError` is all a launcher can report about that. Neither runner is
        involved now: FastAPI names the field.
        """
        response = client.post("/streams", json={"camera_id": "quay 1", "url": "x.mp4"})

        assert response.status_code == 422, response.text
        assert "camera_id" in response.text
        assert "whitespace" in response.text
        assert cameras.attempts == [], "an unusable id reached the controller"

    def test_a_blank_camera_id_is_not_a_request_to_mint_one(self, client, cameras) -> None:
        """`"   "` is truthy, so it was never minted over -- it was carried down and refused.

        The distinction that matters: `""` asks the server for a name, `"   "` is a name the
        server cannot use, and the two must not answer the same way.
        """
        response = client.post("/streams", json={"camera_id": "  ", "url": "x.mp4"})

        assert response.status_code == 422, response.text
        assert "camera_id" in response.text
        assert cameras.attempts == []

    def test_an_empty_camera_id_still_means_name_it_for_me(self, client, cameras) -> None:
        """The one value the validator lets through, because it is not an id at all."""
        response = client.post("/streams", json={"camera_id": "", "url": "x.mp4"})

        assert response.status_code == 201, response.text
        assert response.json()["camera_id"] == "cam-000"
        assert [camera.camera_id for camera in cameras.added] == ["cam-000"]

    def test_the_door_and_the_record_apply_one_rule_rather_than_two_copies(self) -> None:
        """A drift guard, not a duplicate: it is `CameraConfig`'s own predicate under test.

        The two checks are three layers apart -- an HTTP schema and a settings record -- and
        a mirrored rule disagrees the moment either is edited. This asserts they are the same
        function, which is why `usable_camera_id` was lifted out of the validator at all.
        """
        from shipinfer.core.settings.ingest import CameraConfig

        for bad in ("quay 1", "  ", "\tcam", "cam-1 "):
            with pytest.raises(ValueError):
                CameraConfig(camera_id=bad, uri="x.mp4")
            with pytest.raises(ValueError):
                StreamRequest(camera_id=bad, url="x.mp4")
        assert StreamRequest(camera_id="quay-1", url="x.mp4").camera_id == "quay-1"

    def test_a_value_the_schema_did_not_constrain_is_400_and_not_500(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The net under everything else the settings tree validates.

        `StreamRequest` cannot mirror all twenty of `CameraConfig`'s fields, so a value it
        does not know about is still refused a layer down with a `ValueError`. That is the
        caller's mistake and a retry sends it again, which makes it a 400 -- not the 500 that
        reads like a ShipInfer bug in the operator's deployment log.

        The net is wider than the posted values, so the same clause also relabels a genuine
        internal `ValueError` as the caller's mistake; the traceback that tells the two apart
        has to be written by this handler, because starlette answers an `HTTPException`
        without logging anything.
        """
        cameras = FakeCameras(refuse=ValueError("codec 'h266' is not supported"))
        with (
            caplog.at_level(logging.ERROR, logger="shipinfer.api"),
            client_over(cameras) as client,
        ):
            response = client.post("/streams", json={"url": "rtsp://host"})

        assert response.status_code == 400, response.text
        assert "h266" in response.json()["detail"]
        refusals = [r for r in caplog.records if "did not constrain" in r.getMessage()]
        assert len(refusals) == 1 and refusals[0].exc_info is not None, caplog.text
        assert "h266" in caplog.text


class TestWhyACameraIsRefused:
    def test_a_duplicate_id_is_400_because_a_retry_would_be_a_duplicate_too(self) -> None:
        cameras = FakeCameras(
            refuse=DuplicateCameraError("camera 'cam-000' is already running")
        )
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

    def test_a_controller_that_cannot_say_which_ids_are_taken_is_503_not_a_minted_400(
        self,
    ) -> None:
        """A control-plane fault must not be reported as the caller's mistake.

        `_health` is lenient for every *read*, which is right: a listing that 500s because one
        shard is unreachable is useless exactly when it is wanted. But `_mint` acts on that
        report -- it hands out the lowest free `cam-<n>` -- and the lenient stand-in carries no
        `cameras` key at all, which does not mean "none are running". So a deployment with
        fifty cameras up and an unreachable control plane minted `cam-000`, the controller
        refused the duplicate, and the caller got a **400 naming an id they never supplied**:
        terminal, so a well-behaved client stops retrying something that would work in a
        minute. 503 says what is true -- the server could not find out.
        """
        cameras = FakeCameras(refuse_ids=frozenset({"cam-000"}))
        cameras._health = RuntimeError("the control channel is gone")  # type: ignore[assignment]
        with client_over(cameras) as client:
            response = client.post("/streams", json={"url": "rtsp://host"})

        assert response.status_code == 503, response.text
        assert "could not report which ids are in use" in response.json()["detail"]
        assert cameras.attempts == [], "it placed a camera under a name it guessed"

    def test_a_caller_who_names_the_camera_is_placed_even_so(self) -> None:
        """The other half of the same decision: strictness is on the *mint*, not the route.

        Nothing about this request depends on what the controller could not say, so refusing
        it would be inventing a failure. This is what keeps the answer above from being a
        blanket "health is down, nothing works".
        """
        cameras = FakeCameras()
        cameras._health = RuntimeError("the control channel is gone")  # type: ignore[assignment]
        with client_over(cameras) as client:
            response = client.post("/streams", json={"camera_id": "quay-1", "url": "x"})

        assert response.status_code == 201, response.text
        assert [camera.camera_id for camera in cameras.added] == ["quay-1"]

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

    def test_a_refusal_that_is_not_a_duplicate_is_not_retried_under_another_name(self) -> None:
        """One placement attempt, not two: re-minting fixes a taken name and nothing else.

        `InprocessRunner.add_camera` raises a plain `ConfigurationError` when the chain names
        a source that is not registered, and that is a 400 whatever the camera is called. On
        a bare `except ConfigurationError` an id-less POST did the entire add twice -- a
        second health report and a second placement -- before answering exactly the same
        thing.
        """
        cameras = FakeCameras(refuse=ConfigurationError("source 'rtsp' is not registered"))
        with client_over(cameras) as client:
            response = client.post("/streams", json={"url": "rtsp://host"})

        assert response.status_code == 400
        assert "not registered" in response.json()["detail"]
        assert len(cameras.attempts) == 1, "an unrelated refusal was retried"

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
                "lost": False,
                "priority": "high",
            },
            {
                "camera_id": "cam-001",
                "url": "",
                "shard": 1,
                "pending": True,
                "state": "",
                "lost": False,
                "priority": None,
            },
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
                "lost": False,
                "priority": "tracking_critical",
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


class TestTheListingSaysWhichLaneACameraLandedIn:
    """`priority` on the way out, in the vocabulary it went in on (#75 note 4).

    A band is the one thing about a camera that nothing else could confirm. It arrives on the
    placement (`POST /streams {"priority": ...}`) or in `ingest.cameras`, and on a fleet shard
    the second of those is stripped (#71) -- so before this field an operator who wrote
    `tracking_critical` had no way to find out whether it took effect, short of reading the
    shard's logs. The listing now answers it, and answers `null` rather than guessing when the
    controller has not said.
    """

    def test_an_in_process_listing_names_the_lane_the_camera_landed_in(self) -> None:
        """The runner resolved it (`_priority_for`), so the report is the lane, not the ask."""
        cameras = FakeCameras(health=INPROCESS_HEALTH)
        with client_over(cameras) as client:
            streams = client.get("/streams").json()["streams"]

        assert streams[0]["priority"] == "tracking_critical"

    def test_a_fleet_listing_reads_the_band_off_the_shard_that_resolved_it(self) -> None:
        """The same join `state` uses, one level down into the shard's own report.

        The launcher's camera map carries `{"shard": n}` and nothing else -- it knows where a
        camera went, not what its shard decided about it -- so a band read only from that map
        would be `null` for every camera on a fleet, which is the deployment that needs it.
        """
        cameras = FakeCameras(health=FLEET_HEALTH)
        with client_over(cameras) as client:
            streams = client.get("/streams").json()["streams"]

        assert [(s["camera_id"], s["priority"]) for s in streams] == [
            ("cam-000", "high"),
            ("cam-001", None),
        ], "cam-001 is on the shard that did not answer"

    def test_a_controller_that_reports_no_band_answers_null_and_not_a_default(self) -> None:
        """`null` and `normal` are different claims, and only one of them is true here.

        A default would read as "your camera is in the middle lane" for a shard that never
        said so -- an answer indistinguishable from the real thing, given to the operator who
        is checking precisely because they suspect the band did not apply.
        """
        cameras = FakeCameras(health=HEALTH_WITHOUT_BANDS)
        with client_over(cameras) as client:
            streams = client.get("/streams").json()["streams"]

        assert streams[0]["priority"] is None
        assert streams[0]["state"] == "streaming", "the rest of the entry still reports"

    def test_a_band_this_deployment_does_not_have_is_null_and_not_a_broken_listing(
        self,
    ) -> None:
        """A launcher's report is assembled from replies written by other processes.

        They cross a `google.protobuf.Struct`, which validates nothing about a band name, so a
        shard from another release is a real source of a word this enum does not have. One
        field of one camera degrades to `null`; a listing that raised would take the whole
        report down at the moment it is being read to find out what is wrong.
        """
        cameras = FakeCameras(
            health={
                "state": "running",
                "shard_id": 0,
                "cameras": {"cam-000": {"state": "streaming", "priority": "urgent"}},
            }
        )
        with client_over(cameras) as client:
            response = client.get("/streams")

        assert response.status_code == 200, response.text
        assert response.json()["streams"][0] == {
            "camera_id": "cam-000",
            "url": "",
            "shard": 0,
            "pending": False,
            "state": "streaming",
            "lost": False,
            "priority": None,
        }

    def test_a_201_carries_the_band_the_controller_resolved(self) -> None:
        """The POST answers from the same report the GET reads, so the two cannot disagree."""
        cameras = FakeCameras(health=FLEET_HEALTH)
        with client_over(cameras) as client:
            body = client.post("/streams", json={"camera_id": "cam-000", "url": "x"}).json()

        assert body["priority"] == "high"

    def test_a_201_for_a_camera_the_report_does_not_mention_echoes_nothing(self) -> None:
        """Not the posted band: the field reports what was resolved, not what was asked for.

        Echoing the request would make a 201 confirm a lane no runner has agreed to -- and it
        would be the one case where this field cannot be trusted, which is worse than a
        `null` an operator can follow up with a `GET`.
        """
        cameras = FakeCameras(health={"state": "running", "cameras": {}})
        with client_over(cameras) as client:
            body = client.post(
                "/streams", json={"url": "rtsp://host", "priority": "tracking_critical"}
            ).json()

        assert cameras.added[0].priority is Priority.TRACKING_CRITICAL
        assert body["priority"] is None

    def test_the_published_schema_offers_the_same_names_the_request_takes(self) -> None:
        """One vocabulary in both directions: a client can post back what it was told.

        Typed `BandName | None` and not `Priority | None` for the reason
        `StreamRequest.priority` is (`test_the_published_schema_offers_the_names_the_validator
        _accepts`): `Priority` is an `IntEnum`, and a generated client would be handed
        integers this API neither sends nor accepts.
        """
        cameras = FakeCameras(health=INPROCESS_HEALTH)
        with client_over(cameras) as client:
            schema = client.get("/openapi.json").json()

        field = schema["components"]["schemas"]["StreamInfo"]["properties"]["priority"]
        assert field["anyOf"] == [
            {"enum": list(BAND_NAMES), "type": "string"},
            {"type": "null"},
        ], field


class TestACameraWhoseShardHasGone:
    """`lost` is the launcher's word for "the process holding this camera exited".

    Terminal, unlike `unreachable`: a wedged shard may answer the next probe, but nothing
    respawns a dead one and nothing re-places its cameras (ADR-018). The distinction is what
    tells an operator to remove-and-re-add rather than to wait.
    """

    def test_a_listing_marks_the_camera_lost(self) -> None:
        cameras = FakeCameras(health=FLEET_HEALTH_WITH_A_DEAD_SHARD)
        with client_over(cameras) as client:
            streams = client.get("/streams").json()["streams"]

        assert [(s["camera_id"], s["lost"]) for s in streams] == [
            ("cam-000", False),
            ("cam-002", True),
        ]

    def test_a_lost_camera_is_listed_rather_than_omitted(self) -> None:
        """An absent entry says "no such camera", which is the wrong answer to "where did my
        camera go" — and it is what a launcher that deleted the placement would give."""
        cameras = FakeCameras(health=FLEET_HEALTH_WITH_A_DEAD_SHARD)
        with client_over(cameras) as client:
            streams = client.get("/streams").json()["streams"]

        assert [s["camera_id"] for s in streams] == ["cam-000", "cam-002"]
        assert [s["shard"] for s in streams] == [0, 1], "a lost camera still says where it was"

    def test_health_carries_the_whole_map(self) -> None:
        """The route passes the runner's report through, so an operator gets the shard too —
        one dead process makes a dozen cameras lost at once and the shard id is what groups
        them. The view lags a death by up to the launcher's supervision poll."""
        cameras = FakeCameras(health=FLEET_HEALTH_WITH_A_DEAD_SHARD)
        with client_over(cameras) as client:
            body = client.get("/health").json()

        assert body["lost"] == {"cam-002": 1}

    def test_nothing_is_lost_on_a_healthy_fleet(self) -> None:
        cameras = FakeCameras(health=FLEET_HEALTH)
        with client_over(cameras) as client:
            body = client.get("/streams").json()

        assert body["streams"] and not any(s["lost"] for s in body["streams"])
        assert client.get("/health").json()["lost"] == {}

    def test_a_runner_that_reports_no_loss_at_all_is_read_as_none(self) -> None:
        """An in-process runner has no shards to lose and writes no `lost` key; the listing
        reads the absence as "nothing lost" rather than raising over the missing level."""
        cameras = FakeCameras(health=INPROCESS_HEALTH)
        with client_over(cameras) as client:
            assert client.get("/streams").json()["streams"][0]["lost"] is False

    def test_a_camera_placed_right_now_is_not_lost(self) -> None:
        """201 answers from the report, and a camera that was just accepted is on a shard
        that was alive a moment ago."""
        cameras = FakeCameras(health=FLEET_HEALTH_WITH_A_DEAD_SHARD)
        with client_over(cameras) as client:
            body = client.post("/streams", json={"url": "rtsp://quay"}).json()

        assert body["lost"] is False


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
