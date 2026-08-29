"""``POST /streams`` on a real runner: a URL goes in one end, frames come out the other.

The other half of ``tests/api/test_streams.py``. That file pins the *mapping* over a fake
controller; this one pins that the protocol is satisfied by the thing that will actually be
behind it — a real :class:`~shipinfer.runners.inprocess.InprocessRunner`, walking a real
chain, with a real ingest actor per camera. Nothing here is stubbed except the frame source,
which is injected through the runner's own ``source_factory`` seam so the whole file runs
offline: no GPU, no GStreamer, no camera (``tests/runners/test_camera_lifecycle.py`` makes
the same trade and explains why the double is copied rather than shared — the ``tests/``
directories are not packages).

The claim being tested is the end-to-end one B3 exists for: an HTTP request starts a decoder
thread, its frames cross a bounded per-camera lane and a worker thread, and they arrive at
the far end of the chain carrying the ``(camera_id, frame_id)`` tag the caller's ``camera_id``
named. And the reverse: ``DELETE`` stops it, and the frames stop.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any, ClassVar, NamedTuple

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from shipinfer.api import create_app
from shipinfer.core.settings import ServerSettings
from shipinfer.core.settings.ingest import CameraConfig
from shipinfer.ingest.base import FrameSource
from shipinfer.ingest.frame import FrameCounter
from shipinfer.runners.inprocess import InprocessRunner
from shipinfer.topology import ChainSpec, Topology

#: The shortest chain that is a chain: a file source and a sink that counts. What is under
#: test here is the HTTP surface in front of a runner, and every model slot in between would
#: be a model repository this file has no use for.
CHAIN = """
name: streamed
elements:
  decode: {impl: replay}
  output: {impl: none, params: {keep_last: 64}}
"""

HEIGHT, WIDTH = 4, 6


class PacedSource(FrameSource):
    """A source that delivers a fixed number of frames, slowly enough to be interrupted.

    ``frames`` is large and ``pause_s`` small: the point is a camera that is *still reading*
    when the test deletes it, which is what makes "the frames stopped" mean the removal
    stopped them rather than the clip ending. The sleep is what keeps that from being a hot
    loop on a shared box — nothing paces this actor but its source.
    """

    name: ClassVar[str] = "paced"

    def __init__(
        self,
        config: CameraConfig,
        counter: FrameCounter | None = None,
        *,
        settings: Any = None,
        frames: int = 4,
        pause_s: float = 0.0,
    ) -> None:
        super().__init__(config, counter, settings=settings)
        self.frames = frames
        self.pause_s = pause_s
        self.index = 0

    @property
    def is_exhausted(self) -> bool:
        return self.index >= self.frames

    def _do_open(self) -> None:
        self._set_format(HEIGHT, WIDTH, self.config.fps or 20.0)

    def _do_read(self) -> np.ndarray | None:
        index = self.index
        self.index += 1
        if index >= self.frames:
            return None
        if self.pause_s:
            time.sleep(self.pause_s)
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = (index + 1) % 256
        return frame

    def _do_close(self) -> None:
        return None


def paced(frames: int = 4, pause_s: float = 0.0):
    def factory(config: CameraConfig, counter: FrameCounter) -> PacedSource:
        return PacedSource(config, counter, frames=frames, pause_s=pause_s)

    return factory


def settings() -> ServerSettings:
    return ServerSettings(
        pipeline={"workers": 1, "queue_capacity": 64},
        ingest={
            "read_timeout_ms": 50,
            "open_timeout_ms": 50,
            "empty_read_sleep_ms": 0,
            "empty_reads_before_reconnect": 2,
            "reconnect_initial_ms": 10,
            "reconnect_max_ms": 50,
            "reconnect_jitter": 0.0,
        },
    )


def until(predicate, timeout_s: float = 10.0, poll_s: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


class Deployment(NamedTuple):
    """What ``shipinfer run --http`` composes: a runner, the chain it walks, a client."""

    runner: InprocessRunner
    chain: Topology
    client: TestClient

    def sink(self):
        """The ``null`` sink behind the chain's ``output`` element, holding what it published."""
        return self.chain.node("output").element.sink


@pytest.fixture()
def deployment() -> Iterator:
    """A started runner behind a TestClient — the shape ``shipinfer run --http`` composes.

    The runner is always stopped, whatever the test did: an actor thread that outlives its
    test is a decoder still publishing into a closed queue, and it takes the *next* test's
    assertions with it.
    """
    made: list[InprocessRunner] = []

    def _make(frames: int = 4, pause_s: float = 0.0, start: bool = True) -> Deployment:
        chain = Topology.from_spec(ChainSpec.from_yaml(CHAIN))
        runner = InprocessRunner(
            chain, settings=settings(), source_factory=paced(frames, pause_s)
        )
        made.append(runner)
        if start:
            runner.start()
        return Deployment(runner, chain, TestClient(create_app(cameras=runner)))

    yield _make
    for runner in made:
        runner.stop(timeout_s=5.0)


class TestAPostedUrlIsRead:
    def test_the_frames_of_a_posted_camera_reach_the_end_of_the_chain(self, deployment) -> None:
        """The end-to-end claim: HTTP in, frames out, tagged with the caller's camera id.

        Four frames, four items at the sink, ``frame_id`` counting from zero — the tag ADR-002
        says must survive every hand-over, asserted at the far end of a chain the frames
        crossed through a queue and a worker thread, having entered over a socket.
        """
        streamed = deployment(frames=4)
        with streamed.client as client:
            response = client.post(
                "/streams", json={"camera_id": "quay-1", "url": "injected://quay-1"}
            )

            assert response.status_code == 201, response.text
            assert until(lambda: streamed.sink().emitted == 4), streamed.sink().stats()

        assert [(e.camera_id, e.frame_id) for e in streamed.sink().events()] == [
            ("quay-1", 0),
            ("quay-1", 1),
            ("quay-1", 2),
            ("quay-1", 3),
        ]

    def test_a_camera_with_no_id_is_named_and_the_name_is_what_tags_its_frames(
        self, deployment
    ) -> None:
        """The minted id is not decoration: it is the fairness key every frame carries."""
        streamed = deployment(frames=2)
        with streamed.client as client:
            body = client.post("/streams", json={"url": "injected://a"}).json()

            assert body["camera_id"] == "cam-000"
            assert until(lambda: streamed.sink().emitted == 2)

        assert {e.camera_id for e in streamed.sink().events()} == {"cam-000"}

    def test_the_listing_shows_the_camera_the_runner_is_reading(self, deployment) -> None:
        """`GET /streams` reads the runner's own health, so it reports ingest reality.

        ``streaming`` while it reads, ``exhausted`` once the clip ends — both come from the
        camera actor, not from anything the router remembered.
        """
        streamed = deployment(frames=200, pause_s=0.002)
        with streamed.client as client:
            client.post("/streams", json={"camera_id": "quay-1", "url": "injected://q"})

            assert until(lambda: _state(client, "quay-1") == "streaming"), client.get(
                "/streams"
            ).json()
            listed = client.get("/streams").json()["streams"]

        assert listed[0]["camera_id"] == "quay-1"
        assert listed[0]["shard"] == 0, "the in-process runner is its own shard"

    def test_an_empty_url_is_422_here_too_and_no_camera_is_started(self, deployment) -> None:
        """The failure the schema constraints exist for, against the runner that hit it.

        Over the fake controller an unconstrained ``url`` merely reached the double; here it
        reached ``CameraConfig``, whose refusal is a pydantic ``ValidationError`` -- a
        ``ValueError``, not a ``ShipInferError`` -- and fell past ``add_stream``'s typed
        mapping into a **500**. The same body over ``--runner fleet`` was refused by every
        shard and answered **503**, which a load balancer retries forever. 422 naming the
        field is the answer both runners now give, and it is FastAPI's, before the handler
        runs at all.
        """
        streamed = deployment(frames=1)
        with streamed.client as client:
            response = client.post("/streams", json={"url": ""})

            assert response.status_code == 422, response.text
            assert "url" in response.text
            assert client.get("/streams").json() == {"streams": []}
        assert streamed.runner.cameras == ()

    def test_a_second_camera_with_the_same_id_is_400(self, deployment) -> None:
        """The ingest manager's duplicate refusal, reaching the client as the caller's error."""
        streamed = deployment(frames=200, pause_s=0.002)
        with streamed.client as client:
            client.post("/streams", json={"camera_id": "quay-1", "url": "injected://q"})
            response = client.post("/streams", json={"camera_id": "quay-1", "url": "x"})

        assert response.status_code == 400
        assert "quay-1" in response.json()["detail"]


class TestTheListingConfirmsTheBand:
    """The round trip #75 asked for: post a lane, read back the lane that took effect.

    Over the real runner rather than a fake controller, because everything interesting here is
    plumbing between two files -- the band has to survive `POST` -> `CameraSpec` ->
    `InprocessRunner._priority_for` -> the health report -> `GET /streams`, and a double for
    the middle of that would assert the test's own opinion of it.
    """

    def test_a_posted_band_is_what_the_listing_reports(self, deployment) -> None:
        """`tracking_critical` for the camera that asked, `normal` for the one that did not.

        The second half is what makes the first mean anything: a listing that reported the
        posted band would pass on `quay-hot` alone, and a runner that ignored the band
        entirely would too, since `normal` is what every camera used to get (#71).
        """
        streamed = deployment(frames=200, pause_s=0.002)
        with streamed.client as client:
            client.post(
                "/streams",
                json={
                    "camera_id": "quay-hot",
                    "url": "injected://hot",
                    "priority": "tracking_critical",
                },
            )
            client.post("/streams", json={"camera_id": "quay-cold", "url": "injected://cold"})

            assert until(lambda: len(client.get("/streams").json()["streams"]) == 2)
            bands = _bands(client)

        assert bands == {"quay-hot": "tracking_critical", "quay-cold": "normal"}

    def test_the_band_is_reported_before_the_camera_has_delivered_a_frame(
        self, deployment
    ) -> None:
        """It is resolved at placement, so it does not wait on a decoder that may never open.

        A camera pointed at a dead switch is exactly when an operator asks which lane it is
        in, and a band that only appeared once frames arrived would be absent for every
        camera worth asking about.
        """
        streamed = deployment(frames=0)
        with streamed.client as client:
            body = client.post(
                "/streams",
                json={
                    "camera_id": "quay-dark",
                    "url": "injected://dark",
                    "priority": "background",
                },
            )

            assert body.status_code == 201, body.text
            assert _bands(client) == {"quay-dark": "background"}

    def test_a_re_added_camera_reports_the_band_it_was_re_added_with(self, deployment) -> None:
        """The listing follows the runner's resolution rather than remembering the first post.

        `DELETE` then `POST` with no band is a camera handed back to the deployment's own
        choice (`runners/inprocess.py::_admit_at`), and a listing that answered from anything
        it had cached would still be reporting the dead placement's lane.
        """
        streamed = deployment(frames=200, pause_s=0.002)
        with streamed.client as client:
            client.post(
                "/streams",
                json={
                    "camera_id": "quay-1",
                    "url": "injected://q",
                    "priority": "tracking_critical",
                },
            )
            assert until(lambda: _bands(client) == {"quay-1": "tracking_critical"})

            client.delete("/streams/quay-1")
            client.post("/streams", json={"camera_id": "quay-1", "url": "injected://q"})

            assert _bands(client) == {"quay-1": "normal"}


class TestDeletingAStream:
    def test_a_deleted_camera_stops_being_read_and_leaves_the_listing(self, deployment) -> None:
        """The mirror image, and the reason the listing is not a cache.

        The source has 200 frames left when the DELETE lands, so a sink that stops growing is
        the removal having stopped the decoder rather than the clip having ended.
        """
        streamed = deployment(frames=200, pause_s=0.002)
        with streamed.client as client:
            client.post("/streams", json={"camera_id": "quay-1", "url": "injected://q"})
            assert until(lambda: streamed.sink().emitted > 0)

            response = client.delete("/streams/quay-1")
            assert response.status_code == 200
            assert response.json() == {"clean": True}

            settled = streamed.sink().emitted
            # The sleep is not what makes this deterministic, and it should not be read as
            # one: `clean=True` above is the ingest manager saying it *joined* the actor
            # thread, so nothing can publish after it and the assertion holds with no pause at
            # all. What the pause is for is the opposite case -- an implementation that
            # signalled the decoder and returned without waiting would need somewhere to be
            # caught, and a check taken in the same breath as the DELETE would not catch it.
            time.sleep(0.05)
            assert streamed.sink().emitted == settled, "the decoder is still publishing"
            assert client.get("/streams").json() == {"streams": []}
        assert streamed.runner.cameras == ()

    def test_deleting_a_camera_nobody_holds_is_404(self, deployment) -> None:
        streamed = deployment()
        with streamed.client as client:
            response = client.delete("/streams/quay-9")

        assert response.status_code == 404
        assert "quay-9" in response.json()["detail"]


class TestDrainingAndHealth:
    def test_a_drain_releases_every_camera_and_reports_a_clean_zero(self, deployment) -> None:
        streamed = deployment(frames=200, pause_s=0.002)
        with streamed.client as client:
            client.post("/streams", json={"camera_id": "quay-1", "url": "injected://q"})
            client.post("/streams", json={"camera_id": "quay-2", "url": "injected://q"})
            assert until(lambda: len(client.get("/streams").json()["streams"]) == 2)

            response = client.post("/streams/drain", params={"timeout_s": 5.0})

        assert response.status_code == 200
        assert response.json() == {"abandoned": 0}
        assert streamed.runner.cameras == ()

    def test_health_reports_the_runner_and_its_cameras(self, deployment) -> None:
        streamed = deployment(frames=200, pause_s=0.002)
        with streamed.client as client:
            client.post("/streams", json={"camera_id": "quay-1", "url": "injected://q"})
            body = client.get("/health").json()

        assert body["state"] == "running"
        assert body["runner"] == "inprocess"
        assert "quay-1" in body["cameras"]

    def test_a_runner_that_has_not_started_refuses_with_503(self, deployment) -> None:
        """Not an implicit start: the chain is not open, so the frame would meet a closed
        queue. 503 rather than 400 because starting the runner makes the same request work."""
        streamed = deployment(start=False)
        with streamed.client as client:
            response = client.post("/streams", json={"url": "injected://a"})

        assert response.status_code == 503
        assert "not running" in response.json()["detail"]

    def test_a_stopped_runner_still_answers_health_with_200(self, deployment) -> None:
        streamed = deployment(start=False)
        with streamed.client as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["state"] == "stopped"


def _bands(client: TestClient) -> dict[str, str | None]:
    """Every listed camera's reported lane, keyed by id."""
    return {
        str(stream["camera_id"]): stream["priority"]
        for stream in client.get("/streams").json()["streams"]
    }


def _state(client: TestClient, camera_id: str) -> str:
    for stream in client.get("/streams").json()["streams"]:
        if stream["camera_id"] == camera_id:
            return str(stream["state"])
    return ""
