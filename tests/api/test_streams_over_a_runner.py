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
from shipinfer.topology.elements.mock import MockOutput

CHAIN = """
name: streamed
elements:
  decode: {impl: replay}
  detect: {impl: mock, model: ship_detector}
  output: {impl: mock}
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

    def sink(self) -> MockOutput:
        element = self.chain.node("output").element
        assert isinstance(element, MockOutput)
        return element


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
            assert until(lambda: len(streamed.sink().emitted) == 4), streamed.sink().emitted

        assert [item.key for item in streamed.sink().emitted] == [
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
            assert until(lambda: len(streamed.sink().emitted) == 2)

        assert {item.key[0] for item in streamed.sink().emitted} == {"cam-000"}

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

    def test_a_second_camera_with_the_same_id_is_400(self, deployment) -> None:
        """The ingest manager's duplicate refusal, reaching the client as the caller's error."""
        streamed = deployment(frames=200, pause_s=0.002)
        with streamed.client as client:
            client.post("/streams", json={"camera_id": "quay-1", "url": "injected://q"})
            response = client.post("/streams", json={"camera_id": "quay-1", "url": "x"})

        assert response.status_code == 400
        assert "quay-1" in response.json()["detail"]


class TestDeletingAStream:
    def test_a_deleted_camera_stops_being_read_and_leaves_the_listing(self, deployment) -> None:
        """The mirror image, and the reason the listing is not a cache.

        The source has 200 frames left when the DELETE lands, so a sink that stops growing is
        the removal having stopped the decoder rather than the clip having ended.
        """
        streamed = deployment(frames=200, pause_s=0.002)
        with streamed.client as client:
            client.post("/streams", json={"camera_id": "quay-1", "url": "injected://q"})
            assert until(lambda: streamed.sink().emitted)

            response = client.delete("/streams/quay-1")
            assert response.status_code == 200
            assert response.json() == {"clean": True}

            settled = len(streamed.sink().emitted)
            time.sleep(0.05)
            assert len(streamed.sink().emitted) == settled, "the decoder is still publishing"
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


def _state(client: TestClient, camera_id: str) -> str:
    for stream in client.get("/streams").json()["streams"]:
        if stream["camera_id"] == camera_id:
            return str(stream["state"])
    return ""
