"""Two `InferenceServer`s in one process, configured as shards 0 and 1 of one `service` run.

The engine's own path — settings → `_join_service_tier` → `ServiceMesh` → `Model.attach_remote`
— over real rings and the mock backend, with no launcher and no device: what the two-process
multigpu test does, minus the processes. `start()` blocks in `connect()` until the peer's rings
exist, so the two servers start on two threads.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.request import InferenceRequest, RequestContext
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor
from shipinfer.engine import InferenceServer
from shipinfer.engine.spill.remote_instance import RemoteInstance


def _settings(repository: Path, run: str, shard: int) -> ServerSettings:
    return ServerSettings(
        model_repository=repository,
        runner={
            "runner": "service",
            "service": {
                "shared_models": ["echo"],
                "slots_per_pair": 4,
                "slot_bytes": 64 * 1024,
                "heartbeat_ms": 50.0,
                "connect_timeout_s": 20.0,
                "shard": shard,
                "peers": [0, 1],
                "run_id": run,
            },
        },
    )


@pytest.fixture()
def two_servers(tmp_repository: Path):
    run = uuid.uuid4().hex[:8]
    servers = [InferenceServer(_settings(tmp_repository, run, shard)) for shard in (0, 1)]
    errors: list[BaseException] = []

    def start(server: InferenceServer) -> None:
        try:
            server.start()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=start, args=(s,), daemon=True) for s in servers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    try:
        assert not errors, errors
        assert all(not t.is_alive() for t in threads), "a server never finished starting"
        yield servers
    finally:
        for server in servers:
            server.stop()


class TestTheEngineJoinsTheTier:
    def test_each_server_holds_a_mesh_and_a_proxy_for_the_other(self, two_servers) -> None:
        for shard, server in enumerate(two_servers):
            assert server.service_mesh is not None
            # The engine sized the rings from the model's own config: echo is max_batch 8 of
            # one FP32 [4] input and output, so both directions are 8 x 16 B + 64 KiB heads,
            # page-rounded.
            assert server.service_mesh.slot_bytes_by_model["echo"] == (69_632, 69_632)
            proxies = [
                p
                for p in server.model("echo")._dispatcher.instances
                if isinstance(p, RemoteInstance)
            ]
            assert [p.owner for p in proxies] == [f"{1 - shard}:echo"]
            assert proxies[0].is_ready

    def test_a_request_still_answers_through_the_front_server(self, two_servers) -> None:
        front = two_servers[0]
        request = InferenceRequest(
            model_name="echo",
            inputs={"x": Tensor.from_numpy(np.arange(4, dtype=np.float32).reshape(1, 4))},
            context=RequestContext(camera_id="quay-1", frame_id=3),
        )
        response = front.infer(request).result(timeout=10)
        assert (response.context.camera_id, response.context.frame_id) == ("quay-1", 3)

    def test_a_server_without_a_shard_index_joins_nothing(self, tmp_repository: Path) -> None:
        settings = ServerSettings(
            model_repository=tmp_repository,
            runner={"runner": "service", "service": {"shared_models": ["echo"]}},
        )
        server = InferenceServer(settings).start()
        try:
            assert server.service_mesh is None
        finally:
            server.stop()


class TestAFailedConnectLeaksNothing:
    def test_the_engine_takes_its_rings_down_when_the_peer_never_appears(
        self, tmp_repository: Path
    ) -> None:
        from shipinfer.core.errors import ConfigurationError, RingClosedError
        from shipinfer.engine.spill.mesh import ring_name
        from shipinfer.runtime.memory.shared_ring import RingLayout, SharedRing

        run = uuid.uuid4().hex[:8]
        settings = _settings(tmp_repository, run, 0)
        settings.runner.service.connect_timeout_s = 0.3
        server = InferenceServer(settings)
        with pytest.raises(ConfigurationError, match="never appeared"):
            server.start()
        # The rings shard 0 created for its absent peer are gone, not leaked: a restart can
        # recreate the names.
        created = ring_name(run, 1, 0, "echo", "req")
        with pytest.raises(RingClosedError):
            SharedRing.open(created, RingLayout(slots=4, slot_bytes=64 * 1024))
        server.stop()


class TestAnEnsembleCannotCrossTheTier:
    def test_the_join_refuses_it_by_name_before_any_ring_exists(
        self, tmp_repository: Path
    ) -> None:
        """Round 2: the wire carries one request to one model; an ensemble's DAG references
        local models by identity, so offering it to peers can only fail later and worse."""
        from types import SimpleNamespace

        from shipinfer.core.errors import ConfigurationError

        settings = _settings(tmp_repository, uuid.uuid4().hex[:8], 0)
        settings.runner.service.shared_models = ["fused"]
        server = InferenceServer(settings)
        server._models["fused"] = SimpleNamespace(name="fused")  # ensembles have no infer_local
        with pytest.raises(ConfigurationError, match="'fused' is an ensemble"):
            server._join_service_tier()
