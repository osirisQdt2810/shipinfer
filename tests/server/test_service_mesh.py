"""Two shards' meshes in one process, over real rings: create, connect, and a request that
leaves one shard's dispatcher and comes back from the other's model."""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError, RingClosedError
from shipinfer.core.request import InferenceRequest, InferenceResponse, RequestContext
from shipinfer.core.request.future import ResponseFuture
from shipinfer.core.settings.topology import ServiceSettings
from shipinfer.core.types import Device, Tensor
from shipinfer.scheduling.dispatcher import Dispatcher
from shipinfer.scheduling.policies.locality_spillover import LocalityAwareSpilloverPolicy
from shipinfer.scheduling.work import WorkItem
from shipinfer.server.service_mesh import ServiceMesh, ring_name


class FakeModel:
    """A shared model as the mesh sees it: a name, an `infer`, a load, and an attach."""

    def __init__(self, name: str, shard: int, *, depth: int = 0) -> None:
        self.name = name
        self.shard = shard
        self._depth = depth
        self.remote: list = []
        self.served: list[tuple[str, int]] = []
        self.via: list[str] = []

    def infer(self, request: InferenceRequest) -> Future:
        self.via.append("infer")
        return self._run(request)

    def infer_local(self, request: InferenceRequest) -> Future:
        self.via.append("local")
        return self._run(request)

    def _run(self, request: InferenceRequest) -> Future:
        self.served.append((request.context.camera_id, request.context.frame_id))
        future: Future = Future()

        def work() -> None:
            outputs = {
                n: Tensor.from_numpy(t.numpy() + self.shard) for n, t in request.inputs.items()
            }
            future.set_result(
                InferenceResponse(
                    request_id=request.request_id,
                    model_name=self.name,
                    model_version=1,
                    outputs=outputs,
                    context=request.context,
                    timings=request.timings,
                    executed_on=Device.cuda(self.shard),
                )
            )

        threading.Thread(target=work, daemon=True).start()
        return future

    @property
    def total_depth(self) -> int:
        return self._depth

    @property
    def ewma_latency_us(self) -> float:
        return 100.0 * (self.shard + 1)

    def attach_remote(self, candidates) -> None:
        self.remote = list(candidates)


def _settings(run_id: str, shard: int, peers: list[int]) -> ServiceSettings:
    return ServiceSettings(
        shared_models=["emb"],
        slots_per_pair=4,
        slot_bytes=32 * 1024,
        heartbeat_ms=50.0,
        lost_after_ms=500.0,
        connect_timeout_s=5.0,
        shard=shard,
        peers=peers,
        run_id=run_id,
    )


@pytest.fixture()
def two_shards():
    run = uuid.uuid4().hex[:8]
    models = {0: FakeModel("emb", 0), 1: FakeModel("emb", 1)}
    meshes = {i: ServiceMesh(_settings(run, i, [0, 1]), i, {"emb": models[i]}) for i in (0, 1)}
    for mesh in meshes.values():
        mesh.create()
    for mesh in meshes.values():
        mesh.connect(timeout_s=5.0)
    time.sleep(0.1)  # first heartbeats
    try:
        yield run, models, meshes
    finally:
        for mesh in meshes.values():
            mesh.stop()


class TestNames:
    def test_ring_names_are_deterministic_and_run_scoped(self) -> None:
        assert ring_name("abc", 0, 1, "emb", "req") == "shipinfer-abc-0-to-1-emb-req"
        assert ring_name("abc", 0, 1, "emb", "res") != ring_name("abc", 1, 0, "emb", "res")
        with pytest.raises(ValueError):
            ring_name("abc", 0, 1, "emb", "bogus")

    def test_a_mesh_needs_a_run_id_and_must_be_among_its_peers(self) -> None:
        with pytest.raises(ConfigurationError, match="no run id"):
            ServiceMesh(_settings("", 0, [0, 1]), 0, {})
        with pytest.raises(ConfigurationError, match="not among the peers"):
            ServiceMesh(_settings("run", 2, [0, 1]), 2, {})


class TestTwoShardsJoinTheTier:
    def test_each_shard_gets_a_proxy_for_the_other(self, two_shards) -> None:
        _, models, _meshes = two_shards
        for shard in (0, 1):
            assert [p.owner for p in models[shard].remote] == [f"{1 - shard}:emb"]
            proxy = models[shard].remote[0]
            assert proxy.is_ready, "the peer's ingress has stamped its heartbeat"
            assert proxy.ewma_latency_us == 100.0 * (
                2 - shard
            ), "the peer's load, off the header"

    def test_a_request_leaves_shard_0_and_runs_on_shard_1(self, two_shards) -> None:
        _, models, _ = two_shards
        proxy = models[0].remote[0]
        request = InferenceRequest(
            model_name="emb",
            inputs={"x": Tensor.from_numpy(np.zeros((2, 4), dtype=np.float32))},
            context=RequestContext(camera_id="quay-1", frame_id=5),
        )
        future = ResponseFuture(request)
        proxy.enqueue(WorkItem(request, future))
        response = future.result(timeout=3)
        assert response.executed_on == Device.cuda(1)
        np.testing.assert_array_equal(
            response.outputs["x"].numpy(), np.ones((2, 4), dtype=np.float32)
        )
        assert models[1].served == [("quay-1", 5)] and models[0].served == []
        assert models[1].via == [
            "local"
        ], "the owner runs it on its own instances: never twice across"

    def test_through_a_dispatcher_the_deep_shard_borrows_the_quiet_one(
        self, two_shards
    ) -> None:
        _, models, _ = two_shards
        models[0]._depth = 50  # shard 0 is busy: its header says so to shard 1, and vice versa

        class LocalDeep:
            device = Device.cuda(0)
            depth = 50
            ewma_latency_us = 5000.0
            is_ready = True

            def enqueue(self, item):
                raise AssertionError("the policy should have sent this to the quiet peer")

        dispatcher = Dispatcher(
            model_name="emb",
            instances=[LocalDeep(), *models[0].remote],
            policy=LocalityAwareSpilloverPolicy(spill_threshold=4),
        )
        request = InferenceRequest(
            model_name="emb",
            inputs={"x": Tensor.from_numpy(np.zeros((1, 4), dtype=np.float32))},
            context=RequestContext(camera_id="quay-2", frame_id=1),
        )
        request.resident_device = Device.cuda(0)
        future = ResponseFuture(request)
        result = dispatcher.dispatch(
            WorkItem(request, future), lambda inst, item: inst.enqueue(item)
        )
        assert result.instance is models[0].remote[0]
        assert future.result(timeout=3).executed_on == Device.cuda(1)

    def test_stop_takes_the_rings_down(self, two_shards) -> None:
        run, _, meshes = two_shards
        meshes[1].stop()
        time.sleep(0.05)
        from shipinfer.runtime.memory.shared_ring import RingLayout, SharedRing

        with pytest.raises(RingClosedError):
            SharedRing.open(
                ring_name(run, 0, 1, "emb", "req"), RingLayout(slots=4, slot_bytes=32 * 1024)
            )
        proxy_on_0 = meshes[0].proxies["emb"][0]
        assert not proxy_on_0.is_ready, "shard 1 is gone: its proxy on shard 0 says so"


class TestConnectWaitsForPeers:
    def test_a_peer_that_never_creates_its_rings_is_a_configuration_error(self) -> None:
        run = uuid.uuid4().hex[:8]
        lonely = ServiceMesh(_settings(run, 0, [0, 1]), 0, {"emb": FakeModel("emb", 0)})
        lonely.create()
        try:
            with pytest.raises(ConfigurationError, match="never appeared"):
                lonely.connect(timeout_s=0.3)
        finally:
            lonely.stop()


class TestSlotSizing:
    """Both processes derive a ring's slot size from the model's own config — no negotiation."""

    def test_a_static_model_gets_batch_times_bytes_plus_heads_per_direction(self) -> None:
        from types import SimpleNamespace

        from shipinfer.core.types import DataType
        from shipinfer.server.service_mesh import wire_slot_bytes

        embedder = SimpleNamespace(
            max_batch_size=16,
            input_specs=[SimpleNamespace(shape=(3, 256, 128), dtype=DataType.FP32)],
            output_specs=[SimpleNamespace(shape=(2048,), dtype=DataType.FP32)],
        )
        req, res = wire_slot_bytes(embedder, fallback=1_638_400)
        assert req == 6_356_992, "16 x 3x256x128 fp32 + 64 KiB heads, page-rounded"
        assert res == 196_608, "16 x 2048 fp32 + heads — the directions differ"
        assert req % 4096 == 0 and res % 4096 == 0

    def test_a_dynamic_extent_falls_back_to_the_setting(self) -> None:
        from types import SimpleNamespace

        from shipinfer.core.types import DataType
        from shipinfer.server.service_mesh import wire_slot_bytes

        dyn = SimpleNamespace(
            max_batch_size=8,
            input_specs=[SimpleNamespace(shape=(3, -1, -1), dtype=DataType.FP32)],
            output_specs=[SimpleNamespace(shape=(300, 6), dtype=DataType.FP32)],
        )
        req, res = wire_slot_bytes(dyn, fallback=999_424)
        assert req == 999_424, "unsizeable side: the setting"
        assert res != 999_424, "the sizeable side is still computed"

    def test_the_mesh_lays_out_each_direction_with_its_own_size(self) -> None:
        mesh = ServiceMesh(
            _settings("run", 0, [0, 1]),
            0,
            {"emb": FakeModel("emb", 0)},
            slot_bytes_by_model={"emb": (200_704, 8_192)},
        )
        assert mesh.layout_for("emb", "req").slot_bytes == 200_704
        assert mesh.layout_for("emb", "res").slot_bytes == 8_192
        assert mesh.layout_for("other", "req").slot_bytes == mesh.settings.slot_bytes
