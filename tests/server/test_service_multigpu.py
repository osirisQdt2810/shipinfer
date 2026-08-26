"""Two real shard processes on two GPUs, the `service` topology between them.

The first end-to-end run of the tier: `shipinfer fleet`'s own launcher starts two `serve`
processes through the real `ServiceTopology` (run id, peer set, shard index in the environment),
each pinned to one physical GPU, each serving HTTP on its own port. Every request goes to shard 0.
Its one mock instance takes 40 ms a request and the spill threshold is low, so shard 0's queue is
deep within the first few requests and the policy borrows shard 1's instance through the ring.
Shard 1's statistics say how much of shard 0's work it did — the number the topology exists for.

Inside the container only (the suite's containment gate), on the devices `SHIPINFER_TEST_GPUS`
names (default `0,1`; `deploy/rootless/test.sh` forwards it). The mock backend keeps the run to a
CUDA context per shard, nothing else.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from shipinfer.core.settings import ServerSettings
from shipinfer.server.launcher import Fleet, serve_command
from shipinfer.server.topology import ServiceTopology

pytestmark = [pytest.mark.multigpu, pytest.mark.timeout(300)]

REQUESTS = 24
READY_TIMEOUT_S = 150.0
# The plan and the children's configuration are two views of one fleet, and `serve` refuses a
# shard whose cameras its configuration does not define. `serve` starts no ingest, so the URIs
# are never opened.
CAMERAS = [
    {"camera_id": "quay-a", "uri": "rtsp://127.0.0.1:8554/quay-a"},
    {"camera_id": "quay-b", "uri": "rtsp://127.0.0.1:8554/quay-b"},
]


def _gpus() -> list[int]:
    raw = os.environ.get("SHIPINFER_TEST_GPUS", "0,1")
    gpus = [int(x) for x in raw.split(",") if x.strip()]
    if len(gpus) < 2:
        pytest.skip("SHIPINFER_TEST_GPUS names fewer than two devices")
    return gpus[:2]


def _two_free_ports() -> int:
    """A base such that base and base + 1 are both free right now."""
    for _ in range(50):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            base = probe.getsockname()[1]
        try:
            with socket.socket() as second:
                second.bind(("127.0.0.1", base + 1))
        except OSError:
            continue
        return base
    raise RuntimeError("no two consecutive free ports")


def _write_repository(root: Path) -> None:
    (root / "emb" / "1").mkdir(parents=True)
    (root / "emb" / "config.yaml").write_text("""
name: emb
platform: mock
max_batch_size: 1
inputs:
  - {name: x, data_type: FP32, dims: [4]}
outputs:
  - {name: y, data_type: FP32, dims: [4]}
instance_groups:
  - {kind: KIND_GPU, count: 1}
dynamic_batching:
  enabled: false
parameters:
  latency_ms: 40
""".lstrip())


def _get(url: str, timeout: float = 5.0) -> tuple[int, dict]:
    """Status and body; 0 while nothing listens yet (a shard still importing torch)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as reply:
            raw = reply.read()
            return reply.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        return exc.code, {}
    except (urllib.error.URLError, OSError):
        return 0, {}


def _post(url: str, body: dict, timeout: float = 30.0) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode(errors="replace")}


def _wait_ready(ports: list[int], fleet: Fleet) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    pending = set(ports)
    while pending:
        for port in list(pending):
            status, _ = _get(f"http://127.0.0.1:{port}/v2/health/ready", timeout=2.0)
            if status == 200:
                pending.discard(port)
        if not pending:
            return
        exited = [s for s in fleet.running if s.process.poll() is not None]
        assert not exited, f"shard(s) exited before becoming ready: {exited}"
        assert (
            time.monotonic() < deadline
        ), f"shards on ports {sorted(pending)} never became ready"
        time.sleep(0.5)


def _inference_count(port: int) -> int:
    status, body = _get(f"http://127.0.0.1:{port}/v2/models/emb/stats")
    assert status == 200, body
    return int(body["model_stats"][0]["inference_count"])


class TestTwoShardsShareTheEmbedder:
    def test_a_deep_shard_borrows_its_peer_through_the_ring(self, tmp_path: Path) -> None:
        gpus = _gpus()
        root = tmp_path / "model_repository"
        _write_repository(root)
        settings = ServerSettings(model_repository=root, ingest={"cameras": CAMERAS})
        topology = ServiceTopology()
        plan = topology.plan(
            settings, cameras={c["camera_id"]: 1.0 for c in CAMERAS}, gpus=gpus, shards=2
        )
        base = _two_free_ports()
        ports = [base + shard.index for shard in plan.shards]
        env = {
            **topology.environment(settings),
            "SHIPINFER_INGEST__CAMERAS": json.dumps(CAMERAS),
            # The children read their configuration from the environment, as every shard does.
            "SHIPINFER_TOPOLOGY__SERVICE__SHARED_MODELS": json.dumps(["emb"]),
            "SHIPINFER_TOPOLOGY__SERVICE__HEARTBEAT_MS": "50",
            # Small rings: the payload is four floats, and a container's /dev/shm is 64 MiB by default.
            "SHIPINFER_TOPOLOGY__SERVICE__SLOTS_PER_PAIR": "4",
            "SHIPINFER_TOPOLOGY__SERVICE__SLOT_BYTES": str(256 * 1024),
            "SHIPINFER_TOPOLOGY__SERVICE__CONNECT_TIMEOUT_S": str(READY_TIMEOUT_S),
            "SHIPINFER_SCHEDULER__PLACEMENT_POLICY": "locality_spillover",
            "SHIPINFER_SCHEDULER__PLACEMENT_POLICY_OPTIONS": json.dumps({"spill_threshold": 2}),
        }
        fleet = Fleet(
            plan=plan,
            command=lambda shard: serve_command(
                shard, repository=str(root), http_port_base=base
            ),
            env=env,
            shard_env=topology.shard_environment,
            drain_s=15.0,
        )
        fleet.start()
        try:
            _wait_ready(ports, fleet)
            front, peer = ports

            results: list[tuple[int, dict]] = [None] * REQUESTS  # type: ignore[list-item]

            def one(i: int) -> None:
                results[i] = _post(
                    f"http://127.0.0.1:{front}/v2/models/emb/infer",
                    {
                        "id": f"req-{i}",
                        "inputs": [
                            {
                                "name": "x",
                                "shape": [1, 4],
                                "datatype": "FP32",
                                "data": [float(i)] * 4,
                            }
                        ],
                        "parameters": {"camera_id": "quay-a", "frame_id": i},
                    },
                )

            threads = [threading.Thread(target=one, args=(i,)) for i in range(REQUESTS)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=120)

            failures = [(i, r) for i, r in enumerate(results) if r is None or r[0] != 200]
            assert not failures, failures[:3]
            for i, (_, body) in enumerate(results):
                assert (
                    body["parameters"]["frame_id"] == i
                ), "the tag came back with its own response"
                assert body["parameters"]["camera_id"] == "quay-a"

            time.sleep(0.5)  # the last statistics update lands
            done_here, done_by_peer = _inference_count(front), _inference_count(peer)
            assert (
                done_by_peer >= 1
            ), "shard 1 executed none of shard 0's work: the tier did not carry"
            assert (
                done_here >= 1
            ), "shard 0 executed nothing itself: the policy is not keeping work home"
            assert done_here + done_by_peer == REQUESTS, (done_here, done_by_peer)
        finally:
            fleet.stop(drain_s=15.0)
        # Both processes are gone: their ports refuse, and with them their CUDA contexts.
        for port in ports:
            with pytest.raises((urllib.error.URLError, ConnectionError, OSError)):
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v2/health/live", timeout=2.0)
