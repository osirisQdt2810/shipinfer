"""The response cache, end to end through a real server.

ADR-009 describes a cache that is opt-in per model and sound only for deterministic,
stateless models. It was implemented, documented, wired into a model config — and never
written to: ``ResponseCache.put`` had no call site anywhere in ``src/``. Every request
hashed all of its input bytes and then missed, forever.

These tests use the instance's own execution counter as the ground truth for "did the
model actually run", which is the only way to tell a hit from a fast miss.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.errors import InferenceError
from shipinfer.core.request import InferenceRequest, RequestContext
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor
from shipinfer.engine import InferenceServer
from tests.support.models import materialise


def _repo(tmp_path: Path, *, cache: str | None = "lru", broken: bool = False) -> Path:
    """A one-model repository whose caching can be switched on and off."""
    root = tmp_path / "repo"
    (root / "m" / "1").mkdir(parents=True)
    cache_block = ""
    if cache == "lru":
        cache_block = "  response_cache:\n    type: lru\n    max_entries: 4\n"
    elif cache is not None:
        cache_block = f"  response_cache: {cache}\n"
    (root / "m" / "config.yaml").write_text(
        "platform: pytorch\n"
        "max_batch_size: 4\n"
        "inputs: [{name: x, data_type: FP32, dims: [4]}]\n"
        "outputs: [{name: y, data_type: FP32, dims: [2]}]\n"
        "instance_groups: [{kind: KIND_CPU, count: 1}]\n"
        "dynamic_batching: {enabled: false}\n"
        "parameters:\n"
        "  latency_ms: 0.1\n"
        f"  disagrees_with_its_config: {broken}\n" + cache_block
    )
    materialise(root)
    return root


def _server(root: Path, *, warmup: int = 0) -> InferenceServer:
    # warmup_iterations=0 on purpose. Warm-up runs the backend before the instance reports
    # ready, so a mock configured to fail would fail there first: the instance never
    # becomes ready and start() blocks for its full timeout instead of the test failing.
    return InferenceServer(
        ServerSettings(
            model_repository=root,
            devices={"visible_gpus": []},
            execution={"warmup_iterations": warmup},
        )
    )


def _request(value: float, camera: str = "cam0", frame: int = 0) -> InferenceRequest:
    return InferenceRequest(
        model_name="m",
        inputs={"x": Tensor.from_numpy(np.full((1, 4), value, dtype=np.float32))},
        context=RequestContext(camera_id=camera, frame_id=frame),
    )


def _executions(server: InferenceServer) -> int:
    """How many batches the model actually attempted — the instance's own counters.

    It used to read a counter only `MockBackend` reported. The instance has always tracked
    this for every backend, and a cache test asking "did the model run again?" is asking
    exactly what `batches` answers.
    """
    return sum(
        i.stats()["batches"] + i.stats()["failed_batches"] for i in server.model("m").instances
    )


def _await_cache_write(
    server: InferenceServer, entries: int = 1, timeout_s: float = 5.0
) -> None:
    """Wait until the cache actually HOLDS what the last request should have stored.

    THE WRITE IS NOT ORDERED BEFORE THE RESPONSE: `model.py::_store_when_done` caches in a
    `add_done_callback`, which runs on the worker thread after the future resolves, and
    `infer_sync` returns on that same resolution. So a second identical request issued at
    once can find nothing and re-run the model. Reproduced 14 Sep, 1 run in 20 at load 66:
    `assert 2 == 1  # the second identical request re-ran the model`. Waiting tests what the
    cache promises -- identical inputs hit -- without asserting an ordering it does not.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.model("m").stats()["cache"]["entries"] >= entries:
            return
        time.sleep(0.005)
    raise AssertionError(
        f"the cache still holds {server.model('m').stats()['cache']['entries']} entr(ies) "
        f"after {timeout_s}s; the done-callback never ran, which is a real defect rather "
        f"than the ordering this helper exists to absorb"
    )


class TestCacheKeying:
    """Identical inputs hit; different inputs do not. Measured by the backend's own counter."""

    def test_identical_inputs_run_the_model_once(self, tmp_path: Path) -> None:
        """The whole point, and the test that fails against a cache that is never written."""
        with _server(_repo(tmp_path)) as server:
            first = server.infer_sync(_request(1.0), timeout=10)
            assert _executions(server) == 1
            _await_cache_write(server)

            second = server.infer_sync(_request(1.0), timeout=10)
            assert _executions(server) == 1, "the second identical request re-ran the model"

            np.testing.assert_array_equal(
                first.outputs["y"].numpy(), second.outputs["y"].numpy()
            )
            assert server.metrics.cache_hits.value(model="m") == 1
            assert server.metrics.cache_misses.value(model="m") == 1

    def test_different_inputs_are_not_confused(self, tmp_path: Path) -> None:
        with _server(_repo(tmp_path)) as server:
            a = server.infer_sync(_request(1.0), timeout=10)
            b = server.infer_sync(_request(2.0), timeout=10)
            assert _executions(server) == 2
            assert not np.array_equal(a.outputs["y"].numpy(), b.outputs["y"].numpy())


class TestCachedResponseIntegrity:
    """A hit carries the asking request's identity, and cannot be mutated by its caller."""

    def test_a_hit_carries_this_request_s_identity_not_the_stored_one(
        self, tmp_path: Path
    ) -> None:
        """A cached response must not hand back the tag of whoever populated the entry —
        that would misattribute a frame to the wrong camera, which is the failure the whole
        (camera_id, frame_id) invariant exists to prevent."""
        with _server(_repo(tmp_path)) as server:
            server.infer_sync(_request(1.0, camera="cam_first", frame=1), timeout=10)
            # WITHOUT THIS THE TEST PASSES VACUOUSLY. A miss returns a fresh response, which
            # carries the asking tag anyway -- so a raced write turns this from a cache test
            # into an assertion that `infer_sync` echoes its own request.
            _await_cache_write(server)
            hit = server.infer_sync(_request(1.0, camera="cam_second", frame=99), timeout=10)

        assert hit.context.camera_id == "cam_second"
        assert hit.context.frame_id == 99

    def test_a_cached_response_cannot_be_mutated_by_its_caller(self, tmp_path: Path) -> None:
        """Entries are shared between hits, so a writeable one lets any caller corrupt every
        later hit. numpy raises at the offending line instead."""
        with _server(_repo(tmp_path)) as server:
            server.infer_sync(_request(1.0), timeout=10)
            _await_cache_write(server)
            hit = server.infer_sync(_request(1.0), timeout=10)

        with pytest.raises(ValueError):
            hit.outputs["y"].numpy()[0, 0] = 123.0


class TestCacheAdmissionAndEviction:
    """What never enters the cache, and what leaves it when it is full."""

    def test_a_failed_request_is_not_cached(self, tmp_path: Path) -> None:
        """A stored exception would be served forever — far worse than being slow."""
        with _server(_repo(tmp_path, broken=True)) as server:
            for _ in range(2):
                with pytest.raises(InferenceError):
                    server.infer_sync(_request(1.0), timeout=10)
            # Both attempts reached the backend: nothing was served from the cache.
            assert _executions(server) == 2
            assert server.metrics.cache_hits.value(model="m") == 0

    def test_eviction_is_bounded(self, tmp_path: Path) -> None:
        with _server(_repo(tmp_path)) as server:
            for value in range(6):  # max_entries is 4
                server.infer_sync(_request(float(value)), timeout=10)
            stats = server.model("m").stats()["cache"]

        assert stats["entries"] <= 4
        assert stats["evictions"] >= 2


class TestCacheIsOptIn:
    """With no cache configured the lookup costs nothing, not even the input hash."""

    def test_caching_is_off_by_default_and_costs_no_hash(self, tmp_path: Path) -> None:
        """ADR-009: opt-in only. The null cache must not even compute a key, because the
        BLAKE2b pass over every input byte is the expensive half of a lookup."""
        from shipinfer.engine.cache import NullResponseCache

        with _server(_repo(tmp_path, cache=None)) as server:
            server.infer_sync(_request(1.0), timeout=10)
            server.infer_sync(_request(1.0), timeout=10)
            assert _executions(server) == 2

        assert (
            NullResponseCache().key_for(
                "m", 1, {"x": Tensor.from_numpy(np.zeros((1, 4), np.float32))}
            )
            is None
        )
