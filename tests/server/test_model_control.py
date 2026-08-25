"""Explicit model control: load and unload a model into a running server.

The server loaded everything in the repository at start-up. That is fine for the six-model
pipeline this project ships, and wrong for a repository that grows — a hundred models to
serve three is how a box runs out of VRAM before it answers a request.

Triton's model-repository extension is the shape adopted here, paths and verbs unchanged.
Two of its behaviours are deliberately *not* copied and are pinned below so the difference
is a decision rather than a gap: a load of an already-loaded model is refused rather than
treated as a reload, and there is no polling mode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError, ModelControlError, ModelNotFoundError
from shipinfer.core.request import InferenceRequest, RequestContext
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor
from shipinfer.server import InferenceServer

_MODEL = """
platform: mock
max_batch_size: 4
inputs: [{name: x, data_type: FP32, dims: [2]}]
outputs: [{name: y, data_type: FP32, dims: [2]}]
instance_groups: [{kind: KIND_CPU, count: 1}]
dynamic_batching: {enabled: false}
parameters: {latency_ms: 0.05}
"""

_ENSEMBLE = """
platform: ensemble
max_batch_size: 0
inputs: [{name: x, data_type: FP32, dims: [2]}]
outputs: [{name: y, data_type: FP32, dims: [2]}]
dynamic_batching: {enabled: false}
ensemble:
  steps:
    - model: echo
      input_map: {x: x}
      output_map: {y: y}
"""


def _write_model(root: Path, name: str, body: str = _MODEL, *, versioned: bool = True) -> None:
    directory = root / name
    (directory / "1").mkdir(parents=True) if versioned else directory.mkdir(parents=True)
    (directory / "config.yaml").write_text(body.lstrip())


@pytest.fixture()
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    _write_model(root, "echo")
    _write_model(root, "other")
    return root


def _settings(root: Path, **overrides) -> ServerSettings:
    return ServerSettings(
        model_repository=root,
        devices={"visible_gpus": []},
        execution={"warmup_iterations": 0},
        **overrides,
    )


@pytest.fixture()
def explicit(repository: Path):
    """A server that loads nothing at start-up and takes its models over the API."""
    settings = _settings(
        repository, model_control="explicit", load_all_models=False, startup_models=[]
    )
    with InferenceServer(settings) as server:
        yield server


def _request(model: str = "echo") -> InferenceRequest:
    return InferenceRequest(
        model_name=model,
        inputs={"x": Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))},
        context=RequestContext(camera_id="cam0", frame_id=1),
    )


class TestControlIsOffByDefault:
    """A control plane that can unload the detector can take the deployment down."""

    def test_load_is_refused_when_control_is_not_explicit(self, repository: Path) -> None:
        with (
            InferenceServer(_settings(repository)) as server,
            pytest.raises(ModelControlError, match="model_control='none'"),
        ):
            server.load_model("echo")

    def test_unload_is_refused_when_control_is_not_explicit(self, repository: Path) -> None:
        with InferenceServer(_settings(repository)) as server, pytest.raises(ModelControlError):
            server.unload_model("echo")

    def test_the_default_still_loads_everything(self, repository: Path) -> None:
        with InferenceServer(_settings(repository)) as server:
            assert server.models() == ["echo", "other"]


class TestExplicitControl:
    """Start empty, load what is wanted, unload it again."""

    def test_explicit_control_starts_with_nothing_loaded(self, explicit) -> None:
        assert explicit.models() == []
        assert explicit.is_ready  # nothing loaded is not the same as unhealthy

    def test_load_makes_a_model_servable(self, explicit) -> None:
        with pytest.raises(ModelNotFoundError):
            explicit.infer(_request())

        explicit.load_model("echo")

        assert explicit.models() == ["echo"]
        response = explicit.infer_sync(_request(), timeout=10)
        assert response.outputs["y"].shape == (1, 2)

    def test_unload_removes_it_and_stops_its_instances(self, explicit) -> None:
        model = explicit.load_model("echo")

        explicit.unload_model("echo")

        assert explicit.models() == []
        assert not model.is_ready
        with pytest.raises(ModelNotFoundError):
            explicit.infer(_request())

    def test_loading_twice_is_refused_rather_than_reloading(self, explicit) -> None:
        """A reload has to stop the running copy first — two detectors do not fit on one
        GPU — so a reload that failed halfway would take a working model down. Saying
        ``unload`` then ``load`` says the same thing deliberately."""
        explicit.load_model("echo")

        with pytest.raises(ModelControlError, match="already loaded"):
            explicit.load_model("echo")

        assert explicit.model("echo").is_ready

    def test_loading_an_unknown_model_is_a_not_found(self, explicit) -> None:
        with pytest.raises(ModelNotFoundError):
            explicit.load_model("absent")

    def test_unloading_a_model_that_is_not_loaded_is_a_not_found(self, explicit) -> None:
        with pytest.raises(ModelNotFoundError):
            explicit.unload_model("other")

    def test_a_model_added_after_start_up_can_be_loaded(self, explicit, repository) -> None:
        """The reason the feature exists: the repository grows. The index is re-scanned on
        an explicit request, which is not the polling mode — the operator asked at this
        moment, so a broken config fails their call rather than a timer's."""
        _write_model(repository, "arrived_late")

        explicit.load_model("arrived_late")

        assert "arrived_late" in explicit.models()

    def test_a_failed_load_leaves_the_server_untouched(self, explicit, repository) -> None:
        _write_model(repository, "broken", _MODEL.replace("platform: mock", "platform: nope"))
        explicit.load_model("echo")

        with pytest.raises(ConfigurationError):
            explicit.load_model("broken")

        assert explicit.models() == ["echo"]
        assert explicit.model("echo").is_ready


class TestEnsembleDependencies:
    """Unloading a step out from under an ensemble is refused, not discovered at runtime."""

    @pytest.fixture()
    def with_ensemble(self, repository: Path):
        _write_model(repository, "pipe", _ENSEMBLE, versioned=False)
        settings = _settings(
            repository,
            model_control="explicit",
            load_all_models=False,
            startup_models=["echo", "pipe"],
        )
        with InferenceServer(settings) as server:
            yield server

    def test_a_step_of_a_loaded_ensemble_cannot_be_unloaded(self, with_ensemble) -> None:
        with pytest.raises(ModelControlError, match="pipe"):
            with_ensemble.unload_model("echo")

        assert with_ensemble.model("echo").is_ready

    def test_unloading_the_ensemble_first_releases_the_step(self, with_ensemble) -> None:
        with_ensemble.unload_model("pipe")

        with_ensemble.unload_model("echo")

        assert with_ensemble.models() == []


class TestRepositoryIndex:
    """``/v2/repository/index``: what exists, and what is serving."""

    def test_it_reports_state_per_model(self, explicit) -> None:
        explicit.load_model("echo")

        index = {entry["name"]: entry for entry in explicit.index()}

        assert index["echo"]["state"] == "READY"
        assert index["echo"]["version"] == "1"
        assert index["other"]["state"] == "UNAVAILABLE"
        assert index["other"]["reason"]  # a state with no reason is not a diagnosis


class TestControlOverHttp:
    """The endpoints a Triton control plane already speaks."""

    @pytest.fixture()
    def client(self, explicit):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from shipinfer.server.api import create_app

        with TestClient(create_app(explicit)) as test_client:
            yield test_client

    def test_load_then_infer_then_unload(self, client) -> None:
        assert client.post("/v2/repository/models/echo/load").status_code == 200

        infer = client.post(
            "/v2/models/echo/infer",
            json={
                "id": "r1",
                "inputs": [
                    {"name": "x", "shape": [1, 2], "datatype": "FP32", "data": [1.0, 2.0]}
                ],
                "parameters": {"camera_id": "cam0", "frame_id": 1},
            },
        )
        assert infer.status_code == 200

        assert client.post("/v2/repository/models/echo/unload").status_code == 200
        assert (
            client.post("/v2/models/echo/infer", json={"id": "r2", "inputs": []}).status_code
            == 404
        )

    def test_the_index_lists_the_repository(self, client) -> None:
        body = client.post("/v2/repository/index").json()

        assert {entry["name"] for entry in body} == {"echo", "other"}

    def test_an_unknown_model_is_404_and_a_refusal_is_400(self, client) -> None:
        assert client.post("/v2/repository/models/absent/load").status_code == 404

        client.post("/v2/repository/models/echo/load")
        # Already loaded: the client's mistake, and it will not start working on a retry.
        assert client.post("/v2/repository/models/echo/load").status_code == 400

    def test_control_is_400_not_503_when_the_mode_forbids_it(self, repository: Path) -> None:
        """503 would be retried forever by a load balancer or a deploy script; this never
        starts working without a restart with different settings."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from shipinfer.server.api import create_app

        with (
            InferenceServer(_settings(repository)) as server,
            TestClient(create_app(server)) as client,
        ):
            assert client.post("/v2/repository/models/echo/unload").status_code == 400
