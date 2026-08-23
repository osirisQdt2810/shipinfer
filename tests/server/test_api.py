"""The HTTP ingress, which is where the (camera, frame) tag used to die.

The tag is carried on every internal path — batching, scatter, spillover, the ensemble DAG,
error paths — and then the only ingress that ships threw it away. `RequestContext` was built
with the trace id alone, and the request schema forbade extra fields, so a client could not
have supplied it even knowing to try.

Two consequences, both real: every HTTP request shared the fairness key "-", so all cameras
queued in one lane and the fair queue degenerated to FIFO (the exact defect ADR-005 exists
to prevent); and two responses were indistinguishable, so nothing downstream could reassemble
them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from shipinfer.core.settings import ServerSettings  # noqa: E402
from shipinfer.server import InferenceServer  # noqa: E402
from shipinfer.server.api import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "echo" / "1").mkdir(parents=True)
    (root / "echo" / "config.yaml").write_text(
        "platform: mock\n"
        "max_batch_size: 4\n"
        "inputs: [{name: x, data_type: FP32, dims: [2]}]\n"
        "outputs: [{name: y, data_type: FP32, dims: [2]}]\n"
        "instance_groups: [{kind: KIND_CPU, count: 1}]\n"
        "dynamic_batching: {enabled: false}\n"
        "parameters: {latency_ms: 0.05}\n"
    )
    server = InferenceServer(
        ServerSettings(
            model_repository=root,
            devices={"visible_gpus": []},
            execution={"warmup_iterations": 0},
        )
    ).start()
    try:
        with TestClient(create_app(server)) as test_client:
            yield test_client
    finally:
        server.stop()


def _body(**parameters):
    return {
        "id": "req-1",
        "inputs": [{"name": "x", "shape": [1, 2], "datatype": "FP32", "data": [1.0, 2.0]}],
        "parameters": parameters,
    }


class TestContextTagOverHttp:
    """The regression this file exists for: the tag survives the HTTP ingress."""

    def test_the_tag_round_trips(self, client) -> None:
        """The regression test. Supplied in parameters, returned in parameters."""
        response = client.post(
            "/v2/models/echo/infer", json=_body(camera_id="cam07", frame_id=1234)
        )

        assert response.status_code == 200
        parameters = response.json()["parameters"]
        assert parameters["camera_id"] == "cam07"
        assert parameters["frame_id"] == 1234

    def test_two_cameras_are_distinguishable_in_their_responses(self, client) -> None:
        a = client.post(
            "/v2/models/echo/infer", json=_body(camera_id="camA", frame_id=7)
        ).json()
        b = client.post(
            "/v2/models/echo/infer", json=_body(camera_id="camB", frame_id=3)
        ).json()

        assert (a["parameters"]["camera_id"], a["parameters"]["frame_id"]) == ("camA", 7)
        assert (b["parameters"]["camera_id"], b["parameters"]["frame_id"]) == ("camB", 3)

    def test_an_untagged_request_still_works(self, client) -> None:
        """Legitimate, and documented as sharing one fairness lane."""
        response = client.post("/v2/models/echo/infer", json=_body())
        assert response.status_code == 200
        assert response.json()["parameters"]["camera_id"] == ""

    def test_a_malformed_frame_id_is_a_client_error(self, client) -> None:
        """Coercing it to the default would silently merge that client's frames into the
        untagged lane — the bug the tag exists to make impossible."""
        response = client.post("/v2/models/echo/infer", json=_body(frame_id="not-a-number"))
        assert response.status_code == 400


class TestErrorStatusCodes:
    """A client can tell its own bug from a server fault by the status code alone."""

    def test_error_mapping(self, client) -> None:
        """Retryable and not-retryable must be distinguishable by status code, or a client
        turns its own bug into a retry storm."""
        assert client.post("/v2/models/nope/infer", json=_body()).status_code == 404

        wrong_shape = {
            "id": "x",
            "inputs": [{"name": "x", "shape": [1, 9], "datatype": "FP32", "data": [0.0] * 9}],
            "parameters": {},
        }
        assert client.post("/v2/models/echo/infer", json=wrong_shape).status_code == 400


class TestKServeSurface:
    """The rest of the KServe v2 surface: outputs, health and metadata."""

    def test_output_tensors_come_back(self, client) -> None:
        body = client.post(
            "/v2/models/echo/infer", json=_body(camera_id="c", frame_id=1)
        ).json()
        assert body["model_name"] == "echo"
        assert [o["name"] for o in body["outputs"]] == ["y"]
        assert np.asarray(body["outputs"][0]["data"]).shape == (2,)

    def test_health_and_metadata(self, client) -> None:
        assert client.get("/v2/health/live").status_code == 200
        assert client.get("/v2/health/ready").status_code == 200

        metadata = client.get("/v2/models/echo").json()
        assert metadata["name"] == "echo"
        assert [i["name"] for i in metadata["inputs"]] == ["x"]

        assert "shipinfer_requests_total" in client.get("/metrics").text
