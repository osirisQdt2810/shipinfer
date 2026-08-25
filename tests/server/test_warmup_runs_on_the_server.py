"""Does the server actually *run* the declared warm-up samples?

The machinery — turning samples into batches, and refusing the ones that cannot work — is
`tests/repository/test_warmup.py`. This is the other half of the claim, and it is a claim about
the server: measured on the mock backend's own execution counter, so it counts executions that
happened rather than calls that were made.

It lived in the repository test file until #8 was split by seam, where standing the core piece
up alone showed it building an `InferenceServer` — which is not something a test of the warm-up
machinery has any business doing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shipinfer.core.errors import ServerStateError
from shipinfer.core.settings import ServerSettings
from shipinfer.server import InferenceServer

_MODEL = """
platform: mock
max_batch_size: 4
inputs: [{{name: x, data_type: FP32, dims: [2]}}]
outputs: [{{name: y, data_type: FP32, dims: [2]}}]
instance_groups: [{{kind: KIND_CPU, count: 1}}]
dynamic_batching: {{enabled: false}}
parameters: {{latency_ms: 0.0}}
{warmup}
"""


class TestTheServerRunsTheSamples:
    """The wiring, measured on the mock backend's own execution counter."""

    def _server(self, tmp_path: Path, warmup: str, iterations: int = 0) -> InferenceServer:
        root = tmp_path / "repo"
        (root / "m" / "1").mkdir(parents=True)
        (root / "m" / "config.yaml").write_text(_MODEL.format(warmup=warmup).lstrip())
        return InferenceServer(
            ServerSettings(
                model_repository=root,
                devices={"visible_gpus": []},
                execution={"warmup_iterations": iterations},
            )
        )

    def _executions(self, server: InferenceServer) -> int:
        (instance,) = server.model("m").instances
        return instance.stats()["backend"]["executions"]

    def test_without_samples_the_iteration_count_still_governs(self, tmp_path: Path) -> None:
        with self._server(tmp_path, "", iterations=2) as server:
            assert self._executions(server) == 2

    def test_declared_samples_run_count_times_each(self, tmp_path: Path) -> None:
        warmup = (
            "model_warmup:\n"
            "  - {name: small, batch_size: 1, count: 2, inputs: {x: {zero_data: true}}}\n"
            "  - {name: full, batch_size: 4, count: 3, inputs: {x: {random_data: true}}}\n"
        )

        with self._server(tmp_path, warmup, iterations=0) as server:
            assert self._executions(server) == 5

    def test_samples_are_not_cancelled_by_a_deployment_wide_count_of_zero(
        self, tmp_path: Path
    ) -> None:
        """The count is a *deployment* knob for the implicit warm-up. A per-model
        instruction silently overruled by a deployment default is the settings split
        backwards — and the test above already runs with ``warmup_iterations=0``, so this
        states the rule rather than repeating the measurement."""
        warmup = (
            "model_warmup: [{name: s, batch_size: 2, count: 1, inputs: {x: {zero_data: true}}}]"
        )

        with self._server(tmp_path, warmup, iterations=0) as server:
            assert self._executions(server) == 1

    def test_a_sample_that_cannot_run_stops_the_model_becoming_ready(
        self, tmp_path: Path
    ) -> None:
        """An operator's explicit instruction not being carried out must not leave a model
        that believes it is warm — the first p99 after the deploy would be uninterpretable."""
        warmup = (
            "model_warmup: [{name: from_file, batch_size: 1, "
            "inputs: {x: {input_data_file: missing.bin}}}]"
        )

        with pytest.raises(ServerStateError, match=r"missing\.bin"):
            self._server(tmp_path, warmup).start()


class TestABrokenSampleFailsStartUpFastNotAfterTheTimeout:
    """#10's review: the instance closes its queue with the error immediately, but if nothing
    woke the waiter, `wait_ready(remaining)` would burn the whole `timeout_s` and *then* raise.
    A typo in `input_data_file` would then be a silent two-minute stall for a fault known in
    the first second.

    The machinery for the fast path exists — the worker sets `_settled` on a failed start and
    `wait_ready` returns as soon as it fires — so this pins that it actually engages for a
    warm-up failure, which is the common, config-caused case that routes into it.
    """

    def test_a_missing_sample_file_raises_within_seconds(self, tmp_path: Path) -> None:
        import time

        root = tmp_path / "repo"
        (root / "m" / "1").mkdir(parents=True)
        (root / "m" / "config.yaml").write_text(
            _MODEL.format(
                warmup=(
                    "model_warmup:\n"
                    "  - name: real\n"
                    "    batch_size: 1\n"
                    "    count: 1\n"
                    "    inputs: {x: {input_data_file: does_not_exist.bin}}"
                )
            ).lstrip()
        )
        server = InferenceServer(
            ServerSettings(
                model_repository=root,
                devices={"visible_gpus": []},
                execution={"warmup_iterations": 0},
            )
        )

        started = time.monotonic()
        with pytest.raises(ServerStateError, match="does_not_exist"):
            server.start()
        elapsed = time.monotonic() - started

        assert elapsed < 5.0, (
            f"start-up took {elapsed:.1f}s to report a fault that was known immediately; "
            f"the readiness waiter was not woken by the failure"
        )
        server.stop()
