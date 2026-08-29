"""Declared warm-up samples, executed by a real TensorRT engine — the GPU half of the claim.

`tests/backends/test_warmup_wiring.py` proves the wiring against a stub; `tests/engine/
test_warmup_runs_on_the_server.py` proves the server runs samples against a real TorchScript fixture.
Neither drives `warmup()` through a real backend with `model_warmup` set, so without this file the
first execution of a declared sample against a real engine would happen in production. Review of
#10 said exactly that, and it is the house rule anyway: a claim about the data plane is checked on
the data plane.

The engine is `ship_detector`'s `model.plan` — yolo26n, built per machine and gitignored. It is
**static**: probed with TensorRT directly, the input is `(8, 3, 640, 640)`, not dynamic, one
optimisation profile. A declared sample at batch 4 nonetheless executes, because the backend
copies `batch_size` rows into a binding sized for the plan and runs the plan as built — so
"the engine refuses other batch sizes" is not a negative case this backend can show, and the
first draft of this file asserted it would be. The second draft then *guessed* the opposite
("the plan reports dynamic shapes") from the fact that batch 4 ran; that was also written from
inference rather than measurement, and the probe corrected it. The negative case that is true
for a real engine: a declared sample whose data file is the wrong size for the shape it claims.
That is refused at the engine boundary, names the sample, and must arrive in seconds.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from shipinfer.core.errors import ServerStateError
from shipinfer.core.settings import ServerSettings
from shipinfer.engine import InferenceServer

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "model_repository" / "ship_detector" / "1" / "model.plan"
#: Physical ordinal. Any device works; 5 is the one this box's operator keeps free for tests.
DEVICE = 5

_CONFIG = """
name: ship_detector
platform: tensorrt
max_batch_size: 8
inputs:
  - name: images
    data_type: FP32
    dims: [3, 640, 640]
outputs:
  - name: output0
    data_type: FP32
    dims: [300, 6]
instance_groups: [{{kind: KIND_GPU, count: 1}}]
dynamic_batching: {{enabled: false}}
model_warmup:
{samples}
"""


def _sample(name: str, batch_size: int, *, random: bool = False) -> str:
    source = "random_data: true" if random else "zero_data: true"
    return (
        f"  - name: {name}\n"
        f"    batch_size: {batch_size}\n"
        f"    count: 1\n"
        f"    inputs: {{images: {{{source}}}}}"
    )


def _repository(tmp_path: Path, samples: list[str]) -> Path:
    root = tmp_path / "repo"
    version = root / "ship_detector" / "1"
    version.mkdir(parents=True)
    # A copy rather than a symlink: the warm-up path resolves data files inside the version
    # directory and refuses anything that escapes it, and a symlink target outside is exactly
    # the shape that check exists to refuse.
    shutil.copy2(PLAN, version / "model.plan")
    (root / "ship_detector" / "config.yaml").write_text(
        _CONFIG.format(samples="\n".join(samples)).lstrip()
    )
    return root


def _server(root: Path) -> InferenceServer:
    return InferenceServer(
        ServerSettings(
            model_repository=root,
            devices={"visible_gpus": [DEVICE]},
            execution={"warmup_iterations": 0},
        )
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not PLAN.is_file(), reason="ship_detector's engine is not built on this box"
)
class TestDeclaredSamplesRunOnTheRealEngine:
    def test_two_samples_execute_exactly_twice_and_the_model_is_ready(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Counted by a spy that *delegates* to the real `execute`, so what is counted is the
        real engine running — not a stub standing in for it. `warmup_iterations=0` so the only
        executions before readiness are the declared ones."""
        from shipinfer.backends.tensorrt.backend import TensorRTBackend

        calls: list[int] = []
        real = TensorRTBackend.execute

        def counting(self, inputs, batch_size):
            calls.append(batch_size)
            return real(self, inputs, batch_size)

        monkeypatch.setattr(TensorRTBackend, "execute", counting)
        root = _repository(tmp_path, [_sample("zeros", 8), _sample("noise", 8, random=True)])
        server = _server(root)
        try:
            server.start()

            assert server.is_ready
            assert calls == [
                8,
                8,
            ], f"expected the two declared samples and nothing else: {calls}"
        finally:
            server.stop()

    def test_a_wrong_sized_data_file_fails_start_up_fast_and_names_the_sample(
        self, tmp_path: Path
    ) -> None:
        """The real engine is loaded, then the declared sample cannot be built: the file holds
        one row where the sample claims eight. A warm-up that quietly did not happen is worse
        than none, so the instance never reports ready — and the fault is known in the first
        second, so the operator must not wait out a timeout to hear about it."""
        import numpy as np

        root = _repository(
            tmp_path,
            [
                "  - name: real_crops\n"
                "    batch_size: 8\n"
                "    count: 1\n"
                "    inputs: {images: {input_data_file: crops.bin}}"
            ],
        )
        # One frame's worth of bytes, for a sample that promises eight.
        np.zeros((1, 3, 640, 640), dtype=np.float32).tofile(
            root / "ship_detector" / "1" / "crops.bin"
        )
        server = _server(root)

        started = time.monotonic()
        try:
            with pytest.raises(ServerStateError, match="real_crops"):
                server.start()
        finally:
            server.stop()
        elapsed = time.monotonic() - started

        assert elapsed < 30.0, f"a wrong-sized sample took {elapsed:.0f}s to surface"
