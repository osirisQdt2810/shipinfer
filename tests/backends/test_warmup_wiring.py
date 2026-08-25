"""Declared warm-up samples run at instance start-up, and they outrank the iteration count.

The only behaviour this piece changes, so it is the one thing this piece must prove. Two paths
in `ModelBackend.warmup`, decided by the model's own config: `model_warmup` samples when there
are any, `iterations` zero-filled batches otherwise. They differ in how they fail on purpose —
an implicit batch that cannot be built is a guess that did not work out, a declared sample that
fails is an operator's instruction not being carried out.

Tested against a stub backend rather than through the server, because the server is another
piece and because what is asserted here is *which executions happened*, which a stub can count
exactly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shipinfer.backends.base import ModelBackend
from shipinfer.core.errors import ConfigurationError, InferenceError, ShipInferError
from shipinfer.core.types import Tensor
from shipinfer.repository import ModelConfig


class RecordingBackend(ModelBackend):
    """Counts executions; fails on demand; needs no device."""

    platform = "recording"
    requires_gpu = False

    def __init__(self, context, *, fail_with: Exception | None = None) -> None:
        super().__init__(context)
        self.batches: list[int] = []
        self._fail_with = fail_with

    def _do_initialize(self) -> None:
        pass

    def execute(self, inputs: dict[str, Tensor], batch_size: int) -> dict[str, Tensor]:
        if self._fail_with is not None:
            raise self._fail_with
        self.batches.append(batch_size)
        return {}

    def _warmup_batch(self) -> dict[str, Tensor]:
        """The implicit path's batch, without needing `input_specs` from a real artefact."""
        return {"x": Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))}


def _config(samples: list[dict] | None = None) -> ModelConfig:
    base = {
        "name": "m",
        "platform": "recording",
        "max_batch_size": 4,
        "inputs": [{"name": "x", "data_type": "FP32", "dims": [2]}],
        "outputs": [{"name": "y", "data_type": "FP32", "dims": [2]}],
    }
    if samples is not None:
        base["model_warmup"] = samples
    return ModelConfig(**base)


def _sample(name: str, count: int, batch_size: int = 2) -> dict:
    return {
        "name": name,
        "batch_size": batch_size,
        "count": count,
        "inputs": {"x": {"zero_data": True}},
    }


def _backend(tmp_path: Path, config: ModelConfig, **kwargs) -> RecordingBackend:
    context = SimpleNamespace(
        config=config,
        artifact=SimpleNamespace(path=tmp_path, config=config, name="m", version=1),
        device=SimpleNamespace(is_cuda=False),
        instance_name="m:0",
    )
    return RecordingBackend(context, **kwargs)


class TestDeclaredSamplesOutrankTheIterationCount:
    def test_declared_samples_run_count_times_each(self, tmp_path: Path) -> None:
        backend = _backend(tmp_path, _config([_sample("a", 2), _sample("b", 3, batch_size=4)]))

        backend.warmup(iterations=0)

        assert backend.batches == [2, 2, 4, 4, 4]

    def test_a_deployment_wide_zero_does_not_cancel_a_per_model_instruction(
        self, tmp_path: Path
    ) -> None:
        """`execution.warmup_iterations: 0` now means "no *implicit* warm-up". A model that
        declared samples still runs them — the count is a deployment knob and a per-model
        instruction outranks it."""
        backend = _backend(tmp_path, _config([_sample("a", 1)]))

        backend.warmup(iterations=0)

        assert len(backend.batches) == 1

    def test_the_iteration_count_is_ignored_when_samples_are_declared(
        self, tmp_path: Path
    ) -> None:
        backend = _backend(tmp_path, _config([_sample("a", 2)]))

        backend.warmup(iterations=7)

        assert len(backend.batches) == 2, "iterations must not add implicit batches on top"


class TestWithoutSamplesTheOldBehaviourStands:
    def test_iterations_zero_filled_batches_of_one(self, tmp_path: Path) -> None:
        backend = _backend(tmp_path, _config())

        backend.warmup(iterations=3)

        assert backend.batches == [1, 1, 1]

    def test_zero_iterations_means_no_warm_up_at_all(self, tmp_path: Path) -> None:
        backend = _backend(tmp_path, _config())

        backend.warmup(iterations=0)

        assert backend.batches == []


class TestADeclaredSampleThatFailsIsLoud:
    """An implicit batch that cannot be built is skipped with a debug line. A declared one that
    fails propagates, so the instance never reports ready — a model that believes it is warm
    and is not gives a first p99 nobody can interpret."""

    def test_an_arbitrary_failure_becomes_an_inference_error_naming_the_sample(
        self, tmp_path: Path
    ) -> None:
        backend = _backend(
            tmp_path, _config([_sample("regular", 1)]), fail_with=RuntimeError("cuBLAS")
        )

        with pytest.raises(InferenceError, match="'regular'") as caught:
            backend.warmup(iterations=0)

        assert isinstance(caught.value.__cause__, RuntimeError)

    def test_a_typed_failure_passes_through_unchanged(self, tmp_path: Path) -> None:
        """The project's own errors are not re-wrapped. A `DeviceOutOfMemoryError` and a
        broken sample are different operational events — one is fixed by a smaller batch,
        the other by editing the config — and flattening both into `InferenceError` here,
        then again in the instance, leaves an operator with one word for both."""
        typed = ConfigurationError("the sample's data is the wrong shape")
        backend = _backend(tmp_path, _config([_sample("regular", 1)]), fail_with=typed)

        with pytest.raises(ConfigurationError) as caught:
            backend.warmup(iterations=0)

        assert caught.value is typed, "the typed error must be the very object, not a wrapper"
        assert any(
            "regular" in note for note in getattr(caught.value, "__notes__", [])
        ), "the sample's name must travel with the error as a note"
        assert isinstance(typed, ShipInferError)
