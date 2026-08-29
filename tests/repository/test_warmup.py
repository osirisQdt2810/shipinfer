"""Warm-up from declared samples, rather than from a count of zero-filled batches.

An iteration count decides *how often* a model is warmed. What actually selects the kernels
is *what* it is warmed with: a TensorRT engine picks tactics per shape, CUDA modules load
lazily per kernel, and a detector that never sees a box during warm-up has not warmed its
NMS. Triton's answer is ``model_warmup`` in the model config, so that is the key and the
semantics used here.

Two failure modes are pinned deliberately, because both would otherwise be silent: a sample
whose data file is the wrong size, and a sample that does not cover the model's inputs. A
warm-up that quietly did not happen is worse than none, because the deployment then believes
its first p99 is representative.

This file covers the *machinery* — turning declared samples into batches, and refusing the
ones that cannot work. Whether the server actually runs them is a claim about the server, and
it is asserted in ``tests/server/test_warmup_wiring.py`` against the mock backend's own
execution counter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.repository import ModelConfig, build_warmup_batches


def _config(**overrides) -> ModelConfig:
    base = {
        "name": "m",
        "platform": "pytorch",
        "max_batch_size": 4,
        "inputs": [{"name": "x", "data_type": "FP32", "dims": [2, 3]}],
        "outputs": [{"name": "y", "data_type": "FP32", "dims": [2]}],
    }
    base.update(overrides)
    return ModelConfig(**base)


def _sample(**overrides) -> dict:
    sample = {
        "name": "regular",
        "batch_size": 2,
        "count": 3,
        "inputs": {"x": {"zero_data": True}},
    }
    sample.update(overrides)
    return sample


class TestBuildingTheBatches:
    """Config in, tensors out — with no backend and no accelerator involved."""

    def test_no_samples_means_no_batches(self, tmp_path: Path) -> None:
        """An empty tuple is the caller's signal to fall back to the implicit warm-up, which
        is why this is not an error."""
        assert build_warmup_batches(_config(), tmp_path) == ()

    def test_zero_data_is_shaped_from_the_model_declaration(self, tmp_path: Path) -> None:
        config = _config(model_warmup=[_sample()])

        (batch,) = build_warmup_batches(config, tmp_path)

        assert batch.name == "regular"
        assert batch.count == 3
        assert batch.batch_size == 2
        assert batch.inputs["x"].shape == (2, 2, 3)
        assert not batch.inputs["x"].host.any()

    def test_random_data_is_not_zeros_and_is_reproducible(self, tmp_path: Path) -> None:
        """A warm-up that differs run to run makes two deployments of one engine
        incomparable."""
        config = _config(model_warmup=[_sample(inputs={"x": {"random_data": True}})])

        first = build_warmup_batches(config, tmp_path)[0].inputs["x"].host
        second = build_warmup_batches(config, tmp_path)[0].inputs["x"].host

        assert first.any()
        assert np.array_equal(first, second)

    def test_a_data_file_is_read_from_the_version_directory(self, tmp_path: Path) -> None:
        payload = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        (tmp_path / "sample.bin").write_bytes(payload.tobytes())
        config = _config(
            model_warmup=[_sample(inputs={"x": {"input_data_file": "sample.bin"}})]
        )

        (batch,) = build_warmup_batches(config, tmp_path)

        assert np.array_equal(batch.inputs["x"].host, payload)

    def test_the_tensor_from_a_file_is_writeable(self, tmp_path: Path) -> None:
        """``frombuffer`` yields a read-only view, and a backend that stages into its input
        would fail on it — at load, for a reason with nothing to do with the model."""
        (tmp_path / "s.bin").write_bytes(np.zeros((2, 2, 3), dtype=np.float32).tobytes())
        config = _config(model_warmup=[_sample(inputs={"x": {"input_data_file": "s.bin"}})])

        (batch,) = build_warmup_batches(config, tmp_path)

        batch.inputs["x"].host[0, 0, 0] = 1.0  # must not raise

    def test_dims_may_override_the_declaration_for_a_dynamic_extent(
        self, tmp_path: Path
    ) -> None:
        config = _config(
            inputs=[{"name": "x", "data_type": "FP32", "dims": [-1, 3]}],
            model_warmup=[_sample(inputs={"x": {"zero_data": True, "dims": [8, 3]}})],
        )

        (batch,) = build_warmup_batches(config, tmp_path)

        assert batch.inputs["x"].shape == (2, 8, 3)


class TestFailuresAreLoudAndSpecific:
    """Every message here names the model, the sample and the tensor."""

    def test_a_missing_data_file_names_the_path(self, tmp_path: Path) -> None:
        config = _config(
            model_warmup=[_sample(inputs={"x": {"input_data_file": "absent.bin"}})]
        )

        with pytest.raises(ConfigurationError, match=r"absent\.bin"):
            build_warmup_batches(config, tmp_path)

    def test_a_data_file_of_the_wrong_size_says_what_was_expected(self, tmp_path: Path) -> None:
        """``frombuffer`` on a file with one extra byte raises about buffer lengths, which
        tells an operator nothing about which model or what shape."""
        (tmp_path / "short.bin").write_bytes(np.zeros(5, dtype=np.float32).tobytes())
        config = _config(model_warmup=[_sample(inputs={"x": {"input_data_file": "short.bin"}})])

        with pytest.raises(ConfigurationError, match="needs 48"):
            build_warmup_batches(config, tmp_path)

    def test_a_sample_needs_exactly_one_data_source(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            _config(
                model_warmup=[_sample(inputs={"x": {"zero_data": True, "random_data": True}})]
            )

        with pytest.raises(ValueError, match="exactly one"):
            _config(model_warmup=[_sample(inputs={"x": {}})])

    def test_a_sample_must_cover_every_required_input(self) -> None:
        with pytest.raises(ValueError, match="missing required input"):
            _config(
                inputs=[
                    {"name": "x", "data_type": "FP32", "dims": [2, 3]},
                    {"name": "mask", "data_type": "INT32", "dims": [2]},
                ],
                model_warmup=[_sample()],
            )

    def test_an_unknown_input_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not declare"):
            _config(model_warmup=[_sample(inputs={"nope": {"zero_data": True}})])

    def test_a_sample_above_max_batch_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="above max_batch_size"):
            _config(model_warmup=[_sample(batch_size=99)])

    def test_a_dynamic_extent_needs_explicit_dims(self) -> None:
        """There is nothing to infer, and guessing 1 would warm a shape the model never
        serves."""
        with pytest.raises(ValueError, match="dynamic extent"):
            _config(
                inputs=[{"name": "x", "data_type": "FP32", "dims": [-1, 3]}],
                model_warmup=[_sample()],
            )

    def test_an_ensemble_cannot_declare_warm_up(self) -> None:
        with pytest.raises(ValueError, match="ensemble"):
            ModelConfig(
                name="pipe",
                platform="ensemble",
                ensemble={"steps": [{"model": "m"}]},
                dynamic_batching={"enabled": False},
                model_warmup=[_sample()],
            )
