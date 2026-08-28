"""Model config validation — the errors a deployment should hit at commit time."""

from __future__ import annotations

from pathlib import Path

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.types import DataType, Device
from shipinfer.repository import InstanceGroup, InstanceKind, ModelConfig, load_model_config

BASE = {
    "name": "m",
    "platform": "mock",
    "max_batch_size": 8,
    "inputs": [{"name": "x", "data_type": "FP32", "dims": [4]}],
    "outputs": [{"name": "y", "data_type": "FP32", "dims": [2]}],
}


class TestBatchDimensionConvention:
    """max_batch_size owns dim 0, so a declared shape never mentions it."""

    def test_specs_exclude_the_batch_dimension(self) -> None:
        """Triton's convention, kept: ``max_batch_size`` owns dim 0."""
        config = ModelConfig(**BASE)
        assert config.input_specs[0].shape == (4,)
        assert config.input_specs[0].dtype is DataType.FP32
        assert config.effective_max_batch_size == 8

    def test_dynamic_batching_requires_a_batch_dimension(self) -> None:
        with pytest.raises(Exception, match="dynamic_batching needs max_batch_size"):
            ModelConfig(**{**BASE, "max_batch_size": 0})


class TestConfigRejectsBadWiring:
    """A config that could not work is refused at load, with the reason in the message."""

    def test_preferred_sizes_may_not_exceed_the_maximum(self) -> None:
        with pytest.raises(Exception, match="exceeds max_batch_size"):
            ModelConfig(**{**BASE, "dynamic_batching": {"preferred_batch_sizes": [4, 64]}})

    def test_duplicate_tensor_names_are_refused(self) -> None:
        with pytest.raises(Exception, match="duplicate tensor"):
            ModelConfig(
                **{**BASE, "outputs": [{"name": "x", "data_type": "FP32", "dims": [2]}]}
            )

    def test_ensemble_requires_steps(self) -> None:
        with pytest.raises(Exception, match="requires an `ensemble:` section"):
            ModelConfig(name="e", platform="ensemble", max_batch_size=0)


class TestInstanceGroupExpansion:
    """``count`` is per device, exactly as in Triton — four instances from count=2 over two GPUs."""

    def test_count_is_per_device(self) -> None:
        group = InstanceGroup(kind=InstanceKind.GPU, count=2, gpus=[0, 1])
        placements = group.expand(visible_gpus=(0, 1))
        assert len(placements) == 4
        assert sorted(p.device.index for p in placements) == [0, 0, 1, 1]

    def test_a_shared_device_gets_the_share_not_the_count(self) -> None:
        """A fleet shard that shares its GPU loads count // sharing there. Two shards each
        loading the full count would double the device's engines and VRAM for the same total
        throughput — TensorRT contexts are per-process, so nothing saves it."""
        group = InstanceGroup(kind=InstanceKind.GPU, count=4, gpus=[0, 1])
        placements = group.expand(visible_gpus=(0, 1), shared_by={0: 2})
        assert sorted(p.device.index for p in placements) == [0, 0, 1, 1, 1, 1]

    def test_a_count_that_does_not_divide_gives_its_remainder_to_the_lowest_ranks(self) -> None:
        """`count: 3` over two processes is 2 + 1 — the device still carries the three the
        config asked for, not the two floor division would leave it."""
        group = InstanceGroup(kind=InstanceKind.GPU, count=3, gpus=[0])
        first = group.expand(visible_gpus=(0,), shared_by={0: 2}, share_rank={0: 0})
        second = group.expand(visible_gpus=(0,), shared_by={0: 2}, share_rank={0: 1})
        assert (len(first), len(second)) == (2, 1)

    def test_an_absent_rank_is_rank_zero(self) -> None:
        group = InstanceGroup(kind=InstanceKind.GPU, count=3, gpus=[0])
        assert len(group.expand(visible_gpus=(0,), shared_by={0: 2})) == 2

    def test_a_rank_that_would_get_nothing_is_refused(self) -> None:
        group = InstanceGroup(kind=InstanceKind.GPU, count=1, gpus=[0])
        with pytest.raises(ConfigurationError, match="ranked 1 would get none"):
            group.expand(visible_gpus=(0,), shared_by={0: 2}, share_rank={0: 1})

    def test_a_single_instance_shared_by_two_goes_to_rank_zero(self) -> None:
        # Rank 0 takes the remainder; rank 1 is the one refused (next test) — silently
        # rounding to zero would produce a process that accepts frames and can never execute
        # one, which reads as a throughput result rather than a misconfiguration.
        group = InstanceGroup(kind=InstanceKind.GPU, count=1, gpus=[0])
        assert len(group.expand(visible_gpus=(0,), shared_by={0: 2})) == 1

    def test_placements_pass_the_sharing_through(self) -> None:
        config = ModelConfig(
            **{**BASE, "instance_groups": [InstanceGroup(kind=InstanceKind.GPU, count=2)]}
        )
        assert len(config.placements((0, 1), shared_by={0: 2, 1: 2})) == 2
        assert len(config.placements((0, 1))) == 4

    def test_empty_gpus_means_every_visible_device(self) -> None:
        """The property that lets one config run on 2 GPUs and on 16 unchanged."""
        group = InstanceGroup(kind=InstanceKind.GPU, count=1)
        assert len(group.expand(visible_gpus=(0, 1, 2, 3))) == 4
        assert len(group.expand(visible_gpus=(0, 1))) == 2

    def test_auto_falls_back_to_cpu(self) -> None:
        placements = InstanceGroup(kind=InstanceKind.AUTO, count=3).expand(visible_gpus=())
        assert [p.device for p in placements] == [Device.cpu()] * 3

    def test_requesting_an_invisible_gpu_fails_loudly(self) -> None:
        group = InstanceGroup(kind=InstanceKind.GPU, gpus=[7])
        with pytest.raises(ConfigurationError, match="not visible"):
            group.expand(visible_gpus=(0, 1))

    def test_gpu_group_with_no_devices_fails_loudly(self) -> None:
        with pytest.raises(ConfigurationError, match="none are visible"):
            InstanceGroup(kind=InstanceKind.GPU).expand(visible_gpus=())


class TestVersionPolicy:
    def test_latest_n(self) -> None:
        config = ModelConfig(**{**BASE, "version_policy": {"latest": 2}})
        assert config.version_policy.select([1, 2, 3, 4]) == [3, 4]

    def test_all(self) -> None:
        config = ModelConfig(**{**BASE, "version_policy": {"latest": None, "all": True}})
        assert config.version_policy.select([2, 1, 3]) == [1, 2, 3]

    def test_specific(self) -> None:
        config = ModelConfig(**{**BASE, "version_policy": {"latest": None, "specific": [1, 3]}})
        assert config.version_policy.select([1, 2, 3]) == [1, 3]

    def test_exactly_one_policy(self) -> None:
        with pytest.raises(Exception, match="exactly one"):
            ModelConfig(**{**BASE, "version_policy": {"latest": 1, "all": True}})


class TestLoadingFromDisk:
    """Loading from a path: the error names the file, and the directory names the model."""

    def test_load_attaches_the_path_to_an_error(self, tmp_path: Path) -> None:
        """A validation error with no path is nearly useless in a thirty-model repository."""
        bad = tmp_path / "config.yaml"
        bad.write_text("name: broken\nplatform: pytorch\nmax_batch_size: -1\n")
        with pytest.raises(ConfigurationError, match=str(bad)):
            load_model_config(bad)

    def test_load_defaults_the_name_to_the_directory(self, tmp_path: Path) -> None:
        directory = tmp_path / "implied_name"
        directory.mkdir()
        path = directory / "config.yaml"
        path.write_text(
            "platform: pytorch\nmax_batch_size: 1\n"
            "inputs: [{name: x, data_type: FP32, dims: [1]}]\n"
            "outputs: [{name: y, data_type: FP32, dims: [1]}]\n"
            "dynamic_batching: {enabled: false}\n"
        )
        assert load_model_config(path).name == "implied_name"
