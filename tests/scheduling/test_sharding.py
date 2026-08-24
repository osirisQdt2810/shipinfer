"""Assigning cameras and GPUs to processes.

Pure logic, offline by design: this decides how a 16-GPU box is carved up, and the decision has
to be reviewable on a laptop. Every test here is about one of two properties — **stability**,
because a camera's actor holds state that moving it would throw away, and **balance**, because
a shard is a process and a process is a wall, so the fleet is bounded by its busiest shard.
"""

from __future__ import annotations

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.scheduling.sharding import plan_shards


class TestEveryCameraIsAssignedExactlyOnce:
    def test_nothing_is_lost_and_nothing_is_duplicated(self) -> None:
        cameras = [f"cam{i:02d}" for i in range(50)]

        plan = plan_shards(cameras, shards=4, gpus=[2, 3, 4, 5])

        assert plan.cameras == tuple(sorted(cameras))
        assert sum(len(s.cameras) for s in plan.shards) == 50

    def test_one_shard_is_legal_and_means_no_split(self) -> None:
        """So a caller has one code path rather than two."""
        plan = plan_shards(["a", "b", "c"], shards=1, gpus=[0, 1])

        assert len(plan) == 1
        assert plan.shards[0].cameras == ("a", "b", "c")
        assert plan.shards[0].gpus == (0, 1)
        assert plan.imbalance == 0.0


class TestTheAssignmentIsStable:
    """Ingest is stateful per camera (ADR-011): reconnect backoff, frame-id watermark, health
    history. A camera that changes shard on a restart throws all of that away."""

    def test_the_same_fleet_gives_the_same_plan(self) -> None:
        cameras = {f"cam{i}": float(i % 5 + 1) for i in range(20)}

        first = plan_shards(cameras, shards=3, gpus=[0, 1])
        second = plan_shards(cameras, shards=3, gpus=[0, 1])

        assert [s.cameras for s in first.shards] == [s.cameras for s in second.shards]

    def test_input_order_does_not_change_the_plan(self) -> None:
        """A dict from a config file has whatever order the file had."""
        forward = {f"cam{i}": 10.0 for i in range(12)}
        backward = {f"cam{i}": 10.0 for i in reversed(range(12))}

        assert [s.cameras for s in plan_shards(forward, shards=3, gpus=[0]).shards] == [
            s.cameras for s in plan_shards(backward, shards=3, gpus=[0]).shards
        ]


class TestBalanceIsByLoadNotByCount:
    """The failure this project exists to fix is a busy camera starving a quiet one. Splitting
    by count would rebuild it one level up: all the 30 fps cameras in one process."""

    def test_a_skewed_fleet_is_balanced_by_offered_fps(self) -> None:
        # Four busy cameras and twelve quiet ones. By count, 4 shards of 4 would put every busy
        # camera in the first shard if the order were unlucky.
        cameras = {f"busy{i}": 30.0 for i in range(4)}
        cameras.update({f"quiet{i}": 5.0 for i in range(12)})

        plan = plan_shards(cameras, shards=4, gpus=[0, 1, 2, 3])

        assert plan.imbalance < 0.10, plan.describe()
        for shard in plan.shards:
            assert shard.offered_fps == pytest.approx(45.0, abs=6.0)

    def test_a_uniform_fleet_splits_evenly(self) -> None:
        plan = plan_shards([f"cam{i}" for i in range(12)], shards=4, gpus=[0, 1])

        assert {len(s.cameras) for s in plan.shards} == {3}
        assert plan.imbalance == 0.0

    def test_largest_first_beats_the_naive_order(self) -> None:
        """One 100 fps camera and four 10 fps ones over two shards. Greedy-largest-first puts
        the big one alone; anything else pairs it with a small one and wastes a shard."""
        cameras = {"big": 100.0, **{f"small{i}": 10.0 for i in range(4)}}

        plan = plan_shards(cameras, shards=2, gpus=[0, 1])

        big_shard = next(s for s in plan.shards if "big" in s.cameras)
        assert big_shard.cameras == ("big",)

    def test_imbalance_is_reported_not_hidden(self) -> None:
        """A plan that cannot be balanced still has to say so — the fleet is bounded by its
        busiest shard, so an imbalanced plan wastes the capacity it looks like it added."""
        plan = plan_shards({"huge": 100.0, "tiny": 1.0}, shards=2, gpus=[0])

        assert plan.imbalance == pytest.approx(0.99)
        assert "imbalance 99" in plan.describe()


class TestGpuAssignment:
    def test_one_device_each_when_the_counts_match(self) -> None:
        plan = plan_shards([f"cam{i}" for i in range(8)], shards=4, gpus=[2, 3, 4, 5])

        assert [s.gpus for s in plan.shards] == [(2,), (3,), (4,), (5,)]
        assert plan.shards_per_gpu == {2: 1, 3: 1, 4: 1, 5: 1}

    def test_fewer_shards_than_gpus_gets_contiguous_groups(self) -> None:
        """Contiguous so a shard's devices are as close together as the topology allows, and
        so no device is left idle."""
        plan = plan_shards([f"cam{i}" for i in range(4)], shards=2, gpus=[0, 1, 2, 3])

        assert [s.gpus for s in plan.shards] == [(0, 1), (2, 3)]

    def test_more_shards_than_gpus_shares_devices_round_robin(self) -> None:
        """Deliberately allowed: the measured bottleneck was CPU, so the useful number of
        shards is set by cores rather than by devices."""
        plan = plan_shards([f"cam{i}" for i in range(8)], shards=4, gpus=[0, 1])

        assert [s.gpus for s in plan.shards] == [(0,), (1,), (0,), (1,)]
        assert plan.shards_per_gpu == {0: 2, 1: 2}

    def test_an_uneven_group_split_leaves_no_gpu_out(self) -> None:
        plan = plan_shards([f"cam{i}" for i in range(6)], shards=2, gpus=[0, 1, 2])

        assigned = [g for s in plan.shards for g in s.gpus]
        assert sorted(assigned) == [0, 1, 2]

    def test_the_cuda_visible_devices_string_is_what_the_child_exports(self) -> None:
        plan = plan_shards([f"cam{i}" for i in range(4)], shards=2, gpus=[4, 5, 6, 7])

        assert plan.shards[0].cuda_visible_devices == "4,5"
        assert plan.shards[1].cuda_visible_devices == "6,7"


class TestInstancesAreDividedNotRepeated:
    """Two shards on one device each loading the configured count would double that device's
    engines and its VRAM for the same total throughput, and TensorRT contexts are per-process
    so there is no sharing to save it."""

    def test_sharing_a_device_halves_each_shard_s_instances(self) -> None:
        plan = plan_shards([f"cam{i}" for i in range(8)], shards=4, gpus=[0, 1])

        assert plan.instances_for(4, gpu=0) == 2

    def test_a_device_to_itself_keeps_the_configured_count(self) -> None:
        plan = plan_shards([f"cam{i}" for i in range(4)], shards=2, gpus=[0, 1])

        assert plan.instances_for(2, gpu=0) == 2

    def test_a_division_that_would_give_zero_instances_is_refused(self) -> None:
        """Rounding to zero produces a shard that accepts frames and can never execute one,
        which looks like a throughput result rather than a misconfiguration."""
        plan = plan_shards([f"cam{i}" for i in range(4)], shards=4, gpus=[0])

        with pytest.raises(ConfigurationError, match="each shard would get"):
            plan.instances_for(2, gpu=0)


class TestItRefusesAPlanThatCannotWork:
    def test_an_empty_fleet(self) -> None:
        with pytest.raises(ConfigurationError, match="empty fleet"):
            plan_shards([], shards=1, gpus=[0])

    def test_no_gpus(self) -> None:
        with pytest.raises(ConfigurationError, match="no gpus"):
            plan_shards(["cam0"], shards=1, gpus=[])

    def test_zero_shards(self) -> None:
        with pytest.raises(ConfigurationError, match="at least one shard"):
            plan_shards(["cam0"], shards=0, gpus=[0])

    def test_more_shards_than_cameras(self) -> None:
        """A shard with no cameras is a process that starts, loads engines, holds a CUDA
        context of 220-480 MiB, and reads nothing."""
        with pytest.raises(ConfigurationError, match="at most 3 shards"):
            plan_shards(["a", "b", "c"], shards=4, gpus=[0])
