"""Tests for the fleet-to-process plan.

Every test here runs on a laptop with no driver, which is the whole reason the decision lives
in ``scheduling/`` rather than in the launcher: *which* process owns *which* camera is the
question the previous system got wrong, and it is answerable without a GPU.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.scheduling.sharding import Shard, ShardPlan, plan_shards

#: A few busy cameras and many quiet ones — the shape the fleet actually has. Four at 30 fps
#: and twelve at 5 fps: 180 fps total, so a perfect two-way split is 90 each.
SKEWED = {f"busy{i}": 30.0 for i in range(4)} | {f"quiet{i}": 5.0 for i in range(12)}

#: The same shape with **odd** counts, which is the version that separates the two algorithms.
#: On ``SKEWED`` round-robin also lands on 90/90, because four and twelve both alternate
#: cleanly into two — so that fleet cannot tell greedy from round-robin, and a test that used
#: it to claim greedy is better would be asserting nothing. Five busy and eleven quiet is
#: 205 fps, which no alternation divides evenly.
ODD = {f"busy{i}": 30.0 for i in range(5)} | {f"quiet{i}": 5.0 for i in range(11)}


def round_robin(fleet: dict[str, float], shards: int, *, descending: bool) -> float:
    """The imbalance round-robin would have produced, so a comparison is a measurement.

    ``descending`` picks which round-robin: over the fleet sorted by load (the charitable
    version, and a real thing people write) or over the ids as they come.
    """
    order = sorted(fleet, key=lambda n: (-fleet[n], n)) if descending else sorted(fleet)
    loads = [0.0] * shards
    for i, name in enumerate(order):
        loads[i % shards] += fleet[name]
    return (max(loads) - min(loads)) / max(loads)


class TestThePlanIsAPartitionOfTheFleet:
    """Whatever else it does, it must lose no camera and duplicate none."""

    @pytest.mark.parametrize("shards", [1, 2, 3, 5, 16])
    def test_every_camera_lands_in_exactly_one_shard(self, shards: int) -> None:
        plan = plan_shards(SKEWED, shards=shards, gpus=[2, 3, 4, 5])

        assert plan.cameras == tuple(sorted(SKEWED))
        seen = [c for shard in plan.shards for c in shard.cameras]
        assert len(seen) == len(SKEWED), "a camera was duplicated across shards"

    def test_no_shard_is_empty(self) -> None:
        # A shard with no cameras is a process that loads engines, holds a CUDA context and
        # reads nothing. `plan_shards` refuses to create one; assert it also never does so by
        # accident at a legal shard count.
        plan = plan_shards(SKEWED, shards=len(SKEWED), gpus=[2, 3])

        assert all(shard.cameras for shard in plan.shards)

    def test_offered_fps_is_the_sum_of_the_shards_own_cameras(self) -> None:
        plan = plan_shards(SKEWED, shards=3, gpus=[2, 3, 4])

        for shard in plan.shards:
            assert shard.offered_fps == pytest.approx(sum(SKEWED[c] for c in shard.cameras))
        total = sum(shard.offered_fps for shard in plan.shards)
        assert total == pytest.approx(sum(SKEWED.values()))


class TestBalanceIsByLoadNotByCount:
    """The failure this project exists to fix, one level up."""

    def test_the_skewed_fleet_splits_within_a_frame_per_second_of_even(self) -> None:
        plan = plan_shards(SKEWED, shards=2, gpus=[2, 3])

        loads = sorted(shard.offered_fps for shard in plan.shards)
        assert loads == [90.0, 90.0], plan.describe()
        assert plan.imbalance == pytest.approx(0.0)

    @pytest.mark.parametrize("descending", [True, False])
    def test_it_beats_both_round_robins_on_a_fleet_that_does_not_alternate(
        self, descending: bool
    ) -> None:
        rr = round_robin(ODD, 2, descending=descending)

        plan = plan_shards(ODD, shards=2, gpus=[2, 3])

        assert rr > 0.0, "the fixture is too regular to tell the two apart"
        assert plan.imbalance < rr, plan.describe()

    def test_and_ties_it_on_a_fleet_that_does(self) -> None:
        # Stated rather than hidden: greedy is not better everywhere, and SKEWED is a fleet
        # where round-robin is already optimal. The claim is that greedy is never worse, and a
        # test suite that only showed the favourable fixture would not be evidence of that.
        assert round_robin(SKEWED, 2, descending=True) == pytest.approx(0.0)
        assert plan_shards(SKEWED, shards=2, gpus=[2, 3]).imbalance == pytest.approx(0.0)

    @pytest.mark.parametrize("shards", [2, 3, 4, 5])
    def test_greedy_is_never_worse_than_round_robin(self, shards: int) -> None:
        for fleet in (SKEWED, ODD):
            plan = plan_shards(fleet, shards=shards, gpus=[2, 3, 4, 5])
            best_rr = min(
                round_robin(fleet, shards, descending=True),
                round_robin(fleet, shards, descending=False),
            )
            equal_shares = len({len(s.gpus) for s in plan.shards}) == 1 and all(
                n == 1 for n in plan.shards_per_gpu.values()
            )
            if equal_shares:
                # Equal device shares: the per-shard figure is the one to beat.
                assert plan.imbalance <= best_rr + 1e-9, (shards, plan.describe())
            else:
                # Shards share devices, so per-shard loads are *meant* to differ; what must
                # hold is that no device is more than one camera's load above another.
                loads = list(plan.device_load.values())
                assert max(loads) - min(loads) <= max(fleet.values()) + 1e-9, plan.describe()

    def test_a_camera_count_split_would_have_been_worse(self) -> None:
        # Eight cameras each is what balancing by count gives, and on this fleet that puts
        # every busy camera in reach of one shard. The plan must not do that.
        plan = plan_shards(SKEWED, shards=2, gpus=[2, 3])

        busy_per_shard = sorted(
            sum(1 for c in shard.cameras if c.startswith("busy")) for shard in plan.shards
        )
        assert busy_per_shard == [2, 2]

    def test_a_uniform_fleet_given_as_a_bare_sequence_balances_by_count(self) -> None:
        plan = plan_shards([f"cam{i}" for i in range(12)], shards=4, gpus=[2, 3])

        assert sorted(len(s.cameras) for s in plan.shards) == [3, 3, 3, 3]
        assert plan.imbalance == pytest.approx(0.0)

    def test_one_shard_has_no_imbalance_to_report(self) -> None:
        plan = plan_shards(SKEWED, shards=1, gpus=[2, 3, 4, 5])

        assert len(plan) == 1
        assert plan.imbalance == 0.0
        assert plan.shards[0].cameras == tuple(sorted(SKEWED))

    def test_a_fleet_of_idle_cameras_does_not_divide_by_zero(self) -> None:
        plan = plan_shards({"a": 0.0, "b": 0.0}, shards=2, gpus=[2])

        assert plan.imbalance == 0.0


class TestThePlanIsStableAcrossRestarts:
    """Ingest is stateful per camera (ADR-011), so a shuffle on restart throws state away."""

    def test_the_same_fleet_produces_the_same_assignment(self) -> None:
        first = plan_shards(SKEWED, shards=3, gpus=[2, 3, 4, 5])
        second = plan_shards(SKEWED, shards=3, gpus=[2, 3, 4, 5])

        assert first == second

    def test_insertion_order_does_not_change_the_answer(self) -> None:
        # A config file read in a different order, or a dict built by a different code path,
        # must not move a camera between processes.
        shuffled = dict(reversed(list(SKEWED.items())))

        assert plan_shards(shuffled, shards=3, gpus=[2, 3]) == plan_shards(
            SKEWED, shards=3, gpus=[2, 3]
        )

    def test_cameras_are_sorted_within_a_shard(self) -> None:
        plan = plan_shards(SKEWED, shards=3, gpus=[2, 3])

        for shard in plan.shards:
            assert list(shard.cameras) == sorted(shard.cameras)


class TestGpusAreHandedOutWithoutLeavingOneIdle:
    def test_fewer_shards_than_gpus_gives_contiguous_groups_and_uses_them_all(self) -> None:
        plan = plan_shards(SKEWED, shards=2, gpus=[2, 3, 4, 5])

        assert [s.gpus for s in plan.shards] == [(2, 3), (4, 5)]
        assert plan.shards_per_gpu == {2: 1, 3: 1, 4: 1, 5: 1}

    def test_an_uneven_split_gives_the_remainder_to_the_early_shards(self) -> None:
        plan = plan_shards(SKEWED, shards=3, gpus=[2, 3, 4, 5])

        assert [s.gpus for s in plan.shards] == [(2, 3), (4,), (5,)]
        assert set(plan.shards_per_gpu) == {2, 3, 4, 5}

    def test_one_shard_per_gpu_when_the_counts_match(self) -> None:
        plan = plan_shards(SKEWED, shards=4, gpus=[2, 3, 4, 5])

        assert [s.gpus for s in plan.shards] == [(2,), (3,), (4,), (5,)]
        assert plan.shards_per_gpu == {2: 1, 3: 1, 4: 1, 5: 1}

    def test_more_shards_than_gpus_share_devices_as_evenly_as_the_counts_allow(self) -> None:
        plan = plan_shards(SKEWED, shards=6, gpus=[2, 3, 4, 5])

        assert [s.gpus for s in plan.shards] == [(2,), (3,), (4,), (5,), (2,), (3,)]
        assert plan.shards_per_gpu == {2: 2, 3: 2, 4: 1, 5: 1}

    def test_the_ordinals_are_physical_and_survive_to_the_environment_variable(self) -> None:
        # The child exports CUDA_VISIBLE_DEVICES itself, before it imports torch, and at that
        # moment no restriction is in force — so these have to be the host's own ordinals.
        shard = plan_shards(SKEWED, shards=2, gpus=[2, 3, 4, 5]).shards[1]

        assert shard.cuda_visible_devices == "4,5"

    def test_the_gpu_order_given_is_the_order_handed_out(self) -> None:
        plan = plan_shards(SKEWED, shards=2, gpus=[5, 4, 3, 2])

        assert [s.gpus for s in plan.shards] == [(5, 4), (3, 2)]


class TestLoadIsBalancedPerDeviceNotPerShard:
    """The device is what saturates. Six shards on four GPUs used to print an 11% *shard*
    imbalance while two GPUs carried 340 fps and two carried 160 — the founding bug rebuilt one
    level up, invisible to a metric that measured processes rather than the resource."""

    UNIFORM = {f"cam{i:02d}": 20.0 for i in range(50)}

    def test_the_reviewers_case_lands_within_one_camera_per_device(self) -> None:
        plan = plan_shards(self.UNIFORM, shards=6, gpus=[0, 1, 2, 3])

        loads = plan.device_load
        assert set(loads) == {0, 1, 2, 3}
        assert max(loads.values()) - min(loads.values()) <= 20.0, plan.describe()
        assert plan.device_imbalance < 0.1, plan.describe()
        # And the shards on unshared devices carry about twice what a sharing shard does.
        alone = [s.offered_fps for s in plan.shards if plan.shards_per_gpu[s.gpus[0]] == 1]
        sharing = [s.offered_fps for s in plan.shards if plan.shards_per_gpu[s.gpus[0]] == 2]
        assert min(alone) > max(sharing)

    def test_the_skewed_fleet_too(self) -> None:
        plan = plan_shards(SKEWED, shards=6, gpus=[2, 3, 4, 5])

        loads = list(plan.device_load.values())
        assert max(loads) - min(loads) <= 30.0, plan.describe()

    def test_equal_shares_reduce_to_the_old_plan(self) -> None:
        """With one shard per GPU the capacity weights are all 1 and nothing changes."""
        plan = plan_shards(SKEWED, shards=4, gpus=[2, 3, 4, 5])

        assert plan.device_imbalance == plan.imbalance
        assert plan.device_load == {s.gpus[0]: s.offered_fps for s in plan.shards}

    def test_a_shard_with_two_gpus_spreads_its_load_over_both(self) -> None:
        plan = plan_shards(SKEWED, shards=2, gpus=[2, 3, 4, 5])

        for shard in plan.shards:
            for gpu in shard.gpus:
                assert plan.device_load[gpu] == shard.offered_fps / 2

    def test_describe_names_every_device_and_the_figure_the_plan_is_judged_by(self) -> None:
        text = plan_shards(self.UNIFORM, shards=6, gpus=[0, 1, 2, 3]).describe()

        assert "device imbalance" in text and "shard imbalance" in text
        for gpu in range(4):
            assert f"  gpu {gpu}: " in text


class TestTheLauncherIsToldHowManyShardsShareEachGpu:
    """`sharing_for` is what rides to the child as `SHIPINFER_DEVICES__SHARED_BY`; the child
    divides every model's configured instance count by it where the instance groups expand."""

    def test_a_sole_owner_shares_with_nobody(self) -> None:
        plan = plan_shards(SKEWED, shards=4, gpus=[2, 3, 4, 5])

        assert all(plan.sharing_for(shard) == (1,) for shard in plan.shards)

    def test_two_shards_on_one_gpu_are_each_told_two(self) -> None:
        plan = plan_shards(SKEWED, shards=6, gpus=[2, 3, 4, 5])

        assert plan.shards_per_gpu[2] == 2
        sharing = [s for s in plan.shards if tuple(s.gpus) == (2,)]
        alone = [s for s in plan.shards if tuple(s.gpus) == (4,)]
        assert len(sharing) == 2 and all(plan.sharing_for(s) == (2,) for s in sharing)
        assert alone and plan.sharing_for(alone[0]) == (1,), "an unshared device is unaffected"

    def test_the_answer_is_aligned_with_the_shards_device_order(self) -> None:
        plan = plan_shards(SKEWED, shards=2, gpus=[2, 3, 4, 5])

        for shard in plan.shards:
            assert len(shard.gpus) == 2
            assert plan.sharing_for(shard) == (1, 1)

    def test_an_unplanned_gpu_is_treated_as_unshared(self) -> None:
        plan = plan_shards(SKEWED, shards=2, gpus=[2, 3])
        bare = ShardPlan(shards=plan.shards, shards_per_gpu={})

        assert bare.sharing_for(plan.shards[0]) == (1,)

    def test_the_rank_says_who_gets_the_remainder(self) -> None:
        """Six shards over four GPUs: the first owner of a shared device is rank 0, the second
        rank 1 — so `count: 3` becomes 2 + 1 across them, not 1 + 1 with a third gone."""
        plan = plan_shards(SKEWED, shards=6, gpus=[2, 3, 4, 5])

        assert [plan.rank_for(s) for s in plan.shards] == [(0,), (0,), (0,), (0,), (1,), (1,)]

    def test_a_sole_owner_is_rank_zero_on_every_device(self) -> None:
        plan = plan_shards(SKEWED, shards=2, gpus=[2, 3, 4, 5])

        assert all(plan.rank_for(s) == (0, 0) for s in plan.shards)


class TestAFleetWithNoCamerasYet:
    """A shard is spawned before anybody has decided what it reads (arch.md section 2).

    Under the gRPC control plane the cameras arrive one ``AddCamera`` at a time, so an empty
    fleet is the *normal* start state and the plan is the device assignment alone. It used to
    be refused, on the argv-era reasoning that a shard with no cameras reads nothing forever.
    """

    def test_the_plan_is_the_devices_and_names_no_cameras(self) -> None:
        plan = plan_shards({}, shards=3, gpus=[2, 3, 4])

        assert len(plan) == 3
        assert plan.cameras == ()
        assert [s.gpus for s in plan.shards] == [(2,), (3,), (4,)]
        assert all(s.offered_fps == 0.0 for s in plan.shards)

    def test_shards_still_share_devices_when_there_are_more_of_them(self) -> None:
        """The sharing is what the control plane sends on: it must be right with no cameras."""
        plan = plan_shards({}, shards=4, gpus=[2, 3])

        assert [plan.sharing_for(s) for s in plan.shards] == [(2,), (2,), (2,), (2,)]
        assert sorted(plan.rank_for(s) for s in plan.shards) == [(0,), (0,), (1,), (1,)]


class TestAnImpossiblePlanFailsAtPlanTime:

    @pytest.mark.parametrize("shards", [0, -1])
    def test_fewer_than_one_shard(self, shards: int) -> None:
        with pytest.raises(ConfigurationError, match="at least one shard"):
            plan_shards(SKEWED, shards=shards, gpus=[2])

    def test_no_gpus(self) -> None:
        with pytest.raises(ConfigurationError, match="no gpus"):
            plan_shards(SKEWED, shards=1, gpus=[])

    def test_more_shards_than_cameras(self) -> None:
        with pytest.raises(ConfigurationError, match="loads engines and holds a CUDA context"):
            plan_shards({"only": 20.0}, shards=2, gpus=[2, 3])


class TestDescribeIsWhatTheLauncherPrints:
    def test_it_names_every_shard_with_its_load_and_devices(self) -> None:
        text = plan_shards(SKEWED, shards=2, gpus=[2, 3, 4, 5]).describe()

        assert "2 shard(s)" in text
        assert [line for line in text.splitlines() if line.startswith("  shard ")] != []
        assert len(text.splitlines()) == 1 + 2 + 4, "header, one line per shard, one per gpu"
        assert "90 fps offered" in text
        assert "gpu(s) [2, 3]" in text and "gpu(s) [4, 5]" in text

    def test_it_reports_the_imbalance_the_plan_is_judged_by(self) -> None:
        # Two cameras that cannot be balanced: 30 and 10, so 66.7% imbalance.
        text = plan_shards({"a": 30.0, "b": 10.0}, shards=2, gpus=[2]).describe()

        assert "imbalance 66.7%" in text


class TestShardIsAValueObject:
    def test_it_is_frozen_so_a_launcher_cannot_edit_a_plan_it_was_handed(self) -> None:
        shard = Shard(index=0, cameras=("a",), gpus=(2,), offered_fps=20.0)

        with pytest.raises(FrozenInstanceError):
            shard.index = 1  # type: ignore[misc]
