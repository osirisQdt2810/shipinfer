"""The sharded harness, offline: the plan it launches, the child's slice, the sum it reports."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from benchmarks.harness import shards
from benchmarks.harness.config import BenchConfig
from benchmarks.harness.shipinfer import _cameras


def _config(**overrides) -> BenchConfig:
    base = {"cameras": 8, "fps": 5.0, "gpus": (3, 4, 5), "seconds": 20.0, "warmup_s": 5.0}
    base.update(overrides)
    return BenchConfig(**base)


class TestTheConfigKnowsItsShape:
    def test_single_is_the_default_and_has_no_split(self) -> None:
        cfg = _config()
        assert (cfg.topology, cfg.shards, cfg.shard_cameras, cfg.camera_ids) == (
            "single",
            0,
            (),
            (),
        )
        assert cfg.offered_total == 40.0

    def test_a_slice_offers_its_own_cameras(self) -> None:
        cfg = _config(camera_ids=("cam00", "cam01", "cam02"))
        assert cfg.camera_count == 3 and cfg.offered_total == 15.0
        assert cfg.cameras == 8, "the count is the run's topology; the slice is this process's"

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"topology": "mesh"}, "topology must be one of"),
            ({"shards": -1}, "shards must be >= 0"),
            ({"shard_cameras": (4, 4)}, "multi-process topology"),
            ({"topology": "fleet", "shard_cameras": (5, 4)}, "sums to 9, not the 8"),
            (
                {"topology": "fleet", "shard_cameras": (4, 4), "shards": 3},
                "names 2 shards but shards=3",
            ),
            (
                {"topology": "fleet", "shard_cameras": (2, 2, 2, 2)},
                "an explicit split gives each shard one GPU",
            ),
            ({"topology": "service", "shard_cameras": (8, 0)}, "at least one camera"),
        ],
    )
    def test_it_refuses_a_shape_that_cannot_run(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            _config(**overrides)

    def test_from_dict_is_the_inverse_of_as_dict_for_the_configuration(self) -> None:
        cfg = _config(
            topology="service",
            shard_cameras=(4, 2, 2),
            camera_ids=("cam00",),
            out_dir=Path("/tmp/x"),
        )
        again = BenchConfig.from_dict(cfg.as_dict())
        for name in (
            "cameras",
            "fps",
            "gpus",
            "seconds",
            "warmup_s",
            "topology",
            "shards",
            "shard_cameras",
            "camera_ids",
            "out_dir",
            "source",
            "rtsp_port",
        ):
            assert getattr(again, name) == getattr(cfg, name), name
        assert again.model_repository == cfg.resolved().model_repository


class TestThePlanTheHarnessLaunches:
    def test_the_explicit_split_is_contiguous_one_gpu_per_shard(self) -> None:
        plan = shards.plan_for(_config(topology="fleet", shard_cameras=(4, 2, 2)))
        assert len(plan) == 3
        assert [s.cameras for s in plan.shards] == [
            ("cam00", "cam01", "cam02", "cam03"),
            ("cam04", "cam05"),
            ("cam06", "cam07"),
        ]
        assert [s.gpus for s in plan.shards] == [(3,), (4,), (5,)]
        assert [s.offered_fps for s in plan.shards] == [20.0, 10.0, 10.0]
        assert plan.sharing_for(plan.shards[0]) == (1,), "one shard per GPU: nothing is shared"

    def test_the_default_plan_is_the_launchers_balanced_one(self) -> None:
        plan = shards.plan_for(_config(topology="fleet"))
        assert len(plan) == 3 and sorted(
            c for s in plan.shards for c in s.cameras
        ) == shards.camera_names(_config())
        assert plan.imbalance < 0.5

    def test_service_is_told_the_explicit_plan_so_it_names_every_peer(self) -> None:
        import json

        from shipinfer.core.settings.runner import SERVICE_PEERS_ENV, SERVICE_SHARD_ENV

        config = _config(topology="service", shard_cameras=(6, 2))
        plan = shards.plan_for(config)
        env, per_shard = shards.tier_env(config, plan)

        assert json.loads(env[SERVICE_PEERS_ENV]) == [0, 1]
        assert per_shard is not None
        assert [per_shard(s) for s in plan.shards] == [
            {SERVICE_SHARD_ENV: "0"},
            {SERVICE_SHARD_ENV: "1"},
        ]

    def test_single_has_no_plan(self) -> None:
        with pytest.raises(ValueError, match="single"):
            shards.plan_for(_config())


class TestTheChild:
    def test_its_config_is_the_parents_narrowed_to_its_slice(self, tmp_path: Path) -> None:
        parent = _config(topology="service", shard_cameras=(4, 2, 2), out_dir=tmp_path)
        child = shards.child_config(
            parent, cameras=("cam04", "cam05"), gpus=(4,), out_dir=tmp_path / "shard-1"
        )
        assert (child.topology, child.shards, child.shard_cameras) == ("single", 0, ())
        assert child.gpus == (4,) and child.cuda_visible_devices() == "4"
        assert child.camera_ids == ("cam04", "cam05") and child.offered_total == 10.0
        assert child.cameras == 8 and child.fps == parent.fps

    def test_it_drives_exactly_its_cameras_with_the_parents_naming_and_split(
        self, tmp_path: Path
    ) -> None:
        parent = _config(
            topology="fleet",
            shard_cameras=(4, 4),
            gpus=(3, 4),
            person_frames=tmp_path,
            ship_frames=tmp_path,
        )
        child = shards.child_config(
            parent, cameras=("cam03", "cam04"), gpus=(4,), out_dir=tmp_path
        )
        cameras = _cameras(child)
        assert [c["camera_id"] for c in cameras] == ["cam03", "cam04"]
        assert cameras[0]["fps"] == 5.0
        with pytest.raises(ValueError, match="cameras this run does not have"):
            _cameras(
                shards.child_config(parent, cameras=("cam99",), gpus=(4,), out_dir=tmp_path)
            )

    def test_its_command_is_this_interpreter_running_this_module(self, tmp_path: Path) -> None:
        plan = shards.plan_for(_config(topology="fleet", shard_cameras=(4, 2, 2)))
        argv = shards.child_command(plan.shards[2], tmp_path / "config.json", tmp_path)
        assert argv[:3] == [sys.executable, "-m", "benchmarks.harness.shards"]
        assert argv[argv.index("--out") + 1] == str(tmp_path / "shard-2")
        assert argv[argv.index("--config") + 1] == str(tmp_path / "config.json")
        assert argv[argv.index("--cameras") + 1] == "cam06,cam07"

    def test_the_child_parses_the_cameras_the_parent_put_on_its_line(
        self, tmp_path: Path
    ) -> None:
        """The two ends of the harness's own contract. They used to be joined by
        `SHIPINFER_SHARD_CAMERAS`, which is gone with the argv mechanism it belonged to."""
        plan = shards.plan_for(_config(topology="fleet", shard_cameras=(4, 2, 2)))
        argv = shards.child_command(plan.shards[1], tmp_path / "config.json", tmp_path)
        index = argv.index("--cameras")

        assert tuple(argv[index + 1].split(",")) == plan.shards[1].cameras


def _summary(
    shard: int,
    gpu: int,
    rate,
    verdict: str,
    *,
    saturated: bool = False,
    binding=None,
    per_device=None,
) -> dict:
    return {
        "shard": shard,
        "gpus": [gpu],
        "cameras": ["a", "b"],
        "offered_total": 10.0,
        "achieved": 9.5,
        "throughput": {
            "images_per_s": rate,
            "verdict": verdict,
            "saturated": saturated,
            "binding_module": binding,
        },
        "per_device": per_device or {},
    }


class TestTheSumTheParentReports:
    def test_throughput_sums_and_the_worst_verdict_wins(self) -> None:
        agg = shards.aggregate(
            [
                _summary(0, 3, 100.0, "SUSTAINED"),
                _summary(1, 4, 50.0, "SATURATED", saturated=True, binding="ship_detector"),
            ]
        )
        assert agg["images_per_s"] == 150.0
        assert (agg["verdict"], agg["saturated"], agg["binding_module"]) == (
            "SATURATED",
            True,
            "ship_detector",
        )
        assert [r["shard"] for r in agg["shards"]] == [0, 1]

    def test_a_shard_without_a_number_leaves_the_fleet_without_one(self) -> None:
        agg = shards.aggregate(
            [_summary(0, 3, 100.0, "SUSTAINED"), _summary(1, 4, None, "UNMEASURED")]
        )
        assert agg["images_per_s"] is None and agg["verdict"] == "UNMEASURED"

    def test_per_device_counts_add_up_across_shards_where_the_work_ran(self) -> None:
        agg = shards.aggregate(
            [
                _summary(
                    0, 3, 1.0, "SUSTAINED", per_device={"person_embedder": {"cuda:3": 700}}
                ),
                _summary(
                    1,
                    4,
                    1.0,
                    "SUSTAINED",
                    per_device={"person_embedder": {"cuda:4": 300, "cuda:3": 5}},
                ),
            ]
        )
        assert agg["per_device"] == {"person_embedder": {"cuda:3": 705, "cuda:4": 300}}

    def test_a_child_relabels_its_devices_to_physical_ordinals(self) -> None:
        assert shards._relabel({"m": {"cuda:0": 7, "cpu": 1}}, gpus=(4,)) == {
            "m": {"cuda:4": 7, "cpu": 1}
        }
