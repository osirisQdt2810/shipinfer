"""``shipinfer fleet`` — the plan, the refusals, and the one path that runs without a GPU.

The command itself is thin: it builds the plan, prints it, and hands it to
:class:`~shipinfer.server.launcher.Fleet`. What is worth testing here is the part that is not
thin — what it refuses, and whether the number an operator reads before spawning fifty camera
connections is the number the plan actually has.

``--dry-run`` is the only path that runs outside a container, deliberately: it spawns nothing
and touches no accelerator, so the containment gate does not apply to it and this file needs
no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shipinfer.cli.commands.fleet import fleet
from shipinfer.core.errors import ConfigurationError

REPO = Path("model_repository")


def config(monkeypatch, cameras, gpus="2,3,4,5"):
    """Point `build_settings` at a fleet without writing a file for it."""
    import json

    monkeypatch.setenv(
        "SHIPINFER_INGEST__CAMERAS",
        json.dumps(
            [
                {"camera_id": name, "uri": f"rtsp://host/{name}", "fps": fps}
                for name, fps in cameras
            ]
        ),
    )
    monkeypatch.delenv("SHIPINFER_SHARD_CAMERAS", raising=False)
    return gpus


class TestTheDryRunPrintsThePlanAndStops:
    def test_it_reports_one_line_per_shard(self, monkeypatch, capsys) -> None:
        gpus = config(monkeypatch, [("a", 20.0), ("b", 20.0), ("c", 20.0), ("d", 20.0)])

        assert fleet(REPO, shards=2, gpus=gpus, dry_run=True) == 0

        printed = capsys.readouterr().out
        assert "2 shard(s)" in printed
        assert printed.count("shard 0") == 1 and printed.count("shard 1") == 1

    def test_it_balances_by_offered_load_not_by_count(self, monkeypatch, capsys) -> None:
        """The failure this project exists to fix, one level up: four 30 fps cameras and
        twelve 5 fps ones split by count puts every busy camera within reach of one shard."""
        busy = [(f"busy{i}", 30.0) for i in range(4)]
        quiet = [(f"quiet{i}", 5.0) for i in range(12)]
        gpus = config(monkeypatch, busy + quiet)

        fleet(REPO, shards=2, gpus=gpus, dry_run=True)

        printed = capsys.readouterr().out
        assert "imbalance 0.0%" in printed
        assert printed.count("8 camera(s)") == 2

    def test_an_unspecified_frame_rate_counts_as_one_rather_than_zero(
        self, monkeypatch, capsys
    ) -> None:
        """`fps` is 0 for "whatever the source delivers", which is most RTSP cameras.
        Weighting by zero would make every camera equal *and* make the imbalance undefined —
        so the plan would silently go back to balancing by count."""
        gpus = config(monkeypatch, [(f"cam{i}", 0.0) for i in range(6)], gpus="2,3,4")

        fleet(REPO, shards=3, gpus=gpus, dry_run=True)

        printed = capsys.readouterr().out.replace("\n", " ")  # the console wraps long lines
        assert "device imbalance 0.0%" in printed and "shard imbalance 0.0%" in printed
        assert printed.count("2 camera(s)") == 3

    def test_a_lopsided_fleet_is_warned_about_rather_than_refused(
        self, monkeypatch, capsys
    ) -> None:
        """It can be the best available split. It is printed loudly because the fleet is
        bounded by its busiest shard, so this is the number that says whether sharding
        helped."""
        gpus = config(monkeypatch, [("huge", 200.0), ("tiny", 5.0)])

        assert fleet(REPO, shards=2, gpus=gpus, dry_run=True) == 0

        assert "warning" in capsys.readouterr().out

    def test_a_balanced_fleet_is_not_warned_about(self, monkeypatch, capsys) -> None:
        gpus = config(monkeypatch, [("a", 20.0), ("b", 20.0)])

        fleet(REPO, shards=2, gpus=gpus, dry_run=True)

        assert "warning" not in capsys.readouterr().out

    def test_it_spawns_nothing(self, monkeypatch, capsys) -> None:
        """The reason this path needs no container: a dry run that started a process would
        be a GPU held by a command whose whole purpose is to not hold one."""
        # By name out of `sys.modules`, not `import ... as`: the package re-exports the
        # function under the module's own name, so the plain import resolves to the function.
        import sys

        module = sys.modules["shipinfer.cli.commands.fleet"]

        def refuse(*_args, **_kwargs):
            raise AssertionError("--dry-run must not construct a Fleet")

        monkeypatch.setattr(module, "Fleet", refuse)
        gpus = config(monkeypatch, [("a", 20.0), ("b", 20.0)])

        assert fleet(REPO, shards=2, gpus=gpus, dry_run=True) == 0


class TestWhatItRefusesBeforeSpawningAnything:
    def test_a_configuration_with_no_cameras(self, monkeypatch) -> None:
        gpus = config(monkeypatch, [])

        with pytest.raises(ConfigurationError, match="nothing to shard"):
            fleet(REPO, shards=2, gpus=gpus, dry_run=True)

    def test_no_visible_gpus_and_none_from_the_driver(self, monkeypatch) -> None:
        import shipinfer.runtime.platform as platform

        config(monkeypatch, [("a", 20.0)])
        monkeypatch.setenv("SHIPINFER_DEVICES__VISIBLE_GPUS", "[]")
        monkeypatch.setattr(platform, "device_count", lambda: 0)

        with pytest.raises(ConfigurationError, match="driver reports none"):
            fleet(REPO, shards=1, gpus=None, dry_run=True)

    def test_an_empty_visible_list_means_every_device_the_driver_reports(
        self, monkeypatch, capsys
    ) -> None:
        """As it does for `serve` and for `DeviceManager`: a fleet that refused what a single
        process accepts would be the odd one out."""
        import shipinfer.runtime.platform as platform

        config(monkeypatch, [("a", 20.0), ("b", 20.0)])
        monkeypatch.setenv("SHIPINFER_DEVICES__VISIBLE_GPUS", "[]")
        monkeypatch.setattr(platform, "device_count", lambda: 2)

        assert fleet(REPO, shards=2, gpus=None, dry_run=True) == 0

        out = capsys.readouterr().out
        assert "gpu(s) [0]" in out and "gpu(s) [1]" in out

    def test_more_shards_than_cameras(self, monkeypatch) -> None:
        """A shard with no cameras still loads engines and holds a CUDA context."""
        gpus = config(monkeypatch, [("only", 20.0)])

        with pytest.raises(ConfigurationError, match="loads engines"):
            fleet(REPO, shards=4, gpus=gpus, dry_run=True)

    def test_zero_shards(self, monkeypatch) -> None:
        gpus = config(monkeypatch, [("a", 20.0)])

        with pytest.raises(ConfigurationError, match="at least one shard"):
            fleet(REPO, shards=0, gpus=gpus, dry_run=True)
