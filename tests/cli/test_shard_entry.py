"""The shard process's entry point: two flags, and the sharing that used to be an env var.

Everything here is offline and hardware-free. What the entry point *does* with a topology —
load models, build a runner — needs a GPU and a repository and is not asserted here; what is
asserted is the part that has been silently wrong before: the argv contract with the launcher,
and the device sharing reaching the settings the engine reads.

The sharing tests are ``tests/server/test_shard_settings.py``'s, carried onto the RPC. That
file checked that ``SHIPINFER_DEVICES__SHARED_BY`` in a child's environment made a shard load
its *share* of every model's instances. The environment no longer carries it — the
``UpdateTopology`` RPC does — so the same property is asserted at the new seam. The failure it
guards against is silent and expensive: two shards on one GPU each loading the full instance
count is twice the engines and twice the VRAM for the same throughput, with every process
reporting healthy.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from shipinfer.cli.shard import _ShardProcess, apply_sharing, build_parser
from shipinfer.core.errors import ServerStateError
from shipinfer.core.settings import ServerSettings
from shipinfer.repository.model_config import (
    InstanceGroup,
    InstanceKind,
    ModelConfig,
)
from shipinfer.topology import ChainSpec


class TestTheTwoFlags:
    def test_it_takes_the_identity_and_the_port(self) -> None:
        args = build_parser().parse_args(["--shard-id", "3", "--control-port", "50103"])

        assert (args.shard_id, args.control_port) == (3, 50103)

    def test_a_third_flag_is_refused_rather_than_ignored(self) -> None:
        """The argv is the contract (arch.md section 2). A flag the launcher grew and this
        parser silently dropped would be configuration that vanishes on every shard at once."""
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(
                ["--shard-id", "0", "--control-port", "1", "--cameras", "quay-1"]
            )

        assert exit_info.value.code == 2

    @pytest.mark.parametrize("argv", [[], ["--shard-id", "0"], ["--control-port", "50100"]])
    def test_both_flags_are_required(self, argv: list[str]) -> None:
        """A shard that guessed its own id would place its metrics under another shard's."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)

    def test_the_module_really_runs_as_dash_m(self) -> None:
        """The launcher spawns ``python -m shipinfer.cli.shard``; this is that spelling, run.

        Cheap and worth it: ``--help`` exits before the container gate and before anything is
        loaded, so it proves the module is executable and its parser builds, on any host.
        """
        result = subprocess.run(
            [sys.executable, "-m", "shipinfer.cli.shard", "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, result.stderr
        assert "--shard-id" in result.stdout and "--control-port" in result.stdout

    def test_the_launcher_and_this_parser_agree(self) -> None:
        from shipinfer.runners.fleet import FleetRunner

        argv = FleetRunner.shard_command(2, 50102)

        assert argv[1:3] == ["-m", "shipinfer.cli.shard"]
        assert build_parser().parse_args(argv[3:]).shard_id == 2


class TestTheSharingRidesOnTheRpcNow:
    def test_it_lands_where_the_engine_reads_it(self) -> None:
        settings = apply_sharing(ServerSettings(), (2, 1), (1, 0))

        assert settings.devices.shared_by == [2, 1]
        assert settings.devices.share_rank == [1, 0]

    def test_nothing_said_means_one_process_per_device(self) -> None:
        """Which is what a single-process run is, and why nothing had to change for one."""
        settings = ServerSettings()

        assert apply_sharing(settings, (), ()) is settings

    def test_the_rest_of_the_settings_are_untouched(self) -> None:
        whole = ServerSettings()
        narrowed = apply_sharing(whole, (2,), (0,))

        assert narrowed.scheduler == whole.scheduler
        assert narrowed.runner == whole.runner
        assert narrowed.devices.model_dump(exclude={"shared_by", "share_rank"}) == (
            whole.devices.model_dump(exclude={"shared_by", "share_rank"})
        )

    def test_a_zero_share_is_refused(self) -> None:
        """Rebuilt through `DeviceSettings` rather than `model_copy`d precisely so the field
        validators run: a launcher bug reaches the launcher as a refused UpdateTopology."""
        with pytest.raises(ValidationError, match="at least 1"):
            apply_sharing(ServerSettings(), (0,), (0,))

    def test_a_negative_rank_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="0 or more"):
            apply_sharing(ServerSettings(), (1,), (-1,))


class TestTwoShardsOnOneGpuDoNotDoubleLoad:
    """The whole reason the sharing is on the wire at all.

    ``ModelConfig.placements`` is where the division happens, and
    ``tests/repository/test_model_config.py`` pins that. What is pinned here is the *link*:
    the numbers the RPC delivered, read back out of the settings tree in the alignment
    ``DeviceManager`` uses (``visible_gpus[i]`` shares with ``shared_by[i]``), halve the
    instance count instead of leaving it whole.
    """

    @staticmethod
    def _config(count: int) -> ModelConfig:
        return ModelConfig(
            name="ship_detector",
            platform="mock",
            max_batch_size=8,
            inputs=[{"name": "images", "data_type": "FP32", "dims": [3, 640, 640]}],
            outputs=[{"name": "boxes", "data_type": "FP32", "dims": [300, 6]}],
            instance_groups=[InstanceGroup(kind=InstanceKind.GPU, count=count)],
        )

    @staticmethod
    def _mappings(settings: ServerSettings, visible: list[int]):
        shared = dict(zip(visible, settings.devices.shared_by, strict=False))
        ranks = dict(zip(visible, settings.devices.share_rank, strict=False))
        return shared, ranks

    def test_each_shard_loads_its_half(self) -> None:
        config = self._config(4)

        for rank in (0, 1):
            settings = apply_sharing(ServerSettings(), (2,), (rank,))
            shared, ranks = self._mappings(settings, [0])
            placements = config.placements([0], shared_by=shared, share_rank=ranks)

            assert len(placements) == 2, f"shard at rank {rank} loaded {len(placements)}"

    def test_a_shard_told_nothing_loads_the_whole_count(self) -> None:
        """The failure mode, spelled out: this is what a shard that never hears about its
        co-tenant does, and it is why `UpdateTopology` carries the two lists."""
        settings = apply_sharing(ServerSettings(), (), ())
        shared, ranks = self._mappings(settings, [0])

        assert len(self._config(4).placements([0], shared_by=shared, share_rank=ranks)) == 4

    def test_a_count_that_does_not_divide_gives_the_remainder_to_rank_zero(self) -> None:
        """The device still carries the three instances the config asked for, 2 + 1."""
        config = self._config(3)
        loaded = []
        for rank in (0, 1):
            settings = apply_sharing(ServerSettings(), (2,), (rank,))
            shared, ranks = self._mappings(settings, [0])
            loaded.append(len(config.placements([0], shared_by=shared, share_rank=ranks)))

        assert loaded == [2, 1]


CHAIN = """
name: two_step
elements:
  decode: {impl: mock}
  output: {impl: mock}
"""


class TestTheHopThatActuallyCarriesTheSharing:
    """``_ShardProcess.build`` — the RPC path, not the helper it calls.

    Everything above tests ``apply_sharing`` as a function. That is worth having and it is not
    enough: the hop that decides whether a GPU holds one set of engines or two is
    ``build`` calling it *with the right arguments, before the engine exists*. Swapping the
    two lists at that call site passes every test above — they never look at the call — and
    produces a shard that loads ``shared_by`` slices of a ``share_rank``-sized count, which is
    silent and expensive (``model.py`` divides the instance count at load).

    So this drives ``build`` itself, with the engine and the runner factory replaced. No GPU,
    no repository, no model: what is asserted is the *order* and the *arguments*.
    """

    @staticmethod
    def _process(monkeypatch: pytest.MonkeyPatch, settings: ServerSettings | None = None):
        """A ``_ShardProcess`` whose engine and runner are recorders."""
        seen: dict[str, object] = {}

        class FakeEngine:
            def __init__(self, engine_settings: ServerSettings) -> None:
                # Captured at CONSTRUCTION: an engine decides how many instances of each
                # model to load while it starts, and cannot be told afterwards.
                seen["engine_settings"] = engine_settings
                seen["engine"] = self
                self.stopped = 0

            def start(self):
                seen["started"] = True
                return self

            def stop(self) -> None:
                self.stopped += 1

        def fake_build_runner(name: str, topology, runner_settings, **kwargs):
            seen["runner"] = (name, topology, runner_settings, kwargs)
            return object()

        monkeypatch.setattr("shipinfer.engine.InferenceServer", FakeEngine)
        monkeypatch.setattr("shipinfer.runners.build_runner", fake_build_runner)
        return _ShardProcess(settings or ServerSettings(), 1), seen

    def test_the_engine_is_built_from_settings_the_sharing_already_landed_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two shards on one GPU: rank 1 of 2. The engine must be constructed from settings
        that already say so, or it loads the full instance count and the device silently
        holds twice the engines for the same throughput."""
        process, seen = self._process(monkeypatch)

        process.build(ChainSpec.from_yaml(CHAIN), (2,), (1,))

        engine_settings = seen["engine_settings"]
        assert engine_settings.devices.shared_by == [2]
        assert engine_settings.devices.share_rank == [1]
        assert seen["started"] is True

    def test_two_shards_on_one_gpu_load_two_of_four_instances_each(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The assertion ``tests/server/test_shard_settings.py`` used to make of a child's
        environment, made of the RPC path end to end: the numbers the launcher sent, read back
        out of the settings the engine was handed, halve a four-instance model."""
        config = TestTwoShardsOnOneGpuDoNotDoubleLoad._config(4)
        loaded = []
        for rank in (0, 1):
            process, seen = self._process(monkeypatch)
            process.build(ChainSpec.from_yaml(CHAIN), (2,), (rank,))
            engine_settings = seen["engine_settings"]
            shared = dict(zip([0], engine_settings.devices.shared_by, strict=False))
            ranks = dict(zip([0], engine_settings.devices.share_rank, strict=False))
            loaded.append(len(config.placements([0], shared_by=shared, share_rank=ranks)))

        assert loaded == [2, 2], "a shard loaded the whole count despite being told it shares"

    def test_the_runner_is_the_in_process_one_and_gets_the_same_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A shard *is* the process the fleet placed, so a shard that built a `fleet` runner
        would spawn shards of its own."""
        process, seen = self._process(monkeypatch)

        process.build(ChainSpec.from_yaml(CHAIN), (2,), (1,))

        name, topology, runner_settings, kwargs = seen["runner"]
        assert name == "inprocess"
        assert topology.name == "two_step"
        assert runner_settings is seen["engine_settings"]
        assert kwargs["shard_id"] == 1

    def test_nothing_said_leaves_the_settings_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single-process run sends empty lists, and that is what "one process per device"
        has always meant."""
        settings = ServerSettings()
        process, seen = self._process(monkeypatch, settings)

        process.build(ChainSpec.from_yaml(CHAIN), (), ())

        assert seen["engine_settings"] is settings

    def test_a_second_build_gives_the_first_engine_back_before_making_another(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`UpdateTopology` calls this again after a start it refused (``runners/service.py``
        drops the runner so the retry rebuilds it). Assigning a second engine over the first
        would leave the first one's CUDA context held by nothing that can ever stop it."""
        process, seen = self._process(monkeypatch)

        process.build(ChainSpec.from_yaml(CHAIN), (2,), (1,))
        first = seen["engine"]

        process.build(ChainSpec.from_yaml(CHAIN), (2,), (1,))

        assert seen["engine"] is not first, "the second build reused the recorder"
        assert first.stopped == 1, "the first engine was orphaned with its context held"
        assert seen["engine"].stopped == 0

    def test_a_runner_that_cannot_be_built_gives_the_gpu_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLAUDE.md's GPU hygiene rule: a factory that raised half-way must leave no CUDA
        context behind on a box other people share."""
        process, seen = self._process(monkeypatch)

        def explode(*_a: object, **_k: object):
            raise RuntimeError("no such element implementation")

        monkeypatch.setattr("shipinfer.runners.build_runner", explode)

        with pytest.raises(RuntimeError, match="no such element"):
            process.build(ChainSpec.from_yaml(CHAIN), (2,), (1,))

        assert seen["engine"].stopped == 1
        process.release()  # idempotent: the engine is already gone, and this must not re-stop
        assert seen["engine"].stopped == 1

    def test_an_engine_whose_start_raises_is_still_given_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The engine is assigned **before** it is started, and this is why.

        `InferenceServer.start()` loads models one at a time; a strict start that fails on
        the fifth has four running, each with worker threads and a CUDA context. Those
        belong to the engine object — so if `build` only assigns after `start()` returns,
        the failure path assigns nothing at all and `release()` has nothing to reach. The
        contexts are then held for the life of the process by an object no reference finds,
        on a box other people share (CLAUDE.md's GPU hygiene rule).
        """
        process, seen = self._process(monkeypatch)

        class HalfStartedEngine:
            def __init__(self, engine_settings: ServerSettings) -> None:
                seen["engine"] = self
                self.stopped = 0

            def start(self):
                # What a strict start does with models already loaded: raise, holding them.
                raise ServerStateError("instance ship_detector_0_cuda:1 failed to start")

            def stop(self) -> None:
                self.stopped += 1

        monkeypatch.setattr("shipinfer.engine.InferenceServer", HalfStartedEngine)

        with pytest.raises(ServerStateError, match="failed to start"):
            process.build(ChainSpec.from_yaml(CHAIN), (2,), (1,))

        assert seen["engine"].stopped == 1, "the half-started engine was never stopped"
        process.release()  # idempotent, as on every other failure path
        assert seen["engine"].stopped == 1
