"""``shipinfer run --inputs``: files and URLs become cameras on the runner.

arch.md §2's offline mode — *"the M inputs are sharded evenly at start"* — reduced to the two
pure helpers that decide it, so the whole thing is testable outside a container. ``run()``
itself loads engines and drives GPUs and is gated accordingly; ``cameras_from_inputs`` and
``place_cameras`` are the parts with the decisions in them, and they are ordinary functions
over a :class:`~shipinfer.runners.base.Runner`.

The runners here are real :class:`Runner` subclasses rather than mocks, because half of what
is under test is the ABC's own default: a runner that manages no cameras must refuse in a way
the operator can act on, and a ``Mock`` would cheerfully accept the camera.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, ClassVar

import pytest

from shipinfer.cli.commands.run import (
    cameras_from_inputs,
    cameras_from_settings,
    cameras_to_place,
    place_cameras,
    run,
)
from shipinfer.core.errors import ConfigurationError, NoShardAvailableError
from shipinfer.core.request import Priority, ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.launch.control import CameraSpec
from shipinfer.runners import RUNNERS
from shipinfer.runners.base import Runner
from shipinfer.topology import ChainItem, ChainSpec, Topology

CHAIN = textwrap.dedent("""
    name: mock_chain
    elements:
      decode: {impl: replay}
      detect: {impl: mock, model: ship_detector}
      output: {impl: mock}
    """)


@pytest.fixture()
def chain_file(tmp_path: Path) -> Path:
    path = tmp_path / "mock_chain.yaml"
    path.write_text(CHAIN)
    return path


def chain() -> Topology:
    return Topology.from_spec(ChainSpec.from_yaml(CHAIN))


# Registered into the process-wide `RUNNERS`, so `shipinfer runners` lists a test double for
# the rest of any session that imports this file. The same bargain the element registries
# already take in `tests/runners/test_inprocess.py`, and `test_runner_registry.py` asserts
# membership rather than equality, so nothing breaks -- but the runner registry had been clean
# until now, and a reader who finds `still` in a listing deserves this line.
@RUNNERS.register("still")
class StillRunner(Runner):
    """A runner that executes a chain and owns no ingest plane — the ABC's default.

    Registered, so ``run(--runner still)`` can reach it: both shipped runners manage cameras,
    and the refusal under test is about the ones that do not (``deepstream`` will be the first
    real one). Registration is module scope and therefore process-wide, which is the same
    bargain ``tests/runners/test_camera_lifecycle.py`` strikes for its gate elements.
    """

    name: ClassVar[str] = "still"

    def _do_start(self) -> None:
        return None

    def _do_stop(self, timeout_s: float) -> None:
        return None

    def _do_submit(self, item: ChainItem) -> ResponseFuture:  # pragma: no cover - unused
        raise NotImplementedError


class CountingRunner(StillRunner):
    """A runner that does manage cameras, and records what it was given."""

    name: ClassVar[str] = "counting"
    manages_cameras: ClassVar[bool] = True

    def __init__(self, topology: Topology, **kwargs: Any) -> None:
        super().__init__(topology, **kwargs)
        self.added: list[CameraSpec] = []
        self.refuse: str = ""

    def add_camera(self, camera: CameraSpec) -> None:
        if camera.camera_id == self.refuse:
            raise ConfigurationError(f"camera {camera.camera_id!r} is already running")
        self.added.append(camera)


class RefusingRunner(CountingRunner):
    """A runner that answers the way a full fleet does: nowhere to put this camera."""

    name: ClassVar[str] = "refusing"

    def add_camera(self, camera: CameraSpec) -> None:
        raise NoShardAvailableError(camera.camera_id, ["shard 0: draining"])


class TestInputsBecomeCameraSpecs:
    def test_each_input_becomes_one_camera_named_by_position(self) -> None:
        """A path is not a camera id — two directories hold the same ``clip.mp4``, and a
        path is not a legal metric label either. Position is the only answer nobody has to
        be asked for."""
        cameras = cameras_from_inputs(["a.mp4", "rtsp://cam/live", "/frames"])

        assert [camera.camera_id for camera in cameras] == ["cam-000", "cam-001", "cam-002"]
        assert [camera.url for camera in cameras] == ["a.mp4", "rtsp://cam/live", "/frames"]

    def test_the_ids_are_padded_so_they_sort_as_strings(self) -> None:
        """Every log line, dashboard and metric label sorts these as text, not as numbers."""
        cameras = cameras_from_inputs([f"clip-{index}.mp4" for index in range(11)])
        ids = [camera.camera_id for camera in cameras]

        assert sorted(ids) == ids
        assert ids[-1] == "cam-010"

    def test_the_fps_is_left_to_the_source(self) -> None:
        """``0.0`` means "whatever it delivers"; the pacing knob is per-camera config."""
        assert cameras_from_inputs(["a.mp4"])[0].fps == 0.0

    def test_a_file_replays_in_a_loop_unless_no_loop_says_otherwise(self) -> None:
        """The knob the help text used to promise and nothing implemented.

        ``--inputs`` cameras are minted here, so no entry in ``ingest.cameras`` can carry
        their ``loop:`` — ``shipinfer run --inputs clip.mp4`` replayed the file forever and no
        configuration anywhere would stop it. It rides on the spec, so it reaches a shard too.
        """
        assert cameras_from_inputs(["clip.mp4"])[0].loop is True
        assert cameras_from_inputs(["clip.mp4"], loop=False)[0].loop is False

    @pytest.mark.parametrize("inputs", [None, [], ()])
    def test_no_inputs_is_no_cameras_and_not_an_error(self, inputs: Any) -> None:
        """Bringing a chain up empty and adding cameras over the control plane is normal."""
        assert cameras_from_inputs(inputs) == []


def settings_with(*cameras: dict[str, object]) -> ServerSettings:
    """A settings tree whose ``ingest.cameras`` is the deployment's fleet."""
    return ServerSettings(ingest={"cameras": list(cameras)})


class TestTheConfiguredFleetIsPlacedTheSameWay:
    """``ingest.cameras`` / ``ingest.camera_db`` reach a runner through ``add_camera``.

    They used to reach it through ``InprocessRunner._do_start``, which read the settings tree
    it happened to be built from. A shard is an ``InprocessRunner`` built from an env-only
    tree, so every shard read the operator's whole fleet and opened all of it — eight shards
    x fifty cameras, and an ``AddCamera`` that could then place nothing because every id was
    already taken. Deriving the specs here instead means one list, one door, and the runner
    the operator chose decides where they land.
    """

    def test_a_configured_camera_becomes_a_spec_a_runner_takes(self) -> None:
        cameras = cameras_from_settings(
            settings_with(
                {"camera_id": "cam-quay", "uri": "rtsp://10.0.0.7/live", "fps": 20.0},
            )
        )

        assert cameras == [
            CameraSpec(
                camera_id="cam-quay",
                url="rtsp://10.0.0.7/live",
                fps=20.0,
                priority=Priority.NORMAL,
            )
        ]

    def test_a_configured_band_travels_with_the_camera(self) -> None:
        """``priority:`` used to reach no shard at all, and the failure was silent.

        A fleet shard's ``ingest.cameras`` is cleared before its manager is built
        (``runners/inprocess.py::_ingest``), so the shard has nothing to resolve a band
        against and every camera it is given by RPC was admitted at ``normal``. This process
        is the last one that still holds the operator's fleet config, so the band is read here
        and carried.
        """
        cameras = cameras_from_settings(
            settings_with(
                {
                    "camera_id": "cam-hot",
                    "uri": "rtsp://a/live",
                    "priority": Priority.TRACKING_CRITICAL,
                },
                {"camera_id": "cam-cold", "uri": "rtsp://b/live"},
            )
        )

        assert [camera.priority for camera in cameras] == [
            Priority.TRACKING_CRITICAL,
            Priority.NORMAL,
        ]

    def test_the_configured_bands_reach_the_runner_through_place_cameras(self) -> None:
        """End to end through the one door, because that is where a dropped field shows."""
        runner = CountingRunner(chain())
        tree = settings_with(
            {
                "camera_id": "cam-hot",
                "uri": "rtsp://a/live",
                "priority": Priority.TRACKING_CRITICAL,
            },
            {"camera_id": "cam-bg", "uri": "rtsp://b/live", "priority": Priority.BACKGROUND},
        )

        place_cameras(runner, cameras_to_place(tree, ["clip.mp4"]))

        assert [camera.priority for camera in runner.added] == [
            Priority.TRACKING_CRITICAL,
            Priority.BACKGROUND,
            # `--inputs` names no band: it is minted here and appears in no config, so the
            # honest answer is "the deployment decides" rather than a lane invented by the CLI.
            None,
        ]

    def test_a_disabled_camera_is_not_placed(self) -> None:
        """``enabled: false`` keeps a camera in the database and out of the fleet."""
        cameras = cameras_from_settings(
            settings_with(
                {"camera_id": "cam-a", "uri": "rtsp://a/live"},
                {"camera_id": "cam-off", "uri": "rtsp://b/live", "enabled": False},
            )
        )

        assert [camera.camera_id for camera in cameras] == ["cam-a"]

    def test_a_configured_camera_keeps_its_own_loop(self) -> None:
        """``--loop`` is for ``--inputs``; a configured camera already said what it wants."""
        cameras = cameras_to_place(
            settings_with({"camera_id": "cam-file", "uri": "clip.mp4", "loop": False}),
            ["other.mp4"],
            loop=True,
        )

        assert [camera.loop for camera in cameras] == [False, True]

    def test_the_configured_fleet_is_offered_before_the_inputs(self) -> None:
        """Order is a decision: the deployment first, then what this invocation adds.

        A least-loaded placement spreads the standing fleet evenly before the extras land on
        top of it, and a collision is reported against the ``--inputs`` camera that caused it
        rather than against the configured one that was there first.
        """
        runner = CountingRunner(chain())
        tree = settings_with(
            {"camera_id": "cam-cfg-a", "uri": "rtsp://a/live"},
            {"camera_id": "cam-cfg-b", "uri": "rtsp://b/live"},
        )

        place_cameras(runner, cameras_to_place(tree, ["clip.mp4"]))

        assert [camera.camera_id for camera in runner.added] == [
            "cam-cfg-a",
            "cam-cfg-b",
            "cam-000",
        ]
        assert [camera.url for camera in runner.added] == [
            "rtsp://a/live",
            "rtsp://b/live",
            "clip.mp4",
        ]

    def test_no_configured_cameras_is_just_the_inputs(self) -> None:
        assert cameras_to_place(ServerSettings(), ["a.mp4"]) == cameras_from_inputs(["a.mp4"])

    def test_a_refusal_names_the_camera_whichever_door_it_came_through(self) -> None:
        """``ingest/manager.py`` names the id; for an ``--inputs`` camera that id was minted
        from a position and appears nowhere in what the operator typed, so both travel."""
        runner = CountingRunner(chain())
        runner.refuse = "cam-000"
        tree = settings_with({"camera_id": "cam-cfg", "uri": "rtsp://a/live"})

        with pytest.raises(ConfigurationError) as caught:
            place_cameras(runner, cameras_to_place(tree, ["clip.mp4"]))

        assert "cam-000" in str(caught.value)
        assert "clip.mp4" in str(caught.value)
        assert [camera.camera_id for camera in runner.added] == ["cam-cfg"]


class TestPlacingThemOnARunner:
    def test_every_camera_reaches_the_runner_in_order(self) -> None:
        runner = CountingRunner(chain())
        cameras = cameras_from_inputs(["a.mp4", "b.mp4"])

        place_cameras(runner, cameras)

        assert runner.added == cameras

    def test_a_runner_that_manages_no_cameras_is_refused_naming_the_flag(self) -> None:
        """The operator's next action is to choose another ``--runner``, so say so.

        Refused rather than ignored: a flag that accepted three videos and opened none of
        them is a deployment that looks healthy and produces nothing, which is the failure
        mode this whole project is a correction of.
        """
        with pytest.raises(ConfigurationError, match="--runner") as caught:
            place_cameras(StillRunner(chain()), cameras_from_inputs(["a.mp4"]))

        assert "manages no cameras" in str(caught.value)

    def test_placing_nothing_on_a_runner_that_manages_nothing_is_fine(self) -> None:
        """The refusal is about ``--inputs``, not about the runner: no inputs, no problem."""
        place_cameras(StillRunner(chain()), [])

    def test_the_first_refusal_names_the_input_the_operator_typed(self) -> None:
        """``ingest/manager.py`` names the camera id; the operator typed a path.

        Both are in the message because either one alone sends them looking in the wrong
        place — the id is minted here and appears nowhere in what they wrote.
        """
        runner = CountingRunner(chain())
        runner.refuse = "cam-001"

        with pytest.raises(ConfigurationError) as caught:
            place_cameras(runner, cameras_from_inputs(["a.mp4", "b.mp4", "c.mp4"]))

        assert "b.mp4" in str(caught.value)
        assert "cam-001" in str(caught.value)
        assert "already running" in str(caught.value)

    def test_a_fleet_with_no_room_travels_untouched_and_stays_a_503(self) -> None:
        """A capacity refusal is not a fact about the input, so it is not re-labelled.

        `place_cameras` prefixes a `ConfigurationError` with the path the operator typed,
        because the id in that message was minted here. `NoShardAvailableError` is the other
        case: the camera is fine and the fleet has nowhere to put it, and wrapping it in a
        `ConfigurationError` would turn a retryable 503 into a terminal 400 for the same
        condition reached over `POST /streams` (`api/errors.py`).
        """
        runner = RefusingRunner(chain())

        with pytest.raises(NoShardAvailableError) as caught:
            place_cameras(runner, cameras_from_inputs(["a.mp4"]))

        assert not isinstance(caught.value, ConfigurationError)
        assert "cam-000" in str(caught.value)

    def test_it_stops_at_the_first_refusal_rather_than_placing_the_rest(self) -> None:
        """The reachable failures are configuration ones, and they apply to all of them.

        Carrying on would report the last camera's message for a mistake made in the first.
        """
        runner = CountingRunner(chain())
        runner.refuse = "cam-000"

        with pytest.raises(ConfigurationError):
            place_cameras(runner, cameras_from_inputs(["a.mp4", "b.mp4"]))

        assert runner.added == []


class TestTheCommandItself:
    def test_a_dry_run_reports_the_cameras_and_opens_none_of_them(
        self, chain_file: Path, capsys
    ) -> None:
        """Placing a camera means starting a decoder thread, so a dry run must not.

        It still says how many there are, because "did it see my three videos" is exactly
        what a dry run is being asked.
        """
        assert run(chain_file, runner="inprocess", inputs=["a.mp4", "b.mp4"], dry_run=True) == 0

        out = capsys.readouterr().out
        assert "cameras: 2 from --inputs" in out
        assert "cam-000" in out

    def test_no_inputs_prints_no_camera_line(self, chain_file: Path, capsys) -> None:
        run(chain_file, runner="inprocess", dry_run=True)

        assert "--inputs" not in capsys.readouterr().out

    def test_the_configured_fleet_is_in_the_plan_and_the_environment_is_where_it_came_from(
        self, chain_file: Path, capsys, monkeypatch
    ) -> None:
        """``SHIPINFER_INGEST__CAMERAS`` is how an operator configures a fleet, and this
        command is now what places it — so a dry run has to show it, and showing it is the
        proof that ``run`` reads the settings tree rather than leaving it to a runner."""
        monkeypatch.setenv(
            "SHIPINFER_INGEST__CAMERAS",
            '[{"camera_id": "cam-quay", "uri": "rtsp://10.0.0.7/live"}]',
        )

        assert run(chain_file, runner="inprocess", inputs=["a.mp4"], dry_run=True) == 0

        out = capsys.readouterr().out
        assert "cameras: 1 configured (cam-quay ...)" in out
        assert "cameras: 1 from --inputs (cam-000 ...)" in out

    def test_the_two_camera_counts_are_split_by_position_not_by_a_second_build(
        self, chain_file: Path, capsys, monkeypatch
    ) -> None:
        """A disabled camera is dropped from the fleet, and the report must still add up.

        The split is a slice: ``cameras_from_inputs`` mints one camera per input and
        ``cameras_to_place`` puts them last, so the tail of the placed list *is* the inputs.
        ``run`` used to build those specs a second time only to count them, which is two lists
        that agree only while both are passed the same ``loop``.
        """
        monkeypatch.setenv(
            "SHIPINFER_INGEST__CAMERAS",
            '[{"camera_id": "cam-a", "uri": "rtsp://a/live"},'
            ' {"camera_id": "cam-off", "uri": "rtsp://b/live", "enabled": false}]',
        )

        assert run(chain_file, runner="inprocess", inputs=["a.mp4", "b.mp4"], dry_run=True) == 0

        out = capsys.readouterr().out
        assert "cameras: 1 configured (cam-a ...)" in out
        assert "cameras: 2 from --inputs (cam-000 ...)" in out

    def test_a_dry_run_refuses_inputs_on_a_runner_that_manages_no_cameras(
        self, chain_file: Path
    ) -> None:
        """The combination can never work, so the mode for reading a plan must say so.

        The check used to be inside ``place_cameras``, which runs after ``start()`` — so
        ``--dry-run --inputs`` printed a plan and exited ``0`` for a ``--runner`` that would
        open none of the files, and the operator found out on the real run. It is a fact about
        the runner they named, known as soon as it is built.
        """
        with pytest.raises(ConfigurationError, match="manages no cameras"):
            run(chain_file, runner="still", inputs=["a.mp4"], dry_run=True)

    def test_a_dry_run_on_that_runner_is_still_fine_without_inputs(
        self, chain_file: Path
    ) -> None:
        """The refusal is about ``--inputs``, not about the runner."""
        assert run(chain_file, runner="still", dry_run=True) == 0
