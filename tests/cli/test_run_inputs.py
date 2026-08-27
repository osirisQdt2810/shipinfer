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

from shipinfer.cli.commands.run import cameras_from_inputs, place_cameras, run
from shipinfer.core.errors import ConfigurationError
from shipinfer.core.request import ResponseFuture
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

    @pytest.mark.parametrize("inputs", [None, [], ()])
    def test_no_inputs_is_no_cameras_and_not_an_error(self, inputs: Any) -> None:
        """Bringing a chain up empty and adding cameras over the control plane is normal."""
        assert cameras_from_inputs(inputs) == []


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
