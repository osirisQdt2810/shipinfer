"""Drive one scenario over the real Python :class:`~shipinfer.ingest.IngestManager`.

The instrument is a scripted :class:`~shipinfer.ingest.base.FrameSource` and a recording
sink -- nothing else is faked. The manager, the actor, the backoff and the health
bookkeeping are the production ones, which is the only reason the trace says anything.

Everything is observed from the **actor's own thread**, at the source and sink calls, so
what the trace records is that thread's program order rather than a poll racing it. The
actor's state and failure count are read out of its real ``health`` snapshot at each of
those points; the un-jittered backoff peek is the one number computed rather than measured,
because the jitter draws from a different generator in each plane (see ``README.md``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.parity.scenario import (
    INT_SETTINGS,
    CameraScript,
    Scenario,
    load_scenario,
)
from benchmarks.parity.trace import Trace, TraceWriter
from shipinfer.core.errors import (
    ConfigurationError,
    FrameDecodeError,
    QueueFullError,
    RequestCancelledError,
    ServerStateError,
    SourceOpenError,
    SourceUnavailableError,
)
from shipinfer.core.settings.ingest import CameraConfig, IngestSettings
from shipinfer.ingest.base import FrameSource
from shipinfer.ingest.camera.actor import CameraActor
from shipinfer.ingest.frame.frame import Frame
from shipinfer.ingest.frame.tag import FrameCounter
from shipinfer.ingest.manager import IngestManager

__all__ = ["GOLDEN", "HERE", "SCENARIOS", "run_scenario"]

HERE = Path(__file__).resolve().parent
SCENARIOS = HERE / "scenarios"
GOLDEN = HERE / "golden"

#: Every scripted frame is this size. Small because no pixel is ever looked at: the parity
#: property is which frames were produced and what happened to them, not what was in them.
_HEIGHT, _WIDTH = 4, 6

#: How long the whole fleet may take to finish its script. Every camera terminates on its
#: own (an exhausted read, a fatal open, a closed sink), so overrunning this means the run
#: is stuck -- which is a failure to report, never a timeout to sleep through.
_RUN_BUDGET_S = 20.0


class CameraRecorder:
    """One camera's trace, and the counters that turn its script into call outcomes.

    Touched only by that camera's actor thread (the driver reads it once, at the end), which
    is why nothing here is locked: one actor per camera for the actor's whole life is what
    ADR-002 buys.
    """

    def __init__(self, script: CameraScript, scenario: Scenario, writer: TraceWriter) -> None:
        self.script = script
        self.exhausted = False
        self.actor: CameraActor | None = None
        self._scenario = scenario
        self._writer = writer
        self._counts = {"open": 0, "read": 0, "close": 0, "sink": 0}
        self._state = "idle"
        self._failures = 0

    def emit(
        self, kind: str, numbers: tuple[int, ...] = (), text: tuple[str, ...] = ()
    ) -> None:
        self._writer.record(kind, self.script.camera_id, numbers, text)

    def step(self, what: str) -> int:
        """The 0-based index of this call, and advance."""
        index = self._counts[what]
        self._counts[what] = index + 1
        return index

    def observe(self) -> None:
        """Emit whatever the actor has done to itself since the last call boundary.

        Called first at every source and sink hook, so a state change and the retries that
        caused it land before the record of the call that observed them.
        """
        actor = self.actor
        if actor is None:
            return
        health = actor.health
        if health.state.value != self._state:
            self.emit("state", (), (self._state, health.state.value))
            self._state = health.state.value
        while health.consecutive_failures > self._failures:
            self.emit("retry", (self._failures, self._scenario.peek_us(self._failures)))
            self._failures += 1
        self._failures = min(self._failures, health.consecutive_failures)


class ScriptedSource(FrameSource):
    """A decode path with no camera, no network and no decoder -- only a script.

    Written twice, once per plane, and that duplication is the instrument's own risk: every
    call emits a ``source_*`` record, so the two scripts drifting apart shows up in the
    source-event stream before it can be mistaken for a divergence in the actor.
    """

    name = "scripted"

    def __init__(
        self,
        config: CameraConfig,
        counter: FrameCounter | None = None,
        *,
        settings: IngestSettings | None = None,
        recorder: CameraRecorder,
    ) -> None:
        super().__init__(config, counter, settings=settings)
        self._recorder = recorder
        self._image = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)

    @property
    def is_exhausted(self) -> bool:
        return self._recorder.exhausted

    def _do_open(self) -> None:
        self._recorder.observe()
        index = self._recorder.step("open")
        outcome, detail = self._recorder.script.open_at(index)
        self._recorder.emit("source_open", (index,), (outcome,))
        if outcome == "SourceOpenError":
            raise SourceOpenError(self.camera_id, self.config.uri, detail)
        if outcome == "SourceUnavailableError":
            raise SourceUnavailableError("scripted", detail)
        self._set_format(_HEIGHT, _WIDTH, 0.0)

    def _do_read(self) -> np.ndarray | None:
        self._recorder.observe()
        index = self._recorder.step("read")
        outcome, detail = self._recorder.script.read_at(index)
        self._recorder.emit("source_read", (index,), (outcome,))
        if outcome == "FrameDecodeError":
            raise FrameDecodeError(self.camera_id, detail)
        if outcome == "exhaust":
            self._recorder.exhausted = True
            return None
        if outcome == "empty":
            return None
        return self._image

    def _do_close(self) -> None:
        self._recorder.observe()
        self._recorder.emit("source_close", (self._recorder.step("close"),))


class RecordingSink:
    """The consumer, scripted per camera: accept, refuse, or say the consumer has gone.

    Records the frame it was offered and the refusal it answered with. What the *actor* then
    charged to the camera is not read here at all -- it is in the final ``health`` record, so
    a plane that forgot to count a drop diverges there rather than being covered for.
    """

    def __init__(self, recorders: dict[str, CameraRecorder]) -> None:
        self._recorders = recorders

    def put(self, frame: Frame) -> None:
        recorder = self._recorders[frame.camera_id]
        recorder.observe()
        recorder.emit("frame", (frame.frame_id,))
        outcome = recorder.script.sink_at(recorder.step("sink"))
        if outcome == "full":
            recorder.emit("drop", (), ("sink_full",))
            raise QueueFullError(f"parity:{frame.camera_id}", 1, 1)
        if outcome == "closed":
            recorder.emit("drop", (), ("sink_closed",))
            raise RequestCancelledError(f"parity sink for {frame.camera_id} is closed")


def _settings(scenario: Scenario) -> IngestSettings:
    """The fleet, resolved: the scenario's numbers plus one camera per enabled script."""
    values: dict[str, object] = {}
    for key, raw in scenario.settings.items():
        values[key] = int(raw) if key in INT_SETTINGS else float(raw)
    return IngestSettings(
        backend="scripted",
        cameras=[
            CameraConfig(
                camera_id=script.camera_id,
                uri=f"scripted://{script.camera_id}/stream",
                source="scripted",
                enabled=script.enabled,
            )
            for script in scenario.cameras
        ],
        **values,  # type: ignore[arg-type]
    )


def run_scenario(scenario: Scenario, *, plane: str = "python") -> Trace:
    """Run one scenario to completion and return its trace.

    Raises:
        ServerStateError: a camera did not finish within the run budget. Every script ends
            on its own, so this means the run is stuck and the harness says so rather than
            emitting a truncated trace that would then be compared as if it were whole.
    """
    writer = TraceWriter()
    writer.header(scenario.name, plane)
    recorders = {
        script.camera_id: CameraRecorder(script, scenario, writer)
        for script in scenario.cameras
        if script.enabled
    }
    sink = RecordingSink(recorders)

    def factory(config: CameraConfig, counter: FrameCounter) -> FrameSource:
        # The actor is resolved HERE, on the actor's own thread and before its first hook:
        # the manager publishes an actor into its map before it starts the thread, while the
        # driver only learns of it after `start()` returns -- a window in which the first
        # state change would have gone unobserved on some runs and not on others.
        recorder = recorders[config.camera_id]
        if recorder.actor is None:
            recorder.actor = manager.actor(config.camera_id)
        return ScriptedSource(config, counter, settings=None, recorder=recorder)

    manager = IngestManager(sink, settings=_settings(scenario), source_factory=factory)
    actors: dict[str, CameraActor] = {}
    try:
        manager.start()
        for camera_id, recorder in recorders.items():
            actors[camera_id] = manager.actor(camera_id)
            recorder.actor = actors[camera_id]
        _await_finish(actors)
        healths = {camera_id: actor.health for camera_id, actor in actors.items()}
        abandoned = manager.stop(timeout_s=2.0)
    finally:
        manager.stop(timeout_s=2.0)
    for camera_id in sorted(healths):
        health = healths[camera_id]
        writer.record(
            "health",
            camera_id,
            (
                health.frames_read,
                health.frames_published,
                health.frames_dropped,
                health.empty_reads,
                health.connects,
                health.connect_failures,
                health.consecutive_failures,
            ),
            (health.state.value, health.last_error),
        )
    writer.record("stop", "", (abandoned,))
    writer.record(
        "end",
        "",
        (
            len(healths),
            sum(h.frames_read for h in healths.values()),
            sum(h.frames_published for h in healths.values()),
            sum(h.frames_dropped for h in healths.values()),
        ),
    )
    return writer.trace()


def _await_finish(actors: dict[str, CameraActor]) -> None:
    deadline = time.monotonic() + _RUN_BUDGET_S
    while time.monotonic() < deadline:
        running = [name for name, actor in actors.items() if actor.is_running]
        if not running:
            return
        time.sleep(0.002)
    raise ServerStateError(
        f"parity run did not finish within {_RUN_BUDGET_S:g}s; still running: "
        f"{sorted(name for name, actor in actors.items() if actor.is_running)}"
    )


def main(argv: list[str] | None = None) -> int:
    """``python -m benchmarks.parity.drive_python --scenario reconnect --emit-golden``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, help="a name under scenarios/, or a path")
    parser.add_argument("--out", type=Path, help="write the trace here instead of stdout")
    parser.add_argument(
        "--emit-golden",
        action="store_true",
        help="write golden/<scenario>.jsonl -- the committed file BOTH planes are then held "
        "to. Regenerating one to make a plane pass is what this harness exists to prevent",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing golden")
    args = parser.parse_args(argv)

    named = Path(args.scenario)
    path = named if named.suffix == ".scn" else SCENARIOS / f"{args.scenario}.scn"
    scenario = load_scenario(path)
    trace = run_scenario(scenario)
    lines = [record.to_line() for record in trace.records]
    if len(lines) < scenario.records_min:
        raise ConfigurationError(
            f"{path}: promised at least {scenario.records_min} record(s), produced "
            f"{len(lines)}. A vacuous trace is a golden that proves nothing"
        )
    if args.emit_golden:
        destination = GOLDEN / f"{scenario.name}.jsonl"
        if destination.exists() and not args.force:
            raise ConfigurationError(
                f"{destination} already exists. A golden is captured once and committed; "
                f"pass --force only when the change to the plane IS the decision"
            )
        _write(trace, destination)
        print(f"wrote {destination} ({len(lines)} record(s))")
    elif args.out:
        _write(trace, args.out)
        print(f"wrote {args.out} ({len(lines)} record(s))")
    else:
        print("\n".join(_render(trace)))
    return 0


def _render(trace: Trace) -> list[str]:
    writer = TraceWriter()
    writer.header(trace.scenario, trace.plane)
    for record in trace.records:
        writer.record(record.kind, record.camera, record.numbers, record.text)
    return writer.lines()


def _write(trace: Trace, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_render(trace)) + "\n", encoding="ascii")


if __name__ == "__main__":
    sys.exit(main())
