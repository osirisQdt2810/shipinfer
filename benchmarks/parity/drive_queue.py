"""Drive one queue scenario over the real Python request queue and write its trace.

Single-threaded and clock-free, deliberately. The ingest harness groups its records per
camera because thread interleaving is nondeterministic; here the opposite is true -- which
camera's item comes out next IS the invariant -- so every record is fleet-level and the
whole trace is one ordered sequence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.parity.queue_scenario import Op, QueueScenario, load_queue_scenario
from benchmarks.parity.trace import Trace, TraceWriter
from shipinfer.core.errors import ConfigurationError, QueueFullError, RequestCancelledError
from shipinfer.core.request import InferenceRequest, Priority, RequestContext, ResponseFuture
from shipinfer.core.settings import OverflowPolicy
from shipinfer.core.types import Tensor
from shipinfer.scheduling.queues import QUEUES
from shipinfer.scheduling.queues.base import BatchWindow
from shipinfer.scheduling.work import WorkItem

__all__ = ["GOLDEN", "SCENARIOS", "run_queue_scenario"]

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "benchmarks" / "parity" / "scenarios" / "queues"
GOLDEN = ROOT / "benchmarks" / "parity" / "golden"

#: A deadline one nanosecond into the monotonic epoch: in the past on any machine, on either
#: plane, without a clock seam in the queue. Expiry is then a property of the item and not of
#: how long the run took, which is the only way it can be compared byte for byte.
_LONG_PAST = 1


def _item(op: Op) -> WorkItem:
    """One work item for a ``put``, carrying ``rows`` rows so a row budget can be reached."""
    request = InferenceRequest(
        model_name="parity",
        inputs={"x": Tensor.from_numpy(np.zeros((op.rows, 1), dtype=np.float32))},
        context=RequestContext(camera_id=op.camera),
        priority=Priority[op.priority.upper()],
        deadline_ns=_LONG_PAST if op.expired else 0,
    )
    return WorkItem(request, ResponseFuture(request))


class _Run:
    """One scenario's state: the queue, the items put into it, and the trace being written."""

    def __init__(self, scenario: QueueScenario) -> None:
        self.scenario = scenario
        self.writer = TraceWriter()
        self.writer.header(scenario.name, "python")
        # `Registry.create` takes the implementation name positionally and the queue's own
        # name is the first constructor argument, so both are positional here.
        self.queue = QUEUES.create(
            scenario.queue,
            scenario.name,
            scenario.capacity,
            overflow=OverflowPolicy(scenario.overflow),
        )
        self.window = BatchWindow(
            max_batch_size=scenario.max_batch_size, max_delay_us=scenario.max_delay_us
        )
        self._live: list[WorkItem] = []

    def run(self) -> Trace:
        for index, op in enumerate(self.scenario.ops, start=1):
            getattr(self, f"_{op.verb}")(op, index)
        self._emit_stats()
        return self.writer.trace()

    # -- the operations ----------------------------------------------------------------

    def _put(self, op: Op, index: int) -> None:
        item = _item(op)
        try:
            self.queue.put(item)
        except QueueFullError:
            status = "rejected"
        except RequestCancelledError:
            status = "closed"
        else:
            status = "accepted"
            self._live.append(item)
        self.writer.record(
            "qput", numbers=(op.rows, self.queue.depth), text=(op.camera, status)
        )
        self._sweep(index, "put")

    def _take(self, op: Op, index: int) -> None:
        if self.queue.depth == 0 and not self.queue.is_closed:
            raise ConfigurationError(
                f"{self.scenario.name}: operation {index} takes from an open empty queue, "
                f"where `get_batch` blocks by contract -- put something first, or close"
            )
        batch = self.queue.get_batch(self.window, poll_s=0.0)
        self.writer.record(
            "qbatch", numbers=(len(batch), sum(i.request.batch_size for i in batch))
        )
        for served in batch:
            self._live.remove(served)
            self.writer.record(
                "qserved", numbers=(served.request.batch_size,), text=(served.fairness_key,)
            )
        self._sweep(index, "take")

    def _close(self, op: Op, index: int) -> None:
        for drained in self.queue.close():
            self._live.remove(drained)
            self.writer.record("qdrop", text=(drained.fairness_key, "closed"))

    # -- what the queue did to items it did not hand back ------------------------------

    def _sweep(self, index: int, verb: str) -> None:
        """Emit a ``qdrop`` for every item the queue failed during this operation.

        The Python queue reports a drop by failing the item's future; the C++ queue calls
        `on_drop` with the reason. Same event, same moment -- but the *order* of several
        drops in one operation is each queue's internal order and neither promises it, so
        more than one is refused here rather than compared. ``close`` is the exception: it
        returns its drained items in order, so :meth:`_close` emits them itself.
        """
        dropped = [item for item in self._live if item.future.done()]
        if len(dropped) > 1:
            raise ConfigurationError(
                f"{self.scenario.name}: operation {index} ({verb}) dropped "
                f"{len(dropped)} items at once, whose order neither queue promises. Split "
                f"the scenario so at most one item is dropped per operation"
            )
        for item in dropped:
            self._live.remove(item)
            self.writer.record("qdrop", text=(item.fairness_key, _reason(item)))

    def _emit_stats(self) -> None:
        """The totals, then the per-camera half of ADR-005 in a fixed camera order."""
        stats = self.queue.stats()
        self.writer.record(
            "qstats",
            numbers=(
                stats.accepted,
                stats.rejected,
                stats.evicted,
                stats.expired,
                stats.depth,
                stats.capacity,
            ),
        )
        # Every camera the SCENARIO names, not only those with a non-zero count: a row that
        # appears or vanishes is a length difference, and a length difference says less than
        # "cam_quiet's evicted went 0 -> 1" -- which is the regression this seam exists for.
        for camera in sorted({op.camera for op in self.scenario.ops if op.camera}):
            self.writer.record(
                "qcam",
                numbers=(
                    stats.depth_by_camera.get(camera, 0),
                    stats.rejected_by_camera.get(camera, 0),
                    stats.evicted_by_camera.get(camera, 0),
                    stats.expired_by_camera.get(camera, 0),
                ),
                text=(camera,),
            )


def _reason(item: WorkItem) -> str:
    """``evicted`` / ``expired`` / ``closed``, spelled as ``DropReason`` spells them.

    Read from the error TYPE and not from its words: the Python queue evicts by failing the
    victim with the same ``QueueFullError`` it raises at a rejected producer, so a message
    match would have to know that "full" means "evicted" here and "rejected" there. The two
    ``RequestCancelledError`` cases are the one place the message carries the difference,
    and the queue writes both.
    """
    error = item.future.exception()
    if isinstance(error, QueueFullError):
        return "evicted"
    if isinstance(error, RequestCancelledError):
        return "expired" if "deadline" in str(error) else "closed"
    raise ConfigurationError(f"a dropped item carries no known reason: {error!r}")


def run_queue_scenario(scenario: QueueScenario) -> Trace:
    """Run one scenario and return its trace."""
    return _Run(scenario).run()


def load(name: str) -> QueueScenario:
    """A scenario by name under ``scenarios/queues/``, or by path."""
    named = Path(name)
    return load_queue_scenario(named if named.suffix == ".scn" else SCENARIOS / f"{name}.scn")
