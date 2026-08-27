"""The runner's metric handles, resolved once at construction.

Same shape and the same reason as :class:`shipinfer.pipeline.metrics.PipelineMetrics`,
:class:`shipinfer.ingest.IngestMetrics` and :class:`shipinfer.core.metrics.ServerMetrics`:
at 1000 frames a second a metric looked up by string per frame is a hash and a dict probe
nobody needs to pay for, so every handle is resolved here, once.

**Every counter is labelled by ``camera``**, which is the whole point of using the metrics
primitive rather than four integers behind a lock. The number that matters in this system is
never the shard total — it is *which* camera is being starved or dropped, and that is the
lesson of the previous generation, whose shared 1000-slot buffer could report "1000 entries"
and never "camera 7 lost 40% of its frames" (ADR-005). A plain ``+= 1`` under a lock answers
the first question and destroys the evidence for the second.

The lost-update question the lock used to answer:
:class:`~shipinfer.core.metrics.Counter` takes its lock only when a label set is seen for the
first time — when the dict actually mutates its structure — and accepts a lost increment
under extreme contention on the steady-state path. That is the trade the whole codebase
already makes on the dispatch path, and it is the right one here too: a runner's counters are
an operational signal, not an accounting ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shipinfer.core.metrics import Counter, MetricsRegistry

__all__ = ["RunnerMetrics"]


@dataclass(slots=True)
class RunnerMetrics:
    """Counters for one runner's admission and walk, all labelled by camera."""

    registry: MetricsRegistry = field(default_factory=MetricsRegistry)

    items_accepted: Counter = field(init=False)
    items_walked: Counter = field(init=False)
    items_failed: Counter = field(init=False)
    items_expired: Counter = field(init=False)
    items_timed_out: Counter = field(init=False)
    items_dropped: Counter = field(init=False)
    items_queue_closed: Counter = field(init=False)

    def __post_init__(self) -> None:
        r = self.registry
        self.items_accepted = r.counter(
            "shipinfer_runner_items_accepted_total",
            "Chain items admitted into the runner's lane, per camera.",
        )
        self.items_walked = r.counter(
            "shipinfer_runner_items_walked_total",
            "Chain items walked to the end of the chain, per camera.",
        )
        self.items_failed = r.counter(
            "shipinfer_runner_items_failed_total",
            "Chain items whose walk raised, per camera. An element failure costs one item "
            "and not the worker, so this rising while the chain stays healthy is the "
            "signal to look at.",
        )
        self.items_expired = r.counter(
            "shipinfer_runner_items_expired_total",
            "Chain items whose deadline passed before the walk reached them, per camera. "
            "Rising here means the shard is behind, not that anything is broken.",
        )
        self.items_timed_out = r.counter(
            "shipinfer_runner_items_timed_out_total",
            # Split out of `failed` because the two ask for opposite responses: a model that
            # did not answer within `pipeline.stage_timeout_ms` is a saturation signal and the
            # fix is capacity, while `failed` is a bug and the fix is code. Merged, neither
            # number answers "should I add a GPU or open a ticket".
            "Chain items whose model did not answer within the stage timeout, per camera.",
        )
        self.items_dropped = r.counter(
            "shipinfer_runner_items_dropped_total",
            # The single most important number here, and the reason these are labelled at
            # all: it names the camera that flooded, not the one that suffered.
            "Chain items lost to backpressure, per camera: the runner's own lane refusing a "
            "submission, and a `pool` element's model queue refusing a request mid-walk. One "
            "counter for both because the camera lost a frame and the fix is capacity either "
            "way; `stats()['queue']` says which side of the chain it was refused on.",
        )
        self.items_queue_closed = r.counter(
            "shipinfer_runner_items_queue_closed_total",
            # Without this, everything the queue itself resolved was invisible in `stats()`:
            # the items had typed futures and no counter, so `accepted` outran every outcome.
            "Chain items still queued when the runner stopped, failed by the queue's close, "
            "per camera.",
        )

    def totals(self) -> dict[str, int]:
        """The process-local view an operator reads in ``stats()``.

        The per-camera breakdown stays available on the handles
        (``metrics.items_dropped.value(camera="cam-7")``) and through any
        :class:`~shipinfer.core.metrics.exporters.MetricsExporter` over
        :attr:`registry`; this is only the fleet-wide roll-up, kept because
        "accepted 6, walked 6" is what a test asserts and a health check prints.
        """
        return {
            "accepted": int(self.items_accepted.total()),
            "walked": int(self.items_walked.total()),
            "failed": int(self.items_failed.total()),
            "expired": int(self.items_expired.total()),
            "timed_out": int(self.items_timed_out.total()),
            "dropped": int(self.items_dropped.total()),
            "queue_closed": int(self.items_queue_closed.total()),
        }
