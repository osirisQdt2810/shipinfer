"""Per-model serving counters, under Triton's names and with Triton's arithmetic.

The metrics registry already histograms every span, and that is the right instrument for a
dashboard: it survives cardinality and it answers "what is the p99 across the fleet". What
it cannot answer is "what has *this* model done since it loaded" — a histogram has no
cumulative request count per model, and `/v2/statistics` returned the whole server, so an
operator debugging one camera's model had to read every model's numbers to find one.

That is exactly the gap Triton fills with `/v2/models/{name}/stats`, so the shape here is
Triton's, field for field. Two conventions of theirs are worth stating because they look
like bugs otherwise:

* **`inference_stats.compute_*` counts requests, not executions**, and charges each request
  in a batch the whole batch's span. So `ns / count` reads as "the compute latency a request
  experienced", which is what a latency investigation wants — not device time. Summing it
  across requests deliberately over-counts wall clock.
* **`batch_stats[].compute_*` counts executions**, and its `ns` is the sum of the batch
  spans. That one *is* device time, and dividing the two gives the cost of a batch at that
  size — which is how an operator decides whether the batching window is paying for itself.

Both are recorded here so the difference is visible rather than being a footnote.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

__all__ = ["DurationStat", "ModelStatistics"]

#: Triton's `inference_stats` keys, in the order the protocol lists them. Kept as a tuple so
#: :meth:`ModelStatistics.as_dict` renders every key even when it is zero: an absent key and
#: a zero are different claims, and a dashboard that hides "0 failures" cannot distinguish
#: "nothing failed" from "this server does not report failures".
_INFERENCE_STATS = (
    "success",
    "fail",
    "queue",
    "compute_input",
    "compute_infer",
    "compute_output",
    "cache_hit",
    "cache_miss",
)

_BATCH_PHASES = ("compute_input", "compute_infer", "compute_output")


@dataclass(slots=True)
class DurationStat:
    """A ``(count, ns)`` pair — Triton's unit of statistics.

    Not a histogram on purpose. This is the *cumulative* view an operator diffs between two
    scrapes to get a rate; the distribution lives in the metrics registry, and duplicating
    it here would double the memory for a worse answer.
    """

    count: int = 0
    ns: int = 0

    def observe_total(self, ns_total: int, count: int) -> None:
        """Add ``count`` observations whose durations are **already summed**.

        The sibling of :meth:`observe`, and the two are not interchangeable — telling them
        apart is the whole reason both exist.

        `observe` takes one *per-request* span and credits every request in the batch with
        it; that is right for a phase timing, which is a single span the whole batch shares.
        This takes a total already accumulated per request in a loop, and must not multiply
        it again. `ModelInstance._execute` builds `queue_ns` and `total_ns` exactly that way
        — `for item in items: queue_ns += ...` — and passing them to `observe` made both
        figures **exactly `batch_size` times too large**.

        That is not a rounding error in a debug counter. At `max_batch_size: 32` an operator
        watching `person_embedder` sees a mean queue wait of ~5 ms where the truth is ~167
        us, and the natural response — add instances to a pool that is not backed up — takes
        GPU away from the stage that really is behind. It is the one number in this endpoint
        an autoscaler or a pager keys on.

        The old suite could not see it: the only tests with `requests > 1` did not assert on
        `queue`, and the serving-path test used a batch of one, where `n == n * 1`.
        """
        self.count += count
        self.ns += max(0, ns_total)

    def observe(self, ns: int, count: int = 1) -> None:
        """Add ``count`` observations of ``ns`` each.

        ``count > 1`` adds ``count * ns``, not ``ns``, because that is what the argument
        means: the same span happened to that many requests. It is how Triton charges a
        batch — every request in it is credited the whole batch's duration — so ``ns/count``
        reads as "the latency a request experienced" and the sum deliberately exceeds wall
        clock. Adding it once instead would silently divide the reported latency by the
        batch size, which is a flattering error and therefore the dangerous direction.

        ``ns`` is clamped at zero. A negative span means the two stamps came from different
        clocks or one was never written; folding it in would make the running total drift
        downwards and there is no recovering the real number afterwards.
        """
        self.count += count
        self.ns += max(0, ns) * count

    def as_dict(self) -> dict[str, int]:
        return {"count": self.count, "ns": self.ns}


class ModelStatistics:
    """Cumulative counters for one model, safe to update from every instance thread.

    One lock, taken **once per batch execution** rather than once per request: at the design
    point of 1000 frames a second across 16 GPUs a per-request lock would be the contended
    object in the system, and the batch is the natural unit anyway because every span except
    the queue wait is measured per batch.
    """

    __slots__ = (
        "_batches",
        "_lock",
        "_stats",
        "execution_count",
        "inference_count",
        "last_inference_ns",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, DurationStat] = {
            name: DurationStat() for name in _INFERENCE_STATS
        }
        self._batches: dict[int, dict[str, DurationStat]] = {}
        self.inference_count = 0
        self.execution_count = 0
        #: Unix nanoseconds of the most recent completed inference, or 0 if there has been
        #: none. Wall clock rather than monotonic because it leaves the process: an operator
        #: comparing it against `date` needs the two to mean the same thing.
        self.last_inference_ns = 0

    # -- recording -----------------------------------------------------------------------

    def record_execution(
        self,
        *,
        requests: int,
        batch_size: int,
        queue_ns: int,
        compute_input_ns: int,
        compute_infer_ns: int,
        compute_output_ns: int,
        total_ns: int,
    ) -> None:
        """Record one successful batch.

        Args:
            requests: requests in the batch. Every per-request stat is credited this many
                times, which is Triton's convention (see the module docstring).
            batch_size: **rows** in the batch, which is what keys ``batch_stats``. Not the
                same as ``requests``: one request may carry several rows, and a batch of 4
                rows costs what 4 rows cost however many callers contributed them.
            queue_ns: summed queue wait **across the batch's requests**, not one request's.
            compute_input_ns: assembling the batch and staging it to the device.
            compute_infer_ns: the backend's execute.
            compute_output_ns: reading results back and scattering them.
            total_ns: summed end-to-end duration across the batch's requests.
        """
        with self._lock:
            self.inference_count += requests
            self.execution_count += 1
            self.last_inference_ns = time.time_ns()
            # `observe_total` for the two the caller already summed per request, `observe`
            # for the three that are one span the whole batch shared. Three of five were
            # right, which is what made the other two easy to miss.
            self._stats["success"].observe_total(total_ns, requests)
            self._stats["queue"].observe_total(queue_ns, requests)
            self._stats["compute_input"].observe(compute_input_ns, requests)
            self._stats["compute_infer"].observe(compute_infer_ns, requests)
            self._stats["compute_output"].observe(compute_output_ns, requests)

            per_size = self._batches.get(batch_size)
            if per_size is None:
                per_size = {name: DurationStat() for name in _BATCH_PHASES}
                self._batches[batch_size] = per_size
            per_size["compute_input"].observe(compute_input_ns)
            per_size["compute_infer"].observe(compute_infer_ns)
            per_size["compute_output"].observe(compute_output_ns)

    def record_failure(self, count: int, total_ns: int = 0) -> None:
        """Record ``count`` requests that did not produce a response.

        Kept separate from success rather than derived by subtraction: a failed batch has no
        compute spans worth attributing, and folding it into `success` would inflate exactly
        the number an operator uses to judge whether the model is healthy.
        """
        with self._lock:
            self._stats["fail"].observe(total_ns, count)

    def record_cache_hit(self, count: int = 1) -> None:
        with self._lock:
            self._stats["cache_hit"].observe(0, count)

    def record_cache_miss(self, count: int = 1) -> None:
        with self._lock:
            self._stats["cache_miss"].observe(0, count)

    # -- reading -------------------------------------------------------------------------

    def as_dict(self, name: str, version: int) -> dict[str, Any]:
        """One entry of Triton's ``model_stats`` array.

        ``version`` is stringified and ``last_inference`` is milliseconds since the epoch,
        both because that is what the protocol says and what a Triton client will parse.
        """
        with self._lock:
            return {
                "name": name,
                "version": str(version),
                "last_inference": self.last_inference_ns // 1_000_000,
                "inference_count": self.inference_count,
                "execution_count": self.execution_count,
                "inference_stats": {
                    key: self._stats[key].as_dict() for key in _INFERENCE_STATS
                },
                "batch_stats": [
                    {
                        "batch_size": size,
                        "count": phases["compute_infer"].count,
                        **{phase: phases[phase].as_dict() for phase in _BATCH_PHASES},
                    }
                    for size, phases in sorted(self._batches.items())
                ],
            }

    def __repr__(self) -> str:
        return (
            f"<ModelStatistics inferences={self.inference_count} "
            f"executions={self.execution_count}>"
        )
