"""The named metric handles the server actually uses."""

from __future__ import annotations

from dataclasses import dataclass, field

from shipinfer.core.metrics.counter import Counter
from shipinfer.core.metrics.gauge import Gauge
from shipinfer.core.metrics.histogram import Histogram
from shipinfer.core.metrics.registry import MetricsRegistry

__all__ = ["ServerMetrics"]

_BATCH_BUCKETS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


@dataclass(slots=True)
class ServerMetrics:
    """Metric handles resolved once, at construction.

    Resolving eagerly turns every hot-path call site into an attribute load plus an
    ``+=``. Looking them up by string per request would add a dict lookup and a hash to
    the critical path for no benefit whatsoever.
    """

    registry: MetricsRegistry = field(default_factory=MetricsRegistry)

    requests_total: Counter = field(init=False)
    requests_failed: Counter = field(init=False)
    requests_rejected: Counter = field(init=False)
    requests_expired: Counter = field(init=False)
    spills_total: Counter = field(init=False)
    cache_hits: Counter = field(init=False)
    cache_misses: Counter = field(init=False)
    graph_replays_total: Counter = field(init=False)
    batches_total: Counter = field(init=False)

    batch_size: Histogram = field(init=False)
    queue_wait_us: Histogram = field(init=False)
    compute_us: Histogram = field(init=False)
    h2d_us: Histogram = field(init=False)
    d2h_us: Histogram = field(init=False)
    e2e_us: Histogram = field(init=False)

    queue_depth: Gauge = field(init=False)
    instances_ready: Gauge = field(init=False)
    device_memory_free: Gauge = field(init=False)

    def __post_init__(self) -> None:
        r = self.registry
        self.requests_total = r.counter("shipinfer_requests_total", "Requests accepted.")
        self.requests_failed = r.counter(
            "shipinfer_requests_failed_total", "Requests that raised."
        )
        self.requests_rejected = r.counter(
            "shipinfer_requests_rejected_total", "Requests refused by backpressure."
        )
        self.requests_expired = r.counter(
            "shipinfer_requests_expired_total", "Requests dropped past their deadline."
        )
        self.spills_total = r.counter(
            "shipinfer_spills_total", "Requests routed off their resident GPU."
        )
        self.cache_hits = r.counter(
            "shipinfer_response_cache_hits_total", "Requests answered from the response cache."
        )
        self.cache_misses = r.counter(
            "shipinfer_response_cache_misses_total",
            "Cacheable requests the cache did not have.",
        )
        self.graph_replays_total = r.counter(
            "shipinfer_cuda_graph_replays_total", "Executions served by a captured CUDA graph."
        )
        self.batches_total = r.counter("shipinfer_batches_total", "Batches executed.")

        self.batch_size = r.histogram(
            "shipinfer_batch_size", "Requests per executed batch.", _BATCH_BUCKETS
        )
        self.queue_wait_us = r.histogram("shipinfer_queue_wait_us", "Queue wait, microseconds.")
        self.compute_us = r.histogram("shipinfer_compute_us", "Backend execute, microseconds.")
        self.h2d_us = r.histogram("shipinfer_h2d_us", "Host-to-device staging, microseconds.")
        self.d2h_us = r.histogram("shipinfer_d2h_us", "Device-to-host readback, microseconds.")
        self.e2e_us = r.histogram("shipinfer_e2e_us", "Accept to response, microseconds.")

        self.queue_depth = r.gauge("shipinfer_queue_depth", "Requests waiting per instance.")
        self.instances_ready = r.gauge("shipinfer_instances_ready", "Ready model instances.")
        self.device_memory_free = r.gauge(
            "shipinfer_device_memory_free_bytes", "Free memory per device."
        )
