"""Metrics: values, quantiles and the Prometheus rendering."""

from __future__ import annotations

from shipinfer.core.metrics import EXPORTERS, MetricsRegistry, PrometheusExporter, ServerMetrics


def test_counter_accumulates_per_label_set() -> None:
    registry = MetricsRegistry()
    counter = registry.counter("c", "help")
    counter.inc(model="a")
    counter.inc(2, model="a")
    counter.inc(model="b")
    assert counter.value(model="a") == 3
    assert counter.value(model="b") == 1
    assert counter.total() == 4


def test_gauge_goes_both_ways() -> None:
    gauge = MetricsRegistry().gauge("g", "help")
    gauge.set(5, device="cuda:0")
    gauge.dec(2, device="cuda:0")
    assert gauge.value(device="cuda:0") == 3


def test_histogram_quantiles_are_bucket_resolution() -> None:
    histogram = MetricsRegistry().histogram("h", "help", buckets=(10, 100, 1000))
    for value in (5, 50, 500, 5000):
        histogram.observe(value)
    count, total = histogram.snapshot()
    assert count == 4
    assert total == 5555
    assert histogram.quantile(0.5) == 100
    assert histogram.quantile(0.99) == 1000


def test_labels_are_order_independent() -> None:
    counter = MetricsRegistry().counter("c", "help")
    counter.inc(model="m", device="cuda:0")
    assert counter.value(device="cuda:0", model="m") == 1


def test_prometheus_rendering_is_well_formed() -> None:
    registry = MetricsRegistry()
    registry.counter("requests_total", "Requests.").inc(3, model="m")
    registry.histogram("latency_us", "Latency.", buckets=(100, 1000)).observe(150, model="m")

    text = PrometheusExporter().render(registry)

    assert "# HELP requests_total Requests." in text
    assert "# TYPE requests_total counter" in text
    assert 'requests_total{model="m"} 3' in text
    assert 'latency_us_bucket{model="m",le="1000"} 1' in text
    assert 'latency_us_count{model="m"} 1' in text


def test_jsonl_exporter_is_registered() -> None:
    registry = MetricsRegistry()
    registry.counter("c", "help").inc()
    text = EXPORTERS.create("jsonl").render(registry)
    assert '"metric":"c"' in text


def test_server_metrics_resolve_once() -> None:
    """Handles are resolved at construction so the hot path is an attribute load."""
    metrics = ServerMetrics()
    metrics.requests_total.inc(model="m")
    assert metrics.requests_total.value(model="m") == 1
    assert "shipinfer_requests_total" in metrics.registry
