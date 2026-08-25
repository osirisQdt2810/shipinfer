"""The per-stage latency histogram can tell 5 ms from 100 ms.

`Histogram.quantile` reports the upper edge of the bucket the quantile falls in. With the
registry default's 2-2.5x steps, the first complete algo-tier run reported every stage's p50 as
25 000 / 50 000 / 100 000 us and was misread as saturation — "100 000" meant only "somewhere in
50-100 ms". A tier whose answer is uncertain by 2x cannot say whether a change helped.
"""

from __future__ import annotations

from itertools import pairwise

from shipinfer.pipeline.metrics import _STAGE_BUCKETS_US, PipelineMetrics


class TestTheStageBucketsAreFineEnoughToRankStages:
    def test_consecutive_edges_are_about_one_point_six_apart(self) -> None:
        """Every step to 1 s is ~1.6x; the last edge is the 2 s catch-all for a stalled stage."""
        ratios = [b / a for a, b in pairwise(_STAGE_BUCKETS_US)]
        ladder, catch_all = ratios[:-1], ratios[-1]
        assert max(ladder) <= 1.65, f"a step above 1.65x: {max(ladder):.2f}"
        assert min(ladder) >= 1.5, f"a step below 1.5x buys nothing: {min(ladder):.2f}"
        assert catch_all == 2.0

    def test_they_cover_a_fast_crop_and_a_stalled_stage(self) -> None:
        assert _STAGE_BUCKETS_US[0] <= 250.0
        assert _STAGE_BUCKETS_US[-1] >= 1_000_000.0

    def test_the_pipeline_histogram_uses_them(self) -> None:
        metrics = PipelineMetrics()
        assert tuple(metrics.stage_latency_us.buckets) == tuple(_STAGE_BUCKETS_US)
