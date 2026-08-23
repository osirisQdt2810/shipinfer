"""Reassembly: the inherited bug, and the three properties that fix it.

``references/bitbucket-subfaceid/docs/flow.md`` records the symptom in the operator's own
words: *"camera đông người được nhận diện đầy đủ, camera vắng người thỉnh thoảng bị miss"* —
the crowded cameras complete and the quiet ones intermittently lose frames. The cause was one
shared buffer that evicted the **globally oldest** entry, so a camera producing thirty
detections a frame pushed out the frames of a camera producing two.

:class:`TestEvictionIsFairBetweenCameras` is the regression test for exactly that, and it runs
the inherited policy beside the fixed one so the difference is a number rather than a claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from shipinfer.core.request import RequestContext
from shipinfer.core.settings.pipeline import ReassemblySettings
from shipinfer.pipeline.graph import FrameState
from shipinfer.pipeline.metrics import PipelineMetrics
from shipinfer.pipeline.reassembly import (
    COMPLETE,
    EVICTED,
    EVICTION_POLICIES,
    INCOMPLETE,
    SHUTDOWN,
    TIMEOUT,
    FrameCollector,
    FrameResult,
)

pytestmark = pytest.mark.timeout(30)


class Clock:
    """A monotonic-nanosecond clock a test advances by hand.

    Injected rather than slept through: the reassembly timeout is 1500 ms, and a suite that
    waited it out would take longer than the rest of the offline tier put together.
    """

    def __init__(self, now_ns: int = 1_000_000_000) -> None:
        self.now = now_ns

    def __call__(self) -> int:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += int(ms * 1_000_000)


def make_state(camera: str, frame: int, **kwargs) -> FrameState:
    """A minimal state: reassembly never looks at the pixels."""
    return FrameState(
        context=RequestContext(camera_id=camera, frame_id=frame, captured_ns=1),
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        **kwargs,
    )


def collector(**kwargs) -> tuple[FrameCollector, list[FrameResult]]:
    reported: list[FrameResult] = []
    settings = kwargs.pop("settings", ReassemblySettings())
    return FrameCollector(reported.append, settings=settings, **kwargs), reported


class TestEvictionIsFairBetweenCameras:
    """One camera floods; the quiet one keeps its place. This is the whole point of ADR-005."""

    def test_the_greedy_camera_loses_its_own_entries_and_the_quiet_one_keeps_both(self):
        """The quiet camera publishes **first**, so its frames are the globally oldest —
        precisely the case a drop-oldest buffer gets wrong."""
        subject, _ = collector(settings=ReassemblySettings(capacity=16))

        for frame in range(2):
            subject.open(make_state("quiet", frame), expected=("detect",))
        for frame in range(100):
            subject.open(make_state("loud", frame), expected=("detect",))

        pending = subject.pending_per_camera()
        assert pending == {"quiet": 2, "loud": 14}, "the quiet camera was collateral"
        assert subject.evicted == 86
        assert len(subject) == 16

    def test_every_eviction_is_charged_to_the_camera_that_caused_it(self):
        metrics = PipelineMetrics()
        subject, _ = collector(settings=ReassemblySettings(capacity=16), metrics=metrics)

        for frame in range(2):
            subject.open(make_state("quiet", frame), expected=("detect",))
        for frame in range(100):
            subject.open(make_state("loud", frame), expected=("detect",))

        assert metrics.frames_evicted.value(camera="loud") == 86
        assert metrics.frames_evicted.value(camera="quiet") == 0

    def test_the_inherited_policy_loses_the_quiet_camera_entirely(self):
        """The foil. Shipped so the fix is a comparison rather than an assertion of faith."""
        subject, _ = collector(
            settings=ReassemblySettings(capacity=16),
            policy=EVICTION_POLICIES.create("oldest_frame"),
        )

        for frame in range(2):
            subject.open(make_state("quiet", frame), expected=("detect",))
        for frame in range(100):
            subject.open(make_state("loud", frame), expected=("detect",))

        assert subject.pending_per_camera() == {"loud": 16}
        assert "quiet" not in subject.pending_per_camera(), (
            "this is the inherited bug reproduced on purpose: evicting the oldest entry "
            "deleted both of the quiet camera's frames"
        )

    def test_three_cameras_converge_towards_equal_shares(self):
        """Not a fairness *proof*, but the shape: nobody is starved out of the buffer."""
        subject, _ = collector(settings=ReassemblySettings(capacity=30))

        for frame in range(200):
            for camera in ("a", "b", "c"):
                subject.open(make_state(camera, frame), expected=("detect",))

        pending = subject.pending_per_camera()
        assert set(pending) == {"a", "b", "c"}
        assert max(pending.values()) - min(pending.values()) <= 1

    def test_a_tie_is_broken_deterministically(self):
        """Otherwise a fairness assertion is flaky exactly when two cameras are equally loud."""
        first, _ = collector(settings=ReassemblySettings(capacity=4))
        second, _ = collector(settings=ReassemblySettings(capacity=4))
        for subject in (first, second):
            for camera in ("a", "b"):
                for frame in range(2):
                    subject.open(make_state(camera, frame), expected=("detect",))
            subject.open(make_state("c", 0), expected=("detect",))

        assert first.pending_per_camera() == second.pending_per_camera()


class TestItIsBounded:
    """A camera whose results never arrive cannot grow anything without limit."""

    def test_thousands_of_never_completing_frames_leave_every_structure_bounded(self):
        subject, reported = collector(settings=ReassemblySettings(capacity=64))

        for frame in range(5_000):
            subject.open(make_state("stuck", frame), expected=("detect", "ship_embedder"))

        assert subject.sizes() == {"pending": 64, "cameras": 1, "camera_entries": 64}
        assert subject.opened == 5_000
        assert subject.evicted == 4_936
        # Every opened frame left exactly one way out: still pending, or reported.
        assert subject.opened == subject.reported + len(subject)
        assert len(reported) == 4_936

    def test_the_per_camera_index_forgets_a_camera_that_goes_away(self):
        """A 24/7 process that has seen a thousand transient camera ids must not keep a
        thousand empty dicts."""
        subject, _ = collector(settings=ReassemblySettings(capacity=1_000))

        for index in range(500):
            camera = f"transient-{index}"
            subject.open(make_state(camera, 0), expected=("detect",))
            subject.deliver((camera, 0), "detect")
            subject.seal((camera, 0))

        assert subject.sizes() == {"pending": 0, "cameras": 0, "camera_entries": 0}

    def test_capacity_is_honoured_exactly(self):
        subject, _ = collector(settings=ReassemblySettings(capacity=1))
        subject.open(make_state("a", 0), expected=("detect",))
        subject.open(make_state("a", 1), expected=("detect",))
        assert len(subject) == 1
        assert subject.capacity == 1


class TestATimeoutEmitsRatherThanDrops:
    """A frame that lost a stage is published partial, naming what it lost."""

    def test_a_timed_out_frame_is_emitted_naming_the_missing_stages(self):
        clock = Clock()
        subject, reported = collector(
            settings=ReassemblySettings(capacity=8, timeout_ms=1_500), clock=clock
        )
        subject.open(make_state("cam0", 3))
        subject.expect(("cam0", 3), ("detect", "ship_embedder", "ship_recognizer"))
        subject.deliver(("cam0", 3), "detect")

        assert subject.sweep() == 0, "swept before the deadline"
        clock.advance_ms(1_500)
        assert subject.sweep() == 1

        (result,) = reported
        assert result.key == ("cam0", 3)
        assert result.reason == TIMEOUT
        assert result.missing == ("ship_embedder", "ship_recognizer")
        assert result.delivered == ("detect",)
        assert result.is_partial
        assert len(subject) == 0

    def test_a_partial_emission_is_observable_through_metrics(self):
        clock = Clock()
        metrics = PipelineMetrics()
        subject, _ = collector(
            settings=ReassemblySettings(capacity=8, timeout_ms=100),
            metrics=metrics,
            clock=clock,
        )
        subject.open(make_state("cam7", 1), expected=("detect", "person_embedder"))
        subject.deliver(("cam7", 1), "detect")
        clock.advance_ms(100)
        subject.sweep()

        assert metrics.frames_partial.value(camera="cam7", reason=TIMEOUT) == 1
        assert metrics.frames_emitted.value(camera="cam7", reason=TIMEOUT) == 1
        assert metrics.frames_partial.value(camera="cam7", reason=COMPLETE) == 0

    def test_a_frames_own_deadline_tightens_the_reassembly_timeout(self):
        """No point holding memory past the moment the result stops being actionable."""
        clock = Clock()
        subject, reported = collector(
            settings=ReassemblySettings(capacity=8, timeout_ms=10_000), clock=clock
        )
        subject.open(
            make_state("cam0", 0, deadline_ns=clock.now + 50_000_000), expected=("detect",)
        )

        clock.advance_ms(60)
        assert subject.sweep() == 1
        assert reported[0].reason == TIMEOUT

    def test_a_delivery_after_the_timeout_is_counted_as_late_not_republished(self):
        clock = Clock()
        metrics = PipelineMetrics()
        subject, reported = collector(
            settings=ReassemblySettings(capacity=8, timeout_ms=100),
            metrics=metrics,
            clock=clock,
        )
        subject.open(make_state("cam0", 0), expected=("detect",))
        clock.advance_ms(100)
        subject.sweep()

        subject.deliver(("cam0", 0), "detect")

        assert len(reported) == 1, "a late result must not produce a second event"
        assert subject.late == 1
        assert metrics.late_arrivals.value(camera="cam0", stage="detect") == 1


class TestSealingIsTheNormalPath:
    """The worker says when a frame is finished; the timeout is only the safety net."""

    def test_a_sealed_frame_with_everything_delivered_is_complete(self):
        subject, reported = collector()
        subject.open(make_state("cam0", 0), expected=("detect", "crop"))
        subject.deliver(("cam0", 0), "detect")
        subject.deliver(("cam0", 0), "crop")
        subject.seal(("cam0", 0))

        (result,) = reported
        assert result.reason == COMPLETE
        assert not result.is_partial
        assert subject.complete == 1

    def test_a_sealed_frame_with_a_stage_missing_is_partial_immediately(self):
        """Not 1500 ms later: a fast failure must not become a slow one."""
        subject, reported = collector()
        subject.open(make_state("cam0", 0), expected=("detect", "ship_embedder"))
        subject.deliver(("cam0", 0), "detect")
        subject.seal(("cam0", 0))

        (result,) = reported
        assert result.reason == INCOMPLETE
        assert result.missing == ("ship_embedder",)

    def test_a_frame_is_reported_exactly_once(self):
        """Sealing twice, or sealing a swept frame, must not duplicate an event."""
        subject, reported = collector()
        subject.open(make_state("cam0", 0), expected=("detect",))
        subject.deliver(("cam0", 0), "detect")
        subject.seal(("cam0", 0))
        subject.seal(("cam0", 0))
        subject.sweep()

        assert len(reported) == 1
        assert subject.reported == 1

    def test_completion_is_not_declared_while_the_plan_is_still_growing(self):
        """The bug the first implementation had: ``{detect}`` is momentarily satisfied while
        five stages are still to come, and emitting there published empty events."""
        subject, reported = collector()
        subject.open(make_state("cam0", 0))
        subject.expect(("cam0", 0), ("detect",))
        subject.deliver(("cam0", 0), "detect")

        assert reported == [], "emitted before the frame was sealed"

        subject.expect(("cam0", 0), ("crop", "person_embedder"))
        subject.deliver(("cam0", 0), "crop")
        subject.deliver(("cam0", 0), "person_embedder")
        subject.seal(("cam0", 0))

        assert len(reported) == 1
        assert reported[0].reason == COMPLETE


class TestShutdownPublishesWhatWasInFlight:
    """A shutdown that silently discards four hundred half-finished frames is not orderly."""

    def test_drain_emits_everything_pending(self):
        subject, reported = collector()
        for frame in range(5):
            subject.open(make_state("cam0", frame), expected=("detect",))

        assert subject.drain() == 5
        assert [r.reason for r in reported] == [SHUTDOWN] * 5
        assert subject.sizes() == {"pending": 0, "cameras": 0, "camera_entries": 0}


class TestEvictionsAreReportedButNotPublished:
    """Every frame leaves exactly one way, and the runner decides what reaches the sink."""

    def test_an_evicted_frame_is_reported_with_its_reason(self):
        subject, reported = collector(settings=ReassemblySettings(capacity=1))
        subject.open(make_state("cam0", 0), expected=("detect",))
        subject.open(make_state("cam0", 1), expected=("detect",))

        assert [r.reason for r in reported] == [EVICTED]
        assert reported[0].key == ("cam0", 0)


class TestThePolicyRegistry:
    """Adding a policy is a file and a decorator, and both shipped ones are reachable."""

    def test_both_policies_are_registered_with_aliases(self):
        assert set(EVICTION_POLICIES.names()) == {"greediest_camera", "oldest_frame"}
        assert EVICTION_POLICIES.canonical("fair") == "greediest_camera"
        assert EVICTION_POLICIES.canonical("drop_oldest") == "oldest_frame"

    def test_an_unknown_policy_fails_at_construction(self):
        from shipinfer.core.errors import ConfigurationError

        with pytest.raises(ConfigurationError, match="unknown reassembly eviction policy"):
            FrameCollector(lambda _r: None, settings=ReassemblySettings(eviction_policy="lru"))

    def test_an_empty_buffer_has_no_victim(self):
        policy = EVICTION_POLICIES.create("greediest_camera")
        subject, _ = collector()
        assert policy.choose(subject) is None
