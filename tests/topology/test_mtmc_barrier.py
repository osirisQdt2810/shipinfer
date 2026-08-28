"""The instant barrier: synchronisation, without a tracker anywhere near it.

:class:`~shipinfer.topology.elements.mtmc.InstantBarrier` is pure — it buckets opaque
payloads by capture instant and decides who waits, who closes and who is told to give up.
Every property this file asserts is a property of *that*, so nothing here imports
``shipvision``, constructs a track or needs a submodule: the association is a callback and
the payloads are strings.

That is the point rather than a convenience. The failures being guarded against are a frame
waiting for a camera that was removed an hour ago, an association that only ever closes on a
timeout because every worker is parked inside it, and a result scattered to the wrong frame —
none of which is about cross-camera geometry, and all of which are about this class.

**Windows here are milliseconds, not the 60 ms default.** A test that closes on the window
has to spend the window, so the ones that do use 30-60 ms and the ones that do not use a
window wide enough that it cannot fire by accident. Every test is bounded: a hang in this file
is a bug in the class under test, so `pytest.mark.timeout` is the assertion of last resort.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Any

import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.topology.elements.mtmc import (
    CLOSED_COMPLETE,
    CLOSED_WINDOW,
    DROPPED_EVICTED,
    DROPPED_EXPIRED,
    DROPPED_FAILED,
    DROPPED_SHUTDOWN,
    MISSED_DUPLICATE,
    MISSED_LATE,
    MISSED_WOULD_STARVE,
    InstantBarrier,
    InstantEntry,
)

pytestmark = [pytest.mark.timeout(30)]

#: Wide enough that no test closes on it by accident. Anything asserting the *window* names
#: its own, small one.
WIDE_S = 30.0


def by_camera(entries: Sequence[InstantEntry]) -> dict[tuple[str, int], str]:
    """An association that answers one keyed result per entry, from the payloads it was given.

    Keys are ``(camera_id, index)`` so a test can prove the scatter is keyed rather than
    positional: the whole group's answers arrive in one flattened map and a frame has to find
    its own in it.
    """
    return {
        (entry.camera_id, index): f"{entry.camera_id}:{entry.payload}"
        for entry in entries
        for index in (0,)
    }


def flat(entries: Sequence[InstantEntry]) -> dict[str, tuple[str, ...]]:
    """An association that reports exactly which cameras were in the instant it closed."""
    return {"cameras": tuple(entry.camera_id for entry in entries)}


def barrier(**kwargs: Any) -> InstantBarrier:
    kwargs.setdefault("sync_window_s", WIDE_S)
    kwargs.setdefault("workers", 4)
    return InstantBarrier(**kwargs)


class Submitter(threading.Thread):
    """One pipeline worker, in a thread, holding whatever the barrier answered it."""

    def __init__(
        self,
        target: InstantBarrier,
        camera_id: str,
        capture_s: float,
        payload: Any = "p",
        associate: Any = flat,
    ) -> None:
        super().__init__(daemon=True)
        self._barrier = target
        self._camera = camera_id
        self._capture = capture_s
        self._payload = payload
        self._associate = associate
        self.outcome: Any = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.outcome = self._barrier.submit(
                self._camera, self._capture, self._payload, associate=self._associate
            )
        except BaseException as exc:
            self.error = exc


def until(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


# -- construction ----------------------------------------------------------------------------


class TestBarrierConstruction:
    def test_a_zero_window_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="sync_window_s must be positive"):
            InstantBarrier(sync_window_s=0.0, workers=4)

    def test_zero_workers_is_refused_but_none_is_not(self) -> None:
        with pytest.raises(ConfigurationError, match="workers must be at least 1"):
            InstantBarrier(sync_window_s=1.0, workers=0)
        assert InstantBarrier(sync_window_s=1.0, workers=None).workers == 1

    def test_a_barrier_with_no_room_for_an_instant_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="max_instants must be at least 1"):
            InstantBarrier(sync_window_s=1.0, workers=4, max_instants=0)


# -- the bucket key --------------------------------------------------------------------------


class TestTheInstantKeyComesFromCaptureTime:
    """Which instant a frame belongs to is decided by when it was *taken*.

    Arrival order is the pipeline's business — fair lanes, N workers, spill — and has nothing
    to do with whether two frames show the same moment. A barrier keyed on arrival would put
    two cameras 40 ms apart in the queue into different instants while merging two frames of
    one camera that arrived together.
    """

    def test_two_captures_inside_one_window_share_a_key(self) -> None:
        # The windows are absolute -- `floor(t / window)` -- so the two captures have to fall
        # inside the same one, not merely be less than a window apart.
        held = barrier(sync_window_s=0.06)
        assert held.instant_key(100.02) == held.instant_key(100.05)

    def test_two_captures_either_side_of_a_boundary_do_not(self) -> None:
        held = barrier(sync_window_s=0.06)
        assert held.instant_key(100.02) != held.instant_key(100.09)

    def test_the_key_floors_rather_than_truncating(self) -> None:
        """``int()`` rounds toward zero, which would merge the two windows around zero."""
        held = barrier(sync_window_s=1.0)
        assert held.instant_key(-0.5) != held.instant_key(0.5)

    def test_two_cameras_of_one_instant_land_in_one_bucket(self) -> None:
        held = barrier(sync_window_s=0.06, workers=4)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        first = Submitter(held, "cam-a", 100.02)
        first.start()
        assert until(lambda: held.waiters == 1)
        second = held.submit("cam-b", 100.05, "p", associate=flat)
        first.join(5.0)

        assert second.reason == CLOSED_COMPLETE
        assert second.results == {"cameras": ("cam-a", "cam-b")}
        assert first.outcome.results == second.results
        assert first.outcome.instant == second.instant


# -- closing ---------------------------------------------------------------------------------


class TestABucketClosesOnTheLastLiveCamera:
    def test_the_closer_is_whichever_worker_completed_the_instant(self) -> None:
        held = barrier(workers=4)
        for camera in ("cam-a", "cam-b", "cam-c"):
            held.camera_added(camera)
        waiting = [Submitter(held, "cam-a", 10.0), Submitter(held, "cam-b", 10.0)]
        for worker in waiting:
            worker.start()
        assert until(lambda: held.waiters == 2)

        closed = held.submit("cam-c", 10.0, "p", associate=flat)
        for worker in waiting:
            worker.join(5.0)

        assert closed.reason == CLOSED_COMPLETE
        assert closed.results["cameras"] == ("cam-a", "cam-b", "cam-c")
        assert [w.outcome.results for w in waiting] == [closed.results] * 2
        assert held.open_instants == 0

    def test_a_group_of_one_closes_on_its_own_frame(self) -> None:
        """One camera in a group is a legitimate deployment, not a degenerate one."""
        held = barrier(workers=4)
        held.camera_added("cam-a")

        outcome = held.submit("cam-a", 10.0, "p", associate=flat)

        assert outcome.reason == CLOSED_COMPLETE
        assert outcome.results == {"cameras": ("cam-a",)}
        assert held.waiters == 0

    def test_the_scatter_is_keyed_and_a_frame_reads_its_own_entry(self) -> None:
        """The results cover the whole group; each frame finds itself by key, not position."""
        held = barrier(workers=4)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        first = Submitter(held, "cam-a", 10.0, payload="one", associate=by_camera)
        first.start()
        assert until(lambda: held.waiters == 1)
        second = held.submit("cam-b", 10.0, "two", associate=by_camera)
        first.join(5.0)

        assert second.results[("cam-a", 0)] == "cam-a:one"
        assert second.results[("cam-b", 0)] == "cam-b:two"
        assert first.outcome.results is second.results


class TestABucketClosesOnTheWindow:
    def test_a_missing_camera_costs_one_window_and_the_association_still_runs(self) -> None:
        held = barrier(sync_window_s=0.05, workers=4)
        held.camera_added("cam-a")
        held.camera_added("cam-b")

        started = time.monotonic()
        outcome = held.submit("cam-a", 10.0, "p", associate=flat)
        elapsed = time.monotonic() - started

        assert outcome.reason == CLOSED_WINDOW
        # The instant still associated -- over whoever did report. A camera outage is exactly
        # this, and refusing to associate the rest of the group would be the worse answer.
        assert outcome.results == {"cameras": ("cam-a",)}
        assert 0.04 <= elapsed < 5.0
        assert held.stats()[CLOSED_WINDOW] == 1

    def test_every_wait_is_bounded_by_the_window(self) -> None:
        """No caller is ever parked longer than one window, whatever else happens."""
        held = barrier(sync_window_s=0.03, workers=8)
        for camera in ("cam-a", "cam-b", "cam-c", "cam-d"):
            held.camera_added(camera)

        started = time.monotonic()
        workers = [Submitter(held, "cam-a", 10.0), Submitter(held, "cam-b", 11.0)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(5.0)
        elapsed = time.monotonic() - started

        assert not any(worker.is_alive() for worker in workers)
        assert all(worker.outcome.reason == CLOSED_WINDOW for worker in workers)
        assert elapsed < 2.0


# -- late ------------------------------------------------------------------------------------


class TestALateFrameIsCountedAndNeverRetroFitted:
    def test_a_frame_for_a_closed_instant_is_told_it_is_late(self) -> None:
        held = barrier(workers=4)
        held.camera_added("cam-a")
        assert held.submit("cam-a", 10.0, "p", associate=flat).reason == CLOSED_COMPLETE

        late = held.submit("cam-b", 10.0, "p", associate=flat)

        assert late.reason == MISSED_LATE
        assert late.results is None
        assert held.stats()[MISSED_LATE] == 1

    def test_a_late_frame_does_not_re_open_its_instant(self) -> None:
        """The refusal is the point: a second association would issue contradictory ids."""
        closes: list[tuple[str, ...]] = []

        def record(entries: Sequence[InstantEntry]) -> dict[str, Any]:
            closes.append(tuple(entry.camera_id for entry in entries))
            return {"n": len(entries)}

        held = barrier(workers=4)
        held.camera_added("cam-a")
        held.submit("cam-a", 10.0, "p", associate=record)
        held.submit("cam-b", 10.0, "p", associate=record)
        held.submit("cam-c", 10.0, "p", associate=record)

        assert closes == [("cam-a",)]
        assert held.open_instants == 0

    def test_two_frames_of_one_camera_in_one_instant_do_not_merge(self) -> None:
        """``FrameTrackCluster`` refuses a duplicate camera; the barrier never builds one."""
        held = barrier(workers=4)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        waiting = Submitter(held, "cam-a", 10.0)
        waiting.start()
        assert until(lambda: held.waiters == 1)

        again = held.submit("cam-a", 10.0, "p", associate=flat)

        assert again.reason == MISSED_DUPLICATE
        assert again.results is None
        held.submit("cam-b", 10.0, "p", associate=flat)
        waiting.join(5.0)
        assert waiting.outcome.results == {"cameras": ("cam-a", "cam-b")}


# -- bounded ---------------------------------------------------------------------------------


class TestTheBucketsAreBounded:
    def test_the_oldest_instant_is_evicted_and_counted(self) -> None:
        """A camera whose clock runs away must not be able to grow this map."""
        held = barrier(workers=1, max_instants=3)
        held.camera_added("cam-a")
        held.camera_added("cam-b")

        for instant in range(6):
            held.submit("cam-a", float(instant) * WIDE_S, "p", associate=flat)

        assert held.open_instants == 3
        assert held.stats()[DROPPED_EVICTED] == 3

    def test_eviction_takes_the_oldest_by_capture_time_not_by_arrival(self) -> None:
        held = barrier(workers=1, max_instants=2)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        for capture in (300.0, 100.0):
            held.submit("cam-a", capture * WIDE_S, "p", associate=flat)

        held.submit("cam-a", 200.0 * WIDE_S, "p", associate=flat)

        # 100 was the oldest instant even though it arrived second.
        assert held.instant_key(100.0 * WIDE_S) not in held._buckets
        assert sorted(held._buckets) == [
            held.instant_key(200.0 * WIDE_S),
            held.instant_key(300.0 * WIDE_S),
        ]

    def test_an_evicted_instant_releases_the_frames_waiting_on_it(self) -> None:
        # `workers=2` so the frame that does the evicting is itself refused a wait: this test
        # is about the waiter it displaced, and a second parked thread would only be a second
        # thing to join.
        held = barrier(workers=2, max_instants=1)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        waiting = Submitter(held, "cam-a", 10.0 * WIDE_S)
        waiting.start()
        assert until(lambda: held.waiters == 1)

        newer = held.submit("cam-a", 20.0 * WIDE_S, "p", associate=flat)
        assert newer.reason == MISSED_WOULD_STARVE
        waiting.join(5.0)

        assert not waiting.is_alive()
        assert waiting.outcome.reason == DROPPED_EVICTED
        assert waiting.outcome.results is None

    def test_the_late_key_memory_is_bounded_too(self) -> None:
        held = barrier(workers=1, max_instants=2)
        held.camera_added("cam-a")
        for instant in range(200):
            held.submit("cam-a", float(instant) * WIDE_S, "p", associate=flat)

        assert len(held._recent) <= held._recent_limit


# -- never starve ------------------------------------------------------------------------------


class TestBarrierNeverStarves:
    """The guard the whole design turns on: a barrier may never hold every pipeline worker.

    The walk is synchronous. A worker inside this class is a worker not draining its lane, so
    if all of them park here waiting for cameras whose frames are still *queued*, no instant
    can close on evidence — only on the timeout — and the shard has converted itself into a
    fixed latency with a stalled queue behind it.
    """

    @pytest.mark.parametrize("workers", [1, 2, 4, 8])
    def test_at_most_workers_minus_one_ever_wait(self, workers: int) -> None:
        held = barrier(workers=workers)
        for index in range(workers + 3):
            held.camera_added(f"cam-{index}")

        parked = [Submitter(held, f"cam-{index}", 10.0) for index in range(workers - 1)]
        for worker in parked:
            worker.start()
        assert until(lambda: held.waiters == workers - 1)

        extra = held.submit(f"cam-{workers}", 10.0, "p", associate=flat)

        assert held.waiters == workers - 1
        assert extra.reason == MISSED_WOULD_STARVE
        assert extra.results is None
        assert held.stats()[MISSED_WOULD_STARVE] == 1
        held.close_all()
        for worker in parked:
            worker.join(5.0)

    def test_a_single_worker_runner_never_waits_at_all(self) -> None:
        held = barrier(workers=1)
        held.camera_added("cam-a")
        held.camera_added("cam-b")

        started = time.monotonic()
        outcome = held.submit("cam-a", 10.0, "p", associate=flat)

        assert time.monotonic() - started < 1.0
        assert outcome.reason == MISSED_WOULD_STARVE

    def test_an_unknown_worker_count_is_treated_as_one(self) -> None:
        """``ElementContext.workers is None`` means "the runner did not say" -- never guess."""
        held = barrier(workers=None)
        held.camera_added("cam-a")
        held.camera_added("cam-b")

        assert held.submit("cam-a", 10.0, "p", associate=flat).reason == MISSED_WOULD_STARVE

    def test_a_starved_frame_still_contributes_its_tracks_to_the_instant(self) -> None:
        """It gives up its *answer*, not its evidence: it is in the group at that moment."""
        held = barrier(workers=2)
        for camera in ("cam-a", "cam-b", "cam-c"):
            held.camera_added(camera)
        parked = Submitter(held, "cam-a", 10.0)
        parked.start()
        assert until(lambda: held.waiters == 1)

        assert held.submit("cam-b", 10.0, "p", associate=flat).reason == MISSED_WOULD_STARVE
        held.submit("cam-c", 10.0, "p", associate=flat)
        parked.join(5.0)

        assert parked.outcome.results == {"cameras": ("cam-a", "cam-b", "cam-c")}

    def test_a_worker_that_gave_up_frees_the_slot_for_the_next_one(self) -> None:
        held = barrier(sync_window_s=0.05, workers=2)
        held.camera_added("cam-a")
        held.camera_added("cam-b")

        first = held.submit("cam-a", 10.0, "p", associate=flat)
        second = held.submit("cam-a", 20.0, "p", associate=flat)

        assert first.reason == CLOSED_WINDOW
        assert second.reason == CLOSED_WINDOW
        assert held.waiters == 0


# -- the camera set ----------------------------------------------------------------------------


class TestTheLiveCameraSetDrivesCompleteness:
    def test_an_added_camera_is_waited_for(self) -> None:
        held = barrier(sync_window_s=0.05, workers=4)
        held.camera_added("cam-a")
        assert held.submit("cam-a", 10.0, "p", associate=flat).reason == CLOSED_COMPLETE

        held.camera_added("cam-b")
        assert held.submit("cam-a", 20.0, "p", associate=flat).reason == CLOSED_WINDOW

    def test_a_removed_camera_no_longer_holds_an_instant_open(self) -> None:
        """The half that matters: the *open* bucket stops waiting too, not just the next one."""
        held = barrier(sync_window_s=30.0, workers=4)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        waiting = Submitter(held, "cam-a", 10.0)
        waiting.start()
        assert until(lambda: held.waiters == 1)

        started = time.monotonic()
        held.drop_camera("cam-b")
        waiting.join(5.0)

        assert not waiting.is_alive(), "the open instant was still waiting for a gone camera"
        assert time.monotonic() - started < 5.0
        assert waiting.outcome.reason == CLOSED_COMPLETE
        assert waiting.outcome.results == {"cameras": ("cam-a",)}

    def test_dropping_a_camera_returns_without_running_an_association(self) -> None:
        """It runs under the runner's lifecycle lock; a tracker call there stalls every RPC."""
        ran: list[int] = []

        def record(entries: Sequence[InstantEntry]) -> dict[str, int]:
            ran.append(len(entries))
            time.sleep(0.05)
            return {"n": len(entries)}

        held = barrier(sync_window_s=30.0, workers=4)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        waiting = Submitter(held, "cam-a", 10.0, associate=record)
        waiting.start()
        assert until(lambda: held.waiters == 1)

        started = time.monotonic()
        held.drop_camera("cam-b")
        elapsed = time.monotonic() - started
        waiting.join(5.0)

        assert elapsed < 0.04, "drop_camera ran the association on the lifecycle thread"
        assert ran == [1]

    def test_dropping_a_camera_with_nobody_waiting_leaves_the_bucket_alone(self) -> None:
        held = barrier(sync_window_s=30.0, workers=1, max_instants=4)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        assert held.submit("cam-a", 10.0, "p", associate=flat).reason == MISSED_WOULD_STARVE

        held.drop_camera("cam-b")

        assert held.open_instants == 1
        assert held.stats().get(CLOSED_COMPLETE, 0) == 0

    def test_a_barrier_nobody_announced_learns_its_group_from_traffic(self) -> None:
        """A runner that does not drive the hooks costs one instant, not per-camera MTMC."""
        held = barrier(sync_window_s=0.05, workers=4)

        first = held.submit("cam-a", 10.0, "p", associate=flat)
        second = held.submit("cam-b", 20.0, "p", associate=flat)

        assert first.reason == CLOSED_COMPLETE  # nothing known yet: the group is one camera
        assert second.reason == CLOSED_WINDOW  # cam-a has been seen, so cam-b waits for it
        assert held.live == frozenset({"cam-a", "cam-b"})

    def test_the_first_announcement_latches_the_hooks_on_for_good(self) -> None:
        """Otherwise removing the last camera would resurrect the set this hook clears."""
        held = barrier(sync_window_s=0.05, workers=4)
        held.submit("cam-a", 10.0, "p", associate=flat)
        held.camera_added("cam-b")
        held.drop_camera("cam-b")

        assert held.live == frozenset()
        assert held.submit("cam-c", 20.0, "p", associate=flat).reason == CLOSED_COMPLETE


# -- retirement and shutdown --------------------------------------------------------------------


class TestAnAbandonedInstantIsRetired:
    def test_a_bucket_past_its_deadline_with_no_waiters_is_discarded(self) -> None:
        held = barrier(sync_window_s=0.02, workers=1, max_instants=8)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        assert held.submit("cam-a", 10.0, "p", associate=flat).reason == MISSED_WOULD_STARVE
        assert held.open_instants == 1
        time.sleep(0.05)

        held.submit("cam-a", 20.0, "p", associate=flat)

        assert held.stats()[DROPPED_EXPIRED] == 1
        assert held.open_instants == 1


class TestCloseAllResolvesEveryWaiter:
    def test_a_shutdown_releases_the_parked_workers_at_once(self) -> None:
        held = barrier(sync_window_s=30.0, workers=4)
        for camera in ("cam-a", "cam-b", "cam-c"):
            held.camera_added(camera)
        # Two *different* instants: `floor(t / 30)` puts 10 and 100 in separate buckets, and
        # the point of this test is that a shutdown resolves every open one.
        parked = [Submitter(held, "cam-a", 10.0), Submitter(held, "cam-b", 100.0)]
        for worker in parked:
            worker.start()
        assert until(lambda: held.waiters == 2)

        started = time.monotonic()
        assert held.close_all() == 2
        for worker in parked:
            worker.join(5.0)

        assert not any(worker.is_alive() for worker in parked)
        assert time.monotonic() - started < 5.0
        assert {worker.outcome.reason for worker in parked} == {DROPPED_SHUTDOWN}
        assert all(worker.outcome.results is None for worker in parked)

    def test_a_submit_after_close_all_is_refused_rather_than_parked(self) -> None:
        held = barrier(sync_window_s=30.0, workers=4)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        held.close_all()

        outcome = held.submit("cam-a", 10.0, "p", associate=flat)

        assert outcome.reason == DROPPED_SHUTDOWN
        assert outcome.results is None

    def test_close_all_is_idempotent(self) -> None:
        held = barrier(workers=4)
        held.camera_added("cam-a")
        assert held.close_all() == 0
        assert held.close_all() == 0


class TestAFailedAssociationDoesNotStrandAWaiter:
    def test_the_waiters_are_released_and_the_closer_gets_the_exception(self) -> None:
        def explode(entries: Sequence[InstantEntry]) -> dict[str, Any]:
            raise RuntimeError("the tracker refused this cluster")

        held = barrier(sync_window_s=30.0, workers=4)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        waiting = Submitter(held, "cam-a", 10.0, associate=explode)
        waiting.start()
        assert until(lambda: held.waiters == 1)

        with pytest.raises(RuntimeError, match="refused this cluster"):
            held.submit("cam-b", 10.0, "p", associate=explode)
        waiting.join(5.0)

        assert not waiting.is_alive()
        assert waiting.outcome.reason == DROPPED_FAILED
        assert waiting.outcome.results is None
        assert held.stats()[DROPPED_FAILED] == 1


# -- the observer -------------------------------------------------------------------------------


class TestTheInstantObserver:
    def test_one_event_per_instant_and_not_one_per_frame(self) -> None:
        """A closed instant resolves many frames; counting it per frame reports the wrong number."""
        events: list[str] = []
        held = InstantBarrier(sync_window_s=30.0, workers=4, on_event=events.append)
        for camera in ("cam-a", "cam-b", "cam-c"):
            held.camera_added(camera)
        parked = [Submitter(held, "cam-a", 10.0), Submitter(held, "cam-b", 10.0)]
        for worker in parked:
            worker.start()
        assert until(lambda: held.waiters == 2)

        held.submit("cam-c", 10.0, "p", associate=flat)
        for worker in parked:
            worker.join(5.0)

        assert events == [CLOSED_COMPLETE]

    def test_frame_level_misses_are_not_instant_events(self) -> None:
        events: list[str] = []
        held = InstantBarrier(sync_window_s=30.0, workers=1, on_event=events.append)
        held.camera_added("cam-a")
        held.camera_added("cam-b")

        held.submit("cam-a", 10.0, "p", associate=flat)

        assert events == []
        assert held.stats()[MISSED_WOULD_STARVE] == 1


# -- under threads --------------------------------------------------------------------------------


class TestUnderThreads:
    def test_every_frame_of_a_group_gets_the_same_answer_or_an_honest_gap(self) -> None:
        """The property, hammered: nobody hangs, nobody invents, everybody is accounted for."""
        cameras = [f"cam-{index}" for index in range(4)]
        held = barrier(sync_window_s=0.02, workers=8, max_instants=4)
        for camera in cameras:
            held.camera_added(camera)

        answers: dict[str, list[Any]] = {camera: [] for camera in cameras}
        lock = threading.Lock()

        def drive(camera: str) -> None:
            for instant in range(20):
                outcome = held.submit(camera, instant * 0.02, camera, associate=flat)
                with lock:
                    answers[camera].append(outcome)

        threads = [threading.Thread(target=drive, args=(camera,)) for camera in cameras]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(20.0)

        assert not any(thread.is_alive() for thread in threads)
        for camera, outcomes in answers.items():
            assert len(outcomes) == 20
            for outcome in outcomes:
                if outcome.results is None:
                    continue
                # An associated frame is always inside the instant it was told about.
                assert camera in outcome.results["cameras"]
        assert held.waiters == 0
