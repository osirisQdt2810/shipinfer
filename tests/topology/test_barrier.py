"""The instant barrier: synchronisation, without a tracker anywhere near it.

:class:`~shipinfer.topology.barrier.InstantBarrier` is pure — it buckets opaque payloads by
capture instant and decides who waits, who closes and who is told to give up. Every property
this file asserts is a property of *that*, so nothing here imports ``shipvision``, constructs
a track or needs a submodule: the association is a callback and the payloads are strings.

That is the point rather than a convenience. The failures being guarded against are a frame
waiting for a camera that was removed an hour ago, an association that only ever closes on a
timeout because every worker is parked inside it, a camera colliding with itself once every
few frames, and a result scattered to the wrong frame — none of which is about cross-camera
geometry, and all of which are about this class.

``TestAGenlockedGridGetsEveryFrameAnswered`` is the one that would have caught the bucket key
this class used to have, and it is written the way the deployment runs: one thread per camera,
a :class:`threading.Barrier` so every camera submits its k-th frame together at 20 fps, and
the assertion that **every** frame comes back with an answer.

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

from shipinfer.core.errors import ConfigurationError, ServerStateError
from shipinfer.topology.barrier import (
    CLOSED_ADVANCED,
    CLOSED_COMPLETE,
    CLOSED_WINDOW,
    DEFAULT_SYNC_WINDOW_MS,
    DROPPED_EVICTED,
    DROPPED_EXPIRED,
    DROPPED_FAILED,
    DROPPED_SHUTDOWN,
    MISSED_DUPLICATE,
    MISSED_LATE,
    MISSED_WOULD_STARVE,
    InstantBarrier,
    InstantEntry,
    WaiterBudget,
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


# -- the anchored instant --------------------------------------------------------------------


class TestTheInstantIsAnchoredAndComesFromCaptureTime:
    """Which instant a frame belongs to is decided by when it was *taken*, relative to the
    frame that opened the instant — never by an absolute grid.

    Arrival order is the pipeline's business — fair lanes, N workers, spill — and has nothing
    to do with whether two frames show the same moment. A barrier keyed on arrival would put
    two cameras 40 ms apart in the queue into different instants while merging two frames of
    one camera that arrived together.

    An **absolute** ``floor(capture / window)`` grid has no correct setting, which is why
    these tests are about spans and not keys. A window wider than the frame period makes a
    camera collide with itself every ``window / period`` frames; a window narrower than it
    stops free-running cameras from ever landing together. The anchor removes the first
    constraint entirely: the frame that opens the instant sets the span, and a camera already
    in it can only ever *close* it.
    """

    @pytest.mark.parametrize("base", [100.0, 100.059, 100.999999])
    def test_two_captures_inside_one_window_are_one_instant_wherever_they_sit(
        self, base: float
    ) -> None:
        """No boundary to fall either side of: only the spread matters. Each ``base`` is a
        capture that an absolute ``floor(t / 0.06)`` grid would have split from ``base+30ms``
        or merged with something else — here every one of them is one instant."""
        held = barrier(sync_window_s=0.06, workers=4)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        first = Submitter(held, "cam-a", base)
        first.start()
        assert until(lambda: held.waiters == 1)
        second = held.submit("cam-b", base + 0.03, "p", associate=flat)
        first.join(5.0)

        assert second.reason == CLOSED_COMPLETE
        assert second.results == {"cameras": ("cam-a", "cam-b")}
        assert first.outcome.results == second.results
        assert first.outcome.instant == second.instant

    def test_two_captures_more_than_a_window_apart_are_two_instants(self) -> None:
        held = barrier(sync_window_s=0.05, workers=4)
        held.camera_added("cam-a")
        held.camera_added("cam-b")

        first = held.submit("cam-a", 100.0, "p", associate=flat)
        second = held.submit("cam-b", 100.09, "p", associate=flat)

        assert first.reason == CLOSED_WINDOW and second.reason == CLOSED_WINDOW
        assert first.instant != second.instant
        assert first.results == {"cameras": ("cam-a",)}
        assert second.results == {"cameras": ("cam-b",)}

    def test_captures_either_side_of_zero_are_one_instant_when_they_are_close(self) -> None:
        """The old key floored a ratio; an anchored span has no origin to be wrong about."""
        held = barrier(sync_window_s=1.0, workers=4)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        first = Submitter(held, "cam-a", -0.4)
        first.start()
        assert until(lambda: held.waiters == 1)

        second = held.submit("cam-b", 0.4, "p", associate=flat)
        first.join(5.0)

        assert second.reason == CLOSED_COMPLETE
        assert second.results == {"cameras": ("cam-a", "cam-b")}

    def test_a_camera_never_collides_with_itself_at_any_window(self) -> None:
        """The failure the anchor exists to delete: on an absolute grid a 20 fps camera hit
        its own bucket once every ``window / period`` frames, and every such frame was denied
        an answer. Here the second frame *closes* the instant its predecessor was in."""
        held = barrier(sync_window_s=DEFAULT_SYNC_WINDOW_MS / 1e3, workers=4)
        held.camera_added("cam-a")

        outcomes = [
            held.submit("cam-a", 100.0 + k * 0.05, "p", associate=flat) for k in range(40)
        ]

        assert [o.reason for o in outcomes] == [CLOSED_COMPLETE] * 40
        assert MISSED_LATE not in held.frame_stats()
        assert MISSED_DUPLICATE not in held.frame_stats()

    def test_a_second_frame_from_a_camera_in_the_bucket_closes_it(self) -> None:
        """It is the only evidence available that the instant is over: that camera has moved
        on, so nothing more will arrive from it for the instant it was in."""
        # `workers=2` so the frame that does the sealing is itself refused a wait: this test
        # is about the waiter it released, and a second parked thread would only be a second
        # thing to join.
        held = barrier(sync_window_s=30.0, workers=2)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        waiting = Submitter(held, "cam-a", 10.0)
        waiting.start()
        assert until(lambda: held.waiters == 1)

        started = time.monotonic()
        nxt = held.submit("cam-a", 10.5, "p", associate=flat)
        waiting.join(5.0)

        assert time.monotonic() - started < 5.0, "the sealed instant waited out its window"
        assert waiting.outcome.reason == CLOSED_ADVANCED
        assert waiting.outcome.results == {"cameras": ("cam-a",)}
        assert nxt.instant != waiting.outcome.instant, "the next frame opened the next instant"
        assert held.instant_stats()[CLOSED_ADVANCED] == 1

    def test_a_repeat_of_a_capture_already_in_the_bucket_is_a_duplicate(self) -> None:
        """A *later* capture is the next instant; the same one is a mis-wired source."""
        held = barrier(sync_window_s=30.0, workers=4)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        waiting = Submitter(held, "cam-a", 10.0)
        waiting.start()
        assert until(lambda: held.waiters == 1)

        again = held.submit("cam-a", 10.0, "p", associate=flat)

        assert again.reason == MISSED_DUPLICATE
        assert held.instant_stats().get(CLOSED_ADVANCED, 0) == 0
        held.submit("cam-b", 10.0, "p", associate=flat)
        waiting.join(5.0)
        assert waiting.outcome.results == {"cameras": ("cam-a", "cam-b")}

    def test_another_camera_running_ahead_does_not_make_the_next_frame_a_duplicate(
        self,
    ) -> None:
        """The bucket's ``last`` is the maximum over the whole *group*, so testing a repeat
        against it refused a camera's genuinely next frame whenever a second camera had
        already pushed the span past it: cam-a at 100.000, cam-b at 100.055, and then cam-a's
        own next frame at 100.050 — 50 ms later, which is what 20 fps looks like. It was
        counted ``duplicate``, got no answer, and left the instant to sit out its whole window
        for a camera that had moved on. The test is per camera, against what that camera
        itself put in, which is also what :data:`MISSED_DUPLICATE`'s docstring has always
        said."""
        held = barrier(sync_window_s=DEFAULT_SYNC_WINDOW_MS / 1e3, workers=8)
        for camera in ("cam-a", "cam-b", "cam-c"):
            held.camera_added(camera)
        opener = Submitter(held, "cam-a", 100.000)
        opener.start()
        assert until(lambda: held.waiters == 1)
        ahead = Submitter(held, "cam-b", 100.055)
        ahead.start()
        assert until(lambda: held.waiters == 2)
        assert held.open_spans == ((1, 100.000, 100.055),), "one instant, spanning both"

        nxt = held.submit("cam-a", 100.050, "p", associate=flat)

        opener.join(5.0)
        ahead.join(5.0)
        assert held.frame_stats().get(MISSED_DUPLICATE, 0) == 0, "cam-a's next frame refused"
        assert held.instant_stats()[CLOSED_ADVANCED] == 1, "the instant was sealed, not held"
        assert opener.outcome.reason == CLOSED_ADVANCED
        assert opener.outcome.results == {"cameras": ("cam-a", "cam-b")}
        assert ahead.outcome.results == opener.outcome.results
        assert nxt.instant != opener.outcome.instant, "the next frame is the next instant"
        assert nxt.results == {"cameras": ("cam-a",)}, "and it is in that instant alone"


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
        assert held.instant_stats()[CLOSED_WINDOW] == 1

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
        assert held.frame_stats()[MISSED_LATE] == 1

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
        assert held.instant_stats()[DROPPED_EVICTED] == 3

    def test_eviction_takes_the_instant_that_has_been_open_longest(self) -> None:
        """By open order, which is deadline order — every deadline is one window after its
        bucket opened. Evicting by *capture* time instead would let one camera with a stale
        clock push out the instant the rest of the group is actively filling."""
        held = barrier(workers=1, max_instants=2)
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        for capture in (300.0, 100.0):
            held.submit("cam-a", capture * WIDE_S, "p", associate=flat)

        held.submit("cam-a", 200.0 * WIDE_S, "p", associate=flat)

        # 300 arrived first, so it is the one that has been open longest.
        assert [span[1] for span in held.open_spans] == [100.0 * WIDE_S, 200.0 * WIDE_S]
        assert held.instant_stats()[DROPPED_EVICTED] == 1

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
        assert held.frame_stats()[MISSED_WOULD_STARVE] == 1
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
        assert held.instant_stats().get(CLOSED_COMPLETE, 0) == 0

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

        assert held.instant_stats()[DROPPED_EXPIRED] == 1
        assert held.open_instants == 1

    def test_a_sealed_bucket_nobody_waited_on_keeps_the_reason_it_was_sealed_with(
        self,
    ) -> None:
        """``advanced`` is the share an operator reads before touching ``sync_window_ms``,
        and counting a sealed-but-unwaited bucket ``expired`` made it structurally zero in
        the one regime where it matters: a shard with too few workers, where nobody ever
        parks and so nobody is ever there to close the instant a camera sealed."""
        held = barrier(sync_window_s=0.03, workers=1)  # no permits: nobody ever waits
        held.camera_added("cam-a")
        held.camera_added("cam-b")
        assert held.submit("cam-a", 100.00, "p", associate=flat).reason == MISSED_WOULD_STARVE
        # cam-a's next frame is inside the window and later than its own, so it seals instant
        # 1 -- `advanced` -- and opens instant 2 with itself in it.
        assert held.submit("cam-a", 100.02, "p", associate=flat).reason == MISSED_WOULD_STARVE
        time.sleep(0.06)

        held.submit("cam-b", 100.20, "p", associate=flat)

        assert held.instant_stats() == {CLOSED_ADVANCED: 1, DROPPED_EXPIRED: 1}


class TestCloseAllResolvesEveryWaiter:
    def test_a_shutdown_releases_the_parked_workers_at_once(self) -> None:
        held = barrier(sync_window_s=30.0, workers=4)
        for camera in ("cam-a", "cam-b", "cam-c"):
            held.camera_added(camera)
        # Two *different* instants: 10 and 100 are 90 s apart against a 30 s window, so
        # neither can join the other's span, and the point of this test is that a shutdown
        # resolves every open one.
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
        assert held.instant_stats()[DROPPED_FAILED] == 1


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
        assert held.frame_stats()[MISSED_WOULD_STARVE] == 1


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


# -- the property the whole class exists for -----------------------------------------------


def run_grid(
    cameras: int,
    *,
    frames: int = 24,
    window_ms: float = DEFAULT_SYNC_WINDOW_MS,
    fps: float = 20.0,
    stagger: bool = False,
) -> dict[str, int]:
    """One thread per camera, all submitting their k-th frame together. Tally by reason.

    The shape of the deployment, in miniature: ``cameras`` pipeline workers each carrying one
    camera's frame, a :class:`threading.Barrier` standing in for the genlock so that the k-th
    frame of every camera is offered at the same moment, and capture timestamps a real frame
    period apart. ``stagger`` spreads the group's captures evenly across one frame period,
    which is what free-running cameras look like.
    """
    period = 1.0 / fps
    held = InstantBarrier(sync_window_s=window_ms / 1e3, workers=cameras + 1)
    for index in range(cameras):
        held.camera_added(f"cam-{index}")
    offsets = [index * period / cameras if stagger else 0.0 for index in range(cameras)]
    gate = threading.Barrier(cameras)
    tally: dict[str, int] = {}
    lock = threading.Lock()

    def drive(index: int) -> None:
        for frame in range(frames):
            gate.wait()
            outcome = held.submit(
                f"cam-{index}",
                1_000_000.0 + frame * period + offsets[index],
                "p",
                associate=flat,
            )
            with lock:
                tally[outcome.reason] = tally.get(outcome.reason, 0) + 1

    threads = [threading.Thread(target=drive, args=(i,), daemon=True) for i in range(cameras)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20.0)
    assert not any(thread.is_alive() for thread in threads), tally
    assert sum(tally.values()) == cameras * frames
    return tally


class TestAGenlockedGridGetsEveryFrameAnswered:
    """Every frame of a healthy group comes back with an answer. No exceptions.

    This is the test the absolute ``floor(capture / window)`` grid could not pass, and it is
    written as the deployment runs rather than as a unit: real threads, one per camera, every
    camera offering its k-th frame at the same moment, capture times a real 20 fps period
    apart. On the grid, a 60 ms window against a 50 ms period made each camera land in a
    bucket it had already reported once every six frames, so **83.3%** of frames came back
    ``late`` — with the cameras perfectly synchronised and nothing else wrong. The anchored
    instant makes a camera's second frame *close* the instant its first was in, so the
    collision cannot happen at any window.

    Both the default window and the sizing (`CLAUDE.md`: 50 cameras, 20 fps, groups of a
    berth or a quay) are pinned here, because "the default is fine" was the claim that was
    wrong.
    """

    @pytest.mark.parametrize("cameras", [2, 8])
    def test_every_frame_gets_an_answer_at_the_default_window(self, cameras: int) -> None:
        tally = run_grid(cameras)

        answered = tally.get(CLOSED_COMPLETE, 0) + tally.get(CLOSED_WINDOW, 0)
        assert answered == sum(tally.values()), tally
        assert tally.get(MISSED_LATE, 0) == 0, "a camera collided with itself"
        assert tally.get(MISSED_DUPLICATE, 0) == 0

    @pytest.mark.parametrize("cameras", [2, 8])
    def test_free_running_cameras_staggered_across_a_period_still_land_together(
        self, cameras: int
    ) -> None:
        """The other half of the constraint an absolute grid could not satisfy: a window
        *narrow* enough to stop self-collision is too narrow to hold a group whose captures
        are spread across a frame period. The anchor holds them because the first arrival
        opens the window rather than the clock."""
        tally = run_grid(cameras, stagger=True)

        assert tally.get(CLOSED_COMPLETE, 0) == sum(tally.values()), tally

    def test_a_window_narrower_than_the_frame_period_is_also_fine(self) -> None:
        """The default is not load-bearing for the self-collision property — nothing is."""
        tally = run_grid(8, window_ms=25.0)

        answered = tally.get(CLOSED_COMPLETE, 0) + tally.get(CLOSED_WINDOW, 0)
        assert answered == sum(tally.values()), tally


# -- the waiter budget -------------------------------------------------------------------


class TestTheWaiterBudgetIsPerProcessAndNotPerBarrier:
    """Two ``mtmc`` slots in one chain must not be able to park every worker between them.

    Two independent camera groups is a supported configuration — the chain loader takes an
    explicit ``kind:``, and the element's own metric is labelled by element precisely because
    two of them can exist. With a per-barrier count barrier A admits ``workers - 1`` waiters
    and barrier B, which has seen none, admits the last worker. Every pipeline worker is then
    parked, neither barrier can close on evidence, and the shard has converted itself into a
    fixed window of latency with a stalled queue behind it — bounded rather than a hang, and
    therefore exactly the stall dressed as a wait the guard exists to prevent.
    """

    def test_a_budget_refuses_a_negative_size(self) -> None:
        with pytest.raises(ConfigurationError, match="cannot have -1 permits"):
            WaiterBudget(-1)

    def test_a_budget_of_zero_never_lets_anyone_wait(self) -> None:
        budget = WaiterBudget(0)
        assert budget.acquire() is False
        assert budget.held == 0

    def test_releasing_more_than_was_acquired_is_refused(self) -> None:
        """A silent decrement past zero would hand out permits that do not exist."""
        budget = WaiterBudget(1)
        assert budget.acquire() is True
        budget.release()
        with pytest.raises(ServerStateError, match="released more times"):
            budget.release()

    def test_a_budget_hands_out_its_permits_and_no_more_and_counts_them(self) -> None:
        """The bound is :class:`threading.BoundedSemaphore`'s; ``held`` is not, which is why
        there is a wrapper at all — the health report and every test of the never-starve
        invariant read it, and ``Semaphore`` has no public way to say it."""
        budget = WaiterBudget(2)

        assert [budget.acquire() for _ in range(3)] == [True, True, False]

        assert budget.held == 2
        budget.release()
        assert budget.held == 1

    def test_a_supplied_budget_wins_over_an_unknown_worker_count(self) -> None:
        """``workers`` only ever sizes a *private* budget, so "the runner did not say" does
        not veto a budget that was said: a barrier built this way waits on the budget's
        permits, and a docstring claiming it never waits would send an operator looking for
        the wrong symptom."""
        budget = WaiterBudget(3)

        held = InstantBarrier(sync_window_s=1.0, workers=None, budget=budget)

        assert held.budget is budget
        assert held.budget.permits == 3
        assert held.workers == 1, "and `workers` reports the fallback, governing nothing"

    def test_a_barrier_given_no_budget_sizes_a_private_one_from_workers(self) -> None:
        assert barrier(workers=4).budget.permits == 3
        assert barrier(workers=1).budget.permits == 0
        assert barrier(workers=None).budget.permits == 0

    @pytest.mark.parametrize("workers", [2, 4, 8])
    def test_two_barriers_sharing_a_budget_never_park_every_worker(self, workers: int) -> None:
        budget = WaiterBudget(workers - 1)
        first = barrier(workers=workers, budget=budget)
        second = barrier(workers=workers, budget=budget)
        for held in (first, second):
            for index in range(workers + 3):
                held.camera_added(f"cam-{index}")

        # Fill the budget from the *first* barrier, then offer the last worker to the second.
        parked = [Submitter(first, f"cam-{index}", 10.0) for index in range(workers - 1)]
        for worker in parked:
            worker.start()
        assert until(lambda: budget.held == workers - 1)

        last = second.submit("cam-0", 10.0, "p", associate=flat)

        assert last.reason == MISSED_WOULD_STARVE, "both barriers together parked every worker"
        assert first.waiters + second.waiters == workers - 1
        first.close_all()
        second.close_all()
        for worker in parked:
            worker.join(5.0)
        assert budget.held == 0, "a released waiter kept its permit"

    def test_the_permit_comes_back_when_the_waiter_leaves(self) -> None:
        budget = WaiterBudget(1)
        held = barrier(sync_window_s=0.03, workers=4, budget=budget)
        held.camera_added("cam-a")
        held.camera_added("cam-b")

        assert held.submit("cam-a", 10.0, "p", associate=flat).reason == CLOSED_WINDOW
        assert budget.held == 0
        assert held.submit("cam-a", 20.0, "p", associate=flat).reason == CLOSED_WINDOW
