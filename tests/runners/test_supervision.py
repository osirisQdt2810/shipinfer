"""Being told to stop, and blocking until it happens — on the contract, not on one runner.

``shipinfer run`` holds a :class:`~shipinfer.runners.base.Runner` and nothing else. It used to
ask ``getattr(built, "supervise", None)`` and ``getattr(built, "describe_plan", ...)``, which
is a probe rather than a contract: a fleet whose method was renamed would have been silently
downgraded to a runner that never watched its shards, and the fallback would have made that
read like a design decision. Both are methods on the ABC now, with defaults that are the
honest answer for a runner whose workers are threads in this process, and this is what pins
them.

Offline throughout: the whole point of putting them on the ABC is that neither needs hardware.
"""

from __future__ import annotations

import threading
import time

import pytest

from shipinfer.core.errors import ServerStateError
from shipinfer.core.request import ResponseFuture
from shipinfer.runners.base import Runner
from shipinfer.topology import ChainItem, ChainSpec, Topology

CHAIN = """
name: two_step
elements:
  decode: {impl: replay}
  output: {impl: none}
"""


class BareRunner(Runner):
    """The smallest thing that is a runner: it starts, it stops, it refuses items."""

    name = "bare"

    def __init__(self, topology: Topology) -> None:
        super().__init__(topology)
        self.stops: list[float] = []

    def _do_start(self) -> None:
        return None

    def _do_stop(self, timeout_s: float) -> None:
        self.stops.append(timeout_s)

    def _do_submit(self, item: ChainItem) -> ResponseFuture:
        raise ServerStateError("not this runner")


class FailingRunner(BareRunner):
    """One that gets half-way and raises, which is what the unwind exists for."""

    name = "failing"

    def _do_start(self) -> None:
        raise ServerStateError("the third element would not open")


@pytest.fixture()
def chain() -> Topology:
    return Topology.from_spec(ChainSpec.from_yaml(CHAIN))


class TestBeingAskedToStop:
    def test_supervise_returns_when_the_runner_is_asked_to_stop(self, chain) -> None:
        """The handler records; the supervising thread does the blocking work. That split is
        the whole reason `request_stop` exists — a handler that called `stop()` would block
        inside the lock it takes, and a second Ctrl-C would wait on itself."""
        runner = BareRunner(chain).start()
        threading.Timer(0.05, runner.request_stop).start()
        started = time.monotonic()

        runner.supervise(poll_s=5.0)  # returns on the event, not on the poll

        assert time.monotonic() - started < 2.0, "supervise slept out its poll interval"
        assert runner.stop_requested

    def test_request_stop_does_not_stop_anything(self, chain) -> None:
        """It records. The caller's `finally: stop()` owns the stopping, so "supervise
        returned" means the same thing on every runner."""
        runner = BareRunner(chain).start()

        runner.request_stop()

        assert runner.is_running and runner.stops == []

    def test_it_is_safe_before_start_and_survives_a_restart(self, chain) -> None:
        """A runner asked to stop and then started again supervises: the flag is cleared by
        `start`, or the second cycle would return the moment it began."""
        runner = BareRunner(chain)
        runner.request_stop()

        runner.start()

        assert not runner.stop_requested
        runner.request_stop()
        runner.supervise(poll_s=0.01)  # returns

    def test_supervise_returns_when_the_runner_stops_underneath_it(self, chain) -> None:
        runner = BareRunner(chain).start()
        threading.Timer(0.05, runner.stop).start()

        runner.supervise(poll_s=0.01)

        assert not runner.is_running

    def test_until_is_the_callers_own_reason_to_return(self, chain) -> None:
        runner = BareRunner(chain).start()
        calls: list[int] = []

        runner.supervise(poll_s=0.01, until=lambda: bool(calls.append(1)) or len(calls) > 2)

        assert len(calls) == 3

    def test_forward_signals_accepts_a_runner(self, chain) -> None:
        """`forward_signals` used to be typed on `Fleet` and had no production caller: the CLI
        rolled its own handlers. It takes anything with `request_stop()` now — which is every
        runner — and `shipinfer run` uses it."""
        import os
        import signal

        from shipinfer.launch import forward_signals

        runner = BareRunner(chain).start()
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            forward_signals(runner)
            os.kill(os.getpid(), signal.SIGTERM)
            runner.supervise(poll_s=0.01)  # returns instead of blocking

            assert runner.stop_requested
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


class TestTheDryRunAnswer:
    def test_a_runner_that_places_nothing_says_so(self, chain) -> None:
        """The honest answer, not a placeholder: there is one process and it is this one."""
        assert BareRunner(chain).describe_plan() == "no plan: one process"

    def test_the_fleet_overrides_it_with_the_plan_it_would_run(self, chain) -> None:
        from shipinfer.runners.fleet import FleetRunner

        planned = FleetRunner(chain, chain_yaml=CHAIN, shards=2, gpus=[0, 1]).describe_plan()

        assert "2 shard(s)" in planned and planned != "no plan: one process"


class TestTheSingleUnwind:
    def test_a_failed_start_unwinds_through_do_stop_with_the_hooks_budget(self, chain) -> None:
        """One unwind, owned by `Runner.start`. A subclass that caught its own partial start
        would run the release twice, and the second pass would see the state the first one
        cleared — which is how a fleet came to report zero abandoned camera threads after
        unwinding six of them."""
        runner = FailingRunner(chain)

        with pytest.raises(ServerStateError, match="third element"):
            runner.start()

        assert runner.stops == [0.0], "the ABC's unwind did not run exactly once"
        assert not runner.is_running

    def test_a_subclass_can_ask_for_a_budget(self, chain) -> None:
        """A runner whose release is a *conversation* — a `Stop` RPC to every shard — needs
        one; a runner that closes elements in this process does not."""

        class Patient(FailingRunner):
            def _unwind_timeout_s(self) -> float:
                return 12.0

        runner = Patient(chain)
        with pytest.raises(ServerStateError):
            runner.start()

        assert runner.stops == [12.0]
