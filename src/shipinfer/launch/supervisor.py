"""One OS process per shard — the half of the throughput answer the C++ plane did not give.

WHY A LAUNCHER AND NOT MORE THREADS
-----------------------------------
Measured on the dev box, a saturated run of the Python plane used **390-534% CPU**: five cores
of forty-eight, with every model queue empty and only the pipeline queue growing. Four
candidates were eliminated before this one — not the GPUs (the detector retired 119.9 of 120
offered), not the worker pool (24 / 96 / 192 workers gave 87.6 / 81.4 / 85.0 img/s, an eightfold
range for under 8% and not monotonic), not the reassembly lock (98.9% of its hold removed, no
change), not the load generator (it delivered 100% of what it offered).

The C++ data plane answered the first half of that: it took the per-frame work out of the
interpreter and the pipeline from 77 to 390 img/s. It did not answer the second half, which is
that 390 is still **one process**. A process is a wall, and this is the tool for putting more
than one of them on a forty-eight-core box.

That is also the shape vLLM settled on for the same problem: its ``MultiprocExecutor`` spawns
``context.Process`` per GPU worker and talks over ZMQ, with threads kept for auxiliary work.
Forty-one of its files touch ``multiprocessing``; none of its twenty-one ``threading.Thread``
uses is on the model-execution path.

WHY SUBPROCESSES RATHER THAN ``multiprocessing``
------------------------------------------------
Because of one line: ``CUDA_VISIBLE_DEVICES`` has to be set **before anything imports torch**,
and it has to differ per child. With ``multiprocessing``'s spawn context the child inherits the
parent's ``os.environ``, so every child would see the same value; setting it inside the child
means racing whatever imported torch first, and in this codebase that is a module-scope import
two packages deep. A ``Popen`` with an explicit ``env=`` cannot lose that race — the variable is
in place before the interpreter starts, let alone before it imports anything.

It also means a shard is an ordinary process. It can be inspected with ``ps``, killed on its
own, and profiled without the profiler seeing three other shards' kernels — and the same
arrangement is what a container orchestrator would do anyway, one shard per container, so
nothing here has to be undone to get there.

WHAT THIS DOES NOT DO
---------------------
It does not aggregate. Each shard serves its own cameras end to end and hands its own tracklets
downstream, because cameras are independent of each other all the way to MTMC — which is a
separate plane consuming tracklets rather than frames. A launcher that collected results would
be reintroducing the single process this exists to escape.

AND IT DOES NOT TALK TO ITS CHILDREN — YET
------------------------------------------
This is the spawn-and-supervise half of arch.md §2, moved out of ``server/`` unchanged. The
*control* half — ``AddCamera``, ``UpdateTopology``, ``Health``, ``Drain``, ``Stop`` as RPCs
on a live shard — arrives beside it in :mod:`shipinfer.launch.client`. Until it does, a
child is configured the only way a process with no control channel can be: argv and the
environment it inherits at ``exec``. Everything :meth:`Fleet._spawn` puts in the child's
environment except ``CUDA_VISIBLE_DEVICES`` is therefore temporary, and ``Fleet`` itself is
not: spawning and reaping stay exactly this shape once the RPCs exist.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from shipinfer.core.errors import ConfigurationError, ShardExitedError
from shipinfer.core.logging import get_logger
from shipinfer.core.settings.topology import SHARE_RANK_ENV, SHARED_BY_ENV, VISIBLE_GPUS_ENV
from shipinfer.envs import SHARD_CAMERAS
from shipinfer.scheduling.sharding import Shard, ShardPlan

__all__ = [
    "DEFAULT_DRAIN_S",
    "Fleet",
    "ShardProcess",
]

# The logger name stays "server.launcher" on purpose: an operator's log filter is
# behaviour, and this move promises none changed. It is retargeted to "launch…" when
# server/ is deleted (A2 PR-6).
_LOG = get_logger("server.launcher")

#: How long a shard gets to drain after SIGTERM before it is killed. Longer than a request,
#: shorter than a deploy's patience: a shard mid-batch has one batch to finish, not one video.
DEFAULT_DRAIN_S = 20.0


@dataclass(frozen=True, slots=True)
class ShardProcess:
    """One shard and the process serving it."""

    shard: Shard
    process: subprocess.Popen[bytes]

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def __str__(self) -> str:
        return (
            f"shard {self.shard.index} (pid {self.process.pid}, "
            f"{len(self.shard.cameras)} cameras, gpu(s) {list(self.shard.gpus)})"
        )


@dataclass
class Fleet:
    """Every shard's process, started together and stopped together.

    Args:
        plan: who owns which cameras and which GPUs. Pure, decided by
            :func:`~shipinfer.scheduling.sharding.plan_shards`, and printed before anything is
            spawned — a plan is cheap to read and a mis-sharded deployment is not.
        command: given a shard, the argv for its process. Injected rather than hardcoded so the
            offline tier can supervise a process that is not a server: everything below this
            line is about *supervision*, and testing it against a real server would test CUDA.
        env: extra environment for every child, on top of the parent's. ``CUDA_VISIBLE_DEVICES``
            is added per shard and overrides anything here — a shard's device set is the one
            thing the plan, not the operator, decides.
        drain_s: how long a shard gets after SIGTERM before SIGKILL.
    """

    plan: ShardPlan
    command: Callable[[Shard], Sequence[str]]
    env: Mapping[str, str] = field(default_factory=dict)
    #: Extra environment for one shard, decided by the topology (the `service` tier's shard
    #: index). Injected as a callable so `Fleet` keeps depending on nothing above the plan.
    shard_env: Callable[[Shard], Mapping[str, str]] | None = None
    drain_s: float = DEFAULT_DRAIN_S
    _running: list[ShardProcess] = field(default_factory=list, init=False, repr=False)
    #: Set by :meth:`stop`, cleared by :meth:`start`. :meth:`supervise` returns when it is set —
    #: a signal handler that stops the fleet must also end the loop that was watching it.
    _stopped: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    #: `stop` may be called from a signal handler or another thread while `supervise` is
    #: running; the lock makes the second caller wait for the first to finish rather than
    #: terminate the same processes twice.
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.drain_s <= 0:
            raise ConfigurationError(
                f"drain_s must be positive, got {self.drain_s}; a shard given no time to drain "
                f"is a shard killed mid-batch on every restart"
            )

    # -- lifecycle -----------------------------------------------------------------------

    def start(self) -> None:
        """Spawn one process per shard. Any failure takes down whatever already started.

        Partial start-up is the failure worth naming: three of four shards up means
        three-quarters of the cameras are being watched and nothing says which quarter is not.
        So the first exception unwinds the ones that made it.
        """
        if self._running:
            raise ConfigurationError("this fleet is already running")

        _LOG.info("starting fleet\n%s", self.plan.describe())
        self._stopped.clear()
        try:
            for shard in self.plan.shards:
                self._running.append(self._spawn(shard))
        except BaseException:
            self.stop()
            raise
        _LOG.info("fleet up: %d shard(s)", len(self._running))

    def _spawn(self, shard: Shard) -> ShardProcess:
        argv = list(self.command(shard))
        if not argv:
            raise ConfigurationError(f"empty command for shard {shard.index}")
        # Built from the parent's environment rather than replacing it: a child needs PATH,
        # HOME and whatever the container set, and `CUDA_VISIBLE_DEVICES` is the one value this
        # module owns outright.
        child_env = {**os.environ, **self.env}
        child_env["CUDA_VISIBLE_DEVICES"] = shard.cuda_visible_devices
        # The remap above renumbers the child's devices to 0..n-1, so an inherited
        # `SHIPINFER_DEVICES__VISIBLE_GPUS` naming *physical* ordinals — the documented way to
        # configure a single-process `serve` — would now name devices the child cannot see and
        # fail it at start-up. The child's logical view replaces it, and the sharing rides
        # beside it so a shard loads its share of every model's instances, not the whole count.
        child_env[VISIBLE_GPUS_ENV] = json.dumps(list(range(len(shard.gpus))))
        child_env[SHARED_BY_ENV] = json.dumps(list(self.plan.sharing_for(shard)))
        child_env[SHARE_RANK_ENV] = json.dumps(list(self.plan.rank_for(shard)))
        child_env[SHARD_CAMERAS.name] = ",".join(shard.cameras)
        if self.shard_env is not None:
            child_env.update(self.shard_env(shard))
        # Its own process group, so a Ctrl-C in the parent's terminal does not deliver SIGINT
        # to every shard at once and race this class's own orderly shutdown.
        process = subprocess.Popen(argv, env=child_env, start_new_session=True)
        running = ShardProcess(shard=shard, process=process)
        _LOG.info("spawned %s: CUDA_VISIBLE_DEVICES=%s", running, shard.cuda_visible_devices)
        return running

    def request_stop(self) -> None:
        """Ask :meth:`supervise` to stop the fleet, without doing any of the stopping here.

        The signal handler's whole job. `stop()` blocks up to `drain_s` under a lock, and a
        handler that calls it directly wedges the process the moment a second Ctrl-C arrives
        while the first is still draining — the handler re-enters the frame that holds the
        lock and waits on itself, and the `kill()` that follows the wait never runs, so the
        shards it was meant to end keep their CUDA contexts. Setting an event cannot block.
        """
        self._stopped.set()

    def stop(self, *, drain_s: float | None = None) -> None:
        """SIGTERM every shard, then SIGKILL whatever is still up. Idempotent.

        Kill is not a fallback that should never happen — a shard blocked in a CUDA call is not
        interruptible, and a launcher that waits forever for one leaks the GPU it holds. The
        deadline is shared across all shards rather than per shard, so stopping N of them takes
        ``drain_s``, not ``N * drain_s``.
        """
        self._stopped.set()
        with self._lock:
            if not self._running:
                return
            deadline = time.monotonic() + (self.drain_s if drain_s is None else drain_s)

            for running in self._running:
                if running.alive:
                    running.process.terminate()

            for running in self._running:
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    running.process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    _LOG.warning("%s did not drain in time; killing", running)
                    running.process.kill()
                    # Reaped unconditionally: a killed child that is never waited for is a zombie,
                    # and a supervisor that leaks those is worse than one that never ran.
                    running.process.wait()

            self._running.clear()

    # -- supervision ---------------------------------------------------------------------

    @property
    def running(self) -> tuple[ShardProcess, ...]:
        return tuple(self._running)

    def dead(self) -> tuple[ShardProcess, ...]:
        """Shards that have exited, in plan order. Does not block."""
        return tuple(r for r in self._running if not r.alive)

    def supervise(
        self, *, poll_s: float = 1.0, until: Callable[[], bool] | None = None
    ) -> None:
        """Block until a shard dies, :meth:`stop` is called, or ``until()`` says to stop.

        Returning on :meth:`stop` is what makes a signal handler work: Ctrl-C stops the fleet
        from the handler, and the loop that was watching it has to notice — otherwise the
        parent spins over an empty fleet forever, and only SIGKILL ends it.

        Raises:
            ShardExitedError: a shard exited. Its cameras are dark, and the rest of the fleet is
                still reporting healthy — which is exactly the state a supervisor exists to
                refuse to sit in. The fleet is stopped before this is raised, so the caller
                does not have to remember to.
        """
        if not self._running and not self._stopped.is_set():
            raise ConfigurationError("supervise() called on a fleet that was never started")
        while True:
            if self._stopped.is_set() or not self._running:
                # `stop()` may still be draining on the thread that called it; joining it here
                # means "supervise returned" implies "the fleet is down", which is what the
                # CLI's `finally: stop()` and the operator both assume.
                self.stop()
                return
            casualties = self.dead()
            if casualties:
                detail = ", ".join(
                    f"{c} exited with {c.process.returncode}" for c in casualties
                )
                self.stop()
                raise ShardExitedError(
                    f"{len(casualties)} of {len(self.plan)} shard(s) exited: {detail}. Those "
                    f"cameras are no longer being read, so the fleet is stopped rather than "
                    f"left partially serving"
                )
            if until is not None and until():
                return
            time.sleep(poll_s)

    # -- context manager ------------------------------------------------------------------

    def __enter__(self) -> Fleet:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()
