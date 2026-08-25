"""Where the time actually goes, at three levels of detail and three costs.

The system already answers *"queue or compute?"* — :class:`~shipinfer.core.request.Timings`
stamps six points along the serving path and the metrics registry histograms the derived
spans. What it could not answer was *"which op"*: `compute_us` is one number covering the
host-to-device copy, the network, and the copy back, so an 8 ms batch gave no clue whether
the GPU was busy for 8 ms or busy for 5 and idle for 3.

That gap blocked a real decision. The instance worker executes one batch synchronously —
assemble, copy in, infer, copy out, scatter — so the device is idle during both copies, and
the standard fix is to overlap them on separate streams. Whether that is worth doing depends
entirely on how large the idle fraction is, and there was no way to find out. Guessing is how
this project already shipped an adaptive batching window that measured identically to the
fixed one and had to be reverted.

Three levels, cheapest first:

**1. NVTX ranges — always compiled in, free when nobody is watching.**
`torch.cuda.nvtx.range_push/pop` is a handful of instructions when no profiler is attached.
Without them an `nsys profile` timeline is a wall of unlabelled kernels; with them it reads
`assemble / h2d / execute / d2h / scatter` per batch. This is the one that costs nothing, so
it is not behind a switch.

**2. `torch.profiler` — opt in with SHIPINFER_PROFILE_DIR.**
The vLLM shape: point it at a directory and get a Chrome trace with per-kernel times. Heavy
enough that it must never be on by default, and scoped so it captures a bounded number of
steps rather than filling a disk.

**3. Timed CUDA events — opt in with SHIPINFER_PROFILE_PHASES.**
Splits `compute_us` into `h2d_us`, `execute_us` and `d2h_us` as real histograms an operator
can watch in Prometheus over hours, rather than a trace someone has to open. This is the
level that answers the overlap question directly, and it is a switch because
`Event(enable_timing=True)` plus the synchronise needed to read it serialises what the rest
of the design works to keep asynchronous. Measuring it changes it — so it is opt-in, and the
docstring says so rather than leaving someone to discover a latency regression from enabling
telemetry.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from shipinfer.core.logging import get_logger
from shipinfer.envs import PROFILE_DIR, PROFILE_PHASES, PROFILE_STEPS
from shipinfer.runtime.platform import torch_module

__all__ = [
    "TRACE_EVENTS",
    "PhaseTimer",
    "PhaseTimings",
    "nvtx_range",
    "phase_timer",
    "profiling_enabled",
    "torch_profiler",
]

_LOG = get_logger("runtime.profiling")

#: The phases an instance executes, named as **Triton** names them.
#:
#: Triton is the framework this server is shaped after, and its metrics split a request into
#: `queue`, then `compute_input`, `compute_infer`, `compute_output` — see
#: `nv_inference_queue_duration_us` and `nv_inference_compute_*_duration_us`. Using our own
#: vocabulary here would mean an operator who knows Triton's dashboards has to learn a second
#: one, and a comparison against a Triton deployment would need a translation table nobody
#: maintains. So the names are Triton's, and the mapping is:
#:
#:   compute_input   assembling the batch and copying it to the device
#:   compute_infer   the backend's execute
#:   compute_output  copying results back and scattering them per request
#:
#: `queue` is not here because it is already measured on the request itself (`Timings`), which
#: is where Triton measures it too — a queue span belongs to the request, not to the batch
#: that eventually ran it.
PHASES = ("compute_input", "compute_infer", "compute_output")

#: Trace event names, as Triton's trace API emits them — re-exported from `core/tracing`, which
#: owns them, rather than restated. This file *had* its own copy, byte-identical, and the test
#: pinning the two together was moved out in the same change: the next piece to add an eighth
#: event or rename one to match a Triton release would have left the two files emitting
#: different vocabularies with a green suite — and the whole reason for using Triton's names
#: is that an operator can diff a trace against one from Triton without a translation table.
#: `runtime` → `core` is a legal import direction, so there is no reason for a copy.
from shipinfer.core.tracing import TRACE_EVENTS  # noqa: E402 - see the note above


def profiling_enabled() -> bool:
    """Whether any opt-in level is active. Cheap enough to call per batch."""
    return bool(PROFILE_DIR.get()) or PROFILE_PHASES.get()


@contextlib.contextmanager
def nvtx_range(label: str) -> Iterator[None]:
    """Annotate a span for an external profiler. A no-op cost when none is attached.

    Deliberately not behind a switch. `range_push`/`range_pop` are a few instructions with
    no profiler present, and the alternative — an unlabelled `nsys` timeline — is the
    difference between a trace that answers a question and one that raises three.
    """
    torch = torch_module()
    if torch is None or not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@dataclass(slots=True)
class PhaseTimings:
    """Microseconds per phase for one batch, plus what they imply.

    The three phases are Triton's (`compute_input`, `compute_infer`, `compute_output`), so
    the numbers line up with `nv_inference_compute_*_duration_us` on a Triton deployment.

    :attr:`device_busy_us` and :attr:`idle_fraction` are **ours, not Triton's** — Triton
    reports the three spans and leaves the arithmetic to the reader. They are here because
    that arithmetic is the decision: the instance worker runs one batch synchronously, so the
    device is idle during both copies, and whether overlapping them on separate streams is
    worth the complexity depends entirely on how large that idle fraction is. 3% is not worth
    a redesign; 30% is. Anything derived rather than measured says so in its docstring.
    """

    per_phase: dict[str, float] = field(default_factory=dict)
    wall_us: float = 0.0

    @property
    def device_busy_us(self) -> float:
        """The spans that occupy the device. Ours, not a Triton metric."""
        return sum(self.per_phase.get(name, 0.0) for name in PHASES)

    @property
    def is_measured(self) -> bool:
        """Whether any device phase was actually timed.

        Load-bearing, not a convenience. With phase timing off, `device_busy_us` is 0 and a
        naive `1 - busy/wall` reports **100% idle** — which is not a cautious default but a
        confident falsehood about the exact quantity this module exists to establish. Callers
        must check this before recording or believing :attr:`idle_fraction`.
        """
        return bool(self.per_phase)

    @property
    def idle_fraction(self) -> float:
        """Share of the worker's wall-clock during which the device was doing nothing.

        Zero — meaning "nothing to report" — when nothing was measured or there is nothing to
        divide by. Zero rather than NaN because this reaches a metrics histogram and one NaN
        poisons every aggregate computed over it; and zero rather than one because an
        unmeasured span must not read as a problem. Gate on :attr:`is_measured`.
        """
        if not self.is_measured or self.wall_us <= 0.0:
            return 0.0
        return max(0.0, 1.0 - self.device_busy_us / self.wall_us)

    def __repr__(self) -> str:
        if not self.is_measured:
            return f"<PhaseTimings unmeasured wall={self.wall_us:.0f}us>"
        parts = " ".join(f"{k}={v:.0f}us" for k, v in self.per_phase.items())
        return f"<PhaseTimings {parts} idle={self.idle_fraction:.1%}>"


class PhaseTimer:
    """Times the device phases of one batch with CUDA events, when asked to.

    Off by default, and off means *free*: :meth:`phase` reduces to the NVTX range alone, with
    no event allocation and no synchronise. On, it pairs an event around each device phase and
    reads them all in one synchronise at :meth:`finish`, rather than once per phase — reading
    per phase would serialise the very overlap the numbers are meant to inform.

    Events are reused across batches. Allocating a pair per phase per batch would be roughly
    5000 allocations a second at the design point, which is measurable on its own and would
    make the instrument part of what it measures.
    """

    def __init__(self, *, enabled: bool | None = None) -> None:
        self._torch = torch_module()
        self.enabled = (
            (PROFILE_PHASES.get() if enabled is None else enabled)
            and self._torch is not None
            and self._torch.cuda.is_available()
        )
        self._events: dict[str, tuple[Any, Any]] = {}
        self._seen: list[str] = []
        self._lock = threading.Lock()

    def _pair(self, label: str) -> tuple[Any, Any]:
        pair = self._events.get(label)
        if pair is None:
            assert self._torch is not None
            pair = (
                self._torch.cuda.Event(enable_timing=True),
                self._torch.cuda.Event(enable_timing=True),
            )
            self._events[label] = pair
        return pair

    @contextlib.contextmanager
    def phase(self, label: str) -> Iterator[None]:
        """Time one phase, and annotate it for an external profiler either way."""
        if not self.enabled:
            with nvtx_range(label):
                yield
            return

        start, end = self._pair(label)
        start.record()
        with nvtx_range(label):
            try:
                yield
            finally:
                end.record()
                with self._lock:
                    if label not in self._seen:
                        self._seen.append(label)

    def finish(self, wall_us: float) -> PhaseTimings:
        """One synchronise, then read every phase. Returns empty timings when disabled."""
        if not self.enabled or not self._seen:
            return PhaseTimings(wall_us=wall_us)

        assert self._torch is not None
        self._torch.cuda.synchronize()
        per_phase: dict[str, float] = {}
        with self._lock:
            labels = list(self._seen)
            self._seen.clear()
        for label in labels:
            start, end = self._events[label]
            # elapsed_time is milliseconds; every other duration in this system is
            # microseconds, and mixing the two is how a p99 ends up off by 1000.
            per_phase[label] = start.elapsed_time(end) * 1_000.0
        return PhaseTimings(per_phase=per_phase, wall_us=wall_us)


def phase_timer(*, enabled: bool | None = None) -> PhaseTimer:
    """A timer honouring `SHIPINFER_PROFILE_PHASES` unless told otherwise."""
    return PhaseTimer(enabled=enabled)


@contextlib.contextmanager
def torch_profiler(label: str) -> Iterator[Any | None]:
    """A `torch.profiler` scope writing a Chrome trace, when `SHIPINFER_PROFILE_DIR` is set.

    Bounded by `SHIPINFER_PROFILE_STEPS` — an unbounded profiler against 1000 frames a second
    fills a disk in minutes, and the resulting trace is too large to open, so an unbounded
    one is not a more thorough measurement but a failed one.

    Yields the profiler so a caller can `step()` it, or `None` when profiling is off, which
    keeps the call site a single `with` either way.
    """
    directory = PROFILE_DIR.get()
    torch = torch_module()
    if not directory or torch is None:
        yield None
        return

    from pathlib import Path

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    steps = PROFILE_STEPS.get()
    _LOG.info("torch profiler active: %s steps -> %s", steps, target)
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=steps, repeat=1)
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(target / label)),
        record_shapes=True,
        with_stack=False,
    ) as profiler:
        yield profiler
