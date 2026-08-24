"""The once-a-second occupancy writer. One shape, both systems.

The baseline writes its own log from C++::

    {"det_buffer_size": 2529, "seg_buffer_size": 2557}

so the harness cannot write the baseline's samples — it can only read them. What it *can*
do is guarantee that ShipInfer's log has the identical shape, and that is what this module
is for: the same key suffix, the same one-second cadence, the same one-object-per-line file.
:func:`benchmarks.harness.analysis.read_log` then reads both with one parser, which is the
only way to be sure the two numbers were computed the same way.

Two deliberate deviations, both additive and both skippable by the baseline's own reader
(``parse_log_line`` ignores any line without both of its keys):

* **a leading ``{"meta": {...}}`` line.** It records the configuration, the resolved
  artefact paths, whether the process was in a container, and — the reason it exists — which
  pipeline stages were actually wired. A comparison against a pipeline missing a stage is
  not a comparison, so the omission travels with the data rather than living in a report
  somebody may not have read.
* **a ``"t"`` key** on every sample: seconds since the first sample. The baseline has no
  timestamp, so its analysis has to assume the logger's ``10 x 100 ms`` sleep really was one
  second. Ours does not have to assume it, and the analysis records which of the two it
  used.

The sampler runs on its own thread and never touches the system under test beyond calling
``probe()``. That matters: ``probe`` reads ``Model.total_depth``, which walks the instance
list and reads each queue's depth, and doing that from a pipeline worker would put the
measurement inside the thing being measured.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from benchmarks.harness.analysis import BUFFER_SUFFIX

__all__ = ["OccupancySampler"]

Probe = Callable[[], Mapping[str, int]]


class OccupancySampler:
    """Append ``{"t": s, "<module>_buffer_size": n, ...}`` once per interval.

    Args:
        path: the JSONL to write. Parent directories are created; an existing file is
            **truncated**, because appending to a previous run's log would silently splice
            two experiments into one growth fit.
        probe: returns module -> current occupancy. Called from the sampler thread.
        interval_s: seconds between samples. 1.0 matches the baseline; anything else makes
            the two logs incomparable and the metadata records what was used.
        meta: written as the first line. See the module docstring.

    Raises:
        ValueError: ``interval_s`` is not positive, or ``probe`` returns a key that does not
            end in ``_buffer_size``. The suffix is the contract with the analysis, and a
            silently-mistyped key would produce a log the analysis reads as empty.
    """

    def __init__(
        self,
        path: str | Path,
        probe: Probe,
        *,
        interval_s: float = 1.0,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s}")
        self._path = Path(path)
        self._probe = probe
        self._interval = interval_s
        self._meta = dict(meta or {})
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples = 0
        self._failures = 0
        self._start_ns = 0

    @property
    def samples(self) -> int:
        """How many occupancy lines were written. Zero means the run produced no data."""
        return self._samples

    @property
    def failures(self) -> int:
        """Probes that raised. Counted rather than fatal: the sampler must outlive one bad
        read of a model that is mid-shutdown, or a clean stop would lose the whole log."""
        return self._failures

    @property
    def path(self) -> Path:
        return self._path

    # -- lifecycle ----------------------------------------------------------------------

    def start(self) -> OccupancySampler:
        """Write the metadata line and begin sampling. Not idempotent by design.

        A second ``start()`` on a running sampler would give two threads one file
        descriptor and interleave half-written lines, so it raises instead.
        """
        if self._thread is not None:
            raise RuntimeError("this sampler is already running")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"meta": self._meta}) + "\n")
        self._start_ns = time.monotonic_ns()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="bench-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop sampling and join. Idempotent."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            self._thread = None

    def __enter__(self) -> OccupancySampler:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- the loop -----------------------------------------------------------------------

    def _run(self) -> None:
        # Deadline pacing rather than `sleep(interval)`: the probe itself takes time, and
        # sleeping a fixed interval after it makes every sample slightly late and the
        # lateness cumulative. The analysis divides by the real dt, so a slipped tick is
        # survivable — but a systematically stretched axis would scale every growth rate.
        deadline = time.monotonic()
        with self._path.open("a", encoding="utf-8") as handle:
            while not self._stop.is_set():
                deadline += self._interval
                elapsed_ns = time.monotonic_ns() - self._start_ns
                try:
                    row: dict[str, Any] = {"t": round(elapsed_ns / 1e9, 3)}
                    for name, depth in self._probe().items():
                        if not name.endswith(BUFFER_SUFFIX):
                            raise ValueError(
                                f"probe returned {name!r}; every key must end in "
                                f"{BUFFER_SUFFIX!r} or the analysis will not see it"
                            )
                        row[name] = int(depth)
                except Exception:
                    self._failures += 1
                else:
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    self._samples += 1
                remaining = deadline - time.monotonic()
                if remaining > 0 and self._stop.wait(remaining):
                    return


def depth_probe(models: Mapping[str, Any]) -> Probe:
    """A probe over ``{module_name: object_with_total_depth}``.

    Separate from :class:`OccupancySampler` so the sampler can be tested with a counting
    fake and no server, and so the ShipInfer driver can compose the pipeline queue's depth
    with the models' depths without the sampler knowing what either is.
    """

    def probe() -> dict[str, int]:
        return {
            f"{name}{BUFFER_SUFFIX}": int(handle.total_depth) for name, handle in models.items()
        }

    return probe
