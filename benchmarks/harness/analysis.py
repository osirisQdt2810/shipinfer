"""Turn a buffer-occupancy log into a sustained-throughput number, per module.

This module *is* the measurement. Everything else in the harness only produces the input for
it, so the reasoning is set out in full.

The baseline's methodology
--------------------------
``benchmarks/baseline/sim_pipeline_v2.cpp`` logs one line a second::

    {"det_buffer_size": 2529, "seg_buffer_size": 2557}

and ``auto_experiments_v2_cpp.py`` calls a configuration *saturated* when those grow: the
source threads are offering more images than the GPUs retire, so the difference accumulates
in the queue. Its estimator (``analyze_experiment``, line 319) is

.. code-block:: python

    det_growth = ((det_vals[-1] - det_vals[0]) / det_capacity) / dt

— the endpoint slope, normalised by the buffer capacity, over the samples after
``--warmup-sec``. Dividing by capacity turns it into "fraction of the buffer per second",
which is the right unit for *deciding* saturation and the wrong unit for reporting
throughput. This module keeps the estimator and drops the normalisation:

.. math::

    \\text{sustained} = \\text{offered} - \\text{growth}

Why the mean of first differences, and not least squares on the levels
---------------------------------------------------------------------
A queue depth is a **cumulative sum**: :math:`y_t = y_0 + \\sum_{i \\le t} d_i` where
:math:`d_i` is one second's net accumulation. Two consequences decide the estimator.

1. Ordinary least squares on :math:`y` has strongly autocorrelated residuals, because the
   levels are an integrated series. Its slope estimate is fine, but its standard error is
   badly *understated* — often by several times — so a naive OLS confidence interval would
   declare "significantly positive" on a series that is merely drifting. Since the whole
   verdict is "does the interval contain zero", using that standard error would bias every
   answer toward SATURATED, which is the flattering direction. That is exactly the kind of
   error this file exists to avoid.
2. The first differences :math:`d_i = y_{i+1} - y_i` are far closer to independent — they
   are per-second net arrivals minus departures — and their **mean is algebraically identical
   to the endpoint slope** the baseline uses, :math:`(y_n - y_0)/n`. So the baseline's own
   estimator comes with a defensible standard error for free: :math:`s_d/\\sqrt{n_d}`.

So the reported growth is the mean first difference, its interval is the Student-t interval
of that mean, and the OLS slope is reported alongside as a robustness check. When the two
disagree by more than the interval, the series is not a straight line and
:attr:`GrowthFit.ols_slope` is the flag that says so.

The verdict
-----------
* ``SATURATED``  — the 95 % interval lies entirely **above** zero. The queue is growing;
  sustained throughput is strictly less than offered.
* ``DRAINING``   — the interval lies entirely **below** zero. There is headroom: the module
  retires more than is offered and is working off a backlog. The baseline cannot report
  this, because its own analysis floors growth at the endpoint difference and its search
  only ever asks "is it growing"; a negative slope is information and the harness reports
  how much.
* ``SUSTAINED``  — the interval contains zero. Offered load is being met. This is the only
  outcome that licenses the claim "the system sustains X img/s", and X is then the offered
  rate, not a fitted one.

"The interval contains zero" rather than "slope < 5" is deliberate. A threshold in images per
second is unfalsifiable: 5 img/s is noise on a 1000 img/s run and a wall on a 20 img/s one,
and nothing in the data tells you which. An interval scales with the noise the run actually
had, and it makes the answer depend on how long you measured — which is correct, because a
five-sample run genuinely cannot distinguish +3 img/s from zero.

Per module, never averaged
--------------------------
The baseline reports ``det`` and ``seg`` separately because they saturate independently: one
can be the wall while the other has headroom, and their mean is a number describing neither.
Everything here is keyed by module and the summary names the binding constraint rather than
adding the modules up.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "BUFFER_SUFFIX",
    "DRAINING",
    "SATURATED",
    "SUSTAINED",
    "GrowthFit",
    "ModuleResult",
    "RunAnalysis",
    "SampleLog",
    "analyse",
    "as_json",
    "fit_growth",
    "read_log",
    "render",
]

#: The key suffix both systems use. ``det_buffer_size`` is the baseline's; ShipInfer writes
#: ``ship_detector_buffer_size`` and friends. One suffix is what lets one analysis read both.
BUFFER_SUFFIX = "_buffer_size"

SATURATED = "SATURATED"
SUSTAINED = "SUSTAINED"
DRAINING = "DRAINING"

#: Two-sided 95 % Student-t critical values, by degrees of freedom. Tabulated rather than
#: pulled from scipy so the analysis has the same answer with and without scipy installed --
#: a verdict that moves when an unrelated package appears is not a verdict. Values above 30
#: df are within 2 % of the normal 1.96, so the tail is interpolated against it.
_T95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    50: 2.009,
    60: 2.000,
    80: 1.990,
    100: 1.984,
}


def _t_critical(df: int) -> float:
    """Two-sided 95 % critical value for ``df`` degrees of freedom."""
    if df <= 0:
        return math.inf
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95)
    if df > keys[-1]:
        return 1.960 + (_T95[keys[-1]] - 1.960) * keys[-1] / df
    lower = max(k for k in keys if k < df)
    upper = min(k for k in keys if k > df)
    span = upper - lower
    return _T95[lower] + (_T95[upper] - _T95[lower]) * (df - lower) / span


# ---------------------------------------------------------------------------- reading


@dataclass(frozen=True, slots=True)
class SampleLog:
    """One run's occupancy samples, plus whatever metadata the writer left behind.

    Attributes:
        modules: buffer keys in first-seen order, with ``_buffer_size`` stripped.
        times: seconds since the first sample.
        series: module -> occupancy at each time.
        meta: the ``{"meta": {...}}`` line if the writer emitted one. The baseline emits
            none, which is why this is optional rather than required.
        time_source: ``"timestamp"`` when every row carried ``t``, else ``"index"``. Recorded
            because a fitted slope in units of "per sample" is only a rate if the sample
            interval is what it claims to be.
    """

    modules: tuple[str, ...]
    times: tuple[float, ...]
    series: Mapping[str, tuple[float, ...]]
    meta: Mapping[str, Any] = field(default_factory=dict)
    time_source: str = "index"

    def __len__(self) -> int:
        return len(self.times)


def read_log(path: str | Path, *, sample_interval_s: float = 1.0) -> SampleLog:
    """Parse a buffer-occupancy JSONL, from either system.

    Accepts exactly what the baseline's own ``parse_log_line`` accepts and one thing more: a
    line with no ``*_buffer_size`` key is skipped rather than being an error, which is what
    lets the harness put a ``{"meta": ...}`` header in the same file without breaking the
    baseline's reader.

    Args:
        path: the JSONL to read.
        sample_interval_s: assumed spacing when rows carry no ``t``. The baseline's logger
            sleeps 10x100 ms between samples, so 1.0 is its real cadence; a run whose
            metadata says otherwise should pass its own value.

    Raises:
        ValueError: the file holds no parseable sample. An empty result would otherwise
            travel all the way to a printed table of zeroes.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "meta" in obj and isinstance(obj["meta"], dict):
            meta = obj["meta"]
            continue
        if any(k.endswith(BUFFER_SUFFIX) for k in obj):
            rows.append(obj)

    if not rows:
        raise ValueError(
            f"{path} holds no line with a '*{BUFFER_SUFFIX}' key. Either the run produced no "
            f"samples (it died before the first second) or the log is not an occupancy log"
        )

    modules: list[str] = []
    for row in rows:
        for key in row:
            if key.endswith(BUFFER_SUFFIX):
                name = key[: -len(BUFFER_SUFFIX)]
                if name not in modules:
                    modules.append(name)

    has_t = all("t" in row for row in rows)
    if has_t:
        times = tuple(float(row["t"]) for row in rows)
        source = "timestamp"
    else:
        times = tuple(i * sample_interval_s for i in range(len(rows)))
        source = "index"

    series = {
        name: tuple(float(row.get(f"{name}{BUFFER_SUFFIX}", math.nan)) for row in rows)
        for name in modules
    }
    return SampleLog(
        modules=tuple(modules), times=times, series=series, meta=meta, time_source=source
    )


# ---------------------------------------------------------------------------- fitting


@dataclass(frozen=True, slots=True)
class GrowthFit:
    """A growth rate in units per second, with the uncertainty that decides the verdict."""

    #: Mean of the first differences, rescaled to per-second. Algebraically the endpoint
    #: slope, which is the baseline's own estimator.
    slope: float
    #: Half-width of the two-sided 95 % interval. ``inf`` when there are too few samples.
    half_width: float
    #: Least-squares slope over the same window, as a robustness check only.
    ols_slope: float
    #: Samples used, after the warmup skip.
    samples: int
    #: Seconds spanned by those samples.
    span_s: float

    @property
    def low(self) -> float:
        return self.slope - self.half_width

    @property
    def high(self) -> float:
        return self.slope + self.half_width

    @property
    def verdict(self) -> str:
        if not math.isfinite(self.half_width):
            return SUSTAINED
        if self.low > 0.0:
            return SATURATED
        if self.high < 0.0:
            return DRAINING
        return SUSTAINED

    @property
    def linear(self) -> bool:
        """Whether the endpoint and least-squares slopes agree inside the interval.

        False means the series is curved — the growth rate itself is changing — and a single
        number for it is a summary rather than a description. Worth printing, because a
        saturated run whose growth accelerates is a run whose measurement window was short.
        """
        if not math.isfinite(self.half_width):
            return True
        return abs(self.ols_slope - self.slope) <= self.half_width


def fit_growth(times: Sequence[float], values: Sequence[float]) -> GrowthFit:
    """Fit the growth rate of one buffer over one window.

    Args:
        times: seconds, strictly increasing, already trimmed to the steady tail.
        values: occupancy at each time.

    Returns:
        The fit. With fewer than three samples the interval is infinite, which makes the
        verdict :data:`SUSTAINED` — the honest answer when the data cannot distinguish
        growth from noise, rather than a slope reported as if it were meaningful.

    Raises:
        ValueError: the two sequences differ in length, or the window has no duration.
    """
    if len(times) != len(values):
        raise ValueError(f"times ({len(times)}) and values ({len(values)}) differ in length")
    n = len(values)
    if n < 2:
        return GrowthFit(slope=0.0, half_width=math.inf, ols_slope=0.0, samples=n, span_s=0.0)

    span = times[-1] - times[0]
    if span <= 0.0:
        raise ValueError(f"the window has no duration: {times[0]} .. {times[-1]}")

    # Per-second first differences. Dividing each by its own dt keeps the estimator honest
    # when the sampler slipped a tick, which a 1000 fps run on a loaded box does.
    diffs = [
        (values[i + 1] - values[i]) / (times[i + 1] - times[i])
        for i in range(n - 1)
        if times[i + 1] > times[i]
    ]
    if not diffs:
        raise ValueError("no positive time step between consecutive samples")
    mean = sum(diffs) / len(diffs)

    if len(diffs) < 2:
        half = math.inf
    else:
        var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
        stderr = math.sqrt(var / len(diffs))
        half = _t_critical(len(diffs) - 1) * stderr

    # Ordinary least squares on the levels, for the curvature check only.
    t_mean = sum(times) / n
    v_mean = sum(values) / n
    sxx = sum((t - t_mean) ** 2 for t in times)
    sxy = sum((t - t_mean) * (v - v_mean) for t, v in zip(times, values, strict=True))
    ols = sxy / sxx if sxx > 0 else 0.0

    return GrowthFit(slope=mean, half_width=half, ols_slope=ols, samples=n, span_s=span)


# ---------------------------------------------------------------------------- analysis


@dataclass(frozen=True, slots=True)
class ModuleResult:
    """One module's growth, and its sustained throughput when the offered rate is known."""

    module: str
    fit: GrowthFit
    #: Images per second offered to this module, or ``None`` when it is data-dependent. The
    #: baseline's two modules are fed directly by source threads so it is known exactly;
    #: ShipInfer's downstream models are fed by however many crops the detector found, so it
    #: is *measured* by the driver rather than derived from the config, and is ``None`` when
    #: the driver could not measure it.
    offered: float | None
    first: float
    last: float
    peak: float
    capacity: int | None = None

    @property
    def sustained(self) -> float | None:
        """``offered - growth``, or ``None`` when offered is unknown.

        Clamped at zero: a growth rate larger than the offered rate is arithmetically
        possible from noise on a short window and physically is not, and a negative
        throughput in a report is a bug that looks like a result.
        """
        if self.offered is None:
            return None
        return max(0.0, self.offered - self.fit.slope)

    @property
    def headroom(self) -> float | None:
        """How much spare capacity a draining module showed, in images per second."""
        if self.fit.verdict != DRAINING:
            return None
        return -self.fit.slope

    @property
    def utilisation(self) -> float | None:
        """Peak occupancy as a fraction of capacity — the baseline's ``*_util_max``."""
        if not self.capacity:
            return None
        return self.peak / self.capacity

    def as_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "growth_per_sec": round(self.fit.slope, 2),
            "growth_ci95": [round(self.fit.low, 2), round(self.fit.high, 2)],
            "ols_growth_per_sec": round(self.fit.ols_slope, 2),
            "linear": self.fit.linear,
            "verdict": self.fit.verdict,
            "offered": self.offered,
            "sustained": None if self.sustained is None else round(self.sustained, 1),
            "headroom": None if self.headroom is None else round(self.headroom, 1),
            "samples": self.fit.samples,
            "span_s": round(self.fit.span_s, 1),
            "first": self.first,
            "last": self.last,
            "peak": self.peak,
            "utilisation": None if self.utilisation is None else round(self.utilisation, 4),
        }


@dataclass(frozen=True, slots=True)
class RunAnalysis:
    """Every module's result plus the one-line verdict for the run."""

    system: str
    modules: tuple[ModuleResult, ...]
    warmup_s: float
    skipped_samples: int
    time_source: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        """SATURATED if any module is. One saturated module bounds the whole pipeline."""
        verdicts = {m.fit.verdict for m in self.modules}
        if SATURATED in verdicts:
            return SATURATED
        if verdicts == {DRAINING}:
            return DRAINING
        return SUSTAINED

    @property
    def binding_module(self) -> ModuleResult | None:
        """The module with the fastest growth — the wall, when there is one."""
        saturated = [m for m in self.modules if m.fit.verdict == SATURATED]
        pool = saturated or list(self.modules)
        return max(pool, key=lambda m: m.fit.slope) if pool else None

    @property
    def total_sustained(self) -> float | None:
        """Sum of the modules whose offered rate is known.

        A sum, not a mean, and only over modules fed independently: for the baseline that is
        det + seg and it is the number the two systems are compared on. It is ``None`` when
        any contributing module's offered rate was unmeasurable, because a partial sum
        reported as a total is worse than no total.
        """
        known = [m for m in self.modules if m.offered is not None]
        if not known:
            return None
        return sum(m.sustained or 0.0 for m in known)

    def as_dict(self) -> dict[str, Any]:
        binding = self.binding_module
        return {
            "system": self.system,
            "verdict": self.verdict,
            "binding_module": None if binding is None else binding.module,
            "total_sustained": (
                None if self.total_sustained is None else round(self.total_sustained, 1)
            ),
            "warmup_s": self.warmup_s,
            "skipped_samples": self.skipped_samples,
            "time_source": self.time_source,
            "modules": [m.as_dict() for m in self.modules],
            "meta": dict(self.meta),
        }


def analyse(
    log: SampleLog,
    *,
    system: str,
    warmup_s: float = 10.0,
    offered: Mapping[str, float | None] | None = None,
    capacity: int | None = None,
) -> RunAnalysis:
    """Fit every module in one log and decide the run's verdict.

    Args:
        log: the parsed occupancy samples.
        system: ``"baseline"`` or ``"shipinfer"`` — carried into the summary so two files
            cannot be mixed up later.
        warmup_s: leading seconds to discard. The default matches the ramp both systems
            show: the baseline's source threads stagger their first decode and its runners
            deserialise engines while sources already produce, and ShipInfer's TensorRT
            instances warm up per device. Fitting through that measures start-up.
        offered: module -> images per second offered to it, or ``None`` where it is
            data-dependent. Missing keys are treated as unknown.
        capacity: the buffer bound, for the utilisation figures the baseline reports.

    Raises:
        ValueError: the warmup skip leaves fewer than two samples. Silently analysing one
            sample is how a 3-second run gets reported as a throughput measurement.
    """
    offered = offered or {}
    keep = [i for i, t in enumerate(log.times) if t >= warmup_s]
    if len(keep) < 2:
        raise ValueError(
            f"warmup_s={warmup_s} leaves {len(keep)} of {len(log)} sample(s); the run was "
            f"too short to fit a growth rate. Lengthen the run or lower warmup_s"
        )
    times = [log.times[i] for i in keep]

    results: list[ModuleResult] = []
    for module in log.modules:
        raw = log.series[module]
        values = [raw[i] for i in keep]
        results.append(
            ModuleResult(
                module=module,
                fit=fit_growth(times, values),
                offered=offered.get(module),
                first=values[0],
                last=values[-1],
                peak=max(values),
                capacity=capacity,
            )
        )

    return RunAnalysis(
        system=system,
        modules=tuple(results),
        warmup_s=warmup_s,
        skipped_samples=len(log) - len(keep),
        time_source=log.time_source,
        meta=log.meta,
    )


def render(analyses: Sequence[RunAnalysis]) -> str:
    """A fixed-width table of one or more runs. No rich, so it survives a log file."""
    header = (
        f"{'system':<10} {'module':<18} {'offered':>9} {'growth/s':>10} "
        f"{'ci95':>18} {'sustained':>10} {'verdict':<10}"
    )
    lines = [header, "-" * len(header)]
    for run in analyses:
        for module in run.modules:
            offered = "-" if module.offered is None else f"{module.offered:.0f}"
            sustained = "-" if module.sustained is None else f"{module.sustained:.1f}"
            ci = f"[{module.fit.low:+.1f}, {module.fit.high:+.1f}]"
            flag = "" if module.fit.linear else " *"
            lines.append(
                f"{run.system:<10} {module.module:<18} {offered:>9} "
                f"{module.fit.slope:>+10.1f} {ci:>18} {sustained:>10} "
                f"{module.fit.verdict:<10}{flag}"
            )
        total = run.total_sustained
        lines.append(
            f"{run.system:<10} {'TOTAL':<18} {'':>9} {'':>10} {'':>18} "
            f"{'-' if total is None else f'{total:.1f}':>10} {run.verdict:<10}"
        )
        lines.append("")
    if any(not m.fit.linear for run in analyses for m in run.modules):
        lines.append("* endpoint and least-squares slopes disagree: the series is curved,")
        lines.append("  so one growth rate summarises it rather than describing it.")
    return "\n".join(lines)


def as_json(analyses: Sequence[RunAnalysis]) -> str:
    return json.dumps([a.as_dict() for a in analyses], indent=2)
