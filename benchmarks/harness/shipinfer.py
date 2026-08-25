"""Drive ShipInfer through the whole stack: ingest -> scheduler -> engines -> reassembly.

The requirement this file answers is that the comparison must exercise the *system*, not a
convenient slice of it. So the path a frame takes here is::

    ReplaySource (1080p JPEG, decoded per frame, paced by DeadlinePacer)
      -> CameraActor -> QueueFrameSink
      -> FairPriorityQueue, bucketed by camera_id
      -> PipelineRunner worker
      -> DetectStage      -> ship_detector   (real TensorRT plan, yolo26n)
      -> CropStage        -> GPU crop + letterbox
      -> ObjectStage      -> ship_segmenter  (real TensorRT plan, yolo26n-seg)
      -> ObjectStage      -> ship_embedder   (real TensorRT plan, ResNet-50)
      -> ObjectStage      -> person_embedder (real TensorRT plan, ResNet-50)
      -> FrameCollector   -> PerceptionEvent -> ResultSink

No mock backend, no synthetic tensors, no cost model. Every model is the repository's own
``config.yaml`` pointing at a plan built from the same ONNX the baseline's engines come from.

What is **not** wired, and what that costs the comparison
--------------------------------------------------------
Two of the layers the comparison was asked to cover are not reachable from this repository
as it stands. Both are recorded in the log's metadata under ``stages_missing`` so the
omission travels with the data:

``tracking``
    ``shipvision.tracking`` is not referenced anywhere in ``src/shipinfer`` and
    ``build_perception_graph`` has no tracking stage — the graph's own docstring says
    tracking is stateful and lives in another service behind the message bus. So the
    measured pipeline ends at reassembly. Effect on the comparison: **none in ShipInfer's
    favour that matters**, because the baseline has no tracking either; it is a gap against
    the *task's* description of the target pipeline, not against the baseline.

``fused kernels``
    ``runtime/native.py`` still imports ``shipinfer_imgproc``, the name of a submodule that
    has been replaced by ``shipvision``, and ``shipvision._C`` is not built in this
    environment. So ``get_image_ops`` resolves to :class:`TorchImageOps` — GPU letterbox and
    GPU crop through torch kernels, not the single-pass fused kernel. Effect on the
    comparison: it is measured **without** the fused-kernel speedup, so the pre-processing
    figure here is a floor rather than the system's best. The resolved implementation is
    recorded in the metadata as ``ops`` so no reader has to guess.

Why ``pipeline_workers`` is large
--------------------------------
A pipeline worker drives one frame through the DAG and *blocks* on each stage, so the number
of frames in flight is the worker count. The detector has two instances per GPU with a batch
of 8, so keeping four GPUs busy needs 64 concurrent detector requests before any downstream
stage is considered. That is a real difference from the baseline, whose four inference
threads each assemble their own batch of 8 from a shared queue, and it is the knob that makes
the two comparable rather than an artefact of Python threading.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.harness.analysis import BUFFER_SUFFIX
from benchmarks.harness.config import BenchConfig
from benchmarks.harness.histograms import HistogramCell, read_cell
from benchmarks.harness.sampler import OccupancySampler

__all__ = [
    "PIPELINE_MODULE",
    "ShipInferResult",
    "check_offer",
    "per_module_capacity",
    "reconcile",
    "run_shipinfer",
]

#: The module name used for the ingest queue in the JSONL. It is the direct analogue of the
#: baseline's ``det_buffer_size``: decoded frames waiting for a worker.
PIPELINE_MODULE = "pipeline"

#: Fraction of read frames backpressure may refuse before the run stops being a throughput
#: measurement. Small on purpose: a little shedding is the system working as designed, and a
#: lot of it means the offered rate and the served rate are different experiments.
MAX_DROP_FRACTION = 0.02

#: Fraction of offered work the *scheduler* may refuse before the run stops being a
#: throughput measurement. Separate from `MAX_DROP_FRACTION`, which is ingest-side: a
#: rejected inference request is the more dangerous vanish-path, because it makes the buffers
#: look flat for the worst possible reason. Nothing accumulates when work is being refused,
#: so both slopes read zero and the whole offered rate publishes as sustained.
MAX_REJECT_FRACTION = 0.02

#: How far the analysed rate may sit above what actually came out of the pipeline. The
#: buffer-growth method and the emitted-event count are independent estimates of the same
#: quantity; a large gap means one of them is wrong, and the honest response is to refuse
#: rather than to pick the flattering one.
MAX_RECONCILE_GAP = 0.15

#: Models the graph drives, in stage order. Their queue depths are the second-level buffer
#: the baseline has no equivalent of, and they are sampled because a model that is the wall
#: shows it here before the ingest queue notices.
GRAPH_MODELS = ("ship_detector", "ship_segmenter", "ship_embedder", "person_embedder")


@dataclass(frozen=True, slots=True)
class ShipInferResult:
    """What one ShipInfer run produced."""

    log: Path
    startup_s: float
    elapsed_s: float
    frames_accepted: int
    events_emitted: int
    #: What ingest actually read and dropped over the run. The offered rate is a *claim*
    #: until these are checked against it: 50 Python replay threads doing per-frame OpenCV
    #: decode in one interpreter can fall short of the target, and `timing/pacing.py`
    #: absorbs lateness rather than catching up, so the shortfall is silent.
    frames_read: int = 0
    frames_dropped: int = 0
    #: The same counters, but only over the window the growth fit actually uses: from the
    #: warmup boundary to the end. Rating cumulative counters over the *whole* run while
    #: fitting growth over the steady part alone compares two different windows. With
    #: cameras staggering their first decode and instances deserialising engines through
    #: `t < warmup_s`, a run that offered a clean 1000 img/s in steady state measured
    #: 60000/70 = 857, which either tripped `check_offer` and destroyed a good run or
    #: understated capacity by 143 img/s in `sustained = offered - growth`.
    steady_s: float = 0.0
    steady_frames_read: int = 0
    steady_frames_dropped: int = 0
    steady_requests_total: dict[str, float] = field(default_factory=dict)
    steady_events_emitted: int = 0
    #: The algo tier's denominators, over the same window. `steady_frames_accepted` is what
    #: the pipeline accepted between the warm-up boundary and the end; `steady_stage_latency`
    #: is each stage's latency histogram over exactly that window — the histogram at the end
    #: minus the histogram at the boundary, per bucket, so a steady-window mean and quantile
    #: exist. A per-frame cost is a steady total over a steady frame count; dividing a steady
    #: duration by a whole-run count is the two-window mistake the comment above describes.
    steady_frames_accepted: int = 0
    steady_stage_latency: dict[str, HistogramCell] = field(default_factory=dict)
    #: True when the run ended before its own warm-up. The `steady_*` fields are then the
    #: whole run, and anything built on them has to say so rather than call it steady.
    steady_is_whole_run: bool = False
    #: model -> requests accepted, from ``ServerMetrics.requests_total``. The delta over the
    #: steady window is what the analysis uses as each downstream model's *offered* rate,
    #: since the crop count is data-dependent and cannot be derived from the config.
    requests_total: dict[str, float] = field(default_factory=dict)
    requests_rejected: dict[str, float] = field(default_factory=dict)
    #: model -> device -> requests executed. The per-device breakdown a PR needs.
    per_device: dict[str, dict[str, int]] = field(default_factory=dict)
    instances: dict[str, int] = field(default_factory=dict)
    ops: str = ""
    stages: tuple[str, ...] = ()
    stages_missing: tuple[str, ...] = ()

    @property
    def offered(self) -> dict[str, float | None]:
        """Offered rate per module, for the analysis. Filled by the driver."""
        return {}


def _cameras(config: BenchConfig) -> list[dict[str, Any]]:
    """One camera per source, half on person frames and half on ship frames.

    The split mirrors the baseline exactly: it pushes ``person_2K`` through the detector and
    ``ship_2K`` through the segmenter, so the mix of content — and therefore the number of
    crops the detector produces — has to be the same or the downstream load is not the same
    experiment.

    With ``source == "rtsp"`` the same split is served over a real socket instead of read off
    disk. That is a **different experiment**, not a slower one: replay measures the inference
    plane with the decode path removed, while RTSP includes NVDEC, the jitter buffer and the
    NV12 conversion the deployment actually pays for. `config.as_dict()` records which was
    used, because the two numbers must never be compared as though they were the same
    measurement.
    """
    resolved = config.resolved()
    half = config.cameras // 2
    if config.source == "rtsp":
        return _rtsp_cameras(config)
    cameras: list[dict[str, Any]] = []
    for index in range(config.cameras):
        folder = resolved.person_frames if index < half else resolved.ship_frames
        cameras.append(
            {
                "camera_id": f"cam{index:02d}",
                "uri": str(folder),
                "source": "replay",
                "fps": config.fps,
                "loop": True,
            }
        )
    return cameras


def _rtsp_cameras(config: BenchConfig) -> list[dict[str, Any]]:
    """Cameras pointed at the local RTSP servers, in the same person/ship split as replay.

    Two servers, because `scripts/rtsp_serve.py --data` serves one directory: the person
    frames on ``config.rtsp_port`` and the ship frames on the next port
    (:func:`benchmarks.harness.rtsp.ship_port`). The first version pointed every camera at one
    server fed with person frames, so an RTSP run starved the ship branch of the graph and the
    analysis named the detector as the binding module whatever the truth was — the content
    split decides the crop fan-out, and the crop fan-out is the downstream load.

    The URIs are built by `scripts/rtsp_serve.stream_uri`, not by string-formatting them
    here: the server owns the path layout, and a benchmark that guessed it would fail as a
    connection refusal minutes into a run rather than as a mistake at start-up.
    """
    from benchmarks.harness.rtsp import ship_port
    from scripts.rtsp_serve import stream_uri

    half = config.cameras // 2
    cameras: list[dict[str, Any]] = []
    for index in range(config.cameras):
        if index < half:
            uri = stream_uri(index, port=config.rtsp_port, host="127.0.0.1")
        else:
            uri = stream_uri(index - half, port=ship_port(config), host="127.0.0.1")
        cameras.append(
            {
                "camera_id": f"cam{index:02d}",
                "uri": uri,
                # Left unset so the ingest registry resolves it the way production does — the
                # point of an RTSP run is to exercise the real decoder selection, not to pin it.
                "fps": config.fps,
            }
        )
    return cameras


def _settings(config: BenchConfig) -> Any:
    """Build ``ServerSettings`` for this run. Imported lazily — see :func:`run_shipinfer`."""
    from shipinfer.core.settings import ServerSettings

    resolved = config.resolved()
    return ServerSettings(
        model_repository=resolved.model_repository,
        load_all_models=True,
        # 0..n-1 *inside* the CUDA_VISIBLE_DEVICES restriction the caller already applied.
        devices={"visible_gpus": list(range(len(config.gpus)))},
        ingest={"cameras": _cameras(config)},
        pipeline={
            "workers": config.pipeline_workers,
            # The same depth the baseline's queue has, so "buffer growth" means the same
            # thing on both sides. The pipeline's own default is 256 with REJECT, which is
            # the right production choice and the wrong measurement one: it converts growth
            # into a drop count and the growth rate would be pinned at zero by the bound.
            "queue_capacity": config.buffer_capacity,
            "result_sink": "null",
        },
    )


#: The last run's `PipelineMetrics`, for `benchmarks/stages.py` to read.
#:
#: A module global rather than a return value because `ShipInferResult` is a frozen record of
#: *counts*, and the profile wants the live histogram objects — `quantile` and `snapshot` are
#: methods, not numbers, and copying every bucket into the result would duplicate the metrics
#: registry in a dataclass. One run at a time is the only mode this harness has (it owns
#: `CUDA_VISIBLE_DEVICES` for the process), so a single slot cannot be raced.
_LAST_METRICS: Any = None


def last_pipeline_metrics() -> Any:
    """The `PipelineMetrics` of the most recent :func:`run_shipinfer`, or None."""
    return _LAST_METRICS


def run_shipinfer(
    config: BenchConfig,
    out_dir: Path | None = None,
    *,
    startup_timeout_s: float = 1800.0,
) -> ShipInferResult:
    """Start the server and the pipeline, hold the load, and write the occupancy log.

    Every ``shipinfer`` import is inside this function. Importing the server pulls in torch
    and TensorRT, and the harness's config, analysis and sampler modules have to stay
    importable on a machine with neither — that is what lets the analysis be unit-tested
    offline, which is where the correctness of the measurement is actually pinned.

    Args:
        config: the load. ``CUDA_VISIBLE_DEVICES`` must already be set from
            :meth:`BenchConfig.cuda_visible_devices` — torch reads it once, at first CUDA
            call, so setting it here would be too late.
        out_dir: where the log lands. Defaults to ``config.out_dir / "shipinfer"``.
        startup_timeout_s: how long the server may take to become ready. Generous on
            purpose: this repository expands to two dozen TensorRT instances and each one
            deserialises its own plan, which is minutes rather than seconds.

    Raises:
        RuntimeError: ``CUDA_VISIBLE_DEVICES`` disagrees with ``config.gpus``, which would
            silently benchmark a different number of GPUs than the baseline got.
    """
    expected = config.cuda_visible_devices()
    actual = os.environ.get("CUDA_VISIBLE_DEVICES")
    if actual != expected:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES is {actual!r} but this run is configured for {expected!r}. "
            f"Set it before the process starts; torch reads it once"
        )

    from shipinfer.core.logging import configure
    from shipinfer.ingest import IngestManager
    from shipinfer.pipeline import PipelineRunner
    from shipinfer.server import InferenceServer

    configure(os.environ.get("SHIPINFER_BENCH_LOG", "WARNING"), force=True)

    config = config.resolved()
    config.require_inputs()
    out = out_dir or config.out_dir / "shipinfer"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "buffers.jsonl"

    settings = _settings(config)
    started_at = time.monotonic()
    server = InferenceServer(settings)
    server.start(timeout_s=startup_timeout_s) if _accepts_timeout(server) else server.start()
    startup_s = time.monotonic() - started_at

    # The manager is captured rather than left anonymous inside the factory: its
    # `frames_read`/`frames_dropped` are the only measurement of what load was actually
    # delivered, and the reported throughput is derived from that rather than from the
    # configured target.
    created: dict[str, Any] = {}

    def make_manager(sink: Any) -> Any:
        manager = IngestManager(sink, settings=settings.ingest)
        created["manager"] = manager
        return manager

    runner = PipelineRunner(server, settings=settings, frames=make_manager)
    global _LAST_METRICS
    _LAST_METRICS = runner.metrics
    try:
        runner.start()
        handles = {name: server.model(name) for name in GRAPH_MODELS if name in server.models()}
        missing_models = tuple(m for m in GRAPH_MODELS if m not in handles)
        stages = tuple(runner.graph.stage_names)
        stages_missing = ("tracking", "fused_kernels", *(f"model:{m}" for m in missing_models))

        def probe() -> dict[str, int]:
            row = {f"{PIPELINE_MODULE}{BUFFER_SUFFIX}": runner.queue.depth}
            for name, handle in handles.items():
                row[f"{name}{BUFFER_SUFFIX}"] = handle.total_depth
            return row

        meta = {
            "system": "shipinfer",
            "config": config.as_dict(),
            "startup_s": round(startup_s, 1),
            "stages": list(stages),
            "stages_missing": list(stages_missing),
            "ops": runner.health()["ops"],
            "instances": {n: len(h.instances) for n, h in handles.items()},
            "note": (
                "tracking and the fused kernels are NOT in this measurement; see "
                "benchmarks/harness/shipinfer.py for what that costs the comparison"
            ),
        }

        def counters() -> dict[str, Any]:
            """Every cumulative counter the analysis rates, read at one instant."""
            manager = created.get("manager")
            stats = manager.summary() if manager is not None else None
            return {
                "read": int(getattr(stats, "frames_read", 0)),
                "dropped": int(getattr(stats, "frames_dropped", 0)),
                "emitted": int(runner.sink.emitted),
                "accepted": int(runner.frames_accepted),
                "requests": {n: server.metrics.requests_total.value(model=n) for n in handles},
                # Per-bucket, so the boundary snapshot can be subtracted from the end one.
                "stages": {
                    s: read_cell(runner.metrics.stage_latency_us, stage=s) for s in stages
                },
            }

        sampler = OccupancySampler(log, probe, interval_s=config.sample_interval_s, meta=meta)
        window_started = time.monotonic()
        # Taken inside the sampler window at the same boundary the fit uses, so the offered
        # rate and the growth rate describe the same seconds. Sampled by polling rather than
        # by a timer thread: one more thread contending for this interpreter is exactly the
        # kind of thing that shows up as a shortfall in the load generator.
        warmup_at = window_started + config.warmup_s
        at_warmup: dict[str, Any] | None = None
        with sampler:
            deadline = window_started + config.seconds
            while time.monotonic() < deadline:
                if at_warmup is None and time.monotonic() >= warmup_at:
                    at_warmup = counters()
                    warmup_taken = time.monotonic()
                time.sleep(0.2)
        elapsed = time.monotonic() - window_started
        at_end = counters()
        whole_run = at_warmup is None
        if at_warmup is None:
            # A run shorter than its own warmup. Nothing to subtract, and saying so is
            # better than silently rating the whole window and calling it steady.
            at_warmup = {
                "read": 0,
                "dropped": 0,
                "emitted": 0,
                "accepted": 0,
                "requests": {},
                "stages": {},
            }
            warmup_taken = window_started
        steady_s = max(0.0, time.monotonic() - warmup_taken)

        manager = created.get("manager")
        ingest_stats = manager.summary() if manager is not None else None
        metrics = server.metrics
        requests = {n: metrics.requests_total.value(model=n) for n in handles}
        rejected = {n: metrics.requests_rejected.value(model=n) for n in handles}
        per_device: dict[str, dict[str, int]] = {}
        for name, handle in handles.items():
            breakdown: dict[str, int] = {}
            for instance in handle.instances:
                stats = instance.stats()
                device = str(stats["device"])
                breakdown[device] = breakdown.get(device, 0) + int(stats["requests"])
            per_device[name] = breakdown

        return ShipInferResult(
            log=log,
            startup_s=startup_s,
            elapsed_s=elapsed,
            frames_read=getattr(ingest_stats, "frames_read", 0),
            frames_dropped=getattr(ingest_stats, "frames_dropped", 0),
            steady_s=steady_s,
            steady_frames_read=at_end["read"] - at_warmup["read"],
            steady_frames_dropped=at_end["dropped"] - at_warmup["dropped"],
            steady_requests_total={
                n: at_end["requests"].get(n, 0.0) - at_warmup["requests"].get(n, 0.0)
                for n in at_end["requests"]
            },
            steady_events_emitted=at_end["emitted"] - at_warmup["emitted"],
            steady_frames_accepted=at_end["accepted"] - at_warmup["accepted"],
            steady_stage_latency={
                s: cell.minus(at_warmup["stages"].get(s))
                for s, cell in at_end["stages"].items()
            },
            steady_is_whole_run=whole_run,
            frames_accepted=runner.frames_accepted,
            events_emitted=runner.sink.emitted,
            requests_total=requests,
            requests_rejected=rejected,
            per_device=per_device,
            instances={n: len(h.instances) for n, h in handles.items()},
            ops=str(runner.health()["ops"]),
            stages=stages,
            stages_missing=stages_missing,
        )
    finally:
        # Order matters and both must happen even if the run raised: the pipeline owns the
        # cameras and the workers, the server owns the CUDA contexts, and a leaked context
        # holds VRAM on a shared box until the process dies.
        try:
            runner.stop()
        finally:
            server.stop()
            _release_cuda()


def _accepts_timeout(server: Any) -> bool:
    """Whether this ``InferenceServer.start`` takes a timeout. Version tolerance, not magic."""
    import inspect

    try:
        return "timeout_s" in inspect.signature(server.start).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False


def _release_cuda() -> None:
    """Hand the caching allocator's blocks back to the driver.

    This box is shared and its VRAM is recorded continuously, so a benchmark that keeps a
    couple of gigabytes cached after it has finished measuring is a benchmark that gets
    killed by an operator. Failures are ignored on purpose: there is nothing useful to do
    about them during teardown.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - teardown must not raise
        pass


def achieved_offer(config: BenchConfig, result: ShipInferResult) -> float:  # noqa: ARG001
    """Images per second the generator actually delivered into the pipeline.

    Read from ingest's own counters rather than from the configuration. `frames_read` counts
    what the cameras produced and `frames_dropped` what backpressure refused; their sum over
    the elapsed window is the load the system was really offered, and it is the only figure
    a throughput may be derived from.

    Falls back to the accepted count when ingest reported nothing, and to zero when there
    is no measurement at all — a case :func:`check_offer` refuses rather than papers over.

    ``config`` is unused and deliberately kept: it is what this function must *not* read.
    Falling back to `config.offered_total` made the measurement equal the thing it was being
    compared against, so the tolerance test could never fire. The signature stays uniform
    with `offered_rates` and `check_offer`, and the parameter is now a reminder.
    """
    # The *steady* window, matching the fit. Rating cumulative counters over the whole run
    # while fitting growth over the steady part alone measures two different things: with
    # cameras staggering their first decode and instances deserialising engines through
    # `t < warmup_s`, a clean 1000 img/s run read 60000/70 = 857, which either tripped the
    # check below or understated capacity by 143 img/s in `offered - growth`.
    window = result.steady_s if result.steady_s > 0 else result.elapsed_s
    if window <= 0:
        return 0.0
    # `frames_read` minus what backpressure refused. A dropped frame never enters the
    # queue, so it cannot grow the buffer and cannot be retired — counting it as offered
    # while `sustained = offered - growth` reads a flat buffer would report the whole
    # offered rate as throughput. Concretely: read 1000/s, reject 700/s, retire 300/s, and
    # the run says SUSTAINED at 1000 img/s. A 3.3x overstatement, and one-sided, because
    # the baseline is a separate binary that is not measured this way.
    if result.steady_s > 0:
        entered = max(0, result.steady_frames_read - result.steady_frames_dropped)
    else:
        entered = max(0, result.frames_read - result.frames_dropped)
        if not entered:
            entered = result.frames_accepted
    # Zero, not the target. Falling back to `offered_total` made the measurement equal the
    # thing it was being compared against, so `check_offer`'s tolerance test could never
    # fire: a run in which nothing was read, nothing accepted and nothing emitted reported
    # 1000 img/s SUSTAINED off an all-zero buffer log. That is not hypothetical — it is the
    # 50-camera run in which every per-device counter stayed at zero.
    return entered / window if entered else 0.0


#: Instances per GPU, as the four ``model_repository/*/config.yaml`` files declare them.
#: A **fallback only** — :func:`per_module_capacity` prefers the count the run actually
#: started, because this table is a second copy of a fact that lives somewhere else and will
#: be wrong the first time somebody edits a config without editing this.
DEFAULT_INSTANCES_PER_GPU = {
    "ship_detector": 2,
    "ship_segmenter": 1,
    "ship_embedder": 1,
    "person_embedder": 2,
}


def per_module_capacity(
    config: BenchConfig,
    settings: Any = None,
    instances: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """The bound each module's buffer actually has.

    One capacity for every module was wrong, and wrong in the dangerous direction. The
    analysis uses it to spot a *plateau* -- a buffer sitting at its bound stops growing, so
    its slope stops meaning anything -- and ``buffer_capacity`` (65536) is the pipeline
    queue's bound. A model instance's queue is ``scheduler.max_queue_size``, 64 by default,
    so an eight-instance detector plateaus at 512: two orders of magnitude below the number
    the guard compared against, which is why it could never trip for the queues that
    saturate first.

    Args:
        instances: model -> how many instances the *run* actually started. Pass it. Without
            it this falls back to :data:`DEFAULT_INSTANCES_PER_GPU`, which is a transcription
            of the four ``config.yaml`` files and therefore a second copy of a fact that
            already exists — review's finding: set ``ship_detector`` to ``count: 1`` and the
            real bound becomes 256 while the guard still compares against 512, so a queue
            pegged at its bound reads SUSTAINED and its offered rate publishes as throughput.
            The caller holds `len(handle.instances)` and even writes it into the log's
            metadata, so there is no reason to guess.
    """
    from shipinfer.core.settings import ServerSettings

    resolved = settings or ServerSettings()
    per_instance = int(getattr(resolved.scheduler, "max_queue_size", 64))
    capacities = {PIPELINE_MODULE: int(config.buffer_capacity)}
    # Keyed by the modules this driver actually *samples*, which are `GRAPH_MODELS`.
    # `instances_per_gpu` is the *baseline's* vocabulary — `det` and `seg` — so keying off it
    # produced a mapping the ShipInfer log never matched: every model queue got `None`,
    # `capped` was permanently False, and a detector queue sitting at its 512-deep bound
    # read SUSTAINED. That is precisely the failure this function was written to fix, landed
    # on the wrong keys.
    for name in GRAPH_MODELS:
        if instances is not None and name in instances:
            count = int(instances[name])
        else:
            count = DEFAULT_INSTANCES_PER_GPU.get(name, 1) * len(config.gpus)
        capacities[name] = per_instance * count
    return capacities


def reconcile(result: ShipInferResult, claimed: float) -> None:
    """Refuse a claimed rate that far exceeds what came out of the pipeline.

    Two independent estimates of one quantity: ``offered - growth`` from the buffer log, and
    ``events_emitted / elapsed`` from the far end. The first can be fooled -- a scheduler
    that *refuses* work rather than queueing it leaves every buffer flat, so growth reads
    zero and the entire offered rate publishes as sustained. The second cannot be: an event
    either came out or it did not.

    Raises:
        RuntimeError: the claim exceeds the emitted rate by more than
            ``MAX_RECONCILE_GAP``. Deliberately not clamped to the emitted rate -- silently
            substituting one number for the other would hide that they disagreed, which is
            the fact worth surfacing.
    """
    # The steady window again: `claimed` is derived from it, and comparing it against a
    # rate averaged over a run that includes start-up would forgive a real disagreement.
    steady = result.steady_s > 0
    window = result.steady_s if steady else result.elapsed_s
    events = result.steady_events_emitted if steady else result.events_emitted
    if window <= 0 or not events:
        return
    emitted = events / window
    if emitted <= 0 or claimed <= emitted * (1.0 + MAX_RECONCILE_GAP):
        return
    raise RuntimeError(
        f"the buffer log implies {claimed:.1f} img/s sustained but only {emitted:.1f} img/s "
        f"of events came out of the pipeline ({events} events in "
        f"{window:.1f}s). Work that is refused rather than queued leaves every "
        f"buffer flat, so growth reads zero and the offered rate publishes as throughput. "
        f"One of these two numbers is wrong; the run does not support a throughput claim."
    )


def check_offer(
    config: BenchConfig, result: ShipInferResult, *, tolerance: float = 0.98
) -> None:
    """Refuse a run whose generator never delivered the load being reported on.

    Raises:
        RuntimeError: the achieved offer is below ``tolerance`` of the target. This is not a
            warning because the number it protects is the headline: a run offered 400 img/s
            and reported against 1000 prints a sustained rate it never sustained, in the
            direction that flatters us.
    """
    achieved = achieved_offer(config, result)
    target = float(config.offered_total)
    if achieved <= 0.0:
        raise RuntimeError(
            f"ingest reported no delivered frames at all (read={result.frames_read}, "
            f"accepted={result.frames_accepted}, emitted={result.events_emitted}) over "
            f"{result.elapsed_s:.1f}s. An all-zero buffer log is flat for the worst possible "
            f"reason, and a flat log is what the growth fit reads as 'keeping up'."
        )
    rejected = sum(result.requests_rejected.values())
    offered_requests = max(1.0, target * max(result.elapsed_s, 1.0))
    if rejected / offered_requests > MAX_REJECT_FRACTION:
        raise RuntimeError(
            f"the scheduler refused {rejected:.0f} request(s) against roughly "
            f"{offered_requests:.0f} offered. A rejected request never enters a queue, so it "
            f"cannot grow one — every buffer stays flat and the offered rate would publish "
            f"as sustained throughput. Lower the offered rate to what the GPUs retire."
        )
    read = result.frames_read
    if read and result.frames_dropped / read > MAX_DROP_FRACTION:
        raise RuntimeError(
            f"backpressure refused {result.frames_dropped / read:.0%} of the "
            f"{read} frames ingest read. Past {MAX_DROP_FRACTION:.0%} the run is measuring "
            f"a shedding system rather than a serving one, and the sustained figure would "
            f"describe a load most of which never entered the pipeline."
        )
    if target > 0 and achieved < tolerance * target:
        raise RuntimeError(
            f"the load generator delivered {achieved:.1f} img/s against a target of "
            f"{target:.0f} — {achieved / target:.0%} of it. A throughput measured against a "
            f"load that was never offered is not a measurement.\n\n"
            f"Measured on this host: 20 replay cameras at a 10 fps target deliver ~87 img/s. "
            f"Caching the decoded frames moved that only from 77 to 87, so the wall is not "
            f"decoding — it is one interpreter running the camera threads and the pipeline "
            f"workers together. The architecture document puts ingest in separate *processes* "
            f"for exactly this reason (new-system-architecture.md section 9, '[Decode "
            f"procs]'), and this driver runs it in-process.\n\n"
            f"So: lower --cameras/--fps to a rate this host can generate, or run the "
            f"generator as several processes. Do not raise the tolerance."
        )


def offered_rates(config: BenchConfig, result: ShipInferResult) -> dict[str, float | None]:
    """Offered images per second per module, for :func:`analysis.analyse`.

    The ingest queue and the detector are fed by the config: every camera frame becomes one
    detector request, so both are ``cameras x fps``. The three downstream models are fed by
    however many crops the detector found, which is a property of the *images*, not of the
    configuration — so it is measured from ``requests_total`` over the run rather than
    asserted. A model that never received a request gets ``None``: reporting 0 offered would
    make its sustained throughput look like a perfect 0 - 0.
    """
    # Measured, not configured. `config.offered_total` is what we *asked* for; a starved
    # generator makes the queue flat for the wrong reason and the analysis would then read
    # `offered - 0` and publish the target as though it were a throughput.
    frame_rate = achieved_offer(config, result)
    rates: dict[str, float | None] = {
        PIPELINE_MODULE: frame_rate,
        "ship_detector": frame_rate,
    }
    # Same window as `achieved_offer` and as the fit. `requests_total` is cumulative from
    # process start, so dividing it by the whole run charged every downstream model for the
    # warmup it spent waiting on engines that had not finished deserialising.
    steady = result.steady_s > 0
    window = result.steady_s if steady else result.elapsed_s
    source = result.steady_requests_total if steady else result.requests_total
    for name in ("ship_segmenter", "ship_embedder", "person_embedder"):
        total = source.get(name, 0.0)
        rates[name] = total / window if total and window > 0 else None
    return rates
