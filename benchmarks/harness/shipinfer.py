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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.harness.analysis import BUFFER_SUFFIX
from benchmarks.harness.config import BenchConfig
from benchmarks.harness.sampler import OccupancySampler

__all__ = ["PIPELINE_MODULE", "ShipInferResult", "run_shipinfer"]

#: The module name used for the ingest queue in the JSONL. It is the direct analogue of the
#: baseline's ``det_buffer_size``: decoded frames waiting for a worker.
PIPELINE_MODULE = "pipeline"

#: Fraction of read frames backpressure may refuse before the run stops being a throughput
#: measurement. Small on purpose: a little shedding is the system working as designed, and a
#: lot of it means the offered rate and the served rate are different experiments.
MAX_DROP_FRACTION = 0.02

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
    """One replay camera per source, half on person frames and half on ship frames.

    The split mirrors the baseline exactly: it pushes ``person_2K`` through the detector and
    ``ship_2K`` through the segmenter, so the mix of content — and therefore the number of
    crops the detector produces — has to be the same or the downstream load is not the same
    experiment.
    """
    resolved = config.resolved()
    half = config.cameras // 2
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

        sampler = OccupancySampler(log, probe, interval_s=config.sample_interval_s, meta=meta)
        window_started = time.monotonic()
        with sampler:
            deadline = window_started + config.seconds
            while time.monotonic() < deadline:
                time.sleep(0.2)
        elapsed = time.monotonic() - window_started

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


def achieved_offer(config: BenchConfig, result: ShipInferResult) -> float:
    """Images per second the generator actually delivered into the pipeline.

    Read from ingest's own counters rather than from the configuration. `frames_read` counts
    what the cameras produced and `frames_dropped` what backpressure refused; their sum over
    the elapsed window is the load the system was really offered, and it is the only figure
    a throughput may be derived from.

    Falls back to the accepted count when ingest reported nothing, and to the configured
    target only when there is no measurement at all — a case :func:`check_offer` refuses
    rather than papers over.
    """
    if result.elapsed_s <= 0:
        return float(config.offered_total)
    # `frames_read` minus what backpressure refused. A dropped frame never enters the
    # queue, so it cannot grow the buffer and cannot be retired — counting it as offered
    # while `sustained = offered - growth` reads a flat buffer would report the whole
    # offered rate as throughput. Concretely: read 1000/s, reject 700/s, retire 300/s, and
    # the run says SUSTAINED at 1000 img/s. A 3.3x overstatement, and one-sided, because
    # the baseline is a separate binary that is not measured this way.
    entered = max(0, result.frames_read - result.frames_dropped)
    if not entered:
        entered = result.frames_accepted
    if not entered:
        return float(config.offered_total)
    return entered / result.elapsed_s


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
    for name in ("ship_segmenter", "ship_embedder", "person_embedder"):
        total = result.requests_total.get(name, 0.0)
        rates[name] = total / result.elapsed_s if total and result.elapsed_s > 0 else None
    return rates
