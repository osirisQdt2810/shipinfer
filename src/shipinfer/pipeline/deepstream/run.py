"""One shard's DeepStream process: build the graph, run the loop, publish the events.

WHAT THIS OWNS AND WHAT IT DOES NOT
-----------------------------------
It owns a GStreamer pipeline, a ``GLib.MainLoop``, a bus watch and a pad probe. It does not
own batching, placement, reassembly or a model table: inside the graph those are NVIDIA's, and
that is exactly the trade this topology exists to measure. What it keeps from the rest of the
project is the two ends — :class:`~shipinfer.repository.ModelRepository` decides what the GIEs
run (see :mod:`.configs`) and :class:`~shipinfer.pipeline.sinks.ResultSink` decides where the
results go — because a comparison between two architectures is only a comparison if both
publish the same event to the same place.

THE PROBE IS A STREAMING THREAD, AND IT MUST NOT RAISE
------------------------------------------------------
:meth:`MetadataProbe.on_buffer` is called by GStreamer on the thread pushing buffers through
the graph. An exception there does not propagate anywhere useful: the C caller swallows it, the
buffer is dropped, and the deployment loses frames with nothing in the log. So it catches
everything, counts it against ``pipeline_build_failures_total``, and always returns
``PadProbeReturn.OK``. That is the same bug class as a result sink raising into a pipeline
worker (`pipeline/sinks/base.py`), and it is answered the same way.

THE EMISSION DISCIPLINE IS THE RUNNER'S, DELIBERATELY COPIED
------------------------------------------------------------
:meth:`MetadataProbe._emit` is :meth:`shipinfer.pipeline.runner.PipelineRunner._emit_resolved`
and :meth:`~shipinfer.pipeline.runner.PipelineRunner._record` with the futures and the
reassembly removed — same order, same counters, same reasons. ``emit`` returns a ``bool`` and
never raises, so a dropped event has to be a *return-value* check; treating it as an exception
is what made ``pipeline_sink_failures_total`` unreachable once already, and a broker whose DNS
had stopped resolving dropped every event while ``frames_emitted`` climbed at full rate.
Asynchronous refusals are drained afterwards, with the tag they belong to, so one camera's
publish loss is never charged to another's frame.
"""

from __future__ import annotations

import signal
import tempfile
import time
from pathlib import Path
from typing import Any

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.settings import CameraConfig, ServerSettings
from shipinfer.pipeline.deepstream.builder import build_branch
from shipinfer.pipeline.deepstream.configs import GeneratedConfigs, write_configs
from shipinfer.pipeline.deepstream.probe import (
    FrameGeometry,
    FrameNumbering,
    FrameView,
    build_event,
    walk_batch,
)
from shipinfer.pipeline.metrics import PipelineMetrics
from shipinfer.pipeline.schema import PerceptionEvent
from shipinfer.pipeline.sinks import RESULT_SINKS, ResultSink
from shipinfer.repository import ModelRepository
from shipinfer.runtime.gstreamer import load_pyds

__all__ = ["PR1_MISSING_STAGES", "DeepStreamPipeline", "MetadataProbe"]

_LOG = get_logger("pipeline.deepstream")

#: The stages this topology does not run yet, named on every event it publishes. The Python
#: DAG segments ships and recognises their identity; PR1's DeepStream graph is detector,
#: tracker and the two embedders. An event that stayed silent about the difference would be a
#: partial frame published as a complete one — the exact distinction `missing_stages` exists to
#: make (ADR-005). See `docs/design/topology-deepstream.md` for how each one lands.
PR1_MISSING_STAGES: tuple[str, ...] = ("ship_segmenter", "ship_recognizer")


class MetadataProbe:
    """The pad probe: one batch of metadata in, one event per frame out.

    Split from :class:`DeepStreamPipeline` because it is the part with behaviour worth testing
    and no GStreamer in it — given a fake ``gst`` and a fake ``pyds`` it runs, publishes and
    counts exactly as it does on a DeepStream host.
    """

    def __init__(
        self,
        *,
        gst: Any,
        pyds: Any,
        sink: ResultSink,
        metrics: PipelineMetrics,
        settings: ServerSettings,
        camera_by_pad: dict[int, str],
        cameras: dict[str, CameraConfig],
        missing_stages: tuple[str, ...] = PR1_MISSING_STAGES,
    ) -> None:
        self._gst = gst
        self._pyds = pyds
        self._sink = sink
        self._metrics = metrics
        self._settings = settings
        self._deepstream = settings.topology.deepstream
        self._camera_by_pad = camera_by_pad
        self._cameras = cameras
        self._missing_stages = missing_stages
        self._numbering = FrameNumbering()
        # Read once. The graph stamps wall-clock capture times and `PerceptionEvent.build`
        # measures latency from the monotonic clock; sampling the offset per frame would make
        # every event's latency carry its own clock-read jitter.
        self._epoch_offset_ns = time.time_ns() - time.monotonic_ns()
        #: Frames whose muxer pad is not in the camera map. Counted rather than indexed into:
        #: a KeyError here would be an exception on a streaming thread.
        self.unknown_pad_frames = 0

    @property
    def metrics(self) -> PipelineMetrics:
        """The handles this probe charges. The pipeline's own, so one dashboard covers both."""
        return self._metrics

    def on_buffer(self, _pad: Any, info: Any) -> Any:
        """The probe callback. Always returns ``PadProbeReturn.OK``, whatever happened."""
        ok = self._gst.PadProbeReturn.OK
        try:
            buffer = info.get_buffer()
            if buffer is None:
                return ok
            batch_meta = self._pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if batch_meta is None:
                return ok
            for view in walk_batch(self._pyds, batch_meta):
                self._publish(view)
        except Exception:
            # Not the sink's fault and not counted against it: this is our own metadata walk.
            self._metrics.build_failures.inc(camera="unknown")
            _LOG.exception("the DeepStream metadata probe failed on one buffer")
        return ok

    def _publish(self, view: FrameView) -> None:
        camera_id = self._camera_by_pad.get(view.pad_index)
        if camera_id is None:
            # A muxer pad nobody claimed: skip it and count it. The alternative — indexing the
            # map — is a KeyError on a streaming thread, which loses every *other* camera's
            # frames in the same batch.
            self.unknown_pad_frames += 1
            self._metrics.build_failures.inc(camera="unknown")
            _LOG.warning(
                "frame on muxer pad %d belongs to no configured camera", view.pad_index
            )
            return
        camera = self._cameras.get(camera_id)
        try:
            event = build_event(
                view,
                camera_id=camera_id,
                source_id=self._settings.pipeline.source_id,
                labels=self._settings.pipeline.class_labels,
                geometry=FrameGeometry(
                    mux_width=self._deepstream.mux_width,
                    mux_height=self._deepstream.mux_height,
                    source_width=view.source_width,
                    source_height=view.source_height,
                    padded=self._deepstream.mux_enable_padding,
                ),
                fps=camera.fps if camera is not None else 0.0,
                frame_id=self._numbering.next(camera_id, view.frame_num),
                epoch_offset_ns=self._epoch_offset_ns,
                missing_stages=self._missing_stages,
            )
        except Exception:
            self._metrics.build_failures.inc(camera=camera_id)
            _LOG.exception("failed to build the event for camera %s", camera_id)
            return
        self._emit(camera_id, event)

    def _emit(self, camera_id: str, event: PerceptionEvent) -> None:
        """Publish one event. `PipelineRunner`'s discipline, without the futures."""
        if not self._sink.emit(event):
            self._metrics.sink_failures.inc(sink=self._sink.name)
            return
        # Refusals the transport reported since the last emit, counted with the tag they belong
        # to and deliberately after this frame is settled.
        for camera, frame in self._sink.drain_delivery_failures():
            self._metrics.sink_failures.inc(sink=self._sink.name)
            _LOG.error(
                "sink %s did not deliver camera %s frame %d",
                self._sink.name,
                camera,
                frame,
                extra=log_context(camera_id=camera),
            )
        self._metrics.frames_emitted.inc(camera=camera_id)
        self._metrics.objects_per_frame.observe(len(event.objects), camera=camera_id)
        for record in event.objects:
            self._metrics.objects_total.inc(camera=camera_id, object_class=record.class_name)
        if event.latency_us:
            self._metrics.frame_latency_us.observe(event.latency_us, camera=camera_id)


class DeepStreamPipeline:
    """One shard's graph, from the model repository to the result sink.

    Args:
        settings: the whole tree, already narrowed to this shard's cameras by
            :func:`shipinfer.cli.common.build_settings` — the same ``SHIPINFER_SHARD_CAMERAS``
            every other topology's child is narrowed by, so one fleet description drives all of
            them.
        sink: where events go. ``None`` builds the one ``pipeline.result_sink`` names, which is
            how a deployment configures it; injected in tests and in a harness.
        config_root: where the generated nvinfer files are written. ``None`` takes
            ``topology.deepstream.config_dir`` (which the launcher sets for every child), and
            failing that a per-run directory under ``$TMPDIR``.
    """

    def __init__(
        self,
        settings: ServerSettings,
        *,
        sink: ResultSink | None = None,
        config_root: Path | None = None,
    ) -> None:
        self._settings = settings
        self._deepstream = settings.topology.deepstream
        self._cameras = [c for c in settings.ingest.cameras if c.enabled]
        if not self._cameras:
            raise ConfigurationError(
                "this configuration defines no enabled cameras, so there is nothing for a "
                "DeepStream graph to decode. `shipinfer fleet` narrows each shard's cameras "
                "through SHIPINFER_SHARD_CAMERAS; check that this shard was given some"
            )
        self._shard_index = self._deepstream.shard or 0
        self._gpu_id = _single_gpu(settings)
        self._config_root = Path(
            config_root
            if config_root is not None
            else (
                self._deepstream.config_dir
                if self._deepstream.config_dir is not None
                else Path(tempfile.gettempdir())
                / f"shipinfer-ds-{self._deepstream.run_id or 'local'}"
            )
        )
        self._metrics = PipelineMetrics()
        # The sink is built at start(), never here: both shipped sinks side-effect in
        # __init__ (jsonlines truncates/opens its file, kafka builds a Producer), and this
        # object is also constructed on control boxes for --dry-run, whose contract is
        # "reads a repository and writes text" (#32 round 1 reproduced a dry run truncating
        # a live results file).
        self._injected_sink = sink
        self._sink: ResultSink | None = None
        self._gst: Any = None
        self._glib: Any = None
        self._pyds: Any = None
        self._pipeline: Any = None
        self._loop: Any = None
        self._probe: MetadataProbe | None = None
        self._configs: GeneratedConfigs | None = None
        self._exit_code = 0

    # -- introspection -------------------------------------------------------------------

    @property
    def metrics(self) -> PipelineMetrics:
        return self._metrics

    @property
    def sink(self) -> ResultSink:
        return self._ensure_sink()

    @property
    def gpu_id(self) -> int:
        return self._gpu_id

    @property
    def shard_index(self) -> int:
        return self._shard_index

    @property
    def cameras(self) -> tuple[CameraConfig, ...]:
        return tuple(self._cameras)

    # -- configuration -------------------------------------------------------------------

    def write_configs(self, *, require_engine: bool = True) -> GeneratedConfigs:
        """Generate this shard's nvinfer, tracker and label files. Touches no hardware.

        Public because ``--dry-run`` is exactly this and nothing else: an operator on a control
        box wants to read what the graph would be given before fifty cameras start reconnecting.
        """
        repository = ModelRepository.load(self._settings.model_repository)
        return write_configs(
            repository,
            settings=self._settings,
            shard_index=self._shard_index,
            gpu_id=self._gpu_id,
            cameras=self._cameras,
            root=self._config_root,
            require_engine=require_engine,
        )

    # -- lifecycle -----------------------------------------------------------------------

    def _ensure_sink(self) -> ResultSink:
        if self._sink is None:
            self._sink = self._injected_sink or RESULT_SINKS.create(
                self._settings.pipeline.result_sink,
                **self._settings.pipeline.result_sink_options,
            )
        return self._sink

    def start(self) -> None:
        """Generate the configs, build the graph, and put it into PLAYING."""
        if self._pipeline is not None:
            raise ConfigurationError("this DeepStream pipeline is already started")
        # Typed and at start, never at import: every module in this package loads on a host
        # with neither GStreamer nor pyds, which is what keeps the tier above offline.
        self._gst, self._glib, self._pyds = load_pyds()
        self._configs = self.write_configs()
        _LOG.info(
            "shard %d: %d camera(s) on gpu %d, configs in %s",
            self._shard_index,
            len(self._cameras),
            self._gpu_id,
            self._configs.root,
        )

        self._pipeline = self._gst.Pipeline.new(f"shipinfer_ds_shard{self._shard_index}")
        branch = build_branch(
            self._gst,
            self._pipeline,
            cameras=self._cameras,
            gpu_id=self._gpu_id,
            configs=self._configs,
            deepstream=self._deepstream,
            ingest=self._settings.ingest,
        )
        probe = MetadataProbe(
            gst=self._gst,
            pyds=self._pyds,
            sink=self._ensure_sink(),
            metrics=self._metrics,
            settings=self._settings,
            camera_by_pad=dict(branch.camera_by_pad),
            cameras={c.camera_id: c for c in self._cameras},
        )
        self._probe = probe
        # The bound method, not a lambda over `self._probe`: the probe outlives this frame only
        # because the pad holds this reference, and a closure reading an attribute that a later
        # `stop()` could clear would be a NoneType call on a streaming thread.
        branch.probe_pad.add_probe(self._gst.PadProbeType.BUFFER, probe.on_buffer)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus)

        if self._pipeline.set_state(self._gst.State.PLAYING) == (
            self._gst.StateChangeReturn.FAILURE
        ):
            raise ConfigurationError(
                f"shard {self._shard_index}: the DeepStream pipeline refused to enter PLAYING"
            )

    def run(self) -> int:
        """Start if needed, then run the main loop until EOS, an error, or a signal.

        Returns:
            The process exit code. Non-zero when the bus reported an error, so
            :class:`~shipinfer.server.launcher.Fleet` raises
            :class:`~shipinfer.core.errors.ShardExitedError` for it and stops the rest of the
            fleet — the same failure story a dying ``serve`` shard has.
        """
        if self._pipeline is None:
            self.start()
        assert self._glib is not None
        self._loop = self._glib.MainLoop()
        previous = {
            number: signal.signal(number, self._on_signal)
            for number in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            self._loop.run()
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
            self.stop()
        return self._exit_code

    def stop(self) -> None:
        """NULL the graph and close the sink. Idempotent, and safe from a signal handler.

        NULL is what releases the decoder, the engines and the CUDA context. A process that
        exits without it leaves ~300 MiB per device held until the kernel reaps it, and this
        box is shared — so it is a ``finally``, not a happy path.
        """
        if self._loop is not None and self._loop.is_running():
            self._loop.quit()
        if self._pipeline is not None and self._gst is not None:
            self._pipeline.set_state(self._gst.State.NULL)
            self._pipeline = None
        # Only a sink that was ever built gets closed, and only once: a dry construction
        # never builds one, and stop() advertises idempotency.
        sink, self._sink = self._sink, None
        if sink is not None:
            sink.close()

    # -- the bus -------------------------------------------------------------------------

    def _on_bus(self, _bus: Any, message: Any) -> bool:
        gst = self._gst
        if message.type == gst.MessageType.ERROR:
            error, debug = message.parse_error()
            _LOG.error(
                "shard %d: %s (%s)", self._shard_index, error.message, debug or "no detail"
            )
            self._exit_code = 1
            self._quit()
        elif message.type == gst.MessageType.EOS:
            # Every source ended. For a file fixture that is success; for a live fleet it means
            # the cameras are gone, and the supervisor is what decides whether to restart.
            _LOG.info("shard %d: end of stream", self._shard_index)
            self._quit()
        elif message.type == gst.MessageType.WARNING:
            error, debug = message.parse_warning()
            _LOG.warning("shard %d: %s (%s)", self._shard_index, error.message, debug or "")
        return True

    def _on_signal(self, number: int, _frame: object) -> None:
        _LOG.info("shard %d: received signal %d; stopping", self._shard_index, number)
        self._quit()

    def _quit(self) -> None:
        if self._loop is not None:
            self._loop.quit()


def _single_gpu(settings: ServerSettings) -> int:
    """The one device this graph runs on, as *this process* numbers its devices.

    Under `shipinfer fleet` that is always 0: the launcher set ``CUDA_VISIBLE_DEVICES`` to the
    shard's physical ordinal before the interpreter started and told the child its logical view
    is ``[0]``. Run by hand with ``--gpus 3`` and nothing has remapped anything, so 3 is what
    every element in the graph should be given. Both are the same rule — the first device this
    process can see — which is why there is one line for them.
    """
    visible = list(settings.devices.visible_gpus or [])
    if len(visible) > 1:
        raise ConfigurationError(
            f"the `deepstream` topology runs one graph on one GPU, and this process was given "
            f"{visible}. Every element takes a single `gpu-id`, so a second device would hold a "
            f"CUDA context and no work. Start one shard per GPU (`shipinfer fleet --topology "
            f"deepstream --shards N`)"
        )
    return visible[0] if visible else 0
