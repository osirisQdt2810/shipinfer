"""Assembling the graph: sources, muxer, the GIEs, the tracker, and the pad the probe sits on.

The shape is the reference's (``mtmc_deepstream.py``) with three differences, each of which is
a failure mode that reference has.

**Every make and every link is checked, and the failure is typed.** The reference calls
``sys.exit`` when an element is missing. A shard is supervised by
:class:`~shipinfer.launch.Fleet`, which reports which shard died and stops the rest;
an exit code with a message on stderr tells it a shard exited, and a
:class:`~shipinfer.core.errors.SourceUnavailableError` naming ``nvinfer`` tells an operator to
install the SDK. Same outcome, one of them actionable.

**One branch per process, not many per process.** See
:class:`~shipinfer.server.topology.deepstream.DeepStreamTopology` for why: the child sees one
device because ``CUDA_VISIBLE_DEVICES`` was set before its interpreter started, so per-element
physical ``gpu-id`` values would name devices it cannot see.

**The old/new muxer split is detected rather than met at run time.** DeepStream 6.1 introduced
a second ``nvstreammux`` behind ``USE_NEW_NVSTREAMMUX=1``, and it has no ``batch-size``,
``width`` or ``height`` — the batch is decided by a config file instead. Setting a property
that does not exist raises ``TypeError`` from PyGObject, so :func:`build_branch` asks first and
refuses with the variable named.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from shipinfer.core.errors import ConfigurationError, SourceUnavailableError
from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.settings import CameraConfig, IngestSettings
from shipinfer.core.settings.runner import DeepStreamSettings
from shipinfer.pipeline.deepstream.configs import GeneratedConfigs, source_uri

__all__ = ["Branch", "build_branch", "make"]

_LOG = get_logger("pipeline.deepstream")


def make(gst: Any, factory: str, name: str) -> Any:
    """``Gst.ElementFactory.make``, or a typed refusal naming the element and the install.

    Raises:
        SourceUnavailableError: the plugin providing ``factory`` is not installed. The same
            class a missing PyGObject raises, and for the same reason: no amount of retrying
            installs a plugin, so the shard must fail out to its supervisor rather than loop.
    """
    element = gst.ElementFactory.make(factory, name)
    if element is None:
        raise SourceUnavailableError(
            "deepstream",
            f"GStreamer element {factory!r} (needed for {name!r}) is not installed. "
            f"nvurisrcbin / nvstreammux / nvinfer / nvtracker come with the DeepStream SDK; "
            f"check `gst-inspect-1.0 {factory}` inside the container "
            f"(see deploy/deepstream/image.sh)",
        )
    return element


@dataclass(frozen=True, slots=True)
class Branch:
    """One shard's graph, and the two things the caller needs from it afterwards."""

    mux: Any
    pgie: Any
    tracker: Any
    sgies: tuple[Any, ...]
    sink: Any
    #: Muxer sink-pad index -> camera id. The only way back from a frame to the camera that
    #: produced it: `NvDsFrameMeta` carries `pad_index`, not a name.
    camera_by_pad: Mapping[int, str]
    #: The src pad the metadata probe belongs on — after the last GIE, so every object already
    #: carries its class, its track id and its embedding when the probe reads it.
    probe_pad: Any


def build_branch(
    gst: Any,
    pipeline: Any,
    *,
    cameras: Sequence[CameraConfig],
    gpu_id: int,
    configs: GeneratedConfigs,
    deepstream: DeepStreamSettings,
    ingest: IngestSettings,
) -> Branch:
    """Build and link one shard's whole graph into ``pipeline``.

    ``nvurisrcbin`` per camera into one ``nvstreammux``, then the static chain
    ``mux -> pgie -> tracker -> sgie... -> fakesink``. The sink is a ``fakesink`` because
    nothing downstream wants the *frames*: they are decoded into device memory, inferred on
    there, and dropped. Only the metadata leaves, through the probe pad, which is what makes
    this topology's bus a few kilobytes a frame instead of six megabytes.

    Raises:
        SourceUnavailableError: an element is missing (see :func:`make`).
        ConfigurationError: a property the graph needs does not exist on this DeepStream's
            elements, or a link was refused. A refused link is a graph that would run and
            deliver nothing, so it is not survivable.
    """
    if not cameras:
        raise ConfigurationError("a DeepStream branch needs at least one camera")

    mux = make(gst, "nvstreammux", "shipinfer_mux")
    if mux.find_property("batch-size") is None:
        # The new muxer. Its batch and its extents come from a `config-file-path` instead, so
        # the generated configs would describe a graph that is not the one running.
        raise ConfigurationError(
            "this nvstreammux has no `batch-size` property, which means the *new* muxer is "
            "selected (USE_NEW_NVSTREAMMUX=1). It takes its batch and its extents from its own "
            "config file, so the generated ones would not describe the running graph. Unset "
            f"USE_NEW_NVSTREAMMUX. Properties it does offer: {_property_names(mux)}"
        )
    _set(mux, "gpu-id", gpu_id)
    _set(mux, "batch-size", len(cameras))
    _set(mux, "width", deepstream.mux_width)
    _set(mux, "height", deepstream.mux_height)
    _set(mux, "batched-push-timeout", deepstream.mux_batched_push_timeout_us)
    _set(mux, "live-source", int(deepstream.live_source))
    _set(mux, "enable-padding", int(deepstream.mux_enable_padding))
    # `attach-sys-ts=false` means "use the source's NTP stamp IF ONE EXISTS, else stamp
    # NOTHING" — not "fall back to arrival time" (#32 round 4 corrected the inversion here
    # and in the design doc). A file source never has one, and an RTSP source has one only
    # once sender reports flow (DeepStream's samples call configure_source_for_ntp_sync;
    # verifying whether nvurisrcbin does it internally is a §6 live-run item). The probe
    # covers the absence: an unstamped frame gets the probe's own receipt as its capture
    # time, labelled `extra.capture_origin="probe"`, so latency is never a silent zero.
    _set_if_present(mux, "attach-sys-ts", False)
    pipeline.add(mux)

    camera_by_pad: dict[int, str] = {}
    for index, camera in enumerate(cameras):
        _add_source(gst, pipeline, mux, camera, index=index, gpu_id=gpu_id, ingest=ingest)
        camera_by_pad[index] = camera.camera_id

    pgie = make(gst, "nvinfer", "shipinfer_pgie")
    _set(pgie, "config-file-path", str(configs.pgie))
    pipeline.add(pgie)

    tracker = make(gst, "nvtracker", "shipinfer_tracker")
    _set(tracker, "gpu-id", gpu_id)
    _set(tracker, "tracker-width", deepstream.tracker_width)
    _set(tracker, "tracker-height", deepstream.tracker_height)
    _set(tracker, "ll-lib-file", deepstream.tracker_lib)
    _set(tracker, "ll-config-file", str(configs.tracker))
    pipeline.add(tracker)

    sgies = []
    for name, config_path in configs.sgies:
        sgie = make(gst, "nvinfer", f"shipinfer_sgie_{name}")
        _set(sgie, "config-file-path", str(config_path))
        pipeline.add(sgie)
        sgies.append(sgie)

    sink = make(gst, "fakesink", "shipinfer_sink")
    # Nothing renders and nothing is timed against a clock: `sync=true` would pace the whole
    # graph to the presentation timestamps and turn a live pipeline into a player.
    _set_if_present(sink, "sync", False)
    _set_if_present(sink, "async", False)
    _set_if_present(sink, "enable-last-sample", False)
    pipeline.add(sink)

    for upstream, downstream in pairwise([mux, pgie, tracker, *sgies, sink]):
        _link(upstream, downstream)

    last_gie = sgies[-1] if sgies else tracker
    probe_pad = last_gie.get_static_pad("src")
    if probe_pad is None:  # pragma: no cover - an element with no src pad is not linkable
        raise ConfigurationError(
            f"{last_gie.get_name()} has no src pad to attach the metadata probe to"
        )
    return Branch(
        mux=mux,
        pgie=pgie,
        tracker=tracker,
        sgies=tuple(sgies),
        sink=sink,
        camera_by_pad=camera_by_pad,
        probe_pad=probe_pad,
    )


def _add_source(
    gst: Any,
    pipeline: Any,
    mux: Any,
    camera: CameraConfig,
    *,
    index: int,
    gpu_id: int,
    ingest: IngestSettings,
) -> None:
    """One ``nvurisrcbin``, linked to ``sink_<index>`` on the muxer when its pad appears."""
    source = make(gst, "nvurisrcbin", f"shipinfer_src_{camera.camera_id}")
    uri = source_uri(camera.uri)
    _set(source, "uri", uri)
    _set_if_present(source, "gpu-id", gpu_id)
    if uri.startswith("rtsp://"):
        # nvurisrcbin owns reconnection here — the ingest plane's actor, its backoff and its
        # health tracking are not in this topology's process at all. The interval is seconds,
        # and the ingest cap is the closest thing this configuration has to an opinion about
        # how long a camera may be gone before it is retried again.
        _set_if_present(
            source, "rtsp-reconnect-interval", max(1, ingest.reconnect_max_ms // 1000)
        )
        # Resolved here rather than through `ingest.resolve`: `pipeline` may not import
        # `ingest` (the layering rule), and the resolution order is the documented one —
        # camera, then fleet default.
        latency = camera.latency_ms if camera.latency_ms is not None else ingest.latency_ms
        if latency is not None:
            _set_if_present(source, "latency", latency)
    elif camera.loop:
        _set_if_present(source, "file-loop", True)
    pipeline.add(source)

    sink_pad = _request_pad(mux, f"sink_{index}")

    def _on_pad_added(_element: Any, pad: Any, target: Any) -> None:
        # nvurisrcbin's pads appear when the stream negotiates, which is after PLAYING —
        # and they are named `vsrc_%u`/`asrc_%u`, NOT uridecodebin's `src_%u` (#32 round 6:
        # a name-prefix test matched nothing, no camera ever linked, and the pipeline sat
        # in PLAYING publishing zero frames with nothing in the log). Match on CAPS, which
        # is correct under either naming, and make every skip visible.
        caps = pad.get_current_caps() or pad.query_caps()
        media = caps.get_structure(0).get_name() if caps and caps.get_size() else ""
        if not media.startswith("video/"):
            _LOG.debug(
                "camera %s: ignoring non-video pad %s (%s)",
                camera.camera_id,
                pad.get_name(),
                media or "no caps yet",
            )
            return
        result = pad.link(target)
        if result != gst.PadLinkReturn.OK:
            # Logged, not raised: this runs on a streaming thread, where an exception is
            # swallowed by the C caller and the camera goes dark with nothing said. The log
            # names the camera, and `frames_emitted` for it stays at zero.
            _LOG.error(
                "camera %s: linking %s to the muxer was refused (%r)",
                camera.camera_id,
                pad.get_name(),
                result,
                extra=log_context(camera_id=camera.camera_id),
            )

    source.connect("pad-added", _on_pad_added, sink_pad)


def _request_pad(element: Any, name: str) -> Any:
    """``request_pad_simple`` where it exists, ``get_request_pad`` where it does not.

    GStreamer 1.20 renamed the one to the other and deprecated the old name; DeepStream images
    exist on both sides of that. Same shape as the appsink's ``try_pull_sample`` fallback in
    the ingest source: one place for the difference, so nothing else depends on which version
    an image happens to ship.
    """
    request = getattr(element, "request_pad_simple", None) or getattr(
        element, "get_request_pad", None
    )
    if request is None:  # pragma: no cover - neither name exists on no GStreamer we support
        raise ConfigurationError(
            f"{element.get_name()} offers neither request_pad_simple nor get_request_pad"
        )
    pad = request(name)
    if pad is None:
        raise ConfigurationError(
            f"the muxer refused a request pad {name!r}. Its `batch-size` bounds how many "
            f"sink pads it will hand out, so this is one camera too many for the batch"
        )
    return pad


def _link(upstream: Any, downstream: Any) -> None:
    if not upstream.link(downstream):
        raise ConfigurationError(
            f"could not link {upstream.get_name()} -> {downstream.get_name()}: the two ends "
            f"negotiated no common format. On a DeepStream graph this is usually a plugin from "
            f"a different SDK version, or an element that was never added to the pipeline"
        )


def _set(element: Any, name: str, value: Any) -> None:
    """Set a property, refusing when this build of the element does not have it.

    PyGObject raises ``TypeError`` for an unknown property, which surfaces as an untyped crash
    with no element in the message. Every property below is one the graph needs, so an absent
    one is a configuration failure and says which element and which property.
    """
    if element.find_property(name) is None:
        raise ConfigurationError(
            f"{element.get_name()} has no property {name!r} in this DeepStream build. "
            f"Properties it does have: {_property_names(element)}"
        )
    element.set_property(name, value)


def _set_if_present(element: Any, name: str, value: Any) -> None:
    """Set a property the graph works without — an optimisation, not a requirement."""
    if element.find_property(name) is not None:
        element.set_property(name, value)
    else:
        _LOG.debug("%s has no optional property %r; leaving it", element.get_name(), name)


def _property_names(element: Any) -> list[str]:
    return sorted(spec.name for spec in element.list_properties())
