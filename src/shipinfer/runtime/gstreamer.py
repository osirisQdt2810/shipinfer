"""Importing and initialising GStreamer — once per process, whoever asks first.

WHY THIS IS IN ``runtime`` AND NOT IN ``ingest``
------------------------------------------------
It started in :mod:`shipinfer.ingest.sources.gstreamer`, where the only caller was the RTSP
camera source. It has two callers now: that source, and the DeepStream topology's graph under
:mod:`shipinfer.pipeline.deepstream`. ``pipeline`` may not import ``ingest`` — the layering
DAG gives it no such edge, deliberately, because ``ingest`` publishes *into* a protocol
``pipeline`` implements (ADR-011) — so the shared loader moves down to the layer both may
import. Which is also the honest home for it: initialising a plugin registry that will open a
video engine is accelerator-seam work, not camera work.

The body of :func:`load_gst` is unchanged from the version that ran in ``ingest``, lock and
comments included, because every line of it is the scar of a production failure and a rewrite
in transit would be the easiest way to reintroduce one.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from shipinfer.core.errors import SourceUnavailableError
from shipinfer.core.redact import redact_in

__all__ = ["load_gst", "load_pyds"]

#: Serialises GStreamer's one-time initialisation across camera threads (see `load_gst`).
_GST_INIT_LOCK = threading.Lock()


def load_gst() -> tuple[Any, Any]:
    """Import and initialise GStreamer, or explain what is missing.

    Deliberately inside a function: importing this module must work on a host with no
    PyGObject, so the whole offline test tier can exercise the pipeline builder and the
    camera actor.

    The whole body runs under one lock. Fifty camera actors call this from fifty threads at
    start-up, and neither half is safe to race: PyGObject resolves ``gi.repository`` members
    lazily and a concurrent first touch has come back as ``'GLib' object has no attribute
    'Idle'``; and ``Gst.is_initialized()`` turns true as soon as *some* thread has begun
    initialising, before the plugin registry is populated — a thread that saw "initialised"
    and probed the registry found no decoder and gave its camera up as "no h264 decoder
    found" on an image that has three. One lock, one import, one init, and every caller
    returns only after the registry exists.
    """
    with _GST_INIT_LOCK:
        # `rtspsrc` asks GIO for a proxy resolver before it connects, and GIO's default on a
        # desktop-less system is libproxy, which throws a C++ `std::runtime_error("Unable to
        # read configuration")` when it finds no GSettings or D-Bus to read — uncaught across
        # the C boundary, that is `terminate` for the whole process, which is how the first
        # RTSP run inside the container died with fifty cameras connected and zero frames
        # decoded. GIO's documented override selects its no-op resolver instead; `setdefault`
        # so an operator who has configured a real proxy keeps it.
        os.environ.setdefault("GIO_USE_PROXY_RESOLVER", "dummy")
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import GLib, Gst

            # The appsink's *methods* (`try_pull_sample`) exist on the Python side only when the
            # GstApp typelib has been loaded; without it `get_by_name` returns a bare element
            # whose Python type knows the signals but not the methods, and every read failed
            # with "'GstAppSink' object has no attribute 'try_pull_sample'" on the first
            # containerised RTSP run that reached a read. Loaded here, once; `_do_read` still
            # falls back to the signal when the typelib is absent.
            try:
                gi.require_version("GstApp", "1.0")
                from gi.repository import GstApp  # noqa: F401 - loading it is the effect
            except (ImportError, ValueError):
                pass
        except (ImportError, ValueError) as exc:
            raise SourceUnavailableError(
                "gstreamer",
                "PyGObject with GStreamer 1.0 typelibs is not importable "
                f"({redact_in(str(exc))}). Install python3-gi and "
                "gstreamer1.0-plugins-{base,good,bad}, or select the 'pyav' backend with "
                "SHIPINFER_INGEST_BACKEND=pyav",
            ) from exc
        if not Gst.is_initialized():
            Gst.init(None)
        return Gst, GLib


def load_pyds() -> tuple[Any, Any, Any]:
    """GStreamer plus DeepStream's Python metadata bindings: ``(Gst, GLib, pyds)``.

    ``pyds`` is the only way to read `NvDsBatchMeta` from a buffer, which is the one thing the
    DeepStream topology's process actually has to do in Python — the frames never leave the
    graph. It ships with the DeepStream SDK's `deepstream_python_apps` and is not on PyPI, so
    it is absent on every machine that is not a DeepStream install, this one included.

    Raises:
        SourceUnavailableError: PyGObject or ``pyds`` is missing. The same class a missing
            PyGObject raises, and for the same reason: an install fixes it and a retry never
            will, so a shard must fail out to its supervisor rather than spend a reconnect
            budget on it. Raised at *start*, never at import — every module in
            :mod:`shipinfer.pipeline.deepstream` imports on a host with neither, which is what
            keeps the config generation and the probe logic in the offline tier.
    """
    gst, glib = load_gst()
    try:
        import pyds
    except ImportError as exc:
        raise SourceUnavailableError(
            "deepstream",
            f"the DeepStream Python bindings (pyds) are not importable ({redact_in(str(exc))}). "
            "They ship with the DeepStream SDK — install the SDK and its "
            "deepstream_python_apps bindings (see deploy/deepstream/image.sh), or run the "
            "`fleet` / `service` topology, which needs neither",
        ) from exc
    return gst, glib, pyds
