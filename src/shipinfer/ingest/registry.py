"""Registry of video sources, and the factory that resolves which one a camera gets.

Registration is eager — a decorator on each class — because every source module is
*import-safe*: it pulls in nothing but numpy and ``core`` at import time and loads its
decode runtime inside ``_do_open``. That is what lets this registry list ``gstreamer`` on a
host with no GStreamer installed and still fail usefully, at ``open()``, with a
:class:`~shipinfer.core.errors.SourceUnavailableError` naming the package to install.

The alternative — lazy registration — would hide the name until the import succeeded, so a
misconfigured deployment would be told "unknown video source" when the truth is "PyGObject
is not installed". Those are different problems with different fixes.
"""

from __future__ import annotations

from shipinfer.core.logging import get_logger, log_context
from shipinfer.core.registry import Registry
from shipinfer.core.settings.ingest import CameraConfig, IngestSettings
from shipinfer.ingest.base import FrameSource
from shipinfer.ingest.resolve import resolve_source_name
from shipinfer.ingest.frame.tag import FrameCounter

__all__ = ["SOURCES", "create_source"]

_LOG = get_logger("ingest.registry")

SOURCES: Registry[FrameSource] = Registry("video source", FrameSource)


def create_source(
    config: CameraConfig,
    counter: FrameCounter | None = None,
    *,
    settings: IngestSettings | None = None,
) -> FrameSource:
    """Build the source this camera asks for.

    The selection order is camera field -> ``ingest.backend`` setting ->
    ``$SHIPINFER_INGEST_BACKEND``, implemented once in
    :func:`~shipinfer.ingest.base.resolve_source_name`.

    Raises:
        ConfigurationError: the resolved name is not registered. The message lists every
            registered name, because "unknown video source 'gstremaer'" with no list is a
            twenty-minute detour.
    """
    name = resolve_source_name(config, settings)
    _LOG.debug(
        "creating %r source for camera %s",
        name,
        config.camera_id,
        extra=log_context(camera_id=config.camera_id),
    )
    source_cls = SOURCES.get(name)
    return source_cls(config, counter, settings=settings)
