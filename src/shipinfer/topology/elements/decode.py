"""The decode elements: which ingest source a chain wants, said declaratively.

``decode: {impl: replay}`` is the one line in a chain file that chooses how frames are
pulled off the wire. What it does **not** do is pull them: the camera actors, their
reconnect schedules and their frame counters belong to the runner's ingest manager
(arch.md §5①), and an element that owned them would drag two things into ``topology`` that
must stay out of it — the camera set, and the ingest runtime that decodes it.

So an element of this family is two declarations and a pass-through:

* :attr:`_IngestDecode.source` names an implementation registered in
  :data:`shipinfer.ingest.SOURCES`. The runner reads it (never ``topology``), and puts it on
  the :class:`~shipinfer.core.settings.ingest.CameraConfig` of every camera it starts. A
  string and not an import, for the same reason ``model:`` is a string: a *name* is data a
  laptop can validate, and a decoder is not.
* :attr:`~shipinfer.topology.base.Element.produces` is the head cap of the whole chain — what
  the frames the runner submits actually are. All three implementations here say ``bgr@cpu``,
  because all three of today's sources deliver a host-memory BGR image
  (``ingest/frame/frame.py``). ``gstreamer-gpu``, which keeps NV12 in VRAM, is deliberately
  **not** registered until phase D puts a DataPool behind it (arch.md §10) — a chain naming
  it fails to load, which is what ``tests/topology/test_chain.py`` pins.
* :meth:`_IngestDecode._do_process` is ``item.derive()`` with no ``caps=``: by the time the
  walk reaches this element the frame is already in the item, already carrying the cap the
  loader negotiated for this element's outbound edge. Stamping ``output_caps[0]`` here — the
  shortcut the mocks take, and say they are only entitled to — would overwrite a negotiated
  per-edge cap with a per-element guess.

Nothing here imports a decode runtime, at module scope or anywhere else. That is the rule
this package's ``__init__`` states, and this family keeps it for free: the runtime is loaded
by the ingest source the runner builds, inside its own ``_do_open``.
"""

from __future__ import annotations

from typing import ClassVar

from shipinfer.topology.base import ChainItem, Element, ElementContext, ElementKind
from shipinfer.topology.registry import registry_for

__all__ = ["GStreamerDecode", "PyAvDecode", "ReplayDecode"]


class _IngestDecode(Element):
    """A decode element that names an ingest source and hands the frame straight on.

    Subclasses set :attr:`source` and register themselves. There is nothing else to
    implement, which is the point: three implementations that differ only in a string are
    three registry entries and not three code paths.
    """

    kind: ClassVar[ElementKind] = ElementKind.DECODE
    #: The name of a source in :data:`shipinfer.ingest.SOURCES`, resolved by the runner. A
    #: chain may still override it per camera with ``params: {source: ...}``; the runner
    #: prefers the params, because a slot's own configuration outranks its class default the
    #: same way an element prefers its ``params:`` over its :class:`ElementContext`.
    source: ClassVar[str] = ""
    produces: ClassVar[tuple[str, ...]] = ("bgr@cpu",)

    def _do_open(self, context: ElementContext) -> None:
        """Nothing to acquire. The cameras are the runner's, and it started them."""

    def _do_process(self, item: ChainItem) -> ChainItem | None:
        """Hand the frame on unchanged.

        No ``caps=``: the runner's frame sink already stamped the cap the loader negotiated
        for this element's outbound edge, and re-stamping ``output_caps[0]`` would replace a
        per-edge decision with a per-element guess (``elements/mock.py`` explains why the
        mocks are the only things entitled to that shortcut).
        """
        return item.derive()


@registry_for(ElementKind.DECODE).register("replay", "file")
class ReplayDecode(_IngestDecode):
    """Frames from a video file or a directory of images, paced to the camera's fps.

    The offline mode of arch.md §2 — ``shipinfer run --inputs a.mp4 b.mp4`` — and the one
    implementation that needs no camera and no network, which is why it is the one a test
    and a bench reach for.
    """

    source: ClassVar[str] = "replay"


@registry_for(ElementKind.DECODE).register("gstreamer", "gst")
class GStreamerDecode(_IngestDecode):
    """RTSP through GStreamer, decoded to host memory.

    Not ``gstreamer-gpu``: this one lands a BGR image in ordinary RAM, so it declares
    ``bgr@cpu`` honestly. The zero-copy NV12-in-VRAM path is a different implementation with
    a different head cap, and it waits for the DataPool (arch.md §3, phase D).
    """

    source: ClassVar[str] = "gstreamer"


@registry_for(ElementKind.DECODE).register("pyav")
class PyAvDecode(_IngestDecode):
    """RTSP or a file through PyAV/FFmpeg. The portable fallback where GStreamer is absent."""

    source: ClassVar[str] = "pyav"
