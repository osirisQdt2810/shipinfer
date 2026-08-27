"""The DeepStream topology's child process: one NVIDIA graph per shard, the same events out.

``fleet`` and ``service`` run this project's scheduler over TensorRT engines. ``deepstream``
hands the whole per-frame path to NVIDIA's graph — ``nvurisrcbin -> nvstreammux ->
nvinfer(detector) -> nvtracker -> nvinfer(embedders)`` — and keeps exactly two ends:

* the **model repository** generates the nvinfer configs (:mod:`.configs`), so the engine
  paths, input dims, output names, class labels and thresholds have one owner rather than two;
* the **result sink** receives :class:`~shipinfer.pipeline.schema.PerceptionEvent`, the same
  event every other topology publishes, so a downstream consumer cannot tell which topology
  produced it and the comparison is a comparison.

Layout, one reason each::

    configs.py   model repository -> nvinfer/tracker/label text. Pure; every refusal is here
    probe.py     NvDsBatchMeta -> FrameView -> PerceptionEvent. `pyds` is an argument
    builder.py   the elements, their properties and their links. Every one checked
    run.py       the process: main loop, bus, signals, and the runner's emit discipline

**Every module here imports on a host with no GStreamer and no ``pyds``.** That is the whole
reason the package is shaped this way, and it is pinned by a test: ``gi`` is loaded inside
:func:`shipinfer.runtime.gstreamer.load_pyds`, ``pyds`` is a *parameter* of
:func:`~shipinfer.pipeline.deepstream.probe.walk_batch`, and the failure when either is absent
is a typed :class:`~shipinfer.core.errors.SourceUnavailableError` raised at start-up. So the
config generation, the coordinate mapping and the emission discipline are all verified in the
offline tier, on a laptop, and only the graph itself needs a DeepStream host.
"""

from shipinfer.pipeline.deepstream.builder import Branch, build_branch, make
from shipinfer.pipeline.deepstream.configs import (
    GeneratedConfigs,
    labels_file,
    pgie_config,
    sgie_config,
    source_uri,
    tracker_config,
    write_configs,
)
from shipinfer.pipeline.deepstream.probe import (
    UNTRACKED_OBJECT_ID,
    FrameGeometry,
    FrameNumbering,
    FrameView,
    ObjectView,
    build_event,
    walk_batch,
)
from shipinfer.pipeline.deepstream.run import (
    PR1_MISSING_STAGES,
    DeepStreamPipeline,
    MetadataProbe,
)

__all__ = [
    "PR1_MISSING_STAGES",
    "UNTRACKED_OBJECT_ID",
    "Branch",
    "DeepStreamPipeline",
    "FrameGeometry",
    "FrameNumbering",
    "FrameView",
    "GeneratedConfigs",
    "MetadataProbe",
    "ObjectView",
    "build_branch",
    "build_event",
    "labels_file",
    "make",
    "pgie_config",
    "sgie_config",
    "source_uri",
    "tracker_config",
    "walk_batch",
    "write_configs",
]
