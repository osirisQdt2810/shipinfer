"""The model repository, rendered as nvinfer configuration text.

WHY GENERATE THEM AT ALL
------------------------
Every DeepStream deployment anyone has ever seen carries a directory of hand-written
``pgie_config.txt`` / ``sgie_config.txt`` files. They are a *second description of a model that
is already described*: the engine path, the input dims, the output blob names, the batch size
and the class count are all in ``model_repository/<name>/config.yaml``, where the rest of this
project reads them from. Two descriptions of one model is one that drifts, and the drift is
silent — an ``infer-dims`` that no longer matches the engine is a start-up failure several
seconds into a deploy, and a stale ``num-detected-classes`` is a mislabelled object forever.

So the repository is the source of truth here as it is everywhere else, and this module is the
projection onto NVIDIA's vocabulary. The mapping is the interesting part and it is stated
once, in :func:`pgie_config` and :func:`sgie_config`.

WHAT IS REFUSED, AND WHY HERE
-----------------------------
Everything that would otherwise fail inside a GStreamer element on a machine with a GPU is
refused here instead, in pure Python, on a laptop: a detector that is not a TensorRT model, an
input that is not one 3-D FP32 tensor, an end-to-end detector with no bounding-box parser to
read its output, a class filter naming a label the detector does not have, two secondaries
claiming the same label, a secondary that cannot batch. Each of those is a config-generation
error with the model named — which is the whole reason the generation exists.

**Nothing is written into the model repository.** The generated files live under a run
directory (``$TMPDIR/shipinfer-ds-<run>/shard<N>/`` by default): a stray ``.txt`` beside a
``config.yaml`` is at best noise in a diff, and :class:`~shipinfer.repository.ModelRepository`
refuses a ``config.pbtxt`` in a model directory outright.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.settings import CameraConfig, PipelineSettings, ServerSettings
from shipinfer.core.settings.topology import DeepStreamSettings
from shipinfer.core.types import DataType
from shipinfer.pipeline.graph.detections import UNKNOWN_LABEL
from shipinfer.repository import IOConfig, ModelArtifact, ModelRepository

__all__ = [
    "GeneratedConfigs",
    "labels_file",
    "pgie_config",
    "sgie_config",
    "source_uri",
    "tracker_config",
    "write_configs",
]

#: nvinfer's ``network-mode``, in its own numbering. Only honoured when nvinfer builds the
#: engine from ``onnx-file``; a prebuilt plan already has its precision baked in.
_NETWORK_MODE = {"fp32": 0, "int8": 1, "fp16": 2}

#: ``net-scale-factor`` and the colour format, matching this project's own preprocessing:
#: :class:`~shipinfer.runtime.ops.base.NormalizeParams` defaults to mean 0, std 255 and
#: ``swap_rb=True``, i.e. RGB in [0, 1]. Both planes must agree or the same engine sees
#: different pixels depending on which topology fed it, and the comparison measures that
#: instead of the architecture.
_NET_SCALE_FACTOR = 1.0 / 255.0
_MODEL_COLOR_FORMAT_RGB = 0

#: ``cluster-mode=4`` is "no clustering". The shipped detector is end-to-end — yolo26 emits
#: decoded boxes with NMS already applied — so a second round of clustering here would merge
#: boxes the engine already decided to keep. The DetectNet coverage/bbox path is the
#: opposite (#43 round 1): the grid emits raw per-cell boxes, and publishing them unclustered
#: is hundreds of overlapping boxes per object — that path gets DBSCAN.
_CLUSTER_MODE_NONE = 4
_CLUSTER_MODE_DBSCAN = 1

#: What ``nvurisrcbin`` is asked to open. Two schemes on purpose: a fleet is RTSP cameras, and
#: a file is how a fixture replays one. Adding a third (``http``) is one entry here plus a test.
_SUPPORTED_SCHEMES = ("file", "rtsp")


@dataclass(frozen=True, slots=True)
class GeneratedConfigs:
    """Every file one shard's graph reads, and where it was written."""

    #: ``<root>/shard<N>``. Everything below is inside it, except an operator-supplied tracker
    #: config, which is used where it already lives.
    root: Path
    pgie: Path
    tracker: Path
    labels: Path
    #: ``(model name, config path)`` per secondary GIE, in the order they run.
    sgies: tuple[tuple[str, Path], ...]
    #: Artefacts the generated configs name that are not on this machine — a ``model.plan``
    #: that has not been built here, most often. Empty on a real run, because generation
    #: refuses then; populated by a dry run, which is expected to happen on a control box.
    missing_files: tuple[Path, ...] = ()

    def paths(self) -> tuple[Path, ...]:
        """Every generated file, in the order an operator would read them."""
        return (self.pgie, *(path for _name, path in self.sgies), self.tracker, self.labels)


def source_uri(uri: str) -> str:
    """The URI ``nvurisrcbin`` is given for one camera.

    A bare path becomes ``file://`` — an absolute one, because a relative path resolved
    against the child's working directory is a fixture that works in a test and not in a
    deployment. Anything else keeps its scheme.

    Raises:
        ConfigurationError: a scheme this graph does not open. Refused at *config* time so it
            is one message on one line, rather than a camera that connects to nothing and looks
            like a network problem for the next hour.
    """
    text = uri.strip()
    scheme, separator, _rest = text.partition("://")
    if not separator:
        return Path(text).expanduser().resolve().as_uri()
    if scheme.lower() in _SUPPORTED_SCHEMES:
        return text
    raise ConfigurationError(
        f"camera uri {uri!r} has scheme {scheme!r}, and the `deepstream` topology opens "
        f"{list(_SUPPORTED_SCHEMES)} (a bare path is taken as a file). Add the scheme to "
        f"_SUPPORTED_SCHEMES if nvurisrcbin handles it"
    )


def labels_file(class_labels: Mapping[int, str]) -> str:
    """One label per line, indexed by class id — nvinfer's ``labelfile-path``.

    The file is positional, so a gap in the id space has to be filled: ``{0: person, 8: ship}``
    is nine lines, seven of them ``unknown``. Same word the Python DAG uses for an id its map
    does not mention, and for the same reason — an unmapped class must be visible in the event
    rather than silently absent.
    """
    if not class_labels:
        raise ConfigurationError(
            "pipeline.class_labels is empty, so nvinfer has no labels to give its detections. "
            "Set it to the detector's own numbering (the shipped one is COCO: 0 person, 8 boat)"
        )
    highest = max(class_labels)
    return "".join(
        f"{class_labels.get(index, UNKNOWN_LABEL)}\n" for index in range(highest + 1)
    )


def tracker_config(deepstream: DeepStreamSettings, *, pipeline: PipelineSettings) -> str:
    """The low-level tracker's config, in the OpenCV-FileStorage YAML NvDCF reads.

    Deliberately small. Every key here is one this deployment has an opinion about — the
    detector's own confidence floor, the fan-out cap the Python pipeline enforces, and how long
    a track may be shadow-tracked, which has to grow with ``interval``: the detector only sees
    one frame in ``interval + 1``, so a fixed age in *frames* is a shrinking number of chances
    to re-detect. Everything else is left to the library's defaults rather than guessed at. An
    operator who wants NVIDIA's tuned numbers points ``topology.deepstream.tracker_config`` at
    the SDK's own ``config_tracker_NvDCF_perf.yml``, and this file is then not generated at all.
    """
    return (
        "%YAML:1.0\n"
        "# Generated by shipinfer; see topology.deepstream.tracker_config to use your own.\n"
        "BaseConfig:\n"
        f"  minDetectorConfidence: {pipeline.score_threshold}\n"
        "TargetManagement:\n"
        "  enableBboxUnClipping: 1\n"
        f"  maxTargetsPerStream: {pipeline.max_detections}\n"
        "  probationAge: 3\n"
        f"  maxShadowTrackingAge: {30 * (deepstream.interval + 1)}\n"
        "  minIouDiff4NewTarget: 0.5\n"
        "DataAssociator:\n"
        "  checkClassMatch: 1\n"
        "StateEstimator:\n"
        "  stateEstimatorType: 1\n"
    )


def pgie_config(
    artifact: ModelArtifact,
    *,
    gpu_id: int,
    batch_size: int,
    deepstream: DeepStreamSettings,
    pipeline: PipelineSettings,
    labels_path: Path,
    engine: Path,
    onnx: Path | None = None,
) -> str:
    """The primary GIE's config: one nvinfer over every muxed frame.

    The mapping from this repository's vocabulary to NVIDIA's, which is the whole point of
    generating the file:

    =========================  =================================================================
    ``model-engine-file``      the artefact's ``parameters.engine_file`` in its version dir
    ``onnx-file``              ``parameters.onnx_file``, when the model declares one. Optional
                               and worth setting: an engine is only valid for the TensorRT
                               version that built it, and with an ONNX beside it nvinfer
                               rebuilds instead of refusing to start
    ``infer-dims``             the single input's ``dims``, ``c;h;w``
    ``batch-size``             the number of cameras this shard muxes — nvstreammux's batch
    ``output-blob-names``      the declared outputs, in order
    ``num-detected-classes``   ``max(pipeline.class_labels) + 1``, the label file's length
    ``pre-cluster-threshold``  ``pipeline.score_threshold``
    ``topk``                   ``pipeline.max_detections``
    =========================  =================================================================

    Two keys are opinions rather than translations. ``maintain-aspect-ratio=1`` with
    ``symmetric-padding=1`` is the letterbox this project's own preprocessing does
    (``ImageOps.letterbox``, centre-padded), so the detector sees the geometry it was built for
    and nvinfer scales its boxes back to the muxed frame itself. ``cluster-mode`` follows the
    layout: 4 (none) for decoded end-to-end or custom-parsed outputs, DBSCAN for the raw
    DetectNet coverage/bbox grid, whose per-cell boxes must be clustered or every object
    publishes as hundreds of overlapping rectangles.

    Raises:
        ConfigurationError: the model is not a TensorRT model, does not have exactly one 3-D
            FP32 input, or emits outputs that are not the named DetectNet coverage/bbox pair
            while no bounding-box parser is configured to read them. That last one is the
            refusal that matters: nvinfer's built-in parsers cannot decode a ``(300, 6)``
            end-to-end tensor, an EfficientNMS quartet or a segmentation head, and without
            ``parse-bbox-func-name`` a shard starts, runs, and reports zero detections on
            every frame.
    """
    tensor = _single_image_input(artifact)
    outputs = [io.name for io in artifact.config.outputs]
    if not deepstream.bbox_parser and not _is_coverage_bbox_pair(artifact.config.outputs):
        # Not just the single decoded tensor (#32) or the four-output EfficientNMS quartet
        # (#32 round 7): the count alone let a two-output yolo-seg-shaped model through
        # (#42 round 1 — this repository's own ship_segmenter, [300, 38] + [32, 160, 160],
        # is the counterexample). What nvinfer's built-in parser documents is the DetectNet
        # layout: a 3-D coverage grid of C class confidences beside a 3-D bbox grid of 4*C
        # channels on the same spatial extent. Anything else slips into the same
        # runs-and-reports-zero-detections failure the refusal exists to prevent.
        shapes = ", ".join(f"{io.name}{list(io.dims)}" for io in artifact.config.outputs)
        raise ConfigurationError(
            f"{artifact.name} emits output tensor(s) {shapes}, which nvinfer's built-in "
            f"bounding-box parsers cannot read: they expect the two-tensor DetectNet "
            f"coverage/bbox layout ([C, gh, gw] beside [4*C, gh, gw]), not a decoded "
            f"end-to-end tensor, an EfficientNMS quartet, or a segmentation head. Set "
            f"topology.deepstream.bbox_parser (nvinfer's `parse-bbox-func-name`) and "
            f"custom_lib (`custom-lib-path`) to the function that decodes it — without one "
            f"the graph runs and reports zero detections on every frame, which looks like "
            f"a quiet camera"
        )
    if batch_size < 1:
        raise ConfigurationError(
            f"{artifact.name}: the primary GIE's batch is the number of cameras this shard "
            f"muxes, and it got {batch_size}"
        )
    declared = artifact.config.max_batch_size
    if batch_size > declared:
        raise ConfigurationError(
            f"{artifact.name} declares max_batch_size {declared}, and this shard muxes "
            f"{batch_size} cameras — nvstreammux's batch is the primary GIE's batch, so "
            f"nvinfer would be asked for a batch the engine cannot bind and the shard would "
            f"die inside the element seconds into the deploy. Use more shards (at most "
            f"{declared} cameras each), or rebuild the engine with a larger max_batch_size"
        )

    lines = [
        "# Generated by shipinfer from the model repository. Edits are overwritten at start-up.",
        f"# model: {artifact.name} v{artifact.version} ({artifact.path})",
        "[property]",
        f"gpu-id={gpu_id}",
        # RGB in [0, 1] — the same normalisation `runtime/ops` applies on the Python path.
        f"net-scale-factor={_NET_SCALE_FACTOR!r}",
        f"model-color-format={_MODEL_COLOR_FORMAT_RGB}",
        f"model-engine-file={engine}",
    ]
    if onnx is not None:
        lines.append(f"onnx-file={onnx}")
    lines += [
        f"labelfile-path={labels_path}",
        f"batch-size={batch_size}",
        f"network-mode={_NETWORK_MODE[deepstream.network_mode]}",
        # Deliberately the label MAP's extent, not the network's class count: a COCO
        # yolo26 emits ids up to 79, and declaring 9 makes nvinfer drop everything this
        # deployment does not label. The Python plane keeps those ids and publishes
        # UNKNOWN_LABEL instead — a stated cross-plane divergence (design doc §6), not
        # an accident: filtering at the parser is this topology's only pre-tracker gate.
        f"num-detected-classes={max(pipeline.class_labels) + 1}",
        f"interval={deepstream.interval}",
        # 1 is the primary GIE by convention, and every secondary's `operate-on-gie-id` says 1.
        "gie-unique-id=1",
        "process-mode=1",
        "network-type=0",
        f"infer-dims={_infer_dims(tensor.dims)}",
        f"output-blob-names={';'.join(outputs)}",
        "maintain-aspect-ratio=1",
        "symmetric-padding=1",
        # Decoded boxes must not be re-clustered; raw DetectNet grids must be (#43
        # round 1) — one mode per layout, chosen by the same predicate as the parser gate.
        f"cluster-mode={_CLUSTER_MODE_NONE if deepstream.bbox_parser or not _is_coverage_bbox_pair(artifact.config.outputs) else _CLUSTER_MODE_DBSCAN}",
    ]
    if bool(deepstream.bbox_parser) != bool(deepstream.custom_lib):
        # Belt and braces beside the settings validator (the unfiltered-secondary
        # shape from round 1): a hand-built settings object must not generate a
        # config nvinfer can only fail to dlsym.
        raise ConfigurationError(
            f"{artifact.name}: bbox_parser and custom_lib travel together (one names "
            f"the function, the other the library it is dlsym'd from); got "
            f"parser={deepstream.bbox_parser!r} lib={deepstream.custom_lib!r}"
        )
    if deepstream.bbox_parser:
        lines.append(f"parse-bbox-func-name={deepstream.bbox_parser}")
    if deepstream.custom_lib:
        lines.append(f"custom-lib-path={deepstream.custom_lib}")
    lines += [
        "",
        "[class-attrs-all]",
        f"pre-cluster-threshold={pipeline.score_threshold}",
        f"topk={pipeline.max_detections}",
    ]
    return "\n".join(lines) + "\n"


def sgie_config(
    artifact: ModelArtifact,
    *,
    gpu_id: int,
    deepstream: DeepStreamSettings,
    gie_unique_id: int,
    operate_on_class_ids: Sequence[int],
    engine: Path,
    onnx: Path | None = None,
) -> str:
    """A secondary GIE's config: one nvinfer over the objects the detector found.

    Three keys make it a *secondary* and they are the ones worth knowing.
    ``process-mode=2`` means it runs on object crops rather than on frames.
    ``network-type=100`` is "other": no built-in postprocessing, because an embedder's output is
    a vector and nvinfer has no opinion about it. ``output-tensor-meta=1`` is what makes that
    vector reachable at all — it attaches the raw output tensor to the object's user-meta list,
    which is where :func:`shipinfer.pipeline.deepstream.probe.walk_batch` reads it.

    ``operate-on-gie-id=1`` and ``operate-on-class-ids`` are the class-conditional branch the
    Python DAG expresses as a `condition` on an ensemble step: the person embedder sees people,
    the ship embedder sees ships, and neither pays for the other's crops.

    ``maintain-aspect-ratio`` is deliberately left at nvinfer's default (0, a plain resize),
    because that is what :meth:`~shipinfer.runtime.ops.base.ImageOps.crop_batch` does on the
    Python path — a re-id crop is resized, not letterboxed.

    Raises:
        ConfigurationError: the model is not a TensorRT model, does not have exactly one 3-D
            FP32 input, or declares ``max_batch_size: 0``. A secondary that cannot batch would
            run one inference per object, which at ~15 crops a frame is the difference between
            fitting on the GPU and not.
    """
    tensor = _single_image_input(artifact)
    for output in artifact.config.outputs:
        if output.data_type.name != "FP32":
            # The probe reinterprets secondary tensor meta as float32; a HALF output
            # would read as garbage at double the byte span and publish silently (#32
            # round 6) — re-id would match arbitrary ships at full confidence. Both
            # shipped embedders declare FP32; anything else is refused here, offline.
            raise ConfigurationError(
                f"{artifact.name} output {output.name!r} is {output.data_type.name}, "
                f"and the probe reads secondary tensor meta as FP32 — export the "
                f"embedding head in FP32, or teach the probe the dtype first"
            )
    batch_size = artifact.config.max_batch_size
    if batch_size < 1:
        raise ConfigurationError(
            f"{artifact.name} declares max_batch_size: 0, so nvinfer would run one inference "
            f"per object. Give it a batch (the shipped embedders use 16)"
        )

    lines = [
        "# Generated by shipinfer from the model repository. Edits are overwritten at start-up.",
        f"# model: {artifact.name} v{artifact.version} ({artifact.path})",
        "[property]",
        f"gpu-id={gpu_id}",
        f"net-scale-factor={_NET_SCALE_FACTOR!r}",
        f"model-color-format={_MODEL_COLOR_FORMAT_RGB}",
        f"model-engine-file={engine}",
    ]
    if onnx is not None:
        lines.append(f"onnx-file={onnx}")
    lines += [
        f"batch-size={batch_size}",
        f"network-mode={_NETWORK_MODE[deepstream.network_mode]}",
        # No `interval` on a process-mode=2 GIE: for secondaries it skips batches of
        # OBJECTS, not frames — not what the knob's docstring promises (#32 round 4).
        f"gie-unique-id={gie_unique_id}",
        "operate-on-gie-id=1",
        "process-mode=2",
        "network-type=100",
        "output-tensor-meta=1",
        f"infer-dims={_infer_dims(tensor.dims)}",
        f"output-blob-names={';'.join(io.name for io in artifact.config.outputs)}",
    ]
    if operate_on_class_ids:
        lines.append(f"operate-on-class-ids={';'.join(str(c) for c in operate_on_class_ids)}")
    return "\n".join(lines) + "\n"


def write_configs(
    repository: ModelRepository,
    *,
    settings: ServerSettings,
    shard_index: int,
    gpu_id: int,
    cameras: Sequence[CameraConfig],
    root: Path,
    require_engine: bool = True,
) -> GeneratedConfigs:
    """Generate one shard's whole config set under ``root/shard<N>/``.

    Args:
        require_engine: whether a named artefact must already exist. ``True`` on a real start —
            an nvinfer that cannot open its engine fails inside a GStreamer element, and this
            failure says which model and which directory instead. ``False`` for ``--dry-run``,
            which is expected to run on a control box where the engines were never built: the
            paths are still written, and the ones that are missing come back in
            :attr:`GeneratedConfigs.missing_files` for the caller to print.

    Raises:
        ConfigurationError: any of this module's refusals, plus an empty camera list (a shard
            with no cameras is a graph with a zero-sized batch) and an `operate_on` label the
            detector's own map does not define.
    """
    deepstream = settings.topology.deepstream
    pipeline = settings.pipeline
    if not cameras:
        raise ConfigurationError(
            f"shard {shard_index} has no cameras, so nvstreammux would be given batch-size 0. "
            f"A shard with nothing to read still holds a CUDA context"
        )
    for camera in cameras:
        source_uri(camera.uri)  # refuse an unopenable scheme now, not per camera at start-up

    # Absolute, and every path written inside these files is too. nvinfer resolves a relative
    # `model-engine-file` against the *config file's* directory, so a repository-relative path
    # in a config under $TMPDIR names an engine in $TMPDIR — a start-up failure whose message
    # points at a directory nobody chose.
    directory = Path(root).expanduser().resolve() / f"shard{shard_index}"
    directory.mkdir(parents=True, exist_ok=True)
    missing: list[Path] = []

    labels_path = directory / "labels.txt"
    labels_path.write_text(labels_file(pipeline.class_labels), encoding="utf-8")

    detector = repository.resolve(deepstream.detector)
    _require_tensorrt(detector)
    engine, onnx = _artefacts(detector, require=require_engine, missing=missing)
    pgie_path = directory / f"pgie_{detector.name}.txt"
    pgie_path.write_text(
        pgie_config(
            detector,
            gpu_id=gpu_id,
            batch_size=len(cameras),
            deepstream=deepstream,
            pipeline=pipeline,
            labels_path=labels_path,
            engine=engine,
            onnx=onnx,
        ),
        encoding="utf-8",
    )

    by_label = _labels_to_ids(pipeline.class_labels)
    claimed: dict[str, str] = {}
    sgies: list[tuple[str, Path]] = []
    # 1 is the primary; secondaries take 2..N in the order they run, which is also the order
    # `operate-on-gie-id` would chain them in if one ever ran on another's output.
    for offset, name in enumerate(deepstream.secondaries):
        secondary = repository.resolve(name)
        _require_tensorrt(secondary)
        class_ids = _class_ids_for(name, deepstream.operate_on.get(name, []), by_label, claimed)
        sgie_engine, sgie_onnx = _artefacts(secondary, require=require_engine, missing=missing)
        path = directory / f"sgie_{name}.txt"
        path.write_text(
            sgie_config(
                secondary,
                gpu_id=gpu_id,
                deepstream=deepstream,
                gie_unique_id=offset + 2,
                operate_on_class_ids=class_ids,
                engine=sgie_engine,
                onnx=sgie_onnx,
            ),
            encoding="utf-8",
        )
        sgies.append((name, path))

    if deepstream.tracker_config is not None:
        tracker_path = Path(deepstream.tracker_config).expanduser().resolve()
    else:
        tracker_path = directory / "tracker.yml"
        tracker_path.write_text(tracker_config(deepstream, pipeline=pipeline), encoding="utf-8")

    return GeneratedConfigs(
        root=directory,
        pgie=pgie_path,
        tracker=tracker_path,
        labels=labels_path,
        sgies=tuple(sgies),
        missing_files=tuple(missing),
    )


# -- the refusals ------------------------------------------------------------------------


def _require_tensorrt(artifact: ModelArtifact) -> None:
    if artifact.config.platform != "tensorrt":
        raise ConfigurationError(
            f"{artifact.name} has platform {artifact.config.platform!r}, and nvinfer reads a "
            f"TensorRT engine (or an ONNX it builds one from) directly. The `deepstream` "
            f"topology cannot run it; `fleet` and `service` can, through the {artifact.config.platform!r} backend"
        )


def _is_coverage_bbox_pair(outputs: Sequence[IOConfig]) -> bool:
    """Whether two output tensors are the DetectNet coverage/bbox layout, by shape.

    The one layout nvinfer's built-in parser documents: a coverage grid of ``C`` class
    confidences and a bbox grid of ``4 * C`` channels, both 3-D, on the same spatial
    extent. A heuristic on purpose — shapes cannot prove semantics — but every shape it
    rejects is one the built-in parser demonstrably cannot read, and the remedy in the
    refusal (a custom parser pair) is the same either way (#42 round 1).
    """
    if len(outputs) != 2:
        return False
    # By NAME as well as by shape (#43 round 1): nvinfer's DetectPostprocessor locates the
    # pair with strstr(layerName, "cov") / strstr(layerName, "bbox"), not by position — a
    # correctly-shaped pair named conf/boxes would generate parserless and then fail at
    # start-up with "Could not find output coverage layer". Loud, but foreseeable here.
    names = sorted(io.name for io in outputs)
    if not any("cov" in n for n in names) or not any("bbox" in n for n in names):
        return False
    a, b = (list(io.dims) for io in outputs)
    if len(a) != 3 or len(b) != 3:
        return False
    coverage, bbox = (a, b) if a[0] <= b[0] else (b, a)
    return bbox[0] == 4 * coverage[0] and coverage[1:] == bbox[1:]


def _single_image_input(artifact: ModelArtifact) -> IOConfig:
    """The one 3-D FP32 input nvinfer can be told about, or a refusal naming what it found."""
    inputs = artifact.config.inputs
    if len(inputs) != 1:
        raise ConfigurationError(
            f"{artifact.name} declares {len(inputs)} inputs "
            f"({[io.name for io in inputs]}), and nvinfer feeds a GIE exactly one image "
            f"tensor. A multi-input model belongs behind the `fleet` topology's backend"
        )
    tensor = inputs[0]
    if len(tensor.dims) != 3:
        raise ConfigurationError(
            f"{artifact.name}: input {tensor.name!r} has dims {tensor.dims}, and nvinfer's "
            f"`infer-dims` is c;h;w — three extents, the batch dimension excluded"
        )
    if tensor.data_type is not DataType.FP32:
        raise ConfigurationError(
            f"{artifact.name}: input {tensor.name!r} is {tensor.data_type.value}, and nvinfer's "
            f"preprocessing writes FP32. Convert the engine's input, or run this model behind "
            f"the `fleet` topology"
        )
    return tensor


def _class_ids_for(
    model: str,
    labels: Sequence[str],
    by_label: Mapping[str, int],
    claimed: dict[str, str],
) -> tuple[int, ...]:
    """Resolve a secondary's label filter to detector class ids, once and at config time.

    Two secondaries claiming one label is refused rather than resolved. Both would attach an
    output tensor to the same object, the probe reads the first it finds, and which vector an
    object ends up carrying would depend on GIE order — a wrong embedding that looks like a
    right one.
    """
    if not labels:
        # nvinfer runs an UNFILTERED secondary on every class, so an empty entry is a claim
        # on all of them. The settings validator refuses this before it ever reaches here;
        # this refusal covers a hand-built settings object handed straight to
        # `write_configs` (#32 round 1: an unfiltered secondary displaced the right
        # embedding with whichever tensor came first in the user-meta list).
        raise ConfigurationError(
            f"secondary {model!r} has no operate_on labels: an unfiltered GIE runs on every "
            f"class, and its output tensor silently displaces another secondary's on shared "
            f"objects. Name the labels it operates on (known: {sorted(by_label)})"
        )
    ids: list[int] = []
    for label in labels:
        if label not in by_label:
            raise ConfigurationError(
                f"topology.deepstream.operate_on[{model!r}] names label {label!r}, which "
                f"pipeline.class_labels does not define (it has {sorted(by_label)}). A class "
                f"filter is resolved to ids at config time precisely so this is one message "
                f"rather than a secondary that silently runs on nothing"
            )
        owner = claimed.setdefault(label, model)
        if owner != model:
            raise ConfigurationError(
                f"both {owner!r} and {model!r} operate on label {label!r}. An object would "
                f"carry two output tensors and the probe reads the first it finds, so which "
                f"embedding it keeps would depend on GIE order"
            )
        ids.append(by_label[label])
    return tuple(sorted(ids))


# -- small helpers -----------------------------------------------------------------------


def _infer_dims(dims: Sequence[int]) -> str:
    return ";".join(str(int(d)) for d in dims)


def _labels_to_ids(class_labels: Mapping[int, str]) -> dict[str, int]:
    # `PipelineSettings` already refuses two ids mapping to one label, so this cannot lose one.
    return {label: class_id for class_id, label in class_labels.items()}


def _artefacts(
    artifact: ModelArtifact, *, require: bool, missing: list[Path]
) -> tuple[Path, Path | None]:
    """``(engine, onnx)`` for one model, recording what is not on this machine."""
    engine_file = str(artifact.config.parameters.get("engine_file", "model.plan"))
    engine = _artefact(artifact, engine_file, require=require, missing=missing)
    declared = artifact.config.parameters.get("onnx_file")
    onnx = (
        _artefact(artifact, str(declared), require=require, missing=missing)
        if declared
        else None
    )
    return engine, onnx


def _artefact(
    artifact: ModelArtifact, filename: str, *, require: bool, missing: list[Path]
) -> Path:
    if require:
        # `ModelArtifact.file` already names the file, the directory and what is in it.
        return artifact.file(filename).resolve()
    candidate = (artifact.path / filename).resolve()
    if not candidate.is_file():
        missing.append(candidate)
    return candidate
