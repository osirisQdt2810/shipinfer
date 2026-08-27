"""The DeepStream topology's child, verified on a machine with no DeepStream.

That is the point of the whole package's shape, so it is the first thing asserted here:
``gi`` and ``pyds`` are absent from this box and every module still imports. What follows is
everything that can then be checked offline — which turns out to be all of the parts that have
ever silently produced wrong output in a DeepStream deployment:

* the generated nvinfer configs, key by key, against the repository they came from;
* the refusals, each of which is a start-up failure inside a GStreamer element otherwise;
* the metadata walk, driven by a fake ``pyds`` over fake linked lists;
* the coordinate mapping — ``rect_params`` is ``(left, top, w, h)`` in *muxed* pixels and
  :attr:`~shipinfer.pipeline.schema.ObjectRecord.bbox` is ``(x1, y1, x2, y2)`` in *source*
  pixels, and a conversion that gets either half of that wrong looks almost right;
* the emission discipline, which is `PipelineRunner`'s and must stay it.
"""

from __future__ import annotations

import configparser
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError, SourceUnavailableError
from shipinfer.core.settings import ServerSettings
from shipinfer.pipeline.deepstream import (
    PR1_MISSING_STAGES,
    UNTRACKED_OBJECT_ID,
    DeepStreamPipeline,
    FrameGeometry,
    FrameNumbering,
    FrameView,
    MetadataProbe,
    ObjectView,
    build_event,
    labels_file,
    make,
    source_uri,
    walk_batch,
)
from shipinfer.pipeline.metrics import PipelineMetrics
from shipinfer.pipeline.schema import PerceptionEvent
from shipinfer.pipeline.sinks import NullResultSink
from shipinfer.repository import ModelRepository

pytestmark = pytest.mark.timeout(60)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = ("ship_detector", "person_embedder", "ship_embedder", "ship_segmenter")
CAMERAS = [
    {"camera_id": "cam0", "uri": "rtsp://host/cam0", "fps": 20.0},
    {"camera_id": "cam1", "uri": "rtsp://host/cam1", "fps": 20.0},
]
#: A parser has to be configured for the shipped end-to-end detector, and generation refuses
#: without one — see `TestTheRefusalsHappenOffline`.
PARSER = {"bbox_parser": "NvDsInferParseYolo26", "custom_lib": "/opt/ds/libyolo26.so"}


# -- fixtures -----------------------------------------------------------------------------


@pytest.fixture
def repository(tmp_path: Path) -> ModelRepository:
    """A copy of the shipped repository with empty engines beside the real configs.

    The real ``model.plan`` files are gitignored — an engine is valid only for the GPU and
    TensorRT version that built it — so CI has none. The *configs* are the input under test
    here, and they are the real ones rather than fixtures written to match: a config.yaml that
    changes shape should break this.
    """
    root = tmp_path / "model_repository"
    for name in MODELS:
        (root / name / "1").mkdir(parents=True)
        (root / name / "config.yaml").write_bytes(
            (REPO_ROOT / "model_repository" / name / "config.yaml").read_bytes()
        )
        (root / name / "1" / "model.plan").write_bytes(b"")
    return ModelRepository.load(root)


def settings_for(**deepstream: object) -> ServerSettings:
    return ServerSettings(
        model_repository=Path("model_repository"),
        ingest={"cameras": CAMERAS},
        topology={"kind": "deepstream", "deepstream": {**PARSER, **deepstream}},
    )


def generate(repository: ModelRepository, root: Path, **deepstream: object):
    from shipinfer.pipeline.deepstream import write_configs

    settings = settings_for(**deepstream)
    return write_configs(
        repository,
        settings=settings,
        shard_index=2,
        gpu_id=0,
        cameras=settings.ingest.cameras,
        root=root,
    )


def read(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(inline_comment_prefixes=None)
    parser.read_string(path.read_text(encoding="utf-8"))
    return parser


def tree_digest(root: Path) -> list[tuple[str, str]]:
    return sorted(
        (
            str(path.relative_to(root)),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    )


# -- import purity -------------------------------------------------------------------------


class TestItImportsWithNoDeepStream:
    """The promise the package's shape exists to keep: `gi` and `pyds` are start-up
    dependencies, never import-time ones."""

    def test_every_module_imports_with_gi_and_pyds_blocked(self) -> None:
        code = """
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'gi', 'pyds'}:
            raise ImportError(f'{name} is blocked for this test')
        return None

sys.meta_path.insert(0, _Block())
import shipinfer.pipeline.deepstream
import shipinfer.pipeline.deepstream.configs
import shipinfer.pipeline.deepstream.probe
import shipinfer.pipeline.deepstream.builder
import shipinfer.pipeline.deepstream.run
import shipinfer.runtime.gstreamer
assert 'gi' not in sys.modules and 'pyds' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

        assert result.returncode == 0, result.stdout + result.stderr

    def test_the_missing_bindings_are_a_typed_error_with_a_hint(self, monkeypatch) -> None:
        from shipinfer.runtime import gstreamer

        monkeypatch.setattr(gstreamer, "load_gst", lambda: (object(), object()))
        monkeypatch.setitem(sys.modules, "pyds", None)  # `import pyds` -> ImportError

        with pytest.raises(SourceUnavailableError, match="DeepStream SDK"):
            gstreamer.load_pyds()

    def test_a_missing_plugin_is_a_typed_error_naming_the_element(self) -> None:
        class _Gst:
            class ElementFactory:
                @staticmethod
                def make(factory: str, name: str) -> None:
                    return None

        with pytest.raises(SourceUnavailableError, match=r"nvinfer.*not installed"):
            make(_Gst, "nvinfer", "shipinfer_pgie")


# -- config generation ---------------------------------------------------------------------


class TestTheRepositoryGeneratesTheNvinferConfigs:
    def test_the_primary_gie_is_the_detector_as_the_repository_describes_it(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        configs = generate(repository, tmp_path / "run")

        pgie = read(configs.pgie)["property"]
        assert pgie["gpu-id"] == "0"
        assert pgie["batch-size"] == "2"  # one slot per camera this shard muxes
        assert pgie["infer-dims"] == "3;640;640"
        assert pgie["network-mode"] == "2"  # fp16
        assert pgie["num-detected-classes"] == "9"  # max(class_labels) + 1
        assert pgie["cluster-mode"] == "4"  # the engine already applied NMS
        assert pgie["output-blob-names"] == "output0"
        assert pgie["process-mode"] == "1" and pgie["network-type"] == "0"
        assert pgie["gie-unique-id"] == "1"
        assert pgie["parse-bbox-func-name"] == PARSER["bbox_parser"]
        assert pgie["custom-lib-path"] == PARSER["custom_lib"]
        engine = Path(pgie["model-engine-file"])
        assert engine.is_absolute() and engine == (
            repository.resolve("ship_detector").path / "model.plan"
        )

    def test_the_thresholds_come_from_the_pipeline_section(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        attrs = read(generate(repository, tmp_path / "run").pgie)["class-attrs-all"]

        assert float(attrs["pre-cluster-threshold"]) == 0.25
        assert int(attrs["topk"]) == 100

    def test_the_label_file_is_positional_with_the_gaps_named(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        configs = generate(repository, tmp_path / "run")

        labels = configs.labels.read_text().splitlines()
        assert labels[0] == "person" and labels[8] == "ship"
        assert set(labels[1:8]) == {"unknown"}
        assert len(labels) == 9

    def test_labels_file_refuses_an_empty_map(self) -> None:
        with pytest.raises(ConfigurationError, match="class_labels is empty"):
            labels_file({})

    def test_each_secondary_runs_on_its_own_class_and_carries_its_tensor(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        configs = generate(repository, tmp_path / "run")

        assert [name for name, _ in configs.sgies] == ["person_embedder", "ship_embedder"]
        person = read(dict(configs.sgies)["person_embedder"])["property"]
        ship = read(dict(configs.sgies)["ship_embedder"])["property"]

        for sgie in (person, ship):
            assert sgie["process-mode"] == "2"  # objects, not frames
            assert sgie["network-type"] == "100"  # no built-in postprocessing
            assert sgie["output-tensor-meta"] == "1"  # the embedding is reachable at all
            assert sgie["operate-on-gie-id"] == "1"
            assert sgie["batch-size"] == "16"
            assert sgie["infer-dims"] == "3;256;128"
        # Labels resolved through pipeline.class_labels: person is 0, ship is 8.
        assert person["operate-on-class-ids"] == "0"
        assert ship["operate-on-class-ids"] == "8"
        # 1 is the primary; the secondaries take the next ids, in the order they run.
        assert (person["gie-unique-id"], ship["gie-unique-id"]) == ("2", "3")

    def test_a_retrained_detectors_ids_move_the_class_filter_with_them(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        """The filter is written in labels precisely so a new checkpoint's numbering is a
        settings change and not a hand-edited `.txt`."""
        from shipinfer.pipeline.deepstream import write_configs

        settings = ServerSettings(
            ingest={"cameras": CAMERAS},
            pipeline={"class_labels": {3: "person", 4: "ship"}},
            topology={"kind": "deepstream", "deepstream": PARSER},
        )
        configs = write_configs(
            repository,
            settings=settings,
            shard_index=0,
            gpu_id=0,
            cameras=settings.ingest.cameras,
            root=tmp_path / "run",
        )

        assert (
            read(dict(configs.sgies)["person_embedder"])["property"]["operate-on-class-ids"]
            == "3"
        )
        assert read(configs.pgie)["property"]["num-detected-classes"] == "5"

    def test_everything_lands_under_the_shards_own_directory(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        configs = generate(repository, tmp_path / "run")

        assert configs.root == (tmp_path / "run" / "shard2").resolve()
        for path in configs.paths():
            assert path.parent == configs.root, path

    def test_the_model_repository_is_byte_identical_afterwards(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        """Generated configs live outside the repository. `ModelRepository` refuses a stray
        Triton config in a model directory, and a `.txt` beside a `config.yaml` is noise in
        every future diff."""
        before = tree_digest(repository.root)

        generate(repository, tmp_path / "run")

        assert tree_digest(repository.root) == before

    def test_the_generated_tracker_config_carries_this_deployments_opinions(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        text = generate(repository, tmp_path / "run").tracker.read_text()

        assert text.startswith("%YAML:1.0")
        assert "minDetectorConfidence: 0.25" in text  # pipeline.score_threshold
        assert "maxTargetsPerStream: 100" in text  # pipeline.max_detections

    def test_an_operators_tracker_config_is_used_where_it_lives(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        theirs = tmp_path / "config_tracker_NvDCF_perf.yml"
        theirs.write_text("%YAML:1.0\n")

        configs = generate(repository, tmp_path / "run", tracker_config=str(theirs))

        assert configs.tracker == theirs
        assert not (configs.root / "tracker.yml").exists()

    def test_an_onnx_beside_the_engine_is_offered_to_nvinfer(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        """An engine is only valid for the TensorRT version that built it; with an ONNX
        declared, nvinfer rebuilds instead of refusing to start."""
        detector = repository.resolve("ship_detector")
        (detector.path / "model.onnx").write_bytes(b"")
        object.__setattr__(
            detector.config,
            "parameters",
            {**detector.config.parameters, "onnx_file": "model.onnx"},
        )

        pgie = read(generate(repository, tmp_path / "run").pgie)["property"]

        assert Path(pgie["onnx-file"]) == detector.path / "model.onnx"

    def test_a_dry_run_names_what_is_not_on_this_machine_instead_of_refusing(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        from shipinfer.pipeline.deepstream import write_configs

        (repository.resolve("ship_detector").path / "model.plan").unlink()
        settings = settings_for()

        strict = pytest.raises(ConfigurationError, match=r"'model.plan' not found")
        with strict:
            write_configs(
                repository,
                settings=settings,
                shard_index=0,
                gpu_id=0,
                cameras=settings.ingest.cameras,
                root=tmp_path / "run",
            )

        configs = write_configs(
            repository,
            settings=settings,
            shard_index=0,
            gpu_id=0,
            cameras=settings.ingest.cameras,
            root=tmp_path / "run",
            require_engine=False,
        )
        assert configs.missing_files == (
            repository.resolve("ship_detector").path.resolve() / "model.plan",
        )


class TestTheRefusalsHappenOffline:
    """Each of these is a start-up failure inside a GStreamer element on a GPU host, or worse:
    a graph that runs and publishes nothing."""

    def test_an_end_to_end_detector_with_no_parser_names_the_gap_and_the_shape(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            generate(repository, tmp_path / "run", bbox_parser="", custom_lib="")

        message = str(excinfo.value)
        assert "parse-bbox-func-name" in message  # what to set
        assert "[300, 6]" in message  # and what nvinfer cannot read
        assert "zero detections" in message  # and how it fails if you do not

    def test_an_unknown_operate_on_label_lists_the_ones_that_exist(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        with pytest.raises(ConfigurationError, match=r"'boat'.*\['person', 'ship'\]"):
            generate(
                repository,
                tmp_path / "run",
                operate_on={"person_embedder": ["person"], "ship_embedder": ["boat"]},
            )

    def test_two_secondaries_claiming_one_label_is_refused_not_resolved(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        """Both would attach a tensor to the same object and the probe reads the first it
        finds, so which embedding survives would depend on GIE order."""
        with pytest.raises(ConfigurationError, match=r"both .* operate on label 'ship'"):
            generate(
                repository,
                tmp_path / "run",
                operate_on={"person_embedder": ["ship"], "ship_embedder": ["ship"]},
            )

    def test_a_model_that_is_not_a_tensorrt_model_is_refused(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        object.__setattr__(repository.entry("ship_detector").config, "platform", "onnx")

        with pytest.raises(ConfigurationError, match=r"platform 'onnx'.*nvinfer reads"):
            generate(repository, tmp_path / "run")

    def test_a_secondary_that_cannot_batch_is_refused(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        object.__setattr__(repository.entry("person_embedder").config, "max_batch_size", 0)

        with pytest.raises(ConfigurationError, match=r"max_batch_size: 0"):
            generate(repository, tmp_path / "run")

    def test_a_multi_output_model_used_as_a_secondary_needs_one_image_input(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        """The segmenter's *inputs* are fine; what is refused here is a second input, which
        nvinfer has no way to feed."""
        config = repository.entry("person_embedder").config
        object.__setattr__(config, "inputs", [*config.inputs, *config.inputs])

        with pytest.raises(ConfigurationError, match="declares 2 inputs"):
            generate(repository, tmp_path / "run")

    def test_a_shard_with_no_cameras_is_refused(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        from shipinfer.pipeline.deepstream import write_configs

        with pytest.raises(ConfigurationError, match="batch-size 0"):
            write_configs(
                repository,
                settings=settings_for(),
                shard_index=0,
                gpu_id=0,
                cameras=[],
                root=tmp_path / "run",
            )

    def test_a_uri_nvurisrcbin_cannot_open_is_refused_at_config_time(self) -> None:
        assert source_uri("rtsp://host/cam0") == "rtsp://host/cam0"
        assert source_uri("/videos/harbour.mp4") == "file:///videos/harbour.mp4"

        with pytest.raises(ConfigurationError, match="scheme 'sftp'"):
            source_uri("sftp://host/harbour.mp4")


# -- the metadata walk ----------------------------------------------------------------------


class FakeNode:
    """One link of a `pyds` linked list. `.next` raises StopIteration at the end, as pyds' does."""

    def __init__(self, data: object, following: FakeNode | None, *, stop: bool = False) -> None:
        self.data = data
        self._next = following
        self._stop = stop

    @property
    def next(self) -> FakeNode | None:
        if self._stop:
            raise StopIteration
        return self._next


def chain(items: list[object], *, stop_after: int | None = None) -> FakeNode | None:
    node: FakeNode | None = None
    for index in reversed(range(len(items))):
        node = FakeNode(items[index], node, stop=stop_after is not None and index == stop_after)
    return node


class FakeRect:
    def __init__(self, left: float, top: float, width: float, height: float) -> None:
        self.left, self.top, self.width, self.height = left, top, width, height


class FakeObject:
    def __init__(
        self,
        *,
        class_id: int,
        confidence: float,
        rect: tuple[float, float, float, float],
        object_id: int,
        embedding: np.ndarray | None = None,
        tracker_confidence: float = 0.0,
    ) -> None:
        self.class_id = class_id
        self.confidence = confidence
        self.tracker_confidence = tracker_confidence
        self.rect_params = FakeRect(*rect)
        self.object_id = object_id
        self._embedding = embedding
        self.obj_user_meta_list = (
            chain([_tensor_meta(embedding)]) if embedding is not None else None
        )


class FakeFrame:
    def __init__(
        self,
        *,
        pad_index: int,
        frame_num: int,
        objects: list[FakeObject],
        source: tuple[int, int] = (3840, 2160),
        ntp: int = 0,
        stop_after: int | None = None,
    ) -> None:
        self.pad_index = pad_index
        self.frame_num = frame_num
        self.source_frame_width, self.source_frame_height = source
        self.ntp_timestamp = ntp
        self.obj_meta_list = chain(list(objects), stop_after=stop_after)


class _Layer:
    def __init__(self, array: np.ndarray) -> None:
        self.buffer = array  # `get_ptr` turns it into an address, as pyds does
        self.dims = types.SimpleNamespace(numElements=array.size)
        self.isInput = False


def _tensor_meta(array: np.ndarray) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        base_meta=types.SimpleNamespace(meta_type="NVDSINFER_TENSOR_OUTPUT_META"),
        user_meta_data=types.SimpleNamespace(num_output_layers=1, layer=_Layer(array)),
    )


class FakePyds:
    """Enough of `pyds` to drive the real walk. Everything it returns is a real object, and
    `get_ptr` a real address, so no branch in the code under test is skipped."""

    NvDsMetaType = types.SimpleNamespace(
        NVDSINFER_TENSOR_OUTPUT_META="NVDSINFER_TENSOR_OUTPUT_META"
    )

    class NvDsFrameMeta:
        cast = staticmethod(lambda data: data)

    class NvDsObjectMeta:
        cast = staticmethod(lambda data: data)

    class NvDsUserMeta:
        cast = staticmethod(lambda data: data)

    class NvDsInferTensorMeta:
        cast = staticmethod(lambda data: data)

    @staticmethod
    def get_nvds_LayerInfo(tensor_meta: object, index: int) -> _Layer:
        assert index == 0
        return tensor_meta.layer

    @staticmethod
    def get_ptr(buffer: np.ndarray) -> int:
        return buffer.ctypes.data


def batch_of(frames: list[FakeFrame], *, stop_after: int | None = None) -> object:
    return types.SimpleNamespace(frame_meta_list=chain(list(frames), stop_after=stop_after))


def an_object(**kwargs: object) -> FakeObject:
    defaults = {
        "class_id": 0,
        "confidence": 0.9,
        "rect": (100.0, 200.0, 50.0, 80.0),
        "object_id": 7,
    }
    return FakeObject(**{**defaults, **kwargs})


class TestTheWalkReadsWhatTheGraphWrote:
    def test_one_frame_view_per_frame_with_its_objects_in_order(self) -> None:
        batch = batch_of(
            [
                FakeFrame(pad_index=0, frame_num=11, objects=[an_object(class_id=0)]),
                FakeFrame(
                    pad_index=1,
                    frame_num=12,
                    objects=[an_object(class_id=8), an_object(class_id=0)],
                ),
            ]
        )

        views = walk_batch(FakePyds, batch)

        assert [v.pad_index for v in views] == [0, 1]
        assert [v.frame_num for v in views] == [11, 12]
        assert [o.class_id for o in views[1].objects] == [8, 0]

    def test_the_embedding_comes_out_of_the_tensor_meta(self) -> None:
        vector = np.arange(8, dtype=np.float32) / 8.0
        batch = batch_of(
            [FakeFrame(pad_index=0, frame_num=1, objects=[an_object(embedding=vector)])]
        )

        [view] = walk_batch(FakePyds, batch)

        assert view.objects[0].embedding == pytest.approx(tuple(vector.tolist()))

    def test_an_object_no_secondary_ran_on_has_an_empty_embedding(self) -> None:
        batch = batch_of([FakeFrame(pad_index=0, frame_num=1, objects=[an_object()])])

        [view] = walk_batch(FakePyds, batch)

        assert view.objects[0].embedding == ()

    def test_a_stopiteration_ends_the_walk_without_losing_what_was_collected(self) -> None:
        """pyds raises StopIteration at the end of a list. Discarding the frames already read
        would lose real detections to a list-walking accident."""
        batch = batch_of(
            [
                FakeFrame(pad_index=0, frame_num=1, objects=[an_object()]),
                FakeFrame(pad_index=1, frame_num=1, objects=[an_object()]),
                FakeFrame(pad_index=2, frame_num=1, objects=[an_object()]),
            ],
            stop_after=1,
        )

        views = walk_batch(FakePyds, batch)

        assert [v.pad_index for v in views] == [0, 1]

    def test_a_truncated_object_list_keeps_the_objects_before_it(self) -> None:
        batch = batch_of(
            [
                FakeFrame(
                    pad_index=0,
                    frame_num=1,
                    objects=[an_object(class_id=0), an_object(class_id=8)],
                    stop_after=0,
                )
            ]
        )

        [view] = walk_batch(FakePyds, batch)

        assert [o.class_id for o in view.objects] == [0]

    def test_a_tracker_published_box_keeps_a_usable_confidence(self) -> None:
        """nvinfer sets `confidence` to -1 on a box the detector did not produce this frame; a
        negative score in an event inverts every consumer's threshold."""
        batch = batch_of(
            [
                FakeFrame(
                    pad_index=0,
                    frame_num=1,
                    objects=[an_object(confidence=-0.1, tracker_confidence=0.55)],
                )
            ]
        )

        [view] = walk_batch(FakePyds, batch)

        assert view.objects[0].confidence == pytest.approx(0.55)


# -- geometry -------------------------------------------------------------------------------


class TestBoxesComeBackInSourcePixels:
    def test_a_stretched_mux_scales_linearly_per_axis(self) -> None:
        geometry = FrameGeometry(
            mux_width=1920, mux_height=1080, source_width=3840, source_height=2160
        )

        assert geometry.to_source(100.0, 200.0, 50.0, 80.0) == (200.0, 400.0, 300.0, 560.0)

    def test_a_padded_mux_undoes_the_letterbox_before_the_scale(self) -> None:
        # A 1000x1000 source into a 1920x1080 mux: scale = min(1.92, 1.08) = 1.08, so the
        # frame occupies 1080x1080 centred, with (1920-1080)/2 = 420 px of padding each side.
        geometry = FrameGeometry(
            mux_width=1920,
            mux_height=1080,
            source_width=1000,
            source_height=1000,
            padded=True,
        )

        x1, y1, x2, y2 = geometry.to_source(420.0 + 108.0, 216.0, 108.0, 108.0)

        assert (x1, y1) == pytest.approx((100.0, 200.0))
        assert (x2, y2) == pytest.approx((200.0, 300.0))

    def test_boxes_are_clamped_to_the_source_frame(self) -> None:
        geometry = FrameGeometry(
            mux_width=1920, mux_height=1080, source_width=1920, source_height=1080
        )

        assert geometry.to_source(-40.0, 1000.0, 100.0, 400.0) == (0.0, 1000.0, 60.0, 1080.0)

    def test_an_unknown_source_size_leaves_the_coordinates_alone(self) -> None:
        """Scaling by an unknown is worse than not scaling: the event carries width 0, which is
        visibly missing rather than quietly halved."""
        geometry = FrameGeometry(
            mux_width=1920, mux_height=1080, source_width=0, source_height=0
        )

        assert geometry.to_source(10.0, 20.0, 30.0, 40.0) == (10.0, 20.0, 40.0, 60.0)


# -- the event ------------------------------------------------------------------------------


LABELS = {0: "person", 8: "ship"}
GEOMETRY = FrameGeometry(mux_width=1920, mux_height=1080, source_width=3840, source_height=2160)


def a_view(*objects: ObjectView, ntp: int = 0, frame_num: int = 5) -> FrameView:
    return FrameView(
        pad_index=0,
        frame_num=frame_num,
        source_width=3840,
        source_height=2160,
        ntp_timestamp_ns=ntp,
        objects=objects,
    )


def an_event(*objects: ObjectView, ntp: int = 0, epoch_offset_ns: int = 0) -> PerceptionEvent:
    return build_event(
        a_view(*objects, ntp=ntp),
        camera_id="cam0",
        source_id="shipinfer",
        labels=LABELS,
        geometry=GEOMETRY,
        fps=20.0,
        frame_id=5,
        epoch_offset_ns=epoch_offset_ns,
        missing_stages=PR1_MISSING_STAGES,
    )


class TestOneEventPerFrame:
    def test_the_bbox_is_corners_in_source_pixels_not_origin_and_extents(self) -> None:
        """The regression this test exists for: `rect_params` is (left, top, w, h) in muxed
        pixels. Publishing it unchanged halves every box on a 4K camera *and* puts extents in
        a field every consumer reads as corners."""
        event = an_event(
            ObjectView(class_id=0, confidence=0.9, rect=(100.0, 200.0, 50.0, 80.0), object_id=7)
        )

        assert event.objects[0].bbox == (200.0, 400.0, 300.0, 560.0)

    def test_class_ids_become_labels_and_an_unmapped_one_is_unknown(self) -> None:
        event = an_event(
            ObjectView(class_id=0, confidence=0.9, rect=(0.0, 0.0, 1.0, 1.0), object_id=1),
            ObjectView(class_id=8, confidence=0.8, rect=(0.0, 0.0, 1.0, 1.0), object_id=2),
            ObjectView(class_id=3, confidence=0.7, rect=(0.0, 0.0, 1.0, 1.0), object_id=3),
        )

        assert [o.class_name for o in event.objects] == ["person", "ship", "unknown"]

    def test_det_ids_are_camera_frame_index(self) -> None:
        event = an_event(
            ObjectView(class_id=0, confidence=0.9, rect=(0.0, 0.0, 1.0, 1.0), object_id=1),
            ObjectView(class_id=0, confidence=0.9, rect=(0.0, 0.0, 1.0, 1.0), object_id=2),
        )

        assert [o.det_id for o in event.objects] == ["cam0_5_0", "cam0_5_1"]

    def test_an_untracked_object_has_no_track_id_and_no_state(self) -> None:
        event = an_event(
            ObjectView(
                class_id=0,
                confidence=0.9,
                rect=(0.0, 0.0, 1.0, 1.0),
                object_id=UNTRACKED_OBJECT_ID,
            ),
            ObjectView(class_id=0, confidence=0.9, rect=(0.0, 0.0, 1.0, 1.0), object_id=42),
        )

        assert (event.objects[0].track_id, event.objects[0].track_state) == (None, None)
        assert (event.objects[1].track_id, event.objects[1].track_state) == (42, "tracked")

    def test_every_pr1_event_says_which_stages_never_ran(self) -> None:
        event = an_event()

        assert event.missing_stages == ("ship_segmenter", "ship_recognizer")
        assert event.is_partial and event.as_dict()["partial"] is True

    def test_the_capture_time_is_the_sources_and_the_latency_is_monotonic(self) -> None:
        import time

        offset = time.time_ns() - time.monotonic_ns()
        ntp = time.time_ns() - 30_000_000  # captured 30 ms ago

        event = an_event(ntp=ntp, epoch_offset_ns=offset)

        assert event.captured_unix_ns == ntp
        assert 20_000 <= event.latency_us <= 200_000

    def test_a_source_with_no_sender_report_yet_reports_no_latency(self) -> None:
        """NTP 0 must not read as "captured in 1970" and produce a 56-year latency."""
        event = an_event(ntp=0, epoch_offset_ns=10**18)

        assert (event.captured_unix_ns, event.latency_us) == (0, 0)

    def test_the_event_carries_every_v1_key_the_deployed_consumer_reads(self) -> None:
        """One sink serves every topology, so this event has to be that event."""
        payload = an_event(
            ObjectView(class_id=0, confidence=0.9, rect=(0.0, 0.0, 1.0, 1.0), object_id=1),
            ObjectView(class_id=8, confidence=0.8, rect=(0.0, 0.0, 1.0, 1.0), object_id=2),
        ).as_dict()

        for key in (
            "sub_id",
            "det_id_vec",
            "camera_id",
            "image_id",
            "det_body_score_vec",
            "body_bbox_vec",
            "body_feature_vec",
            "img_width",
            "img_height",
            "img_fps",
            "body_track_id_vec",
            "ship_bbox_vec",
            "ship_feature_vec",
            "ship_track_id_vec",
            "missing_stages",
        ):
            assert key in payload, key
        assert payload["img_width"] == 3840 and payload["img_fps"] == 20
        assert len(payload["body_bbox_vec"]) == 1 and len(payload["ship_bbox_vec"]) == 1
        json.dumps(payload)  # the sink serialises it; nothing here may be unserialisable


class TestFrameIdsSurviveAReconnect:
    def test_ids_keep_going_up_when_the_source_restarts_at_zero(self) -> None:
        numbering = FrameNumbering()

        first = [numbering.next("cam0", n) for n in (0, 1, 2, 3)]
        after_reconnect = [numbering.next("cam0", n) for n in (0, 1, 2)]

        assert first == [0, 1, 2, 3]
        assert after_reconnect == [4, 5, 6]

    def test_cameras_are_numbered_independently(self) -> None:
        numbering = FrameNumbering()

        assert numbering.next("cam0", 100) == 100
        assert numbering.next("cam1", 0) == 0


# -- emission -------------------------------------------------------------------------------


class FakeGst:
    PadProbeReturn = types.SimpleNamespace(OK="OK")


class FakeInfo:
    def __init__(self, buffer: object) -> None:
        self._buffer = buffer

    def get_buffer(self) -> object:
        return self._buffer


class BatchPyds(FakePyds):
    """`FakePyds` plus the buffer -> batch-meta lookup the probe does."""

    batches: dict[int, object] = {}

    @classmethod
    def gst_buffer_get_nvds_batch_meta(cls, address: int) -> object:
        return cls.batches.get(address)


class RefusingSink(NullResultSink):
    """A sink that accepts nothing — a broker whose DNS stopped resolving."""

    def _do_emit(self, event: PerceptionEvent) -> None:
        raise RuntimeError("no broker")


class LateFailureSink(NullResultSink):
    """A sink that accepts synchronously and reports a refusal for an *earlier* message."""

    def __init__(self) -> None:
        super().__init__()
        self.pending: list[tuple[str, int]] = [("cam9", 3)]

    def drain_delivery_failures(self) -> tuple[tuple[str, int], ...]:
        drained, self.pending = tuple(self.pending), []
        return drained


def probe_over(sink, **kwargs) -> MetadataProbe:
    settings = settings_for()
    return MetadataProbe(
        gst=FakeGst,
        pyds=BatchPyds,
        sink=sink,
        metrics=PipelineMetrics(),
        settings=settings,
        camera_by_pad=kwargs.pop("camera_by_pad", {0: "cam0", 1: "cam1"}),
        cameras={c.camera_id: c for c in settings.ingest.cameras},
        **kwargs,
    )


def deliver(probe: MetadataProbe, batch: object) -> object:
    buffer = object()
    BatchPyds.batches = {hash(buffer): batch}
    try:
        return probe.on_buffer(None, FakeInfo(buffer))
    finally:
        BatchPyds.batches = {}


class TestTheProbePublishesAndNeverRaises:
    def test_a_batch_becomes_one_event_per_frame(self) -> None:
        sink = NullResultSink(keep_last=4)
        probe = probe_over(sink)

        result = deliver(
            probe,
            batch_of(
                [
                    FakeFrame(pad_index=0, frame_num=1, objects=[an_object()]),
                    FakeFrame(pad_index=1, frame_num=1, objects=[an_object(class_id=8)]),
                ]
            ),
        )

        assert result == "OK"
        assert [e.camera_id for e in sink.events()] == ["cam0", "cam1"]
        assert sink.events()[0].missing_stages == PR1_MISSING_STAGES
        assert probe.metrics.frames_emitted.value(camera="cam0") == 1.0
        assert probe.metrics.objects_per_frame.snapshot(camera="cam1") == (1, 1.0)

    def test_a_refused_event_is_counted_against_the_sink_and_not_emitted(self) -> None:
        probe = probe_over(RefusingSink())

        deliver(probe, batch_of([FakeFrame(pad_index=0, frame_num=1, objects=[an_object()])]))

        metrics = probe.metrics
        assert metrics.sink_failures.value(sink="null") == 1.0
        assert metrics.frames_emitted.total() == 0.0

    def test_a_late_refusal_is_counted_with_the_tag_it_belongs_to(self) -> None:
        """An asynchronous verdict about an earlier message must not decide this frame's
        outcome — this frame was published."""
        probe = probe_over(LateFailureSink())

        deliver(probe, batch_of([FakeFrame(pad_index=0, frame_num=1, objects=[an_object()])]))

        assert probe.metrics.sink_failures.value(sink="null") == 1.0
        assert probe.metrics.frames_emitted.value(camera="cam0") == 1.0

    def test_a_frame_on_an_unknown_pad_is_skipped_and_counted(self) -> None:
        sink = NullResultSink(keep_last=4)
        probe = probe_over(sink)

        deliver(
            probe,
            batch_of(
                [
                    FakeFrame(pad_index=9, frame_num=1, objects=[an_object()]),
                    FakeFrame(pad_index=0, frame_num=1, objects=[an_object()]),
                ]
            ),
        )

        assert probe.unknown_pad_frames == 1
        assert [e.camera_id for e in sink.events()] == ["cam0"]

    def test_a_broken_batch_returns_ok_and_is_counted_as_ours(self) -> None:
        """An exception on a streaming thread is swallowed by the C caller: the buffer is
        dropped and nothing is said. So it is caught, counted and OK is returned."""
        probe = probe_over(NullResultSink())
        broken = types.SimpleNamespace()  # no `frame_meta_list` at all

        result = deliver(probe, broken)

        assert result == "OK"
        assert probe.metrics.build_failures.value(camera="unknown") == 1.0

    def test_a_buffer_with_no_batch_meta_is_not_an_error(self) -> None:
        probe = probe_over(NullResultSink())

        assert probe.on_buffer(None, FakeInfo(object())) == "OK"
        assert probe.metrics.build_failures.total() == 0.0


# -- the process ----------------------------------------------------------------------------


class TestTheShardsProcess:
    def test_it_refuses_a_configuration_with_no_cameras(self) -> None:
        settings = ServerSettings(topology={"kind": "deepstream", "deepstream": PARSER})

        with pytest.raises(ConfigurationError, match="no enabled cameras"):
            DeepStreamPipeline(settings)

    def test_it_refuses_more_than_one_visible_gpu(self) -> None:
        settings = ServerSettings(
            ingest={"cameras": CAMERAS},
            devices={"visible_gpus": [0, 1]},
            topology={"kind": "deepstream", "deepstream": PARSER},
        )

        with pytest.raises(ConfigurationError, match="one graph on one GPU"):
            DeepStreamPipeline(settings)

    def test_the_gpu_is_the_first_device_this_process_can_see(self, tmp_path: Path) -> None:
        """Under `fleet` that is logical 0; run by hand with `--gpus 3` it is 3, and every
        element in the generated configs is given it."""
        settings = ServerSettings(
            model_repository=tmp_path / "model_repository",
            ingest={"cameras": CAMERAS},
            devices={"visible_gpus": [3]},
            topology={"kind": "deepstream", "deepstream": {**PARSER, "shard": 1}},
        )

        pipeline = DeepStreamPipeline(settings, config_root=tmp_path / "run")

        assert (pipeline.gpu_id, pipeline.shard_index) == (3, 1)

    def test_write_configs_needs_no_gpu_and_no_gstreamer(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        settings = ServerSettings(
            model_repository=repository.root,
            ingest={"cameras": CAMERAS},
            topology={"kind": "deepstream", "deepstream": {**PARSER, "shard": 1}},
        )

        configs = DeepStreamPipeline(
            settings, sink=NullResultSink(), config_root=tmp_path / "run"
        ).write_configs()

        assert configs.root == (tmp_path / "run" / "shard1").resolve()
        assert read(configs.pgie)["property"]["gpu-id"] == "0"


class TestDryRunTouchesNoSink:
    def test_a_dry_construction_leaves_a_live_results_file_byte_identical(
        self, tmp_path: Path
    ) -> None:
        """#32 round 1, reproduced by the review: constructing the pipeline for --dry-run
        built the configured sink, and jsonlines truncates its file in __init__ — a dry run
        on the box that is also collecting results destroyed them. The sink is lazy now:
        construction touches nothing; only start() builds it."""
        from shipinfer.pipeline.deepstream.run import DeepStreamPipeline

        live = tmp_path / "live-results.jsonl"
        live.write_text("PRE-EXISTING LIVE DATA\n")
        settings = ServerSettings(
            model_repository=Path("model_repository"),
            ingest={"cameras": CAMERAS},
            pipeline={
                "result_sink": "jsonlines",
                "result_sink_options": {"path": str(live), "append": False},
            },
            topology={"kind": "deepstream", "deepstream": dict(PARSER)},
        )
        pipeline = DeepStreamPipeline(settings, config_root=tmp_path / "cfg")
        assert live.read_text() == "PRE-EXISTING LIVE DATA\n", "a dry construction wrote"
        pipeline.stop()  # idempotent, and must not build a sink just to close it
        assert live.read_text() == "PRE-EXISTING LIVE DATA\n"


class TestEverySecondaryClaimsItsLabels:
    def test_the_settings_refuse_an_unfiltered_secondary(self) -> None:
        """#32 round 1: a secondary absent from operate_on ran on EVERY class, and the
        probe published whichever tensor came first — a person with a ship's embedding."""
        from shipinfer.core.settings.topology import DeepStreamSettings

        with pytest.raises(ValueError, match="no operate_on entry"):
            DeepStreamSettings(operate_on={"person_embedder": ["person"]})

    def test_write_configs_refuses_a_hand_built_unfiltered_secondary(
        self, repository: ModelRepository, tmp_path: Path
    ) -> None:
        """Defense in depth for a settings object built past validation: config generation
        itself refuses the claim-all secondary, naming the knob and the known labels."""
        from shipinfer.core.settings.topology import DeepStreamSettings
        from shipinfer.pipeline.deepstream import write_configs

        settings = settings_for()
        settings.topology.deepstream = DeepStreamSettings.model_construct(
            **{
                **settings.topology.deepstream.model_dump(),
                "operate_on": {"person_embedder": ["person"]},
            }
        )
        with pytest.raises(ConfigurationError, match="no operate_on labels"):
            write_configs(
                repository,
                settings=settings,
                shard_index=0,
                gpu_id=0,
                cameras=settings.ingest.cameras,
                root=tmp_path,
            )
