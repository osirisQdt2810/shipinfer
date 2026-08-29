"""Repository scanning, version selection and error messages."""

from __future__ import annotations

from pathlib import Path

import pytest

from shipinfer.core.errors import (
    ConfigurationError,
    ModelNotFoundError,
    ModelVersionNotFoundError,
)
from shipinfer.repository import ModelRepository


def _write(root: Path, name: str, config: str, *, versioned: bool = True) -> None:
    """Write one model's config, and a version directory unless the model is an ensemble.

    An ensemble has no artefact to version, which is the property
    `test_ensembles_need_no_version_directory` exists to pin — so `versioned=False` is not a
    convenience, it is the case under test.
    """
    model = root / name
    if versioned:
        (model / "1").mkdir(parents=True)
    else:
        model.mkdir(parents=True)
    (model / "config.yaml").write_text(config.lstrip())


def _repo_with_versions(tmp_path: Path, versions: list[int], latest: int = 1) -> Path:
    root = tmp_path / "repo"
    model = root / "m"
    for version in versions:
        (model / str(version)).mkdir(parents=True)
    (model / "config.yaml").write_text(
        "platform: pytorch\nmax_batch_size: 1\n"
        "inputs: [{name: x, data_type: FP32, dims: [1]}]\n"
        "outputs: [{name: y, data_type: FP32, dims: [1]}]\n"
        "dynamic_batching: {enabled: false}\n"
        f"version_policy: {{latest: {latest}}}\n"
    )
    return root


class TestDemoRepository:
    """The repository shipped with the project must always be valid.

    Deliberately asserts *properties* rather than exact numbers. It used to pin
    `max_batch_size == 32`, which became false the moment the detector was pointed at a real
    engine — the plan is built with a static batch of 8, and the backend refuses to load when
    the config disagrees rather than silently truncating. A test that has to change every time
    an engine is rebuilt is testing the engine, not the repository.
    """

    def test_scans_the_demo_repository(self, demo_repository_path: Path) -> None:
        repo = ModelRepository.load(demo_repository_path)

        assert {"ship_detector", "person_embedder", "ship_segmenter"} <= set(repo.names())

    def test_every_shipped_model_runs_on_a_real_backend(
        self, demo_repository_path: Path
    ) -> None:
        """No stand-ins in the shipped repository.

        Every model here names a backend that executes an artefact. A `mock` platform would
        make the repository loadable anywhere at the cost of the numbers meaning nothing, and
        this project measures itself against another system — so the repository has to be the
        real thing or the comparison is not one.
        """
        repo = ModelRepository.load(demo_repository_path)
        platforms = {name: repo.entry(name).config.platform for name in repo.names()}

        assert "mock" not in platforms.values(), f"stand-in backends remain: {platforms}"

    def test_the_declared_batch_matches_what_the_engine_was_built_for(
        self, demo_repository_path: Path
    ) -> None:
        """Not a fixed number — just that the config commits to one.

        `max_batch_size` has to agree with the plan, and the TensorRT backend fails at load
        when it does not. Asserting it is positive and that the batch dimension is absent from
        `dims` is the part that belongs here; the exact value belongs to whoever built the
        engine.
        """
        repo = ModelRepository.load(demo_repository_path)
        config = repo.entry("ship_detector").config

        assert config.max_batch_size > 0
        assert config.inputs[0].dims == [
            3,
            640,
            640,
        ], "the batch dim is owned by max_batch_size"

    def test_ensembles_need_no_version_directory(self, tmp_path: Path) -> None:
        """A property of the loader, tested against a config written here.

        It used to lean on a demo ensemble, which meant deleting that demo took the coverage
        with it. The behaviour under test belongs to `ModelRepository`, so the fixture does
        too.
        """
        root = tmp_path / "repo"
        _write(
            root,
            "pipe",
            """
name: pipe
platform: ensemble
max_batch_size: 8
inputs: [{name: images, data_type: FP32, dims: [3, 4, 4]}]
outputs: [{name: score, data_type: FP32, dims: [1]}]
ensemble:
  steps:
    - model: leaf
      input_map: {images: images}
      output_map: {score: score}
""",
            versioned=False,
        )
        _write(
            root,
            "leaf",
            """
name: leaf
platform: tensorrt
max_batch_size: 8
inputs: [{name: images, data_type: FP32, dims: [3, 4, 4]}]
outputs: [{name: score, data_type: FP32, dims: [1]}]
""",
        )

        artifact = ModelRepository.load(root).resolve("pipe")

        assert artifact.config.is_ensemble
        assert artifact.version == 1


class TestVersionSelection:
    """Which version resolve() picks, and which ones the policy makes reachable at all."""

    def test_resolve_defaults_to_the_latest_version(self, tmp_path: Path) -> None:
        root = _repo_with_versions(tmp_path, [1, 2, 5], latest=1)
        artifact = ModelRepository.load(root).resolve("m")
        assert artifact.version == 5
        assert artifact.path.name == "5"

    def test_version_policy_limits_what_is_loadable(self, tmp_path: Path) -> None:
        root = _repo_with_versions(tmp_path, [1, 2, 3], latest=1)
        repo = ModelRepository.load(root)
        assert repo.entry("m").versions == (3,)
        with pytest.raises(ModelVersionNotFoundError, match="available: \\[3\\]"):
            repo.resolve("m", 1)


class TestErrorsNameWhatExists:
    """A not-found error lists what IS there, because that is what the reader needs next."""

    def test_unknown_model_lists_what_is_loaded(self, tmp_path: Path) -> None:
        repo = ModelRepository.load(_repo_with_versions(tmp_path, [1]))
        with pytest.raises(ModelNotFoundError, match="loaded: \\['m'\\]"):
            repo.entry("nope")

    def test_artifact_file_lists_what_is_present(self, tmp_path: Path) -> None:
        root = _repo_with_versions(tmp_path, [1])
        artifact = ModelRepository.load(root).resolve("m")
        (artifact.path / "present.txt").write_text("hi")
        assert artifact.file("present.txt").name == "present.txt"
        with pytest.raises(ConfigurationError, match=r"present\.txt"):
            artifact.file("absent.plan")


class TestMalformedRepository:
    """A repository that would fail later fails at load, naming the fix."""

    def test_directory_name_is_authoritative(self, tmp_path: Path) -> None:
        """A config whose ``name`` drifted from its directory is a deployment landmine."""
        root = tmp_path / "repo"
        (root / "actual" / "1").mkdir(parents=True)
        (root / "actual" / "config.yaml").write_text(
            "name: claimed\nplatform: pytorch\nmax_batch_size: 1\n"
            "inputs: [{name: x, data_type: FP32, dims: [1]}]\n"
            "outputs: [{name: y, data_type: FP32, dims: [1]}]\n"
            "dynamic_batching: {enabled: false}\n"
        )
        with pytest.raises(ConfigurationError, match="directory is authoritative"):
            ModelRepository.load(root)

    def test_a_triton_config_is_recognised_and_explained(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        (root / "m" / "1").mkdir(parents=True)
        (root / "m" / "config.pbtxt").write_text('name: "m"\n')
        with pytest.raises(ConfigurationError, match="convert it"):
            ModelRepository.load(root)

    def test_missing_version_directory_is_an_error(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        (root / "m").mkdir(parents=True)
        (root / "m" / "config.yaml").write_text(
            "platform: pytorch\nmax_batch_size: 1\n"
            "inputs: [{name: x, data_type: FP32, dims: [1]}]\n"
            "outputs: [{name: y, data_type: FP32, dims: [1]}]\n"
            "dynamic_batching: {enabled: false}\n"
        )
        with pytest.raises(ConfigurationError, match="no numbered version directory"):
            ModelRepository.load(root)
