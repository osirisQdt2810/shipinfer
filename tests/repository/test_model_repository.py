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


def _repo_with_versions(tmp_path: Path, versions: list[int], latest: int = 1) -> Path:
    root = tmp_path / "repo"
    model = root / "m"
    for version in versions:
        (model / str(version)).mkdir(parents=True)
    (model / "config.yaml").write_text(
        "platform: mock\nmax_batch_size: 1\n"
        "inputs: [{name: x, data_type: FP32, dims: [1]}]\n"
        "outputs: [{name: y, data_type: FP32, dims: [1]}]\n"
        "dynamic_batching: {enabled: false}\n"
        f"version_policy: {{latest: {latest}}}\n"
    )
    return root


class TestDemoRepository:
    """The repository shipped with the project loads, ensembles and all."""

    def test_scans_the_demo_repository(self, demo_repository_path: Path) -> None:
        """The repository shipped with the project must always be valid."""
        repo = ModelRepository.load(demo_repository_path)
        assert {"ship_detector", "person_embedder", "ship_pipeline"} <= set(repo.names())
        assert repo.entry("ship_detector").config.max_batch_size == 32

    def test_ensembles_need_no_version_directory(self, demo_repository_path: Path) -> None:
        repo = ModelRepository.load(demo_repository_path)
        artifact = repo.resolve("ship_pipeline")
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
            "name: claimed\nplatform: mock\nmax_batch_size: 1\n"
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
            "platform: mock\nmax_batch_size: 1\n"
            "inputs: [{name: x, data_type: FP32, dims: [1]}]\n"
            "outputs: [{name: y, data_type: FP32, dims: [1]}]\n"
            "dynamic_batching: {enabled: false}\n"
        )
        with pytest.raises(ConfigurationError, match="no numbered version directory"):
            ModelRepository.load(root)
