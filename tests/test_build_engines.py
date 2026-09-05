"""``scripts/build_engines.py`` installs a plan under the name the repository asks for.

Why this file exists: the builder wrote `model.plan` unconditionally while
`parameters.engine_file` is configurable, so a model naming anything else got its plan under a
name nothing loads. Both halves report success -- the build prints a path, `shipinfer plan`
prints the configured name -- and the two disagree only at the next start-up, minutes of
TensorRT later. `test_analyse_cpp.py` and `test_build_csrc.py` make the same argument about
two other pairs of files that must agree and are checked by nothing at run time.

Offline by design (ADR-001): the artefact NAME is repository config, so deciding it needs no
device. Only writing the bytes does, and that is not what is under test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_engines.py"
REPOSITORY = ROOT / "model_repository"


@pytest.fixture(scope="module")
def build_engines() -> ModuleType:
    """Import the script by path -- ``scripts/`` is not a package on ``sys.path``.

    Registered in ``sys.modules`` under its dotted name before execution, because the module
    defines a ``@dataclass`` and `dataclasses` looks its owner up there while building it.
    """
    spec = importlib.util.spec_from_file_location("scripts.build_engines", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts.build_engines"] = module
    spec.loader.exec_module(module)
    return module


def _repository(root: Path, name: str, engine_file: str | None) -> Path:
    """A minimal one-model repository, with or without an ``engine_file`` parameter."""
    version = root / name / "1"
    version.mkdir(parents=True)
    config: dict[str, object] = {
        "name": name,
        "platform": "tensorrt",
        "max_batch_size": 4,
        "inputs": [{"name": "images", "data_type": "FP32", "dims": [3, 640, 640]}],
        "outputs": [{"name": "output0", "data_type": "FP32", "dims": [300, 6]}],
    }
    if engine_file is not None:
        config["parameters"] = {"engine_file": engine_file}
    (root / name / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return version


class TestTheNameComesFromTheConfig:
    def test_a_configured_engine_file_is_what_gets_written(
        self, build_engines: ModuleType, tmp_path: Path
    ) -> None:
        """The defect: this used to answer `model.plan` whatever the config said."""
        version = _repository(tmp_path, "ship_detector", "yolo26n.plan")

        assert build_engines._artefact_name(version, tmp_path / "flat.engine") == "yolo26n.plan"

    def test_the_conventional_name_is_still_the_default(
        self, build_engines: ModuleType, tmp_path: Path
    ) -> None:
        version = _repository(tmp_path, "ship_detector", None)

        assert build_engines._artefact_name(version, tmp_path / "flat.engine") == "model.plan"

    def test_the_model_is_identified_BY_PATH_and_not_by_the_target_s_name(
        self, build_engines: ModuleType, tmp_path: Path
    ) -> None:
        """`Target.name` and a repository model name are not the same thing.

        `reid` is one build target feeding TWO repository models, and it has no `version_dir`
        at all -- so a name-keyed lookup would be right for the other two targets by luck.
        """
        version = _repository(tmp_path, "person_embedder", "reid_r50.plan")

        assert build_engines._artefact_name(version, tmp_path / "x.engine") == "reid_r50.plan"

    def test_an_unreadable_repository_is_REFUSED_and_not_defaulted(
        self, build_engines: ModuleType, tmp_path: Path
    ) -> None:
        """Guessing `model.plan` here is the defect itself, so the refusal is the fix.

        It names the flat engine, which is already built, so the operator loses only the
        install step and knows exactly which file to correct.
        """
        version = tmp_path / "ship_detector" / "1"
        version.mkdir(parents=True)
        (tmp_path / "ship_detector" / "config.yaml").write_text("{{not yaml", encoding="utf-8")

        with pytest.raises(SystemExit) as refused:
            build_engines._artefact_name(version, tmp_path / "flat.engine")

        message = str(refused.value)
        assert "ship_detector" in message
        assert "flat.engine" in message, "the engine that WAS built is named"
        assert "config.yaml" in message, "and the file to fix"


class TestTheInstallWritesThatName:
    """`_install` itself, which is the line that was wrong and is a file copy, not a build."""

    def test_the_plan_lands_under_the_configured_name(
        self, build_engines: ModuleType, tmp_path: Path
    ) -> None:
        version = _repository(tmp_path, "ship_detector", "yolo26n.plan")
        engine = tmp_path / "yolo26n_fp32.engine"
        engine.write_bytes(b"a plan")
        target = build_engines.Target("ship_detector", tmp_path / "x.onnx", engine, version)

        build_engines._install(target, engine)

        assert (version / "yolo26n.plan").read_bytes() == b"a plan"
        assert not (version / "model.plan").exists(), "the name nothing would have loaded"

    def test_a_target_with_no_version_dir_installs_nothing(
        self, build_engines: ModuleType, tmp_path: Path
    ) -> None:
        """`reid` builds a flat engine only, and must not reach the repository at all."""
        engine = tmp_path / "reid_r50_fp32.engine"
        engine.write_bytes(b"a plan")
        target = build_engines.Target("reid", tmp_path / "x.onnx", engine, None)

        build_engines._install(target, engine)

        assert list(tmp_path.iterdir()) == [engine]

    def test_an_identical_plan_already_there_is_not_rewritten(
        self, build_engines: ModuleType, tmp_path: Path
    ) -> None:
        """The skip path -- and it must skip on the CONFIGURED name, not the conventional one,
        or a second run rewrites a file it just wrote."""
        version = _repository(tmp_path, "ship_detector", "yolo26n.plan")
        engine = tmp_path / "yolo26n_fp32.engine"
        engine.write_bytes(b"a plan")
        (version / "yolo26n.plan").write_bytes(b"a plan")
        before = (version / "yolo26n.plan").stat().st_mtime_ns

        build_engines._install(
            build_engines.Target("d", tmp_path / "x.onnx", engine, version), engine
        )

        assert (version / "yolo26n.plan").stat().st_mtime_ns == before


class TestAgainstTheRealRepository:
    def test_every_installable_target_resolves_to_its_config_s_name(
        self, build_engines: ModuleType
    ) -> None:
        """The committed repository, so a rename there fails here rather than at start-up."""
        from shipinfer.repository import ModelRepository

        models = ModelRepository.load(REPOSITORY)
        installable = [t for t in build_engines.TARGETS if t.version_dir is not None]

        assert installable, "the regex found no target, so this test would pass on anything"
        for target in installable:
            name = target.version_dir.parent.name
            assert build_engines._artefact_name(target.version_dir, target.engine) == (
                models.entry(name).config.engine_file
            )

    def test_a_target_with_no_version_dir_is_never_asked(
        self, build_engines: ModuleType
    ) -> None:
        """`reid` builds a flat engine only; the READMEs carry its two-step copy."""
        flat_only = [t for t in build_engines.TARGETS if t.version_dir is None]

        assert [t.name for t in flat_only] == ["reid"]
