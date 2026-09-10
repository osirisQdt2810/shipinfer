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
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from shipinfer.runtime import containment
from tests.support.subprocess_env import checkout_env

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

        `reid` is one build target feeding TWO repository models, so its own name matches
        neither -- and a name-keyed lookup would be right for the other two targets by luck.
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
        target = build_engines.Target("ship_detector", tmp_path / "x.onnx", engine, (version,))

        build_engines._install(target, engine)

        assert (version / "yolo26n.plan").read_bytes() == b"a plan"
        assert not (version / "model.plan").exists(), "the name nothing would have loaded"

    def test_a_target_with_no_version_dirs_installs_nothing(
        self, build_engines: ModuleType, tmp_path: Path
    ) -> None:
        """An empty tuple still means "flat engine only" and must not reach a repository."""
        engine = tmp_path / "reid_r50_fp32.engine"
        engine.write_bytes(b"a plan")
        target = build_engines.Target("reid", tmp_path / "x.onnx", engine, ())

        build_engines._install(target, engine)

        assert list(tmp_path.iterdir()) == [engine]

    def test_one_engine_fans_out_to_every_model_that_loads_it(
        self, build_engines: ModuleType, tmp_path: Path
    ) -> None:
        """`reid` feeds BOTH embedders, and while it fed neither the bench's byte-identity
        guard refused every run and named `build_engines.py --force` as the remedy -- which
        printed success, installed nothing, and left the operator looping with no exit but the
        manual `cp` in the model's own README."""
        person = _repository(tmp_path, "person_embedder", "model.plan")
        ship = _repository(tmp_path, "ship_embedder", "model.plan")
        engine = tmp_path / "reid_r50_fp32.engine"
        engine.write_bytes(b"one reid plan")
        target = build_engines.Target("reid", tmp_path / "x.onnx", engine, (person, ship))

        build_engines._install(target, engine)

        assert (person / "model.plan").read_bytes() == b"one reid plan"
        assert (ship / "model.plan").read_bytes() == b"one reid plan", (
            "both models that load this engine get it, or the guard refuses a run the "
            "printed remedy cannot fix"
        )

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
            build_engines.Target("d", tmp_path / "x.onnx", engine, (version,)), engine
        )

        assert (version / "yolo26n.plan").stat().st_mtime_ns == before


class TestAgainstTheRealRepository:
    def test_every_installable_target_resolves_to_its_config_s_name(
        self, build_engines: ModuleType
    ) -> None:
        """The committed repository, so a rename there fails here rather than at start-up."""
        from shipinfer.repository import ModelRepository

        models = ModelRepository.load(REPOSITORY)
        installable = [t for t in build_engines.TARGETS if t.version_dirs]

        assert installable, "the regex found no target, so this test would pass on anything"
        for target in installable:
            # EVERY directory, not the first. `reid` has two, and reading only `[0]` left
            # `ship_embedder`'s `engine_file` unresolved by the one test whose job is catching
            # a plan installed under a name nothing loads -- weakened exactly where the new
            # entry was added.
            for version_dir in target.version_dirs:
                name = version_dir.parent.name
                assert build_engines._artefact_name(version_dir, target.engine) == (
                    models.entry(name).config.engine_file
                ), name

    def test_every_target_installs_somewhere_and_reid_installs_twice(
        self, build_engines: ModuleType
    ) -> None:
        """This test asserted the OPPOSITE and its premise was the defect.

        `reid` used to build a flat engine only, with the two-step copy left to a README --
        so the bench's byte-identity guard refused every run over a plan that nothing
        installed, and named `build_engines.py --force` as the remedy. It now feeds both
        embedders, which is what makes that remedy true.
        """
        by_name = {t.name: t for t in build_engines.TARGETS}

        assert all(t.version_dirs for t in build_engines.TARGETS), "none is flat-only now"
        assert [d.parent.name for d in by_name["reid"].version_dirs] == [
            "person_embedder",
            "ship_embedder",
        ]


class TestPrecisionNamesThePlan:
    """One precision, one file name. Overwriting is how two runs compare different engines."""

    def test_each_precision_gets_its_own_name(self, build_engines: ModuleType) -> None:
        target = build_engines.TARGETS[0]

        names = {
            p: build_engines._engine_path(target, p).name for p in build_engines.PRECISIONS
        }

        assert names == {
            "fp32": "yolo26n_fp32.engine",
            "fp16": "yolo26n_fp16.engine",
            "int8": "yolo26n_int8.engine",
        }

    def test_fp16_and_int8_together_is_refused_rather_than_ordered(self) -> None:
        """Not a precedence rule. Both reads as "both", and both is what int8 already does
        per layer, so the two spellings cannot be told apart from a mistake."""
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "--fp16", "--int8", "--check"],
            capture_output=True,
            text=True,
            env=checkout_env(),
        )

        assert done.returncode == 2, done.stdout
        assert "ambiguous" in done.stderr


class TestTheCalibrationSetIsBOTHSubjects:
    """A scale chosen with no ship in it is the preprocessing trap by a different door."""

    def test_the_folders_are_interleaved_and_not_concatenated(
        self, build_engines: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        people, ships = tmp_path / "person", tmp_path / "ship"
        for folder, count in ((people, 4), (ships, 2)):
            folder.mkdir()
            for index in range(count):
                (folder / f"{index}.jpg").write_bytes(b"")
        monkeypatch.setattr(build_engines, "CALIBRATION_DIRS", (people, ships))

        files = build_engines._calibration_files()

        # A truncated set still holds both: person, ship, person, ship, then the people left.
        assert [f.parent.name for f in files] == [
            "person",
            "ship",
            "person",
            "ship",
            "person",
            "person",
        ]

    def test_an_embedders_crop_keeps_the_engines_aspect_ratio(
        self, build_engines: ModuleType
    ) -> None:
        """A stand-in for a detection, and not a letterboxed frame: calibrating an embedder on
        1080p with grey bars would put bars in every calibration image and none in a served
        one."""
        numpy = pytest.importorskip("numpy")
        frame = numpy.zeros((1080, 1920, 3), dtype=numpy.uint8)

        cropped = build_engines._centre_crop(frame, (256, 128))

        height, width = cropped.shape[:2]
        assert (height, width) == (1080, 540)
        assert abs(width / height - 128 / 256) < 0.01


class TestABuildIsGatedAndInspectionIsNot:
    """CLAUDE.md's list of what must run in a container names "any engine build".

    This was the one entry in that list with no in-process gate, so the advisory `PreToolUse`
    hook was the only guard -- and a deny-list over command text cannot be sound, which the
    same document says. Patching `containment.require_container` itself is the assertion: a
    private copy of the check inside this script would not be intercepted and the log below
    would come back without its `gate` entry.
    """

    def _log(self, monkeypatch: pytest.MonkeyPatch, build_engines: ModuleType) -> list[str]:
        """Record the calls that decide this, in the order they happen."""
        seen: list[str] = []

        def gate(what: str) -> None:
            seen.append(f"gate:{what}")

        def report(precision: str) -> int:
            seen.append("report")
            return 0

        def build(targets: tuple[object, ...], *, precision: str, force: bool) -> int:
            seen.append("build")
            return 0

        monkeypatch.setattr(containment, "require_container", gate)
        monkeypatch.setattr(build_engines, "report", report)
        monkeypatch.setattr(build_engines, "build", build)
        return seen

    def test_a_build_asks_the_gate_before_it_builds(
        self, monkeypatch: pytest.MonkeyPatch, build_engines: ModuleType
    ) -> None:
        seen = self._log(monkeypatch, build_engines)

        assert build_engines.main(["--only", "ship_detector"]) == 0
        assert seen == ["gate:an engine build", "build"]

    def test_check_reports_and_is_not_gated(
        self, monkeypatch: pytest.MonkeyPatch, build_engines: ModuleType
    ) -> None:
        """`--check` builds nothing, so it is inspection -- allowed on the host on purpose,
        the way `shipinfer repo ls` is. Gating it would make the one command that tells you
        what is missing unusable exactly where you ask that question."""
        seen = self._log(monkeypatch, build_engines)

        assert build_engines.main(["--check"]) == 0
        assert seen == ["report"]

    def test_an_unknown_model_keeps_its_usage_code(
        self, monkeypatch: pytest.MonkeyPatch, build_engines: ModuleType
    ) -> None:
        """A malformed command line deserves its usage error rather than a container one.
        #182 gated `run_bench.main` on line one and turned three exit-2 assertions into
        container failures; the gate belongs below argv validation, above the work."""
        seen = self._log(monkeypatch, build_engines)

        assert build_engines.main(["--only", "no_such_model"]) == 2
        assert seen == []

    @pytest.mark.skipif(
        containment.detect().in_container,
        reason="the gate is silent in a container, which is what it is for",
    )
    def test_the_refusal_reaches_a_real_invocation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end through `__main__`, because the tests above patch out both calls and
        would still pass if nothing wired `main` to the command line.

        The override is dropped rather than tolerated: it is per-command by design, so a run
        that happens to have it set must not turn this into a silent pass. `checkout_env` then
        makes the spawned interpreter resolve `shipinfer` from THIS tree.
        """
        monkeypatch.delenv(containment.ALLOW_HOST_RUN_ENV, raising=False)
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "--only", "ship_detector"],
            capture_output=True,
            text=True,
            env=checkout_env(),
        )

        assert done.returncode != 0, done.stdout
        assert "an engine build must run inside a container" in done.stderr
