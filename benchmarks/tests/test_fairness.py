"""The comparison has to be fair in the ways that are easy to get wrong silently.

Two asymmetries were found by review, both favouring us, and neither visible in the output:
the baseline was given a quarter of our detector concurrency, and nothing checked that the
two systems loaded the same engine. A speed-up produced by either would look exactly like a
speed-up produced by the architecture.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from benchmarks.harness.config import (
    MODULE_MODELS,
    PRECISIONS,
    BenchConfig,
    read_instances_per_gpu,
)

BUILD_ENGINES = Path(__file__).resolve().parents[2] / "scripts" / "build_engines.py"


@pytest.fixture(scope="module")
def build_engines() -> ModuleType:
    """The real script, by path -- ``scripts/`` is not a package on ``sys.path``.

    Loaded so a remedy this file prints can be handed to the parser that has to accept it,
    rather than compared against a copy of its flags that drifts.
    """
    spec = importlib.util.spec_from_file_location("scripts.build_engines", BUILD_ENGINES)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts.build_engines"] = module
    spec.loader.exec_module(module)
    return module


class TestTheConcurrencyComesFromTheRepositoryAndNotFromAHandKeptTable:
    """The translation was tested; the SOURCE of the number was not, and it drifted.

    `config.py`'s own header calls `instances_per_gpu` "the one number in this file that can
    silently make the comparison unfair". It held `{"det": 2, "seg": 1}` while
    `model_repository/ship_segmenter/config.yaml` went to `count: 2` on 27 Aug -- so on a
    7-GPU box the baseline was given seven segmenter threads where we ran fourteen. The same
    asymmetry the class below exists to prevent, one field along.
    """

    def test_each_module_gets_the_count_its_model_declares(self) -> None:
        from shipinfer.repository import ModelRepository
        from shipinfer.repository.resolved import model_runtimes

        repository = Path(__file__).resolve().parents[2] / "model_repository"
        runtimes = model_runtimes(ModelRepository.load(repository))

        found = read_instances_per_gpu(repository)

        assert found, "no module resolved, so this test would pass on anything"
        for module, model in MODULE_MODELS.items():
            assert found[module] == runtimes[model].instances, module

    def test_resolved_fills_it_and_an_explicit_mapping_still_wins(self) -> None:
        assert dict(
            BenchConfig(gpus=(0,)).resolved().instances_per_gpu
        ) == read_instances_per_gpu(Path(__file__).resolve().parents[2] / "model_repository")
        override = BenchConfig(gpus=(0,), instances_per_gpu={"det": 9}).resolved()

        assert dict(override.instances_per_gpu) == {"det": 9}, "an explicit answer is kept"

    def test_the_segmenter_is_two_per_gpu_which_is_the_drift_this_closed(self) -> None:
        """Named rather than left implicit: 1 is the number that was wrong, and a future
        edit that puts it back should fail here rather than in a benchmark nobody re-reads."""
        repository = Path(__file__).resolve().parents[2] / "model_repository"

        assert read_instances_per_gpu(repository)["seg"] == 2

    def test_the_run_record_carries_the_concurrency_even_unresolved(self) -> None:
        """`as_dict` is "the thing written into every log's metadata line", and the field it
        reads now defaults to empty -- so an unresolved config would have recorded no
        concurrency at all, leaving the run record a second source for this number."""
        record = BenchConfig(cameras=50, fps=20.0).as_dict()

        assert record["baseline_workers"], "the metadata line lost the number it exists for"
        assert record["instances_per_gpu"]["seg"] == 2

    def test_an_unreadable_repository_is_refused_and_not_guessed(self, tmp_path: Path) -> None:
        """ "No repository", "a repository that will not parse" and "a repository that says 2"
        are three different events; a last-resort literal made all three answer the same.

        `resolved()` stays tolerant -- it is documented as usable where the artefacts are
        absent -- so the refusal lands where the number is actually needed."""
        from shipinfer.core.errors import ConfigurationError

        empty = tmp_path / "model_repository"
        empty.mkdir()
        config = BenchConfig(gpus=(0,), model_repository=empty)

        assert dict(config.resolved().instances_per_gpu) == {}, "resolved() still works"
        with pytest.raises(ConfigurationError, match="not derivable"):
            config.workers_for("det")

    def test_the_other_harness_reads_the_same_file(self) -> None:
        """`shipinfer.py` held a SECOND copy of the same table, model-keyed, also stale --
        so the plateau guard for the segmenter sat at half its real bound."""
        from benchmarks.harness.shipinfer import repository_instances

        repository = Path(__file__).resolve().parents[2] / "model_repository"
        counts = repository_instances(repository)

        assert counts["ship_segmenter"] == 2
        for module, model in MODULE_MODELS.items():
            assert counts[model] == read_instances_per_gpu(repository)[module], module


class TestBothSidesGetTheSameConcurrency:
    """The baseline takes a total; we configure per GPU. The translation is the risk."""

    def test_the_baseline_is_given_our_instances_per_gpu_times_the_gpus(self) -> None:
        config = BenchConfig(gpus=(2, 3, 4, 5), instances_per_gpu={"det": 2, "seg": 1})

        assert config.workers_for("det") == 8, "2 per GPU across 4 GPUs, as ours runs"
        assert config.workers_for("seg") == 4

    def test_it_scales_with_the_gpu_count_rather_than_being_a_constant(self) -> None:
        """The field read `4` total whatever the machine — one thread per GPU here, and
        one *quarter* of a thread per GPU on the 16-GPU node the sizing targets."""
        assert BenchConfig(gpus=(0,), instances_per_gpu={"det": 2}).workers_for("det") == 2
        assert (
            BenchConfig(gpus=tuple(range(16)), instances_per_gpu={"det": 2}).workers_for("det")
            == 32
        )

    def test_a_module_we_did_not_declare_still_gets_a_worker(self) -> None:
        assert BenchConfig(gpus=(0, 1)).workers_for("unknown") >= 1

    def test_the_report_states_both_figures(self) -> None:
        """So a reader can check the fairness instead of trusting it."""
        note = BenchConfig(gpus=(2, 3, 4, 5)).concurrency_note

        assert "det=2/gpu" in note
        assert "det=8" in note, "the baseline total, spelled out"
        assert "unpinned" in note


class TestOmpIsSymmetric:
    def test_unpinned_by_default_on_both_sides(self) -> None:
        """Pinning only the baseline gave our torch pre-processing the whole box while it
        letterboxed on the CPU inside its own single-threaded workers."""
        assert BenchConfig().omp_threads is None

    def test_a_pin_is_recorded_so_it_cannot_be_one_sided_by_accident(self) -> None:
        config = BenchConfig(omp_threads=1)

        assert "OMP_NUM_THREADS=1" in config.concurrency_note
        assert config.as_dict()["omp_threads"] == 1


class TestBothSidesLoadTheSameEngine:
    """Existence was checked; identity was not, and identity is the property that matters."""

    #: Every model the guard pairs, so the fixture cannot pass by having no plan to check.
    _PAIRED = ("ship_detector", "ship_segmenter", "person_embedder", "ship_embedder")

    def _repository(
        self, tmp_path: Path, detector: bytes | None, engine_file: str = "model.plan"
    ) -> Path:
        """A repository holding a plan for every paired model, or for none of them.

        WITH A `config.yaml` EACH, because the guard resolves the plan's file name the way the
        installer does -- through `parameters.engine_file`, which is configurable. A fixture
        with no config could only ever exercise the assumed name.
        """
        root = tmp_path / "model_repository"
        for model in self._PAIRED:
            version = root / model / "1"
            version.mkdir(parents=True)
            config: dict[str, object] = {
                "name": model,
                "platform": "tensorrt",
                "max_batch_size": 4,
                "inputs": [{"name": "images", "data_type": "FP32", "dims": [3, 640, 640]}],
                "outputs": [{"name": "output0", "data_type": "FP32", "dims": [300, 6]}],
            }
            if engine_file != "model.plan":
                config["parameters"] = {"engine_file": engine_file}
            (root / model / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            if detector is not None:
                (version / engine_file).write_bytes(detector)
        return root

    def _config(self, tmp_path: Path, flat: bytes, plan: bytes) -> BenchConfig:
        # EVERY engine named explicitly, including the embedders'. Left unset, `resolved()`
        # fills them from the repository root -- so the test would check whatever engines this
        # box happens to hold, pass in a worktree with an empty `models/`, and fail in a
        # checkout that has them. A fixture that depends on the box is not a fixture.
        engine = tmp_path / "yolo26n_fp32.engine"
        engine.write_bytes(flat)
        return BenchConfig(
            det_engine=engine,
            seg_engine=engine,
            emb_engine=engine,
            model_repository=self._repository(tmp_path, plan),
        )

    def test_shipinfer_does_not_need_the_baselines_flat_engines(self, tmp_path: Path) -> None:
        """The defect this scoping fixes: `require_inputs` demanded the segmenter's flat plan
        whichever models a run would load, so a detector-only measurement could not start and
        `--precision int8` could only ever raise -- the segmenter does not build at int8 here.
        """
        config = self._config(tmp_path, b"PLAN-A", b"PLAN-A")
        absent = replace(
            config,
            det_engine=tmp_path / "nope_det.engine",
            seg_engine=tmp_path / "nope_seg.engine",
            person_frames=tmp_path,
            ship_frames=tmp_path,
        )

        absent.require_inputs("shipinfer")  # the repository is what it needs, and it is there

        with pytest.raises(FileNotFoundError, match="detector engine"):
            absent.require_inputs("baseline")

    def test_an_unknown_system_name_is_refused(self, tmp_path: Path) -> None:
        """Rather than silently checking nothing, which is how a typo becomes a green run."""
        with pytest.raises(ValueError, match="system must be"):
            self._config(tmp_path, b"PLAN-A", b"PLAN-A").require_inputs("shipnfier")

    def test_identical_engines_pass(self, tmp_path: Path) -> None:
        self._config(tmp_path, b"PLAN-A", b"PLAN-A").require_same_engines()

    def test_different_engines_are_refused(self, tmp_path: Path) -> None:
        """An fp16 plan against an fp32 one is roughly a 2x 'architecture' win, and the
        precision is not recoverable from a serialised plan — so the check is on bytes."""
        config = self._config(tmp_path, b"PLAN-FP32", b"PLAN-FP16-DIFFERENT")

        with pytest.raises(RuntimeError, match="measures the engines"):
            config.require_same_engines()

    def test_a_named_precision_installs_the_plan_it_names(self, tmp_path: Path) -> None:
        """`BENCH-PRECISION-SELECTS-NO-PLAN`'s other half: the flag now SELECTS.

        It named the baseline's flat engine and nothing else -- our side loaded whatever
        precision `<model>/1/model.plan` held, so the flag decided which file the digest was
        compared against and never which file ran. A named precision installs it.
        """
        config = replace(
            self._config(tmp_path, b"PLAN-FP16", b"PLAN-FP32-INSTALLED"), precision="fp16"
        )
        repository = config.model_repository
        assert repository is not None
        plan = repository / "ship_detector" / "1" / "model.plan"
        assert plan.read_bytes() == b"PLAN-FP32-INSTALLED", "the wrong precision, to start"

        config.require_same_engines("shipinfer")

        assert plan.read_bytes() == b"PLAN-FP16", "the named precision is what is installed"

        # IDEMPOTENT, asserted on the mtime rather than the bytes: a second call that rewrote
        # identical bytes would pass a bytes check identically, so that check was not the
        # evidence it read as (#262's review).
        before = plan.stat().st_mtime_ns
        config.require_same_engines("shipinfer")
        assert plan.stat().st_mtime_ns == before, "the second call copies nothing"

    def test_the_default_run_installs_before_the_baseline_arm_checks(
        self, tmp_path: Path
    ) -> None:
        """THE SEQUENCE, which is what nothing drove before (#262 r2).

        `run_bench.py`'s pre-flight calls one arm then the other, and `--systems` defaults to
        both. With `baseline` first, its digest check refused the very mismatch the install
        exists to resolve -- so the flag selected only on `--systems shipinfer`, and the
        `--help` promising "which precision BOTH sides load" was false in the mode it named.
        The unit tests missed it because they called one arm in isolation, and one of them
        passed the `"both"` default no production caller ever produces.
        """
        config = replace(
            self._config(tmp_path, b"PLAN-FP16", b"PLAN-FP32-INSTALLED"), precision="fp16"
        )
        repository = config.model_repository
        assert repository is not None
        plan = repository / "ship_detector" / "1" / "model.plan"

        # The pre-flight's order, as `run_bench.py` runs it.
        for system in ("shipinfer", "baseline"):
            config.require_same_engines(system)

        assert plan.read_bytes() == b"PLAN-FP16", "installed by our arm"

    def test_the_old_order_is_what_refused(self, tmp_path: Path) -> None:
        """The control: baseline first sees the mismatch before anything installs, which is
        the abort this reordering removes. Kept as a test so the order is a DECISION rather
        than an accident of how the loop happens to be written."""
        config = replace(
            self._config(tmp_path, b"PLAN-FP16", b"PLAN-FP32-INSTALLED"), precision="fp16"
        )

        with pytest.raises(RuntimeError, match="measures the engines"):
            config.require_same_engines("baseline")

    def test_an_abort_partway_installs_nothing_at_all(self, tmp_path: Path) -> None:
        """ALL FOUR MODELS OR NONE, which is the invariant `--precision` already claims.

        `_ENGINE_PAIRS` puts the two that install cleanly FIRST, so a copy inside the loop
        left the repository half fp16 and half fp32 -- permanently, from a run the operator
        asked to measure and which then aborted (#262 r3). The fixture is this file's own
        `_repository_without_embedder_plans`, "the state of the box immediately after a build".
        """
        engine = tmp_path / "yolo26n_fp32.engine"
        engine.write_bytes(b"PLAN-FP16")
        repository = self._repository_without_embedder_plans(tmp_path, b"PLAN-FP32-INSTALLED")
        config = BenchConfig(
            det_engine=engine,
            seg_engine=engine,
            emb_engine=engine,
            model_repository=repository,
            precision="fp16",
        )
        installable = {
            model: (repository / model / "1" / "model.plan")
            for model in ("ship_detector", "ship_segmenter")
        }

        # The embedders have no plan to hold the precision to, so the loop raises -- AFTER the
        # two above have already cleared it.
        with pytest.raises(RuntimeError, match="no plan at"):
            config.require_same_engines("shipinfer")

        for model, plan in installable.items():
            assert plan.read_bytes() == b"PLAN-FP32-INSTALLED", (
                f"{model} was installed before the run aborted, leaving a repository that is "
                f"half one precision and half the other"
            )

    def test_a_baseline_only_run_installs_nothing(self, tmp_path: Path) -> None:
        """`--systems baseline` loads no plan of ours, so it may not rewrite one.

        The scoping above skips only the embedder PAIR, so this fell through and rewrote the
        detector and segmenter while leaving both embedders -- a MIXED repository, from a run
        where nothing of ours runs at all (#262's review, reproduced). The next unnamed run is
        documented to "measure whatever is installed and say so", and would have said so about
        a half-converted one.
        """
        config = replace(
            self._config(tmp_path, b"PLAN-FP16", b"PLAN-FP32-INSTALLED"), precision="fp16"
        )
        repository = config.model_repository
        assert repository is not None
        plans = {
            model: (repository / model / "1" / "model.plan")
            for model in ("ship_detector", "ship_segmenter")
        }

        with pytest.raises(RuntimeError, match="measures the engines"):
            config.require_same_engines("baseline")

        for model, plan in plans.items():
            assert plan.read_bytes() == b"PLAN-FP32-INSTALLED", f"{model} was not rewritten"

    def test_without_a_named_precision_a_mismatch_is_still_refused(
        self, tmp_path: Path
    ) -> None:
        """Installing is what a CLAIM buys. With no precision named there is no claim, so a
        difference between the two engines is the old refusal and not a licence to overwrite
        the operator's repository."""
        config = self._config(tmp_path, b"PLAN-FP32", b"PLAN-FP16-DIFFERENT")
        repository = config.model_repository
        assert repository is not None
        plan = repository / "ship_detector" / "1" / "model.plan"

        with pytest.raises(RuntimeError, match="measures the engines"):
            config.require_same_engines()

        assert plan.read_bytes() == b"PLAN-FP16-DIFFERENT", "and nothing was written"

    def _repository_without_embedder_plans(self, tmp_path: Path, plan: bytes) -> Path:
        """The state of the box immediately after a build, BEFORE the reid fanout existed:
        every flat engine present, det and seg plans installed, the embedders' absent."""
        root = self._repository(tmp_path, plan)
        for model in ("person_embedder", "ship_embedder"):
            (root / model / "1" / "model.plan").unlink()
        return root

    def test_a_baseline_only_run_is_not_refused_over_an_embedder_plan(
        self, tmp_path: Path
    ) -> None:
        """The defect this fix is for, and it is the shape of the one the PR is named after --
        an engine check demanding an artefact the run does not load -- one level down, in the
        guard rather than in `require_inputs`.

        The baseline loads one model per image and never an embedder, so it has no stake in
        that pair. Refusing it there sent the operator to `build_engines.py --force`, which
        (before the fanout) installed the reid plan nowhere: the guard failed identically and
        the loop had no exit but a manual `cp`.
        """
        engine = tmp_path / "yolo26n_fp32.engine"
        engine.write_bytes(b"PLAN-A")
        config = replace(
            BenchConfig(
                det_engine=engine,
                seg_engine=engine,
                emb_engine=engine,
                model_repository=self._repository_without_embedder_plans(tmp_path, b"PLAN-A"),
            ),
            person_frames=tmp_path,
            ship_frames=tmp_path,
        )

        config.require_inputs("baseline")  # the pair it has no stake in is skipped

        with pytest.raises(RuntimeError, match="person_embedder"):
            config.require_inputs("shipinfer")

    def test_the_absent_plan_message_does_not_claim_the_baseline_loads_an_embedder(
        self, tmp_path: Path
    ) -> None:
        """A false diagnosis is worse than a vague one when the next line is a command."""
        engine = tmp_path / "yolo26n_fp32.engine"
        engine.write_bytes(b"PLAN-A")
        config = BenchConfig(
            det_engine=engine,
            seg_engine=engine,
            emb_engine=engine,
            model_repository=self._repository_without_embedder_plans(tmp_path, b"PLAN-A"),
        )

        with pytest.raises(RuntimeError) as raised:
            config.require_same_engines("shipinfer")

        assert "the baseline loads" not in str(raised.value)
        assert "this run's precision names" in str(raised.value)

    def test_the_guard_looks_for_the_name_the_config_asks_for(self, tmp_path: Path) -> None:
        """`parameters.engine_file` is configurable and the installer honours it.

        A guard that assumed `model.plan` would look for a file the installer never writes,
        raise "run `build_engines.py --force`", have the operator run it successfully, and fail
        identically -- verbatim the unfixable loop both embedder READMEs describe as the thing
        this guard's own remedy removes.
        """
        engine = tmp_path / "yolo26n_fp32.engine"
        engine.write_bytes(b"PLAN-A")
        named = BenchConfig(
            det_engine=engine,
            seg_engine=engine,
            emb_engine=engine,
            model_repository=self._repository(tmp_path, b"PLAN-A", "reid_r50.plan"),
        )

        named.require_same_engines()  # the plan is under the configured name, and matches

        assert not (
            tmp_path / "model_repository" / "person_embedder" / "1" / "model.plan"
        ).exists(), "the fixture wrote the configured name, so this test could fail"

    def test_the_embedders_are_inside_the_guard_too(self, tmp_path: Path) -> None:
        """Not a cross-system check -- the baseline runs no embedder -- but the one that says
        our side loaded the precision that was ASKED for.

        With only the detector and segmenter paired, `--precision fp16` moved two of four
        models and said nothing about the two that carry ~9 of the chain's ~11.7 invocations
        per image.
        """
        config = self._config(tmp_path, b"PLAN-A", b"PLAN-A")
        assert config.model_repository is not None
        (config.model_repository / "person_embedder" / "1" / "model.plan").write_bytes(
            b"PLAN-SOMETHING-ELSE"
        )

        with pytest.raises(RuntimeError, match="person_embedder"):
            config.require_same_engines()

    def test_an_unpaired_plan_warns_rather_than_passing_in_silence(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A single-system run may have no flat engine, and then nothing verifies which
        precision the plan holds -- so the run says so instead of looking checked."""
        config = replace(
            self._config(tmp_path, b"PLAN-A", b"PLAN-A"),
            emb_engine=tmp_path / "absent_reid.engine",
        )

        config.require_same_engines()

        assert "person_embedder" in capsys.readouterr().err

    def test_a_named_precision_with_nothing_to_check_it_against_is_refused(
        self, tmp_path: Path
    ) -> None:
        """`BENCH-PRECISION-SELECTS-NO-PLAN`: our side loads `model.plan` whatever precision
        it holds, so with no flat engine to hold it to, `--precision fp16` selects nothing and
        reports itself in `summary.json` anyway. A knob that lies is worse than an absent one,
        so the claim is refused -- and the refusal names both ways out."""
        config = replace(
            self._config(tmp_path, b"PLAN-A", b"PLAN-A"),
            emb_engine=tmp_path / "absent_reid.engine",
            precision="fp16",
        )

        with pytest.raises(RuntimeError) as raised:
            config.require_same_engines()

        message = str(raised.value)
        assert "--precision fp16" in message, "the claim that cannot be kept"
        assert "drop --precision" in message, "and the other"

    def test_a_named_precision_is_refused_when_there_is_no_plan_either(
        self, tmp_path: Path
    ) -> None:
        """The branch that used to fall through in silence, gated on the plan existing. With
        neither a flat engine nor a plan the server autobuilds that model from ONNX *after*
        this guard has passed, and `summary.json` records a precision nobody could check --
        the same lie, by the route with one fewer file in it."""
        config = replace(
            self._config(tmp_path, b"PLAN-A", b"PLAN-A"),
            emb_engine=tmp_path / "absent_reid.engine",
            precision="fp16",
        )
        assert config.model_repository is not None
        (config.model_repository / "person_embedder" / "1" / "model.plan").unlink()

        with pytest.raises(RuntimeError, match="autobuild"):
            config.require_same_engines()

    @pytest.mark.parametrize("precision", PRECISIONS)
    def test_the_refusals_remedy_is_a_command_build_engines_accepts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        build_engines: ModuleType,
        precision: str,
    ) -> None:
        """And builds the precision it promises. The remedy used to read
        `build_engines.py --precision fp16`, a flag that script has never had: the operator
        runs it, argparse exits 2, nothing is built, and the refusal repeats. A substring
        assertion is what let that ship, so the command is handed to the real parser."""
        config = replace(
            self._config(tmp_path, b"PLAN-A", b"PLAN-A"),
            emb_engine=tmp_path / "absent_reid.engine",
            precision=precision,
        )
        with pytest.raises(RuntimeError) as raised:
            config.require_same_engines()

        quoted = re.search(r"`(scripts/build_engines\.py[^`]*)`", str(raised.value))
        assert quoted, "the refusal has to name a command at all"
        built: list[str] = []
        monkeypatch.setattr(build_engines, "report", lambda asked: built.append(asked) or 0)

        assert build_engines.main([*quoted.group(1).split()[1:], "--check"]) == 0
        assert built == [precision], "the remedy builds the precision the message claims"

    def test_the_default_claims_nothing_so_the_same_tree_only_warns(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The DEFAULT is the thing under test, and it is what every chain run here takes:
        `precision` is `None` rather than `"fp32"`, so nobody claimed anything and an unpaired
        plan warns instead of refusing. Asserted on the field, not just on the warning, because
        a default that drifted back to `"fp32"` would turn every such run into a refusal."""
        config = replace(
            self._config(tmp_path, b"PLAN-A", b"PLAN-A"),
            emb_engine=tmp_path / "absent_reid.engine",
        )
        assert config.precision is None, "the default claims no precision"

        config.require_same_engines()

        assert "whatever is installed" in capsys.readouterr().err

    def test_a_missing_plan_is_refused_rather_than_skipped(self, tmp_path: Path) -> None:
        """Skipping an absent plan made the guard useless in the case it exists for.

        `autobuild` builds the server its own engine from ONNX when the plan is missing —
        *after* this check has passed — so the two sides then run different engines and the
        one property this method claims to enforce is the one that silently does not hold.
        """
        engine = tmp_path / "yolo26n_fp32.engine"
        engine.write_bytes(b"PLAN")
        config = BenchConfig(
            det_engine=engine,
            seg_engine=engine,
            model_repository=self._repository(tmp_path, None),
        )

        with pytest.raises(RuntimeError, match="build its own from ONNX"):
            config.require_same_engines()


class TestTheHarnessCallsTheApiThatExists:
    """The offered-rate check reads ingest's counters, and read them off the wrong method.

    `IngestManager` has `summary()`, not `stats()`. The mistake only surfaced at the *end*
    of a 30-second GPU run, after the measurement was already spent — which is exactly the
    kind of failure an offline test should take instead.
    """

    def test_the_manager_exposes_the_counters_the_harness_reads(self) -> None:
        from shipinfer.ingest import IngestManager
        from shipinfer.ingest.camera.health import IngestSummary

        assert hasattr(IngestManager, "summary")
        for field in ("frames_read", "frames_dropped"):
            assert field in IngestSummary.__dataclass_fields__, field

    def test_the_harness_calls_summary(self) -> None:
        import inspect

        from benchmarks.harness import shipinfer as harness

        source = inspect.getsource(harness.run_shipinfer)
        assert "manager.summary()" in source
        # Narrow on purpose: `instance.stats()` in the same function is correct and is a
        # different object's API. A guard that matched the bare word would fail on it.
        assert "manager.stats()" not in source, "IngestManager has no stats()"


class TestTheReimplementedBaselineIsGone:
    """`compare_baseline.py` drove a *re-implementation* of the baseline against a synthetic
    backend, on the premise — stated in its own docstring — that counting-simulation could
    not be run here. `harness/baseline.py` compiles and runs the submodule's own binary, so
    the premise is false and the file was a second, wrong answer to the same question.

    Asserted rather than assumed because deleting it did not stick: a later merge from main
    resurrected it, exactly as it resurrected three model configs, and the PR body claimed
    a deletion that had been undone. A test is the only thing a merge cannot quietly revert.
    """

    def test_the_file_is_absent(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        assert not (repo / "benchmarks" / "compare_baseline.py").exists(), (
            "compare_baseline.py is back. It measures a re-implementation against a "
            "synthetic backend; `benchmarks/run_bench.py` runs the real binary."
        )

    def test_the_real_entry_point_is_present(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        assert (repo / "benchmarks" / "run_bench.py").is_file()
        assert (repo / "benchmarks" / "harness" / "baseline.py").is_file()
