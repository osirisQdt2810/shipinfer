"""The comparison has to be fair in the ways that are easy to get wrong silently.

Two asymmetries were found by review, both favouring us, and neither visible in the output:
the baseline was given a quarter of our detector concurrency, and nothing checked that the
two systems loaded the same engine. A speed-up produced by either would look exactly like a
speed-up produced by the architecture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.harness.config import BenchConfig


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

    def _repository(self, tmp_path: Path, detector: bytes) -> Path:
        root = tmp_path / "model_repository"
        (root / "ship_detector" / "1").mkdir(parents=True)
        (root / "ship_detector" / "1" / "model.plan").write_bytes(detector)
        return root

    def _config(self, tmp_path: Path, flat: bytes, plan: bytes) -> BenchConfig:
        engine = tmp_path / "yolo26n_fp32.engine"
        engine.write_bytes(flat)
        return BenchConfig(
            det_engine=engine,
            seg_engine=engine,
            model_repository=self._repository(tmp_path, plan),
        )

    def test_identical_engines_pass(self, tmp_path: Path) -> None:
        self._config(tmp_path, b"PLAN-A", b"PLAN-A").require_same_engines()

    def test_different_engines_are_refused(self, tmp_path: Path) -> None:
        """An fp16 plan against an fp32 one is roughly a 2x 'architecture' win, and the
        precision is not recoverable from a serialised plan — so the check is on bytes."""
        config = self._config(tmp_path, b"PLAN-FP32", b"PLAN-FP16-DIFFERENT")

        with pytest.raises(RuntimeError, match="measures the engines"):
            config.require_same_engines()

    def test_a_missing_plan_is_not_treated_as_a_mismatch(self, tmp_path: Path) -> None:
        """`require_inputs` already reports what is absent; reporting it twice, as a
        mismatch, would point at the wrong problem."""
        engine = tmp_path / "yolo26n_fp32.engine"
        engine.write_bytes(b"PLAN")
        empty = tmp_path / "model_repository"
        (empty / "ship_detector" / "1").mkdir(parents=True)
        BenchConfig(
            det_engine=engine, seg_engine=engine, model_repository=empty
        ).require_same_engines()
