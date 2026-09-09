"""``scripts/analyse_cpp.py`` against the run record ``cli/bench.cpp`` actually writes.

Why this file exists: the analyser reads `meta.config` keys by name and the binary writes them
from a different language, so a rename crosses the boundary as a `KeyError` in the judge --
after the run, on the artefact that took a GPU and seventy seconds to produce. P5-C renamed
`buffer_capacity` to `pipeline_queue` and nothing noticed, because this script had no test at
all. `test_build_csrc.py` makes the same argument one table along.

Offline by design (ADR-001): it reads two files and runs no measurement.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyse_cpp.py"
BENCH = ROOT / "csrc" / "shipinfer" / "cli" / "bench.cpp"
DRIVER = ROOT / "scripts" / "run_cpp_bench.sh"


def _load(path: Path, name: str) -> ModuleType:
    """Import a script by path -- ``scripts/`` is not a package on ``sys.path``."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analyse_cpp() -> ModuleType:
    return _load(SCRIPT, "analyse_cpp")


class _Samples:
    """`analysis.SampleLog`, reduced to the one attribute `capacities` reads."""

    def __init__(self, *modules: str) -> None:
        self.modules = list(modules)


#: A real record, trimmed: the head of `.artifacts/cpp/p5c.jsonl` from an 8-camera run.
META = {
    "system": "cpp",
    "config": {
        "cameras": 8,
        "fps": 5,
        "seconds": 25,
        "workers": 12,
        "pipeline_queue": 256,
        "instance_queue": 64,
        "enqueue_block_timeout_ms": 50,
        "stage_timeout_ms": 5000,
        "reassembly_capacity": 1024,
        "reassembly_timeout_ms": 1500,
        "reassembly_sweep_ms": 100,
        "policy": "locality_spillover",
        "gpus": [0, 1],
    },
    "models": [
        {"name": "ship_detector", "engine": "x", "instances_per_device": 2},
        {"name": "ship_segmenter", "engine": "x", "instances_per_device": 2},
    ],
}


class TestTheCeilingEachModuleIsScoredAgainst:
    def test_the_pipeline_module_takes_the_frame_queue(self, analyse_cpp: ModuleType) -> None:
        found = analyse_cpp.capacities(META, _Samples("pipeline"))

        assert found["pipeline"] == 256

    def test_a_model_module_takes_the_SMALLER_of_its_two_bounds(
        self, analyse_cpp: ModuleType
    ) -> None:
        """Here the workers (12) bind, below `instance_queue x instances x devices` (256)."""
        found = analyse_cpp.capacities(META, _Samples("ship_detector", "ship_segmenter"))

        assert found == {"ship_detector": 12, "ship_segmenter": 12}

    def test_the_queue_binds_where_it_is_the_smaller(self, analyse_cpp: ModuleType) -> None:
        """The case P5-C created: 64 x 1 x 1 = 64 under a 256-worker run.

        Before P5-C this could not happen -- every instance queue was 65536, so the worker
        count was always the smaller and always the right answer. Passing the loose bound now
        would set a plateau guard that never trips, which this file exists not to do.
        """
        meta = {
            "config": {**META["config"], "workers": 256, "gpus": [0]},
            "models": [{"name": "ship_embedder", "engine": "x", "instances_per_device": 1}],
        }
        found = analyse_cpp.capacities(meta, _Samples("ship_embedder"))

        assert found == {"ship_embedder": 64}

    def test_a_module_the_record_does_not_name_falls_back_rather_than_refusing(
        self, analyse_cpp: ModuleType
    ) -> None:
        found = analyse_cpp.capacities(META, _Samples("something_new"))

        assert found["something_new"] == 12


class TestTheBinaryReportsOccupancyAndNotRawMicroseconds:
    """`per_device_busy_pct` is the one number that says whether a stage is the bottleneck.

    Requests and rows say how much a model was ASKED to do; only time says whether it could.
    The three prior counters could not distinguish "the GPUs are full" from "the GPUs are
    idle and something upstream is short", and that distinction is what redirected the next
    optimisation away from the GPU (`NOT-GPU-BOUND-AT-FIVE-GPUS`).
    """

    def test_the_binary_prints_a_percentage_keyed_per_device(self) -> None:
        body = BENCH.read_text("utf-8")

        assert "per_device_busy_pct" in body, "the occupancy line is gone"
        assert "compute_us" in body, "and it has to be built from the summed time, not the EWMA"
        assert "ewma_latency_us" not in body.split("per_device_busy_pct")[1][:400], (
            "occupancy must not be derived from the EWMA: an exponential average times a "
            "batch count is not a total, which is the whole reason `compute_us` was added"
        )

    def test_the_cpp_window_is_the_whole_run_because_it_has_no_warmup(self) -> None:
        """Why the two planes divide by different windows, and the tripwire for the day they
        should not. The Python harness has `warmup_s` and rates occupancy over the steady
        window (`benchmarks/tests/test_occupancy_window.py`); this binary has no such flag, so
        its whole run IS the measurement and `options.seconds` is the right divisor. If a
        `--warmup` ever lands here, that divisor becomes the understatement the harness just
        stopped making -- which is what this assertion is for.
        """
        body = BENCH.read_text("utf-8")

        assert "--warmup" not in body, (
            "cli/bench.cpp grew a warm-up window, so `per_device_busy_pct` now divides by a "
            "window that contains idle time -- subtract a boundary snapshot the way "
            "`benchmarks/harness/shipinfer.py`'s `busy_pct` does"
        )
        assert "options.seconds" in body, "the occupancy divisor is gone"

    def test_the_summed_time_exists_on_both_planes(self) -> None:
        """The sync rule (V88/V89): a per-frame counter on one plane and not the other is the
        gap that made the crop fan-out invisible for months."""
        cpp = (ROOT / "csrc" / "shipinfer" / "engine" / "instance.cpp").read_text("utf-8")
        py = (ROOT / "src" / "shipinfer" / "engine" / "instance.py").read_text("utf-8")

        assert "stats_.compute_us +=" in cpp, "C++ does not sum execute time"
        assert "_executed_compute_us +=" in py, "Python does not sum execute time"
        assert '"compute_us"' in py, "and Python's stats() does not report it"


class TestTheDriverDoesNotHideTheChainLine:
    """The standard driver's summary must show which slots did NOT run.

    `bench` announces them on stderr at start-up -- `chain 'x': 5 stage(s), not run here:
    decode track mtmc output` -- and `run_cpp_bench.sh` greps the log into a summary with an
    alternation anchored on COUNTER names. So the line was in every log and in none of the
    summaries, and a reader of the documented output could quote a throughput number from a
    run whose chain was missing `track` and `mtmc` without ever seeing it. That happened,
    repeatedly, before this test existed.
    """

    def test_the_summary_grep_includes_the_chain_line(self) -> None:
        body = DRIVER.read_text("utf-8")
        greps = [line for line in body.splitlines() if line.startswith("grep -E")]

        assert greps, "no summary grep in the driver any more; this test is guarding nothing"
        assert any("chain " in line for line in greps), (
            "scripts/run_cpp_bench.sh greps the run log into its summary and the alternation "
            "does not include `chain `, so the slots this plane did not run are filtered out "
            "of the one output a reader actually sees:\n  " + "\n  ".join(greps)
        )

    def test_the_binary_still_prints_it_with_that_prefix(self) -> None:
        """The other half of the pair: a grep for `chain ` is only worth having while the
        binary still writes that prefix, and the two live in different languages."""
        assert '"chain \'"' in BENCH.read_text("utf-8"), (
            "csrc/shipinfer/cli/bench.cpp no longer starts that line with `chain '`, so the "
            "driver's grep above matches nothing"
        )


class TestEveryKeyTheAnalyserReadsIsOneTheBinaryWrites:
    """The cross-language check that P5-C needed and did not have.

    `analyse_cpp.py` names `meta["config"][...]` keys in Python; `meta_json` writes them in
    C++. Nothing at run time compares the two, and the failure arrives only when a real run is
    scored -- so it is read out of both files here instead.
    """

    #: Every JSON key `meta_json` writes as a literal, `\"<key>\":` in the C++ source.
    WRITTEN = re.compile(r'\\"(\w+)\\"\s*:')
    #: Every `<dict>["<key>"]` the analyser reads.
    READ = re.compile(r'\["(\w+)"\]')

    def _written(self) -> set[str]:
        # The eight carried settings are written by a loop over `setting_keys()`, so their
        # names live in the plan format rather than in a literal here.
        from shipinfer.topology.plan import SETTING_KEYS

        return set(self.WRITTEN.findall(BENCH.read_text("utf-8"))) | set(SETTING_KEYS)

    def test_every_key_read_is_one_the_binary_writes(self) -> None:
        read = set(self.READ.findall(SCRIPT.read_text("utf-8")))
        written = self._written()

        assert read, "the regex found no key at all, so this test would pass on anything"
        assert read <= written, (
            f"scripts/analyse_cpp.py reads {sorted(read - written)} from a run record, and "
            f"csrc/shipinfer/cli/bench.cpp writes {sorted(written)}. A renamed key travels to "
            f"the judge as a KeyError, after the run that produced the artefact"
        )

    def test_the_slots_the_plane_did_not_run_are_in_the_record(self) -> None:
        """`stages` says what WAS wired; without its complement a reader cannot subtract.

        The plane already prints `not run here: decode track mtmc output` on stderr at
        start-up, and that line did not survive into the artefact -- so a throughput number
        could be quoted from a run whose chain was missing `track` and `mtmc` with nothing in
        the record to say so. It could, and it was, repeatedly, in this session's own reports.
        """
        written = self._written()

        assert "stages" in written, "the half that was always there"
        assert "unsupported" in written, (
            "`meta_json` must write the slots this plane could not run, not only the ones it "
            "did: `benchmarks/harness/sampler.py`'s contract is that the omission travels "
            "with the data, and half a list is not the omission"
        )

    def test_the_key_this_pr_renamed_is_covered_by_that(self) -> None:
        """The regression itself: `buffer_capacity` was read here and is written nowhere."""
        assert "buffer_capacity" not in self._written()
        assert "buffer_capacity" not in self.READ.findall(SCRIPT.read_text("utf-8"))
        assert "pipeline_queue" in self._written()
