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

    def test_the_key_this_pr_renamed_is_covered_by_that(self) -> None:
        """The regression itself: `buffer_capacity` was read here and is written nowhere."""
        assert "buffer_capacity" not in self._written()
        assert "buffer_capacity" not in self.READ.findall(SCRIPT.read_text("utf-8"))
        assert "pipeline_queue" in self._written()
