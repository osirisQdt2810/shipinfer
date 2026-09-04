"""The queue-seam parity gate, from the Python plane's side.

Same three jobs as the ingest gate: this plane still matches its own committed golden, the
differ still notices when a golden is perturbed, and each golden still *shows* the invariant
it was written for rather than merely being long.

Offline by design (ADR-001), and single-threaded: a queue run has no clock and no threads in
it, so unlike the ingest harness its whole trace is one ordered sequence -- which is the
point, because WHICH camera's item comes out next is the invariant here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from benchmarks.parity import compare
from benchmarks.parity.drive_queue import GOLDEN, load, run_queue_scenario
from benchmarks.parity.queue_scenario import load_queue_scenario
from benchmarks.parity.trace import FIELDS, FLEET_KINDS, read_trace
from shipinfer.core.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[2]
CPP_TRACE = ROOT / "csrc" / "tests" / "parity_trace.h"

NAMES = ("fair_eviction", "reject_is_the_default", "priority_lanes", "expiry_on_take")


def _records(name: str) -> tuple:
    return read_trace(GOLDEN / f"{name}.jsonl").records


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    path = tmp_path / f"{name}.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


class TestScenarioFormat:
    @pytest.mark.parametrize("name", NAMES)
    def test_every_committed_scenario_loads(self, name: str) -> None:
        scenario = load(name)

        assert scenario.name == name and scenario.ops

    @pytest.mark.parametrize(
        ("line", "why"),
        [
            ("queue nosuch", "queue is one of"),
            ("overflow sideways", "overflow is one of"),
            ("put cam0 1 urgent", "priority is one of"),
            ("put cam0 0", "rows must be >= 1"),
            ("take now", "takes no argument"),
            ("sprint cam0", "unknown directive"),
        ],
    )
    def test_a_malformed_line_is_refused_naming_it(
        self, tmp_path: Path, line: str, why: str
    ) -> None:
        """Named, because a scenario that loads in one plane and not the other is the worst
        outcome this harness has."""
        path = tmp_path / "bad.scn"
        path.write_text(
            "scenario bad\nqueue fair\ncapacity 2\noverflow reject\nmax_batch_size 2\n"
            f"put cam0\n{line}\n"
        )
        with pytest.raises(ConfigurationError, match=why):
            load_queue_scenario(path)

    def test_a_scenario_with_no_operations_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.scn"
        path.write_text(
            "scenario empty\nqueue fair\ncapacity 2\noverflow reject\nmax_batch_size 2\n"
        )
        with pytest.raises(ConfigurationError, match="no operations"):
            load_queue_scenario(path)


class TestPythonPlaneMatchesGolden:
    @pytest.mark.parametrize("name", NAMES)
    def test_the_run_reproduces_its_golden_exactly(self, name: str) -> None:
        report = compare(read_trace(GOLDEN / f"{name}.jsonl"), run_queue_scenario(load(name)))

        assert report.ok, report.render()

    @pytest.mark.parametrize("name", NAMES)
    def test_the_run_meets_the_floor_it_promises(self, name: str) -> None:
        scenario = load(name)

        assert len(run_queue_scenario(scenario).records) >= scenario.records_min


class TestTheGoldensShowTheirInvariant:
    """A golden that compared nothing would pass for ever. Each one promises a behaviour."""

    def test_eviction_charges_the_greediest_camera_and_spares_the_quiet_one(self) -> None:
        """The inherited starvation bug, inverted: the flood pays for itself."""
        drops = [r.fields() for r in _records("fair_eviction") if r.kind == "qdrop"]
        served = [
            r.fields()["camera"] for r in _records("fair_eviction") if r.kind == "qserved"
        ]

        assert [d["camera"] for d in drops] == ["cam_loud", "cam_loud"]
        assert {d["reason"] for d in drops} == {"evicted"}
        assert "cam_quiet" in served, "the quiet camera's one frame survived the flood"

    def test_the_default_policy_refuses_instead_of_evicting(self) -> None:
        puts = [
            r.fields()["status"] for r in _records("reject_is_the_default") if r.kind == "qput"
        ]
        stats = next(
            r.fields() for r in _records("reject_is_the_default") if r.kind == "qstats"
        )

        assert puts == ["accepted", "accepted", "rejected"]
        assert (stats["evicted"], stats["rejected"]) == (0, 1)

    def test_a_tracking_critical_frame_does_not_queue_behind_a_background_batch(self) -> None:
        served = [
            r.fields()["camera"] for r in _records("priority_lanes") if r.kind == "qserved"
        ]

        assert served == ["cam_tc", "cam_hi", "cam_bg", "cam_bg"]

    def test_an_expired_request_is_dropped_on_the_way_out_not_the_way_in(self) -> None:
        kinds = [(r.kind, r.fields()) for r in _records("expiry_on_take")]
        put = next(f for kind, f in kinds if kind == "qput" and f["camera"] == "cam_late")
        drop = next(f for kind, f in kinds if kind == "qdrop")

        assert put["status"] == "accepted", "expiry is not a refusal at the door"
        assert (drop["camera"], drop["reason"]) == ("cam_late", "expired")


class TestDifferFindsRealDrift:
    """The vacuity guard. A gate that cannot fail is worse than no gate at all."""

    def test_a_reordered_pair_of_served_records_is_reported(self, tmp_path: Path) -> None:
        """The whole reason these records are fleet-level: order IS the invariant."""
        lines = (GOLDEN / "priority_lanes.jsonl").read_text().splitlines()
        first, second = [i for i, line in enumerate(lines) if '"kind":"qserved"' in line][:2]
        lines[first], lines[second] = lines[second], lines[first]
        report = compare(
            read_trace(GOLDEN / "priority_lanes.jsonl"),
            read_trace(_write(tmp_path, "priority_lanes", lines)),
        )

        assert not report.ok and "camera" in report.differences[0].why

    def test_an_eviction_charged_to_the_wrong_camera_is_reported(self, tmp_path: Path) -> None:
        lines = (GOLDEN / "fair_eviction.jsonl").read_text().splitlines()
        victim = next(i for i, line in enumerate(lines) if '"kind":"qdrop"' in line)
        lines[victim] = lines[victim].replace("cam_loud", "cam_quiet")
        report = compare(
            read_trace(GOLDEN / "fair_eviction.jsonl"),
            read_trace(_write(tmp_path, "fair_eviction", lines)),
        )

        assert not report.ok and "cam_quiet" in report.differences[0].why

    def test_a_dropped_record_is_reported(self, tmp_path: Path) -> None:
        lines = (GOLDEN / "expiry_on_take.jsonl").read_text().splitlines()
        del lines[next(i for i, line in enumerate(lines) if '"kind":"qdrop"' in line)]
        report = compare(
            read_trace(GOLDEN / "expiry_on_take.jsonl"),
            read_trace(_write(tmp_path, "expiry_on_take", lines)),
        )

        assert not report.ok


class TestTheFleetKindsAgree:
    """The other half of `TestTheFieldTablesAgree`, which only compared the field names.

    Whether a kind is fleet-level decides whether its records are compared as ONE sequence or
    split per camera. The C++ writer spelled that as ``kind == "stop" || kind == "end"``, so
    a kind added to one plane and not the other would be grouped differently and the gate
    would compare two different things without saying so.
    """

    def test_the_cpp_set_names_the_same_kinds(self) -> None:
        body = re.search(
            r"static const std::set<std::string> kinds = \{(.*?)\};",
            CPP_TRACE.read_text(),
            re.DOTALL,
        )
        assert body, f"no kFleetKinds() initialiser in {CPP_TRACE}"

        assert set(re.findall(r'"(\w+)"', body.group(1))) == set(FLEET_KINDS)

    def test_every_queue_kind_is_fleet_level(self) -> None:
        assert {kind for kind in FIELDS if kind.startswith("q")} <= set(FLEET_KINDS)
