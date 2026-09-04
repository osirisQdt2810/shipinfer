"""The perception-event gate, from the Python plane's side.

The event is a wire format a deployed `motservice` parses, so what the two planes must agree
on is its **bytes** -- key order included. That is why the C++ half compares strings rather
than parsing: it writes JSON and never reads it, and vendoring a parser for one format is
refused by the ponytail principle.

Offline by design (ADR-001): an event is a value, and building one needs no device.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from benchmarks.parity.drive_events import GOLDEN, SCENARIOS, load, render
from benchmarks.parity.event_scenario import FLOATS, load_event_scenario
from shipinfer.core.errors import ConfigurationError
from shipinfer.core.events import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[2]
CPP_SCHEMA = ROOT / "csrc" / "shipinfer" / "core" / "events" / "schema.h"
CPP_GATE = ROOT / "csrc" / "tests" / "test_event_parity.cpp"

NAMES = ("mixed_frame", "empty_frame", "vietnamese_camera", "evicted_frame")


def _golden(name: str) -> str:
    return (GOLDEN / f"{name}.jsonl").read_text().strip()


class TestPythonPlaneMatchesGolden:
    @pytest.mark.parametrize("name", NAMES)
    def test_the_run_reproduces_its_golden_byte_for_byte(self, name: str) -> None:
        assert render(load(name)) == _golden(name)

    @pytest.mark.parametrize("name", NAMES)
    def test_the_golden_is_one_line_of_parseable_json(self, name: str) -> None:
        payload = json.loads(_golden(name))

        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["type"] == "Det2MOT"


class TestTheGoldensReachTheHalfOfTheSchemaAHappyPathDoesNot:
    """A golden that only carried filled fields would compare nothing about nulls."""

    def test_the_mixed_frame_carries_a_null_in_every_optional_vector(self) -> None:
        payload = json.loads(_golden("mixed_frame"))

        assert None in payload["body_track_id_vec"], "a person with no published track"
        assert None in payload["body_track_state_vec"]
        assert None in payload["body_global_id_vec"]
        assert payload["ship_global_id_vec"] == [None], "a ship with no cross-camera identity"
        assert [] in payload["body_feature_vec"], "a person the embedder did not run for"

    def test_the_mixed_frame_splits_the_two_classes(self) -> None:
        payload = json.loads(_golden("mixed_frame"))

        assert len(payload["det_id_vec"]) == 2 and len(payload["ship_det_id_vec"]) == 1
        assert payload["ship_id_vec"] == [0], "0 is a legitimate gallery id, not 'unset'"
        assert payload["ship_mask_area_vec"] == [100.0]

    def test_the_mixed_frame_rounds_its_fps_rather_than_truncating(self) -> None:
        """`img_fps` is an int on the wire and 19.6 must not become 19."""
        assert json.loads(_golden("mixed_frame"))["img_fps"] == 20

    def test_the_empty_frame_is_a_complete_frame_with_nothing_in_it(self) -> None:
        """A frame with no people and a frame whose embedder timed out look identical in v1;
        `partial` and `missing_stages` are the difference, and both are written."""
        payload = json.loads(_golden("empty_frame"))

        assert payload["partial"] is False and payload["missing_stages"] == []
        assert all(
            payload[key] == []
            for key in payload
            if key.endswith("_vec") or key == "missing_stages"
        )


class TestTheKeyOrderIsTheContract:
    def test_the_first_ten_keys_are_v1s_det2mot_in_v1s_order(self) -> None:
        """A v1 consumer that validates strictly must still see a v1 message first."""
        keys = list(json.loads(_golden("mixed_frame")))

        assert keys[:10] == [
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
        ]

    def test_the_cpp_writer_names_the_same_keys_in_the_same_order(self) -> None:
        """Read out of the C++ source, so the two orders cannot drift silently.

        The C++ side writes the keys out one `out += ",\\"key\\":"` at a time precisely so
        this can be checked; assembling them from a map would sort them and every event
        would differ.
        """
        source = (CPP_SCHEMA.parent / "schema.cpp").read_text()
        theirs = re.findall(r'out \+= "?,?\\"(\w+)\\":', source)

        # `extra` is written only when non-empty, on both planes, so it is not in a golden
        # that carries none -- and it must stay LAST, which is where Python appends it.
        assert theirs[-1] == "extra", "the optional `extra` key is written last or not at all"
        assert theirs[:-1] == list(json.loads(_golden("mixed_frame"))), (
            "csrc/shipinfer/core/events/schema.cpp and core/events/schema.py have drifted on "
            "key order, which is part of the wire contract"
        )


class TestTheScenarioFormatRefusesWhatWouldFlap:
    def test_a_float_the_two_planes_may_spell_differently_is_refused(
        self, tmp_path: Path
    ) -> None:
        """The gate compares bytes, so an unlisted float would fail on formatting."""
        path = tmp_path / "bad.scn"
        path.write_text("scenario bad\nfps 0.30000000000000004\n")

        with pytest.raises(ConfigurationError, match="not in FLOATS"):
            load_event_scenario(path)

    def test_every_listed_float_is_spelled_the_same_by_both_planes(self) -> None:
        """The list is only safe if it is checked; `json.dumps` against the C++ table."""
        cases = dict(re.findall(r'\{([-\d.e+]+), "([^"]+)"\}', CPP_GATE.read_text()))
        for value in sorted(FLOATS):
            spelled = json.dumps(value)
            if spelled in cases.values() or str(value) in cases:
                assert cases.get(str(value), spelled) == spelled
        assert cases, "the C++ gate's number table was not found; the check would be vacuous"
        for literal, spelling in cases.items():
            assert json.dumps(float(literal)) == spelling, (
                f"the C++ gate expects {literal} -> {spelling!r}, Python writes "
                f"{json.dumps(float(literal))!r}"
            )

    @pytest.mark.parametrize(
        ("line", "why"),
        [
            ("sprint cam0", "unknown directive"),
            ("person only-a-det-id", "needs a det_id"),
            ("ship s 0.5 1 2 3 4 wobble 1", "unknown object keyword"),
        ],
    )
    def test_a_malformed_line_is_refused_naming_it(
        self, tmp_path: Path, line: str, why: str
    ) -> None:
        path = tmp_path / "bad.scn"
        path.write_text(f"scenario bad\n{line}\n")

        with pytest.raises(ConfigurationError, match=why):
            load_event_scenario(path)

    def test_a_scenario_with_no_name_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.scn"
        path.write_text("camera cam0\n")

        with pytest.raises(ConfigurationError, match="no scenario line"):
            load_event_scenario(path)


class TestANonAsciiCameraIdIsEscapedNotRefused:
    r"""`json.dumps` has `ensure_ascii=True`, and a Vietnamese camera id is ordinary input.

    An earlier C++ escaper THREW on any non-ASCII byte -- on a worker thread, from a sink that
    had never been able to throw before. Refusing at runtime to protect a byte gate is the
    wrong trade, so both planes write the `\uXXXX` and this scenario is what proves the two
    escapers agree rather than only their unit checks.
    """

    def test_the_golden_carries_the_escape_and_not_the_bytes(self) -> None:
        line = _golden("vietnamese_camera")

        assert '"camera_id":"c\\u1ea7u-c\\u1ea3ng-01"' in line
        assert "cầu" not in line, "ensure_ascii means the raw bytes never reach the wire"

    def test_it_round_trips_back_to_the_id_that_was_configured(self) -> None:
        """The escape is only correct if a consumer reads the original id back out."""
        assert json.loads(_golden("vietnamese_camera"))["camera_id"] == "cầu-cảng-01"


class TestTheReasonIsTheCollectorsWordOnBothPlanes:
    """Five words, and `failed` is not one of them.

    `runner.py` passes `result.reason` through verbatim, so the collector's vocabulary IS the
    wire vocabulary. The C++ port read `schema.py`'s docstring instead -- which said `failed`,
    a word nothing emits -- so a consumer alarming on ADR-005's eviction signal saw nothing
    from a C++ shard. The gate could not see it either: `reason` STATES the string and both
    planes echo it, which is why `finished` names the enum and lets each plane derive its own.
    """

    def test_the_golden_carries_the_collectors_word(self) -> None:
        assert json.loads(_golden("evicted_frame"))["reason"] == "evicted"

    def test_no_golden_carries_a_word_the_collector_never_writes(self) -> None:
        from shipinfer.pipeline.reassembly import collector

        known = {
            collector.COMPLETE,
            collector.INCOMPLETE,
            collector.TIMEOUT,
            collector.SHUTDOWN,
            collector.EVICTED,
        }

        for name in NAMES:
            reason = json.loads(_golden(name))["reason"]
            assert reason in known, f"{name}: {reason!r} is not one of {sorted(known)}"

    def test_the_scenario_format_refuses_a_reason_the_collector_has_no_name_for(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "bad.scn"
        path.write_text("scenario bad\nfinished failed\n")

        with pytest.raises(ConfigurationError, match="is not a FinishReason"):
            load_event_scenario(path)


class TestTheCommittedScenariosLoad:
    @pytest.mark.parametrize("name", NAMES)
    def test_every_committed_scenario_loads(self, name: str) -> None:
        scenario = load(name)

        assert scenario.name == name

    def test_every_scenario_on_disk_is_in_the_gate(self) -> None:
        """A scenario nothing runs is a golden nobody compares."""
        on_disk = {p.stem for p in SCENARIOS.glob("*.scn")}
        in_cpp = set(
            re.findall(r'test_this_plane_matches_the_golden\("(\w+)"\)', CPP_GATE.read_text())
        )

        assert on_disk == set(NAMES) == in_cpp
