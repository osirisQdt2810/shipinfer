"""The ingest parity gate, from the Python plane's side.

Three jobs: the Python plane still matches its own committed golden, the differ still
notices when a golden is perturbed (a gate that cannot fail is worse than none), and the
known-divergence register is still a register rather than a suppression list.

Offline by design (ADR-001), and with no GPU tier at all: everything compared here is
scheduling, bookkeeping and error taxonomy, none of which needs a device.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import threading
import time
from pathlib import Path

import pytest

from benchmarks.parity import KNOWN, Record, compare, load_scenario
from benchmarks.parity.drive_python import GOLDEN, SCENARIOS, peek_us, run_scenario
from benchmarks.parity.trace import FIELDS, read_trace
from shipinfer.core.errors import ConfigurationError, SourceOpenError
from shipinfer.core.settings.ingest import CameraConfig, IngestSettings
from shipinfer.ingest.base import FrameSource
from shipinfer.ingest.camera.actor import CameraActor
from shipinfer.ingest.sink import CountingSink
from shipinfer.ingest.timing.backoff import ExponentialBackoff

ROOT = Path(__file__).resolve().parents[2]
CPP_TEST = ROOT / "csrc" / "tests" / "test_ingest_parity.cpp"
CPP_TRACE = ROOT / "csrc" / "tests" / "parity_trace.h"
README = ROOT / "benchmarks" / "parity" / "README.md"
LEDGER = ROOT / ".claude" / "TASKS.md"
HOOK = ROOT / "scripts" / "hooks" / "require_container.py"
THIS_FILE = Path(__file__)

NAMES = ("reconnect", "backpressure", "fatal_vs_retryable")


@pytest.fixture(scope="module")
def goldens() -> dict[str, tuple[str, ...]]:
    """Every committed golden, as lines -- read once, mutated per test on a copy."""
    return {name: tuple((GOLDEN / f"{name}.jsonl").read_text().splitlines()) for name in NAMES}


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    path = tmp_path / f"{name}.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def _fatal_health(lines: list[str]) -> int:
    """The one health line carrying the type-prefixed error, by content and not by position."""
    return next(
        i
        for i, line in enumerate(lines)
        if '"kind":"health"' in line and "SourceUnavailableError: " in line
    )


@pytest.fixture(scope="module")
def fatal_health() -> Record:
    """The fatal camera's health record from a fresh Python run -- built once, read thrice."""
    trace = run_scenario(load_scenario(SCENARIOS / "fatal_vs_retryable.scn"))
    return next(r for r in trace.records if r.kind == "health" and r.camera == "cam_fatal")


class TestScenarioFormat:
    """The format is read the same way by both planes, and refuses what it cannot run."""

    @pytest.mark.parametrize("name", NAMES)
    def test_every_committed_scenario_loads_and_names_itself(self, name: str) -> None:
        scenario = load_scenario(SCENARIOS / f"{name}.scn")
        assert scenario.name == name
        assert scenario.records_min > 0
        assert scenario.cameras

    def test_a_bad_line_is_refused_with_its_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.scn"
        path.write_text("scenario bad\nrecords_min 1\nwibble 3\n")
        with pytest.raises(ConfigurationError, match=r":3: unknown directive 'wibble'"):
            load_scenario(path)

    def test_a_script_that_never_finishes_is_refused(self, tmp_path: Path) -> None:
        """The last entry of each list repeats for ever, so an endless script would hang."""
        path = tmp_path / "endless.scn"
        path.write_text(
            "scenario endless\nrecords_min 1\nempty_read_sleep_ms 0\nreconnect_initial_ms 2\n"
            "reconnect_max_ms 8\nreconnect_factor 2.0\nreconnect_jitter 0.0\n"
            "camera cam0\nopen ok\nread frame\n"
        )
        with pytest.raises(ConfigurationError, match="never finishes"):
            load_scenario(path)

    def test_a_fleet_of_disabled_cameras_needs_no_terminator(self, tmp_path: Path) -> None:
        """No actor, no records: a camera out of the fleet cannot fail to end."""
        path = tmp_path / "quiet.scn"
        path.write_text(
            "scenario quiet\nrecords_min 0\nempty_read_sleep_ms 0\nreconnect_initial_ms 2\n"
            "reconnect_max_ms 8\nreconnect_factor 2.0\nreconnect_jitter 0.0\n"
            "camera cam0 disabled\n"
        )
        scenario = load_scenario(path)
        assert [c.enabled for c in scenario.cameras] == [False]
        trace = run_scenario(scenario)
        assert [r.kind for r in trace.records] == ["stop", "end"]

    def test_a_scenario_without_the_backoff_numbers_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "bare.scn"
        path.write_text("scenario bare\nrecords_min 1\n")
        with pytest.raises(ConfigurationError, match="missing required setting"):
            load_scenario(path)


class TestPythonPlaneMatchesGolden:
    """A fresh run of the real manager, record for record against the committed file."""

    @pytest.mark.parametrize("name", NAMES)
    def test_the_run_reproduces_its_golden_exactly(self, name: str) -> None:
        fresh = run_scenario(load_scenario(SCENARIOS / f"{name}.scn"))
        golden = read_trace(GOLDEN / f"{name}.jsonl")
        report = compare(fresh, golden)
        assert report.ok, report.render()
        # Same plane on both sides, so the register is not consulted at all: any difference
        # here is drift within one plane and no cross-plane decision can excuse it.
        assert report.accepted == ()


class TestDifferFindsRealDrift:
    """The vacuity guard. A gate that cannot fail is worse than no gate at all."""

    def test_a_deleted_drop_is_reported_against_its_camera(
        self, goldens: dict[str, tuple[str, ...]], tmp_path: Path
    ) -> None:
        lines = list(goldens["backpressure"])
        victim = next(i for i, line in enumerate(lines) if '"kind":"drop"' in line)
        del lines[victim]
        report = compare(
            read_trace(GOLDEN / "backpressure.jsonl"),
            read_trace(_write(tmp_path, "backpressure", lines)),
        )
        assert [d.camera for d in report.differences] == ["cam_loud"]
        assert "drop" in report.differences[0].why

    def test_two_swapped_state_records_are_reported(
        self, goldens: dict[str, tuple[str, ...]], tmp_path: Path
    ) -> None:
        lines = list(goldens["reconnect"])
        first, second = [i for i, line in enumerate(lines) if '"kind":"state"' in line][:2]
        lines[first], lines[second] = lines[second], lines[first]
        report = compare(
            read_trace(GOLDEN / "reconnect.jsonl"),
            read_trace(_write(tmp_path, "reconnect", lines)),
        )
        assert [d.camera for d in report.differences] == ["cam0"]
        assert "state" in report.differences[0].why and "from" in report.differences[0].why

    def test_a_retryable_error_relabelled_fatal_is_reported(
        self, goldens: dict[str, tuple[str, ...]], tmp_path: Path
    ) -> None:
        """And no known entry excuses it: the taxonomy is the contract, not a spelling."""
        lines = list(goldens["reconnect"])
        victim = next(i for i, line in enumerate(lines) if '"SourceOpenError"' in line)
        lines[victim] = lines[victim].replace("SourceOpenError", "SourceUnavailableError")
        report = compare(
            read_trace(GOLDEN / "reconnect.jsonl"),
            read_trace(_write(tmp_path, "reconnect", lines)),
        )
        assert [d.camera for d in report.differences] == ["cam0"]
        assert "outcome" in report.differences[0].why
        assert "SourceUnavailableError" in report.differences[0].why

    def test_a_known_divergence_cannot_excuse_a_same_plane_difference(
        self, goldens: dict[str, tuple[str, ...]], tmp_path: Path
    ) -> None:
        lines = list(goldens["fatal_vs_retryable"])
        lines[_fatal_health(lines)] = lines[_fatal_health(lines)].replace(
            "SourceUnavailableError: ", ""
        )
        report = compare(
            read_trace(GOLDEN / "fatal_vs_retryable.jsonl"),
            read_trace(_write(tmp_path, "fatal_vs_retryable", lines)),
        )
        assert not report.ok and report.accepted == ()

    def test_the_same_difference_across_planes_is_accepted(
        self, goldens: dict[str, tuple[str, ...]], tmp_path: Path
    ) -> None:
        """The other half of the rule: cross-plane, that one difference IS the register's."""
        lines = list(goldens["fatal_vs_retryable"])
        lines[0] = lines[0].replace('"plane":"python"', '"plane":"cpp"')
        lines[_fatal_health(lines)] = lines[_fatal_health(lines)].replace(
            "SourceUnavailableError: ", ""
        )
        report = compare(
            read_trace(GOLDEN / "fatal_vs_retryable.jsonl"),
            read_trace(_write(tmp_path, "fatal_vs_retryable", lines)),
        )
        assert report.ok, report.render()
        assert [a.known_id for a in report.accepted] == ["last_error_type_prefix"]


def _health(name: str, camera: str) -> dict[str, int | str]:
    """One camera's health record out of a committed golden, by name."""
    trace = read_trace(GOLDEN / f"{name}.jsonl")
    record = next(r for r in trace.records if r.kind == "health" and r.camera == camera)
    return record.fields()


class TestGoldenIsNotVacuous:
    """A golden that compared nothing would pass for ever. Each one promises a floor."""

    @pytest.mark.parametrize("name", NAMES)
    def test_the_golden_meets_the_floor_its_scenario_declares(self, name: str) -> None:
        scenario = load_scenario(SCENARIOS / f"{name}.scn")
        golden = read_trace(GOLDEN / f"{name}.jsonl")
        assert len(golden.records) >= scenario.records_min

    @pytest.mark.parametrize("name", NAMES)
    def test_the_golden_says_which_plane_emitted_it(self, name: str) -> None:
        """The register is asymmetric -- ``compare`` reads ``left.plane == "python"`` to
        decide which record is whose -- so a golden from another plane would have both sides
        of every entry the wrong way round and excuse each one's mirror image."""
        assert read_trace(GOLDEN / f"{name}.jsonl").plane == "python"

    def test_the_backpressure_golden_charges_the_drops_to_the_loud_camera(self) -> None:
        """ADR-005, asserted rather than snapshotted.

        A floor and a set of record kinds make the golden a photograph: whatever was captured
        is what passes, including a capture-time bug. `backpressure.scn` says in its own
        comment that "cam_loud is refused twice and then accepted; cam_quiet is never refused,
        and its counters have to prove it" -- so these are the numbers that promise.
        """
        assert _health("backpressure", "cam_loud")["frames_dropped"] == 3
        assert _health("backpressure", "cam_quiet")["frames_dropped"] == 0

    def test_the_fatal_golden_never_connected_the_camera_it_gave_up_on(self) -> None:
        """A fatal open is charged as a connect FAILURE and never as a connect: the whole
        point of the taxonomy is that this camera is not retried, so a `connects` of 1 would
        mean the actor had opened the source it declared unusable."""
        health = _health("fatal_vs_retryable", "cam_fatal")
        assert health["connects"] == 0
        assert health["connect_failures"] == 1

    def test_the_suite_covers_a_drop_a_retry_and_both_halves_of_the_taxonomy(self) -> None:
        records: list[Record] = []
        for name in NAMES:
            records.extend(read_trace(GOLDEN / f"{name}.jsonl").records)
        kinds = {record.kind for record in records}
        outcomes = {r.text[0] for r in records if r.kind == "source_open"}
        assert {"drop", "retry", "health", "state", "frame"} <= kinds
        assert {"SourceOpenError", "SourceUnavailableError"} <= outcomes
        assert any(r.kind == "source_read" and r.text[0] == "FrameDecodeError" for r in records)


class TestTheBackoffColumnIsReadFromThePlane:
    """`retry.peek_us` is the production backoff's own number, so it can actually differ."""

    def test_a_change_to_the_production_peek_breaks_the_golden(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The experiment that used to pass.

        `ExponentialBackoff.peek` was changed to `initial * factor ** (attempts + 1)` --
        doubling every reconnect delay in the fleet -- and this gate stayed green, because
        both planes recomputed `initial * factor ** attempt` from the scenario's own config
        and neither read its own backoff. The column could not fail while the README called
        it coverage. It is now stepped in lockstep with the actor's failure count, off a
        production `ExponentialBackoff` built from the same settings the actor's is.
        """
        original = ExponentialBackoff.peek
        monkeypatch.setattr(
            ExponentialBackoff, "peek", lambda self: original(self) * self.factor
        )
        report = compare(
            run_scenario(load_scenario(SCENARIOS / "reconnect.scn")),
            read_trace(GOLDEN / "reconnect.jsonl"),
        )
        assert not report.ok, "the retry delays doubled and the gate did not notice"
        assert "peek_us" in report.differences[0].why, report.render()

    def test_the_recorded_delay_is_the_one_the_actor_would_have_slept(self) -> None:
        """Not a formula that happens to agree with it: the value comes off the class the
        actor retries on, so the two cannot drift apart without this saying so."""
        scenario = load_scenario(SCENARIOS / "reconnect.scn")
        backoff = ExponentialBackoff(
            scenario.int_setting("reconnect_initial_ms") / 1000.0,
            scenario.int_setting("reconnect_max_ms") / 1000.0,
            factor=scenario.float_setting("reconnect_factor"),
            jitter=scenario.float_setting("reconnect_jitter"),
        )
        expected = []
        for _ in range(3):
            expected.append(peek_us(backoff))
            backoff.next_delay()
        retries = [
            r for r in read_trace(GOLDEN / "reconnect.jsonl").records if r.kind == "retry"
        ]
        assert [r.numbers[0] for r in retries] == [0, 1, 2]
        assert [r.numbers[1] for r in retries] == expected


class TestKnownDivergences:
    """The register is a register: cited, ledgered, reproduced, and mirrored in C++."""

    @pytest.mark.parametrize("entry_id", sorted(KNOWN))
    def test_every_entry_cites_a_file_that_exists_on_both_sides(self, entry_id: str) -> None:
        entry = KNOWN[entry_id]
        for side, citation in (("python", entry.python), ("cpp", entry.cpp)):
            path = re.search(r"(?:src|csrc)/[\w./]+\.(?:py|h|cpp)", citation)
            assert path, f"{entry_id}: the {side} side cites no file: {citation}"
            assert (ROOT / path.group(0)).is_file(), f"{entry_id}: no {path.group(0)}"

    @pytest.mark.parametrize("entry_id", sorted(KNOWN))
    def test_every_entry_has_an_open_ledger_line_and_a_reproducing_case(
        self, entry_id: str
    ) -> None:
        """The ledger line is looked UP, not pattern-matched.

        Asserting only that the string starts with ``"[ ] "`` checked the shape of a
        sentence: all three entries cited ``P6-D1/D2/D3`` while ``.claude/TASKS.md`` had no
        ``P6-D`` line at all, so nobody owned any of the three fixes and the test was green.
        The register's only defence against becoming a suppression list is that each entry
        is somebody's open work.
        """
        entry = KNOWN[entry_id]
        assert entry.ledger.startswith("[ ] "), (
            f"{entry_id}: a known divergence carries an OPEN ledger line naming the fix; a "
            f"closed one means the entry should have been deleted with the fix"
        )
        ledger_id = entry.ledger.split()[2]
        open_item = re.compile(rf"^\s*-?\s*\[ \]\s+`?{re.escape(ledger_id)}\b", re.M)
        assert open_item.search(LEDGER.read_text()), (
            f"{entry_id}: cites ledger item {ledger_id!r}, which is not an open line in "
            f"{LEDGER.relative_to(ROOT)}. An entry nobody owns the fix for is a suppression"
        )
        assert entry.case in THIS_FILE.read_text(), (
            f"{entry_id}: names case {entry.case!r}, which is in no test here -- an entry "
            f"nothing reproduces is a suppression, not a decision"
        )

    def test_the_two_halves_of_the_register_name_the_same_ids(self) -> None:
        """Both directions, because only one was guarded and the other was the hole.

        The C++ gate diffs against the same golden, so it carries the same exceptions -- and
        a reviewer widened its half with one line (``if (field == "frames_dropped") return
        "drops_are_counted_differently";``) and no entry here, re-applied a real C++ bug, and
        got ``38 checks, 0 failure(s)`` plus a green Python suite. Checking only that each
        Python entry appears in the ``.cpp`` cannot see that: the set has to be equal.
        """
        source = CPP_TEST.read_text()
        table = re.search(
            r"static const std::vector<KnownEntry> entries = \{(.*?)\n        \};",
            source,
            re.DOTALL,
        )
        assert table, f"no known_register() table in {CPP_TEST.name}"
        body = re.search(
            r"\n    std::string known_divergence\(.*?\n    \}\n", source, re.DOTALL
        )
        assert body, f"no known_divergence() in {CPP_TEST.name}"
        # The table is the register; the returns are read too, so an id smuggled straight
        # into the function -- past the table -- is not invisible to this test.
        there = set(re.findall(r'\{"(\w+)",', table.group(1)))
        there |= {found for found in re.findall(r'return\s+"([^"]*)"', body.group(0)) if found}
        here = {name for name, entry in KNOWN.items() if entry.explains is not None}
        assert there == here, (
            f"the divergence register has drifted: {sorted(there - here)} are excused by "
            f"{CPP_TEST.name} with no entry in known.py, and {sorted(here - there)} are "
            f"entries known.py's differ uses that the C++ gate does not honour"
        )


class TestTheKnownDivergencesAreStillReal:
    """Each entry, reproduced. When one starts failing, the fix landed -- delete the entry."""

    def test_the_python_plane_still_prefixes_the_exception_type(
        self, fatal_health: Record
    ) -> None:
        assert str(fatal_health.fields()["last_error"]).startswith("SourceUnavailableError: ")
        actor_cpp = (ROOT / "csrc/shipinfer/ingest/camera/actor.cpp").read_text()
        assert "last_error_ = redact_in(reason);" in actor_cpp, (
            "the C++ side of last_error_type_prefix has moved; re-read it before trusting "
            "the entry"
        )

    def test_a_fatal_open_leaves_the_python_failure_count_at_zero(
        self, fatal_health: Record
    ) -> None:
        assert fatal_health.fields()["consecutive_failures"] == 0
        assert fatal_health.fields()["connect_failures"] == 1
        actor_cpp = (ROOT / "csrc/shipinfer/ingest/camera/actor.cpp").read_text()
        assert "++consecutive_failures_;" in actor_cpp

    def test_a_second_stop_still_answers_live_on_the_python_plane(self) -> None:
        """Abandon a thread, let it finish, stop again: Python says clean, C++ says false."""
        gate, inside = threading.Event(), threading.Event()
        config = CameraConfig(camera_id="cam_block", uri="scripted://cam_block", source="x")
        actor = CameraActor(
            config,
            CountingSink(),
            settings=IngestSettings(),
            source_factory=lambda c, n: _BlockingSource(c, n, gate=gate, inside=inside),
        )
        actor.start()
        try:
            # Waited for, not assumed: `_run` checks the stop signal before it connects, so a
            # stop issued before the thread reached its open joins cleanly and the whole point
            # of the case is lost. Under a loaded machine that is most of the time.
            assert inside.wait(5.0), "the actor never reached its blocked open"
            assert actor.stop(timeout_s=0.05) is False, "a blocked open must be abandoned"
        finally:
            gate.set()
        deadline = time.monotonic() + 5.0
        while actor.is_running and time.monotonic() < deadline:
            time.sleep(0.005)
        assert actor.stop(timeout_s=1.0) is True
        header = (ROOT / "csrc/shipinfer/ingest/camera/actor.h").read_text()
        assert "STICKY, deliberately" in header, "the C++ header no longer states the rule"


class TestTheHarnessStaysPure:
    """It may reach `shipinfer.ingest`; it may not reach a runtime."""

    def test_no_module_here_imports_an_accelerator_runtime(self) -> None:
        """Not tidiness: the gate has to run on CI's plain runner and in the offline tier.

        `scripts/hooks/check_layers.py` walks `src/` only, so this is where the same rule is
        applied to `benchmarks/parity/`. An AST walk and not a grep, because the word `torch`
        in a docstring should not fail a commit.
        """
        forbidden = {"torch", "tensorrt", "onnxruntime", "cuda", "cv2", "fastapi", "uvicorn"}
        for module in sorted((ROOT / "benchmarks" / "parity").glob("*.py")):
            for node in ast.walk(ast.parse(module.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                roots = {name.split(".")[0] for name in names}
                assert not roots & forbidden, f"{module.name} imports {roots & forbidden}"


class TestTheFieldTablesAgree:
    """One record-kind table, spelled in two languages -- a field name is what a gate says."""

    def test_the_cpp_table_names_the_same_kinds_and_fields(self) -> None:
        body = re.search(
            r"static const std::map<std::string, FieldNames> table = \{(.*?)\n        \};",
            CPP_TRACE.read_text(),
            re.DOTALL,
        )
        assert body, f"no kFields() initialiser in {CPP_TRACE}"
        table = {
            kind: (
                tuple(re.findall(r'"([^"]+)"', numbers)),
                tuple(re.findall(r'"([^"]+)"', text)),
            )
            for kind, numbers, text in re.findall(
                r'\{"(\w+)",\s*\{\{([^}]*)\},\s*\{([^}]*)\}\}\}', body.group(1)
            )
        }
        assert table == dict(FIELDS), (
            "benchmarks/parity/trace.py's FIELDS and csrc/tests/parity_trace.h's kFields() "
            "have drifted; the two planes would then name the same field differently in a "
            "failing gate, or disagree about how many a record has"
        )


class TestTheReaderRefusesAMalformedTrace:
    """A golden read leniently is a gate that compared something other than it says."""

    def test_a_truncated_record_is_refused_naming_its_line(self, tmp_path: Path) -> None:
        """Arity was checked on write and not on read, so a golden with a short `n[]` got as
        far as `fields()` -- which zips names against values -- and died there as a bare
        `KeyError: 'frames_dropped'`, in whichever assertion happened to touch that field."""
        lines = list((GOLDEN / "backpressure.jsonl").read_text().splitlines())
        victim = next(i for i, line in enumerate(lines) if '"kind":"health"' in line)
        lines[victim] = lines[victim].replace('"n":[5,2,3,0,1,0,0]', '"n":[5,2,3]')
        path = _write(tmp_path, "backpressure", lines)
        with pytest.raises(ConfigurationError, match=rf"{path.name}:{victim + 1}: record"):
            read_trace(path)


class TestTheDocumentedCommandsRun:
    """Every command the README gives is one this project's own hook allows on a host.

    Not pedantry: the emitter was documented as `python -m benchmarks.parity.drive_python`,
    which `require_container.py` denies wholesale by module root -- though it imports numpy
    and `shipinfer.ingest`, touches no device and produces no measurement. A README that
    documents a denied command teaches the reader to reach for `SHIPINFER_ALLOW_HOST_RUN`,
    and that is how the container rule was lost the first time.
    """

    @staticmethod
    def _hook():
        spec = importlib.util.spec_from_file_location("require_container", HOOK)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _commands() -> list[str]:
        """Every runnable command in the README: fenced lines and inline `code` spans."""
        text = README.read_text()
        fenced = re.compile(r"```\w*\n(.*?)```", re.S)
        candidates = [
            line.strip() for block in fenced.findall(text) for line in block.splitlines()
        ]
        # The fences are removed before the inline spans are read: a stray pairing across a
        # ``` makes every later span in the file line up one backtick out.
        candidates += [span.strip() for span in re.findall(r"`([^`]+)`", fenced.sub("", text))]
        starts = ("python ", "pytest ", "./csrc/")
        return [c for c in candidates if c.startswith(starts)]

    def test_the_readme_documents_at_least_the_emitter_and_the_two_gates(self) -> None:
        commands = self._commands()
        assert any("emit_parity_golden" in c for c in commands), commands
        assert len(commands) >= 3, commands

    def test_the_container_hook_allows_every_one_of_them(self) -> None:
        verdict = self._hook().verdict
        refused = {c: verdict(c, str(ROOT)) for c in self._commands()}
        assert not any(refused.values()), {c: why for c, why in refused.items() if why}


class _BlockingSource(FrameSource):
    """Blocks inside ``open`` until released -- the only way to abandon a real thread."""

    name = "blocking"

    def __init__(
        self,
        config,
        counter=None,
        *,
        settings=None,
        gate: threading.Event,
        inside: threading.Event,
    ) -> None:
        super().__init__(config, counter, settings=settings)
        self._gate = gate
        self._inside = inside

    def _do_open(self) -> None:
        self._inside.set()
        self._gate.wait(5.0)
        raise SourceOpenError(self.camera_id, self.config.uri, "released")

    def _do_read(self):
        return None

    def _do_close(self) -> None:
        return None
