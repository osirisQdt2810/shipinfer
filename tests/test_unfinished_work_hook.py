"""The `Stop` hook reads the ledger that actually exists.

Pure text parsing; the hook is not imported by `shipinfer`, so this costs nothing in layering.
The fixture is real ledger text — the first version gated collection on a heading the ledger
never had and reported nine open items as "clear".
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "unfinished_work.py"

LEDGER = """# Open work

| Mark | Meaning |
|---|---|
| `- [ ]` | open |
| `- [~]` | in progress |

## Phase 1 · Foundations

- [x] A1 The registry. Evidence: `tests/core/test_registry.py`.
- [ ] B4 Bench-command docs.
- [~] C1 The 5x measurement.
- [!] D2 Which GPUs may the bench use?

## Z · Final gate

- [ ] Z1 Everything above is `[x]` with evidence.
"""


@pytest.fixture(scope="module")
def hook():
    spec = importlib.util.spec_from_file_location("unfinished_work", HOOK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestOpenItemsReadsTheRealLedger:
    def test_open_and_in_progress_lines_are_found_under_any_heading(self, hook) -> None:
        items = hook.open_items(LEDGER)
        assert items == [
            "- [ ] B4 Bench-command docs.",
            "- [~] C1 The 5x measurement.",
            "- [ ] Z1 Everything above is `[x]` with evidence.",
        ]

    def test_done_waiting_and_the_legend_are_not_open(self, hook) -> None:
        items = hook.open_items(LEDGER)
        assert len(items) == 3
        assert all(item.startswith(hook.OPEN_MARKS) for item in items)

    def test_the_repository_ledger_is_seen(self, hook) -> None:
        """The check that would have failed: the shipped ledger has open items and the hook must
        see them, whatever its headings are called."""
        text = hook.LEDGER.read_text(encoding="utf-8")
        raw = [line for line in text.splitlines() if line.strip().startswith(hook.OPEN_MARKS)]
        assert hook.open_items(text) == [line.strip() for line in raw]

    def test_awaiting_operator_is_a_whole_line(self, hook) -> None:
        assert hook.awaiting_operator(LEDGER) is None
        text = LEDGER + "\nAWAITING-OPERATOR: which topology ships first?\n"
        assert hook.awaiting_operator(text) == "AWAITING-OPERATOR: which topology ships first?"


def _run(hook, monkeypatch, capsys, tmp_path, text: str, *, env_stop: str | None = None):
    ledger = tmp_path / "TASKS.md"
    ledger.write_text(text, encoding="utf-8")
    monkeypatch.setattr(hook, "LEDGER", ledger)
    monkeypatch.setattr(hook, "STATE", tmp_path / ".tasks_state.json")
    monkeypatch.setattr(hook.sys, "stdin", type("S", (), {"read": staticmethod(lambda: "")})())
    if env_stop is None:
        monkeypatch.delenv("SHIPINFER_ALLOW_STOP", raising=False)
    else:
        monkeypatch.setenv("SHIPINFER_ALLOW_STOP", env_stop)
    try:
        hook.main()
    except SystemExit as stop:
        assert stop.code == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


class TestTheDecision:
    def test_open_work_blocks_and_lists_it(self, hook, monkeypatch, capsys, tmp_path) -> None:
        out = _run(hook, monkeypatch, capsys, tmp_path, LEDGER)
        assert out["decision"] == "block"
        assert "B4 Bench-command docs" in out["reason"]
        assert "3 item(s)" in out["reason"]

    def test_a_clear_ledger_allows(self, hook, monkeypatch, capsys, tmp_path) -> None:
        out = _run(hook, monkeypatch, capsys, tmp_path, "# Open work\n\n- [x] Z1 done.\n")
        assert "decision" not in out
        assert "clear" in out["systemMessage"]

    def test_awaiting_operator_allows_and_repeats_the_question(
        self, hook, monkeypatch, capsys, tmp_path
    ) -> None:
        out = _run(
            hook, monkeypatch, capsys, tmp_path, LEDGER + "\nAWAITING-OPERATOR: may I merge?\n"
        )
        assert "decision" not in out
        assert "AWAITING-OPERATOR: may I merge?" in out["systemMessage"]

    def test_the_cap_stands_down_only_when_the_ledger_stops_changing(
        self, hook, monkeypatch, capsys, tmp_path
    ) -> None:
        monkeypatch.setattr(hook, "MAX_CONSECUTIVE", 2)
        assert _run(hook, monkeypatch, capsys, tmp_path, LEDGER)["decision"] == "block"
        assert _run(hook, monkeypatch, capsys, tmp_path, LEDGER)["decision"] == "block"
        out = _run(hook, monkeypatch, capsys, tmp_path, LEDGER)
        assert "decision" not in out and "standing down" in out["systemMessage"]
        # A changed ledger is progress: the budget is fresh again.
        assert (
            _run(hook, monkeypatch, capsys, tmp_path, LEDGER + "\n- [ ] new\n")["decision"]
            == "block"
        )
