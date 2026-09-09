"""``scripts/run_tests.sh`` finds an interpreter that has pytest, or says which problem it is.

The script's own comment records the failure it exists to prevent: falling through to whatever
`python` is on PATH "found the system interpreter and failed with `No module named pytest` --
which reads like a broken test suite rather than a wrong interpreter". That recurred on 9 Sep,
in the shape the fix did not cover: **a git worktree has no `.venv` of its own**, so every
`run_tests.sh` from one fell straight through.

Two halves, and the second matters more: find the main worktree's venv (`--git-common-dir`
points at the primary `.git` from any linked worktree), and when no interpreter has pytest,
fail with the reason instead of with pytest's import error. CI legitimately has no `.venv` and
relies on `setup-python`, which is why the fallback stays.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_tests.sh"


def _run(*args: str, python: str | None = None) -> subprocess.CompletedProcess[str]:
    env = {"PATH": "/usr/bin:/bin"}
    if python is not None:
        env["PYTHON"] = python
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )


class TestItSaysWhichProblemItIs:
    def test_an_interpreter_without_pytest_names_itself(self) -> None:
        """`No module named pytest` reads as a broken suite. This reads as a wrong
        interpreter, which is what it is."""
        done = _run("-q", "--collect-only", python="/usr/bin/python3")

        assert done.returncode == 1
        assert "has no pytest" in done.stderr
        assert "/usr/bin/python3" in done.stderr

    def test_the_message_says_where_it_looked_and_how_to_override(self) -> None:
        """A refusal that does not say what to do next is a refusal someone works around."""
        done = _run("-q", "--collect-only", python="/usr/bin/python3")

        assert ".venv" in done.stderr
        assert "PYTHON=/path/to/python" in done.stderr

    def test_a_named_interpreter_that_works_is_used(self) -> None:
        """`PYTHON` wins over every search, which is how CI and a container name theirs."""
        done = _run("-q", "--collect-only", "tests/runtime/test_containment.py")

        assert done.returncode == 0, done.stderr[-2000:]
        assert "has no pytest" not in done.stderr


class TestAWorktreeBorrowsTheMainCheckoutsVenv:
    """A linked worktree has no `.venv`, and the dependencies are not going to be installed
    once per worktree. `--git-common-dir` resolves to the primary `.git` from any of them, so
    its parent is the checkout that does have one."""

    def test_the_search_reaches_for_the_main_worktree(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")

        assert "--git-common-dir" in text, (
            "the worktree fallback is gone, so `run_tests.sh` from a worktree falls through "
            "to the system interpreter again"
        )

    def test_the_path_format_is_absolute(self) -> None:
        """Without `--path-format=absolute`, `--git-common-dir` answers `.git` relative to the
        cwd -- so the venv lookup would be relative to wherever the caller stood."""
        text = SCRIPT.read_text(encoding="utf-8")

        assert "--path-format=absolute" in text

    def test_the_fallthrough_survives(self) -> None:
        """CI has no `.venv` at all and relies on `setup-python`'s interpreter. Removing the
        last resort would break the one machine that gates a merge."""
        text = SCRIPT.read_text(encoding="utf-8")

        assert "command -v python" in text
