"""``scripts/run_tests.sh`` finds an interpreter that has pytest, or says which problem it is.

The script's own comment records the failure it exists to prevent: falling through to whatever
`python` is on PATH "found the system interpreter and failed with `No module named pytest` --
which reads like a broken test suite rather than a wrong interpreter". That recurred on 9 Sep
in the shape the fix did not cover: **a git worktree has no `.venv` of its own**.

HERMETIC, and that is the point of the fixtures. An earlier draft asserted substrings of the
script and hard-coded `/usr/bin/python3` as an interpreter *without* pytest -- an assertion
about the machine, which `apt install python3-pytest` falsifies. Every interpreter here is a
stub in `tmp_path`, and the worktree case builds a real linked worktree and runs the script
from it, so the resolution is executed rather than spelled.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_tests.sh"

#: A stub that satisfies `import pytest` and then reports which path chose it. The script
#: `exec`s it, so what it writes is the script's own output.
WORKS = '#!/bin/sh\necho "CHOSE $0" >&2\nexit 0\n'

#: A stub with no pytest, whatever the machine happens to have installed.
NO_PYTEST = "#!/bin/sh\nexit 1\n"


def _stub(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _worktree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A real repository with a `.venv` stub, plus a linked worktree that has none."""
    main, linked = tmp_path / "main", tmp_path / "wt"
    git = ("git", "-c", "user.email=t@t", "-c", "user.name=t")
    subprocess.run(["git", "init", "-q", str(main)], check=True)
    (main / "scripts").mkdir()
    shutil.copy(SCRIPT, main / "scripts" / "run_tests.sh")
    # COMMITTED, not merely copied: `git worktree add` checks out the committed tree, so an
    # uncommitted script is absent from the linked worktree and `bash` fails with 127 before
    # the code under test runs at all.
    subprocess.run([*git, "add", "scripts/run_tests.sh"], cwd=main, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "i"], cwd=main, check=True)
    # The venv stays UNCOMMITTED and therefore absent from the worktree, which is the whole
    # situation: the dependencies live in the primary checkout only.
    stub = _stub(main / ".venv" / "bin" / "python", WORKS)
    subprocess.run([*git, "worktree", "add", "-q", str(linked)], cwd=main, check=True)
    return main, linked, stub


def _run(
    *args: str, cwd: Path | None = None, python: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = {"PATH": "/usr/bin:/bin"}
    if python is not None:
        env["PYTHON"] = python
    script = "scripts/run_tests.sh" if cwd is not None else str(SCRIPT)
    return subprocess.run(
        ["bash", script, *args],
        capture_output=True,
        text=True,
        cwd=str(cwd or ROOT),
        env=env,
    )


class TestAWorktreeUsesTheMainCheckoutsVenv:
    """The behaviour this change exists for, EXECUTED rather than spelled.

    An earlier draft asserted that `--git-common-dir` and `--path-format=absolute` appear in
    the file. Both survive a plausible slip: `${MAIN_ROOT%.git/}` for `${MAIN_ROOT%/.git}`
    strips nothing, the `-x` probe then tests `<repo>/.git/.venv/bin/python`, and the script
    falls through to the system interpreter with every character-assertion still green. That is
    the bug this fixes, reintroduced. `tests/test_container_hook.py`'s `TestTheGuardCanFail`
    makes the same argument about a guard that cannot fire.
    """

    def test_the_linked_worktree_reaches_the_primary_checkouts_venv(
        self, tmp_path: Path
    ) -> None:
        _main, linked, stub = _worktree(tmp_path)

        done = _run("-q", cwd=linked)

        assert f"CHOSE {stub}" in done.stderr, done.stderr
        assert done.returncode == 0

    def test_the_main_checkout_still_uses_its_own(self, tmp_path: Path) -> None:
        """The path that already worked, measured so the change is known not to have moved
        it."""
        main, _linked, stub = _worktree(tmp_path)

        done = _run("-q", cwd=main)

        assert f"CHOSE {stub}" in done.stderr, done.stderr


class TestItSaysWhichProblemItIs:
    def test_an_interpreter_without_pytest_names_itself(self, tmp_path: Path) -> None:
        """`No module named pytest` reads as a broken suite. This reads as a wrong
        interpreter, which is what it is."""
        stub = _stub(tmp_path / "nopytest", NO_PYTEST)

        done = _run("-q", python=str(stub))

        assert done.returncode == 1
        assert f"{stub} has no pytest" in done.stderr

    def test_a_named_interpreter_is_not_told_where_nobody_looked(self, tmp_path: Path) -> None:
        """With `PYTHON=` set no search happens, so describing one describes work nobody did.
        The two ways to fix it are still printed, because those still apply."""
        stub = _stub(tmp_path / "nopytest", NO_PYTEST)

        done = _run("-q", python=str(stub))

        assert "Looked for a venv" not in done.stderr
        assert "PYTHON=/path/to/python" in done.stderr
        assert "pip install -e" in done.stderr

    def test_a_search_that_failed_says_where_it_looked(self, tmp_path: Path) -> None:
        """The other half: when the script chose the interpreter, it owes an account of how."""
        _main, linked, stub = _worktree(tmp_path)
        stub.write_text(NO_PYTEST, encoding="utf-8")

        done = _run("-q", cwd=linked)

        assert done.returncode == 1
        assert "Looked for a venv" in done.stderr
        assert "the main checkout's" in done.stderr

    def test_a_working_named_interpreter_is_used_and_says_nothing(self, tmp_path: Path) -> None:
        """`PYTHON` wins over every search, which is how CI and the container name theirs."""
        stub = _stub(tmp_path / "works", WORKS)

        done = _run("-q", python=str(stub))

        assert done.returncode == 0
        assert "has no pytest" not in done.stderr


class TestAnActivatedVenvOutranksTheSearch:
    """A caller who activated one has said which interpreter they mean nearly as plainly as
    `PYTHON=`. Without this a *stale* `.venv` in the primary checkout silently wins from a
    linked worktree."""

    def test_virtual_env_beats_the_main_checkouts_venv(self, tmp_path: Path) -> None:
        _main, linked, main_stub = _worktree(tmp_path)
        active = tmp_path / "active"
        _stub(active / "bin" / "python", WORKS)

        done = subprocess.run(
            ["bash", "scripts/run_tests.sh", "-q"],
            capture_output=True,
            text=True,
            cwd=str(linked),
            env={"PATH": "/usr/bin:/bin", "VIRTUAL_ENV": str(active)},
        )

        assert f"CHOSE {active / 'bin' / 'python'}" in done.stderr, done.stderr
        assert str(main_stub) not in done.stderr


class TestTheFallthroughSurvives:
    """CI has no `.venv` at all and relies on `setup-python`'s interpreter, so removing the
    last resort would break the one machine that gates a merge."""

    def test_path_is_still_the_last_resort(self) -> None:
        assert "command -v python" in SCRIPT.read_text(encoding="utf-8")

    def test_finding_nothing_at_all_is_not_silent(self) -> None:
        """`set -e` on a failed command substitution aborts with ZERO output, and silence is
        what this block exists to remove -- so the substitution ends in `|| true` and the empty
        case gets its own message."""
        text = SCRIPT.read_text(encoding="utf-8")

        assert "|| true" in text
        assert "found no python interpreter at all" in text
