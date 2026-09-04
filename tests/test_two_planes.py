"""The seam inventory: which packages exist on both planes, and why the rest do not.

CLAUDE.md's sync rule says a change to a Python data-plane seam is not finished until the
C++ seam carries it. That only holds if somebody knows which packages *are* data-plane
seams -- and the answer lived in a ledger line ("8 of 11 mirror") that was measured once and
then aged. This file is that answer, asserted: a new package under `src/shipinfer/` or
`csrc/shipinfer/` fails the tier until it is placed in one of the three tables below.

The two undecided rows carry an OPEN ledger item, the way `benchmarks/parity/known.py`'s
register does, so "we have not decided yet" cannot quietly become "we decided not to".
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

#: `git` is absent from `pytorch/pytorch:*-runtime`, which is the image
#: `deploy/rootless/test.sh` collects the offline tier in. One honest skip rather than seven
#: errors about a missing binary: what these tests read is the TREE, which is identical on
#: the host and in the container, and the host and CI both have git.
pytestmark = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="git is not on PATH (the test image has none); the inventory is checked on the host",
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / ".claude" / "TASKS.md"

#: Packages that exist on both planes. These are the seams the sync rule is about.
MIRRORED = (
    "backends",
    "cli",
    "core",
    "engine",
    "ingest",
    "pipeline",
    "runtime",
    "scheduling",
)

#: Python-only, with the decision that made it so. ``None`` means the decision is still open
#: and the value is then the ledger item that owes it -- checked below, not just written.
PYTHON_ONLY = {
    "api": "ADR-014: the KServe surface is control plane, and fastapi enters at one seam only",
    "repository": "ADR-014: the model repository is data-driven config, read once at load",
    "launch": "process supervision and the parent half of the shard RPCs -- control plane, and"
    " `launch/signals.py` must not import torch at all, which is a Python constraint",
    "topology": "ADR-020: the chain is a Python declaration and the C++ plane receives a"
    " RESOLVED PLAN, which `topology/plan.py` writes. A registry, caps negotiation and a YAML"
    " loader are load-time control plane, and a second validator is a second door -- the"
    " failure being one plane accepting a chain the other refuses",
    "runners": "ADR-020: choosing WHERE a chain executes is control plane, and this plane's"
    " one execution loop is its binary's composition root under `csrc/shipinfer/cli/` --"
    " which `cli` already mirrors",
}

#: The ledger item that owes each undecided row. Empty since ADR-020 answered both, and
#: `test_every_deferral_is_in_owed_by` is what keeps it honest in either direction.
OWED_BY: dict[str, str] = {}

#: C++-only, with where its peer lives instead.
CPP_ONLY = {
    "obs": "the buffer-occupancy log; its Python peer is `benchmarks/harness/sampler.py`,"
    " because the sampling belongs to the harness that judges a run, not to the server"
}


# doc: long the two flags and the one skip are each a defect this file already caused
def _packages(root: str, repo: Path = ROOT) -> set[str]:
    """Package directories under ``root``, from git rather than the filesystem.

    `git ls-files`, not `iterdir`: `src/shipinfer/server/` still exists on a long-lived
    checkout as three `__pycache__` directories left by the #57 rename, and a filesystem walk
    calls that a package with no tracked file in it. The same reason
    `scripts/hooks/_paths.py` asks git (#124).

    ``--cached --others --exclude-standard``, which is the rest of that lesson and not
    decoration: `--cached` alone lists the INDEX, so a package added but not yet `git add`ed
    is invisible exactly while adding its row is cheap. `--exclude-standard` is what keeps
    `__pycache__` out, so the phantom stays filtered.
    """
    # `--others --exclude-standard` beside `--cached`: an unstaged package is still one.
    argv = ["ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", root]
    done = subprocess.run(["git", "-C", str(repo), *argv], capture_output=True, text=True)
    assert done.returncode == 0, f"git ls-files failed in {repo}: {done.stderr[:300]}"
    prefix = f"{root}/"
    return {
        name.removeprefix(prefix).split("/", 1)[0]
        for name in done.stdout.split("\0")
        if name.startswith(prefix) and "/" in name.removeprefix(prefix)
    }


class TestEveryPackageIsPlaced:
    """A package with no row is a seam nobody has decided about, which is the failure mode."""

    def test_every_python_package_is_mirrored_or_explained(self) -> None:
        placed = set(MIRRORED) | set(PYTHON_ONLY)
        found = _packages("src/shipinfer")

        assert found - placed == set(), (
            f"new package(s) under src/shipinfer with no row here: {sorted(found - placed)}. "
            f"Add it to MIRRORED (and port it) or to PYTHON_ONLY with the decision that "
            f"keeps it Python-side"
        )

    def test_every_cpp_package_is_mirrored_or_explained(self) -> None:
        placed = set(MIRRORED) | set(CPP_ONLY)
        found = _packages("csrc/shipinfer")

        assert (
            found - placed == set()
        ), f"new package(s) under csrc/shipinfer with no row here: {sorted(found - placed)}"

    def test_no_row_names_a_package_that_is_gone(self) -> None:
        """The other direction: a table that outlives its subject stops being a measurement."""
        python, cpp = _packages("src/shipinfer"), _packages("csrc/shipinfer")

        assert set(PYTHON_ONLY) <= python, sorted(set(PYTHON_ONLY) - python)
        assert set(CPP_ONLY) <= cpp, sorted(set(CPP_ONLY) - cpp)
        assert set(MIRRORED) <= python & cpp, sorted(set(MIRRORED) - (python & cpp))


class TestTheUndecidedRowsAreSomebodysWork:
    """``None`` is a decision deferred, and a deferral with no owner is a decision made."""

    @pytest.mark.parametrize("package", sorted(OWED_BY) or [None])
    def test_an_undecided_row_cites_an_open_ledger_item(self, package: str | None) -> None:
        if package is None:
            pytest.skip("no undecided rows; `test_every_deferral_is_in_owed_by` asserts that")
        assert PYTHON_ONLY[package] is None, f"{package} has a reason now; drop it from OWED_BY"
        item = OWED_BY[package]
        open_line = re.compile(rf"^\s*-?\s*\[!\]\s+\*{{0,2}}{re.escape(item)}\b", re.M)

        assert open_line.search(LEDGER.read_text()), (
            f"{package} defers to ledger item {item!r}, which is not an open `[!]` line in "
            f"{LEDGER.relative_to(ROOT)}. An undecided seam nobody owns is a decision made"
        )

    def test_every_deferral_is_in_owed_by(self) -> None:
        """The invariant that survives an EMPTY register, which the parametrize above cannot:
        an empty parameter set skips, and a skip is indistinguishable from a check that has
        stopped running. Both tables are empty since ADR-020 answered `topology` and
        `runners`, so this is the one that still asserts something."""
        deferred = {name for name, why in PYTHON_ONLY.items() if why is None}

        assert deferred == set(OWED_BY), sorted(deferred ^ set(OWED_BY))


# doc: long why this builds a throwaway repository instead of touching the real tree
class TestTheInventoryComesFromGit:
    """Not from the filesystem, which still carries a package the #57 rename deleted.

    Built in ``tmp_path`` as a real git repository, and NOT by making a directory under the
    checkout: `deploy/rootless/test.sh` mounts the repo `-v "$REPO:/work:ro"` and says why
    right above the mount, so a test that writes into the source tree is red in the container
    for ever while passing on the host. The first version of this test did exactly that.
    """

    @staticmethod
    def _repo(tmp_path: Path) -> Path:
        """A checkout with one tracked package and one that is only `__pycache__`."""
        for command in (
            ["init", "-q"],
            ["config", "user.email", "t@t"],
            ["config", "user.name", "t"],
        ):
            subprocess.run(["git", "-C", str(tmp_path), *command], check=True)
        (tmp_path / "src" / "shipinfer" / "core").mkdir(parents=True)
        (tmp_path / "src" / "shipinfer" / "core" / "__init__.py").write_text('"""Short."""\n')
        (tmp_path / "src" / "shipinfer" / "server" / "__pycache__").mkdir(parents=True)
        (
            tmp_path / "src" / "shipinfer" / "server" / "__pycache__" / "x.cpython-310.pyc"
        ).touch()
        (tmp_path / ".gitignore").write_text("__pycache__/\n")
        subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
        return tmp_path

    def test_a_directory_with_no_tracked_file_is_not_a_package(self, tmp_path: Path) -> None:
        found = _packages("src/shipinfer", repo=self._repo(tmp_path))

        assert "core" in found, "a package with a tracked file is one"
        assert "server" not in found, (
            "a directory that is only `__pycache__` is not a package; a filesystem walk calls "
            "it one and the inventory would then grow a row for something that does not exist"
        )

    def test_a_package_that_is_not_yet_staged_is_still_a_package(self, tmp_path: Path) -> None:
        """`--cached` alone would miss it exactly while adding its row is cheap."""
        repo = self._repo(tmp_path)
        (repo / "src" / "shipinfer" / "newseam").mkdir()
        (repo / "src" / "shipinfer" / "newseam" / "__init__.py").write_text('"""Short."""\n')

        assert "newseam" in _packages("src/shipinfer", repo=repo)
