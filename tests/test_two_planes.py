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
import subprocess
from pathlib import Path

import pytest

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
    "topology": None,
    "runners": None,
}

#: The ledger item that owes each undecided row.
OWED_BY = {"topology": "CSRC-TOPOLOGY-Q", "runners": "CSRC-TOPOLOGY-Q"}

#: C++-only, with where its peer lives instead.
CPP_ONLY = {
    "obs": "the buffer-occupancy log; its Python peer is `benchmarks/harness/sampler.py`,"
    " because the sampling belongs to the harness that judges a run, not to the server"
}


def _packages(root: str) -> set[str]:
    """Package directories under ``root``, from git rather than the filesystem.

    `git ls-files`, not `iterdir`: `src/shipinfer/server/` still exists on a long-lived
    checkout as three `__pycache__` directories left by the #57 rename, and a filesystem
    walk reports it as a package that no longer has a single tracked file in it. The same
    reason `scripts/hooks/_paths.py` asks git (#124).
    """
    done = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--", root],
        capture_output=True,
        text=True,
        check=True,
    )
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

    @pytest.mark.parametrize("package", sorted(OWED_BY))
    def test_an_undecided_row_cites_an_open_ledger_item(self, package: str) -> None:
        assert PYTHON_ONLY[package] is None, f"{package} has a reason now; drop it from OWED_BY"
        item = OWED_BY[package]
        open_line = re.compile(rf"^\s*-?\s*\[!\]\s+\*{{0,2}}{re.escape(item)}\b", re.M)

        assert open_line.search(LEDGER.read_text()), (
            f"{package} defers to ledger item {item!r}, which is not an open `[!]` line in "
            f"{LEDGER.relative_to(ROOT)}. An undecided seam nobody owns is a decision made"
        )

    def test_every_deferral_is_in_owed_by(self) -> None:
        deferred = {name for name, why in PYTHON_ONLY.items() if why is None}

        assert deferred == set(OWED_BY), sorted(deferred ^ set(OWED_BY))


class TestTheInventoryComesFromGit:
    """Not from the filesystem, which still carries a package the rename deleted."""

    def test_a_directory_with_no_tracked_file_is_not_a_package(self, tmp_path: Path) -> None:
        """`src/shipinfer/server/` is the live example: three `__pycache__` dirs and no file.

        A filesystem walk reports it as a fourteenth package, which would either fail the
        tier on a developer checkout or -- worse -- be "fixed" by adding a row for a package
        that does not exist. Same lesson as #124, one directory up.
        """
        phantom = ROOT / "src" / "shipinfer" / "__inventory_probe__" / "__pycache__"
        phantom.mkdir(parents=True)
        try:
            found = _packages("src/shipinfer")
        finally:
            phantom.rmdir()
            phantom.parent.rmdir()

        assert "__inventory_probe__" not in found
