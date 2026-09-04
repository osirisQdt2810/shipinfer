"""Which files a hook is allowed to look at.

One module because three hooks ask the same question and a wrong answer is invisible: a hook
that scans the wrong tree reports findings nobody owns, or none at all.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["python_files"]

ROOT = Path(__file__).resolve().parents[2]


def python_files(root: Path) -> list[Path]:
    """Every ``*.py`` this repository owns under ``root`` -- and nothing a submodule owns.

    `rglob` descended into `benchmarks/baseline`, so this project's conventions were applied
    to vendored source: 995 cap findings against 58, and 48 Napoleon orphans against zero, on
    any checkout that has the submodule. CI never saw it because CI does not check them out.
    `git ls-files` reports a submodule as ONE entry rather than its contents, which is the
    distinction `rglob` cannot see and a name-based skip list would miss on the next one.
    """
    done = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--", str(root)],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        # No git (a source tarball, a vendored copy): walk, and accept that a submodule left
        # in the tree would be scanned. Checking too much beats checking nothing.
        return sorted(root.rglob("*.py"))
    return sorted(ROOT / name for name in done.stdout.split("\0") if name.endswith(".py"))
