"""Which files a hook is allowed to look at.

One module because three hooks ask the same question and a wrong answer is invisible: a hook
that scans the wrong tree reports findings nobody owns, or none at all.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["python_files"]

ROOT = Path(__file__).resolve().parents[2]


# doc: long the three ways this can wrongly report nothing, and why each is closed
def python_files(root: Path) -> list[Path]:
    """Every ``*.py`` this repository owns under ``root`` -- and nothing a submodule owns.

    `rglob` descended into `benchmarks/baseline`, so this project's conventions were applied
    to vendored source: 995 cap findings against 58, and 48 Napoleon orphans against zero, on
    any checkout that has the submodule. `git ls-files` reports a submodule as ONE entry
    rather than its contents, which is the distinction `rglob` cannot see and a name-based
    skip list would miss on the next one.

    **Every way this could report nothing is a bug, not a clean tree**, so:
    ``root.resolve()``, because ``git -C ROOT`` resolves a relative pathspec against ROOT and
    not against the caller's cwd -- a relative argument from anywhere else matched no index
    entry, and git exits 0 with no output, which read as "no Python here";
    ``--others --exclude-standard`` beside ``--cached``, because a file that is not yet
    ``git add``ed is not an index entry, so a new module was invisible to all three gates
    until it was staged (a populated submodule is still one gitlink entry either way);
    and ``exists()``, because the index still lists a tracked file deleted from the worktree.
    """
    root = root.resolve()
    # `--others --exclude-standard` beside `--cached`, so an unstaged file is checked too.
    argv = ["ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", str(root)]
    done = subprocess.run(["git", "-C", str(ROOT), *argv], capture_output=True, text=True)
    if done.returncode != 0:
        # No git (a source tarball, a vendored copy): walk, and accept that a submodule left
        # in the tree would be scanned. Checking too much beats checking nothing.
        return sorted(root.rglob("*.py"))
    named = (ROOT / name for name in done.stdout.split("\0") if name.endswith(".py"))
    return sorted(path for path in named if path.exists())
