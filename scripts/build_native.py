#!/usr/bin/env python3
"""Build the fused kernels, which now live in a submodule.

Kept as a thin delegator rather than deleted: `python scripts/build_native.py` is in the
docs, in muscle memory, and in a CI job, and a script that disappears leaves a
"command not found" that says nothing about where the code went.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMODULE = ROOT / "3rdparty" / "shipinfer-imgproc"


def main(argv: list[str]) -> int:
    driver = SUBMODULE / "build.py"
    if not driver.is_file():
        print(
            "the fused kernels live in the shipinfer-imgproc submodule, which is not "
            "checked out. Fetch it with:\n"
            "    git submodule update --init 3rdparty/shipinfer-imgproc",
            file=sys.stderr,
        )
        return 2

    print(f"# delegating to {driver.relative_to(ROOT)}")
    build = subprocess.run([sys.executable, str(driver), *argv])
    if build.returncode != 0:
        return build.returncode

    # The server loads the package, not a file in its own tree, so an editable install of
    # the submodule is what actually makes the build visible. Checking here turns a
    # confusing "provider fell back to python" into a sentence that says what to run.
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import shipinfer_imgproc; assert shipinfer_imgproc.is_available()",
        ],
        cwd=ROOT,
    )
    if check.returncode != 0:
        print(
            "\nthe kernels built, but this interpreter cannot import them. Install the "
            "submodule in editable mode:\n"
            "    pip install -e 3rdparty/shipinfer-imgproc",
            file=sys.stderr,
        )
    return check.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
