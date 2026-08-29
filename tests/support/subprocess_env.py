"""The environment a probe subprocess needs to be about *this* checkout.

A test that spawns ``python -c "import shipinfer..."`` is asserting something about the
source tree it lives in. Without help it asserts about whichever tree the editable install
points at — which in a git worktree is the *primary* checkout, at whatever commit that
happens to be on. The layering probes then pass or fail on code nobody is reviewing.

CI has one checkout so the two coincide there; this makes them coincide everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["checkout_env"]

_SRC = Path(__file__).resolve().parents[2] / "src"


def checkout_env() -> dict[str, str]:
    """``os.environ`` with this checkout's ``src`` ahead of anything installed."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_SRC}{os.pathsep}{existing}" if existing else str(_SRC)
    return env
