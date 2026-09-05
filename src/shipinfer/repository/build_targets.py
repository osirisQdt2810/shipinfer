"""Which models `scripts/build_engines.py` installs into a version directory.

A shipped fact about an UNSHIPPED script, and it lives here for the direction of the
dependency: `scripts/` is not in the wheel (`pyproject.toml`'s `packages.find` is `src` only)
and not in the runtime image (`deploy/docker/Dockerfile` copies `src`, `pyproject.toml` and
`model_repository`), so anything under `src/shipinfer/` that imports it crashes exactly where
the code is meant to run. The script imports this instead.

`tests/cli/test_plan_command.py` pins the two together, because the tests can see both.
"""

from __future__ import annotations

__all__ = ["INSTALLED_BY_BUILD_ENGINES", "REID_ENGINE"]

#: The models whose plan `scripts/build_engines.py` writes into `<model>/<version>/`.
#: The two embedders are absent by design: the script's `reid` target builds ONE engine that
#: both of them use, and installs it nowhere -- `--only ship_embedder` exits 2.
INSTALLED_BY_BUILD_ENGINES = frozenset({"ship_detector", "ship_segmenter"})

#: What that `reid` target writes, so a remedy can name the file to copy.
REID_ENGINE = "models/reid_r50_fp32.engine"
