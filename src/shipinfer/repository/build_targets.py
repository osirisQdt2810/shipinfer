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
INSTALLED_BY_BUILD_ENGINES = frozenset({"ship_detector", "ship_segmenter"})

#: The models the script's `reid` target builds ONE engine for and installs NOWHERE, so their
#: remedy is that target plus a copy. NAMED rather than "everything else": a model this script
#: never heard of would otherwise be told to build a ReID ResNet-50 and copy it into a
#: detector's directory -- confidently wrong, and wrong for every repository but this demo.
BUILT_BY_REID_TARGET = frozenset({"ship_embedder", "person_embedder"})

#: What that `reid` target writes at the default precision, so a remedy can name the file.
#: `--fp16` writes `reid_r50_fp16.engine`; the remedy says the default because that is what
#: `--only reid` with no flag produces.
REID_ENGINE = "models/reid_r50_fp32.engine"
