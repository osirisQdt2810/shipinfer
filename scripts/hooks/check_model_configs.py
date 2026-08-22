#!/usr/bin/env python3
"""Parse every staged model ``config.yaml`` so a typo fails at commit, not at start-up.

A model repository whose config does not validate is a server that refuses to boot. Finding
that out during ``git commit`` costs milliseconds; finding it out during a deploy costs a
rollback.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from shipinfer.core.errors import ConfigurationError
from shipinfer.repository import load_model_config


def main(argv: list[str]) -> int:
    failures = 0
    for name in argv:
        path = Path(name)
        try:
            config = load_model_config(path)
        except ConfigurationError as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            failures += 1
        else:
            print(f"{path}: ok ({config.name}, platform={config.platform})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
