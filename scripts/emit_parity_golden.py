#!/usr/bin/env python3
"""Emit one ingest-parity golden by running the scenario over the real Python plane.

The emitter's entry point lives here rather than under ``benchmarks/`` because
``scripts/hooks/require_container.py`` refuses ``python -m benchmarks.*`` wholesale --
correctly, for the bench runners, and wrongly for this one, which imports numpy and
``shipinfer.ingest``, touches no device and produces no measurement. Documenting a command
the project's own hook denies teaches the reader to reach for ``SHIPINFER_ALLOW_HOST_RUN``,
which is how the container rule was lost the first time, so the command moved instead.

A golden is emitted **once** and committed. Regenerating one to make a plane pass is the one
thing the parity harness exists to prevent, which is why ``--force`` is a separate word.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Run as a path (`python scripts/emit_parity_golden.py`), so the repository root is not
    # on the path the way `-m` would have put it there.
    sys.path.insert(0, str(ROOT))

from benchmarks.parity.drive_python import GOLDEN, SCENARIOS, run_scenario  # noqa: E402
from benchmarks.parity.scenario import load_scenario  # noqa: E402
from benchmarks.parity.trace import Trace, TraceWriter  # noqa: E402
from shipinfer.core.errors import ConfigurationError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """``python scripts/emit_parity_golden.py --scenario reconnect --emit-golden``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, help="a name under scenarios/, or a path")
    parser.add_argument("--out", type=Path, help="write the trace here instead of stdout")
    parser.add_argument(
        "--emit-golden",
        action="store_true",
        help="write golden/<scenario>.jsonl -- the committed file BOTH planes are then held "
        "to. Regenerating one to make a plane pass is what this harness exists to prevent",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing golden")
    args = parser.parse_args(argv)

    named = Path(args.scenario)
    path = named if named.suffix == ".scn" else SCENARIOS / f"{args.scenario}.scn"
    scenario = load_scenario(path)
    trace = run_scenario(scenario)
    lines = [record.to_line() for record in trace.records]
    if len(lines) < scenario.records_min:
        raise ConfigurationError(
            f"{path}: promised at least {scenario.records_min} record(s), produced "
            f"{len(lines)}. A vacuous trace is a golden that proves nothing"
        )
    if args.emit_golden:
        destination = GOLDEN / f"{scenario.name}.jsonl"
        if destination.exists() and not args.force:
            raise ConfigurationError(
                f"{destination} already exists. A golden is captured once and committed; "
                f"pass --force only when the change to the plane IS the decision"
            )
        _write(trace, destination)
        print(f"wrote {destination} ({len(lines)} record(s))")
    elif args.out:
        _write(trace, args.out)
        print(f"wrote {args.out} ({len(lines)} record(s))")
    else:
        print("\n".join(_render(trace)))
    return 0


def _render(trace: Trace) -> list[str]:
    """Back through the canonical writer, so the file on disk is the writer's own bytes."""
    writer = TraceWriter()
    writer.header(trace.scenario, trace.plane)
    for record in trace.records:
        writer.record(record.kind, record.camera, record.numbers, record.text)
    return writer.lines()


def _write(trace: Trace, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_render(trace)) + "\n", encoding="ascii")


if __name__ == "__main__":
    sys.exit(main())
