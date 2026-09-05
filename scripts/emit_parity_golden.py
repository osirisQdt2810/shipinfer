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
# Run as a path, so the repository root is not on sys.path the way `-m` would put it --
# and `src` ahead of it, because in a git worktree an editable install still resolves
# `shipinfer` to the PRIMARY checkout, and a golden emitted from another commit's plane
# says nothing about this one. `pythonpath = [".", "src"]` is the same fix for pytest.
for entry in (str(ROOT), str(ROOT / "src")):
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)

from benchmarks.parity.drive_events import GOLDEN as EVENT_GOLDEN  # noqa: E402
from benchmarks.parity.drive_events import load as load_event  # noqa: E402
from benchmarks.parity.drive_events import render as render_event  # noqa: E402
from benchmarks.parity.drive_masks import GOLDEN as MASK_GOLDEN  # noqa: E402
from benchmarks.parity.drive_masks import load as load_mask  # noqa: E402
from benchmarks.parity.drive_masks import render_masks  # noqa: E402
from benchmarks.parity.drive_plan import GOLDEN as PLAN_GOLDEN  # noqa: E402
from benchmarks.parity.drive_plan import load_plan_scenario, render_plan  # noqa: E402
from benchmarks.parity.drive_python import GOLDEN, SCENARIOS, run_scenario  # noqa: E402
from benchmarks.parity.drive_queue import GOLDEN as QUEUE_GOLDEN  # noqa: E402
from benchmarks.parity.drive_queue import SCENARIOS as QUEUE_SCENARIOS  # noqa: E402
from benchmarks.parity.drive_queue import run_queue_scenario  # noqa: E402
from benchmarks.parity.drive_records import GOLDEN as RECORD_GOLDEN  # noqa: E402
from benchmarks.parity.drive_records import load as load_record  # noqa: E402
from benchmarks.parity.drive_records import render as render_record  # noqa: E402
from benchmarks.parity.queue_scenario import load_queue_scenario  # noqa: E402
from benchmarks.parity.scenario import load_scenario  # noqa: E402
from benchmarks.parity.trace import Trace, TraceWriter  # noqa: E402
from shipinfer.core.errors import ConfigurationError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """``python scripts/emit_parity_golden.py --scenario reconnect --emit-golden``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, help="a name under scenarios/, or a path")
    parser.add_argument(
        "--kind",
        choices=("ingest", "queue", "event", "plan", "record", "mask"),
        default="ingest",
        help="which seam: the camera actors, the request queue (scenarios/queues/), one\n        perception event (scenarios/events/), a resolved chain (scenarios/plans/), or\n        one frame's stage outputs through the production record builder\n        (scenarios/records/), or a segmentation engine's two outputs through the mask\n        fold (scenarios/masks/)",
    )
    parser.add_argument("--out", type=Path, help="write the trace here instead of stdout")
    parser.add_argument(
        "--emit-golden",
        action="store_true",
        help="write golden/<scenario>.jsonl -- the committed file BOTH planes are then held "
        "to. Regenerating one to make a plane pass is what this harness exists to prevent",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing golden")
    args = parser.parse_args(argv)

    if args.kind == "plan":
        # A chain file in, the plan text out. No trace and nothing to run: the artefact that
        # crosses the plane boundary IS the golden, so `records_min` has no meaning here.
        chain = load_plan_scenario(args.scenario)
        text = render_plan(chain)
        destination = PLAN_GOLDEN / f"{chain.stem}.plan" if args.emit_golden else args.out
        if destination is None:
            print(text, end="")
            return 0
        if args.emit_golden and destination.exists() and not args.force:
            raise ConfigurationError(
                f"{destination} already exists. A golden is captured once and committed; "
                f"pass --force only when the change to the plane IS the decision"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        print(f"wrote {destination} ({len(text.splitlines())} line(s))")
        return 0

    # One LINE rather than a trace, like `event` below -- the difference is which driver
    # builds it: `event` states finished records and compares the two JSON writers, `record`
    # states the frame's stage outputs and compares the two record BUILDERS.
    if args.kind == "record":
        named = Path(args.scenario)
        scenario = load_record(str(named) if named.suffix == ".scn" else args.scenario)
        line = render_record(scenario)
        destination = RECORD_GOLDEN / f"{scenario.name}.jsonl" if args.emit_golden else args.out
        if destination is None:
            print(line)
            return 0
        if args.emit_golden and destination.exists() and not args.force:
            raise ConfigurationError(
                f"{destination} already exists. A golden is captured once and committed; "
                f"pass --force only when the change to the plane IS the decision"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(line + "\n", encoding="ascii")
        print(f"wrote {destination} (1 record)")
        return 0

    # One area per crop, one per line -- the FOLD, which is one seam upstream of `record`:
    # a record scenario states already-reduced rows and cannot see a fold that is missing.
    if args.kind == "mask":
        named = Path(args.scenario)
        scenario = load_mask(str(named) if named.suffix == ".scn" else args.scenario)
        text = render_masks(scenario)
        destination = MASK_GOLDEN / f"{scenario.name}.txt" if args.emit_golden else args.out
        if destination is None:
            print(text, end="")
            return 0
        if args.emit_golden and destination.exists() and not args.force:
            raise ConfigurationError(
                f"{destination} already exists. A golden is captured once and committed; "
                f"pass --force only when the change to the plane IS the decision"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="ascii")
        print(f"wrote {destination} ({len(text.splitlines())} area(s))")
        return 0

    if args.kind == "event":
        # One line, not a trace: an event is a single value and what the planes must agree
        # on is its bytes. `--out`/`--emit-golden` mean the same as below.
        named = Path(args.scenario)
        scenario = load_event(str(named) if named.suffix == ".scn" else args.scenario)
        line = render_event(scenario)
        destination = EVENT_GOLDEN / f"{scenario.name}.jsonl" if args.emit_golden else args.out
        if destination is None:
            print(line)
            return 0
        if args.emit_golden and destination.exists() and not args.force:
            raise ConfigurationError(
                f"{destination} already exists. A golden is captured once and committed; "
                f"pass --force only when the change to the plane IS the decision"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(line + "\n", encoding="ascii")
        print(f"wrote {destination} (1 record)")
        return 0

    queues = args.kind == "queue"
    root = QUEUE_SCENARIOS if queues else SCENARIOS
    named = Path(args.scenario)
    path = named if named.suffix == ".scn" else root / f"{args.scenario}.scn"
    scenario = load_queue_scenario(path) if queues else load_scenario(path)
    trace = run_queue_scenario(scenario) if queues else run_scenario(scenario)
    lines = [record.to_line() for record in trace.records]
    if len(lines) < scenario.records_min:
        raise ConfigurationError(
            f"{path}: promised at least {scenario.records_min} record(s), produced "
            f"{len(lines)}. A vacuous trace is a golden that proves nothing"
        )
    if args.emit_golden:
        # Each seam writes under its own root, so a queue scenario and an ingest scenario
        # of the same name cannot overwrite one another with --force.
        destination = (QUEUE_GOLDEN if queues else GOLDEN) / f"{scenario.name}.jsonl"
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
