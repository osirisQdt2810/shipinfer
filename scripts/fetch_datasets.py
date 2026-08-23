#!/usr/bin/env python3
"""Fetch the evaluation datasets, and report what they actually contain.

Two datasets, because no single public one has what this system needs:

**MOT17** gives crowded pedestrians *with ground truth*, which is the only way to measure
tracking quality — HOTA, IDF1 and ID switches all need to know who was really who. Its
sequences span roughly 8 to 45 people per frame, and this script reports the real figure
per sequence rather than asserting a remembered one, so the selection can be checked
against the target operating point (10-20 people per frame) instead of assumed.

**The Singapore Maritime Dataset** gives ships from onshore and onboard cameras, which is
the actual subject matter. It is detection-only: boxes and classes, no track ids.

What this does NOT provide, and no public dataset does: one scene containing both ships and
a crowd, annotated. So algorithm quality is measured on these two separately, and the
end-to-end 50-camera behaviour is measured by replaying them as synthetic streams — two
different kinds of evidence, and conflating them would let a fast system look accurate.

Everything lands under ``data/``, which is gitignored: these are third-party datasets under
their own licences and must not be committed.

    python scripts/fetch_datasets.py --list          # what is available, download nothing
    python scripts/fetch_datasets.py mot17           # the sequences at the target density
    python scripts/fetch_datasets.py mot17 --all-sequences
    python scripts/fetch_datasets.py smd
    python scripts/fetch_datasets.py --report        # density of what is already on disk
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

#: Upper bound on mean people-per-frame for the default selection. The system is specified
#: for 10-20 people per frame (references/bitbucket-subfaceid/docs), so a sequence far above
#: that measures a different problem — MOT20's ~100/frame is a crowd-density benchmark, not
#: this one.
TARGET_MAX_DENSITY = 20.0

DATASETS: dict[str, dict[str, object]] = {
    "mot17": {
        "repo_id": "Lekim89/MOT17",
        "repo_type": "dataset",
        "note": "MOTChallenge MOT17 — pedestrians with tracking ground truth (CC BY-NC-SA 3.0)",
        # The canonical train split. `train/` carries gt/, det/ and img1/; the FRCNN detector
        # variant is the usual choice because its public detections are the ones most
        # published numbers are computed against.
        "patterns": ["train/MOT17-{seq}-FRCNN/**"],
        "sequences": ["02", "04", "05", "09", "10", "11", "13"],
    },
    "smd": {
        "repo_id": "ARG-NCTU/Singapore_Maritime_Dataset_coco",
        "repo_type": "dataset",
        "note": "Singapore Maritime Dataset in COCO form — ships from shore and onboard",
        "patterns": ["data/**", "README.md"],
        "sequences": [],
    },
}


def _hub():
    # The Hub's xet transfer backend rate-limits anonymous callers hard (HTTP 429 on the
    # xet-read-token endpoint), and its failure is a bare ConnectionError mid-download rather
    # than a retry. The classic HTTP path is slower per file and actually completes, which is
    # the trade that matters for a one-off fetch.
    import os

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:  # pragma: no cover - the message is the point
        sys.exit(
            "huggingface_hub is required: pip install 'huggingface_hub>=0.24'\n"
            "(motchallenge.net is not reachable from this network, which is why the "
            "datasets come from a Hub mirror)"
        )
    return snapshot_download


def sequence_density(gt_file: Path) -> tuple[float, int, int, int]:
    """Mean, max people per frame, frame count, and total boxes, read from a gt.txt.

    MOTChallenge gt format is
    ``frame, id, x, y, w, h, conf, class, visibility``. Only ``class == 1``
    (pedestrian) with ``conf == 1`` counts toward the density — the file also contains
    distractor classes and ignore regions, and counting those would overstate the crowd by
    a third on some sequences.
    """
    per_frame: collections.Counter[int] = collections.Counter()
    total = 0
    with gt_file.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 8:
                continue
            try:
                frame = int(row[0])
                confidence = float(row[6])
                class_id = int(row[7])
            except ValueError:
                continue
            if class_id != 1 or confidence < 1.0:
                continue
            per_frame[frame] += 1
            total += 1
    if not per_frame:
        return 0.0, 0, 0, 0
    counts = list(per_frame.values())
    return sum(counts) / len(counts), max(counts), len(per_frame), total


def _download_with_retry(snapshot_download, *, attempts: int = 6, **kwargs) -> str:
    """Retry the snapshot with exponential backoff.

    Anonymous Hub traffic is rate-limited, and a partially-downloaded snapshot resumes: each
    retry re-checks what is already local and only fetches the rest, so retrying is cheap and
    a fetch of a few thousand files reliably needs several passes.
    """
    import time

    delay = 5.0
    for attempt in range(1, attempts + 1):
        try:
            # Serial rather than the default thread pool: the concurrency is what triggers
            # the rate limiter in the first place.
            return snapshot_download(max_workers=2, **kwargs)
        except Exception as error:  # any transport failure here is worth one more attempt
            if attempt == attempts:
                raise
            print(
                f"  attempt {attempt}/{attempts} failed ({type(error).__name__}); "
                f"resuming in {delay:.0f}s"
            )
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
    raise RuntimeError("unreachable")


def report(root: Path = DATA_ROOT) -> int:
    """Print the real per-sequence density of whatever is on disk."""
    gts = sorted(root.glob("**/gt/gt.txt"))
    if not gts:
        print(f"nothing to report — no gt.txt under {root}")
        return 1

    print(
        f"{'sequence':28s} {'frames':>7s} {'boxes':>9s} {'mean/frame':>11s} {'max':>5s}  target"
    )
    print("-" * 76)
    for gt in gts:
        sequence = gt.parent.parent.name
        mean, peak, frames, total = sequence_density(gt)
        verdict = "yes" if mean <= TARGET_MAX_DENSITY else "TOO CROWDED"
        print(f"{sequence:28s} {frames:7d} {total:9d} {mean:11.1f} {peak:5d}  {verdict}")
    print(f"\n'target' is mean <= {TARGET_MAX_DENSITY:.0f} people/frame, the operating point")
    print("this system is specified for. A sequence above it measures a denser problem.")
    return 0


def fetch(name: str, *, all_sequences: bool) -> Path:
    spec = DATASETS[name]
    snapshot_download = _hub()
    destination = DATA_ROOT / name

    patterns: list[str] = []
    for template in spec["patterns"]:  # type: ignore[union-attr]
        if "{seq}" in template:
            patterns.extend(
                template.format(seq=sequence)
                for sequence in spec["sequences"]  # type: ignore[union-attr]
            )
        else:
            patterns.append(template)

    print(f"{name}: {spec['note']}")
    print(f"  from {spec['repo_id']} -> {destination}")
    print(f"  patterns: {patterns}")
    path = _download_with_retry(
        snapshot_download,
        repo_id=str(spec["repo_id"]),
        repo_type=str(spec["repo_type"]),
        local_dir=str(destination),
        allow_patterns=patterns,
    )
    print(f"  done: {path}")
    if not all_sequences and name == "mot17":
        print(
            f"\n  All {len(spec['sequences'])} train sequences were fetched. Run with "
            f"--report to see which are at the target density; the over-dense ones are "
            f"still useful as an explicit overload case."
        )
    return Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("datasets", nargs="*", choices=[*DATASETS, []], help="which to fetch")
    parser.add_argument("--list", action="store_true", help="show what is available")
    parser.add_argument("--report", action="store_true", help="density of what is on disk")
    parser.add_argument("--all-sequences", action="store_true", help="do not filter by density")
    args = parser.parse_args()

    if args.list:
        for name, spec in DATASETS.items():
            print(f"{name:8s} {spec['note']}")
            print(f"{'':8s} {spec['repo_id']}")
        return 0
    if args.report:
        return report()
    if not args.datasets:
        parser.print_help()
        return 2

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        fetch(name, all_sequences=args.all_sequences)
    print()
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
