# doc: long the container-hook reasoning is why this is a library and not a `-m` entry point
"""Compose crowd frames: an N x N mosaic of the single-person bench JPEGs.

**Measured before you reach for this** — the numbers live in ONE place,
``benchmarks/tests/test_crowd_yield.py``, which asserts them against a real engine; this is a
pointer to that table, not a second copy of it. In short: a single stock ``person`` photo
already yields 9-19 PEOPLE per frame (mean 15.2), which is the 10-20 the sizing assumes, so
this tool is **not** the way to raise detections per frame. A 4x4 mosaic yields 4-6, three
times *worse*, because the detector's input is a fixed 640x640 and sixteen photos land in
~160px cells whose people fall under the model's minimum size. A 2x2 (15-23) is modestly
better than a single photo, so the cliff is a function of cell size rather than of mosaicing
as such -- which is why the CLI defaults to 2 and why 4 is a footgun.

What it is genuinely for: deterministic variation in per-frame load, generated from real data,
which is what the no-fake rule permits (generated is fine; random is not). Point the bench at
the output with ``--person-frames``; nothing in the harness config changes.

    python scripts/compose_crowd_frames.py --src benchmarks/baseline/data/person \\
        --out .artifacts/person_crowd --grid 2 --frames 10

(The entry point is under ``scripts/`` because the container hook refuses ``python -m
benchmarks.*``; this module is the library half and is imported, not run.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

__all__ = ["compose_crowd_frames"]

_SUFFIXES = (".jpg", ".jpeg", ".png")
_PAD_RGB = (114, 114, 114)  # the letterbox gray every YOLO lineage trains against
_JPEG_QUALITY = 95


def _sources(src_dir: Path) -> list[Path]:
    """Every image in ``src_dir``, sorted by name so composition is reproducible.

    Refuses with a message like this module's other refusals, rather than letting
    ``iterdir()`` raise a bare ``FileNotFoundError``/``NotADirectoryError`` that names the
    path without saying which flag was wrong.
    """
    if not src_dir.is_dir():
        raise ValueError(f"--src {src_dir} is not a directory")
    found = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in _SUFFIXES)
    if not found:
        raise ValueError(f"no {'/'.join(_SUFFIXES)} images in {src_dir}")
    return found


def _letterbox(image: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    """``image`` fit inside a cell, aspect preserved, gray-padded — never distorted."""
    scale = min(cell_w / image.width, cell_h / image.height)
    fitted = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    cell = Image.new("RGB", (cell_w, cell_h), _PAD_RGB)
    cell.paste(fitted, ((cell_w - fitted.width) // 2, (cell_h - fitted.height) // 2))
    return cell


def compose_crowd_frames(
    src_dir: Path,
    out_dir: Path,
    *,
    grid: int = 2,
    frames: int = 10,
    size: tuple[int, int] = (1920, 1080),
) -> list[Path]:
    """Write ``frames`` mosaics of ``grid**2`` source photos each; return their paths.

    Frame ``i`` starts its source cycle at offset ``i``, so consecutive frames carry
    different arrangements (per-frame detector load varies, as real streams do) while
    two runs over the same inputs are byte-identical.
    """
    if grid < 1:
        raise ValueError(f"grid must be >= 1, got {grid}")
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    # A tool sold on determinism must not hand back a mixed set: `--frames 20` then
    # `--frames 5` used to leave 15 files from the earlier run, and a bench pointed at the
    # directory consumes all of them. Refusing beats deleting someone else's data.
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"--out {out_dir} exists and is not a directory")
    if out_dir.is_dir() and any(out_dir.iterdir()):
        raise ValueError(
            f"{out_dir} is not empty; composing into it would mix these frames with "
            f"whatever is already there. Choose a fresh directory or empty this one."
        )
    sources = _sources(src_dir)
    images = [Image.open(p).convert("RGB") for p in sources]
    width, height = size
    cell_w, cell_h = width // grid, height // grid
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i in range(frames):
        canvas = Image.new("RGB", (width, height), _PAD_RGB)
        for cell in range(grid * grid):
            source = images[(i + cell) % len(images)]
            row, col = divmod(cell, grid)
            canvas.paste(_letterbox(source, cell_w, cell_h), (col * cell_w, row * cell_h))
        path = out_dir / f"crowd{i + 1}.jpg"
        canvas.save(path, quality=_JPEG_QUALITY)
        written.append(path)
    return written


def _wxh(text: str) -> tuple[int, int]:
    """``WxH`` -> (w, h). An argparse error beats dying on the unpack two frames later."""
    parts = text.split("x")
    if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
        raise argparse.ArgumentTypeError(f"expected WxH with positive integers, got {text!r}")
    return int(parts[0]), int(parts[1])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, required=True, help="directory of source photos")
    p.add_argument("--out", type=Path, required=True, help="directory to write mosaics into")
    p.add_argument(
        "--grid",
        type=int,
        default=2,
        help="cells per side (default 2 -> 4 photos/frame; see the module docstring's "
        "measurement -- 4 puts each photo in a ~160px cell and yields FEWER detections)",
    )
    p.add_argument("--frames", type=int, default=10, help="mosaics to write (default 10)")
    p.add_argument(
        "--size",
        type=_wxh,
        default=(1920, 1080),
        metavar="WxH",
        help="output frame size (default 1920x1080)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    written = compose_crowd_frames(
        args.src, args.out, grid=args.grid, frames=args.frames, size=args.size
    )
    print(f"wrote {len(written)} frame(s) of {args.grid**2} photos each to {args.out}")
    return 0
