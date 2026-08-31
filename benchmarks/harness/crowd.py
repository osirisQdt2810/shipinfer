"""Compose crowd frames: an N x N mosaic of the single-person bench JPEGs.

**Measured before you reach for this** (`benchmarks/tests/test_crowd_yield.py`, real engine):
a single stock ``person`` photo already yields **13-18** detections, which is the 10-20 the
sizing assumes; a 2x2 mosaic gives 18-20; and a **4x4 gives 3-6**, because the detector's input
is a fixed 640x640 and sixteen photos land in ~160px cells whose people fall under the model's
minimum size. Composing more people into a frame does not compose more *detectable* ones. A mosaic of K = grid^2 real photos makes the detector emit ~K boxes per
frame — generated from real data, deterministically, which is what the no-fake rule
permits (generated is fine; random is not). Point the bench at the output with
``--person-frames``; nothing in the harness config changes.

    python scripts/compose_crowd_frames.py --src benchmarks/baseline/data/person \\
        --out .artifacts/person_crowd --grid 4 --frames 10

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
    """Every image in ``src_dir``, sorted by name so composition is reproducible."""
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
        type=lambda s: tuple(int(v) for v in s.split("x")),
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


if __name__ == "__main__":
    raise SystemExit(main())
