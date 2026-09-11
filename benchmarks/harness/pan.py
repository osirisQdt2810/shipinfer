"""Pan a window across one large photo, so consecutive frames hold the SAME people moved.

**Why the bench needed this.** Every measurement so far replayed ten unrelated photographs at
20 fps, so a camera's scene changed completely every 50 ms -- and a tracker confirms a track by
agreeing with itself across frames. Measured 11 Sep: the graph offered `mtmc` 0.36 observations
per frame while its embedders processed ~9.5 rows per frame, and the age gate admitted nothing.
Both follow from the input; on this fixture the same graph offers 4.42 and the gate admits 62%.

A 4K source and a 1080p window give continuity from real pixels. Generated from real data and
deterministic, which is what the no-fake rule permits (see `crowd.py`).

    python scripts/make_pan_fixture.py --src benchmarks/baseline/data/person_4K \\
        --out .artifacts/person_pan --frames 400
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image

__all__ = ["pan_frames", "window_at"]

_SUFFIXES = (".jpg", ".jpeg", ".png")
_JPEG_QUALITY = 95


def _source(src: Path) -> Path:
    """One image: ``src`` itself, or the first of a directory by name.

    ONE, not all of them. Panning a directory would put a cut between every source photo and
    reintroduce exactly the discontinuity this exists to remove.
    """
    if src.is_file():
        return src
    if not src.is_dir():
        raise ValueError(f"--src {src} is neither a file nor a directory")
    found = sorted(p for p in src.iterdir() if p.suffix.lower() in _SUFFIXES)
    if not found:
        raise ValueError(f"no {'/'.join(_SUFFIXES)} images in {src}")
    return found[0]


def window_at(index: int, frames: int, travel: tuple[int, int]) -> tuple[int, int]:
    """The window's top-left at ``index`` of ``frames``, on a path that closes on itself.

    Raised cosines, one horizontal cycle to two vertical, so the last frame is adjacent to the
    first: an RTSP fixture is LOOPED by the server, and a path with ends would put one jump
    cut per lap back into the stream.
    """
    if frames < 2:
        raise ValueError("--frames must be at least 2 for a path to exist")
    phase = 2.0 * math.pi * (index % frames) / frames
    x = travel[0] / 2.0 * (1.0 - math.cos(phase))
    y = travel[1] / 2.0 * (1.0 - math.cos(2.0 * phase))
    return round(x), round(y)


def pan_frames(
    src: Path, out: Path, *, size: tuple[int, int] = (1920, 1080), frames: int = 400
) -> list[Path]:
    """Write ``frames`` crops of one photo, each a window step along the path. Deterministic."""
    source = _source(src)
    with Image.open(source) as image:
        picture = image.convert("RGB")
    travel = (picture.width - size[0], picture.height - size[1])
    # STRICTLY larger in at least one direction. Equal sizes are not a degenerate pan, they
    # are the old fixture: one frame answered N times, with nothing for a tracker to follow.
    if travel[0] < 0 or travel[1] < 0 or (travel[0] == 0 and travel[1] == 0):
        raise ValueError(
            f"{source.name} is {picture.width}x{picture.height} and the window is "
            f"{size[0]}x{size[1]}; the source has to be larger, in at least one direction"
        )
    # THE REFUSAL `crowd.py` MAKES, and this fixture needs it harder: 400 frames then 100
    # leaves 300 of the old lap, the encoder globs and sorts by name, and the stream becomes
    # one pan cut into another -- then cached under the directory's name and reused.
    if out.exists() and not out.is_dir():
        raise ValueError(f"--out {out} exists and is not a directory")
    if out.is_dir() and any(out.iterdir()):
        raise ValueError(
            f"{out} is not empty; panning into it would mix these frames with whatever is "
            f"already there, and the encoder takes the directory in name order. Choose a "
            f"fresh directory or empty this one."
        )
    if frames < 2:
        raise ValueError(f"--frames must be at least 2 for a path to exist, got {frames}")
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in range(frames):
        left, top = window_at(index, frames, travel)
        frame = picture.crop((left, top, left + size[0], top + size[1]))
        # ZERO-PADDED, because the fixture encoder takes the directory in NAME order and a
        # pan's order is its whole point: `frame9.jpg` before `frame10.jpg` is a jump cut.
        path = out / f"pan{index:05d}.jpg"
        frame.save(path, quality=_JPEG_QUALITY)
        written.append(path)
    return written


def _wxh(text: str) -> tuple[int, int]:
    """``1920x1080`` -> ``(1920, 1080)``."""
    parts = text.lower().split("x")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise argparse.ArgumentTypeError(f"expected WxH, got {text!r}")
    return int(parts[0]), int(parts[1])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The CLI `scripts/make_pan_fixture.py` hands through."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="a large photo, or a directory")
    p.add_argument("--out", type=Path, required=True, help="directory to write frames into")
    p.add_argument("--size", type=_wxh, default=(1920, 1080), help="window, default 1920x1080")
    p.add_argument(
        "--frames",
        type=int,
        default=400,
        help="frames on one lap of the path (default 400, which is 20 s at 20 fps)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the frames and say where they went."""
    args = parse_args(argv)
    written = pan_frames(args.src, args.out, size=args.size, frames=args.frames)
    print(f"wrote {len(written)} frame(s) of {args.size[0]}x{args.size[1]} to {args.out}")
    return 0
