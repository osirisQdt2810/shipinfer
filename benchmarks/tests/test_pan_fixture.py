"""The pan fixture's one promise: consecutive frames hold the same scene, barely moved.

A tracker confirms a track by agreeing with itself across frames, so the property that makes
this input usable is a BOUNDED step -- not that it is pretty. These run on a generated source
image rather than the bench data, which the offline tier does not check out.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from benchmarks.harness.pan import pan_frames, window_at

#: A fraction of the window's width. At 400 frames over 1920 px of travel the largest step is
#: 15 px, which is 0.8% -- a 300 px person moves a twentieth of their own height between
#: frames, and IoU association has an easy job. 2% is the loosest this may drift to.
MAX_STEP_FRACTION = 0.02


def source(tmp_path: Path, width: int = 3840, height: int = 2160) -> Path:
    """One image big enough to pan a 1080p window across."""
    path = tmp_path / "source.jpg"
    Image.new("RGB", (width, height), (90, 120, 150)).save(path)
    return path


def test_the_step_between_frames_is_small_enough_to_track() -> None:
    travel = (1920, 1080)
    frames = 400

    steps = []
    for index in range(frames):
        here = window_at(index, frames, travel)
        nxt = window_at(index + 1, frames, travel)
        steps.append(max(abs(nxt[0] - here[0]), abs(nxt[1] - here[1])))

    assert max(steps) <= MAX_STEP_FRACTION * 1920, (
        f"the window jumps {max(steps)} px between frames; a tracker that cannot follow it "
        f"leaves every row untracked, which is the input this fixture exists to replace"
    )


def test_the_path_closes_so_a_looped_stream_has_no_cut() -> None:
    """The RTSP server loops the fixture, so frame N-1 -> frame 0 is a step like any other."""
    travel, frames = (1920, 1080), 400

    assert window_at(frames, frames, travel) == window_at(0, frames, travel)
    last = window_at(frames - 1, frames, travel)
    first = window_at(0, frames, travel)
    assert max(abs(last[0] - first[0]), abs(last[1] - first[1])) <= MAX_STEP_FRACTION * 1920


def test_the_names_sort_into_the_pan_order(tmp_path: Path) -> None:
    """`encode_fixture` takes the directory in NAME order, so `pan9` before `pan10` is a cut."""
    written = pan_frames(source(tmp_path), tmp_path / "out", size=(1920, 1080), frames=12)

    assert [p.name for p in written] == sorted(p.name for p in written)
    assert len(written) == 12


def test_a_source_no_larger_than_the_window_is_refused(tmp_path: Path) -> None:
    """Panning needs somewhere to pan to; a 1080p source would answer one frame 400 times."""
    with pytest.raises(ValueError, match="larger"):
        pan_frames(source(tmp_path, 1920, 1080), tmp_path / "out", frames=8)


def test_a_non_empty_output_directory_is_refused(tmp_path: Path) -> None:
    """`--frames 400` then `--frames 100` would leave 300 frames of the old lap behind, and
    the encoder globs and sorts by name: one stream, two pans, a cut where they meet."""
    src, out = source(tmp_path), tmp_path / "out"
    pan_frames(src, out, size=(1280, 720), frames=6)

    with pytest.raises(ValueError, match="not empty"):
        pan_frames(src, out, size=(1280, 720), frames=4)


def test_an_empty_or_absent_directory_is_fine(tmp_path: Path) -> None:
    """The refusal must not make the ordinary case harder."""
    assert pan_frames(source(tmp_path), tmp_path / "absent", size=(1280, 720), frames=4)


def test_every_frame_is_the_window_size(tmp_path: Path) -> None:
    written = pan_frames(source(tmp_path), tmp_path / "out", size=(1280, 720), frames=4)

    for path in written:
        with Image.open(path) as frame:
            assert frame.size == (1280, 720)
