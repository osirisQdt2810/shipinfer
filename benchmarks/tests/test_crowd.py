"""The mosaic composer: determinism, geometry and refusals, with no device.

What it is NOT for: raising detections per frame. See ``test_crowd_yield.py`` -- a 4x4 mosaic
yields fewer people than a single photo, and the stock photos already carry 10-20.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL", reason="Pillow is needed by the composer; pip install '.[dev]'")

from PIL import Image

from benchmarks.harness.crowd import compose_crowd_frames, main, parse_args


@pytest.fixture()
def sources(tmp_path: Path) -> Path:
    """Three distinguishable source photos of different sizes and aspects."""
    src = tmp_path / "src"
    src.mkdir()
    for name, size, color in (
        ("a.jpg", (60, 40), (255, 0, 0)),
        ("b.jpg", (30, 90), (0, 255, 0)),
        ("c.png", (80, 80), (0, 0, 255)),
    ):
        Image.new("RGB", size, color).save(src / name)
    return src


class TestComposition:
    def test_writes_the_asked_count_at_the_asked_size(self, sources: Path, tmp_path: Path):
        written = compose_crowd_frames(
            sources, tmp_path / "out", grid=3, frames=4, size=(300, 210)
        )
        assert len(written) == 4
        assert all(p.exists() for p in written)
        assert Image.open(written[0]).size == (300, 210)

    def test_two_runs_are_byte_identical(self, sources: Path, tmp_path: Path):
        first = compose_crowd_frames(sources, tmp_path / "one", grid=2, frames=3)
        second = compose_crowd_frames(sources, tmp_path / "two", grid=2, frames=3)
        for a, b in zip(first, second, strict=True):
            assert a.read_bytes() == b.read_bytes()

    def test_consecutive_frames_differ(self, sources: Path, tmp_path: Path):
        """Frame i starts the source cycle at offset i — arrangements must not repeat."""
        written = compose_crowd_frames(sources, tmp_path / "out", grid=2, frames=2)
        assert written[0].read_bytes() != written[1].read_bytes()

    def test_every_cell_is_filled_never_distorted(self, tmp_path: Path):
        """A 2:3 source in a wide cell keeps its aspect: gray pads the sides, no stretch.

        A single tall source, so the assertion does not depend on which photo the cycle put
        in cell 0 for frame 0.
        """
        src2 = tmp_path / "tall"
        src2.mkdir()
        Image.new("RGB", (30, 90), (0, 255, 0)).save(src2 / "only.jpg")
        (tall,) = compose_crowd_frames(src2, tmp_path / "out2", grid=1, frames=1, size=(90, 30))
        img = Image.open(tall)
        left = img.getpixel((1, 15))
        center = img.getpixel((45, 15))
        assert center[1] > 200 and center[0] < 130  # the green source, centred
        assert left == (114, 114, 114)  # the pad, not a stretched source
        assert img.size == (90, 30)  # the asked frame size, not the source's


class TestRefusals:
    def test_an_empty_source_directory_is_refused(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match=r"no .*images"):
            compose_crowd_frames(empty, tmp_path / "out")

    def test_a_nonpositive_grid_or_count_is_refused(self, sources: Path, tmp_path: Path):
        with pytest.raises(ValueError, match="grid"):
            compose_crowd_frames(sources, tmp_path / "out", grid=0)
        with pytest.raises(ValueError, match="frames"):
            compose_crowd_frames(sources, tmp_path / "out", frames=0)


class TestCli:
    def test_main_writes_and_reports(self, sources: Path, tmp_path: Path, capsys):
        out = tmp_path / "cli"
        code = main(
            [
                "--src",
                str(sources),
                "--out",
                str(out),
                "--grid",
                "2",
                "--frames",
                "3",
                "--size",
                "100x60",
            ]
        )
        assert code == 0
        assert len(list(out.glob("crowd*.jpg"))) == 3
        assert "3 frame(s) of 4 photos" in capsys.readouterr().out


class TestItRefusesToMixRuns:
    """A tool sold on determinism must not hand back a set it did not write.

    `--frames 20` then `--frames 5` left fifteen files from the earlier run in place, and a
    bench pointed at the directory consumes all twenty.
    """

    def test_a_non_empty_output_directory_is_refused(self, sources: Path, tmp_path: Path):
        out = tmp_path / "out"
        compose_crowd_frames(sources, out, grid=2, frames=3)

        with pytest.raises(ValueError, match="not empty"):
            compose_crowd_frames(sources, out, grid=2, frames=2)

    def test_an_empty_or_absent_directory_is_fine(self, sources: Path, tmp_path: Path):
        """The refusal must not make the ordinary case harder."""
        assert compose_crowd_frames(sources, tmp_path / "absent", grid=1, frames=1)

        empty = tmp_path / "empty"
        empty.mkdir()
        assert compose_crowd_frames(sources, empty, grid=1, frames=1)


class TestBadPathsRefuseLikeEverythingElseHere:
    """A message naming the flag, not a bare OSError naming a path."""

    def test_a_missing_source_directory_says_which_flag(self, tmp_path: Path):
        with pytest.raises(ValueError, match="--src"):
            compose_crowd_frames(tmp_path / "nope", tmp_path / "out", grid=1, frames=1)

    def test_an_output_path_that_is_a_file_says_which_flag(self, sources: Path, tmp_path: Path):
        out = tmp_path / "afile"
        out.write_text("not a directory")

        with pytest.raises(ValueError, match="--out"):
            compose_crowd_frames(sources, out, grid=1, frames=1)


class TestTheSizeFlagFailsAsAFlag:
    def test_a_malformed_size_is_an_argparse_error_not_an_unpack(self):
        """`--size 1920x1080x3` used to build a 3-tuple and die on the unpack later."""
        with pytest.raises(SystemExit):
            parse_args(["--src", "s", "--out", "o", "--size", "1920x1080x3"])

    def test_a_well_formed_size_parses(self):
        assert parse_args(["--src", "s", "--out", "o", "--size", "800x600"]).size == (800, 600)
