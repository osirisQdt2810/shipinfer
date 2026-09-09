"""Every `importorskip` in the suite must be satisfiable by what CI installs.

A skip is invisible. CI run 34344651408 reported **248 skipped** against 9 in a local run, and
189 of those were the pinned submodule -- fair, ADR-010 keeps it out of any wheel -- but 26
were `fastapi`/`uvicorn` and 16 were `cv2`. So the KServe surface and the replay fixture writer
were covered on developers' machines and on no machine that gates a merge, while the tier they
live in is the one CLAUDE.md calls "must stay green".

`pyproject.toml` already argues this for `grpcio` ("CI installs `.[dev,cli]`, so without these
every proto test would skip there") and for `pillow`. This derives the rule instead of
restating it per package: read the `importorskip` targets out of the tree, and require each one
to come from something `pip install -e ".[dev,cli]"` resolves.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

#: Import name -> the distribution that provides it, where they differ. Identity otherwise.
DISTRIBUTION_OF = {
    "cv2": "opencv-python-headless",
    "PIL": "pillow",
    "grpc": "grpcio",
    "grpc_tools": "grpcio-tools",
    "google": "protobuf",
}

#: The one thing no `pip install` can supply: ADR-010 pins the kernels as a submodule at a
#: commit, deliberately not a wheel. An exemption list is the shape that rots, so it is
#: asserted below rather than trusted -- adding to it has to be a deliberate edit here.
NOT_ON_PYPI = frozenset({"shipvision"})

_TARGET = re.compile(r"""importorskip\(\s*["']([A-Za-z0-9_.]+)["']""")


def _distribution(requirement: str) -> str:
    """The normalised distribution name a requirement string names."""
    head = re.split(r"[<>=!;\s]", requirement, maxsplit=1)[0]
    return head.split("[")[0].replace("_", "-").lower()


def _requested() -> dict[str, list[str]]:
    """Every `importorskip` target under the collected test roots, with where it came from."""
    found: dict[str, list[str]] = {}
    for root in ("tests", "benchmarks/tests"):
        for path in sorted((ROOT / root).rglob("*.py")):
            for target in _TARGET.findall(path.read_text(encoding="utf-8")):
                found.setdefault(target, []).append(path.relative_to(ROOT).as_posix())
    return found


def _installed_by_ci() -> set[str]:
    """The distribution names `pip install -e ".[dev,cli]"` resolves, normalised.

    `torch` and the other hard dependencies count too: the CPU wheel is what CI installs
    before the package itself.
    """
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    entries = list(data["project"]["dependencies"]) + extras["dev"] + extras["cli"]
    return {_distribution(entry) for entry in entries}


class TestNoTestIsSkippedForSomethingPipCouldHaveInstalled:
    def test_there_are_targets_to_check(self) -> None:
        """Without this, a regex that stopped matching would pass the rule below."""
        assert len(_requested()) >= 8

    def test_every_target_comes_from_what_ci_installs(self) -> None:
        installed = _installed_by_ci()
        missing = {
            target: where
            for target, where in _requested().items()
            if target.split(".")[0] not in NOT_ON_PYPI
            and DISTRIBUTION_OF.get(target.split(".")[0], target.split(".")[0])
            .replace("_", "-")
            .lower()
            not in installed
        }

        assert not missing, (
            f"these are skipped on any machine that runs `pip install -e '.[dev,cli]'`, which "
            f"is every machine that gates a merge: {missing}. Add the distribution to `dev` "
            f"(the file already makes this argument for grpcio and pillow), or say in "
            f"`NOT_ON_PYPI` why no wheel can supply it."
        )

    def test_the_exemption_is_the_submodule_and_nothing_else(self) -> None:
        """The risky half of the rule above. `shipvision` is exempt because ADR-010 pins it as
        a submodule at a commit; anything else in here would be a skip excused by a list."""
        assert sorted(NOT_ON_PYPI) == ["shipvision"]

    def test_the_facade_and_opencv_are_the_ones_this_added(self) -> None:
        """Named, so a trim of `dev` re-hides them loudly rather than quietly. These are the
        26 + 16 skips that CI reported and a local run did not."""
        installed = _installed_by_ci()

        for distribution in ("fastapi", "uvicorn", "opencv-python-headless"):
            assert distribution in installed, (
                f"{distribution} left `dev`, so the tests that `importorskip` it are covered "
                "locally and on no machine that gates a merge"
            )

    def test_av_is_not_dragged_in(self) -> None:
        """The `video` extra would satisfy `cv2` too, and bring `av` with it -- a package no
        test asks for, which would then first run in CI having never run on a developer's
        machine. The narrow entry is the point, so it is pinned."""
        assert "av" not in _installed_by_ci()

    def test_the_facades_floors_are_not_two_different_numbers(self) -> None:
        """`dev` restates the `server` extra's requirements instead of referring to it, so
        this is what keeps the two from drifting -- the comment there asks for it and a comment
        cannot check. Every `server` requirement must appear in `dev` verbatim."""
        extras = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"][
            "optional-dependencies"
        ]

        assert set(extras["server"]) <= set(extras["dev"]), (
            "the `server` extra and `dev` now disagree about the facade's floors, so CI and a "
            f"deployment resolve different versions: {sorted(set(extras['server']) - set(extras['dev']))}"
        )
