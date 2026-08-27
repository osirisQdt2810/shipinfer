"""The two planes agree on where the engine lives (ADR-014, arch.md §9).

`csrc/` is a mirror of the Python tree, not a second tree that happens to look similar. A
rename that lands on one side only compiles fine — an include path is resolved against
`csrc/`, so `shipinfer/server/model.h` keeps working for exactly as long as the directory
survives — and the drift is then discovered by the next person who goes looking for the
native counterpart of `engine/model.py` and does not find one.

So the check is a grep rather than a build: it needs no toolchain, runs in the offline tier,
and fails on the include path itself rather than on a symptom of it.
"""

from __future__ import annotations

from pathlib import Path

CSRC = Path(__file__).resolve().parents[2] / "csrc"

#: The path this rename retired. Written as a fragment with both slashes so it matches an
#: include directive and a comment alike, and matches nothing in `csrc/shipinfer/engine/`.
RETIRED_PATH = "shipinfer/server/"

#: Build products are not source: `csrc/build/` holds objects and binaries whose bytes may
#: contain any string at all, including a path from a previous checkout.
_SKIP_DIRS = frozenset({"build"})


def _names_retired_path(path: Path) -> bool:
    return RETIRED_PATH in path.read_text(encoding="utf-8", errors="ignore")


def _source_files() -> list[Path]:
    return [
        path
        for path in sorted(CSRC.rglob("*"))
        if path.is_file() and not _SKIP_DIRS.intersection(path.relative_to(CSRC).parts)
    ]


class TestTheNativePlaneMirrorsThePythonOne:
    def test_the_engine_has_a_native_counterpart_at_the_matching_path(self) -> None:
        assert (CSRC / "shipinfer" / "engine" / "model.h").is_file()
        assert (CSRC / "shipinfer" / "engine" / "instance.h").is_file()
        assert not (CSRC / "shipinfer" / "server").exists()

    def test_no_file_under_csrc_still_names_the_old_path(self) -> None:
        """One offender is enough: an include that resolves is not evidence of anything."""
        files = _source_files()
        assert files, f"nothing found under {CSRC} — the walk, not the tree, is broken"

        offenders = [str(p.relative_to(CSRC)) for p in files if _names_retired_path(p)]

        assert not offenders, (
            f"csrc/ still names the pre-rename path {RETIRED_PATH!r}; the planes have "
            "drifted:\n" + "\n".join(offenders)
        )

    def test_the_predicate_would_catch_a_reintroduced_include(self, tmp_path: Path) -> None:
        """Without this the test above passes on a predicate that inspects nothing."""
        offender = tmp_path / "regression.h"
        offender.write_text('#include "shipinfer/server/model.h"\n')
        innocent = tmp_path / "fine.h"
        innocent.write_text('#include "shipinfer/engine/model.h"\n')

        assert _names_retired_path(offender)
        assert not _names_retired_path(innocent)
