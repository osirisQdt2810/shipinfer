"""``scripts/build_csrc.py``'s external lanes, and the C++ table that has to agree with them.

Why this file exists: the build script is the only thing that knows which external lanes it
left out of a build, and the binary is the only thing that can *use* that fact — so the list
crosses the language boundary as ``-DSHIPINFER_OMITTED_LANES``, and
``csrc/shipinfer/ingest/omitted_lanes.h`` holds the other half of the mapping (which video
source names each lane registers). Two halves in two languages drift silently, and the failure
is invisible: a renamed lane makes ``create_source("gstreamer")`` fall back to "unknown video
source", which is exactly the message the mechanism exists to replace. Nothing at runtime would
notice. This does.

Offline by design (ADR-001): it reads two files and compiles nothing.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_csrc.py"
LANES_HEADER = ROOT / "csrc" / "shipinfer" / "ingest" / "omitted_lanes.h"
CSRC = ROOT / "csrc"


def _load(path: Path, name: str) -> ModuleType:
    """Import a script by path — ``scripts/`` is not a package on ``sys.path``."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_csrc() -> ModuleType:
    return _load(BUILD_SCRIPT, "build_csrc")


@pytest.fixture(scope="module")
def cpp_table() -> dict[str, set[str]]:
    """``kTable`` from ``ingest/omitted_lanes.h``: lane -> the source names it registers."""
    text = LANES_HEADER.read_text()
    body = re.search(r"kTable = \{(.*?)\};", text, re.DOTALL)
    assert body is not None, f"no kTable initialiser in {LANES_HEADER}"
    table = {
        lane: {name.strip().strip('"') for name in names.split(",")}
        for lane, names in re.findall(r'\{"([^"]+)",\s*\{([^}]*)\}\}', body.group(1))
    }
    assert table, "kTable parsed as empty; the regex and the header have diverged"
    return table


class TestTheLaneTablesAgree:
    """One lane list, spelled in two languages."""

    def test_every_lane_has_a_row_and_no_row_invents_a_lane(
        self, build_csrc: ModuleType, cpp_table: dict[str, set[str]]
    ) -> None:
        assert set(cpp_table) == set(build_csrc.EXTERNAL), (
            "scripts/build_csrc.py's EXTERNAL and ingest/omitted_lanes.h's kTable name "
            "different lanes. A lane in only one of them makes the refusal for its sources "
            "fall back to 'unknown video source' with nothing to notice it."
        )

    def test_each_row_lists_exactly_what_its_units_register(
        self, build_csrc: ModuleType, cpp_table: dict[str, set[str]]
    ) -> None:
        """The table's names must be the ones the lane's own ``SourceRegistrar``s use.

        Parsed out of the ``.cpp`` rather than trusted, because these strings are what an
        operator puts in a camera's ``source`` field: an alias added to a registrar and not to
        the table is a name that keeps getting the unhelpful answer. Whitespace-tolerant, so
        clang-format rewrapping a registrar does not fail this; a registration shape it cannot
        read fails loudly, which is the right outcome for a table that has to stay true.
        """
        registrar = re.compile(r'SourceRegistrar\s+\w+\(\s*"([^"]+)"\s*,\s*\{([^}]*)\}', re.S)
        for lane, spec in build_csrc.EXTERNAL.items():
            registered: set[str] = set()
            for unit in spec.units:
                found = registrar.findall((CSRC / unit).read_text())
                assert found, f"no SourceRegistrar found in {unit} (lane '{lane}')"
                for name, aliases in found:
                    registered.add(name)
                    registered |= {
                        a.strip().strip('"') for a in aliases.split(",") if a.strip()
                    }
            assert cpp_table[lane] == registered, (
                f"lane '{lane}': omitted_lanes.h lists {sorted(cpp_table[lane])}, but "
                f"{', '.join(spec.units)} registers {sorted(registered)}"
            )


class TestTheParityBinaryStaysOffline:
    """``test_ingest_parity`` has to be in every offline build, or its gate stops running."""

    def test_its_closure_reaches_no_driver_and_no_external_lane(
        self, build_csrc: ModuleType
    ) -> None:
        """CI runs whatever ``--offline`` produced, by glob, and never by name.

        So a parity binary whose includes reached ``core/platform.h`` or a ``pkg-config``
        lane would simply drop out of that build -- and the ingest parity gate would stop
        running with nothing red to say so. The binary is named ``test_ingest_parity``
        precisely to be picked up by that glob; this is the other half of the promise.
        """
        closure = build_csrc.include_closure(CSRC / "tests" / "test_ingest_parity.cpp")
        assert not build_csrc.needs_accelerator(closure)
        assert build_csrc.lanes_in(closure) == set(), (
            "the parity binary reached an external lane; `--offline` would leave it out of "
            "the build and CI's `for candidate in csrc/build/test_*` loop would find nothing"
        )


class TestTheDefineSaysWhatIsMissing:
    """``-DSHIPINFER_OMITTED_LANES`` is the whole contract with the C++ side."""

    def test_the_lanes_left_out_are_the_ones_named(self, build_csrc: ModuleType) -> None:
        assert build_csrc.lane_defines(frozenset()) == [
            '-DSHIPINFER_OMITTED_LANES="gstreamer,opencv"'
        ]
        assert build_csrc.lane_defines(frozenset({"gstreamer"})) == [
            '-DSHIPINFER_OMITTED_LANES="opencv"'
        ]

    def test_a_build_with_every_lane_defines_an_empty_list(
        self, build_csrc: ModuleType
    ) -> None:
        # Not "no define at all": every unit in every build gets the macro, so the reader in
        # omitted_lanes.h has one definition across the link and an undefined macro means only
        # "this binary was not built by that script".
        assert build_csrc.lane_defines(frozenset(build_csrc.EXTERNAL)) == [
            '-DSHIPINFER_OMITTED_LANES=""'
        ]

    def test_the_value_is_a_c_string_literal(self, build_csrc: ModuleType) -> None:
        # The quotes are part of the argument, and they have to be: `subprocess` is given an
        # argument list, so no shell strips them and the preprocessor must see a string literal
        # rather than a bare identifier that fails to compile.
        (define,) = build_csrc.lane_defines(frozenset())
        _, _, value = define.partition("=")
        assert value.startswith('"') and value.endswith('"')
