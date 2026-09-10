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
        # `if name.strip()` because `{}` -- a lane that owns no video source, which
        # `shipvision` is -- otherwise splits to one EMPTY STRING and the row compares unequal
        # to the empty set it means. No lane had an empty row until one did.
        lane: {name.strip().strip('"') for name in names.split(",") if name.strip()}
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
                for name, aliases in found:
                    registered.add(name)
                    registered |= {
                        a.strip().strip('"') for a in aliases.split(",") if a.strip()
                    }
            # A lane need not own a video source (`shipvision` carries the tracker), and the
            # empty row saying so is CHECKED rather than tolerated: the assertion is
            # symmetric, so an empty row is right only when the units register nothing.
            assert cpp_table[lane] == registered, (
                f"lane '{lane}': omitted_lanes.h lists {sorted(cpp_table[lane])}, but "
                f"{', '.join(spec.units)} registers {sorted(registered)}"
                + (
                    ". An empty row is right only for a lane that registers no source at "
                    "all; this one registers some."
                    if registered and not cpp_table[lane]
                    else ""
                )
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


class TestOnlyGstLaneUnitsReachTheBus:
    """``sources/gstreamer_shared.h`` includes ``gst/gst.h``, and no other header here does.

    The closure walker attributes a lane to a ``.cpp`` unit, so it cannot see that a *header*
    needs somebody's ``-dev`` package: an offline unit that included this one would fail with
    ``gst/gst.h: No such file`` rather than the lane machinery's own message. That is why the
    invariant stated at the top of ``ingest/registry.cpp`` is held here instead of by a comment.
    """

    #: The lane package every unit allowed to reach that header already declares.
    PACKAGE = "gstreamer-1.0"
    HEADER = CSRC / "shipinfer" / "ingest" / "sources" / "gstreamer_shared.h"

    def _allowed(self, build_csrc: ModuleType) -> set[Path]:
        """The lane units carrying ``gstreamer-1.0`` — derived, so a third source is covered."""
        return {
            CSRC / unit
            for spec in build_csrc.EXTERNAL.values()
            if self.PACKAGE in spec.packages
            for unit in spec.units
        }

    def test_the_header_is_the_only_one_naming_gstreamer(self, build_csrc: ModuleType) -> None:
        del build_csrc
        offenders = [
            path.relative_to(CSRC).as_posix()
            for path in CSRC.rglob("*.h")
            if path != self.HEADER and "#include <gst/" in path.read_text(errors="replace")
        ]
        assert offenders == [], (
            f"{offenders} include a GStreamer header; only {self.HEADER.name} may, and only "
            f"because every unit that can reach it declares the {self.PACKAGE} lane"
        )

    def test_every_unit_reaching_it_declares_the_lane(self, build_csrc: ModuleType) -> None:
        allowed = self._allowed(build_csrc)
        assert allowed, "no lane declares " + self.PACKAGE
        reaching = {
            path
            for path in list(CSRC.rglob("*.cpp")) + list(CSRC.rglob("*.cu"))
            if self.HEADER in build_csrc.include_closure(path)
        }
        assert reaching <= allowed, sorted(
            p.relative_to(CSRC).as_posix() for p in reaching - allowed
        )

    def test_both_sources_shipped_today_do_reach_it(self, build_csrc: ModuleType) -> None:
        """Not vacuous: the two copies #156 found -- of the bus drain, and then of `gst_init` --
        are the reason this header exists."""
        for name in ("gstreamer.cpp", "nvdec.cpp"):
            unit = CSRC / "shipinfer" / "ingest" / "sources" / name
            assert self.HEADER in build_csrc.include_closure(unit), name


class TestTheNvdecSectionsRunFirst:
    """``test_ingest.cpp`` must exercise the NVDEC sections before anything gst-linked.

    That order is the only thing in the tree that can notice a source which does not initialise
    GStreamer itself. #156 shipped one, and the gate stayed green because the plugin probe and
    the GStreamer source ran first and initialised the library on its behalf -- a test that
    passed because of its neighbours, until the bench segfaulted on it.

    A comment saying "do not move these back down" is what #156 relied on too. This is the
    mechanical version, and it runs on a plain runner: no GStreamer, no GPU, no compiler.
    """

    INGEST = CSRC / "tests" / "test_ingest.cpp"

    def _order(self) -> list[str]:
        """The `test_*()` calls in `main()`, in the order the binary runs them."""
        text = self.INGEST.read_text()
        main = text[text.index("int main(") :]
        calls = re.findall(r"^\s*(test_\w+)\(\);", main, re.M)
        assert len(calls) > 20, f"parsed {len(calls)} calls from main(); the regex has drifted"
        return calls

    def _bodies(self) -> dict[str, str]:
        """Each test function's source, from its signature to the next one's."""
        text = self.INGEST.read_text()
        starts = [
            (m.group(1), m.start())
            for m in re.finditer(r"^\s*void (test_\w+)\(\) \{", text, re.M)
        ]
        assert starts, "no test function definitions found"
        ends = [at for _, at in starts[1:]] + [len(text)]
        return {name: text[at:end] for (name, at), end in zip(starts, ends, strict=True)}

    def _lane_of(self, body: str, table: dict[str, set[str]]) -> set[str]:
        """Which lanes' sources this test BUILDS, by the names they register.

        The lane's names come from `omitted_lanes.h` rather than a list here. What counts as
        using one is narrower than mentioning it, and it has to be: a redaction test that puts
        `"gstreamer"` in an error message reaches no library at all, and reading a bare literal
        flagged it. So a use is the source being SELECTED -- assigned to a camera's `source`, or
        asked of the registry -- which is what a test does immediately before building one.
        """
        used = set()
        for lane, names in table.items():
            for name in names:
                selects = rf'(\.source\s*=\s*"{name}")|(SOURCES\(\)\.contains\("{name}"\))'
                if re.search(selects, body):
                    used.add(lane)
        return used

    def test_no_gst_lane_section_runs_before_an_nvdec_one(
        self, cpp_table: dict[str, set[str]]
    ) -> None:
        table = cpp_table
        bodies = self._bodies()
        nvdec: list[int] = []
        gstreamer: list[tuple[int, str]] = []
        for index, name in enumerate(self._order()):
            lanes = self._lane_of(bodies.get(name, ""), table)
            if "nvdec" in lanes:
                nvdec.append(index)
            elif "gstreamer" in lanes:
                gstreamer.append((index, name))
        assert nvdec, "no test in main() asks for the nvdec source; this check went vacuous"
        assert gstreamer, "no gst-linked test either, so the ordering claim means nothing"
        too_early = [name for index, name in gstreamer if index < max(nvdec)]
        assert too_early == [], (
            f"{too_early} run before the NVDEC sections and initialise GStreamer for them, "
            f"which is what hid a missing `initialise_gstreamer()` in #159 -- move the NVDEC "
            f"calls back above them"
        )


class TestTheDefineSaysWhatIsMissing:
    """``-DSHIPINFER_OMITTED_LANES`` is the whole contract with the C++ side."""

    def test_the_lanes_left_out_are_the_ones_named(self, build_csrc: ModuleType) -> None:
        # Derived from `EXTERNAL` rather than spelled out, because a literal here means adding
        # a lane is a two-place edit and the second place gets forgotten -- which is the exact
        # class of drift `TestTheLaneTablesAgree` above exists to catch one file along. The
        # SHAPE is what matters: sorted, comma-joined, quoted.
        every = ",".join(sorted(build_csrc.EXTERNAL))
        assert build_csrc.lane_defines(frozenset()) == [f'-DSHIPINFER_OMITTED_LANES="{every}"']
        without_gstreamer = ",".join(sorted(set(build_csrc.EXTERNAL) - {"gstreamer"}))
        assert build_csrc.lane_defines(frozenset({"gstreamer"})) == [
            f'-DSHIPINFER_OMITTED_LANES="{without_gstreamer}"'
        ]
        assert "gstreamer" in every and "opencv" in every, "the two that have always been there"

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


class TestTheInTreeLaneAxis:
    """A lane may be a submodule rather than a ``pkg-config`` package, and one is.

    The axis exists because `3rdparty/shipvision` is a pinned submodule (ADR-010) with no
    `.pc` file, and the alternative — checking a generated `.pc` into the tree to satisfy a
    build script — is a build artefact pretending to be source. Both halves are asserted here
    because getting one right and the other wrong is silent: an include root with no sources
    compiles and fails at the link, and sources with no include root fail on the link line
    having compiled cleanly (which is what happened while writing this).
    """

    def test_exactly_the_in_tree_lanes_carry_a_root_and_sources_together(
        self, build_csrc: ModuleType
    ) -> None:
        for lane, spec in build_csrc.EXTERNAL.items():
            has_root, has_sources = bool(spec.include_root), bool(spec.sources)
            assert has_root == has_sources, (
                f"lane '{lane}' sets only one of include_root/sources. An include root with "
                f"no sources links against nothing; sources with no root fail on the link "
                f"line after compiling cleanly."
            )
            assert not (has_root and spec.packages), (
                f"lane '{lane}' is both in-tree and pkg-config. Pick one: the flags come "
                f"from the filesystem or from pkg-config, and both would double the -I."
            )

    def test_an_in_tree_lanes_include_root_reaches_the_link_line(
        self, build_csrc: ModuleType
    ) -> None:
        """The bug this caught: `link_flags` dropped every `-I`, so the lane's own sources
        were compiled on the link line without their headers. Four `fatal error:`s on a link
        whose compile step had been perfectly happy."""
        in_tree = [n for n, s in build_csrc.EXTERNAL.items() if s.include_root]
        if not in_tree:
            pytest.skip("no in-tree lane to check; the axis is unused")
        for lane in in_tree:
            if not (ROOT / build_csrc.EXTERNAL[lane].include_root).is_dir():
                pytest.skip(f"{lane}'s submodule is not checked out here")
            flags = build_csrc.link_flags({lane})
            assert any(f.startswith("-I") for f in flags), (
                f"lane '{lane}' contributes sources to the link but no -I, so g++ compiles "
                f"them there without their own headers"
            )
            assert any(
                f.endswith(".cpp") for f in flags
            ), f"lane '{lane}' declares sources but none reach the link line"

    def test_a_missing_submodule_is_refused_by_name_with_its_hint(
        self, build_csrc: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A submodule nobody checked out must read as "this lane is unavailable" — the same
        `SystemExit` a missing `-dev` package gives, because every caller already treats that
        as the answer. Silence here would compile the unit and fail on the header."""
        lane = next((n for n, s in build_csrc.EXTERNAL.items() if s.include_root), None)
        if lane is None:
            pytest.skip("no in-tree lane to check")
        spec = build_csrc.EXTERNAL[lane]
        monkeypatch.setattr(
            build_csrc,
            "EXTERNAL",
            {
                **build_csrc.EXTERNAL,
                lane: spec._replace(include_root="3rdparty/definitely-not-checked-out"),
            },
        )
        monkeypatch.setattr(build_csrc, "_PKG_CONFIG_CACHE", {})
        with pytest.raises(SystemExit) as raised:
            build_csrc.pkg_config_flags(lane)
        assert "definitely-not-checked-out" in str(raised.value)
        assert "git submodule update --init" in str(raised.value), "the hint has to travel"
