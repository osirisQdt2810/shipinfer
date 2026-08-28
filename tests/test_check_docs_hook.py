"""The documentation cap is itself a guard, so it is tested like one.

Its two failure directions are not symmetric. A **false negative** — prose over the cap that
goes unreported — is silent, and the rule decays back into an intention. A **false positive**
is loud but worse in the end: it is what makes someone delete the hook, after which it checks
nothing. The escape hatch is where both live, and it is what `TestTheEscapeHatch` pins.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "check_docs.py"


def _hook():
    spec = importlib.util.spec_from_file_location("check_docs", HOOK)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(body, encoding="utf-8")
    return path


def _long_docstring(lines: int) -> str:
    """A ``def f()`` whose docstring is exactly ``lines`` long, closing quotes included."""
    body = "\n".join(f"    line {n}" for n in range(lines - 3))
    return f'def f():\n    """first\n\n{body}\n    """\n'


class TestItReportsWhatIsOverTheCap:
    def test_a_docstring_over_the_symbol_cap_is_named_with_its_length(self, tmp_path: Path):
        path = _write(tmp_path, _long_docstring(12))
        assert _hook().check(path) == [f"{path}:1: docstring of f is 12 lines (max 10)"]

    def test_a_comment_block_over_the_cap_is_named(self, tmp_path: Path):
        path = _write(tmp_path, "# one\n# two\n# three\n# four\n# five\nx = 1\n")
        assert _hook().check(path) == [f"{path}:1: comment block is 5 lines (max 4)"]

    def test_prose_within_the_caps_is_not_reported(self, tmp_path: Path):
        """Without this, a hook that reported nothing would pass every other test here."""
        path = _write(tmp_path, '"""Short."""\n\n# one\n# two\nx = 1\n')
        assert _hook().check(path) == []


class TestTheEscapeHatch:
    def test_a_marker_above_a_symbol_exempts_its_docstring(self, tmp_path: Path):
        path = _write(
            tmp_path, "# doc: long the wire format is normative\n" + _long_docstring(12)
        )
        assert _hook().check(path) == []

    def test_a_marker_heading_a_comment_block_exempts_the_block(self, tmp_path: Path):
        """The marker is itself a comment, so it joins the run it exempts."""
        body = (
            "# doc: long the wire format is normative\n# one\n# two\n# three\n# four\nx = 1\n"
        )
        assert _hook().check(_write(tmp_path, body)) == []

    def test_a_marker_heading_a_block_does_not_exempt_the_symbol_below_it(self, tmp_path: Path):
        """The quiet failure: one marker silently buying a docstring nobody marked."""
        body = "# doc: long the wire format is normative\n# one\n# two\n" + _long_docstring(12)
        reported = _hook().check(_write(tmp_path, body))
        assert reported == [f"{tmp_path / 'sample.py'}:4: docstring of f is 12 lines (max 10)"]


class TestAMarkerOnADecoratedSymbol:
    """Both honest placements work, because a false positive is how a hook gets deleted."""

    @pytest.mark.parametrize(
        "prefix",
        [
            "# doc: long the params table is the contract\n@register('x')\n",
            "@register('x')\n# doc: long the params table is the contract\n",
            "# doc: long the params table is the contract\n@register(\n    'x',\n)\n",
        ],
        ids=["above-the-decorator", "between-decorator-and-def", "multi-line-decorator"],
    )
    def test_the_marker_is_found_wherever_it_honestly_sits(self, tmp_path: Path, prefix: str):
        assert _hook().check(_write(tmp_path, prefix + _long_docstring(12))) == []

    def test_a_marker_with_no_reason_does_not_exempt_anything(self, tmp_path: Path):
        """An exemption that need not be argued is one nobody argues."""
        path = _write(tmp_path, "# doc: long\n" + _long_docstring(12))
        assert _hook().check(path) == [f"{path}:2: docstring of f is 12 lines (max 10)"]


class TestAFileItCannotParse:
    def test_it_says_so_rather_than_reporting_the_file_clean(self, tmp_path: Path, capsys):
        """Silence here would be an invisible pass once the gate is armed."""
        path = _write(tmp_path, "def f(:\n")
        assert _hook().check(path) == []
        assert "not parsed, so not checked" in capsys.readouterr().err


class TestWhatIsNotAComment:
    def test_hashes_inside_a_string_are_not_a_comment_block(self, tmp_path: Path):
        path = _write(tmp_path, 'S = """\n# one\n# two\n# three\n# four\n# five\n"""\n')
        assert _hook().check(path) == []

    def test_trailing_comments_do_not_form_a_block(self, tmp_path: Path):
        body = "".join(f"x{n} = {n}  # why\n" for n in range(6))
        assert _hook().check(_write(tmp_path, body)) == []


class TestTheCommandLine:
    def test_a_missing_path_is_a_message_and_not_a_traceback(self, tmp_path: Path, capsys):
        assert _hook().main([str(tmp_path / "nope.py")]) == 2
        assert "no such path" in capsys.readouterr().err

    def test_a_file_over_the_cap_exits_one(self, tmp_path: Path, capsys):
        path = _write(tmp_path, _long_docstring(12))
        assert _hook().main([str(path)]) == 1
        assert "docstring of f" in capsys.readouterr().out

    @pytest.mark.parametrize("body", ['"""Short."""\n', "x = 1\n"])
    def test_a_clean_file_exits_zero(self, tmp_path: Path, body: str):
        assert _hook().main([str(_write(tmp_path, body))]) == 0
