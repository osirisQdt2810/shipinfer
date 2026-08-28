#!/usr/bin/env python3
"""Cap how long documentation may be (V145).

A docstring earns its lines by saying what the signature cannot: a constraint, a failure
mode, a decision and its reason. Longer reasoning belongs in an ADR or a PR body, once.
Put ``# doc: long`` on the line above a symbol (or above a comment block) to exempt it,
with the reason on that line.
"""

from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_MAX, SYMBOL_MAX, COMMENT_MAX = 15, 10, 4
EXEMPT = "# doc: long"
DEFAULT_ROOTS = ("src/shipinfer", "scripts", "tests", "benchmarks")


def _exempted(lines: list[str], lineno: int) -> bool:
    """True when the first non-blank line above ``lineno`` (1-based) is a marker with a reason.

    The *first* line, not any line above: a marker further up heads the comment block between
    them, and reading it as this symbol's would silently exempt a docstring nobody marked. The
    reason is required — an exemption that need not be argued is one nobody argues.
    """
    for i in range(lineno - 2, -1, -1):
        text = lines[i].strip()
        if not text:
            continue
        return text.startswith(EXEMPT) and bool(text[len(EXEMPT) :].strip())
    return False


def _comment_lines(source: str) -> list[int]:
    """Line numbers of standalone comments — tokenised, so a ``#`` in a string is not one.

    Trailing comments (``x = 1  # why``) are excluded: they annotate the line they sit on and
    do not form a block, and counting them made a run of annotated assignments look like prose.
    """
    try:
        return [
            token.start[0]
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT and not token.line[: token.start[1]].strip()
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []


def _blocks(numbers: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs of comment lines as ``(first line, length)``."""
    runs: list[tuple[int, int]] = []
    for number in numbers:
        if runs and number == runs[-1][0] + runs[-1][1]:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((number, 1))
    return runs


def check(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        print(f"{path}: not utf-8, so not checked: {exc}", file=sys.stderr)
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        print(f"{path}: not parsed, so not checked: {exc}", file=sys.stderr)
        return []
    lines = source.splitlines()
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    bad = []
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        text = ast.get_docstring(node, clean=False)
        if text is None:
            continue
        count = text.count("\n") + 1
        cap = MODULE_MAX if isinstance(node, ast.Module) else SYMBOL_MAX
        # The module's own line is its docstring's. A decorated symbol has two places a
        # marker can honestly sit -- above the decorators, or between them and the `def` --
        # and both are accepted, because a false positive is how a hook gets deleted.
        if isinstance(node, ast.Module):
            anchors = [node.body[0].lineno]
        else:
            anchors = [
                node.lineno,
                min((d.lineno for d in node.decorator_list), default=node.lineno),
            ]
        if count > cap and not any(_exempted(lines, anchor) for anchor in anchors):
            name = "<module>" if isinstance(node, ast.Module) else node.name
            bad.append(f"{rel}:{anchors[0]}: docstring of {name} is {count} lines (max {cap})")
    for start, run in _blocks(_comment_lines(source)):
        # The marker is itself a comment, so it joins the run it exempts: a block that opts
        # out says so on its own first line, and neither the marker nor the block is counted.
        marker = lines[start - 1].strip()
        if marker.startswith(EXEMPT):
            if marker[len(EXEMPT) :].strip():
                continue
            bad.append(f"{rel}:{start}: `{EXEMPT}` needs a reason")
            continue
        if run > COMMENT_MAX and not _exempted(lines, start):
            bad.append(f"{rel}:{start}: comment block is {run} lines (max {COMMENT_MAX})")
    return bad


def main(argv: list[str]) -> int:
    given = [Path(a) for a in argv] or [ROOT / root for root in DEFAULT_ROOTS]
    if missing := [p for p in given if not p.exists()]:
        print(f"no such path: {', '.join(str(p) for p in missing)}", file=sys.stderr)
        return 2
    if wrong := [p for p in given if p.is_file() and p.suffix != ".py"]:
        # Silent under a pre-commit `files:` filter is fine; silent when a human names a file
        # is how you conclude a file is clean because the tool never looked at it.
        print(f"not python: {', '.join(str(p) for p in wrong)}", file=sys.stderr)
        return 2
    paths = [f for p in given for f in (sorted(p.rglob("*.py")) if p.is_dir() else [p])]
    bad = [line for path in paths if path.suffix == ".py" for line in check(path)]
    for line in bad:
        print(line)
    if bad:
        print(
            f"\n{len(bad)} over the cap. Cut it, or mark it `{EXEMPT} <reason>`.",
            file=sys.stderr,
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
