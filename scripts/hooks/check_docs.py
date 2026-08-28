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
    """True when the marker sits directly above ``lineno`` (1-based).

    Directly, not anywhere above: a marker separated from the symbol by other comments heads
    the *block* those comments form, and reading it as the symbol's would silently exempt a
    docstring nobody marked — the quiet half of the bug this rule exists to prevent.
    """
    for i in range(lineno - 2, max(lineno - 12, -1), -1):
        text = lines[i].strip()
        if text.startswith(EXEMPT):
            return True
        if text and not text.startswith("@"):
            # Any other comment in between means the marker heads that block, not this symbol.
            return False
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
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
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
        # The module's own line is its docstring's, so a `# doc: long` comment above it works.
        line = node.body[0].lineno if isinstance(node, ast.Module) else node.lineno
        if count > cap and not _exempted(lines, line):
            name = "<module>" if isinstance(node, ast.Module) else node.name
            bad.append(f"{rel}:{line}: docstring of {name} is {count} lines (max {cap})")
    for start, run in _blocks(_comment_lines(source)):
        # The marker is itself a comment, so it joins the run it exempts: a block that opts
        # out says so on its own first line, and neither the marker nor the block is counted.
        if lines[start - 1].strip().startswith(EXEMPT):
            continue
        if run > COMMENT_MAX and not _exempted(lines, start):
            bad.append(f"{rel}:{start}: comment block is {run} lines (max {COMMENT_MAX})")
    return bad


def main(argv: list[str]) -> int:
    given = [Path(a) for a in argv] or [ROOT / root for root in DEFAULT_ROOTS]
    if missing := [p for p in given if not p.exists()]:
        print(f"no such path: {', '.join(str(p) for p in missing)}", file=sys.stderr)
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
