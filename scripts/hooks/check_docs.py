#!/usr/bin/env python3
"""Cap how long documentation may be (V145).

A docstring earns its lines by saying what the signature cannot: a constraint, a failure
mode, a decision and its reason. Longer reasoning belongs in an ADR or a PR body, once.
Put ``# doc: long`` on the line above a symbol (or above a comment block) to exempt it,
with the reason on that line.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_MAX, SYMBOL_MAX, COMMENT_MAX = 15, 10, 4
EXEMPT = "# doc: long"
DEFAULT_ROOTS = ("src/shipinfer", "scripts", "tests", "benchmarks")


def _exempted(lines: list[str], lineno: int) -> bool:
    """True when the line above ``lineno`` (1-based) opts out."""
    for i in range(lineno - 2, max(lineno - 12, -1), -1):
        text = lines[i].strip()
        if text.startswith(EXEMPT):
            return True
        if text and not text.startswith(("@", "#")):
            return False
    return False


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
    run, start = 0, 0
    for number, line in enumerate([*lines, ""], start=1):
        if line.strip().startswith("#"):
            if not run:
                start = number
            run += 1
        else:
            if run > COMMENT_MAX and not _exempted(lines, start):
                bad.append(f"{rel}:{start}: comment block is {run} lines (max {COMMENT_MAX})")
            run = 0
    return bad


def main(argv: list[str]) -> int:
    given = [Path(a) for a in argv] or [ROOT / root for root in DEFAULT_ROOTS]
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
