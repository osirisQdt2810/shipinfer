#!/usr/bin/env python3
"""Refuse a Napoleon field entry whose continuation sits at the item indent.

Inside ``Args:``/``Raises:``/``Attributes:``, a line at the *item* indent starts a new field
and must carry a ``name: description`` colon. A line at that indent with no colon is a
continuation that lost four spaces, and Sphinx renders it as a field named after its first
word -- so a re-wrap that produced ``still`` documents a parameter called "still".

Invisible to black (it does not touch prose), to ruff (E501 is off) and to the caps in
``check_docs.py`` (these lines are short). This is the only thing that looks.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = ("src/shipinfer", "scripts", "tests", "benchmarks")

SECTIONS = (
    "Args:",
    "Arguments:",
    "Keyword Args:",
    "Raises:",
    "Attributes:",
    "Returns:",
    "Yields:",
    "Note:",
    "Example:",
    "Examples:",
)
#: Sections whose body is prose, not a field list, so there is nothing to orphan.
PROSE = ("Returns:", "Yields:", "Note:", "Example:", "Examples:")
#: A field entry: one or more names, then a colon.
FIELD = re.compile(r"^[\w*.\[\]`~/,\- ]+:")


def _docstrings(tree: ast.AST) -> list[ast.Constant]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            found.append(body[0])
    return found


def check(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (UnicodeDecodeError, SyntaxError) as exc:
        print(f"{path}: not parsed, so not checked: {exc}", file=sys.stderr)
        return []
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    bad = []
    for doc in _docstrings(tree):
        section: str | None = None
        item_indent: int | None = None
        for offset, line in enumerate(doc.value.value.split("\n")):
            text = line.strip()
            if not text:
                continue
            indent = len(line) - len(line.lstrip())
            if text in SECTIONS:
                section, item_indent = text, None
                continue
            if section is None:
                continue
            if item_indent is None:
                # Napoleon reads a section as a field list only when its first entry has the
                # `name: description` shape; a section of free prose has no fields to orphan.
                if not FIELD.match(text):
                    section = None
                    continue
                item_indent = indent
                continue
            if indent < item_indent:
                section, item_indent = None, None
                continue
            if indent == item_indent and section not in PROSE and not FIELD.match(text):
                bad.append(
                    f"{rel}:{doc.lineno + offset}: {section} continuation sits at the item "
                    f"indent, so Sphinx reads {text.split()[0]!r} as a field name"
                )
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
            f"\n{len(bad)} orphaned continuation line(s). Indent them four more spaces.",
            file=sys.stderr,
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
