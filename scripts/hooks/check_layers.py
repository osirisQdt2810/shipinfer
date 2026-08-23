#!/usr/bin/env python3
"""Enforce the one-way import rule (ADR-001) at commit time.

The single most damaging mistake available in this codebase is a heavy import creeping into
the pure core: the day ``shipinfer.core`` imports torch, the offline suite silently starts
needing a GPU and nobody notices until CI is moved to a cheaper runner.

This is an AST check, not a grep: ``# import torch`` in a docstring should not fail a
commit, and ``from x import y`` must be checked as carefully as ``import x``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "shipinfer"

#: Third-party modules a layer may never import.
FORBIDDEN_EXTERNAL: dict[str, set[str]] = {
    "core": {
        "torch",
        "tensorrt",
        "onnxruntime",
        "cuda",
        "cv2",
        "fastapi",
        "uvicorn",
        "confluent_kafka",
    },
    "scheduling": {"torch", "tensorrt", "onnxruntime", "cuda", "cv2", "fastapi", "uvicorn"},
    "repository": {"torch", "tensorrt", "onnxruntime", "cuda", "fastapi", "uvicorn"},
    "runtime": {"fastapi", "uvicorn", "confluent_kafka"},
    "backends": {"fastapi", "uvicorn", "confluent_kafka"},
}

#: Top-level modules that are not layers and may therefore be imported from anywhere.
#: ``_C`` is the compiled extension; ``envs`` is the single place the process environment is
#: read, which is only useful if every layer can reach it.
NON_LAYER_MODULES = frozenset({"_C", "envs"})

#: Which sibling packages a layer may import. A layer may always import itself.
ALLOWED_INTERNAL: dict[str, set[str]] = {
    "core": set(),
    "repository": {"core"},
    "scheduling": {"core"},
    "runtime": {"core"},
    "backends": {"core", "repository", "runtime"},
    "server": {"core", "repository", "runtime", "backends", "scheduling"},
    "pipeline": {"core", "repository", "runtime", "backends", "scheduling", "server"},
    # ingest publishes frames straight into the fair, bounded queue from `scheduling`
    # rather than owning a buffer of its own — that shared evict-oldest buffer is the
    # inherited bug ADR-005 exists to remove, so the dependency is deliberate.
    "ingest": {"core", "runtime", "scheduling"},
    "observability": {"core"},
}


def layer_of(path: Path) -> str | None:
    try:
        parts = path.relative_to(SRC).parts
    except ValueError:
        return None
    return parts[0] if len(parts) > 1 else None


def imported_modules(tree: ast.AST) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module, node.lineno))
    return found


def check(path: Path) -> list[str]:
    layer = layer_of(path)
    if layer is None:
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"]

    problems: list[str] = []
    forbidden = FORBIDDEN_EXTERNAL.get(layer, set())
    allowed = ALLOWED_INTERNAL.get(layer)
    rel = path.relative_to(ROOT)

    for module, lineno in imported_modules(tree):
        root = module.split(".")[0]
        if root in forbidden:
            problems.append(
                f"{rel}:{lineno}: layer {layer!r} must not import {module!r} "
                f"(ADR-001: it stays importable without a GPU)"
            )
        if allowed is not None and module.startswith("shipinfer."):
            target = module.split(".")[1]
            if target != layer and target not in allowed and target not in NON_LAYER_MODULES:
                problems.append(
                    f"{rel}:{lineno}: layer {layer!r} must not import shipinfer.{target} "
                    f"(allowed: {sorted(allowed) or 'nothing'})"
                )
    return problems


def main() -> int:
    problems = [p for path in sorted(SRC.rglob("*.py")) for p in check(path)]
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} layering violation(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
