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
ROOT_PACKAGE = "shipinfer"
SRC = ROOT / "src" / ROOT_PACKAGE

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
    # `topology` is the Element ABC, the caps and the chain loader (arch.md §1). It is pure
    # for a reason that bites daily: `Topology.from_spec` instantiates every element in the
    # chain to read its declared caps, so a chain file has to be *validatable* on a host
    # with no accelerator, no GStreamer and no engine. Implementations honour that by
    # importing their runtime inside `_do_open`, never at module scope — see
    # `src/shipinfer/topology/elements/__init__.py`. If a future implementation genuinely
    # cannot (a module whose import needs `pyds`), it registers lazily; and if that ever
    # stops being enough, the relaxation is a separate `topology.elements` layer with its
    # own row here, never a hole in this one.
    "topology": {
        "torch",
        "tensorrt",
        "onnxruntime",
        "cuda",
        "cv2",
        "fastapi",
        "uvicorn",
        "confluent_kafka",
    },
    "repository": {"torch", "tensorrt", "onnxruntime", "cuda", "fastapi", "uvicorn"},
    "runtime": {"fastapi", "uvicorn", "confluent_kafka"},
    "backends": {"fastapi", "uvicorn", "confluent_kafka"},
    # The engine is the model pool (arch.md §6) and the KServe endpoint is a *side-door* into
    # it, not part of it: the HTTP layer lives in `api/`. Naming fastapi here would put a web
    # framework behind `shipinfer.engine`, which is the import an in-process caller — a runner
    # walking a chain — must not pay for.
    "engine": {"fastapi", "uvicorn", "confluent_kafka"},
    # `api` is the KServe surface (arch.md §6): the engine's side-door for callers who bring
    # their own tensors. It is the one layer whose row *omits* fastapi and uvicorn — every
    # other row below and above it names them, so a web framework can enter this codebase at
    # exactly one seam and `import shipinfer.<anything else>` never pays for one. It keeps
    # `confluent_kafka` from the engine's row: publishing results is a `pipeline` sink's job,
    # and an HTTP handler that reached for a broker would be doing dispatch.
    "api": {"confluent_kafka"},
    # `server` is what the split has not carried off yet — the shard launcher and the
    # topology-as-placement classes (both leave in A2 PR-4/PR-6). It held the only fastapi
    # import in the tree until `server/api/` became `api/`; this row is what stops it growing
    # a second one on the way out.
    "server": {"fastapi", "uvicorn", "confluent_kafka"},
    # The three layers left without a ban, banned: an output sink pushes, a camera actor
    # pulls, and the CLI calls `serve_http` rather than building an app itself. Rows rather
    # than trust — "fastapi enters at `api/` and nowhere else" is worth being a statement this
    # script can check, and with these every layer on disk has a row.
    "pipeline": {"fastapi", "uvicorn"},
    "ingest": {"fastapi", "uvicorn"},
    "cli": {"fastapi", "uvicorn"},
}

#: Top-level modules that are not layers, and may therefore be imported by any layer
#: *except* ``core``. ``_C`` is the compiled extension; ``envs`` is the single place the
#: process environment is read, which is only useful if every layer above ``core`` can reach
#: it. ``core`` is deliberately excluded: nothing in this project sits below it, so it may
#: not import even these, and that is what keeps it importable in isolation (ADR-001).
NON_LAYER_MODULES = frozenset({"_C", "envs"})

#: Which sibling packages a layer may import. A layer may always import itself.
ALLOWED_INTERNAL: dict[str, set[str]] = {
    "core": set(),
    "repository": {"core"},
    "scheduling": {"core"},
    "topology": {"core"},
    "runtime": {"core"},
    "backends": {"core", "repository", "runtime"},
    "engine": {"core", "repository", "runtime", "backends", "scheduling"},
    # `api` sits on the engine and on nothing else: `/v2/models/{name}` answers with an
    # entry's versions, platform and tensor specs, but it reads them through the engine
    # object (`server.repository`), not by importing `repository` — so the grant is exactly
    # what the code does today. It may not import `scheduling`, `runtime` or `backends`: an
    # HTTP handler that decided what to batch or where to run would be the dispatch layer
    # wearing a router (arch.md §6). It grows `repository` if the metadata route ever
    # annotates `RepositoryEntry` directly, and `launch` in phase B when `/streams` needs the
    # shards — each as a diff with an argument, not a standing permission.
    "api": {"core", "engine"},
    # What is left of `server` is the launcher and the topology-as-placement classes; they sit
    # *on* the engine and may not import `runtime` or `backends` directly any more
    # (transitively they still reach both through the engine — this check is about direct
    # imports, which is what keeps the remaining callers greppable). `repository` stays because `server/topology/deepstream.py`
    # reads the model repository to plan its shards — a config question, answered without a
    # device. `pipeline` names `engine` rather than `server` for the same reason the row above
    # exists: it wants the pool, and the two stopped being the same package.
    "server": {"core", "repository", "scheduling", "engine"},
    "pipeline": {"core", "repository", "runtime", "backends", "scheduling", "engine"},
    # `ingest` does NOT depend on `scheduling`: it publishes into the `FrameSink` protocol
    # it owns, and `pipeline` supplies the queue-backed implementation. Mapping a frame onto
    # a request is dispatch policy, and it belongs next to the code that undoes the mapping.
    "ingest": {"core", "runtime"},
    "observability": {"core"},
}


def layer_of(path: Path) -> str | None:
    try:
        parts = path.relative_to(SRC).parts
    except ValueError:
        return None
    return parts[0] if len(parts) > 1 else None


def is_submodule(name: str) -> bool:
    """Whether ``shipinfer.<name>`` is a module rather than a re-exported attribute."""
    return (SRC / f"{name}.py").is_file() or (SRC / name / "__init__.py").is_file()


def imported_modules(tree: ast.AST) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == ROOT_PACKAGE:
                # `from shipinfer import envs` reaches the same module as
                # `import shipinfer.envs`, so it must be checked the same way. Without this
                # the two spellings have different rules and the lax one wins by accident.
                # `from shipinfer import __version__` names an attribute, not a module, so
                # the filesystem is what tells the two apart.
                found.extend(
                    (f"{ROOT_PACKAGE}.{alias.name}", node.lineno)
                    for alias in node.names
                    if is_submodule(alias.name)
                )
            else:
                found.append((node.module, node.lineno))
    return found


#: Top-level modules that every layer may import, and the strictest rule that must hold
#: for each. `envs` is importable from `core` upwards, so it inherits `core`'s ban: one
#: `import torch` there would put torch behind `shipinfer.scheduling` while this checker
#: still exited 0, because a top-level file has no layer and was skipped outright.
NON_LAYER_RULES: dict[str, str] = {"envs": "core"}


def check(path: Path) -> list[str]:
    layer = layer_of(path)
    if layer is None:
        name = path.stem
        if name not in NON_LAYER_RULES:
            return []
        # Checked against the strictest layer that may import it, because that is the
        # constraint it actually has to satisfy.
        layer = NON_LAYER_RULES[name]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"]

    problems: list[str] = []
    forbidden = FORBIDDEN_EXTERNAL.get(layer, set())
    allowed = ALLOWED_INTERNAL.get(layer)
    # `relative_to` raises for anything outside the repository, which made the checker
    # crash rather than report when handed a path from elsewhere — including a test's own
    # fixture. A report is not worth failing over.
    try:
        rel: Path | str = path.relative_to(ROOT)
    except ValueError:
        rel = path

    for module, lineno in imported_modules(tree):
        root = module.split(".")[0]
        if root in forbidden:
            problems.append(
                f"{rel}:{lineno}: layer {layer!r} must not import {module!r} "
                f"(ADR-001: it stays importable without a GPU)"
            )
        exempt = frozenset() if layer == "core" else NON_LAYER_MODULES
        if allowed is not None and module.startswith("shipinfer."):
            target = module.split(".")[1]
            if target != layer and target not in allowed and target not in exempt:
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
