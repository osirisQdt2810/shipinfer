"""The layering rule, asserted rather than trusted.

The single most damaging mistake available here is a heavy import creeping into the pure
core: the day ``shipinfer.core`` imports torch, the offline suite silently starts needing a
GPU and nobody notices until CI moves to a cheaper runner. That is a slow, expensive failure
to diagnose and a two-line test to prevent.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "shipinfer"

PURE_LAYERS = ("core", "scheduling", "repository")
FORBIDDEN_IN_PURE = {"torch", "tensorrt", "onnxruntime", "cuda", "cv2", "fastapi", "uvicorn"}


def _modules_imported_by(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
    return found


class TestPureLayersAreAcceleratorFree:
    """The pure layers name no accelerator runtime, which is what keeps the offline tier offline."""

    @pytest.mark.parametrize("layer", PURE_LAYERS)
    def test_pure_layers_import_no_accelerator_runtime(self, layer: str) -> None:
        offenders: list[str] = []
        for path in (SRC / layer).rglob("*.py"):
            for module in _modules_imported_by(path):
                if module.split(".")[0] in FORBIDDEN_IN_PURE:
                    offenders.append(f"{path.relative_to(SRC)} imports {module}")
        assert not offenders, "pure layers must stay accelerator-free:\n" + "\n".join(offenders)


class TestImportsGoOneWay:
    """core is the bottom of the stack and scheduling sits directly on it, never above."""

    def test_core_imports_no_other_shipinfer_layer(self) -> None:
        """``core`` is the bottom of the stack; nothing in the project may be below it."""
        offenders: list[str] = []
        for path in (SRC / "core").rglob("*.py"):
            for module in _modules_imported_by(path):
                if module.startswith("shipinfer.") and not module.startswith("shipinfer.core"):
                    offenders.append(f"{path.relative_to(SRC)} imports {module}")
        assert not offenders, "core must not import upward:\n" + "\n".join(offenders)

    def test_scheduling_only_imports_core(self) -> None:
        allowed = {"shipinfer.core", "shipinfer.scheduling"}
        offenders: list[str] = []
        for path in (SRC / "scheduling").rglob("*.py"):
            for module in _modules_imported_by(path):
                if module.startswith("shipinfer.") and not any(
                    module.startswith(prefix) for prefix in allowed
                ):
                    offenders.append(f"{path.relative_to(SRC)} imports {module}")
        assert not offenders, "scheduling may only import core:\n" + "\n".join(offenders)


class TestEnforcementAgrees:
    """The pre-commit hook and this suite check the same rule, so neither can drift alone."""

    def test_layer_check_hook_passes(self) -> None:
        """The pre-commit hook and this test must agree; run the hook itself."""
        hook = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "check_layers.py"
        result = subprocess.run([sys.executable, str(hook)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


class TestImportIsCheap:
    """import shipinfer must not drag in a backend, so the CLI stays usable on a bare host."""

    def test_importing_shipinfer_does_not_import_backends(self) -> None:
        """``import shipinfer`` must stay cheap.

        The top-level package resolves ``InferenceServer`` lazily precisely so a CLI that only
        lists a repository does not pay for the backend registry and everything behind it.
        """
        code = (
            "import sys, shipinfer; "
            "assert 'tensorrt' not in sys.modules; "
            "assert 'shipinfer.server' not in sys.modules, sorted(m for m in sys.modules if m.startswith('shipinfer'))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
