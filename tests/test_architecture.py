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

PURE_LAYERS = ("core", "scheduling", "repository", "topology")
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

    def test_topology_only_imports_core(self) -> None:
        """The chain sits directly on ``core``, and that is what makes it loadable anywhere.

        Not on ``scheduling`` and not on the engine: everything an element needs from the
        surrounding runner arrives through the ``ElementContext`` handed to ``open()``
        (arch.md §1). The inversion is the reason a chain file can be validated on a laptop —
        ``Topology.from_spec`` instantiates every element to read its caps.
        """
        allowed = {"shipinfer.core", "shipinfer.topology"}
        offenders: list[str] = []
        for path in (SRC / "topology").rglob("*.py"):
            for module in _modules_imported_by(path):
                if module.startswith("shipinfer.") and not any(
                    module.startswith(prefix) for prefix in allowed
                ):
                    offenders.append(f"{path.relative_to(SRC)} imports {module}")
        assert not offenders, "topology may only import core:\n" + "\n".join(offenders)


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
        ``shipinfer.engine`` is the module named here rather than ``shipinfer.server``: the
        pool moved, and asserting the absence of a package that is now a thin shim would pass
        while the whole engine loaded behind it.
        """
        code = (
            "import sys, shipinfer; "
            "assert 'tensorrt' not in sys.modules; "
            "assert 'shipinfer.engine' not in sys.modules, sorted(m for m in sys.modules if m.startswith('shipinfer'))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    def test_importing_topology_pulls_in_no_accelerator(self) -> None:
        """``import shipinfer.topology`` must load no accelerator, no decoder, no server.

        A *runtime* check next to the static one above, and it is the load-bearing half. The
        static rule ("topology imports only core") is a rule about module scope, and the
        element implementations that arrive in later phases will legitimately need
        GStreamer, TensorRT and the engine — inside ``_do_open``. This test is what keeps
        that promise honest when the day comes to relax the static rule: importing the
        package, and therefore every registered element class, still costs nothing.
        """
        code = (
            "import sys, shipinfer.topology as t; "
            "assert t.ELEMENTS, 'nothing registered'; "
            "heavy = [m for m in ('torch', 'tensorrt', 'cv2', 'gi', 'shipinfer.engine', "
            "'shipinfer.runtime', 'shipinfer.scheduling') if m in sys.modules]; "
            "assert not heavy, heavy"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


class TestTheServerShimIsTheSameObjects:
    """`shipinfer.server` re-exports `shipinfer.engine`, and re-export means *identity*.

    The pool moved to `engine/` and `server/__init__.py` is a shim until the rest of
    `server/` has moved too (arch.md §9). The failure worth preventing is not an import
    error — that is loud and immediate — but a shim that grew a second definition of one of
    these names. `isinstance(cache, shipinfer.server.ResponseCache)` would then be False for
    a cache the engine built, from a call site that never mentioned either package, and no
    test that only checks importability would notice.

    Silent by design, too: `pyproject.toml` turns a `DeprecationWarning` from `shipinfer.*`
    into an error in the offline tier, so a shim that warned would fail the suite instead of
    nudging anybody. This class is what stands in for the warning.
    """

    def test_every_name_the_shim_exports_is_the_engine_s_own_object(self) -> None:
        import shipinfer.engine as engine
        import shipinfer.server as server

        assert server.__all__, "the shim exports nothing"
        mismatched = [
            name
            for name in server.__all__
            if getattr(server, name) is not getattr(engine, name, object())
        ]
        assert not mismatched, "shipinfer.server re-exports a *copy* of: " + ", ".join(
            sorted(mismatched)
        )

    def test_the_inference_server_class_is_one_class(self) -> None:
        """Spelled out separately because it is the name every caller outside this tree holds."""
        import shipinfer
        import shipinfer.engine
        import shipinfer.server

        assert shipinfer.server.InferenceServer is shipinfer.engine.InferenceServer
        assert shipinfer.InferenceServer is shipinfer.engine.InferenceServer
        assert shipinfer.engine.Engine is shipinfer.engine.InferenceServer

    def test_the_shim_does_not_export_a_submodule_named_engine(self) -> None:
        """`from shipinfer.server import engine` used to reach the pool module.

        Re-exporting the `shipinfer.engine` *package* under that attribute would make
        `shipinfer.server.engine.InferenceServer` keep working by accident, and every
        remaining caller would then be invisible to the grep that has to find them. The
        spelling is `from shipinfer.engine import pool`.
        """
        import shipinfer.engine.pool  # populates the attribute if it can
        import shipinfer.server

        assert not hasattr(shipinfer.server, "engine")


class TestTheLayerCheckerCoversSharedModules:
    """`envs.py` sits above every layer, and the checker used to skip it entirely.

    `layer_of` returns None for any top-level file, and `check` returned an empty list on
    None — so one `import torch` in `envs.py` would have put torch behind
    `shipinfer.scheduling` with the hook still exiting 0. Every layer from `core` up may
    import `envs`, so it inherits `core`'s ban.
    """

    def _checker(self):
        import importlib.util

        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "check_layers", root / "scripts" / "hooks" / "check_layers.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, root

    def test_envs_is_checked_rather_than_skipped(self) -> None:
        checker, root = self._checker()
        assert checker.check(root / "src" / "shipinfer" / "envs.py") == []

    def test_a_device_import_in_envs_would_be_caught(self, tmp_path: Path) -> None:
        """Without this the previous test passes on a checker that inspects nothing."""
        checker, _ = self._checker()
        offender = tmp_path / "envs.py"
        offender.write_text("import torch\n")

        problems = checker.check(offender)

        assert problems, "the checker skipped a top-level module again"
        assert "torch" in problems[0]

    def test_an_unlisted_top_level_module_is_still_skipped(self, tmp_path: Path) -> None:
        """The rule is a named allowance, not a blanket one: `__main__.py` may import
        whatever the CLI needs."""
        checker, _ = self._checker()
        other = tmp_path / "__main__.py"
        other.write_text("import torch\n")

        assert checker.check(other) == []
