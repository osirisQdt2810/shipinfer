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


def _checker():
    """The hook, imported as a module, so its tables are read rather than restated.

    Loaded by path because `scripts/` is not a package: a table asserted here and defined
    there could otherwise drift, and the drift would be a rule silently switched off.
    """
    import importlib.util

    hook = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "check_layers.py"
    spec = importlib.util.spec_from_file_location("check_layers", hook)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestEnforcementAgrees:
    """The pre-commit hook and this suite check the same rule, so neither can drift alone."""

    def test_layer_check_hook_passes(self) -> None:
        """The pre-commit hook and this test must agree; run the hook itself."""
        hook = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "check_layers.py"
        result = subprocess.run([sys.executable, str(hook)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def test_every_package_on_disk_has_a_forbidden_externals_row(self) -> None:
        """A layer with no row is a layer where nothing is forbidden.

        `FORBIDDEN_EXTERNAL` is a dict, and an absent key means "anything goes" — which is how
        four layers had no fastapi ban until A2 PR-2 added their rows. The hook's comment says
        "every layer on disk has a row"; this makes that a check rather than a sentence, so
        the next top-level package gets a failing test instead of silence.
        """
        module = _checker()
        packages = {p.name for p in SRC.iterdir() if (p / "__init__.py").is_file()}
        missing = packages - set(module.FORBIDDEN_EXTERNAL)
        assert not missing, f"layers with no FORBIDDEN_EXTERNAL row: {sorted(missing)}"

    def test_every_package_on_disk_has_an_allowed_internals_row(self) -> None:
        """The twin, and the one that was actually off.

        `check` reads `ALLOWED_INTERNAL.get(layer)` and skips the internal check entirely when
        it is `None` — so `cli`, which had no row, could have imported anything at all and the
        hook would still have exited 0. A row that grants everything is a decision; a missing
        row is a rule nobody turned on. The reverse check is below: nothing may import `cli`.
        """
        module = _checker()
        packages = {p.name for p in SRC.iterdir() if (p / "__init__.py").is_file()}

        missing = packages - set(module.ALLOWED_INTERNAL)
        assert not missing, f"layers with no ALLOWED_INTERNAL row: {sorted(missing)}"

    def test_no_row_names_a_package_that_is_not_on_disk(self) -> None:
        """A row for a package that does not exist is a rule that cannot be violated.

        `observability` had one for both tables long after the package went; it read as a
        constraint and constrained nothing. Grants are checked here rather than only keys,
        because a stale *grant* is the one that would quietly allow an import later.
        """
        module = _checker()
        packages = {p.name for p in SRC.iterdir() if (p / "__init__.py").is_file()}

        keys = set(module.ALLOWED_INTERNAL) | set(module.FORBIDDEN_EXTERNAL)
        granted = {name for grants in module.ALLOWED_INTERNAL.values() for name in grants}
        stale = (keys | granted) - packages - set(module.NON_LAYER_MODULES)
        assert not stale, f"rows naming packages that are not on disk: {sorted(stale)}"

    def test_nothing_below_the_command_line_may_import_it(self) -> None:
        """`cli` is the composition root, and the direction is what makes that harmless.

        It may import every layer; no layer may import it. A library whose scheduler reached
        for a typer command could not be embedded in anything that is not this CLI.
        """
        module = _checker()

        importers = [
            layer
            for layer, allowed in module.ALLOWED_INTERNAL.items()
            if layer != "cli" and "cli" in allowed
        ]
        assert not importers, f"layers allowed to import the CLI: {importers}"


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

    def test_importing_runners_pulls_in_neither_the_engine_nor_torch(self) -> None:
        """``import shipinfer.runners`` must cost no accelerator and no model pool.

        The static rule is the narrow one — ``runners`` may import ``core``, ``topology``
        and ``scheduling``, and neither ``engine`` nor ``runtime`` — because the ``inprocess``
        runner reaches the model pool through the ``ModelResolver`` it is *handed* rather than
        by importing anything. That is what lets a chain be started with mock elements on a
        host with no driver, and what lets ``tests/runners/`` run in the offline tier at all.
        This runtime check is the load-bearing half: the static rule is about module scope,
        and the ``fleet`` runner will legitimately bind its shard's device through ``runtime``
        one day. This is the test that notices the day that changes.

        ``shipinfer.ingest`` is in the list and is the reason it is load-bearing *today*. The
        layering hook grants ``runners -> ingest`` — the in-process runner owns the camera
        actors (arch.md §5①) — and an AST checker cannot tell a module-scope import from one
        inside a method, so the whole cost of that grant is checked here and nowhere else.
        Importing ``shipinfer.ingest`` reaches ``sources/gstreamer.py`` and through it
        ``shipinfer.runtime`` and torch, so a single import moved to module scope in
        ``runners/`` would put a CUDA-capable stack behind ``import shipinfer.runners`` and
        take the offline tier's no-driver promise with it.
        """
        code = (
            "import sys, shipinfer.runners as r; "
            "assert 'inprocess' in r.RUNNERS, r.RUNNERS.names(); "
            "heavy = [m for m in ('torch', 'tensorrt', 'cv2', 'fastapi', 'shipinfer.engine', "
            "'shipinfer.runtime', 'shipinfer.backends', 'shipinfer.ingest') if m in sys.modules]; "
            "assert not heavy, heavy"
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
            "'shipinfer.api', 'shipinfer.runtime', 'shipinfer.scheduling') if m in sys.modules]; "
            "assert not heavy, heavy"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


class TestTheWebFrameworkEntersAtOneSeam:
    """FastAPI is `api/`'s alone, and `api/` does not pay for it at import time either.

    Two rules that look like one and fail differently. `check_layers.py` says the *string*
    ``fastapi`` appears in no layer but `api` — a static rule, checked there. These are the
    runtime halves, and they are the ones that catch the mistake nobody makes on purpose: an
    import that arrives transitively.

    Why it matters is arch.md §6. The engine is the model pool and the KServe endpoint is a
    side-door *into* it; an in-process caller — a runner walking a chain, a benchmark, a
    notebook — reaches the pool without ever wanting a web framework in the process. A
    starlette import chain behind ``shipinfer.engine`` costs that caller ~100 ms and a
    dependency it cannot uninstall, and no test that only imports ``shipinfer`` would see it.
    """

    def test_importing_the_engine_does_not_import_fastapi(self) -> None:
        code = (
            "import sys, shipinfer.engine; "
            "assert 'fastapi' not in sys.modules; "
            "assert 'starlette' not in sys.modules; "
            "assert 'uvicorn' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    def test_importing_the_api_does_not_import_fastapi_either(self) -> None:
        """`api/` owns fastapi, and still imports it *inside* the functions that need it.

        Deliberate, not accidental: FastAPI and uvicorn are the ``server`` extra, so the
        package has to be importable — and ``shipinfer serve`` without ``--http`` has to
        run — on a host that never installed them. Module-scope imports would turn that into
        an ``ImportError`` at start-up instead of the typed refusal
        ``tests/api/test_optional_dependency.py`` pins.
        """
        code = (
            "import sys, shipinfer.api; "
            "assert callable(shipinfer.api.create_app); "
            "assert callable(shipinfer.api.serve_http); "
            "eager = [m for m in ('fastapi', 'starlette', 'uvicorn') if m in sys.modules]; "
            "assert not eager, eager"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


class TestTheLauncherNeedsNoDeviceAndNoEngine:
    """`import shipinfer.launch` costs a CUDA context on nothing and a model pool on nothing.

    Two separate promises, both load-bearing, and neither visible to a static rule.

    **No torch.** The reason a shard is a subprocess rather than a thread is that
    ``CUDA_VISIBLE_DEVICES`` must be in the child's environment *before* its interpreter
    imports torch — a module-scope import two packages deep otherwise wins the race
    (``launch/supervisor.py``). The parent is the process that sets that variable, and a
    parent that imported torch itself would hold a CUDA context of ~220-480 MiB on every
    device it can see, on the one process in the deployment that needs no device at all. The
    hook checks the string; this checks the import graph, which is where the mistake actually
    arrives.

    **No engine, and no runners.** A launcher spawns the thing that owns models; it does not
    own them. If ``shipinfer.launch`` pulled in the pool, the parent would pay for the backend
    registry and there would be no reason left for the shard to be a separate process.
    ``shipinfer.runners`` does not exist yet (A2 PR-3/PR-6) — it is asserted absent anyway,
    because the direction of that dependency is the whole point: the fleet runner will import
    ``launch``, never the reverse, and a cycle between them is what would make the gRPC
    client's home ambiguous.
    """

    def test_importing_launch_pulls_in_no_accelerator_and_no_engine(self) -> None:
        code = (
            "import sys, shipinfer.launch as launch; "
            "assert callable(launch.forward_signals); "
            "assert launch.Fleet is not None; "
            "heavy = [m for m in ('torch', 'tensorrt', 'fastapi', 'starlette', 'uvicorn', "
            "'shipinfer.engine', 'shipinfer.api', 'shipinfer.runners', 'shipinfer.runtime', "
            "'shipinfer.backends') if m in sys.modules]; "
            "assert not heavy, heavy"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


class TestTheControlPlaneEntersAtTwoSeams:
    """grpcio and protobuf are an optional extra, and no import may make them mandatory.

    Exactly the argument `TestTheWebFrameworkEntersAtOneSeam` makes for FastAPI, one layer
    over. The static half is in `check_layers.py`: `grpc` and `google` are named in every
    `FORBIDDEN_EXTERNAL` row but `launch`'s, and `grpc` additionally in `runners`'. These are
    the runtime halves, and they catch what a string check cannot — an import that arrives
    transitively, or one at module scope inside the two layers that *are* allowed to name it.

    That second case is the interesting one. `launch/client.py` and `runners/service.py` may
    import grpc; they must not do it at module scope, because `import shipinfer.launch` runs
    in the launcher — the one CPU-only process in the deployment, on a host that may well
    have installed neither package — and `import shipinfer.runners` runs on a laptop driving
    the in-process runner. The refusal, when a call is finally made, is
    `tests/launch/test_client_without_grpcio.py`.
    """

    def test_importing_launch_loads_no_grpc_and_no_protobuf(self) -> None:
        code = (
            "import sys, shipinfer.launch as launch; "
            "assert callable(launch.ShardClient); "
            "assert launch.CameraSpec('c', 'rtsp://x').camera_id == 'c'; "
            "eager = [m for m in sys.modules if m == 'grpc' or m.startswith('grpc.') "
            "or m == 'google.protobuf' or m.startswith('google.protobuf.') "
            "or m.startswith('shipinfer.launch.proto.')]; "
            "assert not eager, eager"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    def test_importing_runners_loads_no_grpc_and_no_protobuf(self) -> None:
        """Including `runners.service`, which is the module that holds the servicer."""
        code = (
            "import sys, shipinfer.runners, shipinfer.runners.service as service; "
            "assert callable(service.serve_shard); "
            "eager = [m for m in sys.modules if m == 'grpc' or m.startswith('grpc.') "
            "or m == 'google.protobuf' or m.startswith('google.protobuf.') "
            "or m.startswith('shipinfer.launch.proto.')]; "
            "assert not eager, eager"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    def test_the_layer_rows_say_where_the_control_plane_may_be_named(self) -> None:
        """`launch` may name both; `runners` may name grpc; nobody else may name either.

        Asserted against the checker's own tables rather than restated, so a row deleted in
        `check_layers.py` fails here instead of quietly switching the rule off.
        """
        module = _checker()

        may_name_grpc = {
            layer for layer, banned in module.FORBIDDEN_EXTERNAL.items() if "grpc" not in banned
        }
        may_name_protobuf = {
            layer
            for layer, banned in module.FORBIDDEN_EXTERNAL.items()
            if "google" not in banned
        }

        assert may_name_grpc == {"launch", "runners"}
        assert may_name_protobuf == {"launch"}

    def test_the_launcher_may_not_import_the_runner_it_launches(self) -> None:
        """The direction is what decides where the client and the servicer live.

        `runners` -> `launch` for the control vocabulary; never the reverse. A launcher that
        imported the executor would pay for it in the parent process, and a cycle between the
        two would make the generated stubs' home ambiguous (arch.md §9).
        """
        module = _checker()

        assert "launch" in module.ALLOWED_INTERNAL["runners"]
        assert "runners" not in module.ALLOWED_INTERNAL["launch"]


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
