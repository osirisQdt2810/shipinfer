"""The offline tier must be the same run on a GPU host as on a runner with no driver."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


class TestTheOfflineTierHidesTheAccelerators:
    """``pytest`` with no device marker selected must not open a CUDA context (ADR-001)."""

    @pytest.mark.parametrize(
        ("expression", "requested"),
        [
            ("not gpu and not multigpu", False),  # the default in pyproject.toml
            ("scheduling and not gpu and not multigpu", False),
            ("", True),  # no expression selects everything, device tests included
            ("gpu", True),
            ("multigpu", True),
            ("gpu and slow", True),
            ("not gpu", True),  # still selects a multigpu-only test
            ("not multigpu", True),  # still selects a gpu-only test
            ("gpu or multigpu", True),
            ("not slow", True),  # selects every fast device test
            ("slow and not gpu and not multigpu", False),
            # a multigpu test that also carries `scheduling` is selected; identifiers other
            # than the device markers are free, and the predicate must say what *could* run
            ("(gpu or scheduling) and not gpu", True),
        ],
    )
    def test_the_expression_decides_whether_a_device_tier_is_wanted(
        self, tier_predicate, expression: str, requested: bool
    ) -> None:
        assert tier_predicate(expression) is requested

    def test_an_unparseable_expression_does_not_hide_anything(self, tier_predicate) -> None:
        # pytest rejects the expression itself; the predicate must not pre-empt that by
        # hiding a device from a run that may have meant to use one.
        assert tier_predicate("gpu and (") is True

    def test_this_run_sees_no_accelerator(self, request, tier_predicate) -> None:
        """The live assertion: the process this test runs in has no device.

        This test carries no device marker, so it is part of the offline tier; if that tier
        selected a device test as well (``-m "not gpu"`` on a multi-GPU box) the hook leaves
        the devices alone by design, and this assertion would be about a different run.
        """
        if tier_predicate(request.config.getoption("markexpr") or ""):
            pytest.skip("this run selected a device tier; the devices are meant to be visible")
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        assert os.environ.get("HIP_VISIBLE_DEVICES") == ""
        torch = pytest.importorskip("torch")
        assert torch.cuda.is_available() is False, "the offline tier opened a CUDA device"


class TestAskingTheDriverCannotAbortCollection:
    """`_device_count_or_zero` takes its probe as an argument for exactly this test."""

    def test_a_driver_that_answers_is_believed(self, device_count_or_zero) -> None:
        assert device_count_or_zero(lambda: 3) == 3

    def test_a_driver_that_raises_is_a_machine_with_none(self, device_count_or_zero) -> None:
        def broken():
            raise RuntimeError("CUDA driver initialization failed")

        with pytest.warns(UserWarning, match="could not ask the driver"):
            assert device_count_or_zero(broken) == 0

    def test_the_failure_is_kept_for_the_skip_reason(self, probe_device_count) -> None:
        def broken():
            raise RuntimeError("CUDA driver initialization failed")

        with pytest.warns(UserWarning):
            count, failure = probe_device_count(broken)
        assert count == 0
        assert failure is not None and "CUDA driver initialization failed" in failure
        assert probe_device_count(lambda: 4) == (4, None)


class TestEveryBenchmarkEntryPointGatesItself:
    """A `benchmarks/` module you can run with `python -m` is a process that may measure.

    `containment.py` is the enforcement point BECAUSE it runs inside that process -- CLAUDE.md
    says so, and calls the `PreToolUse` hook an advisory fast path. That is a claim about which
    files call it, so it is worth testing rather than repeating: three of the six did not, and
    `run_bench.py` was one -- the tier the headline >=5x number comes from, where the advisory
    hook was therefore the only guard. `link_probe.py` and `ipc_context_cost.py` were the
    others. Derived from the tree, so a fourth tier is covered without editing this.
    """

    #: `benchmarks/baseline` is the counting-simulation SUBMODULE -- someone else's tree, and
    #: full of `__main__` scripts. Excluded by name, because a glob that includes it passes on
    #: a machine where CI's un-checked-out submodule is absent and fails on this box.
    NOT_OURS = ("baseline", "tests")

    def _entry_points(self) -> list[Path]:
        root = Path(__file__).resolve().parents[1] / "benchmarks"
        return sorted(
            path
            for path in root.rglob("*.py")
            if not set(self.NOT_OURS) & set(path.parts)
            and 'if __name__ == "__main__"' in path.read_text(encoding="utf-8")
        )

    def test_there_are_some(self) -> None:
        """Without this, a glob that matched nothing would pass the assertion below."""
        assert len(self._entry_points()) >= 6

    def test_the_exclusion_is_the_submodule_and_nothing_of_ours(self) -> None:
        """The exclusion is the risky half of a glob, so it is asserted rather than trusted:
        every path it drops must be under `benchmarks/baseline` or a `tests` directory."""
        root = Path(__file__).resolve().parents[1] / "benchmarks"
        stray = [
            path
            for path in root.rglob("*.py")
            if set(self.NOT_OURS) & set(path.parts)
            and 'if __name__ == "__main__"' in path.read_text(encoding="utf-8")
            and "baseline" not in path.parts
            and "tests" not in path.parts
        ]
        assert not stray, f"the exclusion is dropping our own entry points: {stray}"

    def test_each_one_calls_the_gate(self) -> None:
        ungated = [
            path.name
            for path in self._entry_points()
            if "require_container" not in path.read_text(encoding="utf-8")
        ]
        assert not ungated, (
            f"{ungated} can be run with `python -m` and measure without asking "
            "`runtime.containment`. The hook over command text is the advisory half; the "
            "gate in the process is the enforcement point, and a benchmark that skips it "
            "has neither."
        )

    #: Callees that constitute doing the work. A gate reached after one of these has already
    #: let a model load or a device be touched.
    #: `main` dispatches through `MEASURE[system]`, so the callee unparses to a subscript
    #: rather than to a name -- matching only `measure_*` missed the one call that matters.
    WORK = (
        "MEASURE[",
        "sweep_system",
        "measure_baseline",
        "measure_shipinfer",
        "run_shipinfer",
    )

    def test_the_gate_is_called_before_the_work(self) -> None:
        """The call has to precede the measurement, compared as CALLS in the entry function
        rather than as text positions in the file.

        Two drafts got that wrong in the two ways this repository spent a day on: "within N
        characters of the top" pushed the gate ahead of `run_bench`'s own argv validation, so
        a malformed command line raised a container error instead of its usage code; and
        comparing the offset of `torch.cuda` tripped on an import above `main` -- a name
        appearing is not a call happening, the hook's own defect.
        """
        for path in self._entry_points():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            entry = next(
                (
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name in {"main", "_child_main"}
                ),
                None,
            )
            assert entry is not None, f"{path.name} has an entry function this cannot find"
            gate = [
                node.lineno
                for node in ast.walk(entry)
                if isinstance(node, ast.Call) and "require_container" in ast.unparse(node.func)
            ]
            assert gate, f"{path.name}'s entry function does not call the gate"
            work = [
                node.lineno
                for node in ast.walk(entry)
                if isinstance(node, ast.Call)
                and any(w in ast.unparse(node.func) for w in self.WORK)
            ]
            assert not [w for w in work if w < min(gate)], (
                f"{path.name} calls the gate at line {min(gate)}, after work at "
                f"{sorted(w for w in work if w < min(gate))}; an in-process gate has to run "
                "before a device is touched"
            )


class TestEveryScriptThatBuildsForADeviceGatesItself:
    """The same rule one directory over, because the rule is one rule.

    CLAUDE.md's list of what must run in a container is "the GPU test tiers, every benchmark,
    `shipinfer bench|serve`, and any engine build". #182 gave the benchmarks their gate; the
    engine build was the last entry without one, and its own docstring already claimed "this
    refuses to run without a device". Derived from the imports rather than from a list of file
    names, so the next script that reaches for TensorRT is covered by this and not by memory.
    """

    #: A top-level import of any of these means the script cannot do its job without a device,
    #: whatever it calls itself. `shipvision` is the fused-kernel library, so it counts.
    DEVICE_STACKS = frozenset({"torch", "tensorrt", "onnxruntime", "pycuda", "shipvision"})

    def _device_scripts(self) -> list[Path]:
        root = Path(__file__).resolve().parents[1] / "scripts"
        found = []
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if 'if __name__ == "__main__"' not in source:
                continue
            if self.DEVICE_STACKS & self._imports(ast.parse(source)):
                found.append(path)
        return found

    @staticmethod
    def _imports(tree: ast.AST) -> set[str]:
        """Every module imported anywhere, function bodies included -- a device import is
        usually lazy precisely because the script has to start without one."""
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        return names

    def test_the_sweep_finds_the_engine_build(self) -> None:
        """Names the one member there is today, so a predicate that silently matches nothing
        fails here instead of passing the assertion below."""
        assert [p.name for p in self._device_scripts()] == ["build_engines.py"]

    def test_each_one_calls_the_gate(self) -> None:
        ungated = [
            path.name
            for path in self._device_scripts()
            if "require_container" not in path.read_text(encoding="utf-8")
        ]
        assert not ungated, (
            f"{ungated} reaches for a device stack and can be run directly, without asking "
            "`runtime.containment`. Host nvcc here is 11.5 against a 12.6 driver, so what a "
            "host build produces is the wrong artefact rather than a slow one."
        )


class TestEveryDeviceTierTestSitsUnderTheGate:
    """A `mark.gpu` module in a directory whose conftest chain lacks the guards runs the
    device tier with no container gate and no no-device skip.

    `benchmarks/tests/test_crowd_yield.py` was the first such module, and the invocation its
    own test plan documented — `pytest -m gpu benchmarks/tests/test_crowd_yield.py` — loaded
    neither guard, because `tests/conftest.py` is directory-scoped and `tests/` was not among
    the collected paths.
    """

    #: The hooks that carry `containment.require_container` and the no-device skip.
    GUARDS = ("pytest_configure", "pytest_collection_modifyitems")

    def _gpu_modules(self) -> list[Path]:
        root = Path(__file__).resolve().parents[1]
        return [
            path
            for path in root.rglob("test_*.py")
            if "3rdparty" not in path.parts
            and ".venv" not in path.parts
            and "mark.gpu" in path.read_text(encoding="utf-8")
        ]

    def test_the_benchmarks_conftest_re_exports_the_very_same_hooks(self) -> None:
        """Re-exported, not reimplemented, so the two directories cannot drift apart.

        Identity and not equality: a copy that merely behaves the same today is the thing this
        asserts against.
        """
        import benchmarks.tests.conftest as bench_conftest
        import tests.conftest as root_conftest

        for guard in self.GUARDS:
            assert getattr(bench_conftest, guard) is getattr(root_conftest, guard), (
                f"benchmarks/tests/conftest.py provides its own {guard} instead of the one in "
                f"tests/conftest.py; the two gates will drift"
            )

    def test_there_is_at_least_one_to_check(self) -> None:
        """Without this the next test passes on an empty sweep."""
        assert self._gpu_modules(), "found no `mark.gpu` module; the sweep is broken"

    def test_each_one_has_an_ancestor_conftest_providing_the_guards(self) -> None:
        root = Path(__file__).resolve().parents[1]
        unguarded = []
        for module in self._gpu_modules():
            guarded = False
            for parent in [module.parent, *module.parent.parents]:
                conftest = parent / "conftest.py"
                if conftest.is_file():
                    text = conftest.read_text(encoding="utf-8")
                    if all(guard in text for guard in self.GUARDS):
                        guarded = True
                        break
                if parent == root:
                    break
            if not guarded:
                unguarded.append(str(module.relative_to(root)))

        assert not unguarded, (
            f"these `mark.gpu` modules have no ancestor conftest providing {self.GUARDS}, so "
            f"a targeted `pytest -m gpu <path>` on them runs the device tier ungated: "
            f"{unguarded}"
        )
