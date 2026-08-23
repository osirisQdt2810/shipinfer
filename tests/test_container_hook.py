"""The container-rule hook is itself covered, because it is a guard.

A guard that silently stops guarding is worse than no guard: it converts "we
checked" into "we believed we checked".  The two failure directions are not
symmetric and both are tested here.

- A **false negative** (a host test run allowed through) is the failure the hook
  exists to prevent, and it is silent -- the number gets measured on the host
  and reported as if it came from the container.
- A **false positive** (`grep pytest` refused) is loud but corrosive: it is what
  makes an operator switch the hook off, after which it prevents nothing.

So `TestRefuses` and `TestAllows` carry roughly equal weight, and
`TestTheGuardCanFail` asserts the guard is capable of firing at all -- without
it, a hook that returned "allow" unconditionally would pass every other test in
this file.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / "scripts" / "hooks" / "require_container.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("require_container", HOOK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


hook = _load_hook()


def refused(command: str, cwd: Path | None = None) -> str | None:
    """The reason the hook gives, or None when it allows the command."""
    return hook.verdict(command, str(cwd or REPO_ROOT))


class TestRefuses:
    """Commands that would run the suite or touch a device on the host."""

    @pytest.mark.parametrize(
        "command",
        [
            "pytest tests/",
            "pytest -m gpu",
            "python -m pytest tests/scheduling -q",
            "python3 -m pytest -m multigpu 2>&1 | tail -20",
            "./scripts/run_tests.sh",
            "cd /somewhere && .venv/bin/python -m pytest -m gpu",
            "shipinfer bench person_embedder --cameras 50",
            "shipinfer serve --port 8000",
            "trtexec --onnx=models/yolo26n.onnx",
            "python benchmarks/run_bench.py --cameras 50",
            "python benchmarks/compare_baseline.py",
            "ls && pytest tests/ && echo done",
            "nice -n 5 pytest tests/",
            'bash -c "pytest tests/ -q"',
        ],
    )
    def test_a_host_test_or_benchmark_run_is_refused(self, command: str) -> None:
        assert refused(command) is not None, command

    @pytest.mark.parametrize(
        "command",
        [
            'python -c "import torch; print(torch.cuda.device_count())"',
            'python -c "import tensorrt as trt; print(trt.__version__)"',
            "CUDA_VISIBLE_DEVICES=2,3 python probe.py --cuda",
            'python -c "import torch; torch.cuda.synchronize()"',
        ],
    )
    def test_inline_python_reaching_for_a_device_is_refused(self, command: str) -> None:
        assert refused(command) is not None, command

    def test_a_semicolon_inside_quotes_does_not_hide_the_device_call(self) -> None:
        """The regression that mattered: splitting the command on `;` as a
        character cut `python -c "import torch; print(...)"` in half, and the
        device call ended up in a fragment that no longer parsed -- so it was
        skipped and the command sailed through."""
        command = 'python -c "import torch; print(torch.cuda.device_count())"'
        assert refused(command) is not None

    def test_a_wrappers_own_operand_does_not_shadow_the_real_command(
        self, tmp_path: Path
    ) -> None:
        """`timeout 900 python probe.py` used to resolve its executable to
        `900`, so the python behind it was never examined.  This is the exact
        shape of the one process observed holding a CUDA context on this box."""
        script = tmp_path / "opaque_name.py"
        script.write_text("import torch\nprint(torch.cuda.device_count())\n")
        assert refused(f"timeout 900 ./.venv/bin/python {script}") is not None

    def test_a_script_is_judged_by_its_contents_not_its_name(self, tmp_path: Path) -> None:
        script = tmp_path / "totally_innocent.py"
        script.write_text("import tensorrt\n")
        assert refused(f"python {script}") is not None


class TestAllows:
    """Commands that must not be refused, or the hook gets switched off."""

    @pytest.mark.parametrize(
        "command",
        [
            "grep -rn pytest tests/ | head",
            "rg 'torch.cuda' src/ -l",
            "cat tests/test_architecture.py",
            "sed -n 1,20p scripts/hooks/require_container.py",
            'echo "pytest is how we run tests"',
            'git commit -m "test: add pytest case for cuda graphs"',
            "git status",
            "gh pr list --limit 5",
            "ruff check src/ && black --check src/",
            "shipinfer repo ls",
            "nvidia-smi --query-compute-apps=pid,used_memory --format=csv",
        ],
    )
    def test_reading_about_a_test_is_not_running_one(self, command: str) -> None:
        assert refused(command) is None, command

    @pytest.mark.parametrize(
        "command",
        [
            "deploy/rootless/test.sh tests/scheduling -q",
            "deploy/rootless/prove.sh",
            "docker run --rm --device nvidia.com/gpu=all img pytest tests/",
            "podman run --rm img pytest tests/",
            "make shell",
        ],
    )
    def test_an_already_containerised_command_passes(self, command: str) -> None:
        assert refused(command) is None, command

    def test_the_documented_override_works(self) -> None:
        assert refused("SHIPINFER_ALLOW_HOST_RUN=1 pytest tests/ -q") is None

    def test_a_version_check_is_inspection_not_measurement(self) -> None:
        """Deliberately allowed.  Refusing this would add friction with no
        integrity gain, and friction is what gets a hook disabled."""
        assert refused('python -c "import torch; print(torch.__version__)"') is None

    def test_the_layer_checker_is_not_mistaken_for_device_work(self) -> None:
        """`scripts/hooks/check_layers.py` contains the string `torch` because
        its job is to forbid that import.  Matching on the bare word rather than
        on an import statement would block the layer checker itself."""
        assert refused("python3 scripts/hooks/check_layers.py") is None

    def test_a_script_it_cannot_read_is_allowed(self) -> None:
        """Fail open on missing files: blocking work over an unreadable path
        would make the hook a source of mystery failures."""
        assert refused("python3 /nonexistent/mystery.py") is None

    def test_an_unparseable_command_is_allowed(self) -> None:
        """A half-typed command with an unbalanced quote must not become a
        refusal -- that would be the hook blocking on its own parse bug."""
        assert refused('python -c "unterminated') is None


class TestTheGuardCanFail:
    """Without this, a hook that always allowed would pass everything above."""

    def test_the_hook_denies_through_its_real_entry_point(self) -> None:
        """Exercises the process boundary the harness actually uses: stdin JSON
        in, a `permissionDecision` out.

        `SHIPINFER_IN_CONTAINER=0` forces the host view. Without it this test
        would pass vacuously in the very place the rule requires tests to run:
        inside the container the hook is deliberately a no-op, so it emits
        nothing and there is no decision to assert on."""
        payload = json.dumps(
            {
                "tool_name": "Bash",
                "cwd": str(REPO_ROOT),
                "tool_input": {"command": "pytest tests/"},
            }
        )
        result = subprocess.run(
            [sys.executable, str(HOOK)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "SHIPINFER_IN_CONTAINER": "0"},
        )
        assert result.returncode == 0, result.stderr
        decision = json.loads(result.stdout)["hookSpecificOutput"]
        assert decision["permissionDecision"] == "deny"
        assert "deploy/rootless/test.sh" in decision["permissionDecisionReason"]

    def test_an_allowed_command_produces_no_output(self) -> None:
        payload = json.dumps(
            {
                "tool_name": "Bash",
                "cwd": str(REPO_ROOT),
                "tool_input": {"command": "git status"},
            }
        )
        result = subprocess.run(
            [sys.executable, str(HOOK)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "SHIPINFER_IN_CONTAINER": "0"},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_the_hook_is_wired_into_project_settings(self) -> None:
        """A tested hook that nothing invokes enforces nothing."""
        settings = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text())
        commands = [
            h["command"]
            for entry in settings.get("hooks", {}).get("PreToolUse", [])
            if entry.get("matcher") == "Bash"
            for h in entry.get("hooks", [])
            if h.get("type") == "command"
        ]
        assert any("require_container.py" in c for c in commands), commands

    def test_inside_a_container_the_hook_stands_down(self) -> None:
        """The hook gates the host, not the container. If it refused inside one
        too, `deploy/rootless/test.sh` could never run the suite."""
        payload = json.dumps(
            {
                "tool_name": "Bash",
                "cwd": str(REPO_ROOT),
                "tool_input": {"command": "pytest tests/"},
            }
        )
        result = subprocess.run(
            [sys.executable, str(HOOK)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "SHIPINFER_IN_CONTAINER": "1"},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""
