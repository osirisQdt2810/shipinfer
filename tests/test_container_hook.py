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
            "pytest -m gpu",
            "pytest -m multigpu",
            "python3 -m pytest -m multigpu 2>&1 | tail -20",
            "cd /somewhere && .venv/bin/python -m pytest -m gpu",
            "shipinfer bench person_embedder --cameras 50",
            "shipinfer serve --port 8000",
            "csrc/build/bench --plan model.plan --cameras 50",
            "./csrc/build/test_dataplane",
            "trtexec --onnx=models/yolo26n.onnx",
            "python benchmarks/run_bench.py --cameras 50",
            "python benchmarks/compare_baseline.py",
            "ls && pytest -m gpu && echo done",
            "nice -n 5 pytest -m gpu",
            'bash -c "pytest -m gpu -q"',
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

    def test_a_later_line_cannot_hide_behind_an_earlier_one(self) -> None:
        """The silent bypass. ``shlex`` treats a newline as whitespace, so lexing a
        multi-line command whole made the second line's command an *argument* of the
        first line's — and a guard that only inspects executables then never saw it."""
        assert refused("echo starting\npytest -m gpu tests/\necho done") is not None

    def test_every_line_of_a_script_is_inspected(self) -> None:
        assert refused("git status\ncd /tmp\nshipinfer bench person_embedder") is not None


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
            "docker run --rm --device nvidia.com/gpu=all img pytest -m gpu",
            "podman run --rm img pytest -m gpu",
            "make shell",
        ],
    )
    def test_an_already_containerised_command_passes(self, command: str) -> None:
        assert refused(command) is None, command

    @pytest.mark.parametrize(
        "command",
        [
            "csrc/build/test_scheduling",
            "./csrc/build/test_engine",
            "csrc/build/test_containment",
        ],
    )
    def test_the_cuda_free_cpp_tests_are_the_offline_tier(self, command: str) -> None:
        """They link no accelerator library (`ldd` shows neither libcuda nor libnvinfer), so
        they are to the C++ plane what a bare `pytest` is to the Python one."""
        assert refused(command) is None

    def test_the_offline_runner_script_is_allowed(self) -> None:
        """`run_tests.sh` adds `-m "not gpu and not multigpu"` and exports empty
        `CUDA_VISIBLE_DEVICES`, so it is *strictly more* offline than the bare `pytest` this
        hook allows. Refusing it while permitting `pytest` taught the developer to reach for
        the override — which is how the rule was lost the first time."""
        assert refused("./scripts/run_tests.sh") is None
        assert refused("scripts/run_tests.sh -q") is None

    def test_the_documented_override_works(self) -> None:
        assert refused("SHIPINFER_ALLOW_HOST_RUN=1 pytest -m gpu -q") is None

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

    def test_a_later_lines_argument_is_not_charged_to_an_earlier_command(self) -> None:
        """The false positive the same bug caused, and the one that would have got this
        hook switched off: a `wc -l` of the benchmark sources on a later line was read as
        an argument to an earlier `python3`, and refused as "invokes a benchmark runner"."""
        command = (
            'gh pr view 1 --json labels | python3 -c "import sys; print(sys.stdin.read())"\n'
            "wc -l benchmarks/harness/*.py benchmarks/compare_baseline.py"
        )
        assert refused(command) is None

    def test_a_heredoc_body_is_data_not_commands(self) -> None:
        """`python3 - <<'PY'` is one python invocation. Reading its script's lines as
        commands would refuse a block for merely containing the word pytest."""
        command = "python3 - <<'PY'\nprint('pytest tests/ would be a lie here')\nPY"
        assert refused(command) is None


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
                "tool_input": {"command": "pytest -m gpu"},
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

    def test_inside_a_container_the_hook_stands_down(self) -> None:
        """The hook gates the host, not the container. If it refused inside one too,
        `deploy/rootless/test.sh` could never run the suite."""
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

    # Deliberately no assertion that `.claude/settings.json` wires the hook up. The
    # offline tier must not depend on the agent harness's own configuration: a review
    # environment that rewrites that file made this the tier's only failure, and the
    # remaining tests in this module already cover the behaviour that matters.


class TestTheBypassesFoundInReview:
    """Six ways past the guard, all reported by review and all verified against it.

    Each is the same species of mistake in a different place: the hook looked at *text*
    where it should have looked at what a segment actually executes. They are kept as one
    class because a future change that reintroduces any of them almost certainly
    reintroduces the rest.
    """

    IMPORT_LINE = "import " + "torch"
    DEVICE_CALL = "torch" + ".cuda.device_count()"

    def test_a_substitution_in_command_position_is_refused(self) -> None:
        """`$(which pytest)` resolves to pytest at runtime and to an opaque token here."""
        assert refused("$(which pytest) -m gpu tests/") is not None

    def test_a_variable_in_command_position_is_refused(self) -> None:
        assert refused("PYTEST=pytest; $PYTEST -m gpu") is not None

    def test_the_attached_m_form_is_refused(self) -> None:
        """Only the detached `-m pytest` was inspected."""
        assert refused("python -mpytest -m gpu tests/") is not None

    def test_a_module_path_is_refused_not_just_a_file_path(self) -> None:
        """The blocked list held `run_bench.py`, so the module form walked through."""
        assert refused("python -m benchmarks.run_bench --seconds 70") is not None

    def test_the_allowlist_is_not_matched_on_a_substring(self) -> None:
        """`make test` appearing anywhere in the text used to allow the whole command —
        the same mention-versus-invocation error, on the permissive side."""
        assert refused('echo "make test" >/dev/null; pytest -m gpu tests/') is not None

    def test_an_executed_heredoc_is_the_program(self) -> None:
        command = f"python3 - <<'XX'\n{self.IMPORT_LINE}\nprint(1)\nXX"
        assert refused(command) is not None

    def test_a_shell_heredoc_read_from_stdin_is_the_program(self) -> None:
        """`bash -s` reads its script from the heredoc and takes no `-`."""
        assert refused("bash -s <<'XX'\npytest -m gpu tests/\nXX") is not None


class TestTheBypassFixesDidNotCostTooMuch:
    """The other half of every one of those fixes: what must still pass."""

    IMPORT_LINE = "import " + "torch"
    DEVICE_CALL = "torch" + ".cuda.device_count()"

    def test_a_substitution_as_an_argument_is_allowed(self) -> None:
        """Refusing every `$(...)` would refuse the container scripts themselves."""
        assert refused("docker run --rm --user $(id -u) img pytest -m gpu") is None

    def test_writing_a_file_that_mentions_a_device_is_allowed(self) -> None:
        """A heredoc redirected into a file is data. Refusing it would make the hook
        unusable for writing any test that imports torch."""
        assert refused(f"cat > t.py <<'XX'\n{self.IMPORT_LINE}\nXX") is None

    def test_an_executed_heredoc_merely_quoting_a_device_call_is_allowed(self) -> None:
        """Matching the bare word refused a heredoc whose body quoted a device call as
        test data — and, on its first outing, refused the edit that was fixing that."""
        assert refused(f"python3 - <<'XX'\nprint('{self.DEVICE_CALL}')\nXX") is None

    def test_a_backslash_continuation_is_one_command(self) -> None:
        """Line-by-line lexing closed the multi-line bypass and opened this: a `\\`
        continuation is the same command wrapped, not a new one, and splitting there
        refused a real `docker run` whose argv sat on the last line."""
        command = (
            "docker run --rm --pid=host \\\n"
            '  -v "$PWD:/work:ro" -w /work img \\\n'
            "  python3 scripts/build_engines.py --check"
        )
        assert refused(command) is None

    def test_a_real_newline_still_starts_a_new_command(self) -> None:
        """The other half: folding continuations must not fold genuine line breaks."""
        assert refused("echo start\npytest -m gpu tests/") is not None

    def test_a_redirection_target_is_not_a_command(self) -> None:
        """`>` used to be treated as a command separator, so `cat > "$S/out.md"` produced a
        segment whose executable was `$S/out.md` — which the variable-in-command-position
        rule then refused. Redirecting into a path held in a variable is ordinary."""
        assert refused("cat > \"$OUT/reply.md\" <<'EOF'\nhello\nEOF") is None

    def test_a_redirection_does_not_hide_the_command_after_it(self) -> None:
        assert refused("echo hi >/dev/null; pytest -m gpu tests/") is not None


class TestTheOfflineTierRunsAnywhere:
    """ADR-001, and the reason the hook was refusing more than the rule says.

    The offline tier must pass on a machine with no driver: that is what CI does on a plain
    runner, and it is the promise that makes the pure layers verifiable. The hook refused it
    anyway, which contradicted the rule rather than enforcing it — and blocked verifying the
    in-process gate that now covers the part that matters.
    """

    def test_a_plain_offline_run_is_allowed(self) -> None:
        assert refused("pytest tests/") is None

    def test_an_explicit_offline_selection_is_allowed(self) -> None:
        """`-m "not gpu"` mentions the marker and selects the opposite. A substring test
        would refuse the default run — mention versus selection, again."""
        assert refused("pytest -m 'not gpu' tests/") is None

    def test_the_offline_tier_through_the_module_form_is_allowed(self) -> None:
        assert refused("python -m pytest tests/core -q") is None

    def test_the_device_tier_is_still_refused_both_ways(self) -> None:
        assert refused("pytest -m gpu") is not None
        assert refused("python -m pytest -m multigpu") is not None
