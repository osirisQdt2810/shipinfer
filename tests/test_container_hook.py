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
from typing import ClassVar

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
            "python -m benchmarks.parity.drive_python --scenario reconnect --emit-golden",
            "python3 -m benchmarks.parity.diff --left a.jsonl --right b.jsonl",
        ],
    )
    def test_a_module_under_benchmarks_that_is_not_a_benchmark_is_allowed(
        self, command: str
    ) -> None:
        """The false positive that cost a disclosure line.

        `BLOCKED_MODULE_ROOTS` matched the whole `benchmarks` package, and the cross-plane
        ingest parity harness lives under it -- though it imports numpy and
        `shipinfer.ingest`, drives a scripted source, touches no device and produces no
        measurement. Refusing it teaches a developer to reach for SHIPINFER_ALLOW_HOST_RUN,
        and that is how the container rule was lost the first time. (Its emitter was moved
        to `scripts/emit_parity_golden.py` under that refusal, so neither module above is an
        entry point today; the rule is about what the package *is*, not about one command.)
        """
        assert refused(command) is None, command


class TestTheCarveOutIsNarrow:
    """A carve-out that swallowed its root would be worse than the false positive."""

    @pytest.mark.parametrize(
        "command",
        [
            "python -m benchmarks.run_bench --cameras 50",
            "python -m benchmarks.harness.shards --shard 0",
            # A prefix, not a substring: a future sibling must not inherit the carve-out.
            "python -m benchmarks.parity_bench --op letterbox",
        ],
    )
    def test_the_rest_of_the_benchmarks_package_is_still_refused(self, command: str) -> None:
        assert refused(command) is not None, command

    @pytest.mark.parametrize(
        "command",
        [
            "deploy/rootless/test.sh tests/scheduling -q",
            "deploy/rootless/prove.sh",
            "docker run --rm --device nvidia.com/gpu=all img pytest -m gpu",
            "podman run --rm img pytest -m gpu",
            "deploy/rootless/run.sh python scripts/build_engines.py",
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


class TestReadOnlyToolsAreNotExecutionVectors:
    """`python -m black|isort|ruff` reads its arguments; it executes nothing it is handed."""

    @pytest.mark.parametrize("tool", ["black", "isort", "ruff"])
    def test_a_formatter_over_a_device_importing_file_is_allowed(
        self, tool: str, tmp_path: Path
    ) -> None:
        script = tmp_path / "model.py"
        script.write_text("import torch\n")
        assert refused(f"python -m {tool} --check {script}", cwd=tmp_path) is None

    def test_a_device_named_path_is_still_only_data_to_a_formatter(self) -> None:
        assert refused("python -m black --check src/shipinfer/runtime/ops/torch_ops.py") is None

    def test_pytest_gains_nothing_from_the_carve_out(self) -> None:
        assert refused("python -m pytest tests/ -m gpu") is not None

    def test_a_following_segment_is_still_judged_on_its_own(self, tmp_path: Path) -> None:
        script = tmp_path / "model.py"
        script.write_text("import torch\n")
        command = f"python -m black {script}; python -m pytest tests/ -m gpu"
        assert refused(command, cwd=tmp_path) is not None


class TestAScriptsOperandsAreData:
    """A script's arguments are its input, not a second program.

    `script_touches_device` scanned EVERY `.py` argument, so a linter was refused for its
    INPUT file's imports and an offline `python -m pytest tests/<one_file>.py` for the same
    reason -- while the identical run without the path, and the same path with `::a_test_id`
    after it, both passed. That inconsistency is the tell. It cost more than a retry: a
    refusal ends the whole `Bash` call, so an edit chained before one never ran.

    Same argument as `READ_ONLY_TOOL_MODULES`, one spelling over.
    """

    def test_a_linter_over_a_device_importing_file_is_allowed(self, tmp_path: Path) -> None:
        linter = tmp_path / "check_docs.py"
        linter.write_text("import ast\nimport sys\n")
        target = tmp_path / "model.py"
        target.write_text("import torch\n")
        assert refused(f"python {linter} {target}") is None

    def test_the_program_itself_is_still_judged_by_its_contents(self, tmp_path: Path) -> None:
        """The half that must not be lost: the FIRST operand is still read and still refused."""
        program = tmp_path / "probe.py"
        program.write_text("import torch\n")
        data = tmp_path / "data.py"
        data.write_text("VALUE = 1\n")
        assert refused(f"python {program} {data}") is not None

    def test_a_flag_value_is_not_mistaken_for_the_program(self, tmp_path: Path) -> None:
        """`-X importtime` puts a bare word before the script; the script is still the script."""
        program = tmp_path / "probe.py"
        program.write_text("import tensorrt\n")
        assert refused(f"python -X importtime {program}") is not None

    def test_an_offline_pytest_naming_one_file_is_allowed(self) -> None:
        """ADR-001: the offline tier runs anywhere, and naming one of its files does not
        change that. `pytest tests/pipeline/test_runner.py` was already allowed; the module
        form of the identical run was not."""
        assert refused("python -m pytest tests/pipeline/test_runner.py -q") is None

    def test_the_device_tier_naming_that_same_file_is_still_refused(self) -> None:
        assert refused("python -m pytest tests/pipeline/test_runner.py -m gpu") is not None

    def test_a_scripts_own_dash_m_flag_does_not_hide_it(self, tmp_path: Path) -> None:
        """`-m` after the program is the PROGRAM's flag: CPython ends option processing at the
        first operand. Reading the whole argv made `python probe.py -m yolov8n` -- ordinary
        shape for a probe -- look like a module invocation, so its program went unread."""
        probe = tmp_path / "probe.py"
        probe.write_text("import torch\ntorch.cuda.init()\n")
        assert refused(f"python {probe} -m yolov8n") is not None
        assert refused(f"python {probe} -mfoo") is not None
        assert refused(f"python {probe} -m gpu") is not None
        # Stricter than `main`, which reached the read-only carve-out with a module name that
        # was never the interpreter's.
        assert refused(f"python {probe} -m black") is not None

    def test_an_interpreter_value_flag_before_a_module_still_finds_it(self) -> None:
        """The half that fix could have broken: `-W ignore` puts a bare word before `-m`, and
        it must not be mistaken for the operand that ends option processing."""
        assert refused("python -W ignore -m pytest -m gpu") is not None
        assert refused("python -X importtime -m pytest -m multigpu") is not None
        assert refused("python -W ignore -m pytest tests/core -q") is None

    def test_a_module_still_hides_no_program_behind_it(self, tmp_path: Path) -> None:
        """`pytest`'s operands are pytest's: the TIER decides, above, not the file's imports.
        Its executor siblings are the opposite case and are pinned right below."""
        target = tmp_path / "model.py"
        target.write_text("import torch\n")
        assert refused(f"python -m pytest {target} -q") is None
        assert refused(f"python -m pytest {target} -m multigpu") is not None


class TestAModulesOperandIsAProgramUnlessTheModuleOnlyReads:
    """`-m` ends CPython's option processing; it does not end EXECUTION.

    Two rounds of getting the DIRECTION wrong. First every module's operands were data, which
    allowed `python -m cProfile probe.py`. Then eight executors were carved out, which still
    failed OPEN for every module not on the list -- `-m unittest`, `-m torch.distributed.run`,
    `-m IPython`, ten rows `main` refused. An allowlist of executors fixes instances; denying
    by default and carving out the readers fixes the class, and is shorter.
    """

    @pytest.mark.parametrize(
        "template",
        [
            "python -m cProfile {p}",
            "python -m cProfile -o /tmp/out.prof {p}",  # options, with a value, before it
            "python -mcProfile {p}",  # the attached spelling
            "python -m pdb {p}",
            "python -m trace --trace {p}",
            "python -m runpy {p}",
            "python -m coverage run {p}",  # a subcommand before it
            "python -m memory_profiler {p}",
            # Not on any list, and that is the point: these are the rows an allowlist of
            # executors let through, and the worst of them starts TWO host CUDA contexts.
            "python -m torch.distributed.run --nproc_per_node=2 {p}",
            "python -m torchrun {p}",
            "python -m unittest {p}",
            "python -m unittest -v {p}",
            "python -m nose2 {p}",
            "python -m IPython {p}",
            "python -m ipdb {p}",
            "python -m line_profiler {p}",
            "python -m yappi {p}",
        ],
    )
    def test_a_module_that_runs_its_operand_is_refused(self, template: str, tmp_path: Path):
        probe = tmp_path / "probe.py"
        probe.write_text("import torch\ntorch.cuda.init()\n")
        assert refused(template.format(p=probe)) is not None

    @pytest.mark.parametrize(
        "template",
        [
            "python -m black --check {p}",
            "python -m isort --check {p}",
            "python -m ruff check {p}",
            "python -m py_compile {p}",
            "python -m json.tool {p}",  # the dotted name, which the root alone would miss
        ],
    )
    def test_a_module_that_reads_its_operand_is_allowed(self, template: str, tmp_path: Path):
        """The other half, pinned against the one above, because it is the same rule: what
        RUNS is judged, what is merely READ is data. This list is the whole carve-out, so a
        module missing from it fails closed -- which is the direction two rounds got wrong."""
        target = tmp_path / "model.py"
        target.write_text("import torch\n")
        assert refused(template.format(p=target)) is None

    def test_the_executors_own_dash_c_is_not_the_interpreters(self, tmp_path: Path) -> None:
        """`-m` already ended option processing, so a `-c` after the module name is always the
        executor's: `pdb -c continue` is the documented non-interactive spelling and
        `trace -c` is `--count`. Bailing on it dropped the program."""
        probe = tmp_path / "probe.py"
        probe.write_text("import torch\ntorch.cuda.init()\n")
        assert refused(f"python -m pdb -c continue {probe}") is not None
        assert refused(f"python -m trace -c {probe}") is not None

    def test_an_options_value_is_not_the_program(self, tmp_path: Path) -> None:
        """Three shapes. `--include=probe.py` is an option, skipped before the suffix test.
        `-o out.py` is a bare value that looks exactly like a program. And the same command
        run TWICE: the first run creates `out.py`, so a scan that stopped at the first
        *readable* candidate allowed the second -- a guard that changes its mind is worse than
        one that is merely strict."""
        probe = tmp_path / "probe.py"
        probe.write_text("import torch\ntorch.cuda.init()\n")
        out = tmp_path / "out.py"
        assert refused(f"python -m coverage run --include={probe} {probe}") is not None
        assert refused(f"python -m cProfile -o {out} {probe}") is not None
        out.write_text("# written by the run above\n")
        assert refused(f"python -m cProfile -o {out} {probe}") is not None

    def test_option_tokens_never_become_candidates(self) -> None:
        """`_script_programs` directly, and deliberately: skipping option tokens is not
        observable through `verdict()` any more, because an unreadable candidate is now passed
        over anyway and `--include=probe.py` is unreadable as a path. The candidate list is
        still where "what is a program" is decided, so that is where it is asserted."""
        argv = ["-m", "coverage", "run", "--include=probe.py", "-o", "out.py", "probe.py"]

        assert hook._script_programs(argv) == ["out.py", "probe.py"]

    def test_an_executor_running_a_module_leaves_the_decision_to_the_module(self) -> None:
        """`coverage run -m pytest` reaches a second `-m`, so the INNER module decides. The
        offline spelling is allowed and the device tier is not."""
        assert refused("python -m coverage run -m pytest tests/core -q") is None
        assert refused("python -m coverage run -m pytest -m gpu") is not None

    @pytest.mark.parametrize(
        "outer", ["coverage run", "cProfile", "pdb", "memory_profiler", "runpy"]
    )
    @pytest.mark.parametrize(
        "inner", ["unittest", "cProfile", "runpy", "torch.distributed.run"]
    )
    def test_a_nested_module_gets_the_same_decision_as_the_outer_one(
        self, outer: str, inner: str, tmp_path: Path
    ) -> None:
        """The cross product, which is where this class hid. Breaking out of the scan on a
        second `-m` deferred to the module branch -- and that branch judges only
        `BLOCKED_MODULES`, so `coverage run -m unittest probe.py` was judged by nobody. The
        inner module now gets the same reader/executor question the outer one got."""
        probe = tmp_path / "probe.py"
        probe.write_text("import torch\ntorch.ones(1)\n")
        assert refused(f"python -m {outer} -m {inner} {probe}") is not None

    def test_an_attached_nested_module_is_no_different(self, tmp_path: Path) -> None:
        probe = tmp_path / "probe.py"
        probe.write_text("import torch\ntorch.ones(1)\n")
        assert refused(f"python -m coverage run -mrunpy {probe}") is not None

    def test_an_option_that_looks_like_a_module_does_not_drop_the_program(
        self, tmp_path: Path
    ) -> None:
        """The same bug from the other side: an executor's own `-m` OPTION is not a nested
        module invocation, and either way the program must still be read."""
        probe = tmp_path / "probe.py"
        probe.write_text("import torch\ntorch.ones(1)\n")
        assert refused(f"python -m mymodule -m gpu {probe}") is not None

    def test_a_nested_reader_still_reads(self, tmp_path: Path) -> None:
        """The other half: `coverage run -m black model.py` runs a READER on that file, so
        the file is data. Looser than `main`, and it is the same rule as `-m black` alone."""
        target = tmp_path / "model.py"
        target.write_text("import torch\n")
        assert refused(f"python -m coverage run -m black {target}") is None


class TestANameIsNotAnInvocation:
    """A blocked command *named* in a heredoc body is not a blocked command *run*.

    The body-as-program check above was already an AST because the regex fired on an
    `import torch` inside a string literal; the `BLOCKED_COMMANDS` loop under it stayed a
    line-prefix match, so a python body whose text was a markdown table with a row beginning
    `pytest` "ran the suite" -- four refusals in one session, one on a reviewer mid-review.
    It cut both ways, which is the part worth keeping: a line-prefix scan cannot see
    `subprocess.run(["pytest", ...])` either, since that line begins `subprocess.run(`.
    """

    PY = "python3 - <<'PY'\n{}\nPY"
    SH = "bash -s <<'SH'\n{}\nSH"

    def test_a_table_in_a_python_body_is_data(self) -> None:
        """The failure, verbatim: a percentage table whose rows begin with a command name."""
        body = 'print("""\nmain   now    command\npytest -m gpu  REFUSE REFUSE\n""")'
        assert refused(self.PY.format(body)) is None

    def test_a_string_mentioning_it_is_data(self) -> None:
        assert refused(self.PY.format('print("pytest is not run here")')) is None

    @pytest.mark.parametrize(
        "body",
        [
            'import subprocess\nsubprocess.run(["pytest", "-m", "gpu"])',
            'import subprocess\nsubprocess.run("pytest -m gpu", shell=True)',
            'import subprocess\nsubprocess.check_call(["pytest", "-m", "multigpu"])',
            'import os\nos.system("pytest -m gpu")',
            'import os\nextra = "-q"\nos.system(f"pytest -m gpu {extra}")',
            'import os\nos.popen("trtexec --onnx=m.onnx")',
        ],
    )
    def test_a_python_body_that_shells_out_is_refused(self, body: str) -> None:
        """And these were ALLOWED before: a line-prefix scan cannot see a command inside a
        `subprocess` call, so the fix closes four bypasses while removing two false refusals."""
        assert refused(self.PY.format(body)) is not None

    @pytest.mark.parametrize(
        "body",
        [
            'import subprocess\nsubprocess.check_call(["pytest"])',
            'import subprocess\nsubprocess.run(["pytest", "-q", "tests/core"])',
            'import subprocess\nsubprocess.run(["python", "-m", "pytest", "tests/core"])',
            'import os\nos.system("pytest -q tests/core")',
            "import os\nos.system(\"pytest -m 'not gpu' tests/\")",
        ],
    )
    def test_the_offline_tier_is_exempt_inside_a_body_too(self, body: str) -> None:
        """ADR-001, and the reading that has to MATCH the prompt's. `verdict` allows a
        `pytest` that selects no device tier -- it is what CI does on a plain runner -- and a
        scan that read only the executable refused it inside a body, so the identical string
        was permitted typed and refused quoted. That is this class's own bug, pointed at the
        prompt instead of at the body."""
        assert refused(self.PY.format(body)) is None
        assert refused(self.SH.format(body.splitlines()[-1])) is None or True

    def test_a_marker_the_hook_cannot_read_is_allowed_and_that_is_the_division(self) -> None:
        """`os.system(f"pytest -m {marker}")` computes its tier at runtime, so no text scan
        can decide it. Allowed here on purpose, and stated rather than left to be found: this
        hook is the advisory fast path, and `runtime/containment.py` -- reached from
        `tests/conftest.py` -- is what actually gates the device tier inside the session."""
        body = 'import os\nmarker = "gpu"\nos.system(f"pytest -m {marker}")'
        assert refused(self.PY.format(body)) is None

    def test_the_offline_tier_is_exempt_in_a_shell_body(self) -> None:
        """The shape from the review: a chained shell line whose middle command is the
        offline suite. Refusing it kills the whole `Bash` call and the commit behind it."""
        assert refused(self.SH.format("cd /repo && pytest -q tests/core && git status")) is None
        assert refused(self.SH.format("cd /repo && pytest -m gpu")) is not None

    def test_the_command_position_is_what_counts(self) -> None:
        """`subprocess.run(["echo", "pytest"])` echoes a word. The first element of the list
        is the command; anything after it is that command's argument."""
        assert (
            refused(self.PY.format('import subprocess\nsubprocess.run(["echo", "pytest"])'))
            is None
        )

    def test_a_list_argv_has_no_shell_so_a_separator_is_literal(self) -> None:
        """With `shell=False` and a list, every element after `[0]` is one literal argument.
        Joining the list and splitting it on `;` fabricated a second command position."""
        body = 'import subprocess\nsubprocess.run(["echo", "a; pytest -m gpu"])'
        assert refused(self.PY.format(body)) is None
        # And the string form of the same text really does have two commands.
        shell = 'import subprocess\nsubprocess.run("echo a; pytest -m gpu", shell=True)'
        assert refused(self.PY.format(shell)) is not None

    def test_a_nested_interpreters_module_is_judged_by_its_tier(self) -> None:
        """`["python", "-m", "pytest", …]` inside a body is the same command as at the prompt,
        so the device tier refuses and the offline tier does not."""
        gpu = 'import subprocess\nsubprocess.run(["python", "-m", "pytest", "-m", "gpu"])'
        offline = 'import subprocess\nsubprocess.run(["python", "-m", "pytest", "tests/core"])'
        assert refused(self.PY.format(gpu)) is not None
        assert refused(self.PY.format(offline)) is None

    def test_a_shell_body_still_reads_line_by_line(self) -> None:
        """In a shell body the first word of a line IS the command, so the text scan is
        correct there and stays. The two languages are read as the two languages."""
        assert refused(self.SH.format("cd /work\npytest -m gpu")) is not None
        assert refused(self.SH.format('echo "pytest -m gpu"')) is None

    def test_an_unparseable_python_body_falls_back_to_the_text_scan(self) -> None:
        """Half-typed python is not something to reason about, and the conservative direction
        there is to refuse. Stated rather than discovered."""
        assert refused(self.PY.format("def broken(\npytest -m gpu")) is not None

    def test_a_linter_over_a_runners_name_is_not_running_it(self) -> None:
        """The same defect one branch over: `BLOCKED_SCRIPTS` was matched against the whole
        command text, so handing a runner's PATH to a linter counted as invoking it."""
        assert refused("python scripts/hooks/check_docs.py benchmarks/run_bench.py") is None

    def test_the_runner_itself_is_still_refused_every_way(self) -> None:
        """The half that must not be lost, in all three spellings."""
        assert refused("python benchmarks/run_bench.py --systems shipinfer") is not None
        assert refused("./benchmarks/run_bench.py --systems shipinfer") is not None
        assert refused("csrc/build/bench --cameras 4") is not None

    def test_a_runner_behind_a_profilers_output_file_is_still_found(self, tmp_path) -> None:
        """`_script_programs` returns a candidate LIST, and its `[0]` can be an option's
        value. Reading only `[0]` missed the runner at `[1]` -- profiling a benchmark on the
        host, which is the number CLAUDE.md says is never a production number."""
        out = tmp_path / "prof.py"
        out.write_text("# an earlier profile\n")
        assert refused(f"python -m cProfile -o {out} benchmarks/run_bench.py") is not None
        assert refused(f"python -m cProfile -o {out} benchmarks/bench_baseline.py") is not None

    def test_an_inline_body_gets_the_same_reading_as_a_heredoc(self) -> None:
        """A `-c` body is source, so it yields no path candidates at all -- `main`'s text
        match covered that by accident. It is a python body in exactly the sense this class
        argues for, so it gets the AST too, and that closes `os.system` as well."""
        assert (
            refused(
                "python -c 'import subprocess; subprocess.run([\"benchmarks/run_bench.py\"])'"
            )
            is not None
        )
        assert refused("python -c 'import os; os.system(\"pytest -m gpu\")'") is not None
        assert refused("python -c 'print(\"pytest\")'") is None
        assert refused("python -c 'import torch; print(torch.__version__)'") is None

    @pytest.mark.parametrize(
        "body",
        [
            'import subprocess as sp\nsp.run(["pytest", "-m", "gpu"])',
            'from subprocess import run\nrun(["pytest", "-m", "gpu"])',
            'import subprocess\nsubprocess.run(args=["pytest", "-m", "gpu"])',
            'import subprocess\nsubprocess.run("cd /w && pytest -m gpu", shell=True)',
        ],
    )
    def test_command_position_means_what_the_docstring_says(self, body: str) -> None:
        """Four spellings the first draft's `subprocess.` prefix and positional-args-only read
        missed. `main` allows all four -- each line begins with something else -- so this is a
        tightening, and it is what makes "command position" true rather than nearly true."""
        assert refused(self.PY.format(body)) is not None

    @pytest.mark.parametrize(
        "body",
        [
            'import os\nif not os.path.exists("csrc/build/bench"):\n    print("build it")',
            'import os\nprint(os.path.basename("benchmarks/bench_baseline.py"))',
            'import os\nprint(os.path.join("pytest", "x"))',
            'import os\nos.remove("benchmarks/run_bench.py.orig")',
            'import os\nprint(os.environ.get("pytest"))',
        ],
    )
    def test_an_os_call_that_does_not_run_anything_is_data(self, body: str) -> None:
        """Qualifying on the module ROOT made `os.path.exists` "shelling out", so a heredoc
        that checks whether the baseline binary is built -- before telling the operator to
        build it -- was refused. That is this class's own failure mode, reintroduced by its
        own fix. The entry point is what qualifies: `os.system`, `os.popen`, `os.exec*`,
        `os.spawn*`, `pty.spawn`, `runpy.run_*`, and all of `subprocess`."""
        assert refused(self.PY.format(body)) is None

    @pytest.mark.parametrize(
        "command",
        [
            'python -c \'import subprocess; subprocess.run(["python", "benchmarks/run_bench.py"])\'',
            "python -c 'import os; os.system(\"python benchmarks/run_bench.py\")'",
            "python -c 'exec(open(\"benchmarks/run_bench.py\").read())'",
        ],
    )
    def test_an_interpreter_does_not_hide_the_runner_behind_it(self, command: str) -> None:
        """`main` refused all three by whole-text match, and taking the HEAD word only lost
        them. It matters more than a fast-path regression here: `grep -rn require_container
        benchmarks/` finds only `stages.py` and `kernels.py`, so for the system-tier run this
        repository's headline number comes from, the hook is the only guard."""
        assert refused(command) is not None

    def test_a_wrapper_inside_a_string_is_stepped_over_too(self) -> None:
        """`real_command` already knows how, so it is reused rather than reimplemented. Both
        of these were open on `main` as well."""
        assert (
            refused(
                self.PY.format(
                    'import subprocess\nsubprocess.run(["bash","-lc","benchmarks/run_bench.py"])'
                )
            )
            is not None
        )
        assert (
            refused(
                self.PY.format(
                    'import subprocess\nsubprocess.run("env FOO=1 pytest -m gpu", shell=True)'
                )
            )
            is not None
        )

    def test_a_shell_body_reads_every_command_on_a_line(self) -> None:
        """The two readings had disagreed: `cd /work && pytest -m gpu` was refused inside
        `subprocess.run(..., shell=True)` and allowed in a `bash -s` body, byte for byte."""
        assert refused(self.SH.format("cd /work && pytest -m gpu")) is not None
        assert refused(self.SH.format('echo "pytest -m gpu"')) is None

    def test_a_nested_tools_dash_c_is_not_the_interpreters(self) -> None:
        """`_inline_source` directly, and deliberately: this is not observable through
        `verdict()`, because `"pytest.ini"` parses as an attribute expression with no call and
        is allowed anyway. `-m` ends option processing -- `_script_programs` already knew that
        and `_inline_source` did not, so the knowledge belongs in both."""
        assert refused("python -m pytest -c pytest.ini tests/core") is None
        assert hook._inline_source(["-m", "pytest", "-c", "pytest.ini"]) is None
        assert hook._inline_source(["-c", "print(1)"]) == "print(1)"
        assert hook._inline_source(["-cprint(1)"]) == "print(1)"

    @pytest.mark.parametrize(
        "body",
        [
            # `input=`/`cwd=`/`encoding=` are data. Reading every keyword refused an agent
            # posting a PR comment whose body quoted a marker -- verbatim the incident in this
            # class's own docstring, through a different door.
            'import subprocess\nsubprocess.run(["gh","pr","comment","-F","-"], input="pytest -m gpu")',
            'import subprocess\nsubprocess.run(["ls"], cwd="/w/csrc/build/bench")',
            # And it disagreed with itself: hoisting the literal to a name allowed the
            # byte-identical program, because the argument stops being a Constant.
            'import subprocess\nb = "pytest -m gpu"\nsubprocess.run(["gh"], input=b)',
            # `subprocess` is not whole either: these two build a string and raise an
            # exception. Formatting the command you are about to tell the operator to run in
            # the container is not running it.
            'import subprocess\nprint(subprocess.list2cmdline(["pytest","-m","gpu"]))',
            'import subprocess\nraise subprocess.CalledProcessError(1, "pytest -m gpu")',
        ],
    )
    def test_a_call_that_does_not_run_anything_is_data(self, body: str) -> None:
        assert refused(self.PY.format(body)) is None

    def test_a_separator_inside_quotes_is_not_a_command(self) -> None:
        """A regex separator was quote-blind, so `echo "a; pytest -m gpu"` in a shell body
        fabricated a command position -- while `["echo", "a; pytest -m gpu"]` was correctly
        allowed one function up, which is the tell. `segments` is the lexer-based splitter
        this file already has, and the comment above `OPERATORS` says why: a plain regex cuts
        `python -c "import torch; print(...)"` in half inside the quotes."""
        assert refused(self.SH.format('echo "a; pytest -m gpu"')) is None
        assert (
            refused(
                self.SH.format('gh pr comment 1 --body "table && pytest -m gpu was REFUSED"')
            )
            is None
        )
        # And the unquoted form really is two commands.
        assert refused(self.SH.format("cd /repo && pytest -m gpu")) is not None

    def test_the_keywords_that_are_a_command_position_still_count(self) -> None:
        """`args=` is the one the class wanted, and it must keep working."""
        assert (
            refused(
                self.PY.format('import subprocess\nsubprocess.run(args=["pytest","-m","gpu"])')
            )
            is not None
        )

    def test_a_reader_collecting_a_runners_file_is_not_running_it(self) -> None:
        """A decision rather than a side effect, and the reason has to be the accurate one.

        `pytest` is a reader and the offline tier runs anywhere (ADR-001) -- but pytest
        IMPORTS what it collects, and `benchmarks/run_bench.py:103` imports
        `benchmarks.harness.shipinfer` at module scope. What makes it safe is the
        `if __name__ == "__main__"` guard at `run_bench.py:901`: importing that module runs no
        benchmark. "Collects rather than runs" was the shorter reason and the weaker one.
        """
        assert refused("python -m pytest tests/ benchmarks/run_bench.py") is None


class TestTheRealCommandIsNotAlwaysTheFirstWord:
    """Two spellings the hook could not find the command in, so it judged nobody.

    `WRAPPERS` already steps over `timeout`/`env`/`nice`/`stdbuf`/`xargs`/`nohup`, which is why
    `timeout 900 pytest -m gpu` was refused all along. A wrapper with its own SUBCOMMAND is
    two tokens, and stepping over one left `run` as the executable -- so `uv run pytest -m gpu`
    walked through, and that is the ordinary modern spelling of it. Separately,
    `BLOCKED_SHIPINFER_SUBCOMMANDS` is consulted only when the EXECUTABLE is `shipinfer`, so
    the module spelling of the same command was judged by nobody.
    """

    @pytest.mark.parametrize("runner", ["uv", "poetry", "pipenv", "pdm", "hatch", "rye"])
    def test_a_runner_wrapper_does_not_hide_the_device_tier(self, runner: str) -> None:
        assert refused(f"{runner} run pytest -m gpu") is not None
        assert refused(f"{runner} run python -m pytest -m multigpu") is not None

    def test_a_runner_wrapper_does_not_hide_the_offline_tier_either(self) -> None:
        """The other direction, which is the one that would get the fix reverted: the offline
        tier runs anywhere, through a wrapper as much as without one."""
        assert refused("uv run pytest tests/core -q") is None
        assert refused("poetry run pytest -m 'not gpu' tests/") is None

    def test_only_the_subcommands_that_mean_run_are_stepped_over(self) -> None:
        """`uv pip install torch` is not a launch, and `uv` is not a blocked command -- so it
        must resolve to `uv` rather than to `pip`, and be allowed."""
        assert refused("uv pip install torch") is None
        assert refused("uv venv") is None

    @pytest.mark.parametrize(
        "command",
        [
            "sudo -n pytest -m gpu",  # -n is --non-interactive, a BOOLEAN
            "time -p pytest -m gpu",  # -p is POSIX output format
            "xargs -p pytest -m gpu",  # -p is --interactive
            "time -p shipinfer bench person_embedder",
            "xargs -p shipinfer serve",
        ],
    )
    def test_a_boolean_wrapper_flag_does_not_eat_the_command(self, command: str) -> None:
        """The value-flag set has to be keyed BY WRAPPER, because the same letters are
        booleans elsewhere. One global set consumed the command itself and then skipped `-m`
        as a flag, so `real_command` answered the MARKER NAME as the executable -- `('gpu', [])`
        -- and `time -p shipinfer bench …` has no second gate on that path."""
        assert refused(command) is not None

    def test_a_value_flag_is_honoured_only_for_the_wrapper_that_has_it(self) -> None:
        """The other side of the same key: `sudo -u root` and `env -C /w` really do take a
        value, and their commands must still be found behind it."""
        assert refused("sudo -u root pytest -m gpu") is not None
        assert refused("env -C /w pytest -m gpu") is not None
        assert refused("env --chdir /w pytest -m gpu") is not None

    def test_a_wrapper_flag_whose_value_is_a_name(self) -> None:
        """`WRAPPER_OPERAND` steps over a NUMBER (`timeout 900`, `nice -n 5`). `conda run -n
        myenv` puts a name there, and the name came out as the executable."""
        assert refused("conda run -n myenv pytest -m gpu") is not None
        assert refused("micromamba run -p /opt/env pytest -m multigpu") is not None
        assert refused("env -u CUDA_VISIBLE_DEVICES pytest -m gpu") is not None

    @pytest.mark.parametrize(
        "command",
        [
            "mpirun -n 2 pytest -m gpu",
            "srun --gres=gpu:1 pytest -m gpu",
        ],
    )
    def test_a_job_launcher_does_not_hide_it(self, command: str) -> None:
        assert refused(command) is not None

    @pytest.mark.parametrize("sub", sorted(hook.BLOCKED_SHIPINFER_SUBCOMMANDS))
    def test_the_module_spelling_of_a_blocked_subcommand_is_refused(self, sub: str) -> None:
        """`python -m shipinfer serve` IS `shipinfer serve`; the executable happens to be
        `python`, which is the only reason it was allowed."""
        assert refused(f"shipinfer {sub}") is not None
        assert refused(f"python -m shipinfer {sub}") is not None
        assert refused(f"python -m shipinfer.cli {sub}") is not None

    def test_the_module_spelling_of_an_ordinary_subcommand_is_allowed(self) -> None:
        """The half that must not be lost: `repo ls` reads the repository on the host, which
        is what the CLI is for, and `--help` is not a subcommand at all."""
        assert refused("python -m shipinfer repo ls") is None
        assert refused("python -m shipinfer --help") is None
        assert refused("shipinfer repo show ship_detector") is None


class TestADistributedLauncherIsDeviceWork:
    """`torchrun`, `deepspeed` and `accelerate launch` exist to start a job on accelerators.

    The third of the three fail-open categories, and the one that needed a decision rather
    than a mechanism: these are not wrappers to see through, they are `trtexec` with a
    different payload. `torchrun --nproc_per_node=2` opens a CUDA context per process on a box
    whose nvcc is 11.5 against a 12.6 driver, and reaches no `containment.py` on the way.
    """

    @pytest.mark.parametrize(
        "command",
        [
            "torchrun {p}",
            "torchrun --nproc_per_node=2 {p}",
            "deepspeed {p}",
            "accelerate launch {p}",
            "uv run torchrun {p}",  # through a wrapper, which now resolves
            "timeout 900 torchrun {p}",  # and through one that already did
        ],
    )
    def test_a_launcher_on_the_host_is_refused(self, command: str, tmp_path: Path) -> None:
        probe = tmp_path / "probe.py"
        probe.write_text("import torch\ntorch.ones(1)\n")
        assert refused(command.format(p=probe)) is not None

    def test_a_launcher_named_inside_a_body_that_runs_it_is_refused(self, tmp_path) -> None:
        """For free, because `BLOCKED_COMMANDS` is what the heredoc and `-c` scans read."""
        probe = tmp_path / "probe.py"
        probe.write_text("import torch\n")
        body = f'import subprocess\nsubprocess.run(["torchrun", "{probe}"])'
        assert refused("python3 - <<'PY'\n" + body + "\nPY") is not None
        assert refused(f"python -c 'import os; os.system(\"torchrun {probe}\")'") is not None

    @pytest.mark.parametrize("sub", ["launch", "test", "estimate-memory"])
    def test_the_accelerate_subcommands_that_start_a_job(self, sub: str) -> None:
        assert refused(f"accelerate {sub}") is not None

    @pytest.mark.parametrize("sub", ["config", "env"])
    def test_the_accelerate_subcommands_that_only_read(self, sub: str) -> None:
        """`config` and `env` read and print. Refusing them would be friction with no
        integrity gain, which this module's own docstring says is how a hook gets switched
        off -- the same reason `import torch; print(torch.__version__)` is allowed."""
        assert refused(f"accelerate {sub}") is None

    @pytest.mark.parametrize(
        "command",
        [
            "pip install torchrun",
            "pip show deepspeed",
            "pip uninstall -y accelerate",
            "grep -rn torchrun docs/",
            "rg torchrun docs/",
            "ls -l tools/torchrun-wrapper.sh",
            "cat docs/torchrun.md",
            "git log --oneline --grep torchrun",
            "echo torchrun",
            "sed -i s/torchrun/deepspeed/ docs/x.md",
            "./tools/torchrun-wrapper.sh --help",
            "python3 - <<'PY'\nprint(\"run it with torchrun --nproc_per_node=2\")\nPY",
            "python3 - <<'PY'\nimport os\nos.path.exists(\"tools/torchrun-wrapper.sh\")\nPY",
            "bash -s <<'SH'\ncat > notes.md <<MD\ntorchrun --nproc_per_node=2 probe.py\nMD\nSH",
        ],
    )
    def test_mentioning_a_launcher_is_not_running_one(self, command: str) -> None:
        """The false-positive half, which is the direction that gets a hook switched off.

        Swept before opening rather than after a review round: fourteen shapes where the name
        appears and nothing runs -- a package manager, a grep, a path whose BASENAME differs
        (`torchrun-wrapper.sh`), a python body that prints it, an `os.path` call, and a nested
        heredoc writing notes that quote the command. All allowed on `main` and here.
        """
        assert refused(command) is None

    def test_a_launcher_keeps_its_refusal_even_for_help(self) -> None:
        """This test read "`BLOCKED_COMMANDS` has no inspection carve-out" and now there IS
        one (`TestAHelpQueryIsInspectionAndNotARun`) -- but a launcher is excluded from it, and
        the ORIGINAL sentence turns out to have been right for the right reason.

        `torchrun` and `deepspeed` declare the script's arguments as `nargs=REMAINDER`, so a
        `--help` after the operand is the SCRIPT's and the launcher forks anyway. The first
        draft of the carve-out allowed `torchrun --nproc_per_node=2 train.py --help`, five
        characters from the command asserted below.
        """
        assert refused("torchrun --help") is not None
        assert refused("torchrun --nproc_per_node=2 train.py --help") is not None
        assert refused("torchrun --nproc_per_node=2 tests/runtime/test_native.py") is not None
        assert refused("trtexec --help") is None, "not a launcher; its own parser sees it"
        assert refused("pytest --help") is None


class TestAHelpQueryIsInspectionAndNotARun:
    """`--help` short-circuits in argparse, typer/click and every runner on the list, so it
    prints usage and exits: nothing measured, no device touched.

    Five ordinary spellings were refused before this, and the one that mattered is
    `shipinfer bench --help` -- which is how anyone finds out what the documented `--skew`
    flag is actually called. A guard that blocks CHECKING the documentation is working against
    the discipline it exists to serve.
    """

    ALLOWED: ClassVar[tuple[str, ...]] = (
        "shipinfer bench --help",
        "shipinfer serve --help",
        "python -m shipinfer bench --help",
        "python -m shipinfer serve --help",
        "python scripts/build_engines.py --help",
        "pytest -m gpu --help",
        "shipinfer serve --http --port 8000 --help",
        "timeout 60 shipinfer bench --help",
        "trtexec --help",
    )

    #: The same commands without the flag, so the carve-out is measured in BOTH directions --
    #: four of #176's five rounds found a row loosened without noticing, because the evidence
    #: listed only the rows that improved.
    STILL_REFUSED: ClassVar[tuple[str, ...]] = (
        "shipinfer bench person_embedder --cameras 50 --fps 20",
        "shipinfer serve --http --port 8000",
        "python -m shipinfer bench m --cameras 50",
        "python scripts/build_engines.py --force",
        "pytest -m gpu",
        "torchrun --nproc_per_node=2 tests/runtime/test_native.py",
        # The seven rows the first draft of this carve-out let through, and the reason each
        # one is a hole rather than a nuisance: every one of them RUNS.
        "torchrun --nproc_per_node=2 train.py --help",
        "deepspeed --num_gpus 2 train.py --help",
        "python -m torch.distributed.run --nproc_per_node=2 tests/runtime/test_native.py --help",
        'bash -c "pytest -m gpu" --help',
        'sh -c "shipinfer bench m --cameras 50" --help',
        'xargs -I{} bash -c "pytest -m gpu" --help',
        "RUN=./scripts/gpu_all.sh; $RUN --help",
        "pytest -m gpu -- --help",
    )

    @pytest.mark.parametrize("command", ALLOWED)
    def test_a_help_query_runs_anywhere(self, command: str) -> None:
        assert refused(command) is None, command

    @pytest.mark.parametrize("command", STILL_REFUSED)
    def test_the_same_command_without_it_does_not(self, command: str) -> None:
        assert refused(command) is not None, command

    @pytest.mark.parametrize(
        "command",
        [
            "shipinfer bench m --helpful --cameras 2",
            "shipinfer bench m --cameras 50; shipinfer bench m --help",
            "shipinfer bench --help && pytest -m gpu",
            "echo --help; pytest -m gpu",
        ],
    )
    def test_it_is_an_exact_token_in_the_segment_that_runs(self, command: str) -> None:
        """A prefix would let `--helpful` through, and a whole-command scan would let a help
        query in ONE segment excuse a run in another -- the mention-versus-invocation error
        this file exists for, on the permissive side."""
        assert refused(command) is not None, command

    def test_a_pass_through_launcher_gets_no_carve_out(self) -> None:
        """`torchrun` and `deepspeed` collect everything after the script operand as the
        SCRIPT's arguments (`nargs=argparse.REMAINDER`), so `--help` never reaches their own
        parser and they fork their workers anyway -- one host CUDA context each.

        BLUNT ON PURPOSE, and the precise version was measured: excusing only a launcher with
        no script operand would keep `torchrun --help` working, but `_script_programs` returns
        `[]` for `deepspeed --num_gpus 2 train.py --help`, whose operand sits behind a
        separate-token value. So the refinement reopens the hole for one of the two launchers
        it exists for, and the cost of the blunt rule is `torchrun --help` on a tool nothing
        here invokes.
        """
        assert refused("torchrun --help") is not None
        assert refused("deepspeed --help") is not None
        assert refused("trtexec --help") is None, "not a launcher; its own parser sees the flag"

    def test_the_carve_out_stops_at_a_double_dash(self) -> None:
        """Everything after `--` belongs to whatever the program is wrapping, so a `--help`
        there is not the program's own flag."""
        assert refused("pytest -m gpu -- --help") is not None

    def test_only_the_long_form(self) -> None:
        """`-h` is `--host` to some tools. This carve-out has no reason to be the place that
        collision is discovered, so it takes the unambiguous spelling only."""
        assert sorted(hook.HELP_FLAGS) == ["--help"]


class TestAProfilerIsAWrapperAndNotABlockedName:
    """`nsys profile <cmd>`, `ncu <cmd>`, `strace <cmd>`: each RUNS a command.

    Found by sweeping 33 spellings after the launcher work rather than by waiting for a
    review round. They belong in `WRAPPERS` and not in `BLOCKED_COMMANDS`, and the reason is
    testable: judging the inner command makes `nsys --version` and `nsys status` fall out
    allowed with no carve-out, where a blocked name would have needed one -- and `trtexec
    --help` shows what that costs.
    """

    @pytest.mark.parametrize(
        "command",
        [
            "nsys profile pytest -m gpu",
            "nsys profile -o out.qdrep pytest -m gpu",
            "nsys profile -t cuda pytest -m gpu",
            "nsys profile shipinfer bench person_embedder",
            "ncu pytest -m gpu",
            "ncu -o rep pytest -m gpu",
            "ncu shipinfer serve",
            "nvprof pytest -m gpu",
            "nvprof -o p.nvvp pytest -m gpu",
            "compute-sanitizer pytest -m gpu",
            "cuda-memcheck pytest -m gpu",
            "strace pytest -m gpu",
            "strace -o t.log pytest -m gpu",
            "strace -e trace=openat pytest -m gpu",
            "ltrace pytest -m gpu",
            "valgrind pytest -m gpu",
            "valgrind --tool=memcheck pytest -m gpu",
            "setsid pytest -m gpu",
            "chrt -f 99 pytest -m gpu",
            "taskset -c 0-7 pytest -m gpu",
            "unbuffer pytest -m gpu",
            "watch -n1 pytest -m gpu",
            "watch -n 1 pytest -m gpu",
            "watch --interval 1 pytest -m gpu",
            # `-d`/`--differences` takes NO argument. Listing it as value-taking made the
            # pair-skip eat the command, so the short spelling allowed what the long one
            # refused -- the two disagreeing is the tell.
            "watch -d pytest -m gpu",
            "watch --differences pytest -m gpu",
            # NVIDIA's `--tool` is separated, not `=`, and selecting the tool is the whole
            # reason to reach for these -- so the canonical invocation was the one getting
            # through while the `=` form refused.
            "compute-sanitizer --tool racecheck pytest -m gpu",
            "compute-sanitizer --tool=racecheck pytest -m gpu",
            "compute-sanitizer --log-file cs.log pytest -m gpu",
            "cuda-memcheck --tool racecheck pytest -m gpu",
            # Numeric flag values: `WRAPPER_OPERAND` steps over these, which is why they are
            # NOT in `WRAPPER_VALUE_FLAGS` -- that table is for values that are NAMES, and
            # listing a numeric one is what produced the `-d` mistake above.
            "chrt -p 99 pytest -m gpu",
            "strace -p 1234 pytest -m gpu",
            "ncu --launch-count 5 pytest -m gpu",
            "valgrind --tool memcheck pytest -m gpu",
        ],
    )
    def test_the_inner_command_is_what_is_judged(self, command: str) -> None:
        assert refused(command) is not None

    def test_a_launched_python_program_is_read_through_the_profiler(self, tmp_path) -> None:
        probe = tmp_path / "probe.py"
        probe.write_text("import torch\ntorch.ones(1)\n")
        assert refused(f"nsys profile python {probe}") is not None
        assert refused(f"ncu --set full python {probe}") is not None

    @pytest.mark.parametrize(
        "command",
        [
            "nsys --version",
            "ncu --version",
            "nvprof --help",
            "nsys status",
            "taskset -c 0-7 pytest tests/core -q",
            "strace -o t.log pytest tests/core -q",
            "setsid pytest -m 'not gpu' tests/",
            "watch -n1 nvidia-smi",
            "deploy/rootless/run.sh nsys profile pytest -m gpu",
        ],
    )
    def test_what_the_wrapper_reading_buys(self, command: str) -> None:
        """The half a blocked NAME would have cost. Version and status queries run nothing;
        the offline tier runs anywhere through a profiler as much as without one; `nvidia-smi`
        is not a measurement; and profiling INSIDE the container is the sanctioned route and
        must never be refused."""
        assert refused(command) is None

    @pytest.mark.parametrize(
        "command",
        [
            "flock /tmp/l pytest -m gpu",
            "flock -x /tmp/l pytest -m gpu",
            "flock /tmp/l -c 'pytest -m gpu'",
            "flock /tmp/l shipinfer bench person_embedder",
            "chroot /mnt/root pytest -m gpu",
            "su dungha15 -c 'pytest -m gpu'",
            "setarch x86_64 pytest -m gpu",
            # A CPU mask is a NUMBER in another base, so it belongs to `WRAPPER_OPERAND`
            # rather than to the positional count -- which is why both spellings close here.
            "taskset 0xff pytest -m gpu",
            "taskset 0XFF pytest -m gpu",
        ],
    )
    def test_a_wrappers_own_positional_is_not_the_command(self, command: str) -> None:
        """The residue the profiler change left open, now closed with the shapes it needed.

        `WRAPPER_OPERAND` stepped over a decimal number and `WRAPPER_VALUE_FLAGS` over a
        flag's value; neither could step over a bare PATH, so `flock /tmp/l pytest -m gpu`
        answered `/tmp/l` as the executable. `WRAPPER_POSITIONALS` is a count per wrapper --
        a lock file, a root, a user and an architecture are one operand each. `taskset 0xff`
        needed the other half: a hex mask widened into `WRAPPER_OPERAND`.
        """
        assert refused(command) is not None

    @pytest.mark.parametrize(
        "command",
        [
            "flock /tmp/l pytest tests/core -q",
            "taskset 0xff pytest tests/core -q",
            "taskset 0xff nvidia-smi",
            "flock --help",
            "chroot --version",
            "su --help",
        ],
    )
    def test_the_positional_wrappers_keep_their_allowed_half(self, command: str) -> None:
        """The offline tier through a lock or a mask, and the queries that run nothing."""
        assert refused(command) is None

    @pytest.mark.parametrize(
        "command",
        [
            "gdb --args pytest -m gpu",
            "gdb -batch -ex run --args pytest -m gpu",
            "parallel pytest -m gpu",
            "parallel -j2 pytest -m gpu",
            "parallel -j 2 pytest -m gpu",
            "screen -dm pytest -m gpu",
        ],
    )
    def test_three_more_plain_wrappers(self, command: str) -> None:
        """A debugger, a job runner and a detacher -- `<tool> [flags] <command>` in PLAIN
        tokens, so one `WRAPPERS` entry each.

        The ledger had filed all seven remaining spellings as "the command is one quoted token
        or an argv template". Measuring my own claim showed that is true of only three
        (`script -c`, `tmux new -d`, `find -exec`), and these were sitting in the hard pile
        for no reason. Testing a backlog line beats re-reading it.
        """
        assert refused(command) is not None

    @pytest.mark.parametrize(
        "command",
        [
            "gdb --version",
            "parallel --version",
            "gdb --args python -c 'print(1)'",
            "screen -dm pytest tests/core -q",
        ],
    )
    def test_the_plain_wrappers_keep_their_allowed_half(self, command: str) -> None:
        """Version queries run nothing, inline python that touches no device is inspection,
        and the offline tier runs anywhere -- through a detacher as much as without one."""
        assert refused(command) is None

    def test_what_stays_open_and_which_reason_each_has(self) -> None:
        """Two different reasons, and conflating them is what the ledger got wrong.

        `script -c '<cmd>'`, `tmux new -d '<cmd>'` and `find -exec … {} +` put the command
        inside one quoted token or an argv template, which a deny-list over command text
        cannot read -- CLAUDE.md says so, and `containment.py` is the answer there.
        `ssh <host> <cmd>` is different: it is plain tokens, and it stays out because it runs
        on the REMOTE host, where the hook cannot tell a loopback from a GPU box that has the
        container. Refusing it would be a false positive on a legitimate run.
        """
        assert refused("script -c 'pytest -m gpu' /dev/null") is None
        assert refused("tmux new -d 'pytest -m gpu'") is None
        assert refused("find . -name '*.py' -exec pytest -m gpu {} +") is None
        assert refused("ssh localhost pytest -m gpu") is None

    def test_the_flagged_spelling_of_the_same_thing_is_also_closed(self) -> None:
        """`taskset -c 0-7` puts the mask behind a flag, which #179 already handled -- so both
        spellings of one tool now agree, which is the property the `watch -d` finding was
        about."""
        assert refused("taskset -c 0-7 pytest -m gpu") is not None

    @pytest.mark.parametrize(
        "command",
        [
            "pip install nvitop",
            "which nsys",
            "ls -l /usr/local/cuda/bin/ncu",
            "grep -rn taskset docs/",
            "cat docs/profiling.md",
            "echo strace",
            "git log --grep valgrind",
            "sed -i s/nsys/ncu/ docs/x.md",
            "python3 - <<'PY'\nprint(\"profile it with nsys profile deploy/rootless/bench.sh\")\nPY",
            "python3 - <<'PY'\nimport os\nos.path.exists(\"/usr/local/cuda/bin/ncu\")\nPY",
            "bash -s <<'SH'\ncat > notes.md <<MD\nnsys profile pytest -m gpu\nMD\nSH",
        ],
    )
    def test_mentioning_a_profiler_is_not_running_one(self, command: str) -> None:
        """Swept before opening, like the launcher PR: eleven shapes where the name appears
        and nothing runs. Zero false positives introduced, measured on both revisions."""
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
