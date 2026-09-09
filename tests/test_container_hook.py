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
            'import subprocess\nsubprocess.check_call(["pytest"])',
            'import os\nos.system("pytest")',
            'import os\nargs = 1\nos.system(f"pytest {args}")',
            'import os\nos.popen("trtexec --onnx=m.onnx")',
        ],
    )
    def test_a_python_body_that_shells_out_is_refused(self, body: str) -> None:
        """And these were ALLOWED before: a line-prefix scan cannot see a command inside a
        `subprocess` call, so the fix closes four bypasses while removing two false refusals."""
        assert refused(self.PY.format(body)) is not None

    def test_the_command_position_is_what_counts(self) -> None:
        """`subprocess.run(["echo", "pytest"])` echoes a word. The first element of the list
        is the command; anything after it is that command's argument."""
        assert (
            refused(self.PY.format('import subprocess\nsubprocess.run(["echo", "pytest"])'))
            is None
        )

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
