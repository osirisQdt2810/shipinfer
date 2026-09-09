#!/usr/bin/env python3
"""Refuse to run tests, benchmarks or GPU work outside a container.

The project rule is that every test, benchmark and measurement runs in the
container (`.claude/CLAUDE.md`, "Where commands run").  That rule was written
down and then quietly broken, repeatedly, because iterating on the host is
faster -- so it is now a `PreToolUse` hook on `Bash` rather than a paragraph
somebody is trusted to remember.  A host-measured number reported as a
container one is the specific failure this exists to make impossible.

Design constraint: **a hook with false positives gets switched off**, and a
switched-off hook enforces nothing.  So this refuses narrowly.  It looks for a
command that *invokes* a runner, not one that merely mentions it -- `grep -rn
pytest tests/` is a search and passes; `pytest tests/` is a test run and does
not.  Each shell segment is inspected separately, so a pipeline is judged by
what it actually executes.

Escape hatch: put `SHIPINFER_ALLOW_HOST_RUN=1` in the command.  It is
deliberately noisy and deliberately per-command -- there is no session-wide
switch, because "I turned it off an hour ago and forgot" is how the rule was
lost the first time.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import sys
from pathlib import Path

# --- what counts as "already in a container" -------------------------------

CONTAINER_MARKERS = ("/.dockerenv", "/run/.containerenv")

# A command that is itself the containerised entry point.  Matched as a
# substring because these appear after `bash`, `cd ... &&`, env prefixes and
# other noise that segment parsing would otherwise have to model.
CONTAINERISED = (
    "docker run",
    "docker exec",
    "docker compose run",
    "docker-compose run",
    "podman run",
    "podman exec",
    "nerdctl run",
    "deploy/rootless/test.sh",
    "deploy/rootless/run.sh",
    "deploy/rootless/bench.sh",
    "deploy/rootless/cpp.sh",
    "deploy/rootless/profile.sh",
    "deploy/rootless/prove.sh",
    "deploy/rootless/gst-image.sh",
    "deploy/rootless/wheels.sh",
)
# `make shell|test|bench` were here and there is no Makefile (the compose path they came from
# was replaced by the scripts above -- `setup.sh`, KERNEL LIMIT). Allowing a command that
# cannot run is worse than refusing it: the developer is handed it, gets "no makefile found",
# and reaches for SHIPINFER_ALLOW_HOST_RUN -- the outcome this hook exists to prevent.

# --- what must not run on the host ----------------------------------------

# Executables whose whole purpose is running the suite or the data plane.
BLOCKED_COMMANDS = {
    "pytest",
    "py.test",
    "trtexec",
    "polygraphy",
}

# Scripts that run the suite or a benchmark, wherever they are invoked from.
BLOCKED_SCRIPTS = (
    # `run_tests.sh` is deliberately absent. It adds `-m "not gpu and not multigpu"` and
    # exports empty `CUDA_VISIBLE_DEVICES`, so it is *strictly more* offline than the bare
    # `pytest` this hook allows. Refusing it while permitting `pytest` taught the developer
    # to reach for SHIPINFER_ALLOW_HOST_RUN, which is how the rule was lost the first time.
    "run_bench.py",
    "compare_baseline.py",
    "bench_baseline.py",
    "run_baseline.py",
    # The C++ plane's binaries that open a device. Run directly, the C++ benchmark binary used
    # to pass both enforcement points; it now consults the containment gate itself
    # (`runtime/containment.h`), and the hook knows its name so the common case stops here.
    # The CUDA-free test binaries (`test_scheduling`, `test_engine`, `test_containment`) are
    # deliberately absent: they link no accelerator library and are the C++ offline tier.
    "csrc/build/bench",
    "csrc/build/test_pipeline",
    "csrc/build/test_dataplane",
)

# `shipinfer <subcommand>` -- only the ones that touch a device or serve.
BLOCKED_SHIPINFER_SUBCOMMANDS = {"bench", "serve", "profile", "warmup"}

# Inline python (`-c`) or a module (`-m`) is blocked when it reaches for a
# device.  A bare `import torch; print(torch.__version__)` is version
# inspection, not a measurement, and is allowed on purpose: refusing it would
# add friction with no integrity gain, and friction is what gets hooks killed.
DEVICE_TOKENS = re.compile(
    r"\b(?:"
    r"cuda|hip|rocm|nvml|nvtx|tensorrt|trt\b|pycuda|cupy|"
    r"cudaSetDevice|current_device|device_count|set_device|"
    r"is_available|pin_memory|synchronize"
    r")",
    re.IGNORECASE,
)

# `python some_script.py` says nothing about what the script does, and the case
# that actually leaked a CUDA context on this box was exactly that shape.  So
# when the target is a readable local .py, the hook reads it and looks for a
# real device import.  Anchored at line start and requiring an import statement
# on purpose: `scripts/hooks/check_layers.py` contains the *string* "torch"
# because its job is to forbid that import, and blocking the layer checker
# would be a false positive with no upside.
DEVICE_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(?:torch|tensorrt|cupy|pycuda|pynvml|shipinfer\.runtime)\b",
    re.MULTILINE,
)
SCRIPT_READ_LIMIT = 256 * 1024

# Env assignments and wrappers to skip when finding a segment's real command.
WRAPPERS = {
    "sudo",
    "env",
    "time",
    "nice",
    "ionice",
    "nohup",
    "stdbuf",
    "xargs",
    "timeout",
    "command",
    "exec",
    "bash",
    "sh",
    "-c",
    # Job launchers whose own operands are flags and counts: `mpirun -n 2 …`, `srun
    # --gres=gpu:1 …`. Flags and numbers are already stepped over, so no special case.
    "mpirun",
    "srun",
}

#: Wrappers that put a SUBCOMMAND between themselves and the real command, so one token is
#: not enough: `uv run pytest -m gpu` stopped at `run`, which is not a blocked command, so the
#: device tier walked through the ordinary modern spelling of it. Valued by the subcommands
#: that mean "then run this" -- `uv pip install …` is not one, so `uv` still resolves to `uv`.
WRAPPER_SUBCOMMANDS = {
    # `run` only: `uv tool run pytest` is THREE tokens and `uvx` is a separate executable,
    # so listing `tool` here would imply coverage this does not give. Both are in the ledger
    # item that owns the rest of the launcher sweep.
    "uv": {"run"},
    "poetry": {"run"},
    "pipenv": {"run"},
    "pdm": {"run"},
    "hatch": {"run"},
    "conda": {"run"},
    "micromamba": {"run"},
    "rye": {"run"},
}

PYTHON_RE = re.compile(r"(?:^|/)(python|python3|python3\.\d+)$")

# Shell operators that end one command and begin the next.  Splitting on these
# with a plain regex would cut `python -c "import torch; print(...)"` in half at
# the semicolon *inside the quotes*, hiding the device call in a fragment that
# no longer parses -- so the split happens on lexer tokens, not on characters.
OPERATORS = {"&&", "||", "|", ";", "&", "\n"}

#: Redirections. Separated from the operators above because what follows one is a *file*,
#: not the next command: lumping them together made `cat > out.md` produce a segment whose
#: "executable" was `out.md`.
REDIRECTS = {">", ">>", "<", "<<", "2>", "2>>", "&>", ">&"}


#: Modules that run the suite or a benchmark when given to `python -m`.
BLOCKED_MODULES = frozenset({"pytest", "py.test"})
#: Whole package roots whose modules do the same — `python -m benchmarks.run_bench` is the
#: bench runner however deeply the module path is nested, so the root is what is matched.
BLOCKED_MODULE_ROOTS = frozenset({"benchmarks"})

#: Modules that only READ the paths they are handed -- `python -m black model.py` formats
#: the file, it never imports it, so a torch import inside is data, not execution. Kept
#: deliberately short: pytest and pip EXECUTE what they are given and must never join it.
READ_ONLY_TOOL_MODULES = frozenset({"black", "isort", "ruff"})

#: Sub-packages carved back out of the roots above. `benchmarks.parity` is the cross-plane
#: ingest parity harness: it imports numpy and `shipinfer.ingest`, drives a scripted source,
#: touches no device and produces no measurement — so the container rule has nothing to say
#: about it, and refusing the whole package was a false positive. Its emitter has since been
#: moved to `scripts/emit_parity_golden.py` precisely because this hook denied it, so nothing
#: under the package is an entry point today; the carve-out is the principled half of that
#: fix, and it is what keeps the next one from being moved out of the package too.
#:
#: A prefix and not a root, so `benchmarks.run_bench` beside it is still refused, and matched
#: with a trailing dot so a future `benchmarks.parity_bench` does not inherit it.
ALLOWED_MODULE_PREFIXES = ("benchmarks.parity",)


#: Runners whose *offline* use is allowed on a host, per ADR-001. Their device tiers are not.
_TEST_RUNNERS = frozenset({"pytest", "py.test"})

#: Marker expressions that select work needing an accelerator.
_DEVICE_MARKERS = ("multigpu", "gpu")


def _device_marker(args: list[str]) -> str:
    marker = _marker_expression(args)
    return f"-m {marker!r}" if marker else ""


def _marker_expressions(args: list[str]) -> list[str]:
    """Every ``-m`` value in the argv, in either spelling.

    All of them, not the first. ``python -m pytest -m gpu`` carries two: the interpreter's
    module and pytest's marker expression. Reading only the first found ``pytest`` and
    concluded the run was offline, so the device tier walked through.
    """
    found: list[str] = []
    for index, token in enumerate(args):
        if token == "-m":
            found.append(args[index + 1] if index + 1 < len(args) else "")
        elif token.startswith("-m") and len(token) > 2 and not token.startswith("--"):
            found.append(token[2:])
    return found


def _marker_expression(args: list[str]) -> str | None:
    """The last ``-m`` value, which for a pytest run is the marker expression."""
    values = _marker_expressions(args)
    return values[-1] if values else None


def _selects_device_tier(args: list[str]) -> bool:
    """Whether this pytest invocation asks for the GPU tiers.

    Read off ``-m`` rather than guessed. ``-m "not gpu"`` mentions the marker and selects
    the opposite, so a substring test would refuse the default offline run — the same
    mention-versus-selection error this file keeps having to relearn.
    """
    for expression in _marker_expressions(args):
        if "not gpu" in expression or "not multigpu" in expression:
            continue
        if any(marker in expression for marker in _DEVICE_MARKERS):
            return True
    return False


def _module_at(args: list[str]) -> tuple[int, str] | None:
    """``(index of the module's first operand, module name)``, or None.

    The one place that models CPython's option grammar, and it stops where CPython stops: at
    the FIRST non-option operand. Scanning the whole argv read a *script's* own `-m` as the
    interpreter's, so `python probe.py -m yolov8n` -- an ordinary shape for a probe -- looked
    like a module invocation and its program was never inspected (#174 review).
    """
    skip = False
    for index, token in enumerate(args):
        if skip:
            skip = False
            continue
        if token == "-m":
            return index + 2, (args[index + 1] if index + 1 < len(args) else "")
        if token.startswith("-m") and len(token) > 2 and not token.startswith("--"):
            return index + 1, token[2:]
        if token in _PYTHON_VALUE_FLAGS:
            # `-W ignore -m pytest -m gpu`: the value is not the operand that ends options.
            skip = True
            continue
        if not token.startswith("-"):
            return None
    return None


def _module_argument(args: list[str]) -> str | None:
    """The module in `-m pytest` or `-mpytest`, or None."""
    found = _module_at(args)
    return None if found is None else found[1]


def _is_containerised(tokens: list[str]) -> bool:
    """Whether this one segment is itself the containerised entry point.

    Matched against the segment's own executable and its first operands, never against the
    whole command: allowlisting on a substring let `echo "make test"; pytest tests/` pass.
    """
    if not tokens:
        return False
    head = " ".join(tokens[:3])
    exe = tokens[0].rsplit("/", 1)[-1]
    for marker in CONTAINERISED:
        first = marker.split()[0]
        if (exe == first or tokens[0].endswith(marker.split("/")[-1])) and head.startswith(
            (marker, exe)
        ):
            if marker.startswith(("docker", "podman", "nerdctl")):
                # `docker run …`: the operands matter, so compare the whole prefix. `make`
                # was here too and no `make` entry survives in CONTAINERISED -- there is no
                # Makefile in this repository and never was.
                return head.startswith(marker)
            return True
    return False


def in_container() -> bool:
    # `SHIPINFER_IN_CONTAINER=0` forces the host view even where a marker file
    # exists.  It only ever makes the hook stricter, which is why it is safe to
    # honour: it cannot be used to get a host run past the guard.  The suite
    # needs it because inside the container the hook is deliberately a no-op, so
    # without it the deny path would be untestable exactly where the rule says
    # tests must run.
    forced = os.environ.get("SHIPINFER_IN_CONTAINER")
    if forced == "0":
        return False
    if forced == "1":
        return True
    return any(Path(m).exists() for m in CONTAINER_MARKERS)


#: The start of a heredoc, and the word that ends it.  Its body is data, not
#: commands: a `python3 - <<'PY'` block whose script says `import torch` is one
#: python invocation, and reading its lines as commands would refuse it for
#: containing the word.
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")


def _join_continuations(command: str) -> str:
    """Fold ``\\``-continued lines back into one line before anything else looks at them.

    Lexing line by line is what closed the multi-line bypass, but a backslash continuation
    is not a new command -- it is the *same* command wrapped for readability. Splitting
    there refused a perfectly good `docker run ... \\` whose image and argv sat on the last
    line: the hook saw a bare `python3 scripts/build_engines.py` with the `docker run` that
    contained it three lines earlier, and blocked the sanctioned path. It caught me within
    minutes of writing it.
    """
    return re.sub(r"\\\n[ \t]*", " ", command)


def _split_heredocs(command: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Separate the command's own lines from its heredoc bodies.

    Returns ``(lines, [(opening_line, body), ...])``. Both halves are needed: the lines are
    commands, the bodies are data — except when the thing consuming a body is an interpreter
    reading stdin, in which case the body *is* the program. See :func:`_heredoc_runs_device`.
    """
    lines: list[str] = []
    bodies: list[tuple[str, str]] = []
    terminator: str | None = None
    opener = ""
    current: list[str] = []
    for line in _join_continuations(command).splitlines():
        if terminator is not None:
            if line.strip() == terminator:
                bodies.append((opener, "\n".join(current)))
                terminator, opener, current = None, "", []
            else:
                current.append(line)
            continue
        lines.append(line)
        match = HEREDOC.search(line)
        if match:
            terminator, opener = match.group(2), line
    if terminator is not None:
        bodies.append((opener, "\n".join(current)))  # unterminated, still worth inspecting
    return lines, bodies


def _command_lines(command: str) -> list[str]:
    """The command's lines with heredoc bodies removed."""
    return _split_heredocs(command)[0]


#: An interpreter told to read its program from stdin: `python3 - <<PY`, `bash -s <<SH`.
#: Only for these is a heredoc body executable rather than data — `cat > file <<EOF` writes
#: a file, and refusing that would refuse writing a test that mentions torch.
#: Interpreters whose heredoc body is a program rather than data.
_INTERPRETERS = re.compile(r"^(?:python3?(?:\.\d+)?|bash|sh|zsh)$")
#: The flags that mean "read the program from standard input".
_STDIN_FLAGS = frozenset({"-", "-s"})


def _stdin_interpreter(opening_line: str) -> str | None:
    """The interpreter that will *execute* the heredoc following this line, or None.

    Read as tokens rather than matched as a pattern. The regex this replaces had an
    optional-flags group that swallowed the very flag it then required, so `bash -s <<SH`
    — a shell reading its script from the heredoc — was never recognised. WHICH interpreter
    matters and not merely whether there is one: a python body and a shell body are different
    languages, and the same text means different things in them.
    """
    try:
        tokens = shlex.split(opening_line.split("<<")[0], comments=False)
    except ValueError:
        return None
    for index, token in enumerate(tokens):
        base = token.rsplit("/", 1)[-1]
        if base in {"sudo", "env", "exec", "nohup", "time"}:
            continue
        if _INTERPRETERS.match(base):
            if any(flag in _STDIN_FLAGS for flag in tokens[index + 1 :]):
                return base
            return None
        return None
    return None


def _reads_program_from_stdin(opening_line: str) -> bool:
    """Whether the heredoc following this line will be executed at all."""
    return _stdin_interpreter(opening_line) is not None


#: Module roots whose import means device work.
_DEVICE_ROOTS = frozenset(
    {"torch", "tensorrt", "cupy", "pycuda", "pynvml", "shipinfer.runtime"}
)


def _imports_device_stack(source: str) -> bool:
    """Whether ``source`` really imports a device stack, by parsing it.

    Falls back to the line-anchored regex when the body is not Python — a `bash -s` heredoc
    has no AST — and returns False when it is Python that does not parse, because refusing
    a half-typed script is the hook blocking on its own inability to read it.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return bool(DEVICE_IMPORT.search(source))
    except ValueError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            if root in _DEVICE_ROOTS or name.startswith("shipinfer.runtime"):
                return True
    return False


#: Modules whose names a body may bind, so an alias can be resolved back to the canonical
#: dotted path before it is matched. Which of their members actually run something is
#: `_SHELLING_OUT_CALL`'s question -- `subprocess.list2cmdline` builds a string and
#: `subprocess.CalledProcessError` is an exception, so not even `subprocess` is whole.
_SHELLING_OUT_MODULES = frozenset({"subprocess", "os", "pty", "runpy"})

#: The ENTRY POINTS that hand a string to a shell or exec a program. Qualifying on the module
#: ROOT instead made `os.path.exists("csrc/build/bench")` "shelling out", so a heredoc that
#: checks whether the baseline binary is built before telling the operator to build it was
#: refused -- the false-positive direction this whole change exists to repair.
_SHELLING_OUT_CALL = re.compile(
    r"^(?:subprocess\.(?:run|call|check_call|check_output|Popen|getoutput|getstatusoutput)"
    r"|os\.(?:system|popen|exec|spawn|posix_spawn)"
    r"|pty\.(?:spawn|fork)|runpy\.run_)"
)

#: Builtins that run source they are handed. Their payload is nested rather than positional
#: (`exec(open(P).read())`), so `_first_words` walks a call to one of these.
_SOURCE_BUILTINS = frozenset({"exec", "eval"})

#: The KEYWORDS that are a command position. Reading every keyword made `input=`, `cwd=` and
#: `encoding=` into commands, so posting a PR comment through `gh pr comment --body-file -`
#: from a python heredoc was refused for quoting `pytest -m gpu` -- verbatim the incident this
#: change exists to fix, through a different door. Everything else in a signature is data.
_COMMAND_KEYWORDS = frozenset({"args", "cmd", "executable", "path", "mod_name"})


def _bound_names(tree: ast.Module) -> dict[str, str]:
    """Name bound in *this body* -> the canonical dotted path it stands for.

    `import subprocess as sp` binds `sp` to `subprocess`; `from subprocess import run` binds
    `run` to `subprocess.run`. Without this, matching the unparsed callee missed both, and
    both are ordinary.
    """
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _SHELLING_OUT_MODULES:
                    bound[alias.asname or alias.name.split(".")[0]] = alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and (node.module or "") in (
            _SHELLING_OUT_MODULES
        ):
            for alias in node.names:
                bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bound


def _is_shelling_out(func: ast.expr, bound: dict[str, str]) -> bool:
    """Whether this callee actually runs something, after resolving the body's own aliases."""
    target = ast.unparse(func)
    if target in _SOURCE_BUILTINS:
        return True
    head, _, rest = target.partition(".")
    if head in bound:
        target = bound[head] + ("." + rest if rest else "")
    return bool(_SHELLING_OUT_CALL.match(target))


def _blocked_word(commands: list[list[str]]) -> str | None:
    """The first of ``commands`` that runs something blocked, or None.

    COMMANDS and not words, because the offline-tier carve-out needs a command's own
    arguments: `verdict` allows a `pytest` that selects no device tier (ADR-001, what CI does
    on a plain runner), and reading only the executable dropped that -- so the identical
    string was allowed typed and refused quoted, which is this change's own bug. Basename for
    `BLOCKED_COMMANDS`, substring for `BLOCKED_SCRIPTS`, as the executable check reads them.
    """
    for tokens in commands:
        if not tokens:
            continue
        exe, rest = tokens[0], tokens[1:]
        base = exe.rsplit("/", 1)[-1]
        if base in BLOCKED_COMMANDS:
            if base in _TEST_RUNNERS and not _selects_device_tier(rest):
                continue
            return base
        if any(script in exe for script in BLOCKED_SCRIPTS):
            return exe
        # `python -m pytest -m gpu` inside a body is the device tier just as much as at the
        # prompt, and `python -m pytest tests/core` is the tier ADR-001 exempts.
        if (PYTHON_RE.search(base) or base == "python") and _selects_device_tier(rest):
            module = _module_argument(rest)
            if module in BLOCKED_MODULES:
                return module
    return None


def _one_command(tokens: list[str]) -> list[list[str]]:
    """One argv, with wrappers stepped over, as the commands it really runs.

    A `python <program>` head yields the interpreter's command AND the program's -- #174's
    rule, inside a string. That matters for the one path where the hook is the only guard:
    `grep -rn require_container benchmarks/` finds `stages.py` and `kernels.py` only, so
    `run_bench.py` and `harness/` reach `containment` nowhere.
    """
    exe, rest = real_command(tokens)
    if exe is None:
        return []
    out = [[exe, *rest]]
    base = exe.rsplit("/", 1)[-1]
    if (PYTHON_RE.search(base) or base == "python") and _module_at(rest) is None:
        program = next((t for t in rest if not t.startswith("-")), None)
        if program is not None:
            out.append([program])
    return out


def _shell_commands(text: str) -> list[list[str]]:
    """Every command in a SHELL string, split by the LEXER rather than by a regex.

    `segments` is the quote-aware splitter this file already has, and the comment above
    `OPERATORS` says exactly why a regex is wrong: it cuts at a separator inside quotes.
    Reproducing that here refused `echo "a; pytest -m gpu"` in a shell body -- while the list
    form of the same argument was correctly allowed one function up, which is the tell.
    """
    out: list[list[str]] = []
    for tokens in segments(text):
        if tokens:
            out.extend(_one_command(tokens))
    return out


def _first_words(call: ast.Call) -> list[list[str]]:
    """Every command ``call`` would run, from its string arguments and its keywords.

    COMMAND POSITION, not every word: `subprocess.run(["echo", "pytest"])` echoes a word. A
    STRING argument is a shell line and splits on `;`/`&&`; a LIST argv is one command with no
    shell, so it does not.
    """
    if isinstance(call.func, ast.Name) and call.func.id in _SOURCE_BUILTINS:
        # `exec(open("benchmarks/run_bench.py").read())`: the payload is nested, not an
        # argument, so every string inside the call is a candidate program.
        return [
            command
            for node in ast.walk(call)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            for command in _shell_commands(node.value)
        ]
    out: list[list[str]] = []
    keywords = [kw.value for kw in call.keywords if kw.arg in _COMMAND_KEYWORDS]
    for arg in [*call.args, *keywords]:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.extend(_shell_commands(arg.value))
        elif isinstance(arg, (ast.List, ast.Tuple)) and arg.elts:
            # A LIST argv is one command and there is no shell, so every element after the
            # first is a literal argument -- splitting it on `;` fabricated a command position
            # and refused `subprocess.run(["echo", "a; pytest -m gpu"])`.
            words = [
                e.value
                for e in arg.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
            if words:
                out.extend(_one_command(words))
        elif isinstance(arg, ast.JoinedStr):
            for part in arg.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    out.extend(_shell_commands(part.value)[:1])
                    break
    return out


def _text_runs_blocked(source: str) -> str | None:
    """A blocked command in *command position* anywhere in a SHELL body, or None.

    One call, because `segments` splits by line as well as by operator -- and the whole body
    is read the way the same string inside `subprocess.run(..., shell=True)` is, since the two
    readings disagreeing is the bug in miniature.
    """
    return _blocked_word(_shell_commands(source))


def _python_runs_blocked(source: str) -> str | None:
    """A blocked command this PYTHON body actually invokes, or None.

    Parsed, for the same reason `_imports_device_stack` is: a name in a string literal is not
    an invocation. The text scan refused a heredoc whose body was a markdown table with a row
    beginning `pytest`, and refused the very comment reporting that -- four times in one
    session, each time killing the whole `Bash` call and the edit chained ahead of it.

    An unparseable body falls back to the text scan: half-typed python is not something to
    reason about, and the conservative direction there is to refuse.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return _text_runs_blocked(source)
    bound = _bound_names(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_shelling_out(node.func, bound):
            continue
        found = _blocked_word(_first_words(node))
        if found is not None:
            return found
    return None


def _heredoc_runs_device(command: str) -> str | None:
    """A refusal when a heredoc body that will be *executed* reaches for an accelerator.

    The body is the program here, so it gets the same reading a script file gets. Bodies fed
    to anything else are left alone: writing a file whose text contains `import torch` is not
    running it, and treating it as such would make the hook unusable for ordinary editing.
    """
    for opener, body in _split_heredocs(command)[1]:
        interpreter = _stdin_interpreter(opener)
        if interpreter is None:
            continue
        # Parsed, not pattern-matched. The line-anchored regex fired on an import that
        # appeared inside a *string literal* — a heredoc whose python writes another file
        # containing that line — and refused the edit that was fixing this very check. An
        # AST cannot make that mistake: a string is not an import statement.
        if _imports_device_stack(body):
            return (
                "a heredoc executed by an interpreter imports a device stack. The body is "
                "the program, not a file being written."
            )
        # In a SHELL body a line's first word is a command; in a PYTHON body it is a name,
        # and only a `subprocess`/`os.system` call makes it an invocation. Reading both the
        # same way is the mistake the AST above was already introduced to fix, one loop down.
        if interpreter.startswith("python"):
            exe = _python_runs_blocked(body)
        else:
            exe = _text_runs_blocked(body)
        if exe is not None:
            return f"a heredoc executed by an interpreter runs `{exe}`."
    return None


def segments(command: str) -> list[list[str]]:
    """Split a shell command into segments and tokenise each one.

    Lexed **line by line**, which is the whole point.  ``shlex`` treats a newline
    as ordinary whitespace, so lexing the command whole collapses a multi-line
    script into one token run and the second line's command becomes the first
    line's argument.  That is not a cosmetic bug in both directions:

        echo starting
        pytest tests/ -m gpu

    lexed whole yields a single segment whose executable is ``echo`` and whose
    arguments merely *contain* the word ``pytest`` -- a silent bypass of the
    guard.  The same collapse also refuses innocent commands, by attributing a
    later line's file argument to an earlier ``python3``.

    A line that will not lex (an unbalanced quote, a half-typed command) is
    skipped rather than guessed at, and skipping only costs that one line.
    Failing open is right here: blocking real work over the hook's own parse bug
    is how a guard gets switched off.
    """
    out: list[list[str]] = []
    for line in _command_lines(command):
        if not line.strip():
            continue
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        try:
            tokens = list(lexer)
        except ValueError:
            continue

        current: list[str] = []
        expect_target = False
        for token in tokens:
            if expect_target:
                # The word after `>` is a file, not a command. Treating it as one made
                # `cat > "$S/out.md"` look like a segment whose executable is `$S/out.md`,
                # which the variable-in-command-position rule then refused — a redirection
                # into a path held in a variable is completely ordinary.
                expect_target = False
                continue
            if token in REDIRECTS:
                expect_target = True
                continue
            if token in OPERATORS:
                if current:
                    out.append(current)
                    current = []
                continue
            current.append(token)
        if current:
            out.append(current)
    return out


ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# `timeout 900`, `nice -n 5`, `stdbuf -oL`: a wrapper's own operands must be
# stepped over too, or the "executable" comes out as `900` and the real command
# behind it is never examined.  That was the miss on the one process that had
# actually leaked a CUDA context here.
WRAPPER_OPERAND = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")

#: Wrapper flags whose value is a NAME, which `WRAPPER_OPERAND` cannot step over: `conda run
#: -n myenv pytest` answered `myenv`. Keyed BY WRAPPER -- the same letters are booleans
#: elsewhere (`sudo -n`, `time -p`), and one global set consumed the command itself and
#: answered `gpu` (#177 review).
WRAPPER_VALUE_FLAGS = {
    "conda": {"-n", "--name", "-p", "--prefix"},
    "micromamba": {"-n", "--name", "-p", "--prefix"},
    "env": {"-u", "--unset", "-C", "--chdir"},
    "sudo": {"-u", "--user", "-g", "--group"},
    "srun": {"-n", "--ntasks", "-p", "--partition", "--gres"},
    "mpirun": {"-n", "-np", "--host", "--hostfile"},
}


def real_command(tokens: list[str]) -> tuple[str | None, list[str]]:
    """Strip env assignments and wrappers; return (executable, remaining args)."""
    i = 0
    #: The wrapper whose own flags we are currently inside, so a value-taking flag is only
    #: honoured for the wrapper that has it.
    owner = ""
    while i < len(tokens):
        tok = tokens[i]
        base = tok.rsplit("/", 1)[-1]
        # `uv run pytest`: two tokens, and only together. Checked before the plain wrapper
        # test so `uv pip install …` still resolves to `uv` rather than to `pip`.
        following = tokens[i + 1] if i + 1 < len(tokens) else ""
        if following in WRAPPER_SUBCOMMANDS.get(base, frozenset()):
            owner = base
            i += 2
            continue
        if tok in WRAPPER_VALUE_FLAGS.get(owner, frozenset()):
            i += 2
            continue
        skip = (
            (ENV_ASSIGN.match(tok) and not tok.startswith("-"))
            or base in WRAPPERS
            or tok.startswith("-")
            or WRAPPER_OPERAND.match(tok)
        )
        if skip:
            if base in WRAPPERS:
                owner = base
            i += 1
            continue
        return tok, tokens[i + 1 :]
    return None, []


#: Interpreter options that take a SEPARATE value, so the token after one is not the program.
#: `-c` is absent because it ends option processing and its code is read by `DEVICE_TOKENS`;
#: `-m` is absent because `_module_at` decides it, and the decision is not one-sided.
_PYTHON_VALUE_FLAGS = frozenset({"-W", "-X", "--check-hash-based-pycs"})

#: Modules that only READ their operand. Everything else is assumed to RUN it, because `-m`
#: ends CPython's option processing and not execution -- and the fail-OPEN direction is the
#: one that cost this file `-m cProfile`, then `-m unittest` and `-m torch.distributed.run`
#: at once. An allowlist of executors fixes instances; this fixes the class (#174 review).
_READER_MODULES = READ_ONLY_TOOL_MODULES | {"pytest", "py.test", "py_compile", "json.tool"}


def _inline_source(args: list[str]) -> str | None:
    """The source after `-c`, in either spelling, or None."""
    for index, token in enumerate(args):
        if token == "-m" or (token.startswith("-m") and len(token) > 2):
            return None  # `-m` ended option processing; a later `-c` is the module's
        if token == "-c":
            return args[index + 1] if index + 1 < len(args) else ""
        if token.startswith("-c") and len(token) > 2 and not token.startswith("--"):
            return token[2:]
    return None


def _script_programs(args: list[str]) -> list[str]:
    """The ``.py`` file this command would RUN, as a candidate list.

    No module: the first non-flag operand, and nothing after it, since what follows is that
    program's own argv -- a script's operands are DATA, which is what `_READER_MODULES` says
    for `python -m black <file>`. Any OTHER module is assumed to run its operand, so the scan
    goes past the module name skipping the module's own options. Several candidates there,
    because a bare `.py` option value looks exactly like a program (`-o out.py probe.py`);
    the caller reads them in order and judges the first that imports a device stack.
    """
    candidates: list[str] = []
    module = _module_at(args)
    if module is not None:
        start, name = module
        # The dotted name as well as the root: `json.tool` reads, `torch.distributed.run`
        # runs, and only the full name tells them apart.
        if name in _READER_MODULES or name.split(".")[0] in _READER_MODULES:
            return []
        for offset, token in enumerate(args[start:]):
            # A nested `-m` gets the SAME reader/executor decision the outer one got.
            # `break`ing here deferred to the module branch, which judges only
            # `BLOCKED_MODULES`, so `coverage run -m unittest probe.py` was judged by nobody.
            if token == "-m" or (token.startswith("-m") and len(token) > 2):
                return _script_programs(args[start + offset :])
            # Before the suffix test, or an option's VALUE wins: `--include=probe.py`
            # answered as the program, and it is not a readable file. `-c` is deliberately not
            # bailed on here -- the interpreter's cannot appear after `-m`, so a `-c` is the
            # executor's own (`pdb -c continue`, `trace -c`).
            if token.startswith("-"):
                continue
            if token.endswith(".py"):
                candidates.append(token)
        return candidates
    skip = False
    for token in args:
        if skip:
            skip = False
            continue
        if token == "-c":
            return []
        if token in _PYTHON_VALUE_FLAGS:
            skip = True
            continue
        if token.startswith("-"):
            continue
        return [token] if token.endswith(".py") else []
    return []


def script_touches_device(args: list[str], cwd: str | None) -> str | None:
    """If a python invocation RUNS a local script that imports a device stack, name it.

    Every candidate, until one imports a device stack. Stopping at the first READABLE one let
    an earlier run's own output file decide: `-m cProfile -o out.py probe.py` was refused once
    and allowed the next time. Only the PROGRAM is judged -- scanning every `.py` argument
    refused a linter for its INPUT's imports, and an offline `python -m pytest tests/<file>.py`
    for the same reason. Both are allowed by the rule, and a refusal kills the whole `Bash`
    call, so an edit chained before one never ran.
    """
    root = Path(cwd) if cwd else Path.cwd()
    for program in _script_programs(args):
        candidate = Path(program)
        path = candidate if candidate.is_absolute() else root / candidate
        try:
            if not path.is_file():
                continue  # `-o out.py probe.py`: an option's value, not the program
            body = path.read_text(errors="replace")[:SCRIPT_READ_LIMIT]
        except OSError:
            continue
        if DEVICE_IMPORT.search(body):
            return program
    return None


def _indirection(tokens: list[str]) -> str | None:
    """Refuse a segment whose real command is hidden behind a substitution or a variable.

    Targeted rather than blanket. Refusing every `$(...)` would refuse `docker run --user
    $(id -u)` and the container scripts themselves, and a hook that blocks the sanctioned
    path is a hook that gets switched off. So two specific shapes are refused:

    - the executable is a variable: `PYTEST=pytest; $PYTEST tests/` runs the suite and
      shows the lexer only `$PYTEST`;
    - a command substitution *in command position* whose own body names something blocked:
      `$(which pytest) tests/` is a test run wearing a hat.

    A substitution used as an argument is left alone — it cannot change which program runs.
    """
    if not tokens:
        return None
    head = tokens[0]
    if head.startswith("$") and head != "$":
        return (
            f"`{head}` puts a variable where the command goes, so the hook cannot see what "
            f"runs. Write the command out, or use the container script."
        )
    if head == "$" and len(tokens) > 1 and tokens[1] == "(":
        inner = []
        for token in tokens[2:]:
            if token == ")":
                break
            inner.append(token.rsplit("/", 1)[-1])
        hit = next(
            (t for t in inner if t in BLOCKED_COMMANDS or any(x in t for x in BLOCKED_SCRIPTS)),
            None,
        )
        if hit is not None:
            return f"`$(... {hit} ...)` resolves to a blocked command in command position."
    return None


def verdict(command: str, cwd: str | None = None) -> str | None:
    """Return a refusal reason, or None to allow."""
    if "SHIPINFER_ALLOW_HOST_RUN=1" in command:
        return None
    # Deliberately NOT `marker in command`. Allowlisting on a substring of the whole
    # command is the same mention-vs-invocation error this hook exists to avoid, on the
    # permissive side: `echo "make test"; pytest tests/ -m gpu` was allowed because the
    # words "make test" appeared somewhere in it. The allowlist is matched per segment,
    # against what that segment actually runs.

    heredoc = _heredoc_runs_device(command)
    if heredoc is not None:
        return heredoc

    for tokens in segments(command):
        if _is_containerised(tokens):
            continue
        exe, args = real_command(tokens)
        if exe is None:
            continue
        indirect = _indirection(tokens)
        if indirect is not None:
            return indirect
        # `bash -c "pytest tests/"` arrives as one quoted token once the wrapper
        # and its `-c` are stepped over.  Re-read it as a command rather than
        # treating the whole string as an executable name.
        if " " in exe.strip():
            nested = verdict(exe, cwd)
            if nested is not None:
                return nested
            continue
        base = exe.rsplit("/", 1)[-1]

        if base in BLOCKED_COMMANDS:
            if base in _TEST_RUNNERS and not _selects_device_tier(args):
                # The offline tier may run anywhere. That is ADR-001, it is what CI does on
                # a plain runner, and it is the promise that makes the pure layers
                # verifiable without a driver — so refusing it here contradicted the rule
                # rather than enforcing it, and it blocked verifying the in-process gate.
                # The device tiers are gated inside the pytest session, where no spelling
                # avoids them (`shipinfer.runtime.containment`).
                pass
            else:
                return f"`{base}` runs the data plane on the host."

        if any(script in exe for script in BLOCKED_SCRIPTS):
            return f"`{base}` is a test or benchmark runner."

        if base == "shipinfer" and args:
            sub = next((a for a in args if not a.startswith("-")), None)
            if sub in BLOCKED_SHIPINFER_SUBCOMMANDS:
                return f"`shipinfer {sub}` runs the server or a benchmark."

        if PYTHON_RE.search(base) or base == "python":
            joined = " ".join(args)
            # `_module_at` once, for both the name and the index the `shipinfer` branch
            # below needs -- it was called twice, which the review noted.
            found = _module_at(args)
            module = None if found is None else found[1]
            if module is not None:
                root = module.split(".")[0]
                if root in READ_ONLY_TOOL_MODULES:
                    # A formatter or linter inspects its arguments; nothing it is handed
                    # runs. Without this, `python -m black engine/model.py` was refused
                    # because the FILE imports torch (the false positive of 28 Aug).
                    continue
                # ANY `-m` value, not just the first. `python -m coverage run -m pytest
                # -m gpu` names `coverage` first, so reading one module let the device tier
                # straight through -- and this branch is already the place that judges a
                # module, so it is the place the second one belongs.
                runner = next(
                    (m for m in _marker_expressions(args) if m in BLOCKED_MODULES), None
                )
                if runner is not None and _selects_device_tier(args):
                    return f"`python -m {runner} {_device_marker(args)}` runs the device tier."
                # `python -m shipinfer serve` is `shipinfer serve`. The check above it only
                # ever fires when the EXECUTABLE is `shipinfer`, so the module spelling of the
                # same command -- what `python -m shipinfer` exists for -- was judged by
                # nobody, on the two subcommands that serve and measure.
                if root == "shipinfer":
                    sub = next(
                        (a for a in args[found[0] :] if not a.startswith("-")),
                        None,
                    )
                    if sub in BLOCKED_SHIPINFER_SUBCOMMANDS:
                        return f"`python -m {module} {sub}` runs the server or a benchmark."
                carved_out = any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in ALLOWED_MODULE_PREFIXES
                )
                if root in BLOCKED_MODULE_ROOTS and not carved_out:
                    return f"`python -m {module}` runs a benchmark on the host."
            # The PROGRAM, not the text. `python scripts/hooks/check_docs.py
            # benchmarks/run_bench.py` lints a runner; it does not run one. EVERY candidate,
            # which is the list's own contract: `-m cProfile -o prof.py run_bench.py` puts the
            # profiler's output file at [0], so reading only that missed the runner behind it.
            programs = _script_programs(args)
            if any(script in program for program in programs for script in BLOCKED_SCRIPTS):
                return "this invokes a test or benchmark runner."
            # A `-c` body is source, so it yields no candidates -- and `main`'s text match
            # covered it by accident. It is a python body in exactly the sense the heredoc
            # scan argues for, so it gets the same reading.
            inline = _inline_source(args)
            if inline is not None:
                ran = _python_runs_blocked(inline)
                if ran is not None:
                    return f"this inline python runs `{ran}`."
            if DEVICE_TOKENS.search(joined):
                return "this python reaches for an accelerator on the host."
            script = script_touches_device(args, cwd)
            if script is not None:
                return f"`{script}` imports a device stack (torch/tensorrt/...)."

    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # unreadable payload: never block on our own bug

    command = ((payload.get("tool_input") or {}).get("command")) or ""
    if not command or in_container():
        return 0

    reason = verdict(command, payload.get("cwd"))
    if reason is None:
        return 0

    detail = (
        f"Refused: {reason}\n\n"
        "Project rule (.claude/CLAUDE.md, 'Where commands run'): every test, "
        "benchmark and measurement runs inside the container. Host nvcc here is "
        "11.5 against a 12.6 driver, so host numbers are not production numbers.\n\n"
        "Run it in the container instead:\n"
        "  deploy/rootless/test.sh [pytest args]      # the suite\n"
        "  deploy/rootless/prove.sh                   # container + GPU attestation\n"
        "  deploy/rootless/run.sh <cmd>               # one command, or a shell\n\n"
        "If the operator has agreed this one may run on the host, prefix the "
        "command with SHIPINFER_ALLOW_HOST_RUN=1 and say so in the report."
    )
    json.dump(
        {
            "systemMessage": f"Blocked host run: {reason} (see .claude/CLAUDE.md)",
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": detail,
            },
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
