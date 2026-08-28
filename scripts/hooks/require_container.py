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
    "deploy/rootless/prove.sh",
    "deploy/rootless/gst-image.sh",
    "deploy/rootless/wheels.sh",
    "make shell",
    "make test",
    "make bench",
)

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


def _module_argument(args: list[str]) -> str | None:
    """The module in `-m pytest` or `-mpytest`, or None.

    Both spellings, because only the detached form was checked and `python -mpytest tests/`
    went straight through.
    """
    for index, token in enumerate(args):
        if token == "-m":
            return args[index + 1] if index + 1 < len(args) else ""
        if token.startswith("-m") and len(token) > 2 and not token.startswith("--"):
            return token[2:]
    return None


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
            if marker.startswith(("docker", "podman", "nerdctl", "make")):
                # `docker run …`, `make test`: the operands matter, so compare the prefix.
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


def _reads_program_from_stdin(opening_line: str) -> bool:
    """Whether the command on this line will *execute* the heredoc that follows it.

    Read as tokens rather than matched as a pattern. The regex this replaces had an
    optional-flags group that swallowed the very flag it then required, so `bash -s <<SH`
    — a shell reading its script from the heredoc — was never recognised.
    """
    try:
        tokens = shlex.split(opening_line.split("<<")[0], comments=False)
    except ValueError:
        return False
    for index, token in enumerate(tokens):
        base = token.rsplit("/", 1)[-1]
        if base in {"sudo", "env", "exec", "nohup", "time"}:
            continue
        if _INTERPRETERS.match(base):
            return any(flag in _STDIN_FLAGS for flag in tokens[index + 1 :])
        return False
    return False


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


def _heredoc_runs_device(command: str) -> str | None:
    """A refusal when a heredoc body that will be *executed* reaches for an accelerator.

    The body is the program here, so it gets the same reading a script file gets. Bodies fed
    to anything else are left alone: writing a file whose text contains `import torch` is not
    running it, and treating it as such would make the hook unusable for ordinary editing.
    """
    for opener, body in _split_heredocs(command)[1]:
        if not _reads_program_from_stdin(opener):
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
        for line in body.splitlines():
            exe = line.strip().split(" ")[0].rsplit("/", 1)[-1]
            if exe in BLOCKED_COMMANDS:
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


def real_command(tokens: list[str]) -> tuple[str | None, list[str]]:
    """Strip env assignments and wrappers; return (executable, remaining args)."""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        base = tok.rsplit("/", 1)[-1]
        skip = (
            (ENV_ASSIGN.match(tok) and not tok.startswith("-"))
            or base in WRAPPERS
            or tok.startswith("-")
            or WRAPPER_OPERAND.match(tok)
        )
        if skip:
            i += 1
            continue
        return tok, tokens[i + 1 :]
    return None, []


def script_touches_device(args: list[str], cwd: str | None) -> str | None:
    """If a python invocation targets a local script that imports a device stack,
    name that script.  Unreadable or missing targets return None -- the hook
    never blocks on a file it could not inspect."""
    root = Path(cwd) if cwd else Path.cwd()
    for arg in args:
        if arg.startswith("-") or not arg.endswith(".py"):
            continue
        candidate = Path(arg)
        path = candidate if candidate.is_absolute() else root / candidate
        try:
            if not path.is_file():
                continue
            body = path.read_text(errors="replace")[:SCRIPT_READ_LIMIT]
        except OSError:
            continue
        if DEVICE_IMPORT.search(body):
            return arg
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
            module = _module_argument(args)
            if module is not None:
                root = module.split(".")[0]
                if root in READ_ONLY_TOOL_MODULES:
                    # A formatter or linter inspects its arguments; nothing it is handed
                    # runs. Without this, `python -m black engine/model.py` was refused
                    # because the FILE imports torch (the false positive of 28 Aug).
                    continue
                if module in BLOCKED_MODULES and _selects_device_tier(args):
                    return f"`python -m {module} {_device_marker(args)}` runs the device tier."
                if root in BLOCKED_MODULE_ROOTS:
                    return f"`python -m {module}` runs a benchmark on the host."
            if any(script in joined for script in BLOCKED_SCRIPTS):
                return "this invokes a test or benchmark runner."
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
        "  make shell                                 # interactive\n\n"
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
