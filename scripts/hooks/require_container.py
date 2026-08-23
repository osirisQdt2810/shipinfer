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
    "run_tests.sh",
    "run_bench.py",
    "compare_baseline.py",
    "bench_baseline.py",
    "run_baseline.py",
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
OPERATORS = {"&&", "||", "|", ";", "&", "\n", ">", ">>", "<"}


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


def segments(command: str) -> list[list[str]]:
    """Split a shell command into segments and tokenise each one.

    Unparseable input (unbalanced quotes, mid-edit heredocs) yields nothing:
    the hook then allows the command.  Failing open is the right call here --
    guessing at a command we cannot read would block real work on a parse bug.
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return []

    out: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
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


def verdict(command: str, cwd: str | None = None) -> str | None:
    """Return a refusal reason, or None to allow."""
    if "SHIPINFER_ALLOW_HOST_RUN=1" in command:
        return None
    if any(marker in command for marker in CONTAINERISED):
        return None

    for tokens in segments(command):
        exe, args = real_command(tokens)
        if exe is None:
            continue
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
            return f"`{base}` runs the suite or the data plane on the host."

        if any(script in exe for script in BLOCKED_SCRIPTS):
            return f"`{base}` is a test or benchmark runner."

        if base == "shipinfer" and args:
            sub = next((a for a in args if not a.startswith("-")), None)
            if sub in BLOCKED_SHIPINFER_SUBCOMMANDS:
                return f"`shipinfer {sub}` runs the server or a benchmark."

        if PYTHON_RE.search(base) or base == "python":
            joined = " ".join(args)
            if "-m" in args:
                idx = args.index("-m")
                module = args[idx + 1] if idx + 1 < len(args) else ""
                if module in ("pytest", "py.test"):
                    return "`python -m pytest` runs the suite on the host."
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
