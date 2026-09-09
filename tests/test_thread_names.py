"""Both planes name every thread they start, and the C++ one only just learned how.

CLAUDE.md's sync rule says a per-frame seam must be the same seam on both planes. Python
named all six of its threads from the start; `csrc/` named none, so `top -H`, a backtrace and
`/proc/<tid>/comm` showed fifty-odd rows called `bench`. That is not cosmetic:
`NOT-GPU-BOUND-AT-FIVE-GPUS` measured the wall as host CPU and left "which threads spend it"
open, and per-thread accounting reads names.

One gap stays open and is recorded rather than asserted: `threading.Thread(name=...)` is a
Python-level label and does NOT reach `/proc/<tid>/comm`, so the Python plane is still
invisible to per-thread OS accounting. See `PYTHON-THREADS-ARE-UNNAMED-TO-THE-KERNEL`.

Scope: the Python half reads `Thread(` calls only, so a `ThreadPoolExecutor` is outside it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc" / "shipinfer"
SRC = ROOT / "src" / "shipinfer"

#: A thread body opens here. `std::thread(` is unambiguous; an `emplace_back` is not, so it
#: counts only on a container this file declares as `std::vector<std::thread>` -- otherwise a
#: future `vector<function<...>>` would be asked for a thread name it has no use for.
_THREAD_CTOR = re.compile(r"std::thread\s*\w*\s*\(\s*\[")
_THREAD_VECTOR = re.compile(r"std::vector\s*<\s*std::thread\s*>\s*(\w+)")

#: How far below the opening line to look. Four, because the body's first statement is the
#: naming and a wrapped lambda capture can push it down a line.
_WITHIN = 4


def _spawn_sites() -> list[tuple[Path, int, str]]:
    """Every line in `csrc/shipinfer/` that opens a thread body."""
    found: list[tuple[Path, int, str]] = []
    # Headers too: `csrc/` is header-heavy (`core/join_on_unwind.h`, `ingest/sink.h`), and a
    # thread spawned from one would otherwise be invisible to this ratchet and ship unnamed
    # with the suite green.
    for path in sorted([*CSRC.rglob("*.cpp"), *CSRC.rglob("*.h")]):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        vectors = set(_THREAD_VECTOR.findall(text))
        alternation = "|".join(re.escape(name) for name in sorted(vectors))
        emplace = (
            re.compile(rf"\b(?:{alternation})\.emplace_back\s*\(\s*\[") if vectors else None
        )
        for index, line in enumerate(lines):
            if _THREAD_CTOR.search(line) or (emplace and emplace.search(line)):
                found.append((path, index, "\n".join(lines[index : index + _WITHIN])))
    return found


def test_the_detector_still_finds_the_threads_this_plane_starts() -> None:
    """Guards the regexes above: a rule that silently matches nothing passes on everything."""
    sites = _spawn_sites()
    files = {path.relative_to(CSRC).as_posix() for path, _, _ in sites}
    assert files >= {
        "cli/bench.cpp",
        "engine/instance.cpp",
        "ingest/camera/actor.cpp",
        "obs/sampler.cpp",
    }, files
    assert len(sites) >= 5, [f"{p}:{i + 1}" for p, i, _ in sites]


def _first_statement(body: str) -> str:
    """The body's first line that is neither blank nor a comment.

    The window's FIRST statement and not merely a line within it: a naming placed after a
    blocking call three lines in would satisfy "present" while leaving the thread anonymous
    for exactly the interval you wanted the name for. A lambda whose captures wrap would trip
    this on its continuation line -- a refusal in the strict direction, and there is none today.
    """
    for line in body.splitlines()[1:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            return stripped
    return ""


def test_every_cpp_thread_names_itself_first() -> None:
    """`pthread_setname_np` takes `pthread_self()`, so the naming has to be IN the body -- and
    first, because a thread that stalls before naming itself is the one you wanted the name
    for."""
    wrong = [
        f"{path.relative_to(ROOT)}:{index + 1} starts with {_first_statement(body)!r}"
        for path, index, body in _spawn_sites()
        if "name_this_thread" not in _first_statement(body)
    ]
    assert not wrong, f"thread bodies that do not name themselves first: {wrong}"


def test_every_python_thread_carries_a_name() -> None:
    """The property the Python plane already had, now a ratchet so it stays true."""
    unnamed: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            name = (
                target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
            )
            if name != "Thread":
                continue
            if not any(keyword.arg == "name" for keyword in node.keywords):
                unnamed.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not unnamed, f"threads started without a name: {unnamed}"


#: Linux caps a thread name at 16 bytes including the NUL, and `core/thread_name.h` keeps
#: the head. The kernel is the authority on this number, not us.
_BUDGET = 15


def test_a_model_instances_label_is_built_from_the_name_the_runtime_passes() -> None:
    """The wiring, because the truncation cannot be checked honestly from here.

    This file's first draft truncated `f"mdl-{model}"` over the repository's directory names --
    a string the runtime never builds. `bench.cpp` passes `model:device:index`, so
    `ship_detector:0:0` and `ship_detector:4:0` both cut to `mdl-ship_detect` and twenty
    threads on five GPUs shared four names, green throughout. The distinctness is enumerated
    in `csrc/tests/test_thread_name.cpp` through the real function (4 models x 8 devices x 2
    instances = 64 labels); what is asserted here is that `instance.cpp` still CALLS it.
    """
    instance = (CSRC / "engine" / "instance.cpp").read_text(encoding="utf-8")
    assert "name_this_thread(instance_thread_label(name_))" in instance, (
        "the model instance's thread name no longer goes through `instance_thread_label`, "
        "which is the only place the device index is kept out of the truncated tail"
    )
    bench = (CSRC / "cli" / "bench.cpp").read_text(encoding="utf-8")
    assert 'spec.name + ":" + std::to_string(device) + ":" + std::to_string(i)' in bench, (
        "the composite instance name changed shape; `instance_thread_label` parses it, and "
        "the C++ test enumerates that shape -- both need to move with it"
    )


@pytest.mark.parametrize(
    ("prefix", "discriminator"),
    [("cam", "camera-0049"), ("pipe", "127"), ("sweeper", ""), ("sampler", "")],
)
def test_the_names_that_do_fit_are_not_truncated_at_all(
    prefix: str, discriminator: str
) -> None:
    """Only the model names need the truncation. A camera id, a worker index and the two
    singletons fit whole, which is what makes `top -H` readable without a decoder ring."""
    name = f"{prefix}-{discriminator}" if discriminator else prefix
    assert len(name) <= _BUDGET, name
