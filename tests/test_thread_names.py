"""Both planes name every thread they start, and both names reach the kernel.

CLAUDE.md's sync rule says a per-frame seam must be the same seam on both planes. Python
named all six of its threads from the start; `csrc/` named none, so `top -H`, a backtrace and
`/proc/<tid>/comm` showed fifty-odd rows called `bench`. That is not cosmetic:
`NOT-GPU-BOUND-AT-FIVE-GPUS` measured the wall as host CPU and left "which threads spend it"
open, and per-thread accounting reads names.

`threading.Thread(name=...)` is a Python-level label and does NOT reach `/proc/<tid>/comm`
(measured), which left the Python plane just as invisible; `core/thread_name.py` sets the OS
name through `ctypes`, since `_thread.set_name` is CPython 3.14 and this tree pins 3.10.

Scope: the Python half reads `Thread(`/`start_thread(` calls, so a `ThreadPoolExecutor` is
outside it.
"""

from __future__ import annotations

import ast
import ctypes
import re
import threading
from pathlib import Path

import pytest

from shipinfer.core.thread_name import (
    instance_thread_label,
    kernel_name,
    start_thread,
)

#: `SYS_gettid` on x86-64. `threading.get_native_id()` exists, but this file is about
#: what the KERNEL holds, so it asks the kernel for the id as well as for the name.
_SYS_GETTID = 186

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


def _thread_calls(name: str) -> list[tuple[Path, int, ast.Call]]:
    """Every call to ``name`` under `src/shipinfer/`, with where it is."""
    found: list[tuple[Path, int, ast.Call]] = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            called = (
                target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
            )
            if called == name:
                found.append((path, node.lineno, node))
    return found


def _thread_subclasses() -> list[tuple[Path, int, ast.ClassDef]]:
    """Every `threading.Thread` SUBCLASS under `src/shipinfer/`.

    A `ClassDef` is not a `Call`, so the scan above cannot see one -- and review found two
    live threads (`ResultReader`, `RingIngress`) that the ratchet was reporting as covered.
    A subclass cannot use the factory at all: it names itself in `run`.
    """
    found: list[tuple[Path, int, ast.ClassDef]] = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                named = (
                    base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                )
                if named == "Thread":
                    found.append((path, node.lineno, node))
    return found


def test_every_python_thread_goes_through_the_factory() -> None:
    """`threading.Thread` directly would keep the Python name and lose the kernel one.

    The factory exists because two of the targets cannot name themselves from inside:
    `pipeline.runner`'s worker takes no index and the HTTP thread's target is uvicorn's
    `Server.run`. `core/thread_name.py` keeps its own `threading.Thread` -- it IS the factory.
    """
    factory = (SRC / "core" / "thread_name.py").resolve()
    bare = [
        f"{path.relative_to(ROOT)}:{line}"
        for path, line, _ in _thread_calls("Thread")
        if path.resolve() != factory
    ]
    assert not bare, f"threads started outside `start_thread`: {bare}"


def test_every_thread_subclass_names_itself_in_run() -> None:
    """The hole the first draft of this file had, and it was worse than missing coverage.

    `_thread_calls` matches `ast.Call`, so a class whose BASE is `threading.Thread` was
    invisible: `ResultReader` and `RingIngress` (ADR-016's control channel, started by
    `spill/mesh.py`) both reported `python` to the kernel while the test above passed and its
    docstring claimed otherwise. A ratchet that keeps passing for a whole spelling records the
    property as enforced when it is not.
    """
    missing: list[str] = []
    for path, line, node in _thread_subclasses():
        run = next(
            (n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "run"), None
        )
        first = run.body[0] if run and run.body else None
        called = (
            getattr(first.value.func, "id", "")
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
            else ""
        )
        if called != "name_this_thread":
            missing.append(
                f"{path.relative_to(ROOT)}:{line} {node.name} starts with {called!r}"
            )
    assert not missing, f"`Thread` subclasses that do not name themselves first: {missing}"


def test_the_subclass_scan_finds_the_two_that_exist() -> None:
    """Guards the scan the way the C++ half guards its regexes: a rule that silently matches
    nothing passes on everything."""
    found = {node.name for _path, _line, node in _thread_subclasses()}
    assert {"ResultReader", "RingIngress"} <= found, found


def test_every_started_thread_carries_a_name() -> None:
    """The property the Python plane already had, now a ratchet on the factory."""
    unnamed = [
        f"{path.relative_to(ROOT)}:{line}"
        for path, line, node in _thread_calls("start_thread")
        if not any(keyword.arg == "name" for keyword in node.keywords)
    ]
    assert not unnamed, f"threads started without a name: {unnamed}"


def test_the_kernel_holds_the_name_the_factory_was_given() -> None:
    """The whole point, asserted against `/proc` rather than against the helper.

    `threading.Thread(name=...)` alone leaves `/proc/<tid>/comm` reading `python`, which is
    what made six named threads invisible to `top -H` and to per-thread CPU accounting.
    """
    seen: dict[str, str] = {}

    def body() -> None:
        tid = ctypes.CDLL("libc.so.6").syscall(_SYS_GETTID)
        seen["kernel"] = Path(f"/proc/self/task/{tid}/comm").read_text().strip()
        seen["python"] = threading.current_thread().name

    start_thread(body, name="shipinfer-ship_detector_0_3", kernel="m3.0-ship_detec").join()

    assert seen["python"] == "shipinfer-ship_detector_0_3", seen
    assert seen["kernel"] == "m3.0-ship_detec", seen


def test_the_two_planes_agree_on_a_model_instances_label() -> None:
    """Same (model, device, index) -> the same fifteen bytes on both planes, so one `top -H`
    reads the same. The composite differs -- `model_ordinal_device` here,
    `model:device:index` there -- and the label is where that stops mattering.
    """
    header = (CSRC / "core" / "thread_name.h").read_text(encoding="utf-8")
    assert (
        'return thread_name("m" + tail, name.substr(0, first))' in header
    ), "the C++ label changed shape; the two planes' strings would diverge"
    assert instance_thread_label("ship_detector_0_3") == "m3.0-ship_detec"
    assert instance_thread_label("ship_segmenter_1_7") == "m7.1-ship_segme"


def test_no_two_instances_share_a_kernel_name() -> None:
    """The 15-byte budget's one real failure mode, over the format the runtime builds:
    `engine/model.py` names an instance `{model}_{ordinal}_{device}`.
    """
    models = sorted(p.name for p in (ROOT / "model_repository").iterdir() if p.is_dir())
    assert len(models) >= 4, models

    labels = {
        instance_thread_label(f"{model}_{ordinal}_{device}")
        for model in models
        for device in range(8)
        for ordinal in range(2)
    }
    assert len(labels) == len(models) * 8 * 2, sorted(labels)
    assert all(len(label) <= _BUDGET for label in labels), sorted(labels)


def test_every_call_site_fits_without_truncation() -> None:
    """A `kernel=` that still needed cutting would be a discriminator thrown away -- which is
    what `pipeline-worker-12` -> `pipeline-worker` was, and why every site passes one."""
    for name in ("pipe-127", "sweeper", "sampler", "cam-camera-0049", "http", "ch7.31"):
        assert kernel_name(name) == name, name


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
