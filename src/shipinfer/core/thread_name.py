"""Give a thread's name to the OS as well as to Python. The mirror of `core/thread_name.h`.

`threading.Thread(name=...)` is a Python-level label: `/proc/<tid>/comm` still reads `python`
(measured), so `top -H`, a `gdb` thread list and per-thread CPU accounting see none of the six
names this package chooses -- and per-thread accounting is what
`NOT-GPU-BOUND-AT-FIVE-GPUS` left open. `_thread.set_name` arrives in CPython 3.14 and this
tree pins 3.10, so the route is `ctypes` into libc.

Stdlib only, because `core` is pure (ADR-001).
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import threading
from collections.abc import Callable, Sequence
from typing import Any

#: Linux caps a thread name at 16 bytes *including* the NUL, so fifteen are usable. The
#: kernel is the authority on this number: `pthread_setname_np` returns `ERANGE` for anything
#: longer and then sets nothing at all.
KERNEL_NAME_MAX = 15


def kernel_name(name: str) -> str:
    """``name`` cut to what the kernel will hold, keeping the head.

    The head, because every scheme here puts the class first and the discriminator second --
    except the model instances, which is why :func:`instance_thread_label` exists.
    """
    return name.encode("utf-8", "replace")[:KERNEL_NAME_MAX].decode("utf-8", "ignore")


def instance_thread_label(name: str) -> str:
    """The label for a model instance, whose name is ``model_ordinal_device``.

    THE DEVICE GOES IN THE HEAD, which inverts :func:`kernel_name`'s rule: here the head is
    the SHARED part, since `shipinfer-ship_detector_0_3` cut to fifteen is `shipinfer-ship_`
    for all four models -- and which DEVICE's instances are hot is the question.

    Produces the same string the C++ plane does for the same (model, device, index), so one
    `top -H` reads alike on both. The two spell the composite differently
    (`model_ordinal_device` here, `model:device:index` there); the label hides that.
    """
    head, _, device = name.rpartition("_")
    model, _, ordinal = head.rpartition("_")
    if not model or not device.isdigit() or not ordinal.isdigit():
        return kernel_name(f"mdl-{name}")
    return kernel_name(f"m{device}.{ordinal}-{model}")


def _resolve() -> Callable[[str], None] | None:
    """`pthread_setname_np`, with its prototype DECLARED, or None on a platform without it.

    Declared, and this is not defensive typing: `pthread_t` is pointer-sized and ctypes
    defaults a return type to `c_int`, so an undeclared `pthread_self()` truncates the handle
    and the call **segfaults the worker**. It did, in `tests/pipeline`, on the first draft of
    this module -- and a `try/except` cannot catch that, which is why the resolution happens
    once here rather than per thread.
    """
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=False)
        libc.pthread_self.restype = ctypes.c_void_p
        libc.pthread_self.argtypes = []
        libc.pthread_setname_np.restype = ctypes.c_int
        libc.pthread_setname_np.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    except (OSError, AttributeError, ValueError):  # pragma: no cover - platform dependent
        return None

    def name_it(name: str) -> None:
        libc.pthread_setname_np(libc.pthread_self(), name.encode("utf-8"))

    return name_it


#: Resolved once at import: a platform without the call gets `None` and every naming is then
#: a no-op, because a worker must not go down for a diagnostic.
_SET_NAME = _resolve()


def name_this_thread(name: str) -> None:
    """Name the calling thread for the OS. Must run ON that thread (`pthread_self`).

    Silent on failure: the Python-level name still works and the kernel one is a convenience
    for whoever is reading `top`.
    """
    if _SET_NAME is None:
        return
    # Suppressed rather than handled: the name is already bounded to fifteen bytes, so the
    # one documented failure (`ERANGE`) cannot happen -- and a diagnostic must not be able to
    # take a worker down whatever a platform does.
    with contextlib.suppress(OSError):
        _SET_NAME(kernel_name(name))


def start_thread(
    target: Callable[..., Any],
    *,
    name: str,
    kernel: str | None = None,
    daemon: bool = True,
    args: Sequence[Any] = (),
) -> threading.Thread:
    """``threading.Thread``, started, with the name given to the OS as well as to Python.

    One factory rather than a line in each target: two of the six cannot carry one --
    `pipeline.runner`'s worker takes no index and the HTTP thread's target is uvicorn's.

    Args:
        name: the Python-level name. Tests read these off `threading.enumerate()`.
        kernel: what the OS sees, when cutting ``name`` to fifteen would throw the
            discriminator away -- `pipeline-worker-12` becomes `pipeline-worker`.
    """
    os_name = kernel_name(kernel if kernel is not None else name)

    def run() -> None:
        name_this_thread(os_name)
        target(*args)

    thread = threading.Thread(target=run, name=name, daemon=daemon)
    thread.start()
    return thread
