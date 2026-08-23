"""``shipinfer doctor`` — what this host can actually run."""

from __future__ import annotations

from shipinfer.cli.common import console, print_table
from shipinfer.runtime.native import is_native_available, native_version
from shipinfer.runtime.platform import describe, device_count, device_properties, memory_info

__all__ = ["doctor"]


def doctor() -> int:
    """Report devices, the CUDA provider and the native data plane.

    The first command to run on a new box, and the first thing to paste into a bug report.
    Most "why is it slow" questions are answered by two lines of this output: which CUDA
    provider was chosen, and whether the native extension was found.
    """
    out = console()

    out.print(f"[bold]Accelerator[/bold]: {describe()}")
    native = (
        f"yes ({native_version()})"
        if is_native_available()
        else "no — data plane runs in Python"
    )
    out.print(f"[bold]Native extension[/bold]: {native}")

    count = device_count()
    if count == 0:
        out.print("\nNo CUDA devices. CPU backends only; GPU-marked tests will be skipped.")
        return 0

    rows = []
    for index in range(count):
        props = device_properties(index)
        free, total = memory_info(index)
        major, minor = props.compute_capability
        rows.append(
            [
                str(index),
                props.name,
                f"sm_{major}{minor}",
                f"{props.multi_processor_count}",
                f"{free // (1 << 20)} / {total // (1 << 20)} MiB",
            ]
        )
    print_table("Devices", ["idx", "name", "arch", "SMs", "free / total"], rows)
    return 0
