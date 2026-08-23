"""Source-level guards on the CUDA/HIP layer.

Every rule here encodes a defect that actually shipped in this file set, and every one is
checkable by reading the source — no GPU, no compiler, so they run in the default tier
where a mistake is caught in seconds rather than on a production node.

The GPU-marked tests at the bottom cover the two things a grep cannot: that an odd-sized
frame does not trip the sticky misalignment error, and that consecutive batches of
*different* content do not bleed into each other through reused staging buffers.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

NATIVE = Path(__file__).resolve().parents[2] / "native"
PLATFORM = NATIVE / "include" / "shipinfer" / "platform.hpp"
SOURCES = sorted(
    [*(NATIVE / "src").glob("*.cu"), *(NATIVE / "bindings").glob("*.cpp")]
    + [p for p in (NATIVE / "include" / "shipinfer").glob("*.hpp") if p != PLATFORM]
)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def test_there_are_sources_to_check() -> None:
    """A guard that silently checks nothing is worse than no guard."""
    assert len(SOURCES) >= 4, f"expected the native tree, found {SOURCES}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_vendor_specific_runtime_call_outside_platform_hpp(path: Path) -> None:
    """The whole point of the gpu* aliases: one source tree, CUDA or HIP.

    A single raw ``cudaMalloc`` compiles fine on NVIDIA and breaks the ROCm build, which
    nobody notices until someone tries to build it.
    """
    body = _strip_comments(path.read_text())
    offenders = sorted(set(re.findall(r"\b(?:cuda|hip)[A-Z]\w*", body)))
    assert not offenders, (
        f"{path.name} calls the vendor API directly: {offenders}. "
        f"Use the gpu* aliases from platform.hpp, or the ROCm build stops compiling."
    )


def test_every_kernel_launch_checks_the_async_error_slot() -> None:
    """A launch reports configuration errors immediately and execution errors later.

    Without ``check_launch`` after each one, an out-of-bounds write surfaces as an
    unrelated failure in whatever call happens next — one of the hardest CUDA bugs to trace.
    """
    body = _strip_comments((NATIVE / "src" / "image_ops.cu").read_text())
    launches = [m.end() for m in re.finditer(r"<<<", body)]
    assert launches, "no kernel launches found; has the file moved?"
    for position in launches:
        following = body[position : position + 700]
        assert "check_launch(" in following, (
            "a kernel launch is not followed by check_launch(); an out-of-bounds write "
            "would then surface somewhere unrelated"
        )


def test_the_device_output_entry_points_do_not_synchronise() -> None:
    """The ``_into`` path exists to be asynchronous.

    An earlier version allocated its descriptor table per call, which forced a
    ``gpuStreamSynchronize`` before the temporary died — so every "async" call blocked
    until the kernel retired, and the next batch's upload could never overlap this one's
    compute. Reuse is made safe by the staging ring's events instead.
    """
    body = _strip_comments((NATIVE / "bindings" / "module.cpp").read_text())
    for name in ("letterbox_into", "crop_into"):
        start = (
            body.index(f"void ImageOps::{name}")
            if f"void ImageOps::{name}" in body
            else body.index(name)
        )
        # Up to the next member definition, which is close enough to bound the body.
        segment = body[start : start + 3000]
        assert "gpuStreamSynchronize" not in segment, (
            f"{name} synchronises; the fast path must not wait. If reuse needs ordering, "
            f"record an event on the staging slot instead."
        )


def test_the_staging_ring_records_an_event_for_every_slot_it_hands_out() -> None:
    """Reuse without ordering is a data race that identical test inputs cannot reveal.

    A slot's pinned buffer is overwritten by the next call's host memcpy, and its device
    frames by the next upload; both can still be in use. Every ``acquire`` must be paired
    with a ``record``.
    """
    body = _strip_comments((NATIVE / "bindings" / "module.cpp").read_text())
    assert body.count("ring_.acquire()") == body.count("slot.record("), (
        "every staging slot acquired must have an event recorded on it before the call "
        "returns, or the rotation reuses buffers that are still being read"
    )


def test_packed_allocations_align_their_float_view() -> None:
    """A frame is h*w*3 uint8 bytes, a multiple of 4 only by luck.

    A 1079x1919 frame is 6,211,803 bytes, so a float* placed straight after it is
    misaligned; CUDA then raises cudaErrorMisalignedAddress, which is sticky and poisons
    the context for the life of the process.
    """
    body = _strip_comments((NATIVE / "bindings" / "module.cpp").read_text())
    assert (
        "align_up(frame_bytes)" in body
    ), "the boxes share an allocation with the frame and must start at an aligned offset"
    assert "device + frame_bytes" not in body, "the unaligned offset is back"


def test_the_library_half_never_mentions_python() -> None:
    """``include/`` and ``src/`` are a plain C++/CUDA library.

    They transform data and run kernels on a CPU or a GPU; an interpreter is not their
    concern. Keeping Python out of them is what lets the same sources be linked into a C++
    service, tested without an interpreter, and reasoned about without asking which thread
    holds a lock.
    """
    offenders = []
    for path in [
        *(NATIVE / "include" / "shipinfer").glob("*.hpp"),
        *(NATIVE / "src").glob("*.cu"),
    ]:
        body = _strip_comments(path.read_text())
        hits = sorted(set(re.findall(r"\bpy::|\bpybind11\b|\bgil_\w+|\bPy_\w+", body)))
        if hits:
            offenders.append(f"{path.name}: {hits}")
    assert (
        not offenders
    ), "the library half must not know Python exists; move it to bindings/:\n" + "\n".join(
        offenders
    )


def test_each_entry_point_crosses_the_gil_exactly_once() -> None:
    """One transition per call, at the boundary — not scattered through helpers.

    An earlier version released the GIL in five places across nested helpers, so a single
    call crossed the boundary three times and no helper's body told you whether the lock
    was held on entry. That is how a ``py::`` access ends up on the wrong side of a
    release, and that failure is an interpreter crash rather than an exception.
    """
    body = _strip_comments((NATIVE / "bindings" / "module.cpp").read_text())
    entry_points = ["letterbox_into", "letterbox_batch", "crop_into", "crop_batch", "nms"]

    total = body.count("gil_scoped_release")
    assert total == len(entry_points), (
        f"expected one GIL release per public entry point ({len(entry_points)}), "
        f"found {total}; a release inside a helper makes the boundary unreadable"
    )
    assert "gil_scoped_acquire" not in body, (
        "re-acquiring the GIL mid-call means the released region touches Python; "
        "prepare the data before releasing instead"
    )


def _function_body(source: str, signature_fragment: str) -> str:
    """The braced body of the function whose signature contains ``signature_fragment``.

    Brace matching rather than a fixed window: a window either misses the tail of a long
    function or runs into the next one, and both failures are silent — the guard passes
    while checking the wrong text.
    """
    start = source.index(signature_fragment)
    opening = source.index("{", start)
    depth = 0
    for i in range(opening, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[opening : i + 1]
    raise AssertionError(f"unbalanced braces after {signature_fragment!r}")


def test_the_gil_free_helpers_touch_no_python_object() -> None:
    """The private ``run_*`` and ``download`` helpers execute with the GIL released.

    A ``py::`` access in one of them is undefined behaviour, and it would look perfectly
    ordinary in review — which is exactly why it is checked here rather than trusted.
    """
    source = _strip_comments((NATIVE / "bindings" / "module.cpp").read_text())
    for helper in ("run_letterbox", "run_crop", "run_nms", "download"):
        signature = (
            f"std::vector<int64_t> {helper}(" if helper == "run_nms" else f"void {helper}("
        )
        body = _function_body(source, signature)
        assert (
            "py::" not in body
        ), f"{helper} runs with the GIL released and must not touch a py:: object"
        assert "gil_scoped" not in body, (
            f"{helper} is already called with the GIL released; nesting a scope here is "
            f"the confusion this structure exists to remove"
        )


def test_the_prepare_helpers_run_with_the_gil_held() -> None:
    """The mirror of the rule above: ``plan_*`` read numpy and must not release."""
    source = _strip_comments((NATIVE / "bindings" / "module.cpp").read_text())
    # The DEFINITION, not the first call site — matching " plan_frames(" finds the call
    # inside letterbox_into, whose enclosing function does release, and the guard then
    # fails for the wrong reason.
    for signature in ("std::vector<FramePlan> plan_frames(", "CropPlan plan_crop("):
        helper = signature.split()[-1].rstrip("(")
        body = _function_body(source, signature)
        assert "gil_scoped" not in body, f"{helper} reads Python objects; it must not release"


# -- the parts a grep cannot check ----------------------------------------------------------


@pytest.mark.gpu
def test_an_odd_sized_frame_does_not_trip_the_alignment_error() -> None:
    """1079 x 1919 x 3 is 3 mod 4. Before the fix this raised cudaErrorMisalignedAddress
    and poisoned the context for every later call on that worker."""
    from shipinfer.runtime.ops import NativeImageOps, NormalizeParams

    ops = NativeImageOps(device_index=0)
    frame = np.random.default_rng(0).integers(0, 255, (1079, 1919, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 300, 300], [500, 400, 900, 800]], dtype=np.float32)

    crops = ops.crop_batch(frame, boxes, (64, 32), NormalizeParams())

    assert crops.shape == (2, 3, 64, 32)
    assert np.isfinite(crops).all()
    # The context must still be usable: a sticky error would fail this second call.
    assert ops.crop_batch(frame, boxes, (64, 32), NormalizeParams()).shape == (2, 3, 64, 32)


@pytest.mark.gpu
def test_consecutive_batches_do_not_bleed_through_reused_staging() -> None:
    """The race the event ring exists to prevent.

    Each batch has distinct constant content, so a slot reused before its DMA finished
    would return the *previous* batch's pixels — invisible to any benchmark that submits
    the same image twice, which is exactly how it went unnoticed.
    """
    import torch

    from shipinfer.runtime.ops import NativeImageOps, NormalizeParams, TorchImageOps

    params = NormalizeParams()
    native = NativeImageOps(device_index=0)
    reference = TorchImageOps(device_index=0)
    got = torch.empty((4, 3, 128, 128), dtype=torch.float32, device="cuda:0")
    want = torch.empty_like(got)

    for i in range(10):
        images = [np.full((240, 320, 3), (i * 7 + j) % 251, np.uint8) for j in range(4)]
        native.letterbox_to_device(images, got, params)
        actual = got.cpu().numpy().copy()
        reference.letterbox_to_device(images, want, params)
        np.testing.assert_allclose(actual, want.cpu().numpy(), atol=1e-5, err_msg=f"batch {i}")
