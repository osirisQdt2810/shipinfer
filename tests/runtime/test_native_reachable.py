"""The fused kernels must be reachable when they are built — three separate reasons they were not.

Found when the kernel tier finally ran: every kernel benchmark reported the native column skipped
and blamed a missing build. The build was fine; the loader could not see it, for three unrelated
reasons, and each has a test here that goes red against the old code.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError
from shipinfer.runtime import native
from shipinfer.runtime.ops import NormalizeParams, native_ops


def _extension(**attrs):
    module = types.ModuleType("shipvision._C")
    module.__version__ = "0.1.0"
    module.platform = "cuda"
    module.device_count = lambda: 1
    for name, value in attrs.items():
        setattr(module, name, value)
    return module


class TestTheExtensionIsImportedByItsOwnName:
    """`import shipvision` followed by `shipvision._C` only works when something has *already*
    imported the submodule — a submodule becomes an attribute of its package as a side effect
    of being imported, and `shipvision/__init__.py` deliberately imports nothing eagerly."""

    def test_a_package_that_has_not_imported_its_extension_still_yields_it(
        self, monkeypatch
    ) -> None:
        package = types.ModuleType(
            "shipvision"
        )  # no `_C` attribute, as after `import shipvision`
        extension = _extension(cuda_available=lambda: True)
        monkeypatch.setitem(sys.modules, "shipvision", package)
        # Patch the seam, not the stdlib: `native.importlib` *is* `importlib`, so replacing its
        # attribute would change every lazy import in the process for the test's duration.
        monkeypatch.setattr(
            native, "importlib", types.SimpleNamespace(import_module=lambda name: extension)
        )
        native.native_module.cache_clear()
        try:
            assert native.native_module() is extension
            assert not hasattr(package, "_C"), "the loader must not depend on the attribute"
        finally:
            native.native_module.cache_clear()

    def test_an_absent_extension_is_none_not_an_error(self, monkeypatch) -> None:
        def refuse(name):
            raise ImportError("no build")

        monkeypatch.setattr(native, "importlib", types.SimpleNamespace(import_module=refuse))
        native.native_module.cache_clear()
        try:
            assert native.native_module() is None
        finally:
            native.native_module.cache_clear()


class TestTheProbeIsTheNameTheExtensionBinds:
    """The extension binds `cuda_available`; the loader read `is_available`, a name it never
    defined, and treated the missing probe as "assume usable" — so the disagreement produced no
    error here and an AttributeError two files away."""

    def test_cuda_available_false_means_no_usable_kernels(self) -> None:
        assert native._reports_devices(_extension(cuda_available=lambda: False)) is False

    def test_cuda_available_true_means_usable(self) -> None:
        assert native._reports_devices(_extension(cuda_available=lambda: True)) is True

    def test_a_build_without_the_probe_is_not_usable(self) -> None:
        # The ops refuse such a build, so the loader must not call it usable: one answer to
        # "is native usable", or the banner says fast path while the data plane runs torch.
        assert native._reports_devices(_extension(is_available=lambda: True)) is False

    def test_a_string_attribute_is_read_not_called(self) -> None:
        assert native._describe(_extension(), "platform") == "cuda"


class TestNativeImageOpsUsesTheExtensionsSurface:
    def _fake(self, monkeypatch, extension) -> None:
        monkeypatch.setattr(native_ops, "require_native", lambda: extension)

    def test_a_build_without_the_probe_is_named(self, monkeypatch) -> None:
        extension = _extension()
        extension.ImageOps = lambda device: object()
        self._fake(monkeypatch, extension)

        with pytest.raises(ConfigurationError, match="cuda_available"):
            native_ops.NativeImageOps(device_index=0)

    def test_a_build_with_no_usable_device_says_so(self, monkeypatch) -> None:
        extension = _extension(cuda_available=lambda: False)
        extension.ImageOps = lambda device: object()
        self._fake(monkeypatch, extension)

        with pytest.raises(ConfigurationError, match="no usable GPU kernels"):
            native_ops.NativeImageOps(device_index=0)

    def test_letterbox_batch_carries_the_extents_the_kernel_reports(self, monkeypatch) -> None:
        """The submodule returns four values since its first review added the applied extents;
        the two- and three-value unpacks raised ValueError on every call and went unnoticed
        because the path was unreachable. A dead path is where a contract change is invisible.
        """

        class _Ops:
            def __init__(self, device: int) -> None:
                self.device = device

            def letterbox_batch(self, images, dst_h, dst_w, mean, std, swap_rb, pad, stream):
                n = len(images)
                tensor = np.zeros((n, 3, dst_h, dst_w), dtype=np.float32)
                scales = np.full(n, 0.5, dtype=np.float32)
                pads = np.zeros((n, 2), dtype=np.float32)
                extents = np.array([[dst_h // 2, dst_w] for _ in range(n)], dtype=np.int32)
                return tensor, scales, pads, extents

        extension = _extension(cuda_available=lambda: True)
        extension.ImageOps = _Ops
        self._fake(monkeypatch, extension)
        ops = native_ops.NativeImageOps(device_index=0)

        result = ops.letterbox_batch(
            [np.zeros((10, 20, 3), dtype=np.uint8)], (8, 8), NormalizeParams()
        )

        assert result.tensor.shape == (1, 3, 8, 8)
        assert result.extents is not None and result.extents.tolist() == [[4, 8]]

    def test_letterbox_to_device_unpacks_the_three_values_the_kernel_returns(
        self, monkeypatch
    ) -> None:
        """The fast path — the one production uses. CI never checks the submodule out, so this
        double is what keeps a change in `letterbox_into`'s return shape visible offline."""

        class _Ops:
            def __init__(self, device: int) -> None:
                self.calls: list[tuple] = []

            def letterbox_into(
                self, images, ptr, nbytes, dst_h, dst_w, mean, std, swap_rb, pad, stream
            ):
                self.calls.append((ptr, nbytes, dst_h, dst_w, pad, stream))
                n = len(images)
                return (
                    np.full(n, 0.5, np.float32),
                    np.zeros((n, 2), np.float32),
                    np.zeros((n, 2), np.int32),
                )

        class _Out:  # duck-typed CUDA tensor: what the launch actually reads off it
            shape = (2, 3, 8, 8)

            def data_ptr(self) -> int:
                return 0x1000

            def numel(self) -> int:
                return 2 * 3 * 8 * 8

            def element_size(self) -> int:
                return 4

        extension = _extension(cuda_available=lambda: True)
        extension.ImageOps = _Ops
        self._fake(monkeypatch, extension)
        # The contract check wants a real CUDA tensor; it is not what this test is about.
        monkeypatch.setattr(native_ops, "check_device_output", lambda *args, **kwargs: None)
        ops = native_ops.NativeImageOps(device_index=0)

        scales, pads = ops.letterbox_to_device(
            [np.zeros((10, 20, 3), np.uint8)] * 2, _Out(), NormalizeParams()
        )

        assert scales.shape == (2,) and pads.shape == (2, 2)
        assert ops._ops.calls == [(0x1000, 2 * 3 * 8 * 8 * 4, 8, 8, 114, 0)]
