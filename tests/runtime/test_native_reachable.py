"""The fused kernels must be reachable when they are built — three separate reasons they were not.

Found when the kernel tier finally ran: every kernel benchmark reported the native column skipped
and blamed a missing build. The build was fine; the loader could not see it, for three unrelated
reasons, and each has a test here that goes red against the old code.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

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
        monkeypatch.setattr(
            native.importlib,
            "import_module",
            lambda name: (
                extension if name == "shipvision._C" else importlib.import_module(name)
            ),
        )
        native.native_module.cache_clear()
        try:
            assert native.native_module() is extension
            assert not hasattr(package, "_C"), "the loader must not depend on the attribute"
        finally:
            native.native_module.cache_clear()

    def test_an_absent_extension_is_none_not_an_error(self, monkeypatch) -> None:
        def refuse(name):
            if name == "shipvision._C":
                raise ImportError("no build")
            return importlib.import_module(name)

        monkeypatch.setattr(native.importlib, "import_module", refuse)
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

    def test_is_available_alone_is_not_consulted(self) -> None:
        # An extension that only has the *wrong* name is one that does not say; the ops decide.
        assert native._reports_devices(_extension(is_available=lambda: False)) is True

    def test_a_string_attribute_is_read_not_called(self) -> None:
        assert native._describe(_extension(), "platform") == "cuda"


class TestNativeImageOpsUsesTheExtensionsSurface:
    def _fake(self, monkeypatch, extension) -> None:
        monkeypatch.setattr(native_ops, "require_native", lambda: extension)

    def test_a_build_without_the_probe_is_named(self, monkeypatch) -> None:
        extension = _extension()
        extension.ImageOps = lambda device: object()
        self._fake(monkeypatch, extension)

        with pytest.raises(RuntimeError, match="cuda_available"):
            native_ops.NativeImageOps(device_index=0)

    def test_a_build_with_no_usable_device_says_so(self, monkeypatch) -> None:
        extension = _extension(cuda_available=lambda: False)
        extension.ImageOps = lambda device: object()
        self._fake(monkeypatch, extension)

        with pytest.raises(RuntimeError, match="no usable GPU kernels"):
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
