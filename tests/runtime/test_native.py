"""The native-extension probes, which have to be safe to call when there is no extension.

`runtime/native.py` is the only place in `shipinfer` that touches `shipvision._C`, and every
function in it is called from a start-up path — `InferenceServer.start()` logs the version and
`health()` reports it. So a probe that raises does not degrade a feature, it stops the server.
"""

from __future__ import annotations


class TestTheVersionProbeCannotTakeOutStartup:
    """`native_version()` called `module.version()`, which the extension has never defined —
    `csrc/bindings/module.cpp` sets `__version__` and nothing else.

    It was unreachable for as long as nothing in `shipinfer` put `shipvision._C` into
    `sys.modules`, so `native_module()` returned None and the line never ran. The moment the
    tracking stage imported `shipvision.tracking` at module scope it became reachable, and
    `InferenceServer.start()` died with `AttributeError` on every host where the submodule is
    built — the production container. CI does not catch it because CI deliberately does not
    check the submodule out (ADR-001), which is exactly the gap this test closes.
    """

    def test_it_reads_the_attribute_the_extension_actually_sets(self, monkeypatch) -> None:
        from shipinfer.runtime import native

        class _Extension:
            __version__ = "0.1.0"

        monkeypatch.setattr(native, "native_module", lambda: _Extension())

        assert native.native_version() == "0.1.0"

    def test_an_extension_with_no_version_is_reported_not_fatal(self, monkeypatch) -> None:
        """A version is a diagnostic. An extension that does not announce one is a fact to
        report, not a reason to refuse to start — which is what the old code did."""
        from shipinfer.runtime import native

        class _Silent:
            pass

        monkeypatch.setattr(native, "native_module", lambda: _Silent())

        assert native.native_version() is None

    def test_no_extension_is_none(self, monkeypatch) -> None:
        from shipinfer.runtime import native

        monkeypatch.setattr(native, "native_module", lambda: None)

        assert native.native_version() is None

    def test_calling_version_as_a_method_would_be_caught(self, monkeypatch) -> None:
        """The regression itself: an extension that exposes only `__version__` — which is
        every build of it — must not be asked for `version()`."""
        from shipinfer.runtime import native

        class _RealShape:
            __version__ = "0.1.0"

            def __getattr__(self, name: str):
                raise AttributeError(f"module 'shipvision._C' has no attribute {name!r}")

        monkeypatch.setattr(native, "native_module", lambda: _RealShape())

        assert native.native_version() == "0.1.0"
