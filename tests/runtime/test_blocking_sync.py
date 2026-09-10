"""The blocking-sync knob, on the plane that owes it the C++ plane's measurement.

`THE-INSTANCE-THREADS-SPIN-ON-cudaStreamSynchronize` measured the trade on `csrc/`: CUDA's
default `cudaDeviceScheduleAuto` spins when the active contexts do not outnumber the logical
processors, and asking it to block instead HALVED the model-instance threads' host CPU at the
design load, freed 24% of the process's CPU, and raised events ~15%. This plane's instance
threads wait in the same place, so they take the same knob -- same name, same three parse
rules, same default (off).

Offline: the parse is a string, the wiring is a call, and the flag itself needs a driver. What
is asserted here is that the knob is inert unless asked for and applied BEFORE the first model
-- because the driver refuses `cudaSetDeviceFlags` once a device has a context.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shipinfer.runtime import device as device_module
from shipinfer.runtime.device import (
    BLOCKING_SYNC_ENV,
    blocking_sync_requested,
    prefer_blocking_sync,
)

ROOT = Path(__file__).resolve().parents[2]


class TestTheKnobIsReadTheSameWayOnBothPlanes:
    @pytest.mark.parametrize(
        ("value", "asked"),
        [(None, False), ("", False), ("0", False), ("1", True), ("yes", True)],
    )
    def test_the_three_rules(self, value: str | None, asked: bool) -> None:
        """Empty is the one worth stating: `docker run -e VAR` forwards an unset host variable
        as EMPTY, so empty has to mean "not asked for" or every container run flips it."""
        environ = {} if value is None else {BLOCKING_SYNC_ENV: value}

        assert blocking_sync_requested(environ) is asked

    def test_the_env_var_is_the_one_the_cpp_plane_reads(self) -> None:
        """One knob for two planes, so an operator sets one thing."""
        header = (ROOT / "csrc" / "shipinfer" / "cli" / "bench.cpp").read_text(encoding="utf-8")

        assert f'env_flag("{BLOCKING_SYNC_ENV}")' in header

    def test_the_flag_value_matches_the_cpp_planes_symbol(self) -> None:
        """`ctypes` has no header to read `cudaDeviceScheduleBlockingSync` from, so this plane
        carries the literal 0x04. The C++ plane uses the SYMBOL, and the two cannot be checked
        against each other by the compiler -- so they are checked here."""
        platform = (ROOT / "csrc" / "shipinfer" / "core" / "platform.h").read_text(
            encoding="utf-8"
        )

        assert "gpuDeviceScheduleBlockingSync cudaDeviceScheduleBlockingSync" in platform
        assert device_module._CUDA_DEVICE_SCHEDULE_BLOCKING_SYNC == 0x04


class TestItIsInertUnlessAskedFor:
    def test_no_device_is_touched_when_nothing_asks(self, monkeypatch) -> None:
        """The default has to be the behaviour every existing measurement was taken under."""
        monkeypatch.delenv(BLOCKING_SYNC_ENV, raising=False)

        assert blocking_sync_requested() is False

    def test_an_empty_device_list_calls_nothing(self, monkeypatch) -> None:
        """A CPU-only host asks for no devices, and the helper must not reach for libcudart to
        find that out -- `prefer_blocking_sync` runs from a start-up path that has to work on a
        machine with no driver.

        The assertion is on `loaded`, not on the return value: the first draft checked only
        that the result was empty, which it is whether or not libcudart was opened -- a test
        that could not fail.
        """
        loaded: list[str] = []

        def record(name: str) -> object:
            loaded.append(name)
            raise AssertionError(f"libcudart opened for an empty device list: {name}")

        monkeypatch.setattr(device_module.ctypes, "CDLL", record)
        monkeypatch.setattr(device_module.ctypes.util, "find_library", lambda _n: "libcudart")

        assert prefer_blocking_sync([]) == ()
        assert loaded == []

    def test_a_device_that_already_has_a_context_is_reported_not_raised(
        self, monkeypatch, caplog
    ) -> None:
        """`cudaErrorSetOnActiveProcess` (216) is what the driver answers when the flag arrives
        too late. A diagnostic knob must not be able to stop a server, so it is a warning that
        names the device -- and the return value says which devices actually took it."""

        class Refusing:
            def __init__(self) -> None:
                self.cudaSetDevice = _Fn(0)
                self.cudaSetDeviceFlags = _Fn(216)
                self.cudaGetDevice = _Out(0)

        monkeypatch.setattr(device_module.ctypes, "CDLL", lambda _name: Refusing())

        with caplog.at_level("WARNING"):
            assert prefer_blocking_sync([3, 4]) == ()

        assert "already has a context" in caplog.text
        assert "device 3" in caplog.text and "device 4" in caplog.text

    def test_a_missing_libcudart_is_a_warning_and_no_devices(self, monkeypatch, caplog) -> None:
        """ADR-001: this import path has to work on a machine with no driver at all."""

        def missing(_name: str) -> object:
            raise OSError("libcudart.so: cannot open shared object file")

        monkeypatch.setattr(device_module.ctypes, "CDLL", missing)

        with caplog.at_level("WARNING"):
            assert prefer_blocking_sync([0]) == ()

        assert "libcudart" in caplog.text


class TestItIsAppliedBeforeAnyDeviceHasAContext:
    """The wiring, which is the part that broke -- and the part no test covered.

    The first draft called this from `InferenceServer.start`, one constructor too late:
    `DeviceManager.__init__` validates by reading `memory_info(index)` for every visible
    device, `cudaMemGetInfo` runs under a device guard, and that INITIALISES the primary
    context. The driver then refuses `cudaSetDeviceFlags` (216), so the knob warned once per
    device and the threads kept spinning -- the operator sets it, sees nothing worth stopping
    for, and gets the behaviour the knob exists to remove.
    """

    def test_the_flag_lands_before_the_validation_that_takes_a_context(
        self, monkeypatch
    ) -> None:
        order: list[str] = []
        monkeypatch.setenv(BLOCKING_SYNC_ENV, "1")
        monkeypatch.setattr(
            device_module,
            "prefer_blocking_sync",
            lambda devices: order.append(f"flag{tuple(devices)}") or (),
        )
        monkeypatch.setattr(device_module, "device_count", lambda: 2)
        monkeypatch.setattr(
            device_module,
            "memory_info",
            lambda index: order.append(f"memory_info({index})") or (1, 2),
        )
        monkeypatch.setattr(device_module, "device_properties", lambda index: f"gpu{index}")

        device_module.DeviceManager()

        assert order[0] == "flag(0, 1)", order
        assert order[1:] == ["memory_info(0)", "memory_info(1)"], order

    def test_nothing_is_applied_when_the_knob_is_unset(self, monkeypatch) -> None:
        """The default path must not reach for libcudart at all -- ADR-001: this constructor
        runs on a machine with no driver."""
        reached: list[str] = []
        monkeypatch.delenv(BLOCKING_SYNC_ENV, raising=False)
        monkeypatch.setattr(
            device_module, "prefer_blocking_sync", lambda devices: reached.append("flag") or ()
        )
        monkeypatch.setattr(device_module, "device_count", lambda: 1)
        monkeypatch.setattr(device_module, "memory_info", lambda index: (1, 2))
        monkeypatch.setattr(device_module, "device_properties", lambda index: "gpu")

        device_module.DeviceManager()

        assert reached == []

    def test_both_planes_put_the_callers_device_back(self) -> None:
        """The Python plane learned this in review (#203 round 3) and the C++ plane carried the
        same walk for a week. Source-level, because a driverless tier cannot call
        `cudaGetDevice` -- weaker than the `finally` test above and worth having anyway: the
        two planes are meant to be the same seam, and a wart on one is a wart on both.
        """
        bench = (ROOT / "csrc" / "shipinfer" / "cli" / "bench.cpp").read_text(encoding="utf-8")
        guarded = bench.split('env_flag("SHIPINFER_CUDA_BLOCKING_SYNC")')[1]
        walk = guarded.split("std::printf")[0]

        assert "gpuGetDevice(&previous)" in walk, walk
        assert "gpuSetDevice(previous)" in walk.split("for (int device")[1], walk

    def test_the_server_no_longer_knows_about_the_flag(self) -> None:
        """It moved into the device layer, and `engine/pool.py` keeping a second call site is
        how the two would drift back apart."""
        pool = (ROOT / "src" / "shipinfer" / "engine" / "pool.py").read_text(encoding="utf-8")

        assert "blocking_sync" not in pool
        assert "prefer_blocking_sync" not in pool


class TestTheManagerRemembersWhatTookTheFlag:
    """A flag-on arm that applied to nothing is the flag-off arm, so the bench has to be able
    to ask. Rounds 2 and 3 of #203 were both that failure; nothing could have detected either.
    """

    def _manager(self, monkeypatch, applied: tuple[int, ...]) -> object:
        monkeypatch.setattr(device_module, "device_count", lambda: 2)
        monkeypatch.setattr(device_module, "memory_info", lambda index: (1, 2))
        monkeypatch.setattr(device_module, "device_properties", lambda index: f"gpu{index}")
        monkeypatch.setattr(device_module, "prefer_blocking_sync", lambda devices: applied)
        return device_module.DeviceManager()

    def test_it_names_the_devices_that_took_it(self, monkeypatch) -> None:
        monkeypatch.setenv(BLOCKING_SYNC_ENV, "1")

        assert self._manager(monkeypatch, (0, 1)).blocking_sync == (0, 1)

    def test_asked_for_and_refused_reads_empty_rather_than_absent(self, monkeypatch) -> None:
        """The distinction the caller cannot make for itself: the knob was requested and the
        driver said no, which looks exactly like a flag-off run from the outside."""
        monkeypatch.setenv(BLOCKING_SYNC_ENV, "1")

        assert self._manager(monkeypatch, ()).blocking_sync == ()

    def test_unasked_is_empty_and_never_calls_the_driver(self, monkeypatch) -> None:
        monkeypatch.delenv(BLOCKING_SYNC_ENV, raising=False)

        def refuse(devices: object) -> tuple[int, ...]:
            raise AssertionError("the default path must not reach for libcudart")

        monkeypatch.setattr(device_module, "device_count", lambda: 1)
        monkeypatch.setattr(device_module, "memory_info", lambda index: (1, 2))
        monkeypatch.setattr(device_module, "device_properties", lambda index: "gpu")
        monkeypatch.setattr(device_module, "prefer_blocking_sync", refuse)

        assert device_module.DeviceManager().blocking_sync == ()


class _Out:
    """`cudaGetDevice`'s shape: it answers through an out-parameter, which is the third place
    the #200 prototype trap applies -- an undeclared `POINTER(c_int)` argument is where a
    truncated pointer would go."""

    def __init__(self, current: int) -> None:
        self.current = current
        self.restype: object = None
        self.argtypes: object = None
        self.calls: list[tuple] = []

    def __call__(self, out: object) -> int:
        self.calls.append((out,))
        out._obj.value = self.current  # what `ctypes.byref(x)` hands a real library
        return 0


class _Fn:
    """A libcudart entry point double: records the prototype the caller declares."""

    def __init__(self, status: int) -> None:
        self.status = status
        self.restype: object = None
        self.argtypes: object = None
        self.calls: list[tuple] = []

    def __call__(self, *args: object) -> int:
        self.calls.append(args)
        return self.status


class TestThePrototypesAreDeclared:
    def test_both_entry_points_get_a_restype_and_argtypes(self, monkeypatch) -> None:
        """An undeclared `ctypes` call returns `c_int`, so a pointer-sized value is truncated
        -- and `#200` proved that segfaults a worker, which no `try/except` catches. So the
        declaration is the property, not a style choice."""

        class Recording:
            def __init__(self) -> None:
                self.cudaSetDevice = _Fn(0)
                self.cudaSetDeviceFlags = _Fn(0)
                self.cudaGetDevice = _Out(2)

        library = Recording()
        monkeypatch.setattr(device_module.ctypes, "CDLL", lambda _name: library)

        assert prefer_blocking_sync([5]) == (5,)

        for entry in (
            library.cudaSetDevice,
            library.cudaSetDeviceFlags,
            library.cudaGetDevice,
        ):
            assert entry.restype is not None, entry
            assert entry.argtypes is not None, entry
        assert library.cudaSetDeviceFlags.calls == [(0x04,)]
        # The device it was on comes back: the walk touches every visible GPU, and an A/B whose
        # flag-on arm ends up current on a different device measures two things at once.
        assert library.cudaSetDevice.calls == [(5,), (2,)], library.cudaSetDevice.calls
