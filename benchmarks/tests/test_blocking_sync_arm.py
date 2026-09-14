"""A flag-on arm has to prove the flag applied, in the run's own output.

Four design-load arms of the blocking-sync A/B (10 Sep) printed nothing about the knob two of
them were named after: `runtime/device.py` logs it at INFO and the harness configures WARNING.
The C++ arm has never had that problem because `csrc/shipinfer/cli/bench.cpp` *prints* it. So
this plane prints it too -- and refuses the arm the driver declined WHEN SOMEBODY ASKED for
the flag by name, because a requested-and-unapplied arm is the flag-off arm under another one.
Since the knob became the default, "nobody refused it" is not an ask: a `--sweep` rung past
the first gets `cudaErrorSetOnActiveProcess` on every device from its own predecessor's live
contexts, and aborting there would abort the climb rather than the arm.

Offline by design: the helper reads one property off whatever it is handed, so a double is a
two-attribute object and no device is involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from benchmarks.harness.shipinfer import _announce_blocking_sync
from shipinfer.runtime.device import BLOCKING_SYNC_ENV


@dataclass
class _Devices:
    blocking_sync: tuple[int, ...] = ()


@dataclass
class _Server:
    devices: _Devices = field(default_factory=_Devices)


class TestTheArmSaysWhatItIs:
    def test_the_devices_that_took_it_are_printed(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv(BLOCKING_SYNC_ENV, "1")

        _announce_blocking_sync(_Server(_Devices((0, 2))))

        assert "blocking synchronise on device(s) [0, 2]" in capsys.readouterr().out

    def test_an_arm_that_asked_and_applied_to_nothing_is_refused(self, monkeypatch) -> None:
        """Not a warning. The A/B's whole method is "one binary, one env var", and an arm that
        silently lost its only difference publishes a null result as a measurement."""
        monkeypatch.setenv(BLOCKING_SYNC_ENV, "1")

        with pytest.raises(RuntimeError, match="flag-off arm under another name"):
            _announce_blocking_sync(_Server())

    def test_the_default_arm_says_so_and_runs_on(self, monkeypatch, capsys) -> None:
        """The whole of review round 1's first finding. Nobody asked, so nothing applying is
        not a failed request: no visible CUDA device (reproduced in the container -- a
        CPU-only arm reaches it from a real server build), no loadable libcudart, a device
        that already has a context. Raising there aborted the run, and under `--sweep`
        `sweep_system` catches the RuntimeError and BREAKS the climb.
        """
        monkeypatch.delenv(BLOCKING_SYNC_ENV, raising=False)

        _announce_blocking_sync(_Server())

        out = capsys.readouterr().out
        assert "blocking synchronise on device(s) none" in out
        assert "synchronises by spinning" in out
        assert "was set" not in out, "nobody set it, and a message that says so misleads"

    def test_an_empty_value_is_the_default_arm_and_not_an_ask(self, monkeypatch) -> None:
        """`docker run -e SHIPINFER_CUDA_BLOCKING_SYNC` with the host variable unset forwards
        it EMPTY, which is how every containerised run arrives: the knob is on, and the run
        must not be refused for a flag the operator never named."""
        monkeypatch.setenv(BLOCKING_SYNC_ENV, "")

        _announce_blocking_sync(_Server())

    def test_it_is_silent_and_permissive_only_when_refused(self, monkeypatch, capsys) -> None:
        """A REFUSED arm prints nothing and does not care that no device took the flag. The
        default arm is no longer that arm: since the two-load measurement the knob is on unless
        refused, so an unset variable announces and is held to the same proof."""
        monkeypatch.setenv(BLOCKING_SYNC_ENV, "0")

        _announce_blocking_sync(_Server())

        assert capsys.readouterr().out == ""

    def test_an_unset_variable_is_the_default_arm_and_still_announces(
        self, monkeypatch, capsys
    ) -> None:
        monkeypatch.delenv(BLOCKING_SYNC_ENV, raising=False)

        _announce_blocking_sync(_Server(_Devices((0,))))

        assert "blocking synchronise on device(s) [0]" in capsys.readouterr().out


class TestItIsWiredIntoTheRunAndNotOnlyDefined:
    def test_the_only_server_the_harness_builds_is_announced(self) -> None:
        """The link a unit test cannot reach: `run_shipinfer` is where both topologies build
        their server -- the parent for `single`, each child for `fleet`."""
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / "harness" / "shipinfer.py"
        text = source.read_text(encoding="utf-8")

        built = text.split("server = InferenceServer(settings)")
        assert (
            len(built) == 2
        ), "one construction site, or this assertion is checking the wrong one"
        assert built[1].lstrip().startswith("_announce_blocking_sync(server)")
