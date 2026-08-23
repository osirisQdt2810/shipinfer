"""On-demand engine building: 200 lines of start-up path that had no tests.

Everything here is filesystem and naming logic, which is why it belongs in the offline tier
— the one thing it does *not* need is a GPU. `resolve_engine` even takes a `builder`
argument documented as "injected for testing", and nothing was injecting it.

Two of the bugs below were found by review and are pinned here because they are exactly the
kind that only bite in production: a lock left behind by a killed builder made every later
start-up wait 900 seconds and fail, and a cache key that could not name the GPU
architecture matched every machine that also could not.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from shipinfer.backends.tensorrt import autobuild
from shipinfer.core.errors import BackendLoadError


def onnx_at(directory: Path, name: str = "model.onnx", body: bytes = b"\x08\x07onnx") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(body)
    return path


class TestCacheName:
    """Everything that makes a plan valid has to be in its filename."""

    def _name(self, onnx: Path, **overrides) -> str:
        kwargs = {"trt_version": "10.14.1", "capability": "86", "fp16": False}
        kwargs.update(overrides)
        return autobuild.cache_name(onnx, **kwargs)

    def test_the_same_inputs_give_the_same_name(self, tmp_path: Path) -> None:
        onnx = onnx_at(tmp_path)
        assert self._name(onnx) == self._name(onnx)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"trt_version": "10.13.0"},
            {"capability": "89"},
            {"fp16": True},
        ],
    )
    def test_every_invalidating_input_changes_the_name(
        self, tmp_path: Path, overrides: dict
    ) -> None:
        """A plan built for sm_86 on TensorRT 10.14 is a different file from one built for
        sm_89 on 10.13. Naming them the same is how a stale artefact gets loaded."""
        onnx = onnx_at(tmp_path)
        assert self._name(onnx) != self._name(onnx, **overrides)

    def test_changing_the_onnx_changes_the_name(self, tmp_path: Path) -> None:
        first = onnx_at(tmp_path / "a", body=b"\x08\x07one")
        second = onnx_at(tmp_path / "b", body=b"\x08\x07two")
        assert self._name(first).split(".", 1)[1] != self._name(second).split(".", 1)[1]

    def test_the_precision_is_visible_in_the_name(self, tmp_path: Path) -> None:
        onnx = onnx_at(tmp_path)
        assert "fp32" in self._name(onnx)
        assert "fp16" in self._name(onnx, fp16=True)


class TestTheCapabilityIsRequired:
    """The cache key exists to stop a plan crossing architectures."""

    def test_an_undeterminable_capability_refuses_rather_than_guessing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It used to return "unknown", so an sm_86 plan cached as `smunknown` and matched
        every machine that also failed to introspect — including an sm_89 one."""
        import builtins

        real_import = builtins.__import__

        def refuse_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse_torch)

        with pytest.raises(BackendLoadError, match="compute capability"):
            autobuild._capability(0)


class TestAnAbandonedLockIsTakenOver:
    """A killed builder never reaches its own cleanup."""

    def test_a_lock_naming_a_dead_process_is_abandoned(self, tmp_path: Path) -> None:
        lock = tmp_path / "model.building"
        # A pid that cannot exist: the kernel's maximum plus one.
        lock.write_text("4194305\n")

        assert autobuild._lock_is_abandoned(lock)

    def test_a_lock_naming_this_live_process_is_not(self, tmp_path: Path) -> None:
        lock = tmp_path / "model.building"
        lock.write_text(f"{os.getpid()}\n")

        assert not autobuild._lock_is_abandoned(lock)

    def test_an_unreadable_lock_falls_back_to_its_age(self, tmp_path: Path) -> None:
        """A lock whose pid cannot be read is judged by how long it has been held. Younger
        than the timeout means a build may still be running."""
        lock = tmp_path / "model.building"
        lock.write_text("not-a-pid\n")

        assert not autobuild._lock_is_abandoned(lock)

        old = time.time() - autobuild._BUILD_TIMEOUT_S - 60
        os.utime(lock, (old, old))
        assert autobuild._lock_is_abandoned(lock)

    def test_a_missing_lock_counts_as_abandoned(self, tmp_path: Path) -> None:
        assert autobuild._lock_is_abandoned(tmp_path / "absent.building")


class TestResolveEngine:
    """Resolution order, and the failures it has to name."""

    def test_a_configured_plan_is_used_as_is(self, tmp_path: Path) -> None:
        """An operator who supplied a plan gets that plan, with no build and no ONNX."""
        (tmp_path / "model.plan").write_bytes(b"PLAN")

        resolved = autobuild.resolve_engine(
            tmp_path, engine_file="model.plan", trt=None, device_index=0
        )

        assert resolved == tmp_path / "model.plan"

    def test_neither_a_plan_nor_an_onnx_is_named_clearly(self, tmp_path: Path) -> None:
        """The message has to say what is missing *and* what to do; a bare "not found" on a
        start-up path costs someone an afternoon."""
        with pytest.raises(BackendLoadError) as caught:
            autobuild.resolve_engine(
                tmp_path, engine_file="model.plan", trt=None, device_index=0
            )

        message = str(caught.value)
        assert str(tmp_path) in message, "it names the directory it looked in"
        assert ".onnx" in message and "engine_file" in message, "and both remedies"

    def test_two_onnx_files_are_ambiguous_rather_than_arbitrary(self, tmp_path: Path) -> None:
        """Picking one would be a coin flip that produces a plan for the wrong model."""
        onnx_at(tmp_path, "first.onnx")
        onnx_at(tmp_path, "second.onnx")

        with pytest.raises(BackendLoadError, match="onnx"):
            autobuild.resolve_engine(
                tmp_path, engine_file="model.plan", trt=None, device_index=0
            )

    def test_a_built_plan_is_reused_on_the_next_call(self, tmp_path: Path) -> None:
        """Restarts are free; a second start-up must not rebuild."""
        onnx = onnx_at(tmp_path)
        builds: list[Path] = []

        def builder(source, destination, **kwargs):
            builds.append(Path(destination))
            Path(destination).write_bytes(b"PLAN")
            return Path(destination)

        class Trt:
            __version__ = "10.14.1"

        kwargs = {
            "engine_file": "model.plan",
            "trt": Trt(),
            "device_index": 0,
            "onnx_file": onnx.name,
            "builder": builder,
        }
        first = autobuild.resolve_engine(tmp_path, **kwargs)
        second = autobuild.resolve_engine(tmp_path, **kwargs)

        assert first == second
        assert len(builds) == 1, "the cached plan was rebuilt"

    def test_a_failed_build_leaves_no_partial_plan_and_no_lock(self, tmp_path: Path) -> None:
        """A half-written plan deserialises to None, which is a confusing way to fail."""
        onnx = onnx_at(tmp_path)

        def builder(source, destination, **kwargs):
            Path(destination).write_bytes(b"HALF")
            raise RuntimeError("builder exploded")

        class Trt:
            __version__ = "10.14.1"

        with pytest.raises(RuntimeError, match="exploded"):
            autobuild.resolve_engine(
                tmp_path,
                engine_file="model.plan",
                trt=Trt(),
                device_index=0,
                onnx_file=onnx.name,
                builder=builder,
            )

        assert not list(tmp_path.glob("*.partial")), "a partial plan survived"
        assert not list(tmp_path.glob("*.building")), "the lock survived the failure"
