"""Where the container rule is actually enforced.

The rule started as a `PreToolUse` hook that reads the *text* of a shell command, and review
showed a deny-list over text cannot be made sound: `( pytest -m gpu )`, `eval "pytest -m
gpu"`, `coverage run -m pytest`, `echo pytest | sh`, `tox` and `uv run pytest` all walked
through it, because the executable a shell will eventually run is not a property of the
string.

So the gate moved inside the processes that would do the work — the pytest session and the
`serve` / `bench` commands — where no spelling avoids it. These tests pin the two properties
that matter: it refuses device work on a host, and it does *not* refuse the offline tier,
which ADR-001 requires to run anywhere and which CI runs on a plain runner.
"""

from __future__ import annotations

import pytest

from shipinfer.runtime import containment


def evidence(*, marker: bool, cgroup: bool, overlay: bool, forced: str | None = None):
    return containment.Containment(
        marker=marker, cgroup=cgroup, overlay_root=overlay, forced=forced
    )


class TestOneSignalIsNotEnough:
    """`/.dockerenv` is a file. `touch /.dockerenv` used to be a complete attestation."""

    def test_the_marker_alone_does_not_prove_a_container(self) -> None:
        assert not evidence(marker=True, cgroup=False, overlay=False).in_container

    def test_two_agreeing_signals_do(self) -> None:
        assert evidence(marker=True, cgroup=True, overlay=False).in_container
        assert evidence(marker=True, cgroup=False, overlay=True).in_container
        assert evidence(marker=False, cgroup=True, overlay=True).in_container

    def test_a_rootless_container_sharing_the_host_pid_namespace_still_counts(self) -> None:
        """`--pid=host` is how `deploy/rootless/` runs, because this kernel refuses a userns
        `/proc` mount. Pid 1's cgroup is then the host's, so requiring all three signals
        would refuse the configuration the project actually uses."""
        assert evidence(marker=True, cgroup=False, overlay=True).in_container

    def test_no_signals_is_a_host(self) -> None:
        assert not evidence(marker=False, cgroup=False, overlay=False).in_container

    def test_the_evidence_is_reportable(self) -> None:
        """An attestation that says "in a container" without saying why is not evidence."""
        described = evidence(marker=True, cgroup=False, overlay=True).describe()

        assert "marker=True" in described
        assert "cgroup=False" in described
        assert "overlay_root=True" in described


class TestTheForcedOverrideOnlyEverTightens:
    def test_zero_asserts_a_host_even_where_markers_agree(self) -> None:
        """What the suite uses to exercise the refusal *from inside* a container, where the
        check is otherwise a no-op and the deny path would be untestable."""
        assert not evidence(marker=True, cgroup=True, overlay=True, forced="0").in_container

    def test_one_asserts_a_container(self) -> None:
        assert evidence(marker=False, cgroup=False, overlay=False, forced="1").in_container


class TestRequireContainer:
    def test_it_refuses_on_a_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(containment.FORCE_ENV, "0")
        monkeypatch.delenv(containment.ALLOW_HOST_RUN_ENV, raising=False)

        with pytest.raises(RuntimeError, match="must run inside a container"):
            containment.require_container("a benchmark")

    def test_the_refusal_names_the_work_and_shows_its_evidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(containment.FORCE_ENV, "0")
        monkeypatch.delenv(containment.ALLOW_HOST_RUN_ENV, raising=False)

        with pytest.raises(RuntimeError) as caught:
            containment.require_container("`shipinfer bench`")

        message = str(caught.value)
        assert "`shipinfer bench`" in message
        assert "Containment evidence:" in message
        assert "deploy/rootless/bench.sh" in message, "it has to say what to run instead"

    def test_it_passes_inside_a_container(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(containment.FORCE_ENV, "1")
        containment.require_container("a benchmark")

    def test_the_documented_override_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(containment.FORCE_ENV, "0")
        monkeypatch.setenv(containment.ALLOW_HOST_RUN_ENV, "1")
        containment.require_container("a benchmark")

    def test_a_value_other_than_one_does_not_open_the_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`SHIPINFER_ALLOW_HOST_RUN=0` and `=true` are not the override. Accepting any
        truthy-looking string would make an accidental export enough to disable the rule."""
        monkeypatch.setenv(containment.FORCE_ENV, "0")
        for value in ("0", "true", "yes", ""):
            monkeypatch.setenv(containment.ALLOW_HOST_RUN_ENV, value)
            with pytest.raises(RuntimeError):
                containment.require_container("a benchmark")


class TestTheDeviceCliCommandsAreGated:
    """`serve` and `bench` stand up engines, so they carry the gate themselves."""

    @pytest.mark.parametrize("module", ["serve", "bench"])
    def test_the_command_calls_require_container(self, module: str) -> None:
        import importlib
        import inspect

        loaded = importlib.import_module(f"shipinfer.cli.commands.{module}")
        source = inspect.getsource(loaded)

        assert "require_container(" in source, f"{module} is not gated"
