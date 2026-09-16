"""One degraded card must not take the whole GPU tier down.

It did, for three days from 1 Sep: every script in `deploy/rootless/` hard-coded
`--device nvidia.com/gpu=all`, the driver enumerated eight cards while CUDA could open
seven, and `torch.cuda.__init__`'s queued `_check_capability` walks EVERY visible device --
so a test that wanted GPUs 0-3 died at CUDA init because of a card it never asked for::

    RuntimeError: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED
    at ATen/cuda/CUDAContext.cpp:52 ... device=7, num_gpus=7

These tests are offline: they read the scripts and run the helper, so the rule holds on a
machine with no driver -- which is the only place the rule can be checked cheaply.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROOTLESS = ROOT / "deploy" / "rootless"
HELPER = ROOTLESS / "_gpus.sh"

#: The doctor keeps `all`, and that is the point of it: it exists to enumerate every card and
#: to SAY which one is faulted. Every script that does work takes the knob.
DOCTOR = "setup.sh"

#: Scripts that start a container for a job with no GPU in it: `gst-image.sh` bakes the
#: GStreamer image with `docker run` + `docker commit` (this kernel refuses `docker build`),
#: and apt needs no card. Exempt from the knob, held to asking for no device at all. One
#: entry: `wheels.sh` runs no container, so an entry for it would assert nothing.
NO_GPU = ("gst-image.sh",)


def _scripts() -> list[Path]:
    return sorted(p for p in ROOTLESS.glob("*.sh") if p.name != HELPER.name)


def _runs_docker(path: Path) -> bool:
    return "docker run" in path.read_text()


def _effective(path: Path, seen: frozenset[str] = frozenset()) -> str:
    """A script's text plus that of every sibling it sources, recursively.

    The knob became reachable through a shared file rather than directly: `test.sh` and
    `run.sh` now source `_container.sh`, which sources `_gpus.sh`. Read literally, the
    original check skipped both -- so the one script written to be a door had no GPU
    assertion at all while the suite stayed green.
    """
    text = path.read_text()
    parts = [text]
    for name in re.findall(r"_[a-z]+\.sh", text):
        if name != path.name and name not in seen and (ROOTLESS / name).is_file():
            parts.append(_effective(ROOTLESS / name, seen | {path.name, name}))
    return "\n".join(parts)


def _expand(value: str | None, *, probe: str = "0") -> subprocess.CompletedProcess[str]:
    """Source the helper with `SHIPINFER_GPUS` set, and print what it built.

    `probe` defaults to OFF so these cases stay about the expansion. With it on, `_gpus.sh`
    asks which cards CUDA can open and routes around the ones it cannot -- asserted separately
    below, with a stub, because the real answer depends on the box the suite runs on.
    """
    env = {"PATH": "/usr/bin:/bin", "SHIPINFER_GPU_PROBE": probe} | (
        {"SHIPINFER_GPUS": value} if value is not None else {}
    )
    return subprocess.run(
        ["bash", "-c", f'. "{HELPER}" && echo "${{GPU_DEVICES[*]}}"'],
        capture_output=True,
        text=True,
        env=env,
    )


class TestEveryRunTakesTheKnob:
    @pytest.mark.parametrize("script", [p.name for p in _scripts() if _runs_docker(p)])
    def test_no_script_hard_codes_every_device(self, script: str) -> None:
        lines = [
            line
            for line in (ROOTLESS / script).read_text().splitlines()
            if "nvidia.com/gpu=all" in line and not line.lstrip().startswith("#")
        ]

        if script == DOCTOR:
            assert lines, f"{DOCTOR} is the doctor and enumerates every card on purpose"
            return
        assert not lines, (
            f'{script} hard-codes every device: {lines}. Use "${{GPU_DEVICES[@]}}" so '
            f"SHIPINFER_GPUS can route around a faulted card"
        )

    @pytest.mark.parametrize("script", [p.name for p in _scripts()])
    def test_a_script_that_uses_the_array_sources_the_helper(self, script: str) -> None:
        """Every script, not only the ones that `docker run`: `_container.sh` builds the argv."""
        path = ROOTLESS / script
        if "${GPU_DEVICES[@]}" not in path.read_text():
            pytest.skip(f"{script} does not use the array")

        assert "_gpus.sh" in _effective(path), (
            f"{script} expands GPU_DEVICES without sourcing _gpus.sh, so it would run with "
            f"no --device flag at all -- which reads exactly like a host with no driver"
        )

    @pytest.mark.parametrize(
        "script", [p.name for p in _scripts() if _runs_docker(p) and p.name != DOCTOR]
    )
    def test_every_docker_run_reaches_the_knob(self, script: str) -> None:
        """The positive half, which the skip above cannot give: a runner that never mentions
        the array is a runner with no `--device` flag, and it skips rather than fails."""
        text = _effective(ROOTLESS / script)
        if script in NO_GPU:
            assert "--device" not in text, f"{script} is exempt and asks for a device anyway"
            return

        assert "${GPU_DEVICES[@]}" in text and "_gpus.sh" in text, (
            f"{script} starts a container without reaching SHIPINFER_GPUS, directly or "
            f"through a file it sources"
        )


class TestTheHelperExpandsWhatItIsGiven:
    @pytest.mark.parametrize("value", [None, "", "all"])
    def test_unset_empty_and_all_mean_every_device(self, value: str | None) -> None:
        done = _expand(value)

        assert done.returncode == 0, done.stderr
        assert done.stdout.split() == ["--device", "nvidia.com/gpu=all"]

    def test_unset_still_means_every_device_when_the_probe_is_off(self) -> None:
        """The pre-16-Sep contract, kept reachable: `SHIPINFER_GPU_PROBE=0` restores it."""
        assert _expand(None, probe="0").stdout.split() == ["--device", "nvidia.com/gpu=all"]

    def test_a_list_becomes_one_flag_per_index(self) -> None:
        """One `--device` per index, not a comma list: a typo in a comma list is a device
        named `0,1` that resolves to nothing and hands the container no GPU at all."""
        done = _expand("0,1,2,3")

        assert done.returncode == 0, done.stderr
        assert done.stdout.split() == [
            arg for index in range(4) for arg in ("--device", f"nvidia.com/gpu={index}")
        ]

    def test_a_single_index_is_a_list_of_one(self) -> None:
        assert _expand("2").stdout.split() == ["--device", "nvidia.com/gpu=2"]

    @pytest.mark.parametrize("value", ["0,x", "all,0", "-1", "0,,1"])
    def test_anything_that_is_not_an_index_is_refused_by_name(self, value: str) -> None:
        """Refused rather than dropped: a silently ignored entry is a run on fewer GPUs than
        the operator asked for, and the per-device table would then be quietly wrong."""
        done = _expand(value)

        assert done.returncode == 2, f"{value!r} was accepted: {done.stdout!r}"
        assert "is not an index" in done.stderr


class TestACardCudaCannotOpenIsRoutedAround:
    """`_gpus.sh`'s whole reason for existing, finally the default rather than an incantation.

    Its header has quoted `device=7, num_gpus=7` since 1 Sep -- `torch.cuda.__init__` walks
    every VISIBLE device, so one card the driver enumerates and CUDA cannot open takes the tier
    down. The default stayed `all` and it happened again on 16 Sep. Stubbed here because the
    real probe's answer depends on the machine.
    """

    def _with_probe(self, script: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
        probe = tmp_path / "probe.py"
        probe.write_text(script)
        return subprocess.run(
            ["bash", "-c", f'. "{HELPER}" && echo "${{GPU_DEVICES[*]}}"'],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "SHIPINFER_GPU_PROBE": str(probe)},
        )

    def test_the_healthy_cards_become_the_default(self, tmp_path: Path) -> None:
        done = self._with_probe("print('0,1,2')\n", tmp_path)

        assert done.returncode == 0, done.stderr
        assert done.stdout.split() == [
            "--device",
            "nvidia.com/gpu=0",
            "--device",
            "nvidia.com/gpu=1",
            "--device",
            "nvidia.com/gpu=2",
        ]

    def test_it_says_which_card_it_dropped(self, tmp_path: Path) -> None:
        """A silently smaller device set is how somebody benchmarks seven cards and reports eight."""
        done = self._with_probe(
            "import sys\nprint('0,1,2')\nprint('7 (00000000:D2:00.0)', file=sys.stderr)\n",
            tmp_path,
        )

        assert "GPU(s) 7 (00000000:D2:00.0)" in done.stderr
        assert "0,1,2" in done.stderr

    def test_a_probe_that_finds_nothing_leaves_the_default_alone(self, tmp_path: Path) -> None:
        """Agreement, no runtime, no `nvidia-smi` -- all of them print nothing and exit 1."""
        done = self._with_probe("raise SystemExit(1)\n", tmp_path)

        assert done.stdout.split() == ["--device", "nvidia.com/gpu=all"]

    def test_an_explicit_choice_is_never_second_guessed(self, tmp_path: Path) -> None:
        probe = tmp_path / "probe.py"
        probe.write_text("print('0,1,2')\n")
        done = subprocess.run(
            ["bash", "-c", f'. "{HELPER}" && echo "${{GPU_DEVICES[*]}}"'],
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "SHIPINFER_GPU_PROBE": str(probe),
                "SHIPINFER_GPUS": "all",
            },
        )

        assert done.stdout.split() == ["--device", "nvidia.com/gpu=all"]


class TestTheDefaultProbePathIsTheOneProductionUses:
    def test_the_path_the_helper_computes_when_nothing_overrides_it_exists(self) -> None:
        """`SHIPINFER_GPU_PROBE` is a test seam; the path used in anger is asserted by nothing
        else, and a rename would leave every box silently back on the old default."""
        assert (ROOT / "scripts" / "usable_gpus.py").is_file()


class TestTheProbeCannotTakeTheCallerDownWithIt:
    """Every script sourcing `_gpus.sh` runs under `set -euo pipefail` (`_container.sh`).

    Round 1 of #295 ran the probe TWICE — once for stdout, once for stderr — and the second
    assignment had no `|| ...`. A probe that succeeded then failed aborted the sourcing script
    with nothing on either stream: a worse failure than the `device=7, num_gpus=7` assert this
    exists to remove, because at least that one said something.
    """

    def _under_set_e(self, script: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
        probe = tmp_path / "probe.py"
        probe.write_text(script)
        return subprocess.run(
            [
                "bash",
                "-c",
                # `;` and NOT `&&`: an AND-list suppresses `set -e` inside the sourced file,
                # so a `&&` here would pass whatever the helper does. Real callers source it on
                # a line of its own (`_container.sh`), which is what this reproduces.
                f'set -euo pipefail; . "{HELPER}"; echo "GPU_DEVICES=${{GPU_DEVICES[*]}}"',
            ],
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "SHIPINFER_GPU_PROBE": str(probe),
                "TMPDIR": str(tmp_path),
            },
        )

    def test_a_probe_that_fails_after_its_first_call_does_not_abort_the_caller(
        self, tmp_path: Path
    ) -> None:
        """THE round-1 regression: it is only reachable if the probe runs more than once."""
        marker = tmp_path / "called"
        done = self._under_set_e(
            "import pathlib, sys\n"
            f"m = pathlib.Path({str(marker)!r})\n"
            "if m.exists(): sys.exit(3)\n"
            "m.write_text('x')\nprint('0,1,2')\n",
            tmp_path,
        )

        assert done.returncode == 0, f"the caller survived: {done.stderr}"
        assert "GPU_DEVICES=" in done.stdout, done.stdout
        assert marker.exists(), "and the probe did run"

    def test_a_probe_that_always_fails_leaves_the_default_and_says_nothing_extra(
        self, tmp_path: Path
    ) -> None:
        done = self._under_set_e("import sys; sys.exit(9)\n", tmp_path)

        assert done.returncode == 0, done.stderr
        assert done.stdout.split() == ["GPU_DEVICES=--device", "nvidia.com/gpu=all"]
