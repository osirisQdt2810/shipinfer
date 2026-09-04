"""The container has a door for one command, and the docs name the one that exists.

The rule is that anything touching an accelerator runs in a container, and `deploy/` holds the
recipes. It held three: `test.sh` for pytest, `bench.sh` for the benchmark, `prove.sh` for the
attestation. For everything else -- an engine build, `shipinfer repo ls`, one `python -c`, a
shell to look around in -- CLAUDE.md and `container.md` both said `make shell`, and there is no
Makefile anywhere in this repository. A documented command that fails is how somebody ends up
reaching for `SHIPINFER_ALLOW_HOST_RUN=1`.

Offline: these read the scripts and the docs. Whether the container actually starts is proved
by the runs in the PR body, not here.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOTLESS = ROOT / "deploy" / "rootless"
RUN = ROOTLESS / "run.sh"
INSIDE = ROOTLESS / "_inside.sh"
CONTAINER = ROOTLESS / "_container.sh"
#: The two runners that start the same container.
RUNNERS = (ROOTLESS / "test.sh", RUN)


class TestTheDoorExists:
    def test_run_sh_is_present_and_executable(self) -> None:
        assert RUN.is_file(), f"{RUN} is the documented door; it has to exist"
        assert RUN.stat().st_mode & 0o111, "and be executable, like its siblings"

    def test_it_takes_the_gpu_knob_like_every_other_runner(self) -> None:
        """Otherwise a faulted card takes this door down too (#128).

        Through the shared definition, not a third copy of `_gpus.sh` -- which is the point of
        `TestTheTwoRunnersCannotDrift` below.
        """
        assert "_container.sh" in RUN.read_text()
        assert "_gpus.sh" in CONTAINER.read_text()

    def test_it_asks_for_a_tty_only_when_there_is_one(self) -> None:
        """`-it` unconditionally makes docker refuse in every script, CI job and agent.

        "cannot attach stdin to a TTY-enabled container because stdin is not a terminal" --
        so the door would work by hand and fail everywhere else, which is the half that
        matters most for a command whose whole purpose is being reachable.
        """
        text = RUN.read_text()

        assert "[ -t 0 ]" in text, "the TTY has to be conditional"
        assert "--rm -it" not in text, "an unconditional -it is the defect"


class TestTheTwoRunnersCannotDrift:
    """`test.sh` and `run.sh` differ in what they `exec`, and in nothing else.

    Both halves of that are load-bearing, and both were proved by this branch. `_inside.sh`
    was factored out because two copies of a wheel list is how a container ends up without
    grpcio while the run still reads as green -- and then `run.sh` copied `test.sh`'s
    *docker* block and had drifted before review: TensorRT at `/opt/tensorrt` where all four
    siblings use `/tensorrt`, so a `trtexec` line copied out of `cpp.sh` failed in the one
    door built to run it.
    """

    def test_neither_builds_its_own_docker_argv(self) -> None:
        """The container is defined in `_container.sh` and nowhere else."""
        for script in RUNNERS:
            body = script.read_text()
            assert "_container.sh" in body, f"{script.name} has to source the definition"
            assert '"${DOCKER_ARGV[@]}"' in body, f"{script.name} has to use it"
            for own in ('-v "$REPO:/work', '-v "$WHEELS', "--device nvidia.com/gpu"):
                assert own not in body, f"{script.name} still builds its own argv: {own}"

    def test_the_shared_definition_carries_what_both_need(self) -> None:
        text = CONTAINER.read_text()

        assert "DOCKER_HOST" in text and "docker info" in text, "the socket, once"
        # `"/tensorrt:ro" in text` was the first spelling and it is a substring of
        # `/opt/tensorrt:ro`, so it passed on the very drift it was written to catch.
        assert '-v "$TRT_DIR:/tensorrt:ro"' in text, "the path every sibling script uses"
        code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        assert not [ln for ln in code if "/opt/tensorrt" in ln], "the drift, back in the argv"
        assert "_gpus.sh" in text and "SHIPINFER_TEST_GPUS" in text
        assert 'CONTAINER_MOUNT="${CONTAINER_MOUNT:-ro}"' in text, "read-only by default"

    def test_the_door_mounts_footage_like_the_test_runner(self) -> None:
        """Otherwise `run.sh python -m pytest -m gpu tests/system` skips the tier silently."""
        for script in RUNNERS:
            body = script.read_text()
            assert "SHIPINFER_SYSTEM_VIDEO" in body and ":/footage:ro" in body, script.name

    def test_both_source_the_shared_preamble(self) -> None:
        for script in RUNNERS:
            assert "_inside.sh" in script.read_text(), (
                f"{script.name} installs its own wheels; two copies of that list drift, and "
                f"a container missing grpcio silently skips 119 tests"
            )

    def test_the_preamble_installs_the_control_plane_as_required(self) -> None:
        """Not optional: without grpcio/protobuf, 119 tests never COLLECT."""
        text = INSIDE.read_text()

        assert "grpcio" in text and "protobuf" in text
        assert "grpcio-tools==" in text, "pinned -- 1.83 emits a different stub for one .proto"

    def test_neither_installs_wheels_itself_any_more(self) -> None:
        for script in RUNNERS:
            body = script.read_text()
            assert (
                "pip install" not in body
            ), f"{script.name} should install through _inside.sh, not inline"


# doc: long the scan is over the TRACKED tree, and why a mention needs a marker
class TestTheDocsNameACommandThatExists:
    """Nothing tells a developer to run `make shell`, and there is no Makefile to run.

    Scanned over the **tracked** tree rather than two hand-listed docs, because the first
    version of this test listed `CLAUDE.md` and `container.md` and passed green while the
    string was still live in the four paths a developer actually hits: the containment gate's
    refusal, the hook's `permissionDecisionReason`, the hook's own allowlist, and
    `.claude/memory/run-in-container.md` -- which is durable, so it survives a compaction and
    comes back.

    A surviving mention has to carry a MARKER saying the command does not exist, so the
    history can be recorded without handing anybody a command that fails.
    """

    #: A line may mention it only while saying it is gone.
    MARKERS = ("no Makefile", "SUPERSEDED", "does not exist", "never did", "was a compose")
    #: The ledger is a record of what happened, not instructions.
    EXEMPT = (".claude/TASKS.md", ".claude/JOURNAL.md", "tests/test_container_door.py")

    @staticmethod
    def _tracked() -> list[str]:
        """Tracked paths, so a gitignored `references/` checkout is not this repo's problem."""
        done = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [name for name in done.stdout.split("\0") if name]

    def test_no_tracked_file_hands_anybody_make_shell(self) -> None:
        offending: list[str] = []
        for name in self._tracked():
            if name in self.EXEMPT:
                continue
            path = ROOT / name
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if "make shell" in line and not any(m in line for m in self.MARKERS):
                    offending.append(f"{name}:{number}: {line.strip()[:90]}")

        assert not offending, (
            "these hand a developer a command that does not exist; say "
            f"`deploy/rootless/run.sh` instead, or mark the line with one of {self.MARKERS}:\n"
            + "\n".join(offending)
        )

    def test_the_scan_reaches_the_places_that_print_it(self) -> None:
        """Non-vacuity: the two paths a refusal actually shows a developer are in the scan."""
        tracked = set(self._tracked())

        assert "src/shipinfer/runtime/containment.py" in tracked
        assert "scripts/hooks/require_container.py" in tracked

    def test_there_really_is_no_makefile(self) -> None:
        """Tracked paths only. `ROOT.glob("**/Makefile")` walked into `references/`, the
        submodule and any local `.venv` -- so a bare `pytest` would fail on a file no commit
        here controls, on a developer box and never in CI. #124's lesson, a third time."""
        found = [name for name in self._tracked() if Path(name).name == "Makefile"]

        assert not found, f"a Makefile is tracked now: {found}; revisit the docs and this test"

    def test_the_hook_allowlists_the_scripts_that_exist(self) -> None:
        """A hook that allows a command which cannot run is worse than one that refuses it."""
        hook = (ROOT / "scripts" / "hooks" / "require_container.py").read_text()
        allowlist = re.search(r"CONTAINERISED = \((.*?)\)", hook, re.DOTALL)
        assert allowlist, "no CONTAINERISED tuple found"
        named = set(re.findall(r'"([^"]+)"', allowlist.group(1)))

        assert "deploy/rootless/run.sh" in named, "the door has to be allowed"
        assert (
            not {"make shell", "make test", "make bench"} & named
        ), "the allowlist names commands that do not exist"
        for script in named:
            if script.startswith("deploy/"):
                assert (ROOT / script).is_file(), f"{script} is allowed and does not exist"
