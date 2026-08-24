"""Am I in a container, and may this device work run here?

The project rule is that every measurement runs in a container. It was first enforced by a
`PreToolUse` hook that reads the *text* of a shell command, and review demonstrated what
should have been obvious: a deny-list over command text cannot be made sound. Nine ordinary
spellings walked through it — ``( pytest tests/ )``, ``eval "pytest tests/"``,
``coverage run -m pytest``, ``echo pytest | sh``, ``tox``, ``uv run pytest`` — because the
executable a shell will eventually run is not a property of the string.

So the gate lives here instead, at the two places device work actually begins: the GPU test
tier and the CLI commands that stand up the server. Those cannot be quoted around. The text
hook stays as a fast advisory that catches the common case before a command runs at all,
which is genuinely useful and no longer load-bearing.

**What is refused, and what is not.** The offline tier must keep running anywhere — that is
ADR-001, it is what CI does, and it is the promise that the pure layers need no driver. So
this refuses only work that touches an accelerator: the ``gpu``/``multigpu`` test tiers, and
``shipinfer bench`` / ``serve``.

**Detecting a container.** ``/.dockerenv`` alone is a file anyone can ``touch``, and it was
the whole basis of the attestation, so a host run could self-certify. Agreement between
independent signals is required instead: the marker file, pid 1's cgroup, and the root
mount's device. Any two is enough — a rootless container with ``--pid=host`` legitimately
shows the host's cgroup, so demanding all three would refuse the configuration this project
actually runs in.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "ALLOW_HOST_RUN_ENV",
    "Containment",
    "detect",
    "require_container",
]

#: The one documented way to run device work on the host. Per-command and deliberately
#: noisy: there is no session-wide switch, because "I turned it off and forgot" is how the
#: rule was lost the first time.
ALLOW_HOST_RUN_ENV = "SHIPINFER_ALLOW_HOST_RUN"

#: Overrides detection outright. ``1`` asserts a container, ``0`` asserts a host. Only ever
#: makes the check *stricter* in the ``0`` direction, which is what the test suite uses to
#: exercise the refusal from inside a container.
FORCE_ENV = "SHIPINFER_IN_CONTAINER"

_MARKERS = ("/.dockerenv", "/run/.containerenv")


class Containment:
    """The evidence, and what it adds up to.

    Kept as an object rather than a bool so a caller can *report* what it saw. An
    attestation that says "in a container" without saying why is not evidence.
    """

    __slots__ = ("cgroup", "forced", "marker", "overlay_root")

    def __init__(
        self, *, marker: bool, cgroup: bool, overlay_root: bool, forced: str | None
    ) -> None:
        self.marker = marker
        self.cgroup = cgroup
        self.overlay_root = overlay_root
        self.forced = forced

    @property
    def signals(self) -> int:
        return sum((self.marker, self.cgroup, self.overlay_root))

    @property
    def in_container(self) -> bool:
        if self.forced == "1":
            return True
        if self.forced == "0":
            return False
        # Two of three. One is not enough: `/.dockerenv` is a file, and `touch /.dockerenv`
        # made every host run self-attest as containerised.
        return self.signals >= 2

    def describe(self) -> str:
        parts = [
            f"marker={self.marker}",
            f"cgroup={self.cgroup}",
            f"overlay_root={self.overlay_root}",
        ]
        if self.forced is not None:
            parts.append(f"forced={self.forced}")
        return " ".join(parts)


def _marker_present() -> bool:
    return any(Path(m).exists() for m in _MARKERS)


def _cgroup_says_container() -> bool:
    """Pid 1's cgroup naming a container runtime.

    Reads pid 1 rather than self, because a rootless container run with ``--pid=host`` shares
    the host's PID namespace and `self` then sits in the *host's* cgroup — which is exactly
    the configuration `deploy/rootless/` uses, so reading self would report "host" from
    inside a container.
    """
    try:
        text = Path("/proc/1/cgroup").read_text(errors="replace")
    except OSError:
        return False
    needles = ("docker", "containerd", "kubepods", "libpod", "podman", "lxc")
    return any(needle in text for needle in needles)


def _root_is_overlay() -> bool:
    """The root filesystem being an overlay, which a container image is and a host is not."""
    try:
        for line in Path("/proc/self/mountinfo").read_text(errors="replace").splitlines():
            fields = line.split(" - ")
            if len(fields) < 2:
                continue
            mount_point = fields[0].split()[4]
            fs_type = fields[1].split()[0]
            if mount_point == "/":
                return fs_type in {"overlay", "overlayfs"}
    except OSError:
        return False
    return False


def detect() -> Containment:
    """Gather the evidence. Cheap enough to call at start-up, never cached across a fork."""
    return Containment(
        marker=_marker_present(),
        cgroup=_cgroup_says_container(),
        overlay_root=_root_is_overlay(),
        forced=os.environ.get(FORCE_ENV),
    )


def host_run_allowed() -> bool:
    return os.environ.get(ALLOW_HOST_RUN_ENV) == "1"


def require_container(what: str) -> None:
    """Refuse ``what`` unless this process is in a container, or the operator said so.

    Raises:
        RuntimeError: not containerised and no override. Deliberately not one of the
            project's typed errors: this fires before any of them mean anything, and the
            caller is a CLI or a test session rather than a pipeline stage.
    """
    if host_run_allowed():
        return
    containment = detect()
    if containment.in_container:
        return
    raise RuntimeError(
        f"{what} must run inside a container (.claude/CLAUDE.md, 'Where commands run'). "
        f"Containment evidence: {containment.describe()}.\n"
        f"Host nvcc here is 11.5 against a 12.6 driver, so a host measurement does not "
        f"describe the deployment.\n"
        f"  deploy/rootless/test.sh -m gpu     the GPU tier\n"
        f"  deploy/rootless/bench.sh           the benchmark\n"
        f"  make shell                         an interactive shell\n"
        f"If the operator has agreed this one may run on the host, set "
        f"{ALLOW_HOST_RUN_ENV}=1 and say so in the report."
    )
