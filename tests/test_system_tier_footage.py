"""The footage mount the system tier depends on, checked from inside the container.

`tests/system` runs the real chain on real video, and the video cannot live in the
repository: `references/` is gitignored and committing frames of identifiable people to
make a test runnable is the wrong trade. So the operator points `SHIPINFER_SYSTEM_VIDEO` at
a file or a frame directory and `deploy/rootless/test.sh` mounts it read-only. This asserts
the plumbing, so a broken mount is a named failure rather than eight silent skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

#: The mount this asserts is established by ``deploy/rootless/test.sh``, so the assertion is
#: only meaningful inside that container -- the same tier the footage exists for. Unmarked it
#: would land in the offline selection, where there is no mount and nothing to check.
pytestmark = pytest.mark.gpu


class TestTheFootageMount:
    @pytest.mark.skipif(
        not os.environ.get("SHIPINFER_SYSTEM_VIDEO"),
        reason="SHIPINFER_SYSTEM_VIDEO is unset; set it to run the system tier",
    )
    def test_what_the_operator_pointed_at_is_readable_here(self) -> None:
        path = Path(os.environ["SHIPINFER_SYSTEM_VIDEO"])
        assert path.exists(), f"{path} is not visible in this container; check the mount"
        assert os.access(path, os.R_OK), f"{path} is not readable"
        if path.is_dir():
            assert any(path.iterdir()), f"{path} is an empty frame directory"
        else:
            assert path.stat().st_size > 0, f"{path} is empty"
