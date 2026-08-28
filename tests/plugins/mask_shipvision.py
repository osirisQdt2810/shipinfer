"""Hide ``shipvision`` the way a checkout with no submodule does.

`3rdparty/shipvision` is optional and **CI does not check it out** (`.claude/CLAUDE.md`), so
"the suite is green" on a developer box says nothing about the run that gates a PR. Every
file that touches the submodule is written to be green both ways — the classes that drive a
real tracker skip, the classes that assert a *contract* run everywhere — and this plugin is
what makes the second half checkable without deleting anything::

    pytest -p tests.plugins.mask_shipvision tests/topology -q

It is committed rather than retyped per session. It was retyped per session: the mask was a
throwaway `sitecustomize.py` in a scratch directory, so every reviewer who wanted the
CI-shaped numbers wrote their own — and a mask written slightly differently is a *different
test run*, which is the one thing an evidence artefact must not be.

**The failure has to be `ModuleNotFoundError`, not `ImportError`.** ``pytest.importorskip``
skips on the first and propagates the second, so a mask that raised the wrong one turns four
of ``tests/topology/test_bridge.py``'s skips into failures and the run stops meaning
"shipvision is absent" and starts meaning "the mask is broken". ``None`` in ``sys.modules``
is exactly right: CPython's import machinery reads it as "known not importable" and raises
``ModuleNotFoundError("import of shipvision halted; None in sys.modules")`` — the same class
a genuinely missing package raises, which is why every ``except ImportError`` in
``topology/bridge.py`` catches it unchanged.

Loaded with ``-p`` and not as a ``conftest.py``, because it has to run **before** anything
imports the package: a conftest is imported during collection, by which time a module-scope
``import shipvision`` in a test file has already succeeded and been cached.
"""

from __future__ import annotations

import sys

#: The submodule and the four subpackages the elements reach through
#: ``shipinfer.topology.bridge``. The parent alone would be enough — importing
#: ``shipvision.mot`` imports ``shipvision`` first and stops there — but a name that is
#: *already* in ``sys.modules`` is never looked up again, and this file cannot know what a
#: plugin loaded before it imported. Naming them costs five lines and removes the question.
_MASKED = (
    "shipvision",
    "shipvision.errors",
    "shipvision.mot",
    "shipvision.mtmc",
    "shipvision.reid",
    "shipvision.types",
)


def _mask() -> None:
    for name in [name for name in sys.modules if name.split(".")[0] == "shipvision"]:
        del sys.modules[name]
    for name in _MASKED:
        sys.modules[name] = None  # type: ignore[assignment]


# At import time, not in `pytest_configure`: plugins named with `-p` are imported before
# collection, and collection is where a test module's own imports run.
_mask()


def pytest_report_header() -> str:
    """Say so in the header, so an evidence paste carries its own provenance.

    A masked run and an unmasked one differ by ~130 skips and by nothing else visible, and a
    number pasted into a PR without this line is indistinguishable from the other run.
    """
    return "shipvision: MASKED (tests/plugins/mask_shipvision.py) — the CI-shaped run"
