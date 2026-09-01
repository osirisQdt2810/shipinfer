"""The device-tier guards, for the half of ``testpaths`` that is not ``tests/``.

A conftest is directory-scoped, so ``pytest -m gpu benchmarks/tests/...`` never loads
``tests/conftest.py`` — and would then run the GPU tier with **no container gate and no
no-device skip**, breaking two promises that hold everywhere else: that the gate runs inside
the process that would do the work so no spelling avoids it, and that ``-m gpu`` on a host
with no driver reports "skipped" rather than "failed".

Re-exported rather than reimplemented, so the two directories cannot drift. Both hooks are
idempotent, so a whole-suite run that loads both conftests is unaffected.
"""

from __future__ import annotations

from tests.conftest import (  # noqa: F401
    pytest_collection_modifyitems,
    pytest_configure,
)
