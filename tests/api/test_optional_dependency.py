"""The HTTP surface refuses in a typed way when its extra was never installed.

FastAPI and uvicorn are the ``server`` extra (``pip install "shipinfer[server]"``), and
`api/` imports both *inside* the functions that need them. That is what lets a host with no
extra import ``shipinfer.api``, run ``shipinfer serve`` without ``--http``, and collect this
test file — and it is what turns "the dependency is missing" from an ``ImportError`` at
start-up into a `ConfigurationError` naming the fix.

The pattern is a template, not a one-off: gRPC arrives as an optional extra in a later PR of
this split and refuses the same way, so the property is worth a test of its own rather than a
line in a docstring. `tests/api/test_api.py` cannot host it — that file skips wholesale
without fastapi, which is exactly the condition under test.
"""

from __future__ import annotations

import sys

import pytest

from shipinfer.api import create_app, serve_http
from shipinfer.core.errors import ConfigurationError


@pytest.fixture()
def without_the_server_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import fastapi`` / ``import uvicorn`` fail the way an absent wheel does.

    ``None`` in ``sys.modules`` is the interpreter's own "this import is blocked" marker and
    raises ``ImportError`` on the next import of that name, so this reproduces the missing
    extra without unloading a package the rest of the session may already hold.
    """
    for name in ("fastapi", "uvicorn"):
        monkeypatch.setitem(sys.modules, name, None)


class TestAMissingExtraIsATypedRefusal:
    def test_create_app_names_the_extra_to_install(
        self, without_the_server_extra: None
    ) -> None:
        with pytest.raises(ConfigurationError, match=r"shipinfer\[server\]"):
            # `server=None` is deliberate: the guard runs before the argument is touched, so
            # a caller with no FastAPI is told what to install rather than shown a traceback
            # from three frames deeper.
            create_app(None)  # type: ignore[arg-type]

    def test_serve_http_names_the_extra_to_install(
        self, without_the_server_extra: None
    ) -> None:
        with pytest.raises(ConfigurationError, match=r"shipinfer\[server\]"):
            serve_http(None)  # type: ignore[arg-type]

    def test_the_refusal_mentions_the_dependency_by_name(
        self, without_the_server_extra: None
    ) -> None:
        """A message that says only "install the extra" sends the reader to the docs."""
        with pytest.raises(ConfigurationError, match="uvicorn"):
            serve_http(None)  # type: ignore[arg-type]
