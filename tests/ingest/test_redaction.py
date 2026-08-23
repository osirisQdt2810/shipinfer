"""No stream credential reaches a log, an error message or the health API.

An RTSP fleet shares one password across every camera, so a single URI printed once is the
whole fleet's credential — and the paths that print it repeat: the actor logs its URI on
every restart, and a camera that cannot be opened backs off and retries forever. A typo'd
stream path was therefore enough to write the password to the log on every attempt and to
serve it from the ingest health endpoint to anyone who asked.

These tests pin the boundary, not the helper: `redact` having the right behaviour is worth
little if a call site forgets it, so the assertions are made against the message the error
actually carries and against the text the logger actually emits.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from shipinfer.core.errors import SourceOpenError
from shipinfer.core.redact import PLACEHOLDER, redact, redact_in

SECRET = "s3cr3t-fleet-password"
URI = f"rtsp://admin:{SECRET}@10.0.0.5/stream"


class TestRedact:
    def test_the_password_is_replaced_and_the_rest_survives(self) -> None:
        out = redact(URI)
        assert SECRET not in out
        assert out == f"rtsp://admin:{PLACEHOLDER}@10.0.0.5/stream"

    def test_the_username_is_kept_because_it_is_not_the_secret(self) -> None:
        assert "admin" in redact(URI)

    def test_the_mask_does_not_leak_the_length(self) -> None:
        short = redact("rtsp://u:a@h/s")
        long = redact("rtsp://u:aaaaaaaaaaaaaaaaaaaaaaaaaaa@h/s")
        assert short == long

    @pytest.mark.parametrize(
        "uri",
        ["rtsp://10.0.0.5/stream", "/data/frames", "file:///data/clip.mp4", ""],
    )
    def test_a_uri_with_no_password_is_untouched(self, uri: str) -> None:
        assert redact(uri) == uri

    def test_a_port_survives(self) -> None:
        assert redact(f"rtsp://admin:{SECRET}@10.0.0.5:8554/s") == (
            f"rtsp://admin:{PLACEHOLDER}@10.0.0.5:8554/s"
        )

    def test_it_never_raises(self) -> None:
        """It runs inside logging and error construction; throwing there would turn a
        diagnostic into a second failure on the path that is already failing."""
        for hostile in ["://", "rtsp://[", "%%%", "rtsp://u:p@", "\x00"]:
            assert isinstance(redact(hostile), str)

    def test_an_embedded_uri_is_redacted_in_place(self) -> None:
        description = f"rtspsrc location={URI} latency=200 ! rtph264depay ! appsink"
        out = redact_in(description)
        assert SECRET not in out
        assert "latency=200" in out, "only the password is replaced"


class TestTheErrorDoesNotCarryTheSecret:
    def test_source_open_error_message_is_redacted(self) -> None:
        """This message becomes `CameraHealth.last_error` and is served by the health API."""
        error = SourceOpenError("cam0", URI, "connection refused")
        assert SECRET not in str(error)
        assert PLACEHOLDER in str(error)

    def test_the_attribute_keeps_the_real_uri(self) -> None:
        """Redaction is a formatting decision. Whatever needs to reopen the stream needs
        the credential, so the value survives on the attribute and only the text is masked."""
        assert SourceOpenError("cam0", URI, "boom").uri == URI


class TestNoLogCallFormatsARawUri:
    """A source-level guard over the logging calls, parsed rather than grepped.

    Asserted by reading the code, because a test that emits its own log line proves the
    *test's* formatting is safe and would keep passing after someone dropped the `redact()`
    from `actor.py`.

    Scoped to logger calls on purpose. Two other places legitimately handle the raw URI and
    must keep doing so: the GStreamer pipeline description has to contain the real
    credential or the stream will not open, and `SourceOpenError` is *given* the raw URI and
    redacts it inside its own message — which `TestTheErrorDoesNotCarryTheSecret` pins.
    """

    SITES = (
        "src/shipinfer/ingest/camera/actor.py",
        "src/shipinfer/ingest/sources/gstreamer.py",
        "src/shipinfer/ingest/sources/pyav.py",
        "src/shipinfer/ingest/sources/replay.py",
        "src/shipinfer/ingest/manager.py",
    )
    LOG_LEVELS = {"debug", "info", "warning", "error", "exception", "critical"}

    @staticmethod
    def _mentions_uri(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id == "uri":
                return True
            if isinstance(child, ast.Attribute) and child.attr == "uri":
                return True
        return False

    @staticmethod
    def _is_redacted(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in {"redact", "redact_in"}
            ):
                return True
        return False

    def _offenders(self, source: str, relative: str) -> list[str]:
        found: list[str] = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in self.LOG_LEVELS:
                continue
            for argument in list(node.args) + [k.value for k in node.keywords]:
                if self._mentions_uri(argument) and not self._is_redacted(argument):
                    found.append(f"{relative}:{node.lineno}")
        return found

    def test_no_logger_receives_an_unredacted_uri(self) -> None:
        root = Path(__file__).resolve().parents[2]
        offenders: list[str] = []
        for relative in self.SITES:
            path = root / relative
            if path.is_file():
                offenders += self._offenders(path.read_text(), relative)
        assert not offenders, "a raw URI is handed to a logger here: " + ", ".join(offenders)

    def test_the_guard_would_catch_a_regression(self) -> None:
        """Without this the guard could be vacuous: a walker that matches nothing passes."""
        bad = "_LOG.info('camera %s at %s', cam, self.config.uri)"
        good = "_LOG.info('camera %s at %s', cam, redact(self.config.uri))"
        assert self._offenders(bad, "x.py"), "the guard misses the very thing it exists for"
        assert not self._offenders(good, "x.py"), "the guard rejects the correct form"
