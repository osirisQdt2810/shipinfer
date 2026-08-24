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


class TestPasswordsThatBreakTheEasyParse:
    """The three shapes that made the first implementation fail *open*.

    Review found all three by running the function rather than reading it, and every one
    reaches the ingest health endpoint: `SourceOpenError` embeds `redact(uri)`, the actor
    stores that as `last_error`, and the API serialises it. A URI this malformed also never
    opens, so the actor retries forever and republishes the leak on every attempt.
    """

    #: `urlsplit` follows RFC 3986, where `/` ends the authority — so the URI has no `@` in
    #: its authority at all, `parts.password` is None, and the old early return echoed it.
    SLASH = "rtsp://admin:pa/ss@10.0.0.5/stream"
    #: Stopping at the first `@` left `ss123` in the clear, in output that *looks* redacted.
    AT = "rtsp://admin:p@ss123@10.0.0.5/stream"
    #: Both at once, with a port, which is what a real fleet credential tends to look like.
    BOTH = "rtsp://admin:Ab/c@123@cam.local:554/h264"

    @pytest.mark.parametrize("uri", [SLASH, AT, BOTH])
    def test_no_fragment_of_the_password_survives(self, uri: str) -> None:
        redacted = redact(uri)
        assert "pa/ss" not in redacted
        assert "ss123" not in redacted
        assert "Ab/c" not in redacted
        assert PLACEHOLDER in redacted

    @pytest.mark.parametrize("uri", [SLASH, AT, BOTH])
    def test_the_host_survives_so_the_line_is_still_diagnostic(self, uri: str) -> None:
        """Redacting by throwing the URI away would be safe and useless."""
        redacted = redact(uri)
        assert "admin" in redacted
        assert uri.rsplit("@", 1)[1].split("/")[0] in redacted

    @pytest.mark.parametrize("uri", [SLASH, AT, BOTH])
    def test_the_same_holds_for_a_uri_embedded_in_a_decoder_message(self, uri: str) -> None:
        """The route that actually leaks: PyAV renders `[Errno 111] Connection refused:
        '<uri>'` and `gst_parse` reports the location property verbatim, and `redact_in` is
        what stands between those and the log."""
        for template in (
            "[Errno 111] Connection refused: '{}'",
            'could not set property "location" to "{}"',
        ):
            redacted = redact_in(template.format(uri))
            assert "pa/ss" not in redacted
            assert "ss123" not in redacted
            assert "Ab/c" not in redacted
            assert PLACEHOLDER in redacted


class TestItFailsClosed:
    """When the parse is uncertain the answer is to print nothing, not to print the input."""

    def test_a_credential_with_nothing_after_it_is_not_echoed(self) -> None:
        assert redact("rtsp://admin:secret@") == "<unparseable uri>"

    def test_a_string_with_no_scheme_is_not_a_uri_and_is_left_alone(self) -> None:
        """A local clip path is a legitimate source and carries no secret."""
        assert redact("/var/lib/clips/cam0.mp4") == "/var/lib/clips/cam0.mp4"

    def test_over_masking_is_the_direction_it_errs_in(self) -> None:
        """A portful host with an `@` in the path and no credentials at all comes back
        masked. Pinned rather than fixed: the alternative reading of this string is the one
        that leaks `p@ss123`, and a mangled log line is the cheaper mistake."""
        assert redact("rtsp://host:554/a@b") == "rtsp://host:***@b"


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
        "src/shipinfer/ingest/camera/db.py",
        "src/shipinfer/ingest/camera/health.py",
        "src/shipinfer/ingest/sources/gstreamer.py",
        "src/shipinfer/ingest/sources/pyav.py",
        "src/shipinfer/ingest/sources/replay.py",
        "src/shipinfer/ingest/manager.py",
        "src/shipinfer/ingest/resolve.py",
    )
    #: Errors that redact inside their own constructor, so handing them a raw URI is
    #: correct. Anything else that formats one into a message is not.
    REDACTING_ERRORS = frozenset({"SourceOpenError", "FrameDecodeError"})
    LOG_LEVELS = {"debug", "info", "warning", "error", "exception", "critical"}

    #: Names that may carry a URI. `exc` and `reason` are here because that is where the
    #: credential actually travelled: `av.FFmpegError.__str__` embeds the container name,
    #: which is the whole RTSP URI, and the actor logs that on every reconnect. Matching only
    #: the identifier `uri` tested the *argument name* rather than the invariant, so
    #: `str(exc)` was invisible and `FrameDecodeError`'s leak read as covered.
    CARRIERS = frozenset({"uri", "exc", "reason", "error", "record", "options"})

    @classmethod
    def _mentions_uri(cls, node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in cls.CARRIERS:
                return True
            if isinstance(child, ast.Attribute) and child.attr in cls.CARRIERS:
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
        """Every place a URI could become text a human reads: a log line or an error.

        Raise sites are inspected as well as logger calls, because an exception message is
        logged on every retry *and* served as `CameraHealth.last_error`. Errors that redact
        inside their own constructor are exempt — handing them the raw URI is the point.
        """
        found: list[str] = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr not in self.LOG_LEVELS:
                    continue
                for argument in list(node.args) + [k.value for k in node.keywords]:
                    if self._mentions_uri(argument) and not self._is_redacted(argument):
                        found.append(f"{relative}:{node.lineno} (log)")
            elif isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                name = getattr(node.exc.func, "id", None) or getattr(
                    node.exc.func, "attr", None
                )
                if name in self.REDACTING_ERRORS:
                    continue
                for argument in list(node.exc.args) + [k.value for k in node.exc.keywords]:
                    if self._mentions_uri(argument) and not self._is_redacted(argument):
                        found.append(f"{relative}:{node.lineno} (raise)")
        return found

    def test_no_logger_receives_an_unredacted_uri(self) -> None:
        root = Path(__file__).resolve().parents[2]
        offenders: list[str] = []
        for relative in self.SITES:
            path = root / relative
            if path.is_file():
                offenders += self._offenders(path.read_text(), relative)
        assert not offenders, "a raw URI is handed to a logger here: " + ", ".join(offenders)

    def test_the_guard_would_catch_a_logging_regression(self) -> None:
        """Without this the guard could be vacuous: a walker that matches nothing passes."""
        bad = "_LOG.info('camera %s at %s', cam, self.config.uri)"
        good = "_LOG.info('camera %s at %s', cam, redact(self.config.uri))"
        assert self._offenders(bad, "x.py"), "the guard misses the very thing it exists for"
        assert not self._offenders(good, "x.py"), "the guard rejects the correct form"

    def test_the_guard_would_catch_a_raise_regression(self) -> None:
        """An exception message is logged on every retry and served by the health API, so
        a raw URI in one is worse than a raw URI in a single log line, not better."""
        bad = "raise ConfigurationError(f'bad camera: {uri}')"
        good = "raise ConfigurationError(f'bad camera: {redact(uri)}')"
        assert self._offenders(bad, "x.py"), "raise sites were not inspected at all"
        assert not self._offenders(good, "x.py")

    def test_an_error_that_redacts_internally_may_take_the_raw_uri(self) -> None:
        """`SourceOpenError` is *given* the real URI and masks it in its own message; the
        attribute has to keep the value so a caller can reopen the stream."""
        raw = "raise SourceOpenError(self.camera_id, self.config.uri, str(exc))"
        assert not self._offenders(raw, "x.py")
