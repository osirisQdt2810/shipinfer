"""``SourceUnavailableError`` redacts its own message, like its two siblings do.

The rule is stated in ``core/errors/ingest.py`` and applied to ``SourceOpenError`` and
``FrameDecodeError``: the message is what gets logged **and what becomes**
``CameraHealth.last_error`` in the health API, so a credential left in it is served to every
reader of ``GET /streams`` on every retry. ``SourceUnavailableError`` was the sibling that did
not, and it is the one the fatal-open path stores (P6-D1/D2).
"""

from __future__ import annotations

import pytest

from shipinfer.core.errors import FrameDecodeError, SourceOpenError, SourceUnavailableError
from shipinfer.core.redact import PLACEHOLDER

URL = "rtsp://admin:s3cret@cam-01.quay/stream1"


class TestNoIngestErrorCarriesACredentialInItsMessage:
    @pytest.mark.parametrize(
        "error",
        [
            SourceUnavailableError(URL, "install pygobject"),
            SourceUnavailableError("gstreamer", f"set location={URL}"),
            SourceOpenError("cam-000", URL, "connection refused"),
            FrameDecodeError("cam-000", f"[Errno 111] Connection refused: {URL!r}"),
        ],
        ids=["unavailable-source", "unavailable-hint", "open", "decode"],
    )
    def test_the_password_never_reaches_the_message(self, error: Exception) -> None:
        """Every argument, not only the one named for a URI.

        The decoders put the URI inside the *hint* as well: a GStreamer hint reads
        ``set location=rtsp://...``, which is the door ``SourceOpenError`` had to close for
        its ``reason`` and this one had open.
        """
        assert "s3cret" not in str(error)
        assert PLACEHOLDER in str(error)

    def test_the_attribute_keeps_the_credential_so_a_retry_can_use_it(self) -> None:
        """Redacted in the message, intact on the attribute — ``SourceOpenError``'s rule."""
        error = SourceUnavailableError(URL, "install pygobject")

        assert error.source == URL, "the actor reconnects with this; redacting it breaks retry"

    def test_a_message_with_no_credential_is_left_alone(self) -> None:
        error = SourceUnavailableError("file:///data/clip.mp4", "install pyav")

        assert "file:///data/clip.mp4" in str(error)
        assert PLACEHOLDER not in str(error)
