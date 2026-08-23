"""Strip credentials out of a stream URI before it reaches a log, an error or an API.

An RTSP camera URI carries its password inline — ``rtsp://admin:s3cret@10.0.0.5/stream`` —
and a fleet typically shares one credential across every camera. So a single URI written to
a log is the whole fleet's password, and it is written on a path that repeats: the actor
logs its URI on **every** restart, and a camera that cannot be opened backs off and retries
forever.

The failure this prevents is not hypothetical. A typo'd stream path raises ``SourceOpenError``
carrying the raw URI; that message is stored as the camera's ``last_error`` and served by the
ingest health endpoint, so the credential ends up in the log on every attempt *and* in an API
response any reader can fetch.

Redaction happens at the boundary rather than at the source: the URI must stay intact for the
thing that opens the stream, so it is the *formatting* that is unsafe, not the value. Every
site that turns a URI into text for a human calls :func:`redact`.

It lives in ``core`` rather than in ``ingest`` because ``core.errors.ingest`` needs it too,
and ``core`` may not import upwards (ADR-001).
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

__all__ = ["PLACEHOLDER", "redact", "redact_in"]

#: What a password becomes. Fixed rather than length-preserving on purpose — a mask that
#: tracked the real length would leak it.
PLACEHOLDER = "***"


def redact(uri: str | None) -> str:
    """``rtsp://admin:s3cret@host/stream`` -> ``rtsp://admin:***@host/stream``.

    The username is kept: it is useful for diagnosis and is not the secret. Anything that
    does not parse as a URL, or carries no password, is returned unchanged — a local file
    path and a plain ``rtsp://host/stream`` are both common and neither is sensitive.

    Never raises. This runs inside logging and error construction, and a redaction helper
    that can throw would turn a diagnostic into a second failure on the path that is already
    failing.
    """
    if not uri:
        return ""
    try:
        parts = urlsplit(uri)
        if not parts.password:
            return uri
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        userinfo = f"{parts.username or ''}:{PLACEHOLDER}"
        return urlunsplit(
            (parts.scheme, f"{userinfo}@{host}", parts.path, parts.query, parts.fragment)
        )
    except ValueError:
        # A URI too malformed to split cannot be redacted piecewise, and echoing it raw
        # risks printing whatever credential it does contain. Callers always have the
        # camera id alongside, so nothing identifying is lost.
        return "<unparseable uri>"


#: ``scheme://user:password@`` inside a larger string. The password is group 2 and is the
#: only part replaced, so the scheme, the username and everything around it survive.
_EMBEDDED = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:/@]+):([^\s@/]+)@")


def redact_in(text: str) -> str:
    """Redact every credential-bearing URI embedded in ``text``.

    For strings that are not themselves a URI but contain one — a GStreamer pipeline
    description is the case that forced this: ``rtspsrc location=rtsp://admin:s3cret@host
    latency=200 ! ...`` is logged verbatim so an operator can paste it into
    ``gst-launch-1.0``, and that convenience published the fleet password.
    """
    if not text:
        return ""
    return _EMBEDDED.sub(lambda m: f"{m.group(1)}:{PLACEHOLDER}@", text)
