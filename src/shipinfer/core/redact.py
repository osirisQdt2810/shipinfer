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

__all__ = ["PLACEHOLDER", "redact", "redact_in"]

#: What a password becomes. Fixed rather than length-preserving on purpose — a mask that
#: tracked the real length would leak it.
PLACEHOLDER = "***"


def redact(uri: str | None) -> str:
    """``rtsp://admin:s3cret@host/stream`` -> ``rtsp://admin:***@host/stream``.

    The username is kept: it is useful for diagnosis and is not the secret. A URI with no
    ``@`` in it carries no credential and is returned unchanged — a local file path and a
    plain ``rtsp://host/stream`` are both common and neither is sensitive.

    **Where the credential ends is decided by the last ``@``, not the first, and not by
    ``urlsplit``.** Both of the obvious readings fail open on passwords that real fleets
    actually use:

    - ``urlsplit`` follows RFC 3986, where a ``/`` ends the authority. So in
      ``rtsp://admin:pa/ss@10.0.0.5/stream`` the authority is ``admin:pa`` with no ``@``,
      ``parts.password`` is ``None``, and an early return on that echoed the whole URI.
    - Stopping at the first ``@`` leaves the tail of a password like ``p@ss123`` in the
      clear, which is worse than not redacting at all because the output *looks* redacted.

    So: everything between the first ``:`` after the scheme and the final ``@`` is the
    password, whatever characters it contains.

    This over-masks in one case — ``rtsp://host:554/a@b``, a portful host with an ``@``
    somewhere in the path and no credentials at all, comes back as ``rtsp://host:***@b``.
    That is the direction to be wrong in: a mangled log line costs a reader a moment, and
    the alternative costs a fleet its password.

    Never raises. This runs inside logging and error construction, and a redaction helper
    that can throw would turn a diagnostic into a second failure on the path that is
    already failing.
    """
    if not uri:
        return ""
    try:
        scheme, separator, rest = uri.partition("://")
        if not separator or "@" not in rest:
            return uri
        userinfo, _, host = rest.rpartition("@")
        username, colon, _password = userinfo.partition(":")
        if not colon:
            # ``rtsp://user@host``: a username with no password is not a secret.
            return uri
        if not host:
            raise ValueError("nothing after the credential")
        return f"{scheme}://{username}:{PLACEHOLDER}@{host}"
    except (ValueError, AttributeError):
        # Too malformed to split, and echoing it raw risks printing whatever credential it
        # does contain. Callers always have the camera id alongside, so nothing identifying
        # is lost. Fail closed: this branch exists because the *only* safe default when the
        # parse is uncertain is to print nothing.
        return "<unparseable uri>"


#: ``scheme://user:password@`` inside a larger string. The password is group 3 and is the
#: only part replaced, so the scheme, the username and everything around it survive.
#:
#: The password class is ``[^\s]+`` — deliberately including ``/`` and ``@`` — and it is
#: greedy, so it runs to the **last** ``@`` in the token rather than the first. A narrower
#: class could not match ``pa/ss`` at all, and a non-greedy one left ``ss123`` of ``p@ss123``
#: in the clear. It cannot cross whitespace, so two URIs in one line stay two matches.
_EMBEDDED = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)([^\s:/@]+):([^\s]+)@")


def redact_in(text: str) -> str:
    """Redact every credential-bearing URI embedded in ``text``.

    For strings that are not themselves a URI but contain one — a GStreamer pipeline
    description is the case that forced this: ``rtspsrc location=rtsp://admin:s3cret@host
    latency=200 ! ...`` is logged verbatim so an operator can paste it into
    ``gst-launch-1.0``, and that convenience published the fleet password.
    """
    if not text:
        return ""
    return _EMBEDDED.sub(lambda m: f"{m.group(1)}{m.group(2)}:{PLACEHOLDER}@", text)
