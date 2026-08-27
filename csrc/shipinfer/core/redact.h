// Strip credentials out of a stream URI before it reaches a log, an error or an API —
// `core/redact.py`, ported.
//
// An RTSP camera URI carries its password inline (`rtsp://admin:s3cret@10.0.0.5/stream`) and a
// fleet typically shares one credential across every camera, so a single URI written to a log
// is the whole fleet's password. It is written on a path that repeats: the camera actor logs
// its URI on every connect, and a camera that cannot be opened backs off and retries forever.
//
// The URI must stay intact for whatever opens the stream, so it is the *formatting* that is
// unsafe rather than the value. Every site that turns a URI into text for a human calls
// `redact_uri`; every site that logs an error message that might have a URI embedded in it
// calls `redact_in`.
//
// It lives in `core/` rather than in `ingest/` because `core/types.h` needs it too —
// `SourceOpenError` builds its message from a URI — and `core` may not import upwards
// (ADR-001). That is the same reason the Python module sits in `core`.
#pragma once

#include <cstddef>
#include <string>

namespace shipinfer {

    // What a password becomes. Fixed rather than length-preserving on purpose — a mask that
    // tracked the real length would leak it.
    inline constexpr const char* kRedactPlaceholder = "***";

    namespace detail {

        inline bool is_space(char c) {
            return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
        }
        inline bool is_alpha(char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        }
        // The scheme class of RFC 3986: ALPHA *( ALPHA / DIGIT / "+" / "-" / "." ).
        inline bool is_scheme_char(char c) {
            return is_alpha(c) || (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.';
        }

    }  // namespace detail

    // `rtsp://admin:s3cret@host/stream` -> `rtsp://admin:***@host/stream`.
    //
    // The username is kept: it is useful for diagnosis and is not the secret. A URI with no
    // `@` carries no credential and comes back unchanged — a local file path and a plain
    // `rtsp://host/stream` are both common and neither is sensitive.
    //
    // **Where the credential ends is decided by the LAST `@`.** Both of the obvious readings
    // fail open on passwords real fleets use: a spec-conformant parse stops the authority at
    // the first `/`, so `rtsp://admin:pa/ss@10.0.0.5/stream` parses as having no password at
    // all; and stopping at the first `@` leaves the tail of `p@ss123` in the clear, which is
    // worse than not redacting because the output *looks* redacted.
    //
    // This over-masks one case — `rtsp://host:554/a@b`, a portful host with an `@` in the path
    // and no credentials, becomes `rtsp://host:***@b`. That is the direction to be wrong in.
    //
    // Never throws. It runs inside error construction and logging, and a redaction helper that
    // can throw turns a diagnostic into a second failure on the path that is already failing.
    inline std::string redact_uri(const std::string& uri) {
        if (uri.empty()) return "";
        const size_t separator = uri.find("://");
        if (separator == std::string::npos) return uri;
        const std::string rest = uri.substr(separator + 3);
        const size_t at = rest.rfind('@');
        if (at == std::string::npos) return uri;
        const std::string userinfo = rest.substr(0, at);
        const std::string host = rest.substr(at + 1);
        const size_t colon = userinfo.find(':');
        // `rtsp://user@host`: a username with no password is not a secret.
        if (colon == std::string::npos) return uri;
        if (host.empty()) {
            // Too malformed to split, and echoing it raw risks printing whatever credential it
            // does contain. Callers always have the camera id alongside, so nothing
            // identifying is lost. Fail closed.
            return "<unparseable uri>";
        }
        return uri.substr(0, separator) + "://" + userinfo.substr(0, colon) + ":" +
               kRedactPlaceholder + "@" + host;
    }

    // Redact every credential-bearing URI *embedded* in a larger string.
    //
    // For text that is not itself a URI but contains one. A decoder's error message is the
    // case that forces this: the runtimes put the URI inside their own text, so redacting only
    // the argument named `uri` leaves the credential in the message by the other door — and
    // the actor stores that message as `CameraHealth::last_error`, which a health endpoint
    // serves to every reader on every retry.
    //
    // Matches `scheme://user:password@`, where the password runs to the **last** `@` inside the
    // whitespace-delimited token (so `pa/ss` and `p@ss123` are both covered) and cannot cross
    // whitespace (so two URIs on one line stay two matches).
    inline std::string redact_in(const std::string& text) {
        if (text.empty()) return "";
        std::string out;
        out.reserve(text.size());
        size_t cursor = 0;
        while (cursor < text.size()) {
            const size_t mark = text.find("://", cursor);
            if (mark == std::string::npos) {
                out.append(text, cursor, std::string::npos);
                break;
            }
            size_t scheme = mark;
            while (scheme > 0 && detail::is_scheme_char(text[scheme - 1])) --scheme;
            // The run may begin with characters a scheme cannot *start* with — digits, `.`,
            // `+`, `-`, as in `"2.rtsp://admin:pw@host"`. The scheme is the run from its
            // first ALPHA onwards, which is where Python's regex anchors too. Giving up on
            // the whole run here fails OPEN: the password behind a numeric prefix would pass
            // through unredacted (#33 round 2).
            while (scheme < mark && !detail::is_alpha(text[scheme])) ++scheme;
            if (scheme == mark) {
                out.append(text, cursor, mark + 3 - cursor);  // not a scheme; nothing to do
                cursor = mark + 3;
                continue;
            }
            // The user part: one or more characters that are none of space, `:`, `/`, `@`.
            size_t colon = mark + 3;
            while (colon < text.size() && !detail::is_space(text[colon]) &&
                   text[colon] != ':' && text[colon] != '/' && text[colon] != '@') {
                ++colon;
            }
            if (colon == mark + 3 || colon >= text.size() || text[colon] != ':') {
                out.append(text, cursor, colon - cursor);
                cursor = colon;
                continue;
            }
            // The password: greedy to the last `@` before the token ends.
            size_t end = colon + 1;
            while (end < text.size() && !detail::is_space(text[end])) ++end;
            const size_t at = text.rfind('@', end - 1);
            if (at == std::string::npos || at < colon + 2) {
                out.append(text, cursor, end - cursor);
                cursor = end;
                continue;
            }
            out.append(text, cursor, colon + 1 - cursor);
            out.append(kRedactPlaceholder);
            out.push_back('@');
            cursor = at + 1;
        }
        return out;
    }

}  // namespace shipinfer
