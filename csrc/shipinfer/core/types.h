// Small value types the whole data plane shares.
//
// Deliberately tiny and header-only: these cross every layer boundary, and a type that needs
// a translation unit becomes a link-order problem in a codebase with this many threads.
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "shipinfer/core/redact.h"

namespace shipinfer {

    // The two clocks, one place. Every deadline and every latency in this plane is steady
    // (`scheduling/queues/fair.h`, `fifo.h`, `engine/request.h`), so the arithmetic reads
    // `monotonic_ns`; anything a human or another process reads is wall time.
    inline int64_t monotonic_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
    inline int64_t unix_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    // The identity that must survive every path, including every error path. The Python side
    // calls this `RequestContext`; the invariant is the same and it is the one the previous
    // generation lost, which is how a crowded camera came to evict a quiet one's work.
    //
    // TWO CLOCKS, AND THEY ARE NOT INTERCHANGEABLE
    // --------------------------------------------
    // Both are read at the moment of decode, because that is the last place that knows when
    // the frame actually existed — a timestamp taken later measures the queue, not the camera.
    struct FrameTag {
        std::string camera_id;
        int64_t frame_id = 0;
        // STEADY nanoseconds. Every deadline in this plane is built from this field, and
        // `InferenceRequest::is_expired` compares against `monotonic_ns()`. Stamping wall time
        // here is not a cosmetic error: a wall-clock value is ~1.7e18 against a steady value of
        // ~1e13, so `deadline_ns = captured_ns + budget` would land roughly 54 years out and
        // nothing would ever expire. `tests/test_engine.cpp` pins both halves of that.
        int64_t captured_ns = 0;
        // WALL nanoseconds — for humans and for cross-process joins. NEVER for deadline
        // arithmetic: NTP can step it, including backwards.
        int64_t captured_unix_ns = 0;

        std::string key() const { return camera_id + ":" + std::to_string(frame_id); }
    };

    // One detection. Boxes are in the *original* image's pixel coordinates, not the letterboxed
    // model input's — every downstream crop is taken from the full-resolution frame, which is
    // both cheaper and sharper than cropping a resized crop.
    struct Detection {
        float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        float score = 0;
        int class_id = 0;
        int index = 0;  // position within the frame, so a scattered row can find its owner
    };

    // The typed failure vocabulary, mirroring `core/errors/` on the Python side. One exception
    // type per domain, because "something went wrong" is not a diagnosis and a caller that
    // cannot tell a missing engine from a full queue cannot do anything useful about either.
    struct ConfigError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    struct BackendError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    // Carries the numbers (ADR-005): an operator paged on a refusal must be able to tell a
    // queue at 5/4096 from one at 4096/4096, and the parity harness compares them.
    struct QueueFullError : std::runtime_error {
        QueueFullError(const std::string& message, size_t depth_, size_t capacity_)
            : std::runtime_error(message), depth(depth_), capacity(capacity_) {}
        size_t depth;
        size_t capacity;
    };
    // The server is not in a state to do this: no instance is ready, a model has no instances.
    struct ServerStateError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    // `core.errors.RequestCancelledError`: the request left the system without an answer.
    // Lives here rather than beside the request it cancels because the camera actor catches it
    // — a sink that has shut down says so with this — and `ingest/` may not include `engine/`.
    struct RequestCancelledError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    // -- the ingest vocabulary, mirroring `core/errors/ingest.py` ---------------------------
    //
    // Four distinct events, four types, because the operator response to each is different:
    // fix the install, fix the network, fix the camera, or wait. A single `RuntimeError` for
    // all four is why the previous generation logged "Can not read frame" and nothing else.

    // Base for every ingest-plane failure, so a camera actor can catch one thing.
    struct IngestError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    // A source could not produce what it was asked for. The general case, kept because the
    // replay library throws it for a folder that is not a folder.
    struct SourceError : IngestError {
        using IngestError::IngestError;
    };
    // The decode runtime this source needs is not installed on this host.
    //
    // Distinct from `SourceOpenError` on purpose: a missing library is fixed by an install and
    // will never fix itself, whereas an unreachable camera might come back in a second. The
    // actor must not burn a reconnect budget on the former — it gives up, says so once, and
    // stays UNHEALTHY so a dashboard can tell it from a camera an operator removed.
    struct SourceUnavailableError : IngestError {
        SourceUnavailableError(const std::string& source_, const std::string& hint_)
            // Redacted in the *message*, intact on the members — `SourceOpenError`'s rule,
            // which this class did not follow. The message becomes
            // `CameraHealth::last_error`, and the hint is redacted for the same reason that
            // one redacts its `reason`: a GStreamer hint reads `set location=<uri>`.
            : IngestError("video source '" + redact_uri(source_) +
                          "' is unavailable: " + redact_in(hint_)),
              source(source_),
              hint(hint_) {}
        std::string source;
        std::string hint;
    };
    // The stream could not be opened, or negotiated no usable video. Carries the camera id and
    // the URI because an ingest log with fifty cameras in it is useless without them.
    struct SourceOpenError : IngestError {
        SourceOpenError(const std::string& camera_id_, const std::string& uri_,
                        const std::string& reason_)
            // Redacted in the *message*, intact on the attribute. The message is what gets
            // logged and what becomes `CameraHealth::last_error`, so a fleet password would
            // otherwise be served to every reader of that payload on every retry. `reason` is
            // redacted too, and that is not belt-and-braces: decoders put the URI inside it.
            : IngestError("camera '" + camera_id_ + "': cannot open '" + redact_uri(uri_) +
                          "': " + redact_in(reason_)),
              camera_id(camera_id_),
              uri(uri_),
              reason(reason_) {}
        std::string camera_id;
        std::string uri;
        std::string reason;
    };
    // A frame read failed in a way that ends the stream (EOS, decoder error). Thrown rather
    // than returning nothing: "no frame yet" and "this stream is over" demand opposite
    // responses — keep waiting, or reconnect — and one empty result for both is exactly the
    // ambiguity that makes a stalled camera invisible.
    struct FrameDecodeError : IngestError {
        FrameDecodeError(const std::string& camera_id_, const std::string& reason_)
            : IngestError("camera '" + camera_id_ + "': decode failed: " + redact_in(reason_)),
              camera_id(camera_id_),
              reason(reason_) {}
        std::string camera_id;
        std::string reason;
    };
    // One or more cameras never produced a frame within the start-up window, so a deploy
    // against a mistyped camera database fails at start-up instead of looking healthy and
    // producing no detections.
    struct CameraUnavailableError : IngestError {
        CameraUnavailableError(std::vector<std::string> cameras_, double timeout_s_)
            : IngestError(build_message(cameras_, timeout_s_)),
              cameras(std::move(cameras_)),
              timeout_s(timeout_s_) {}
        std::vector<std::string> cameras;
        double timeout_s;

      private:
        static std::string build_message(const std::vector<std::string>& cameras_,
                                         double timeout_s_) {
            std::string named;
            for (const std::string& camera : cameras_) {
                named += (named.empty() ? "" : ", ") + camera;
            }
            return std::to_string(cameras_.size()) + " camera(s) produced no frame within " +
                   std::to_string(timeout_s_) + "s: " + named;
        }
    };

}  // namespace shipinfer
