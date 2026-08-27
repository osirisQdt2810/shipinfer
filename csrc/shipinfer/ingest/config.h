// One camera: where it is, how to decode it, and what a drop-out costs.
//
// FLAT ON PURPOSE. The Python plane splits this in two — `CameraConfig` for the camera and
// `IngestSettings` for the fleet default it inherits — because pydantic gives it a `None`
// that reads as "inherit" and one resolution order stated once (`ingest/resolve.py`). There is
// no settings tree in this plane yet, so a two-level structure here would be two levels of
// nothing: this struct is the *resolved* record, shaped so P5's settings tree can fill it
// field for field rather than reshaped around it.
//
// The reconnect numbers are the part worth reading. The reference implementation counted to
// ten and then retried every twenty seconds forever, from a separate monitor thread; fifty
// cameras behind one switch therefore reconnected in lockstep every twenty seconds for as long
// as the switch was down. Here the backoff is exponential, jittered and capped, and it lives in
// the camera's own thread.
#pragma once

#include <cctype>
#include <cstdint>
#include <map>
#include <string>

#include "shipinfer/core/options.h"
#include "shipinfer/core/types.h"

namespace shipinfer {

    struct IngestConfig {
        // -- identity ----------------------------------------------------------------------
        // Half of the `(camera_id, frame_id)` tag that rides every request through the whole
        // system, so it must not change when a camera is re-added (ADR-002).
        std::string camera_id;
        // `rtsp://…` for a camera; a file or a directory of frames for `replay`.
        std::string uri;
        // A name registered in `SOURCES()`. Empty is refused by `create_source` rather than
        // defaulted: this plane has no environment layer yet, and guessing a backend that is
        // not linked into this binary reports "unknown video source" for a problem that is
        // actually "you did not say".
        std::string source;

        // -- decode ------------------------------------------------------------------------
        // 0 keeps the native resolution, which is what the fused letterbox wants anyway.
        int width = 0;
        int height = 0;
        // Target frame rate. 0 means "whatever the source delivers"; for `replay` it is the
        // pacing target, which is how one video file simulates a 20 fps camera.
        double fps = 0.0;
        // Jitter buffer the source keeps, in milliseconds. A direct latency cost, so the
        // 1000 ms of the previous generation is not the default here.
        int latency_ms = 200;
        // RTSP lower transport. TCP by default: UDP loses packets under load and the resulting
        // decode artefacts look exactly like a model regression.
        std::string transport = "tcp";
        // The video codec, which is what picks the depayloader, the parser and the decoder
        // (`sources/gstreamer_pipeline.h`). `auto` builds a `decodebin` pipeline that
        // negotiates the codec at connect time: the safe choice for a mixed fleet and the
        // slightly slower one, because the decoder is then chosen by plugin rank rather than by
        // us.
        std::string codec = "h264";
        // Prefer hardware decode where the backend supports it. A backend that cannot always
        // resolves this to false, so a log line says "software decode" instead of implying an
        // NVDEC path that does not exist.
        bool hwaccel = true;
        // Where this camera's `frame_id` sequence starts. Non-zero when a restarted process
        // must not reuse tags a downstream tracker has already seen.
        int64_t first_frame_id = 0;
        // `replay` only: restart at end of input. True keeps a stress test running; false makes
        // a finite fixture terminate, which is what a test wants.
        bool loop = true;
        // Set false to keep a camera in the database but out of the fleet.
        bool enabled = true;

        // -- timing and reconnect ----------------------------------------------------------
        // How long one frame read may block before it counts as an empty read. Also the worst
        // case for an actor to notice a stop request, which is why every source bounds its
        // read.
        int read_timeout_ms = 5000;
        int open_timeout_ms = 10000;
        // Consecutive empty reads before the source is torn down and reopened. An RTSP source
        // that has stopped delivering never says so; it simply times out forever.
        int empty_reads_before_reconnect = 5;
        // Pause between empty reads, so a source that returns immediately (a file at EOF, a
        // fake in a test) cannot spin a core at 100%.
        int empty_read_sleep_ms = 5;
        int reconnect_initial_ms = 500;
        int reconnect_max_ms = 30000;
        double reconnect_factor = 2.0;
        // Fraction of each delay removed at random. Non-zero on purpose: 50 cameras that fail
        // together must not retry together.
        double reconnect_jitter = 0.2;
        // Consecutive failed attempts before the camera reports UNHEALTHY. It keeps retrying at
        // the capped delay — a camera down for an hour must still come back on its own — but
        // health, and therefore the operator's dashboard, says so.
        int failures_before_unhealthy = 3;

        // Escape hatch for backend-specific options. Anything used by more than one deployment
        // belongs in a real field.
        KeywordOptions options;

        // Refuse a camera that cannot work, before a thread is started for it.
        //
        // Every check here is one an operator can act on from the message alone; that is the
        // difference between a start-up failure and fifty actors failing one at a time.
        void validate() const {
            if (camera_id.empty()) throw ConfigError("camera_id must not be empty");
            for (char c : camera_id) {
                // It becomes a metric label, a log field and the fair queue's per-camera
                // lane key; whitespace in any of those is a debugging session nobody enjoys.
                if (std::isspace(static_cast<unsigned char>(c))) {
                    throw ConfigError("camera_id '" + camera_id +
                                      "' must not contain whitespace");
                }
            }
            if (uri.empty())
                throw ConfigError("camera '" + camera_id + "': uri must not be empty");
            // `Codec` is a pydantic `Literal` on the Python plane (`settings/ingest.py`), so a
            // typo is refused there before a camera object exists. There is no Literal here, so
            // this is where it lands — and it lands at start-up rather than at the first
            // connect, because a codec no source can build a pipeline for never negotiates and
            // the camera would otherwise spend its reconnect budget discovering that.
            if (codec != "auto" && codec != "h264" && codec != "h265") {
                throw ConfigError("camera '" + camera_id + "': unsupported codec '" + codec +
                                  "'; expected one of [auto, h264, h265]");
            }
            if ((width == 0) != (height == 0)) {
                throw ConfigError("camera '" + camera_id +
                                  "': width and height must be set together, or neither");
            }
            // The same bounds pydantic enforces on the Python plane (`settings/ingest.py`),
            // so a config the server would refuse is refused here too (two planes, one rule).
            if (width != 0 && (width < 16 || height < 16)) {
                throw ConfigError("camera '" + camera_id +
                                  "': width and height must be >= 16 when set");
            }
            if (fps < 0.0) throw ConfigError("camera '" + camera_id + "': fps must be >= 0");
            if (first_frame_id < 0) {
                throw ConfigError("camera '" + camera_id + "': first_frame_id must be >= 0");
            }
            if (latency_ms < 0) {
                throw ConfigError("camera '" + camera_id + "': latency_ms must be >= 0");
            }
            if (empty_read_sleep_ms < 0) {
                throw ConfigError("camera '" + camera_id +
                                  "': empty_read_sleep_ms must be >= 0");
            }
            if (reconnect_initial_ms <= 0) {
                throw ConfigError("camera '" + camera_id +
                                  "': reconnect_initial_ms must be > 0");
            }
            if (reconnect_max_ms < reconnect_initial_ms) {
                throw ConfigError("camera '" + camera_id + "': reconnect_max_ms (" +
                                  std::to_string(reconnect_max_ms) +
                                  ") must be >= " + "reconnect_initial_ms (" +
                                  std::to_string(reconnect_initial_ms) + ")");
            }
            if (reconnect_factor <= 1.0) {
                throw ConfigError("camera '" + camera_id + "': reconnect_factor must be > 1");
            }
            if (reconnect_jitter < 0.0 || reconnect_jitter >= 1.0) {
                throw ConfigError("camera '" + camera_id +
                                  "': reconnect_jitter must be in [0, 1)");
            }
            if (empty_reads_before_reconnect < 1) {
                throw ConfigError("camera '" + camera_id +
                                  "': empty_reads_before_reconnect must be >= 1");
            }
            if (failures_before_unhealthy < 1) {
                throw ConfigError("camera '" + camera_id +
                                  "': failures_before_unhealthy must be >= 1");
            }
            if (read_timeout_ms < 1) {
                throw ConfigError("camera '" + camera_id + "': read_timeout_ms must be >= 1");
            }
        }

        double read_timeout_s() const { return read_timeout_ms / 1000.0; }
        double open_timeout_s() const { return open_timeout_ms / 1000.0; }
    };

}  // namespace shipinfer
