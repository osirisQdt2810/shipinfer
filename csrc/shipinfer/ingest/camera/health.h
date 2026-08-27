// What an operator needs to know about one camera, and about the fleet —
// `ingest/camera/health.py`.
//
// A camera is not up or down. It is connecting, streaming, dropping frames because the
// inference pool is saturated, or retrying every thirty seconds against a dead switch — and the
// right operator response differs for each. The reference system had a single `_online[cam_id]`
// boolean, which is why a camera that quietly stopped delivering took a week to diagnose.
#pragma once

#include <cstdint>
#include <string>

namespace shipinfer {

    // The lifecycle of one camera actor. The names match the Python enum's *values*, so a
    // health payload from either plane reads the same.
    enum class CameraState {
        Idle,        // constructed, not started
        Connecting,  // trying to open the stream; normal for a second or two at start-up
        Streaming,   // frames are arriving
        Degraded,    // was streaming, has stopped delivering, is retrying — recoverable
        // Has failed `failures_before_unhealthy` times in a row. Still retrying at the capped
        // delay — a camera down overnight must come back on its own — but this is the state a
        // dashboard should be paging on.
        Unhealthy,
        Exhausted,  // the source reported end of stream and will not produce more
        Stopped,    // stopped on request
    };

    const char* to_string(CameraState state);
    bool is_healthy(CameraState state);

    // An immutable snapshot of one camera, safe to read from another thread: the actor builds
    // it under its own lock and hands it out, so a caller cannot see half of an update.
    struct CameraHealth {
        std::string camera_id;
        CameraState state = CameraState::Idle;
        // Frames the source produced.
        uint64_t frames_read = 0;
        // Frames actually accepted by the sink. The gap to `frames_read` is backpressure.
        uint64_t frames_published = 0;
        // Frames the sink refused. Counted, never silent — that is the whole point of ADR-005.
        uint64_t frames_dropped = 0;
        // Reads that returned nothing (a timeout on a live stream, EOF on a file).
        uint64_t empty_reads = 0;
        uint64_t connects = 0;
        uint64_t connect_failures = 0;
        // Consecutive failures right now; resets on the first frame after a reconnect.
        uint64_t consecutive_failures = 0;
        // Measured over the last window, not since start: an average since start-up hides the
        // camera that stopped ten minutes ago.
        double fps = 0.0;
        int64_t last_frame_unix_ns = 0;
        std::string last_error;

        bool is_healthy() const { return shipinfer::is_healthy(state); }
        double drop_ratio() const {
            return frames_read
                       ? static_cast<double>(frames_dropped) / static_cast<double>(frames_read)
                       : 0.0;
        }
    };

    // The fleet in one object, for a health endpoint and the stats log.
    struct IngestSummary {
        size_t cameras = 0;
        size_t streaming = 0;
        size_t unhealthy = 0;
        double total_fps = 0.0;
        uint64_t frames_read = 0;
        uint64_t frames_published = 0;
        uint64_t frames_dropped = 0;

        // Deliberately strict: with 50 cameras, "most of them work" is the state the previous
        // system lived in for months.
        bool is_healthy() const { return cameras > 0 && streaming == cameras; }
    };

    // Window over which the actor measures fps. Long enough to be stable at 20 fps, short
    // enough that a camera that stopped a few seconds ago reads as 0.
    inline constexpr double kFpsWindowS = 2.0;
    // Shortest partial window that gives a usable estimate before the first full one.
    inline constexpr double kFpsMinWindowS = 0.25;

}  // namespace shipinfer
