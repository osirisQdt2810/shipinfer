#include "shipinfer/ingest/base.h"

#include <utility>

namespace shipinfer {

    FrameSource::FrameSource(IngestConfig config, FrameCounter& counter, StopSignal& stop)
        : config_(std::move(config)), counter_(counter), stop_(stop) {
        if (counter_.camera_id() != config_.camera_id) {
            // A source stamping another camera's frames would produce a `(camera_id,
            // frame_id)` key that already belongs to somebody else — the one thing ADR-002
            // relies on never happening — so it is refused at construction rather than
            // discovered downstream.
            throw ConfigError("frame counter belongs to camera '" + counter_.camera_id() +
                              "', not '" + config_.camera_id + "'");
        }
    }

    void FrameSource::open() {
        if (is_open_) return;
        try {
            do_open();
        } catch (...) {
            // Unwind unconditionally and best-effort. The suppression is the point: the
            // original failure is the one worth propagating, and a `do_close` that throws while
            // cleaning up after a failed open would otherwise replace a useful diagnosis
            // ("cannot open rtsp://…: connection refused") with a useless one.
            try {
                do_close();
            } catch (...) {  // NOLINT — deliberate, see above
            }
            throw;
        }
        is_open_ = true;
    }

    std::optional<Frame> FrameSource::read() {
        if (!is_open_) {
            throw SourceOpenError(config_.camera_id, config_.uri, "read() before open()");
        }
        std::optional<HostFrame> image = do_read();
        if (!image) return std::nullopt;
        return counter_.stamp(std::move(*image));
    }

    void FrameSource::close() {
        if (!is_open_) return;
        is_open_ = false;
        do_close();
    }

    void FrameSource::set_format(int height, int width, double fps) {
        height_ = height;
        width_ = width;
        fps_ = fps;
    }

}  // namespace shipinfer
