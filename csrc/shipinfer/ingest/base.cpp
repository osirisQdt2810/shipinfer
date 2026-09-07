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
        // Device first, because a source that has one answers only there; `do_read_device`
        // defaults to nothing, so this costs one branch for every host source.
        if (std::optional<DeviceImage> on_device = do_read_device()) {
            latch_where(true);
            return counter_.stamp(std::move(*on_device));
        }
        std::optional<HostFrame> image = do_read();
        if (!image) return std::nullopt;
        latch_where(false);
        return counter_.stamp(std::move(*image));
    }

    void FrameSource::close() {
        if (!is_open_) return;
        is_open_ = false;
        do_close();
    }

    void FrameSource::latch_where(bool on_device) {
        // Latched on the FIRST frame and enforced after: where a camera's pixels live is a
        // property of the camera, not of the frame, and a source that changed its answer --
        // most plausibly by falling back to software decode on a reconnect -- would move the
        // whole graph onto the slow path with nothing said. Refusing names both answers.
        if (!where_latched_) {
            where_latched_ = true;
            reads_device_ = on_device;
            return;
        }
        if (reads_device_ != on_device) {
            // `ConfigError` and not `FrameDecodeError`: this is a contract violation, not a
            // stream ending. A decode error asks the actor to reconnect, which for a source
            // that changes its own answer would retry the bug forever.
            throw ConfigError(
                "camera '" + config_.camera_id + "': source answered from do_read" +
                (on_device ? "_device" : "") + " after answering from do_read" +
                (reads_device_ ? "_device" : "") +
                "; where a camera's pixels live is a property of the camera, and the chain is "
                "built once from it");
        }
    }

    void FrameSource::set_format(int height, int width, double fps) {
        height_ = height;
        width_ = width;
        fps_ = fps;
    }

}  // namespace shipinfer
