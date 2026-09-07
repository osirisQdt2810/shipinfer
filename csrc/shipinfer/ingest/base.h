// The video-source contract: open a stream, pull frames, close it — `ingest/base.py`.
//
// A `FrameSource` does one job. It does not own the frame id, does not know about queues,
// models or GPUs, and does not decide when to retry — that is `CameraActor`'s job. Keeping the
// split there is what lets the whole ingest plane be tested with a source that returns a
// pointer to a static buffer.
//
// `open` / `read` / `close` are **template methods**, not virtual entry points. The subclass
// hooks are `do_open` / `do_read` / `do_close`, and the wrappers own three invariants that
// would otherwise have to be re-implemented (and eventually mis-implemented) per backend:
//
//   1. every frame leaving any source is stamped by the actor's counter, so the
//      `(camera_id, frame_id)` tag and both clocks cannot be forgotten by a new backend;
//   2. `open` is idempotent and cleans up after a partial failure;
//   3. `close` is idempotent, so a reconnect path and a shutdown path can both call it.
#pragma once

#include <optional>
#include <string>

#include "shipinfer/core/stop_signal.h"
#include "shipinfer/ingest/config.h"
#include "shipinfer/ingest/frame.h"

namespace shipinfer {

    class FrameSource {
      public:
        // A source is **not** expected to survive an error. The actor throws it away and builds
        // a new one, which is why the reconnect state — the backoff, the frame counter, the
        // stop signal — lives outside it and is passed by reference. Both references outlive
        // every source the actor builds, because the actor destroys its source before itself.
        FrameSource(IngestConfig config, FrameCounter& counter, StopSignal& stop);

        // NOT a `close()`. A destructor cannot dispatch to a subclass, so `~FrameSource`
        // calling `close()` would call *this* class's pure `do_close` — undefined behaviour at
        // best. The actor closes; a concrete source may close defensively in its own
        // destructor, where the dispatch still works.
        virtual ~FrameSource() = default;

        FrameSource(const FrameSource&) = delete;
        FrameSource& operator=(const FrameSource&) = delete;

        // -- lifecycle -------------------------------------------------------------------

        // Connect and negotiate. Idempotent.
        //
        // A `do_open` that throws leaves the source closed and gets exactly one best-effort
        // `do_close` to unwind with: a half-open source leaks a socket and a decoder thread,
        // and the subclass may not be able to tell how far it got. The *original* exception is
        // the one that propagates, even if `do_close` throws too.
        //
        // Throws:
        //   SourceUnavailableError: the decode runtime is not installed. Not retryable.
        //   SourceOpenError: the stream could not be opened, or carries no video.
        void open();

        // One frame, or nothing if none arrived within the read timeout.
        //
        // Nothing means "not yet" — a live stream that has gone quiet, or a paced replay
        // between frames — and consumes no frame id. It never means "broken": that is a
        // `FrameDecodeError`, so the actor can reconnect immediately instead of waiting out an
        // empty-read budget.
        //
        // Throws:
        //   SourceOpenError: called before `open()`.
        //   FrameDecodeError: the stream ended or the decoder failed.
        //   ConfigError: this camera's pixels changed which memory they live in
        //     (`FrameCounter::latch_where`). A contract violation, so `CameraActor::pump`
        //     stops the camera rather than reconnecting around it.
        std::optional<Frame> read();

        // Release everything. Idempotent, and safe after a failed `open()`.
        void close();

        // -- what the stream negotiated --------------------------------------------------

        bool is_open() const { return is_open_; }
        const std::string& camera_id() const { return config_.camera_id; }
        // 0 until `open()` has run.
        int height() const { return height_; }
        int width() const { return width_; }
        // 0 when the source does not advertise one.
        double fps() const { return fps_; }

        // True when this source will never produce another frame. Only a finite source (a
        // replay file with `loop=false`) ever says true. It is what lets a bench or a test
        // terminate on its own instead of being reconnected forever — a live camera at
        // end-of-stream is a fault, a finished file is not.
        virtual bool is_exhausted() const { return false; }

        // Whether this backend can decode on a GPU *at all*. False makes `hwaccel()` always
        // resolve to false, so a log line says "software decode" instead of implying an NVDEC
        // path that does not exist.
        virtual bool supports_hwaccel() const { return false; }
        bool hwaccel() const { return supports_hwaccel() && config_.hwaccel; }

        const IngestConfig& config() const { return config_; }

      protected:
        // Connect, and call `set_format` with what was actually negotiated.
        virtual void do_open() = 0;
        // One image as HWC BGR uint8 with its `owner` set, or nothing if none is available yet.
        //
        // STILL PURE, deliberately, now that `do_read_device` exists beside it: a device-only
        // source writes one line of `return std::nullopt;` and the compiler makes it choose.
        // Defaulting both and checking at run time that one was overridden is not possible --
        // "not overridden" and "no frame yet" are the same answer from here -- so the check
        // would have to be a count, which cannot tell a quiet stream from a missing hook.
        virtual std::optional<HostFrame> do_read() = 0;
        // doc: long the second hook, why it is defaulted, and what stops a source using both
        // One image in DEVICE memory -- NV12 with its `owner` set -- for a source that decodes
        // straight into VRAM and never brings the pixels back (V156: `rtsp -> nv12 -> tren
        // vram het`). Nothing by default, which is every source that answers `do_read`.
        //
        // A CAMERA ANSWERS FROM ONE HOOK FOR ITS WHOLE LIFE. `read()` latches which on the
        // first frame and refuses a change -- and the reason is NOT the cost of a branch, which
        // at 1000 fps is free and which `Frame::on_device()` already is. It is that where the
        // pixels live is a PLAN-CONSTRUCTION fact: the chain is built once from it, so it
        // cannot change under a chain that is already running.
        //
        // The honest objection is that falling back to software decode keeps 1 of 50 cameras
        // alive where refusing kills it. Refusing anyway, because the alternative is that
        // camera silently feeding a graph built for device frames -- and the operator learns
        // from `CameraHealth` either way, which is the difference between a stopped camera and
        // a wrong one.
        virtual std::optional<DeviceImage> do_read_device() { return std::nullopt; }
        // Release resources. Must tolerate being called after a partial `do_open`.
        virtual void do_close() = 0;

        // Record what the stream actually negotiated. Called from `do_open`.
        void set_format(int height, int width, double fps);

        StopSignal& stop() const { return stop_; }
        FrameCounter& counter() const { return counter_; }

      private:
        IngestConfig config_;
        FrameCounter& counter_;
        StopSignal& stop_;
        bool is_open_ = false;
        int height_ = 0;
        int width_ = 0;
        double fps_ = 0.0;
    };

}  // namespace shipinfer
