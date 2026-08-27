// Replay a directory of frames, paced at a target frame rate — `ingest/sources/replay.py`.
//
// This is the backend that makes the ingest plane *measurable*. Everything above it — the
// per-camera actor, the frame tag, the reconnect policy, the fair queue's per-camera lanes, the
// 50-camera skew that reproduces the inherited starvation bug — is exercised by a source that
// needs no camera and no network. A design where that is impossible is a design whose ingest
// path is only ever tested in production.
//
// **This unit is the ingest plane's one accelerator-facing file.** It reaches OpenCV (to
// decode) and `core/platform.h` (to page-lock the decoded pages), which is why nothing else
// under `ingest/` may include this header: `scripts/build_csrc.py` follows a header to the
// `.cpp` beside it, so one such include would pull the driver's headers into the closure of the
// whole ingest plane and the offline C++ tier would stop building on a machine with no CUDA.
// See the note at the top of `ingest/registry.cpp`.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "shipinfer/ingest/base.h"
#include "shipinfer/ingest/frame.h"
#include "shipinfer/ingest/timing/pacing.h"

namespace shipinfer {

    // Every image under one folder, decoded once and shared by every camera replaying it.
    //
    // Decoding the same JPEG a thousand times measures libjpeg, not this server: at benchmark
    // scale the decode was the *dominant* cost and the load generator, not the inference
    // system, was the wall. Shared rather than per-camera because fifty cameras replaying one
    // folder would otherwise hold fifty copies — a 1920x1080 BGR frame is 6.2 MB, so ten files
    // across fifty cameras is 3.1 GB against 62 MB shared.
    class ReplayLibrary {
      public:
        // The library for `folder`, decoding it at most once per process while anybody holds a
        // handle. `limit > 0` takes only the first N files in name order.
        //
        // Throws SourceError when the folder is not a directory or holds nothing decodable — a
        // benchmark that silently offers zero frames is not a slower measurement, it is a
        // different experiment.
        static std::shared_ptr<const ReplayLibrary> acquire(const std::string& folder,
                                                            int limit = 0);

        ~ReplayLibrary();

        ReplayLibrary(const ReplayLibrary&) = delete;
        ReplayLibrary& operator=(const ReplayLibrary&) = delete;

        size_t size() const { return frames_.size(); }
        // The image at `index`, with `owner` left empty: the *source* fills it, because the
        // source is what knows which handle is keeping this library alive.
        HostFrame at(size_t index) const;
        // Whether every image is page-locked. False means the run is still correct and its
        // host->device copies take the slow path — worth printing, not worth failing over. One
        // refused registration used to flip a single flag and the destructor then unregistered
        // nothing, leaving the pages that had registered locked; each image records its own
        // outcome now.
        bool pinned() const;
        // Files under the folder that did not decode — skipped, and reported rather than
        // hidden.
        size_t undecodable() const { return undecodable_; }

      private:
        ReplayLibrary(const std::string& folder, int limit);

        struct Image {
            std::vector<uint8_t> pixels;
            int height = 0;
            int width = 0;
        };
        std::vector<Image> frames_;
        std::vector<char> registered_;  // per image, 1 when gpuHostRegister succeeded
        size_t undecodable_ = 0;
    };

    // A frame directory delivered at `config.fps`.
    //
    // `config.loop` decides what end-of-input means. True (the default) rewinds, which is what
    // a long-running stress test wants; false marks the source exhausted, which is how a test
    // that wrote six frames asserts it received exactly six and then terminated on its own.
    class ReplaySource : public FrameSource {
      public:
        using FrameSource::FrameSource;

        bool is_exhausted() const override { return exhausted_; }

        // Exposed so a bench can report how often the consumer fell behind.
        const DeadlinePacer& pacer() const { return pacer_; }

      protected:
        void do_open() override;
        std::optional<HostFrame> do_read() override;
        void do_close() override;

      private:
        std::shared_ptr<const ReplayLibrary> library_;
        DeadlinePacer pacer_;
        size_t index_ = 0;
        bool exhausted_ = false;
    };

}  // namespace shipinfer
