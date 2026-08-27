// Where a camera actor puts a frame — a contract this package owns.
//
// The ingest plane's job ends at "here is a tagged frame". Turning that frame into an
// inference request for a particular model, in a particular queue, is **dispatch policy**, and
// dispatch policy belongs next to the DAG that consumes it: the same code that maps a frame
// onto a request has to undo that mapping when it reassembles the results, and splitting one
// decision across two packages is how the two halves drift apart. So `ingest` depends on this
// contract instead of on `scheduling`, and the part that has to scale — 50 cameras at 20 fps —
// can be measured against `CountingSink` with no scheduler in the process at all.
//
// **Throwing is the contract, not a failure mode.** An RTSP camera cannot be backpressured: it
// will send the next frame whether or not anyone is ready. So a sink that cannot accept a
// frame says so by throwing, and the *caller* decides what to do — and the caller is the camera
// actor, which is the only thing in the system that knows which camera is being greedy and can
// count the drop against it (ADR-005).
//
// Two errors, both from `core/types.h`, so a sink implementation needs nothing from the
// scheduler either — and they are exactly the two the fair queue's `put` already reports, so
// the production adapter is a frame-to-`WorkItem` mapping and nothing else.
#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#include "shipinfer/ingest/frame.h"

namespace shipinfer {

    struct FrameSink {
        virtual ~FrameSink() = default;

        // Accept a frame, or throw if it cannot be accepted.
        //
        // Throws:
        //   QueueFullError: there is no room. The caller drops this frame, charges it to this
        //     camera and continues; carrying the depth and capacity is what turns "we lost a
        //     frame" into a number an operator can act on.
        //   RequestCancelledError: the consumer has shut down and will accept nothing more.
        //     The caller finishes, rather than logging one line per frame until the process
        //     dies.
        virtual void put(Frame&& frame) = 0;
    };

    // Counts frames per camera and keeps none of them. Never refuses.
    //
    // The measurement harness, not a production sink. Ingest is the part of the system that
    // must hold 1000 frames a second, and the only way to measure *that* rather than the
    // scheduler behind it is to run against a consumer whose cost is one lock and one integer.
    class CountingSink : public FrameSink {
      public:
        void put(Frame&& frame) override {
            std::lock_guard<std::mutex> lock(mutex_);
            ++total_;
            ++per_camera_[frame.tag.camera_id];
        }

        uint64_t total() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return total_;
        }

        // Frames accepted, per camera — what a fairness assertion reads.
        std::map<std::string, uint64_t> counts() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return per_camera_;
        }

      private:
        mutable std::mutex mutex_;
        uint64_t total_ = 0;
        std::map<std::string, uint64_t> per_camera_;
    };

}  // namespace shipinfer
