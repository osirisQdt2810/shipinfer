// One tracker per camera, and the ordering guard that makes that safe.
//
// The C++ twin of `topology/elements/track.py`'s `TrackerShard`. The sharding is a
// CORRECTNESS constraint and not a scaling one: Kalman state, track ids and ageing all belong
// to one camera's view, so two cameras on one tracker associate one camera's objects with the
// other's and report a real identity where nothing happened.
//
// The ordering guard is here for the same reason it is there. The fair lane preserves a
// camera's order into the pipeline, but a pool of workers does not preserve it out again --
// and tracking is stateful, so replaying a frame double-ages every track and double-counts
// the hit that promotes one. A frame that does not advance its camera's stream is refused,
// and the graph publishes it without track ids rather than with wrong ones.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipvision/mot/trackers/bytetrack/tracker.h"

namespace shipinfer {

    //: How far a camera's `frame_id` may go BACKWARDS before this stops calling it a
    //: reordering and starts calling it a restarted stream. The Python element's default, and
    //: the same reasoning: a reconnect means the ids must not continue, and refusing every
    //: frame of a camera until the process ends is the worse failure.
    inline constexpr int64_t kRegressionReset = 64;

    // What one camera's tracker did with one frame. `ids` is parallel to the detections it
    // was given -- `-1` where a detection matched no confirmed track -- because the caller
    // needs to put an id back on a row, not to reason about the pool.
    struct TrackUpdate {
        std::vector<int> ids;
        size_t confirmed = 0;
    };

    // Counters an operator needs to tell a reordering from a reconnect. Both are rare and
    // both mean something different, which is why they are two numbers and not one.
    struct TrackerShardStats {
        uint64_t out_of_order = 0;
        uint64_t implicit_resets = 0;
        size_t cameras = 0;
    };

    class TrackerShard {
      public:
        using OnImplicitReset = std::function<void(const std::string&)>;

        explicit TrackerShard(shipvision::mot::ByteTrackTracker::Options options = {},
                              int64_t regression_reset = kRegressionReset)
            : options_(options), regression_reset_(regression_reset) {}

        // Advance one camera's tracker by one frame.
        //
        // THROWS `InferenceError` when the frame does not advance the stream and the
        // regression is small enough to be a reordering -- per frame, caught where the event
        // is built, so one bad frame is named rather than ending a worker (`core/types.h`).
        TrackUpdate update(const std::string& camera_id, int64_t frame_id,
                           const std::vector<Detection>& detections,
                           const OnImplicitReset& on_implicit_reset = {});

        // Forget one camera's tracks and its stream position, WITHOUT building a tracker for a
        // camera that has none. What a lifecycle hook wants: `camera_added` fires for every
        // camera on the shard, and minting a Kalman filter and a track pool for the
        // forty-nine that were never on this element -- on the thread holding the runner's
        // lifecycle lock -- is work for nothing. Returns whether there was one to reset.
        bool reset_if_present(const std::string& camera_id);

        TrackerShardStats stats() const;

      private:
        struct Camera {
            std::mutex lock;
            std::unique_ptr<shipvision::mot::ByteTrackTracker> tracker;
            //: The highest `frame_id` fed to this tracker. `-1` because a frame id may be 0.
            int64_t last_frame_id = -1;
            uint64_t out_of_order = 0;
            uint64_t implicit_resets = 0;
        };

        // Guards insertion into `cameras_` ONLY. Never held across `tracker.update`, so a slow
        // camera cannot stall the other forty-nine, and never held across a camera's own lock,
        // so a lifecycle call cannot queue behind one camera's frame.
        Camera& camera_for(const std::string& camera_id);

        shipvision::mot::ByteTrackTracker::Options options_;
        int64_t regression_reset_;
        mutable std::mutex map_lock_;
        //: `unique_ptr` because `Camera` holds a mutex and is therefore immovable, and a
        //: rehash of a map of values would have to move them. The pointer is stable for the
        //: shard's life, which is what lets a caller hold `Camera&` outside `map_lock_`.
        std::map<std::string, std::unique_ptr<Camera>> cameras_;
    };

}  // namespace shipinfer
