// The tracking stage, in the LANE's own unit — see `tracking/registry.h` for why it is not in
// `graph/stages.h`. Nothing in the graph plane includes this header.
#pragma once

#include <memory>
#include <string>

#include "shipinfer/pipeline/graph/plan_stages.h"
#include "shipinfer/pipeline/graph/stage.h"
#include "shipinfer/pipeline/tracking/shard.h"

namespace shipinfer::tracking {

    // One camera's tracker over one frame's detections, as an `ObjectBatch` of ids.
    //
    // NOT a `ModelStage`: there is no engine and no queue, so it runs on the worker's own
    // thread like `CropStage`. The shard is SHARED by every worker -- one tracker per camera is
    // the correctness constraint, and `TrackerShard` is built to be called from several threads
    // with one lock per camera.
    //
    // A DETECTION THE TRACKER DID NOT CONFIRM gets no row, which leaves its `track_id` null
    // rather than `-1`. That is what the Python plane emits for the same case, and the chain
    // file's own comment says why it happens at all: a detection between the publish threshold
    // and the tracker's own only ever CONTINUES a track, never starts one.
    class TrackStage : public Stage {
      public:
        TrackStage(std::string name, std::string output, TrackerShard& shard);

      protected:
        size_t do_run(FrameState& state) override;

      private:
        std::string output_;
        TrackerShard& shard_;
    };

    // The one shard this process tracks with. A function-local static because one tracker per
    // camera is the constraint and several workers share it -- and because a shard owned by a
    // Dag would be one per worker, which is the identity-splitting bug `shard.h` exists about.
    TrackerShard& process_shard();

}  // namespace shipinfer::tracking
