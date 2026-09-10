// `TRACKERS`, the tracking-stage registry — and the reason it exists is a build fact.
//
// A tracker reaches `3rdparty/shipvision`, which is an EXTERNAL LANE: `scripts/build_csrc.py`
// compiles a unit reaching outside this tree only when that unit's lane is in the build. The
// build script follows a header to the `.cpp` beside it, so a single `#include
// "pipeline/tracking/shard.h"` from `graph/stages.h` drags shipvision into the closure of the
// WHOLE graph plane -- measured: `stages.o` and `from_plan.o` were dropped from a build without
// the submodule and the binary failed to link at all. `ingest/registry.h` states the same
// invariant for the same reason, and this is its answer applied to a stage.
//
// So the graph asks for a tracker BY NAME and includes nothing of one. A build without the lane
// has no registrar, `create_track_stage` refuses NAMING THE LANE (`ingest/omitted_lanes.h`),
// and every other stage still builds -- which is the promise "a machine with no build still
// runs" cashed out for this plane.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "shipinfer/pipeline/graph/plan_stages.h"
#include "shipinfer/pipeline/graph/stage.h"

namespace shipinfer::tracking {

    //: One tracker implementation's stage, for one plan's track slot.
    using TrackStageFactory = std::function<std::unique_ptr<Stage>(const TrackStageSpec&)>;

    class TrackerRegistry {
      public:
        void add(const std::string& impl, TrackStageFactory factory);
        bool has(const std::string& impl) const;
        std::unique_ptr<Stage> create(const std::string& impl,
                                      const TrackStageSpec& spec) const;
        std::vector<std::string> names() const;

      private:
        std::map<std::string, TrackStageFactory> entries_;
    };

    // Function-local static: every unit's registrar can run before `main`, in any order, and
    // still find the one registry.
    TrackerRegistry& TRACKERS();

    // The stage for `impl`, or a refusal that says whether the name is unknown or the LANE is
    // absent -- two different problems with two different fixes, which is the distinction
    // `ingest/registry.h` was rewritten to make.
    std::unique_ptr<Stage> create_track_stage(const std::string& impl,
                                              const TrackStageSpec& spec);

    // One of these at the bottom of a tracker's unit is the `@TRACKERS.register(...)`.
    struct TrackerRegistrar {
        TrackerRegistrar(const std::string& impl, TrackStageFactory factory) {
            TRACKERS().add(impl, std::move(factory));
        }
    };

}  // namespace shipinfer::tracking
