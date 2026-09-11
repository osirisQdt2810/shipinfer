// What a cross-camera tracker does, with nothing of one in it: an instant in, global ids out.
//
// The same seam `tracking/associator.h` is, for the same two build lines, and the reasoning
// there applies verbatim: `3rdparty/shipvision` is an external lane, so a header that reaches
// it drags the lane into every includer; and `core/platform.h` is the CUDA line, which
// `--offline` refuses. A stage reads a `FrameState`, a matcher reaches the lane, so they cannot
// be one unit. This interface is the seam -- `TrackKey`, embeddings and boxes, all from
// `core/types.h` and `mtmc/identity.h`, which cross neither line.
//
// WHY THE INPUT IS A WHOLE INSTANT. `shipvision.mtmc` consumes every camera of a group at one
// synchronised instant and refuses anything less, because handing it one camera at a time turns
// cross-camera association into within-camera deduplication. `mtmc/barrier.h` is what turns the
// chain's one-frame-at-a-time stream back into instants; this is what gives those instants
// meaning.
// NOT THREAD-SAFE at the interface, and an implementation that is says so itself: the lane
// unit takes its own mutex because ONE tracker is shared by every worker on a slot. What the
// interface promises is only that `ids()` is called with a whole instant.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/mtmc/identity.h"

namespace shipinfer::mtmc {

    // One camera's track at one instant: who it is, what it looks like, where it is.
    //
    // The box and the frame extent are here and not in `IdentityObservation` because they are
    // the GEOMETRY's, not the identity map's: the gated matcher projects them to ground
    // positions and vetoes pairs no single object could occupy. `identity.h` never sees them.
    struct ClusterObservation {
        TrackKey key;
        std::vector<float> embedding;
        //: xyxy in the absolute pixels of the frame it was measured in.
        float box[4] = {0.f, 0.f, 0.f, 0.f};
        int frame_width = 0;
        int frame_height = 0;
        //: Which DETECTION row this came from, carried through so the caller can scatter the
        //: answer back. The tracker never reads it -- identity is keyed by (camera, track),
        //: and a detection index means nothing outside the frame that produced it.
        int box_index = -1;
    };

    // Cross-camera identities for one instant. Stateful across calls by definition -- that is
    // what makes an id an identity rather than a label.
    class ClusterTracker {
      public:
        virtual ~ClusterTracker() = default;

        // Every observation's key mapped to its global id. NEVER PARTIAL: an observation that
        // reached here is a track the caller decided to trust, so one that matches nothing
        // starts a new identity rather than being dropped -- the reference left such tracks at
        // `-1` and published them.
        virtual std::map<TrackKey, int64_t> ids(
            const std::vector<ClusterObservation>& instant) = 0;

        // How many identities and tracks are live. For the run's report, and for a test that
        // wants to see the bounds hold.
        virtual IdentitySizes sizes() const = 0;
    };

    using ClusterTrackerFactory = std::function<std::shared_ptr<ClusterTracker>()>;

    class ClusterRegistry {
      public:
        void add(const std::string& impl, ClusterTrackerFactory factory);
        bool has(const std::string& impl) const;
        std::shared_ptr<ClusterTracker> create(const std::string& impl) const;
        std::vector<std::string> names() const;

      private:
        std::map<std::string, ClusterTrackerFactory> entries_;
    };

    // Function-local static: every unit's registrar can run before `main`, in any order.
    ClusterRegistry& CLUSTERERS();

    // The tracker for `impl` on `slot`, or a refusal that says whether the name is unknown or
    // the LANE is absent -- two problems with two fixes.
    //
    // ONE PER (impl, slot), CACHED FOR THE PROCESS, for the reason `create_associator` gives
    // and one worse: a cross-camera tracker is the identity space for a whole GROUP, so a
    // second instance would issue a second, contradictory set of global ids for the same
    // objects. Two `mtmc` slots are two groups and get two.
    std::shared_ptr<ClusterTracker> create_cluster_tracker(const std::string& impl,
                                                           const std::string& slot);

    struct MadeClusterTracker {
        std::string impl;
        std::string slot;
        std::shared_ptr<ClusterTracker> tracker;
    };
    std::vector<MadeClusterTracker> made_cluster_trackers();

    // One of these at the bottom of a tracker's unit is the `@CLUSTERERS.register(...)`.
    struct ClusterRegistrar {
        ClusterRegistrar(const std::string& impl, ClusterTrackerFactory factory) {
            CLUSTERERS().add(impl, std::move(factory));
        }
    };

}  // namespace shipinfer::mtmc
