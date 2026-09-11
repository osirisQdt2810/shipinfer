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

#include <atomic>
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
    //: "This observation has no identity", which is a different fact from "this observation
    //: does not exist": the gate admits nothing for a track that is too small or too new, and
    //: `records.cpp` leaves the event's `global_id` null for it. NAMED so a second
    //: implementation cannot pick a different sentinel -- `assign()` only ever issues
    //: non-negative ids, so -1 is unambiguous (#221 round 3).
    inline constexpr int64_t kUnidentified = -1;

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

        // AN INSTANT THIS SEAM REFUSED, counted rather than lost -- the same shape
        // `Associator::note_untracked` has and for the same reason: one camera's malformed row
        // (a duplicate key, a zero embedding, a second width) is an ordinary outcome of real
        // data, and the stage publishes the group's frames with null ids instead of failing
        // the frame that happened to close the bucket. The caller counts it HERE, because one
        // tracker serves every worker on a slot and this is the run's answer.
        void note_refused() { refused_.fetch_add(1, std::memory_order_relaxed); }
        uint64_t refused_instants() const { return refused_.load(std::memory_order_relaxed); }

        // HOW MUCH OF AN INSTANT SURVIVED THE GATE, counted by the implementation because it
        // is the only thing that knows. Without it "no identities" is indistinguishable from
        // "nothing to identify": a site whose boxes are all under `min_height_fraction`, or a
        // stream where no track is present in `min_hits` CONSECUTIVE instants, admits nothing
        // and issues nothing -- and the run used to report that as `mtmc_identities 0 0` with
        // no way to tell it from a barrier that never closed. Measured: that is exactly what
        // happened (`MTMC-GATE-ADMITS-NOTHING-AT-THE-MEASURED-LOAD`).
        void note_instant(uint64_t offered, uint64_t admitted) {
            offered_.fetch_add(offered, std::memory_order_relaxed);
            admitted_.fetch_add(admitted, std::memory_order_relaxed);
        }
        uint64_t observations_offered() const {
            return offered_.load(std::memory_order_relaxed);
        }
        uint64_t observations_admitted() const {
            return admitted_.load(std::memory_order_relaxed);
        }

      private:
        std::atomic<uint64_t> refused_{0};
        std::atomic<uint64_t> offered_{0};
        std::atomic<uint64_t> admitted_{0};
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
