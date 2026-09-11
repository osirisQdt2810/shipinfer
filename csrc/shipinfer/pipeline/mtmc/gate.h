// Which tracks are worth asking about at all. The C++ twin of `shipvision/mtmc/gating.py`.
//
// doc: long the one failure both gates are about, and why their order is load-bearing
// TWO CHEAP GATES, applied before anything expensive, and both are about the same failure: a
// bad crop produces a CONFIDENT embedding. A person eight pixels tall, or one the single-camera
// tracker has only just noticed and may drop again next frame, gives the re-ID model too little
// to work with -- but the model does not say so, it returns a unit vector like any other, and
// that vector will happily score 0.9 against a stranger. Admitting it does not add a weak vote
// to the clustering, it adds a WRONG one, and once a global id has been merged across cameras
// nothing later un-merges it.
//
// HEIGHT FIRST, THEN AGE, and the order is the reference's. Age counts consecutive frames in
// which the track was ALSO large enough, so a figure walking in from the far distance starts
// accruing trust only once it is close enough to be worth trusting. Swap the two and a track
// banks three frames of age while it is unusably small, then enters the matrix on its first
// usable frame with the gate already satisfied.
//
// Pure: no lane, no device, no submodule -- so it builds and is tested in the offline tier.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
// `cluster.h` and not `identity.h`: the gate thresholds the BOX and the frame extent, which
// live on `ClusterObservation` because they are the geometry's. `identity.h` deliberately
// never sees them.
#include "shipinfer/pipeline/mtmc/cluster.h"

namespace shipinfer::mtmc {

    //: Consecutive qualifying observations before a track may take part in cross-camera
    //: association. The reference's production value.
    inline constexpr int kDefaultMinHits = 3;
    //: Minimum box height as a fraction of frame height. The reference's production value:
    //: a person must fill about a ninth of the frame, which at 1080p is 120 pixels -- roughly
    //: the smallest crop its re-ID model was trained to handle.
    inline constexpr double kDefaultMinHeightFraction = 1.0 / 9.0;

    struct GateOptions {
        int min_hits = kDefaultMinHits;
        double min_height_fraction = kDefaultMinHeightFraction;
    };

    // Drops tracks that are too small, or too new to be trusted yet.
    //
    // Holds one piece of state -- how many consecutive QUALIFYING frames each track has been
    // seen for -- and it is bounded by construction rather than by a policy: every key not seen
    // in the current instant is dropped at the end of the call, so the map can never hold more
    // than the tracks currently in flight. That also IS the definition of "consecutive".
    class ObservationGate {
      public:
        using Options = GateOptions;

        explicit ObservationGate(Options options = {});

        // The observations that may take part in association, in input order.
        std::vector<ClusterObservation> filter(
            const std::vector<ClusterObservation>& observations);

        //: Consecutive qualifying observations for one track. Zero if it is not being held --
        //: which is what a caller reports as the REASON a track was published unidentified.
        int hits(const TrackKey& key) const;
        void reset();
        size_t size() const { return hits_.size(); }
        int min_hits() const { return options_.min_hits; }
        double min_height_fraction() const { return options_.min_height_fraction; }

      private:
        Options options_;
        std::map<TrackKey, int> hits_;
    };

}  // namespace shipinfer::mtmc
