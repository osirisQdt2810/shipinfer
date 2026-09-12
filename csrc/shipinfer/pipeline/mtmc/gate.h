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
// NOT THREAD-SAFE, deliberately, and `identity.h` says the same for the same reason: `hits_`
// is carried across `filter` calls, so this is state and not a pure function. The stage that
// owns it holds the barrier's lock across one whole instant, which is the level where one
// instant is one atomic step.
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
    //: How many consecutive instants a camera may be absent from before its tracks' streaks
    //: are dropped. Not a tuning knob but a bound: an absent camera says nothing about its
    //: tracks, and without a limit one that goes away for good would leave its streaks behind
    //: for the life of the process. At a 60 ms window 32 is about two seconds.
    inline constexpr int kDefaultMaxAbsentInstants = 32;

    struct GateOptions {
        int min_hits = kDefaultMinHits;
        double min_height_fraction = kDefaultMinHeightFraction;
        int max_absent_instants = kDefaultMaxAbsentInstants;
    };

    // The cameras a bare observation list can account for -- every one with a track in it.
    //
    // WEAKER THAN A ROSTER and named so callers see which they are using: a camera that
    // reported an EMPTY VIEW leaves no observation, so this cannot tell it from one that never
    // landed in the window. Only the barrier knows the difference. For drivers that genuinely
    // have nothing else, which is the parity harness and tests of the gate alone.
    std::vector<std::string> cameras_of(const std::vector<ClusterObservation>& observations);

    // Drops tracks that are too small, or too new to be trusted yet.
    //
    // doc: long what "consecutive" counts, and why an absent camera is not a miss
    // HOLDS TWO PIECES OF STATE: how many consecutive QUALIFYING observations each track has,
    // and how many consecutive instants its camera has been absent from since the last one.
    //
    // AN INSTANT ITS CAMERA WAS NOT IN IS NOT EVIDENCE AGAINST A TRACK. "Consecutive" is about
    // the track, not about the caller's clock: a synchronised instant holds whichever cameras
    // landed inside its window, and at fleet scale that is a fraction of them. Measured on a
    // 50-camera deployment: an instant held 11.8 cameras, so a camera appeared in 24% of
    // instants and three consecutive appearances happened 1.4% of the time -- the gate admitted
    // 2.2% of what it was offered and every global identity held exactly one track. So a streak
    // survives an instant its camera did not report in, and breaks when the camera WAS there
    // and the track was not, including when it was there and saw nothing.
    //
    // BOUNDED WITHOUT A POLICY THE CALLER TUNES: a key is dropped when its camera reported and
    // the track did not qualify, and after `max_absent_instants` consecutive instants its
    // camera was not in at all. So the maps hold the tracks in flight plus, briefly, those of a
    // camera that has just gone quiet.
    class ObservationGate {
      public:
        using Options = GateOptions;

        explicit ObservationGate(Options options = {});

        // The observations that may take part in association, in input order.
        //
        // `cameras` is the roster: every camera the instant HELD, empty views included. It is
        // REQUIRED here where the reference defaults it, and that difference is deliberate --
        // `gating.py` is public library API with callers this repository cannot see, while
        // this gate has one production caller. A defaulted roster would be a silent trapdoor
        // with nobody to justify it. `cameras_of()` is the explicit way to say "only the
        // observations", and it says which callers settled for that.
        std::vector<ClusterObservation> filter(
            const std::vector<ClusterObservation>& observations,
            const std::vector<std::string>& cameras);

        //: Consecutive qualifying observations for one track. Zero if it is not being held --
        //: which is what a caller reports as the REASON a track was published unidentified.
        int hits(const TrackKey& key) const;
        void reset();
        size_t size() const { return hits_.size(); }
        //: How many of those are being carried across an absence. For a test that wants to
        //: see the bound hold rather than infer it from `size()`.
        size_t absent_size() const { return absent_.size(); }
        int min_hits() const { return options_.min_hits; }
        double min_height_fraction() const { return options_.min_height_fraction; }
        int max_absent_instants() const { return options_.max_absent_instants; }

      private:
        Options options_;
        std::map<TrackKey, int> hits_;
        std::map<TrackKey, int> absent_;
    };

}  // namespace shipinfer::mtmc
