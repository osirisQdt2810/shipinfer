#include "shipinfer/pipeline/mtmc/gate.h"

#include <set>

namespace shipinfer::mtmc {

    ObservationGate::ObservationGate(Options options) : options_(options) {
        if (options_.min_hits < 1) {
            throw ConfigError(
                "min_hits must be at least 1; 0 would admit a track on the frame "
                "it was first seen, got " +
                std::to_string(options_.min_hits));
        }
        if (!(options_.min_height_fraction >= 0.0) || options_.min_height_fraction >= 1.0) {
            throw ConfigError(
                "min_height_fraction is a fraction of frame height and must be in "
                "[0, 1), got " +
                std::to_string(options_.min_height_fraction));
        }
    }

    int ObservationGate::hits(const TrackKey& key) const {
        const auto found = hits_.find(key);
        return found == hits_.end() ? 0 : found->second;
    }

    void ObservationGate::reset() {
        hits_.clear();
    }

    std::vector<ClusterObservation> ObservationGate::filter(
        const std::vector<ClusterObservation>& observations) {
        std::map<TrackKey, int> hits;
        std::vector<ClusterObservation> admitted;
        // ONE ROW PER KEY, refused rather than deduplicated. Both copies would have advanced
        // the same run once (correct) and then both been admitted, and the caller zips
        // `admitted` against the clusterer's labels -- so one track would occupy two rows of
        // the matrix and the identity map would see it contest itself. The barrier upstream
        // cannot produce this; a single-camera tracker emitting one track twice in a frame
        // can, and that is a tracking fault to name rather than to average over.
        std::set<TrackKey> seen_here;
        for (const ClusterObservation& observation : observations) {
            if (!seen_here.insert(observation.key).second) {
                throw InferenceError(observation.key.str() +
                                     " appears twice in one instant; a cross-camera instant "
                                     "holds one row per (camera, track), and two rows for one "
                                     "track would have it contest itself");
            }
        }
        for (const ClusterObservation& observation : observations) {
            // A ZERO FRAME HEIGHT admits nothing rather than dividing by it: a frame whose
            // extent nobody filled in is a wiring fault, and letting every track through it
            // would put unusable crops into the matrix -- the exact failure this gate exists
            // for. `frames.h` makes the extent a field and not an optional for the same reason.
            if (observation.frame_height <= 0) continue;
            const double fraction =
                static_cast<double>(observation.box[3] - observation.box[1]) /
                static_cast<double>(observation.frame_height);
            if (!(fraction > options_.min_height_fraction)) continue;
            // AGE COUNTS ONLY QUALIFYING FRAMES, which is why this runs after the height test
            // and not beside it.
            const auto previous = hits_.find(observation.key);
            const int count = (previous == hits_.end() ? 0 : previous->second) + 1;
            hits[observation.key] = count;
            if (count >= options_.min_hits) admitted.push_back(observation);
        }
        // REPLACED, not pruned. That is what enforces "consecutive" -- a track that misses one
        // instant starts again from one -- and what keeps the map bounded by the tracks in
        // flight rather than by uptime.
        hits_ = std::move(hits);
        return admitted;
    }

}  // namespace shipinfer::mtmc
