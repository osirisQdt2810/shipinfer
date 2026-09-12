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
        if (options_.max_absent_instants < 1) {
            throw ConfigError(
                "max_absent_instants must be at least 1; 0 would drop a streak the instant "
                "its camera missed one instant, which is the behaviour this replaced, got " +
                std::to_string(options_.max_absent_instants));
        }
    }

    std::vector<std::string> cameras_of(const std::vector<ClusterObservation>& observations) {
        std::set<std::string> unique;
        for (const ClusterObservation& observation : observations) {
            unique.insert(observation.key.camera_id);
        }
        return {unique.begin(), unique.end()};
    }

    int ObservationGate::hits(const TrackKey& key) const {
        const auto found = hits_.find(key);
        return found == hits_.end() ? 0 : found->second;
    }

    void ObservationGate::reset() {
        hits_.clear();
        absent_.clear();
    }

    std::vector<ClusterObservation> ObservationGate::filter(
        const std::vector<ClusterObservation>& observations,
        const std::vector<std::string>& cameras) {
        std::map<TrackKey, int> hits;
        std::map<TrackKey, int> absent;
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
        // CARRIED, not broken: a camera this instant did not hold said nothing about its
        // tracks. `present` is the caller's roster and not these observations, so a camera
        // that reported an empty view is here and breaks its streaks like any other.
        const std::set<std::string> present(cameras.begin(), cameras.end());
        for (const auto& [key, count] : hits_) {
            if (present.count(key.camera_id) != 0) continue;
            const auto missed_before = absent_.find(key);
            const int missed = (missed_before == absent_.end() ? 0 : missed_before->second) + 1;
            if (missed <= options_.max_absent_instants) {
                hits[key] = count;
                absent[key] = missed;
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
            absent.erase(observation.key);
            if (count >= options_.min_hits) admitted.push_back(observation);
        }
        // REPLACED, not pruned. That is what enforces "consecutive" -- a track whose camera
        // WAS here and did not qualify is simply not copied across -- and what keeps the maps
        // bounded by the tracks in flight rather than by uptime.
        hits_ = std::move(hits);
        absent_ = std::move(absent);
        return admitted;
    }

}  // namespace shipinfer::mtmc
