// shipvision's ByteTrack behind the `Associator` seam — the LANE's only unit besides the shard.
//
// Nothing here reaches `core/platform.h`, which is what lets the lane's CI job build it with
// g++ alone (`--offline --with-external shipvision`). The stage that reads a `FrameState` is on
// the other side of that line, in `graph/stages.cpp`.
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/pipeline/tracking/associator.h"
#include "shipinfer/pipeline/tracking/shard.h"

namespace shipinfer::tracking {

    namespace {

        //: A tracker option's value, converted where the lane's key table knows its type. The
        //: plan carries strings because it is a line format; a bad one is refused naming the
        //: key, since "max_age must be a number" is actionable and a silent 30 is not.
        int as_int(const std::string& key, const std::string& value) {
            try {
                size_t used = 0;
                const int out = std::stoi(value, &used);
                if (used == value.size()) return out;
            } catch (const std::exception&) {
            }
            throw ConfigError("tracker option '" + key + "' must be a whole number, got '" +
                              value + "'");
        }

        float as_float(const std::string& key, const std::string& value) {
            try {
                size_t used = 0;
                const float out = std::stof(value, &used);
                if (used == value.size()) return out;
            } catch (const std::exception&) {
            }
            throw ConfigError("tracker option '" + key + "' must be a number, got '" + value +
                              "'");
        }

        bool as_bool(const std::string& key, const std::string& value) {
            if (value == "true" || value == "1") return true;
            if (value == "false" || value == "0") return false;
            throw ConfigError("tracker option '" + key + "' must be true or false, got '" +
                              value + "'");
        }

        // doc: long the key table, and why an unknown key is refused rather than dropped
        // THE LANE OWNS ITS KEYS, which is why the conversion is here and not in the plan
        // reader: the reader would have to drop what it does not recognise, and a dropped
        // `max_age` is a chain that loads, reports `track` as having run, and emits different
        // ids from the other plane -- exactly the divergence this closes.
        //
        // REFUSED, not ignored, mirroring `TrackerShard.__init__` on the other plane: a
        // `max_ago: 90` typo is a tracker running 30 with nothing said, and the failure is
        // invisible because every frame still gets an id.
        shipvision::mot::ByteTrackTracker::Options byte_track_options(
            const std::map<std::string, std::string>& stated) {
            shipvision::mot::ByteTrackTracker::Options built;
            for (const auto& [key, value] : stated) {
                if (key == "track_threshold") {
                    built.track_threshold = as_float(key, value);
                } else if (key == "low_threshold") {
                    built.low_threshold = as_float(key, value);
                } else if (key == "match_threshold") {
                    built.match_threshold = as_float(key, value);
                } else if (key == "second_match_threshold") {
                    built.second_match_threshold = as_float(key, value);
                } else if (key == "max_age") {
                    built.max_age = as_int(key, value);
                } else if (key == "min_hits") {
                    built.min_hits = as_int(key, value);
                } else if (key == "gate") {
                    built.gate = as_bool(key, value);
                } else {
                    throw ConfigError(
                        "tracker option '" + key +
                        "' is not one bytetrack has; it takes track_threshold, low_threshold, "
                        "match_threshold, second_match_threshold, max_age, min_hits, gate");
                }
            }
            return built;
        }

        // ONE SHARD PER ASSOCIATOR, and `create_associator` decides how many associators
        // there are: one per (impl, slot), shared by every worker. The sharing axis is not
        // this file's to choose -- it was, and choosing "one per process" here gave two
        // tracking slots one shard keyed by camera, so the second slot's every frame was
        // refused as out of order.
        class ShardAssociator : public Associator {
          public:
            explicit ShardAssociator(const TrackerOptions& options)
                : shard_(byte_track_options(options.options),
                         options.regression_reset.value_or(kRegressionReset)) {}

            std::vector<int> ids(const std::string& camera_id, int64_t frame_id,
                                 const std::vector<Detection>& detections) override {
                return shard_.update(camera_id, frame_id, detections).ids;
            }

          private:
            TrackerShard shard_;
        };

        // `impl: shipvision` in the chain, which is the name the plan carries. The algorithm
        // inside it (`params: algorithm: bytetrack`) is the only one this lane has, so the plan
        // writer does not emit it and this does not read it.
        const AssociatorRegistrar kShipvision("shipvision", [](const TrackerOptions& options) {
            return std::make_shared<ShardAssociator>(options);
        });

    }  // namespace

}  // namespace shipinfer::tracking
