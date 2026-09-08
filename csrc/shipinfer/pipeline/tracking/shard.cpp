#include "shipinfer/pipeline/tracking/shard.h"

#include <string>
#include <utility>

namespace shipinfer {

    namespace {

        // shipinfer's box is four floats and a class; shipvision's is an array and a class.
        // One conversion in one place: the tracker's vocabulary is its own and the graph's
        // `Detection` carries an `index` the tracker has no use for.
        std::vector<shipvision::mot::Detection> as_mot(const std::vector<Detection>& from) {
            std::vector<shipvision::mot::Detection> out;
            out.reserve(from.size());
            for (const Detection& d : from) {
                shipvision::mot::Detection m;
                m.box[0] = d.x1;
                m.box[1] = d.y1;
                m.box[2] = d.x2;
                m.box[3] = d.y2;
                m.score = d.score;
                m.class_id = d.class_id;
                out.push_back(m);
            }
            return out;
        }

    }  // namespace

    TrackerShard::Camera& TrackerShard::camera_for(const std::string& camera_id) {
        std::lock_guard<std::mutex> held(map_lock_);
        std::unique_ptr<Camera>& slot = cameras_[camera_id];
        if (!slot) {
            slot = std::make_unique<Camera>();
            slot->tracker = std::make_unique<shipvision::mot::ByteTrackTracker>(options_);
        }
        return *slot;
    }

    TrackUpdate TrackerShard::update(const std::string& camera_id, int64_t frame_id,
                                     const std::vector<Detection>& detections,
                                     const OnImplicitReset& on_implicit_reset) {
        Camera& camera = camera_for(camera_id);
        std::lock_guard<std::mutex> held(camera.lock);
        if (frame_id <= camera.last_frame_id) {
            const int64_t behind = camera.last_frame_id - frame_id;
            if (regression_reset_ > 0 && behind >= regression_reset_) {
                // A restarted stream nobody announced. Forget the tracks and take this frame
                // as the new stream's first: the ids restart, which is what a reconnect means.
                camera.tracker->reset();
                camera.last_frame_id = -1;
                ++camera.implicit_resets;
                if (on_implicit_reset) on_implicit_reset(camera_id);
            } else {
                ++camera.out_of_order;
                throw InferenceError(
                    "camera '" + camera_id + "': frame " + std::to_string(frame_id) +
                    " reached the tracker after frame " + std::to_string(camera.last_frame_id) +
                    ". Tracking is stateful and ordered; replaying a frame double-ages every "
                    "track and double-counts the hit that promotes one, so this frame is "
                    "published without track ids rather than with wrong ones");
            }
        }
        camera.last_frame_id = frame_id;

        const std::vector<shipvision::mot::Track> tracks =
            camera.tracker->update(as_mot(detections));

        // BACK ONTO THE ROWS, which is the only shape a caller wants: `Track::last_match` is
        // the index of the detection that corrected the track, in the list `update` was
        // handed. A track that no detection corrected this frame (a coasting Kalman
        // prediction) has -1 there and therefore claims no row.
        //
        // `is_publishable()` AS WELL, and its own header says why rather than this one: a LOST
        // track's box is a prediction no detector saw, and emitting it as an observation is
        // how a phantom object drifts across a scene. `update` already returns confirmed-and-
        // seen tracks, so this is defence in depth against that contract changing under us --
        // cheap, and the failure it guards is invisible until two identities swap.
        TrackUpdate out;
        out.ids.assign(detections.size(), -1);
        for (const shipvision::mot::Track& track : tracks) {
            if (!track.is_publishable()) continue;
            ++out.confirmed;
            const int row = track.last_match;
            if (row >= 0 && static_cast<size_t>(row) < out.ids.size()) {
                out.ids[static_cast<size_t>(row)] = track.track_id;
            }
        }
        return out;
    }

    bool TrackerShard::reset_if_present(const std::string& camera_id) {
        Camera* camera = nullptr;
        {
            std::lock_guard<std::mutex> held(map_lock_);
            const auto found = cameras_.find(camera_id);
            if (found == cameras_.end()) return false;
            camera = found->second.get();
        }
        // OUTSIDE `map_lock_`: this waits for a frame in flight on that camera, and holding
        // the map lock while it does would stall every other camera's first frame.
        std::lock_guard<std::mutex> held(camera->lock);
        camera->tracker->reset();
        camera->last_frame_id = -1;
        return true;
    }

    TrackerShardStats TrackerShard::stats() const {
        TrackerShardStats out;
        std::lock_guard<std::mutex> held(map_lock_);
        out.cameras = cameras_.size();
        for (const auto& [name, camera] : cameras_) {
            (void)name;
            // Read WITHOUT the camera's lock, deliberately: a counter read one frame stale is
            // a metric, and taking fifty per-camera locks to total two numbers would put a
            // metrics scrape in the path of every camera's frame.
            out.out_of_order += camera->out_of_order;
            out.implicit_resets += camera->implicit_resets;
        }
        return out;
    }

}  // namespace shipinfer
