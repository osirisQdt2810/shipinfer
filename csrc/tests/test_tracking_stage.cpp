// `TrackStage`'s SCATTER — which rows get an id, which get none, and whose index they carry.
//
// AGAINST A FAKE ASSOCIATOR, deliberately. The tracker's own behaviour is
// `test_tracking_associator.cpp` and shipvision's suite; what this stage owns is the mapping
// from ids to `ObjectBatch` rows, and a fake makes `-1` reachable on demand instead of hoping
// ByteTrack declines to confirm. It also keeps this test out of the external lane -- it reaches
// `core/platform.h` through `graph/state.h`, so it could not live in the lane's offline job.
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/pipeline/graph/emission.h"
#include "shipinfer/pipeline/graph/stages.h"
#include "shipinfer/pipeline/graph/state.h"
#include "shipinfer/pipeline/tracking/associator.h"

namespace {

    using namespace shipinfer;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    // Answers whatever it was told to, so `-1` is reachable without a real tracker declining.
    class ScriptedAssociator : public tracking::Associator {
      public:
        explicit ScriptedAssociator(std::vector<int> answer) : answer_(std::move(answer)) {}

        std::vector<int> ids(const std::string& camera_id, int64_t frame_id,
                             const std::vector<Detection>& detections) override {
            seen_camera = camera_id;
            seen_frame = frame_id;
            seen_detections = detections.size();
            return answer_;
        }

        std::string seen_camera;
        int64_t seen_frame = 0;
        size_t seen_detections = 0;

      private:
        std::vector<int> answer_;
    };

    Detection box(float x) {
        Detection det;
        det.x1 = x;
        det.y1 = 10.0f;
        det.x2 = x + 40.0f;
        det.y2 = 90.0f;
        det.score = 0.9f;
        return det;
    }

    std::unique_ptr<FrameState> frame_with(const std::string& camera, int64_t frame_id,
                                           std::vector<Detection> detections) {
        FrameTag tag;
        tag.camera_id = camera;
        tag.frame_id = frame_id;
        auto state = std::make_unique<FrameState>(tag, 1080, 1920, 20.0f);
        state->set_detections(std::move(detections));
        return state;
    }

    void every_confirmed_id_lands_on_its_own_detections_index() {
        auto associator = std::make_shared<ScriptedAssociator>(std::vector<int>{7, 8, 9});
        TrackStage stage("track", "track_out", associator);
        auto state = frame_with("cam0", 42, {box(0), box(100), box(200)});

        const size_t rows = stage.run(*state).rows;
        const ObjectBatch* batch = state->batch("track_out");

        check(rows == 3, "three ids, three rows");
        check(batch != nullptr, "attached under the output name");
        if (batch == nullptr) return;
        check(batch->width == 1, "one id per row");
        check(batch->object_indices == std::vector<int>({0, 1, 2}), "the detections' indices");
        check(batch->row(0)[0] == 7.0f && batch->row(2)[0] == 9.0f, "in order");
    }

    void an_unmatched_detection_gets_no_row_at_all() {
        // The middle one matched nothing. A `-1` in the batch would arrive in an event as
        // `track_id: -1`, which is a lie a consumer cannot tell from an id.
        auto associator = std::make_shared<ScriptedAssociator>(std::vector<int>{7, -1, 9});
        TrackStage stage("track", "track_out", associator);
        auto state = frame_with("cam0", 1, {box(0), box(100), box(200)});

        const size_t rows = stage.run(*state).rows;
        const ObjectBatch* batch = state->batch("track_out");

        check(rows == 2, "two rows for three detections");
        check(batch != nullptr && batch->object_indices == std::vector<int>({0, 2}),
              "and the surviving indices are 0 and 2, not 0 and 1");
    }

    void the_camera_and_frame_reach_the_associator() {
        // The sharding is per CAMERA and the ordering guard is per FRAME, so both have to
        // arrive -- a stage that passed the wrong camera would merge two views silently.
        auto associator = std::make_shared<ScriptedAssociator>(std::vector<int>{1});
        TrackStage stage("track", "track_out", associator);
        auto state = frame_with("cam-seven", 99, {box(0)});

        stage.run(*state);

        check(associator->seen_camera == "cam-seven", "the frame's camera");
        check(associator->seen_frame == 99, "the frame's id");
        check(associator->seen_detections == 1, "and its detections");
    }

    void a_frame_with_no_detections_attaches_an_empty_batch() {
        auto associator = std::make_shared<ScriptedAssociator>(std::vector<int>{});
        TrackStage stage("track", "track_out", associator);
        auto state = frame_with("cam0", 1, {});

        const size_t rows = stage.run(*state).rows;
        const ObjectBatch* batch = state->batch("track_out");

        check(rows == 0, "no rows");
        check(batch != nullptr && batch->empty(), "the name exists and is empty");
    }

    void more_ids_than_detections_cannot_walk_off_the_end() {
        // Not a caller this tree has; a tracker that answered long would be a crash otherwise.
        auto associator = std::make_shared<ScriptedAssociator>(std::vector<int>{1, 2, 3, 4});
        TrackStage stage("track", "track_out", associator);
        auto state = frame_with("cam0", 1, {box(0)});

        const size_t rows = stage.run(*state).rows;

        check(rows == 1, "bounded by the detections, not by the answer");
    }

}  // namespace

int main() {
    every_confirmed_id_lands_on_its_own_detections_index();
    an_unmatched_detection_gets_no_row_at_all();
    the_camera_and_frame_reach_the_associator();
    a_frame_with_no_detections_attaches_an_empty_batch();
    more_ids_than_detections_cannot_walk_off_the_end();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
