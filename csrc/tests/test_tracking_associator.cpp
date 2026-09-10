// The `Associator` seam and its registry — the LANE side, and offline by design.
//
// `tracking/associator.h` explains why this file exists apart from the stage's: a tracker
// reaches an external lane and a stage reaches `core/platform.h`, and the lane's CI job builds
// with g++ alone. So the tracker is tested here, without a `FrameState` anywhere in sight, and
// the stage's scatter is tested in `test_tracking_stage.cpp` against a fake.
#include <cstdio>
#include <string>
#include <vector>

#include "shipinfer/pipeline/tracking/associator.h"

namespace {

    using namespace shipinfer;
    using shipinfer::tracking::Associator;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    Detection box(float x, float y, float score, int class_id) {
        Detection det;
        det.x1 = x;
        det.y1 = y;
        det.x2 = x + 40.0f;
        det.y2 = y + 80.0f;
        det.score = score;
        det.class_id = class_id;
        return det;
    }

    void the_lanes_tracker_registers_itself() {
        // This binary exists only when the lane is in the build, so the registrar HAS run.
        check(tracking::ASSOCIATORS().has("shipvision"), "shipvision is registered");
        check(tracking::create_associator("shipvision") != nullptr, "and it builds");
    }

    void one_associator_serves_every_caller() {
        // One tracker per camera is the constraint, so two callers asking for the same impl
        // must get the SAME associator -- two would be two identity spaces for one camera.
        const auto first = tracking::create_associator("shipvision");
        const auto second = tracking::create_associator("shipvision");

        check(first.get() == second.get(), "the same associator, not a second identity space");
    }

    void a_confirmed_track_gets_a_positive_id() {
        const auto associator = tracking::create_associator("shipvision");
        std::vector<int> last;
        // ByteTrack confirms on a later hit, so one frame is not enough -- which is the
        // tracker's business and is exactly why the STAGE is tested against a fake instead.
        for (int64_t frame = 1; frame <= 4; ++frame) {
            last = associator->ids("cam-assoc", frame, {box(100, 100, 0.9f, 0)});
            check(last.size() == 1, "one id per detection, always");
        }
        check(!last.empty() && last[0] >= 1, "a repeated detection is eventually confirmed");
    }

    void a_frame_with_no_detections_answers_nothing() {
        const auto associator = tracking::create_associator("shipvision");

        check(associator->ids("cam-empty", 1, {}).empty(), "no detections, no ids");
    }

    void an_unknown_impl_is_refused_by_name() {
        bool refused = false;
        try {
            tracking::create_associator("no_such_tracker");
        } catch (const ConfigError& error) {
            refused = std::string(error.what()).find("unknown tracker") != std::string::npos;
        }

        check(refused, "an unknown impl is a ConfigError naming it");
    }

}  // namespace

int main() {
    the_lanes_tracker_registers_itself();
    one_associator_serves_every_caller();
    a_confirmed_track_gets_a_positive_id();
    a_frame_with_no_detections_answers_nothing();
    an_unknown_impl_is_refused_by_name();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
