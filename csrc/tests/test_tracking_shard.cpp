// The per-camera tracker shard and its ordering guard.
//
// The guard is the reason this file exists. Tracking is stateful, so the failures it prevents
// are all invisible: a replayed frame double-ages every track and double-counts the hit that
// promotes one, and two cameras on one tracker report a real identity where nothing happened.
// None of that shows up as an error, a dropped frame or a wrong count -- it shows up as an
// identity, which is exactly the thing nothing downstream can second-guess.
//
// Offline: g++ alone, no CUDA. The tracker is pure C++ (four .cpp files out of shipvision and
// one include path), which is what lets the correctness-critical half of `track` be tested on
// a machine with no driver -- ADR-001's promise, applied to tracking.

#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/tracking/shard.h"

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

    //: One box, positioned so a sequence of them reads as one object moving right. `min_hits`
    //: is lowered in the options below so a short test sees confirmed ids at all.
    Detection a_box(float x, int class_id = 0, float score = 0.9f) {
        Detection d;
        d.x1 = x;
        d.y1 = 10.f;
        d.x2 = x + 50.f;
        d.y2 = 90.f;
        d.score = score;
        d.class_id = class_id;
        return d;
    }

    shipvision::mot::ByteTrackTracker::Options fast_confirm() {
        shipvision::mot::ByteTrackTracker::Options options;
        options.min_hits = 1;  // confirm on the first frame, so ten frames is enough to test
        return options;
    }

    // -- the thing it is for -------------------------------------------------------------

    void test_one_object_keeps_one_id_across_frames() {
        TrackerShard shard(fast_confirm());
        int first = -1;
        for (int64_t frame = 0; frame < 8; ++frame) {
            const TrackUpdate out =
                shard.update("cam0", frame, {a_box(10.f + 5.f * static_cast<float>(frame))});
            check(out.ids.size() == 1, "one id per detection");
            if (frame == 0) {
                first = out.ids[0];
                check(first > 0,
                      "and a confirmed track has a positive id: " + std::to_string(first));
            } else {
                check(out.ids[0] == first,
                      "the same object keeps the same id across frames: got " +
                          std::to_string(out.ids[0]) + " after " + std::to_string(first));
            }
        }
    }

    void test_two_cameras_do_not_share_a_tracker() {
        // THE CORRECTNESS CONSTRAINT, and the one with no symptom of its own.
        //
        // Two cameras interleaved on the SAME frame ids, which is the production pattern --
        // every camera starts at frame 0. A shared tracker fails this two ways at once and
        // the test reports whichever comes first rather than aborting on it: the second
        // camera's frame 0 arrives "after" the first's on a shared high-water mark, and their
        // boxes (identical coordinates) associate with each other by IoU.
        //
        // CAUGHT rather than left to propagate: an uncaught throw here would abort the binary
        // with no summary line, which is a crash and not a red test.
        TrackerShard shard(fast_confirm());
        std::string refused;
        try {
            for (int64_t frame = 0; frame < 4; ++frame) {
                const float x = 10.f + 5.f * static_cast<float>(frame);
                shard.update("cam0", frame, {a_box(x)});
                shard.update("cam1", frame, {a_box(x)});
            }
        } catch (const InferenceError& error) {
            refused = error.what();
        }
        check(refused.empty(),
              "two cameras interleaving the same frame ids is the ORDINARY case and must not "
              "trip the ordering guard -- one shared high-water mark does exactly that: " +
                  refused);
        check(shard.stats().cameras == 2,
              "one tracker per camera, built lazily: " + std::to_string(shard.stats().cameras));
        check(shard.stats().out_of_order == 0, "and nothing is counted as a reordering: " +
                                                   std::to_string(shard.stats().out_of_order));
    }

    void test_each_cameras_ids_start_from_its_own_pool() {
        // The other half of the isolation, on ids rather than on ordering: separate pools each
        // number from 1, so two cameras' first objects carry the SAME id. A shared pool would
        // hand the second camera a different number -- or, worse, the first camera's id, which
        // is a real identity reported where nothing happened.
        //
        // Same reason as above for the catch: a shared high-water mark makes the second
        // camera's frame 0 a "replay", and an escaping throw is a crash rather than a red.
        TrackerShard shard(fast_confirm());
        try {
            const TrackUpdate a = shard.update("cam0", 0, {a_box(10.f)});
            const TrackUpdate b = shard.update("cam1", 0, {a_box(400.f)});  // far: no IoU
            check(a.ids.size() == 1 && b.ids.size() == 1, "one id each");
            check(a.ids[0] == 1 && b.ids[0] == 1,
                  "each camera's first track is id 1 in its own pool: got " +
                      std::to_string(a.ids[0]) + " and " + std::to_string(b.ids[0]));
        } catch (const InferenceError& error) {
            check(false, std::string("two cameras' first frames must both be accepted: ") +
                             error.what());
        }
    }

    // -- the ordering guard --------------------------------------------------------------

    void test_a_replayed_frame_is_refused_rather_than_tracked() {
        TrackerShard shard(fast_confirm());
        shard.update("cam0", 10, {a_box(10.f)});
        bool refused = false;
        std::string message;
        try {
            shard.update("cam0", 10, {a_box(15.f)});  // the same frame id again
        } catch (const InferenceError& error) {
            refused = true;
            message = error.what();
        }
        check(refused, "a frame that does not advance the stream is REFUSED");
        check(message.find("cam0") != std::string::npos &&
                  message.find("frame 10") != std::string::npos,
              "and the refusal names the camera and the frame: " + message);
        check(shard.stats().out_of_order == 1,
              "counted as a reordering: " + std::to_string(shard.stats().out_of_order));
        check(shard.stats().implicit_resets == 0, "and not as a reconnect");
    }

    void test_a_small_regression_is_a_reordering_and_a_large_one_is_a_restart() {
        // The two sides of `kRegressionReset`, in one test because the boundary is the point.
        TrackerShard shard(fast_confirm(), /*regression_reset=*/64);
        shard.update("cam0", 1000, {a_box(10.f)});

        bool refused = false;
        try {
            shard.update("cam0", 1000 - 63, {a_box(10.f)});  // 63 behind: a reordering
        } catch (const InferenceError&) {
            refused = true;
        }
        check(refused, "63 frames behind is a reordering and is refused");

        std::string reset_camera;
        const TrackUpdate out =
            shard.update("cam0", 1000 - 64, {a_box(10.f)},
                         [&](const std::string& camera) { reset_camera = camera; });
        check(out.ids.size() == 1, "64 frames behind is a RESTARTED STREAM and is accepted");
        check(reset_camera == "cam0", "and the callback names the camera: " + reset_camera);
        const TrackerShardStats stats = shard.stats();
        check(stats.implicit_resets == 1 && stats.out_of_order == 1,
              "one of each, kept apart: resets=" + std::to_string(stats.implicit_resets) +
                  " out_of_order=" + std::to_string(stats.out_of_order));
    }

    void test_a_restart_lets_a_lower_frame_id_continue() {
        // What the reset is FOR: after it, the stream carries on from the new low id rather
        // than refusing every subsequent frame because they are all below the old high water.
        TrackerShard shard(fast_confirm(), /*regression_reset=*/8);
        shard.update("cam0", 500, {a_box(10.f)});
        shard.update("cam0", 0, {a_box(10.f)});  // a reconnect: 500 behind
        bool refused = false;
        try {
            shard.update("cam0", 1, {a_box(15.f)});
            shard.update("cam0", 2, {a_box(20.f)});
        } catch (const InferenceError&) {
            refused = true;
        }
        check(!refused,
              "frames 1 and 2 continue the restarted stream rather than being "
              "measured against the abandoned 500");
    }

    void test_a_reset_that_is_disabled_refuses_every_regression() {
        TrackerShard shard(fast_confirm(), /*regression_reset=*/0);
        shard.update("cam0", 1000, {a_box(10.f)});
        bool refused = false;
        try {
            shard.update("cam0", 0, {a_box(10.f)});  // 1000 behind, and no reset threshold
        } catch (const InferenceError&) {
            refused = true;
        }
        check(refused, "with the threshold off, even a huge regression is a reordering");
        check(shard.stats().implicit_resets == 0, "and nothing was reset");
    }

    // -- lifecycle ------------------------------------------------------------------------

    void test_reset_if_present_does_not_mint_a_tracker() {
        // `camera_added` fires for every camera on the shard, including the forty-nine that
        // were never on this element. Building a Kalman filter and a track pool for each --
        // on the thread holding the runner's lifecycle lock -- is work for nothing.
        TrackerShard shard(fast_confirm());
        check(!shard.reset_if_present("never-seen"),
              "a camera with no tracker reports nothing to reset");
        check(shard.stats().cameras == 0,
              "and no tracker was built: " + std::to_string(shard.stats().cameras));

        shard.update("cam0", 0, {a_box(10.f)});
        check(shard.reset_if_present("cam0"), "a camera with one reports that it was reset");
        check(shard.stats().cameras == 1, "and the tracker is kept, not dropped");
    }

    void test_a_reset_lets_the_stream_start_again_from_zero() {
        TrackerShard shard(fast_confirm());
        for (int64_t frame = 0; frame < 4; ++frame) {
            shard.update("cam0", frame, {a_box(10.f + 5.f * static_cast<float>(frame))});
        }
        shard.reset_if_present("cam0");
        bool refused = false;
        try {
            shard.update("cam0", 0, {a_box(10.f)});  // the reconnect's first frame
        } catch (const InferenceError&) {
            refused = true;
        }
        check(!refused,
              "after an explicit reset, frame 0 is the new stream's first and is "
              "not measured against the old high water");
    }

    // -- threading ------------------------------------------------------------------------

    void test_fifty_cameras_in_parallel_each_keep_their_own_ids() {
        // The map lock is held only for insertion, so this is where a bad lock discipline
        // shows: fifty threads inserting concurrently while forty-nine others run `update`.
        TrackerShard shard(fast_confirm());
        std::vector<std::thread> threads;
        std::vector<int> first_ids(50, -2);
        std::vector<std::string> thread_errors(50);
        for (int camera = 0; camera < 50; ++camera) {
            threads.emplace_back([&shard, &first_ids, &thread_errors, camera] {
                const std::string name = "cam" + std::to_string(camera);
                // CAUGHT INSIDE THE THREAD: an escaping exception calls `std::terminate`, so
                // the shared-tracker bug this test exists to catch would abort the binary
                // with no summary rather than fail a check.
                try {
                    for (int64_t frame = 0; frame < 6; ++frame) {
                        const TrackUpdate out = shard.update(
                            name, frame, {a_box(10.f + 5.f * static_cast<float>(frame))});
                        if (frame == 0) first_ids[static_cast<size_t>(camera)] = out.ids[0];
                    }
                } catch (const std::exception& error) {
                    thread_errors[static_cast<size_t>(camera)] = error.what();
                }
            });
        }
        for (std::thread& thread : threads) thread.join();
        std::string first_error;
        for (const std::string& error : thread_errors) {
            if (!error.empty() && first_error.empty()) first_error = error;
        }
        check(first_error.empty(),
              "fifty cameras on their own monotonic streams raise nothing: " + first_error);
        check(shard.stats().cameras == 50, "fifty concurrent cameras give fifty trackers: " +
                                               std::to_string(shard.stats().cameras));
        bool all_first = true;
        for (const int id : first_ids) all_first = all_first && id == 1;
        check(all_first,
              "and each camera's first track is id 1 in its OWN pool -- a shared "
              "pool would have handed out fifty different numbers");
        check(shard.stats().out_of_order == 0 && shard.stats().implicit_resets == 0,
              "with no spurious reordering under concurrency");
    }

}  // namespace

int main() {
    test_one_object_keeps_one_id_across_frames();
    test_two_cameras_do_not_share_a_tracker();
    test_each_cameras_ids_start_from_its_own_pool();
    test_a_replayed_frame_is_refused_rather_than_tracked();
    test_a_small_regression_is_a_reordering_and_a_large_one_is_a_restart();
    test_a_restart_lets_a_lower_frame_id_continue();
    test_a_reset_that_is_disabled_refuses_every_regression();
    test_reset_if_present_does_not_mint_a_tracker();
    test_a_reset_lets_the_stream_start_again_from_zero();
    test_fifty_cameras_in_parallel_each_keep_their_own_ids();

    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
