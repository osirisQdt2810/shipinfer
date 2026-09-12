// The two gates before the expensive part: too small, and too new to trust yet.
//
// Both are about one failure -- a bad crop produces a CONFIDENT embedding -- so what these
// tests pin is not arithmetic but the order (height first, then age) and the boundedness of the
// one piece of state. Pure, offline, no submodule.
#include <cstdio>
#include <string>
#include <vector>

#include "shipinfer/pipeline/mtmc/gate.h"

namespace {

    using namespace shipinfer;
    using shipinfer::mtmc::ClusterObservation;
    using shipinfer::mtmc::ObservationGate;
    using shipinfer::mtmc::TrackKey;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    // `height` is the box's height in pixels of a 1080-tall frame, so the fraction is readable:
    // 200 is well over a ninth, 50 is well under.
    ClusterObservation seen(const std::string& camera, int64_t track, float height) {
        ClusterObservation out;
        out.key = TrackKey{camera, track};
        out.embedding = {1.0f, 0.0f};
        out.box[0] = 0.0f;
        out.box[1] = 0.0f;
        out.box[2] = 40.0f;
        out.box[3] = height;
        out.frame_width = 1920;
        out.frame_height = 1080;
        return out;
    }

    // The gate with no roster but the observations' own. Most of these tests are about height
    // and age, where the two are the same thing; the ones that are about the roster call
    // `filter` directly and pass it.
    std::vector<ClusterObservation> admit(ObservationGate& gate,
                                          const std::vector<ClusterObservation>& instant) {
        return gate.filter(instant, mtmc::cameras_of(instant));
    }

    ObservationGate::Options options(int min_hits = 3, double min_height = 1.0 / 9.0) {
        ObservationGate::Options built;
        built.min_hits = min_hits;
        built.min_height_fraction = min_height;
        return built;
    }

    void the_options_are_refused_when_they_would_admit_anything() {
        bool hits = false;
        bool height = false;
        try {
            ObservationGate bad(options(0));
        } catch (const ConfigError& error) {
            hits = std::string(error.what()).find("min_hits must be at least 1") !=
                   std::string::npos;
        }
        try {
            ObservationGate bad(options(3, 1.0));
        } catch (const ConfigError& error) {
            height = std::string(error.what()).find("must be in [0, 1)") != std::string::npos;
        }

        check(hits, "min_hits 0 would admit a track on the frame it was first seen");
        check(height, "and a fraction of 1.0 admits nothing at all, which is not a threshold");
    }

    void a_track_must_be_seen_min_hits_times_before_it_is_admitted() {
        ObservationGate gate(options(3));

        check(admit(gate, {seen("cam0", 1, 200)}).empty(), "not on the first frame");
        check(admit(gate, {seen("cam0", 1, 200)}).empty(), "not on the second");
        check(admit(gate, {seen("cam0", 1, 200)}).size() == 1, "admitted on the third");
        check(gate.hits(TrackKey{"cam0", 1}) == 3, "and its run is three");
    }

    void a_track_that_is_too_small_never_accrues_age() {
        // The order that is load-bearing: age counts consecutive frames in which the track was
        // ALSO large enough. Otherwise a figure walking in from the distance banks three
        // frames while unusably small and enters the matrix on its first usable frame with the
        // gate already satisfied.
        ObservationGate gate(options(3));
        for (int frame = 0; frame < 5; ++frame) {
            check(admit(gate, {seen("cam0", 1, 50)}).empty(),
                  "a small track is never admitted");
        }
        check(gate.hits(TrackKey{"cam0", 1}) == 0, "and banks no age at all");

        // Now it comes close: it still needs its three qualifying frames.
        check(admit(gate, {seen("cam0", 1, 200)}).empty(), "the first usable frame is hit 1");
        check(admit(gate, {seen("cam0", 1, 200)}).empty(), "then 2");
        check(admit(gate, {seen("cam0", 1, 200)}).size() == 1, "and only then admitted");
    }

    void an_instant_its_camera_was_not_in_is_not_a_miss() {
        // At fifty cameras an instant holds about a quarter of the fleet, so absence as a miss
        // makes three consecutive sightings a 1.4% event -- measured, and what collapsed
        // admission to 2.2%. "Consecutive" is about the track, not the caller's clock.
        ObservationGate gate(options(3));
        admit(gate, {seen("cam0", 1, 200)});
        admit(gate, {seen("cam0", 1, 200)});
        check(gate.hits(TrackKey{"cam0", 1}) == 2, "two sightings");

        admit(gate, {seen("cam1", 9, 200)});  // an instant cam0 was not in

        check(gate.hits(TrackKey{"cam0", 1}) == 2, "the run is carried, not broken");
        check(gate.absent_size() == 1,
              "and it is counted as an absence, so the bound can end it");
        check(admit(gate, {seen("cam0", 1, 200)}).size() == 1, "the third SIGHTING admits it");
    }

    void a_camera_that_reported_an_empty_view_breaks_the_run() {
        // The other half, and the one the observations alone cannot express: cam0 WAS in this
        // instant and its track was not. Only the roster tells this from the case above, which
        // is why `filter` takes one rather than deriving it.
        ObservationGate gate(options(3));
        admit(gate, {seen("cam0", 1, 200)});
        admit(gate, {seen("cam0", 1, 200)});

        gate.filter({}, {"cam0"});  // reported, saw nothing

        check(gate.hits(TrackKey{"cam0", 1}) == 0,
              "the camera was there and the track was not");
        check(admit(gate, {seen("cam0", 1, 200)}).empty(), "so the next sighting is hit 1");
    }

    void a_camera_that_goes_quiet_for_good_does_not_leak_its_streaks() {
        // The bound, and it is a bound rather than a knob: an absent camera says nothing about
        // its tracks, so without a limit one that goes away forever leaves them in the map for
        // the life of the process.
        ObservationGate::Options bounded = options(3);
        bounded.max_absent_instants = 2;
        ObservationGate gate(bounded);
        admit(gate, {seen("cam0", 1, 200)});
        admit(gate, {seen("cam0", 1, 200)});

        admit(gate, {seen("cam1", 9, 200)});
        check(gate.hits(TrackKey{"cam0", 1}) == 2, "carried through the first absence");
        admit(gate, {seen("cam1", 9, 200)});
        check(gate.hits(TrackKey{"cam0", 1}) == 2, "and the second, which is the bound");
        admit(gate, {seen("cam1", 9, 200)});

        check(gate.hits(TrackKey{"cam0", 1}) == 0, "past it the streak is dropped");
        check(gate.size() == 1, "and the map holds only the tracks in flight");
    }

    void the_absence_bound_is_refused_when_it_would_break_a_run_at_once() {
        bool refused = false;
        ObservationGate::Options bad = options(3);
        bad.max_absent_instants = 0;
        try {
            ObservationGate gate(bad);
        } catch (const ConfigError& error) {
            refused =
                std::string(error.what()).find("max_absent_instants must be at least 1") !=
                std::string::npos;
        }

        check(refused, "0 is the behaviour this replaced, not a tighter version of it");
    }

    void the_height_test_is_strictly_greater_than_the_threshold() {
        // `> min_height_fraction`, the reference's comparison. A box exactly at the threshold
        // is not admitted, which matters because 1/9 of 1080 is exactly 120.
        ObservationGate gate(options(1, 120.0 / 1080.0));

        check(admit(gate, {seen("cam0", 1, 120)}).empty(), "exactly at the threshold: out");
        check(admit(gate, {seen("cam0", 2, 121)}).size() == 1, "a pixel over: in");
    }

    void a_frame_with_no_extent_admits_nothing_rather_than_dividing_by_zero() {
        ObservationGate gate(options(1));
        ClusterObservation broken = seen("cam0", 1, 200);
        broken.frame_height = 0;

        check(admit(gate, {broken}).empty(),
              "a frame whose extent nobody filled in is a wiring fault, and admitting every "
              "track through it is the failure this gate exists for");
        check(gate.hits(TrackKey{"cam0", 1}) == 0, "and it accrues no age either");
    }

    void the_admitted_keep_their_input_order() {
        // The caller zips the gate's answer against the clusterer's labels, so order is part
        // of the contract rather than an accident of the container.
        ObservationGate gate(options(1));

        const auto admitted =
            admit(gate, {seen("cam2", 1, 200), seen("cam0", 1, 200), seen("cam1", 1, 200)});

        check(admitted.size() == 3, "all three qualify");
        check(admitted[0].key.camera_id == "cam2" && admitted[2].key.camera_id == "cam1",
              "and they come back in INPUT order, not sorted");
    }

    void one_track_twice_in_one_instant_is_refused() {
        // #220's review named the shape: both copies read the same previous count so the run
        // advances once (correct), and then BOTH are admitted -- and the caller zips
        // `admitted` against the clusterer's labels, so one track takes two rows of the matrix
        // and contests itself. No golden can hold it: the emitter's scenario format is
        // line-oriented over distinct keys and the reference's dicts would collapse the pair.
        ObservationGate gate(options(1, 0.0));
        std::string message;

        try {
            admit(gate, {seen("cam0", 1, 200.0), seen("cam0", 1, 200.0)});
        } catch (const InferenceError& error) {
            message = error.what();
        }

        check(message.find("cam0#1") != std::string::npos, "the duplicate is named");
        check(message.find("twice in one instant") != std::string::npos,
              "and the reason is the instant's shape, not the track's size");
    }

    // doc: long why a retried instant counting twice is pinned rather than fixed here
    void a_re_submitted_instant_advances_the_run_again_as_the_reference_does() {
        // `filter` commits its hit map before the caller has applied the instant, so an
        // instant submitted twice advances every run twice and `min_hits` is satisfied one
        // real instant early. That is worth a test because it is EASY to read as a port
        // defect and it is not: the reference does the same, measured 11 Sep --
        //
        //   instant 1            hits=1 admitted=0
        //   instant 2            hits=2 admitted=0
        //   RETRY of instant 2   hits=3 admitted=1
        //
        // -- from `shipvision/mtmc/gating.py`, whose `filter` assigns `self._hits = hits`
        // before `tracker.track()` runs the gram, the clusterer and the assigner, any of
        // which can raise. Making this half atomic alone would diverge the two planes, so the
        // fix is upstream and this pins the shared behaviour until then
        // (`MTMC-GATE-COMMITS-BEFORE-THE-GRAM-CAN-THROW`). Nothing retries today: the stage
        // catches the refusal and publishes the frame unidentified.
        ObservationGate gate(options(3));
        const std::vector<ClusterObservation> instant = {seen("cam0", 1, 200)};

        admit(gate, instant);
        admit(gate, instant);

        check(admit(gate, instant).size() == 1,
              "the third submission admits, whether or not the second was applied");
        check(gate.hits(TrackKey{"cam0", 1}) == 3, "and the run counts every submission");
    }

    void reset_forgets_every_run() {
        ObservationGate gate(options(3));
        admit(gate, {seen("cam0", 1, 200)});
        admit(gate, {seen("cam0", 1, 200)});

        gate.reset();

        check(gate.size() == 0, "nothing held");
        check(admit(gate, {seen("cam0", 1, 200)}).empty(), "and the run starts again from one");
    }

}  // namespace

int main() {
    the_options_are_refused_when_they_would_admit_anything();
    a_track_must_be_seen_min_hits_times_before_it_is_admitted();
    a_track_that_is_too_small_never_accrues_age();
    an_instant_its_camera_was_not_in_is_not_a_miss();
    a_camera_that_reported_an_empty_view_breaks_the_run();
    a_camera_that_goes_quiet_for_good_does_not_leak_its_streaks();
    the_absence_bound_is_refused_when_it_would_break_a_run_at_once();
    the_height_test_is_strictly_greater_than_the_threshold();
    a_frame_with_no_extent_admits_nothing_rather_than_dividing_by_zero();
    the_admitted_keep_their_input_order();
    one_track_twice_in_one_instant_is_refused();
    a_re_submitted_instant_advances_the_run_again_as_the_reference_does();
    reset_forgets_every_run();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
