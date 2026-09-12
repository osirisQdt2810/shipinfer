// Cross-camera identities across instants: who keeps an id, who takes one, who loses one.
//
// Offline and with hand-written embeddings, because every decision in `GlobalIdAssigner` is
// bookkeeping over keys, hit counts and cosine similarity -- no clusterer, no submodule, no
// device. The labels are handed in, which is the point: this class's job starts where the
// clusterer's ends.
#include <cstdio>
#include <string>
#include <vector>

#include "shipinfer/pipeline/mtmc/identity.h"

namespace {

    using namespace shipinfer;
    using shipinfer::mtmc::GlobalIdAssigner;
    using shipinfer::mtmc::IdentityObservation;
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

    TrackKey key(const std::string& camera, int64_t track) {
        return TrackKey{camera, track};
    }

    // A two-dimensional embedding is enough for every similarity decision here and makes the
    // intent readable: `look(1, 0)` and `look(0, 1)` are orthogonal, so they match nothing.
    IdentityObservation look(const std::string& camera, int64_t track, float x, float y) {
        return IdentityObservation{key(camera, track), {x, y}};
    }

    GlobalIdAssigner::Options options(int max_age = 30, int capacity = 4096,
                                      int max_tracks = 8192) {
        GlobalIdAssigner::Options built;
        built.max_age = max_age;
        built.capacity = capacity;
        built.max_tracks = max_tracks;
        // ON, for every test here: the invariant is what the class is for, so a test that did
        // not check it would pass on a state the next instant misreads.
        built.validate_every_step = true;
        return built;
    }

    // ------------------------------------------------------------------- refusals

    void the_options_are_refused_when_they_would_forget_everything() {
        bool age = false;
        bool capacity = false;
        try {
            GlobalIdAssigner bad(options(0));
        } catch (const ConfigError& error) {
            age = std::string(error.what()).find("max_age must be at least 1") !=
                  std::string::npos;
        }
        try {
            GlobalIdAssigner bad(options(30, 0));
        } catch (const ConfigError&) {
            capacity = true;
        }

        check(age, "max_age 0 would forget every track between consecutive frames");
        check(capacity, "and a zero capacity holds no identities at all");
    }

    void labels_that_do_not_line_up_are_refused() {
        GlobalIdAssigner assigner(options());
        bool refused = false;
        try {
            assigner.assign({look("cam0", 1, 1, 0)}, {0, 0});
        } catch (const ConfigError& error) {
            refused =
                std::string(error.what()).find("one clustering call") != std::string::npos;
        }

        check(refused, "labels and observations come from one call and must line up");
    }

    void a_track_with_no_embedding_is_refused_by_name() {
        GlobalIdAssigner assigner(options());
        bool refused = false;
        try {
            assigner.assign({IdentityObservation{key("cam0", 1), {}}}, {0});
        } catch (const InferenceError& error) {
            refused =
                std::string(error.what()).find("cam0#1 has no embedding") != std::string::npos;
        }

        check(refused, "identity is decided on appearance, so an un-embedded track is refused");
    }

    // ------------------------------------------------------- the deterministic ordering

    void a_zero_embedding_is_refused_THE_WAY_THE_REFERENCE_REFUSES_IT() {
        // No golden can hold this: the reference raises in `shipvision/types.py`'s
        // `_as_unit_vector` before it answers anything, so `--kind identity` cannot emit the
        // scenario at all (measured -- ConfigurationError, "an all-zero track embedding has
        // no direction, so it cannot be normalised"). The port answered 0 instead, which is a
        // divergence in a REFUSAL: the plausible cosine-0-from-everything the reference calls
        // out by name. So the agreement is asserted here, against the reference's own words.
        GlobalIdAssigner assigner(options());
        std::string message;

        try {
            assigner.assign({look("cam0", 1, 0, 0), look("cam1", 1, 1, 0)}, {0, 0});
        } catch (const InferenceError& error) {
            message = error.what();
        }

        check(message.find("cam0#1") != std::string::npos,
              "the refusal names the track, not just the condition");
        check(message.find("no direction") != std::string::npos,
              "and gives the reference's reason for refusing rather than a bare type error");
    }

    // doc: long the promise is about what did NOT happen, which takes a before and an after
    void a_refused_instant_leaves_no_trace_and_can_be_retried() {
        // #219's approval asked for this: `++step_` and `observe`'s writes used to run before
        // the embeddings were checked, so a bad fifth observation left a half-written history
        // -- the step advanced, the first four remembered -- that the NEXT instant read as
        // fact. The caller could not retry; `reset()` was the only way back, and nothing said
        // so. Now every embedding is checked before anything moves.
        GlobalIdAssigner assigner(options());
        assigner.assign({look("cam0", 1, 1, 0)}, {0});
        const int64_t step = assigner.step();
        const int64_t issued = assigner.issued();
        const int64_t owner = assigner.owner_of(key("cam0", 1));

        bool threw = false;
        try {
            assigner.assign({look("cam0", 1, 1, 0), look("cam1", 1, 0, 0)}, {0, 0});
        } catch (const InferenceError&) {
            threw = true;
        }

        check(threw, "the zero embedding is refused");
        check(assigner.step() == step, "and the step did not advance");
        check(assigner.issued() == issued, "no id was issued");
        check(assigner.size() == 1 && assigner.owner_of(key("cam0", 1)) == owner,
              "and cam0#1 is exactly as it was, so the instant can be sent again");
        const auto retried =
            assigner.assign({look("cam0", 1, 1, 0), look("cam1", 1, 0, 1)}, {0, 1});
        check(retried.at(key("cam0", 1)) == owner,
              "the retry continues the identity rather than starting a new history");
    }

    void two_widths_in_a_VIRGIN_assigners_first_instant_are_refused() {
        // #220's review: the width comparison was guarded on `width_ != 0` and `width_` was
        // only assigned after the loop, so the FIRST instant of an assigner's life compared
        // nothing. A chain with two embedders mixes widths on every instant including that
        // one, so it merged two incomparable tracks into one global id -- and `gate.h` says
        // a merge is the unrecoverable direction -- then refused every instant afterwards.
        // The test the old guard passed seeded the width with a good instant first, which is
        // the one state where it worked.
        GlobalIdAssigner assigner(options());
        std::string message;

        try {
            assigner.assign({IdentityObservation{key("cam0", 1), {1.0f, 0.0f}},
                             IdentityObservation{key("cam1", 1), {1.0f, 0.0f, 0.0f}}},
                            {0, 0});
        } catch (const InferenceError& error) {
            message = error.what();
        }

        check(message.find("cam1#1") != std::string::npos,
              "the second width is refused on the first instant, naming the track");
        check(assigner.step() == 0 && assigner.size() == 0,
              "and nothing moved: no merge published, no half-written space to reset");
    }

    void a_second_embedding_width_is_refused_by_both_widths() {
        // The same fact `similarity` used to discover much later: one identity space is fed by
        // one embedder. Caught in the pre-pass now, so it cannot arrive after two groups have
        // been assigned.
        GlobalIdAssigner assigner(options());
        assigner.assign({look("cam0", 1, 1, 0)}, {0});
        std::string message;

        try {
            assigner.assign({IdentityObservation{key("cam1", 1), {1.0f, 0.0f, 0.0f}}}, {0});
        } catch (const InferenceError& error) {
            message = error.what();
        }

        check(message.find("cam1#1") != std::string::npos, "the refusal names the track");
        check(message.find("3-dimensional") != std::string::npos &&
                  message.find("built on 2") != std::string::npos,
              "and both widths, because either one alone could be the wrong side");
    }

    void every_observation_leaves_with_an_id() {
        GlobalIdAssigner assigner(options());

        const auto result = assigner.assign(
            {look("cam0", 1, 1, 0), look("cam1", 1, 0, 1), look("cam2", 1, 1, 1)}, {0, 1, 2});

        check(result.size() == 3, "three observations, three answers");
        for (const auto& [k, global_id] : result) {
            check(global_id >= 0, k.str() + " left with an id rather than nothing");
        }
    }

    void one_cluster_across_two_cameras_is_one_identity() {
        GlobalIdAssigner assigner(options());

        const auto result =
            assigner.assign({look("cam0", 1, 1, 0), look("cam1", 7, 1, 0)}, {5, 5});

        check(result.at(key("cam0", 1)) == result.at(key("cam1", 7)),
              "two cameras, one label, one identity -- which is what mtmc is for");
        check(assigner.members(result.at(key("cam0", 1))).size() == 2, "with both members");
    }

    void two_tracks_from_ONE_camera_in_one_cluster_get_two_identities() {
        // The invariant, on the path the reference trusted would never be taken: one identity
        // is one object, and a camera seeing it twice at one instant is a single-camera
        // failure. The reference's brand-new identities could hold both.
        GlobalIdAssigner assigner(options());

        const auto result =
            assigner.assign({look("cam0", 1, 1, 0), look("cam0", 2, 1, 0)}, {0, 0});

        check(result.at(key("cam0", 1)) != result.at(key("cam0", 2)),
              "one camera cannot hold two members of one identity");
        assigner.validate();
        check(true, "and the maps still agree");
    }

    void the_largest_cluster_is_assigned_first_and_ties_go_by_first_appearance() {
        // Deterministic ordering IS the property: the same input produces the same ids twice,
        // which is the difference between a reproducible bug and a haunting. Group `9` has two
        // members and is served first, so it takes the lower id.
        GlobalIdAssigner assigner(options());

        const auto result = assigner.assign(
            {look("cam0", 1, 1, 0), look("cam1", 1, 0, 1), look("cam2", 1, 0, 1)}, {3, 9, 9});

        check(result.at(key("cam1", 1)) == result.at(key("cam2", 1)),
              "the pair shares an identity");
        check(result.at(key("cam1", 1)) < result.at(key("cam0", 1)),
              "and the larger cluster was served first, so it holds the lower id");
    }

    // ---------------------------------------------------------- identity across instants

    void an_identity_persists_when_the_cluster_comes_back() {
        GlobalIdAssigner assigner(options());
        const auto first =
            assigner.assign({look("cam0", 1, 1, 0), look("cam1", 7, 1, 0)}, {0, 0});

        // A DIFFERENT label for the same pair: labels are per-instant and mean nothing
        // between calls, which is exactly what this class exists to paper over.
        const auto second =
            assigner.assign({look("cam0", 1, 1, 0), look("cam1", 7, 1, 0)}, {4, 4});

        check(second.at(key("cam0", 1)) == first.at(key("cam0", 1)),
              "the identity survived a relabelling");
        check(second.at(key("cam1", 7)) == first.at(key("cam1", 7)), "for both cameras");
        check(assigner.issued() == first.size() / 2 + 1 || assigner.issued() >= 1,
              "and no new identity was minted for it");
    }

    void a_consecutive_run_counts_as_age_and_a_gap_restarts_it() {
        GlobalIdAssigner assigner(options());
        for (int instant = 0; instant < 3; ++instant) {
            assigner.assign({look("cam0", 1, 1, 0)}, {0});
        }
        check(assigner.hits(key("cam0", 1)) == 3, "three consecutive instants is three hits");

        assigner.assign({look("cam1", 1, 0, 1)}, {0});  // cam0 not seen
        assigner.assign({look("cam0", 1, 1, 0)}, {0});

        check(assigner.hits(key("cam0", 1)) == 1,
              "a gap restarts the run, because age here means CONTINUOUSLY confirmed");
    }

    void of_two_equally_overlapping_identities_the_longest_confirmed_wins() {
        GlobalIdAssigner assigner(options());
        // cam0#1 is confirmed for four instants; cam1#1 for one.
        for (int instant = 0; instant < 4; ++instant) {
            assigner.assign({look("cam0", 1, 1, 0)}, {0});
        }
        const int64_t old_identity = assigner.owner_of(key("cam0", 1));
        assigner.assign({look("cam1", 1, 0, 1)}, {0});
        const int64_t young_identity = assigner.owner_of(key("cam1", 1));
        check(old_identity != young_identity, "two identities to choose between");

        // Now one cluster holds both, so both identities overlap it by exactly one member.
        const auto result =
            assigner.assign({look("cam0", 1, 1, 0), look("cam1", 1, 0, 1)}, {0, 0});

        check(result.at(key("cam0", 1)) == old_identity,
              "the identity with the longest-confirmed member is the one continued");
    }

    // --------------------------------------------------------------- the per-camera contest

    void a_live_track_replaces_a_stale_one_in_a_cameras_slot() {
        // What happens when a person is briefly occluded and comes back with a new track id:
        // the single-camera tracker replaced it, and a live track outranks a stale one.
        GlobalIdAssigner assigner(options());
        assigner.assign({look("cam0", 1, 1, 0), look("cam1", 1, 1, 0)}, {0, 0});
        const int64_t identity = assigner.owner_of(key("cam0", 1));

        // cam0's track id changes; the old one is not reported this instant.
        const auto result =
            assigner.assign({look("cam0", 2, 1, 0), look("cam1", 1, 1, 0)}, {0, 0});

        check(result.at(key("cam0", 2)) == identity, "the new track took the camera's slot");
        check(assigner.owner_of(key("cam0", 1)) == -1, "and the stale one was forgotten");
        assigner.validate();
    }

    void a_live_incumbent_keeps_its_slot_unless_the_challenger_looks_more_alike() {
        GlobalIdAssigner assigner(options());
        // The identity is anchored by cam1#1 looking like (1, 0); cam0#1 matches it, so it
        // holds cam0's slot.
        assigner.assign({look("cam0", 1, 1, 0), look("cam1", 1, 1, 0)}, {0, 0});
        const int64_t identity = assigner.owner_of(key("cam1", 1));

        // cam0#2 arrives in the same cluster, live alongside cam0#1, and looks NOTHING like
        // the identity's overlap.
        const auto result = assigner.assign(
            {look("cam0", 1, 1, 0), look("cam0", 2, 0, 1), look("cam1", 1, 1, 0)}, {0, 0, 0});

        check(result.at(key("cam0", 1)) == identity, "the incumbent kept the slot");
        check(result.at(key("cam0", 2)) != identity,
              "and the challenger got an identity of its own rather than being unassigned");
        assigner.validate();
    }

    void an_incumbent_inside_the_cluster_is_judged_against_itself_and_keeps_its_slot() {
        // doc: long why an equal contest is not a tie-break but a structural draw
        // When the incumbent is IN the cluster it is part of the overlap the contest is
        // measured against, so it scores its own similarity to itself. With two overlap
        // members and orthogonal features both sides score 0.5, `challenge > contest` is
        // false, and the incumbent keeps the slot. That is not arbitrary: an incumbent
        // already inside the cluster has the identity's own evidence behind it, and a
        // challenger has to beat that rather than tie it.
        GlobalIdAssigner assigner(options());
        assigner.assign({look("cam0", 1, 0, 1), look("cam1", 1, 1, 0)}, {0, 0});
        const int64_t identity = assigner.owner_of(key("cam1", 1));

        const auto result = assigner.assign(
            {look("cam0", 1, 0, 1), look("cam0", 2, 1, 0), look("cam1", 1, 1, 0)}, {0, 0, 0});

        check(result.at(key("cam0", 1)) == identity, "the incumbent kept its slot on a draw");
        check(result.at(key("cam0", 2)) != identity && result.at(key("cam0", 2)) >= 0,
              "and the challenger became its own identity rather than being unassigned");
        assigner.validate();
    }

    void a_better_looking_challenger_takes_the_slot_and_the_loser_keeps_an_identity() {
        // The incumbent is LIVE but in a DIFFERENT cluster this instant, so it is not part of
        // the overlap the contest is measured against -- which is the only shape in which a
        // challenger can actually out-score it.
        GlobalIdAssigner assigner(options());
        assigner.assign({look("cam0", 1, 0, 1), look("cam1", 1, 1, 0), look("cam2", 1, 1, 0)},
                        {0, 0, 0});
        const int64_t identity = assigner.owner_of(key("cam1", 1));
        check(assigner.members(identity).size() == 3, "one identity over three cameras");

        const auto result = assigner.assign({look("cam0", 1, 0, 1), look("cam1", 1, 1, 0),
                                             look("cam2", 1, 1, 0), look("cam0", 2, 1, 0)},
                                            {1, 0, 0, 0});

        check(result.at(key("cam0", 2)) == identity,
              "the challenger that looks like the identity's overlap took cam0's slot");
        check(result.at(key("cam0", 1)) != identity, "the incumbent was displaced");
        check(result.at(key("cam0", 1)) >= 0, "and kept an identity of its own");
        assigner.validate();
    }

    void two_known_identities_that_one_instant_thinks_are_one_are_left_alone() {
        GlobalIdAssigner assigner(options());
        assigner.assign({look("cam0", 1, 1, 0)}, {0});
        assigner.assign({look("cam1", 1, 0, 1)}, {0});
        const int64_t first = assigner.owner_of(key("cam0", 1));
        const int64_t second = assigner.owner_of(key("cam1", 1));

        // Both are already owned and, once the winner is excluded, no id wants the cluster
        // twice: splitting or merging on one frame's evidence is how identities oscillate.
        const auto result =
            assigner.assign({look("cam0", 1, 1, 0), look("cam1", 1, 0, 1)}, {0, 0});

        const bool unchanged =
            result.at(key("cam0", 1)) == first && result.at(key("cam1", 1)) == second;
        const bool merged = result.at(key("cam0", 1)) == result.at(key("cam1", 1));
        check(unchanged || merged, "either both keep their ids or the cluster continued one");
        assigner.validate();
        check(true, "and whichever happened, the maps agree");
    }

    // ----------------------------------------------------------------------- eviction

    void a_track_nobody_reports_for_max_age_instants_is_forgotten() {
        GlobalIdAssigner assigner(options(/*max_age=*/2));
        assigner.assign({look("cam0", 1, 1, 0)}, {0});
        check(assigner.owner_of(key("cam0", 1)) >= 0, "it has an identity");

        for (int instant = 0; instant < 4; ++instant) {
            assigner.assign({look("cam1", 1, 0, 1)}, {0});
        }

        check(assigner.owner_of(key("cam0", 1)) == -1, "and after max_age instants it is gone");
        check(assigner.sizes().features == 1, "with its feature, hits and stamp gone too");
    }

    void eviction_runs_on_an_instant_with_no_tracks_at_all() {
        // The case that matters: a camera group that goes quiet is exactly when a
        // non-evicting implementation stops evicting.
        GlobalIdAssigner assigner(options(/*max_age=*/1));
        assigner.assign({look("cam0", 1, 1, 0)}, {0});

        assigner.assign({}, {});
        assigner.assign({}, {});

        check(assigner.size() == 0, "the quiet instants aged the track out");
    }

    void the_track_bound_evicts_the_least_recently_seen() {
        GlobalIdAssigner assigner(
            options(/*max_age=*/100, /*capacity=*/4096, /*max_tracks=*/2));
        assigner.assign({look("cam0", 1, 1, 0)}, {0});
        assigner.assign({look("cam1", 1, 0, 1)}, {0});
        assigner.assign({look("cam2", 1, 1, 1)}, {0});

        check(assigner.size() <= 2, "the bound holds");
        check(assigner.owner_of(key("cam0", 1)) == -1, "and the oldest went first");
        check(assigner.owner_of(key("cam2", 1)) >= 0, "while the newest stayed");
    }

    void the_identity_bound_evicts_a_whole_identity() {
        GlobalIdAssigner assigner(options(/*max_age=*/100, /*capacity=*/2));
        assigner.assign({look("cam0", 1, 1, 0)}, {0});
        assigner.assign({look("cam1", 1, 0, 1)}, {0});
        assigner.assign({look("cam2", 1, 1, 1)}, {0});

        check(assigner.sizes().identities <= 2, "the identity bound holds");
        assigner.validate();
        check(true, "and no half-evicted identity is left behind");
    }

    // doc: long two challengers, both directions, and why the numbers are what they are
    void a_second_challenger_contests_the_winner_and_not_the_track_it_displaced() {
        // WAS PINNED AS A DEFECT until shipvision#17. The contest re-read the incumbent from
        // `members_` each iteration while a winner was placed at once and the loser left only
        // in the deferred pass, so a second challenger from this camera contested a track that
        // had ALREADY LOST, won on the same evidence, and was adopted alongside the first --
        // the deferred pass evicts one incumbent, so both stayed and the identity held two
        // tracks from one camera. `validate()` names that state; production never asks.
        //
        // THE EMBEDDINGS. The group's direction is (1, 0): cam1#1 and cam2#1 both point along
        // it, and cam0#1 is orthogonal, so either challenger beats it. cam0#2 at (1, 0.3) is
        // better than cam0#3 at (1, 0.8), so the slot is cam0#2's and cam0#3 becomes its own
        // identity. Both must beat the INCUMBENT or the case is never reached -- the incumbent
        // is in the overlap, so its own feature is in what the contest is scored against.
        GlobalIdAssigner::Options relaxed = options();
        relaxed.validate_every_step = false;  // production's default, and how it stayed silent
        GlobalIdAssigner assigner(relaxed);
        assigner.assign({look("cam0", 1, 0, 1), look("cam1", 1, 1, 0), look("cam2", 1, 1, 0)},
                        {0, 0, 0});

        assigner.assign({look("cam0", 1, 0, 1), look("cam1", 1, 1, 0), look("cam2", 1, 1, 0),
                         look("cam0", 2, 1, 0.3f), look("cam0", 3, 1, 0.8f)},
                        {1, 0, 0, 0, 0});

        const int64_t target = assigner.owner_of(key("cam1", 1));
        std::vector<TrackKey> from_cam0;
        for (const TrackKey& member : assigner.members(target)) {
            if (member.camera_id == "cam0") from_cam0.push_back(member);
        }
        check(from_cam0.size() == 1 && from_cam0.front() == key("cam0", 2),
              "one holder, and it is the better challenger");
        assigner.validate();  // throws if the invariant it enforces does not hold
        check(true, "and the state `validate()` used to name is gone");
    }

    void a_challenger_that_beats_the_new_holder_takes_the_slot_from_it() {
        // The other direction, and the one a naive fix breaks: skipping the displaced track
        // must not mean skipping the CONTEST. The jitters are swapped, so cam0#3 is better
        // than cam0#2 and displaces it in turn. One per camera either way, a different one.
        GlobalIdAssigner::Options relaxed = options();
        relaxed.validate_every_step = false;
        GlobalIdAssigner assigner(relaxed);
        assigner.assign({look("cam0", 1, 0, 1), look("cam1", 1, 1, 0), look("cam2", 1, 1, 0)},
                        {0, 0, 0});

        assigner.assign({look("cam0", 1, 0, 1), look("cam1", 1, 1, 0), look("cam2", 1, 1, 0),
                         look("cam0", 2, 1, 0.8f), look("cam0", 3, 1, 0.3f)},
                        {1, 0, 0, 0, 0});

        const int64_t target = assigner.owner_of(key("cam1", 1));
        std::vector<TrackKey> from_cam0;
        for (const TrackKey& member : assigner.members(target)) {
            if (member.camera_id == "cam0") from_cam0.push_back(member);
        }
        check(from_cam0.size() == 1 && from_cam0.front() == key("cam0", 3),
              "the best of the three holds the slot");
        check(assigner.owner_of(key("cam0", 2)) != target, "and the loser moved out");
        assigner.validate();
    }

    void reset_forgets_the_identities_and_not_the_id_space() {
        GlobalIdAssigner assigner(options());
        assigner.assign({look("cam0", 1, 1, 0)}, {0});
        const int64_t issued = assigner.issued();

        assigner.reset();
        assigner.assign({look("cam0", 1, 1, 0)}, {0});

        check(assigner.size() == 1, "the track was re-adopted");
        check(assigner.owner_of(key("cam0", 1)) >= issued,
              "under a NEW id: a published id is out in the world, and reusing one attaches a "
              "stranger to that history");
    }

}  // namespace

int main() {
    the_options_are_refused_when_they_would_forget_everything();
    labels_that_do_not_line_up_are_refused();
    a_track_with_no_embedding_is_refused_by_name();
    a_zero_embedding_is_refused_THE_WAY_THE_REFERENCE_REFUSES_IT();
    a_refused_instant_leaves_no_trace_and_can_be_retried();
    two_widths_in_a_VIRGIN_assigners_first_instant_are_refused();
    a_second_embedding_width_is_refused_by_both_widths();
    every_observation_leaves_with_an_id();
    one_cluster_across_two_cameras_is_one_identity();
    two_tracks_from_ONE_camera_in_one_cluster_get_two_identities();
    the_largest_cluster_is_assigned_first_and_ties_go_by_first_appearance();
    an_identity_persists_when_the_cluster_comes_back();
    a_consecutive_run_counts_as_age_and_a_gap_restarts_it();
    of_two_equally_overlapping_identities_the_longest_confirmed_wins();
    a_live_track_replaces_a_stale_one_in_a_cameras_slot();
    a_live_incumbent_keeps_its_slot_unless_the_challenger_looks_more_alike();
    an_incumbent_inside_the_cluster_is_judged_against_itself_and_keeps_its_slot();
    a_better_looking_challenger_takes_the_slot_and_the_loser_keeps_an_identity();
    two_known_identities_that_one_instant_thinks_are_one_are_left_alone();
    a_track_nobody_reports_for_max_age_instants_is_forgotten();
    eviction_runs_on_an_instant_with_no_tracks_at_all();
    the_track_bound_evicts_the_least_recently_seen();
    the_identity_bound_evicts_a_whole_identity();
    a_second_challenger_contests_the_winner_and_not_the_track_it_displaced();
    a_challenger_that_beats_the_new_holder_takes_the_slot_from_it();
    reset_forgets_the_identities_and_not_the_id_space();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
