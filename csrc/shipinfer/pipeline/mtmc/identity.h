// Cross-camera global ids, maintained across synchronised instants.
//
// The C++ twin of `shipvision/mtmc/identity.py`. It exists HERE and not in the submodule for
// the reason the submodule's own header gives (`shipvision/mtmc/frames.h`): "`TrackKey`, the
// `Track` itself and the embedding do not cross. An identity map keyed on (camera, track) is
// Python's to own -- it is the stateful half." The library ships the stateless (n, n) passes
// and the clusterer; the identity bookkeeping is the caller's, so a C++ mtmc stage needs this.
//
// doc: long what this owns, and the one invariant everything else rests on
// WHAT IT OWNS. Cluster labels say which tracks look like one object AT ONE INSTANT and
// nothing more -- they are not identities and they are not stable between calls. This class
// turns a stream of labellings into identities that persist: it remembers each track's
// appearance, how long it has been continuously confirmed and when it was last seen, and it
// decides which existing identity a cluster continues.
//
// THE INVARIANT. `owner_[k] == g` exactly when `k` appears in `members_[g]`, and one identity
// holds at most one track per camera. Both are checked by `validate()` rather than believed:
// two representations of one fact can disagree, and when they do the output stays PLAUSIBLE --
// an identity quietly holding two tracks from one camera, an id nothing can be found under.
//
// BOTH HALVES HOLD THROUGH A CONTESTED CLUSTER NOW. They did not: a winner was placed at once
// and the loser left in the deferred pass, so a second challenger from that camera contested a
// track that had already lost and was adopted alongside the first. The contest is against the
// CURRENT holder, upstream and here (`MTMC-ONE-CAMERA-TWICE-IN-A-CONTESTED-CLUSTER`).
//
// NOT THREAD-SAFE, deliberately. The stage that owns it takes the barrier's lock for the whole
// association, which is the level where one instant is one atomic step.
#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"

namespace shipinfer::mtmc {

    // One single-camera track, across cameras: which camera, and that camera's track id.
    //
    // ORDERED, and by (camera, track) in that order, because two eviction paths sort by this
    // key to break ties -- so the ordering is part of the behaviour rather than an incidental
    // property of the container.
    struct TrackKey {
        std::string camera_id;
        int64_t track_id = 0;

        bool operator<(const TrackKey& other) const {
            if (camera_id != other.camera_id) return camera_id < other.camera_id;
            return track_id < other.track_id;
        }
        bool operator==(const TrackKey& other) const {
            return camera_id == other.camera_id && track_id == other.track_id;
        }
        std::string str() const { return camera_id + "#" + std::to_string(track_id); }
    };

    // What the assigner needs from one observation. The box and the frame extent go to the
    // MATCHER, which is the submodule's; identity is decided on appearance and the key.
    struct IdentityObservation {
        TrackKey key;
        //: The track's embedding. Normalised on arrival, so similarity is a dot product.
        std::vector<float> embedding;
    };

    struct IdentitySizes {
        size_t identities = 0;
        size_t tracks = 0;
        size_t features = 0;
        int64_t issued = 0;
        int64_t step = 0;
    };

    // NAMESPACE SCOPE and not nested, for the reason `BarrierOptions` is: a nested
    // struct's default member initialisers are not available while the enclosing class
    // is still being defined, so `Options options = {}` below would not compile.
    struct AssignerOptions {
        //: How many consecutive instants a track may go unobserved before it is
        //: forgotten. Counted in INSTANTS because the instant is this component's clock:
        //: it is called once per synchronised group, and wall-clock ageing would make
        //: eviction depend on decoder timestamps that stop advancing exactly when a
        //: camera is in trouble.
        int max_age = 30;
        //: The maximum number of live identities; on overflow the least recently seen is
        //: evicted whole. The backstop a per-track age cannot cover -- an incident that
        //: produces thousands of simultaneous identities.
        int capacity = 4096;
        //: The maximum number of single-camera tracks across all identities. The second
        //: bound: one identity that keeps acquiring members must not fill memory alone.
        int max_tracks = 8192;
        //: Run `validate()` after every instant. Off by default -- it is O(tracked) work
        //: against a bug that should not exist -- on in tests, worth it in a canary.
        bool validate_every_step = false;
    };

    class GlobalIdAssigner {
      public:
        using Options = AssignerOptions;

        explicit GlobalIdAssigner(Options options = {});

        // Advance one instant: every observation leaves with a global id.
        //
        // `labels` is one cluster label per observation, as the clusterer produced them; only
        // EQUALITY between labels is read. Never partial and never absent: an observation
        // that reached this point is a track the caller decided to trust, so one that matches
        // nothing starts a new identity rather than being dropped.
        //: EVERY EMBEDDING IS CHECKED BEFORE ANYTHING IS MUTATED, so a refusal leaves the
        //: instant unapplied and the caller may retry it. `++step_` and `observe`'s writes
        //: used to run first, so a bad fifth observation left a half-written history the next
        //: instant read as fact -- and `reset()` was the only way back.
        std::map<TrackKey, int64_t> assign(const std::vector<IdentityObservation>& observations,
                                           const std::vector<int>& labels);

        // Forget every identity. Does NOT rewind the id space: ids that have been published
        // are out in the world, and reusing one attaches a stranger to that history.
        void reset();

        // Assert that the maps agree, or throw. See the header's INVARIANT.
        void validate() const;

        int64_t step() const { return step_; }
        int64_t issued() const { return counter_; }
        size_t size() const { return owner_.size(); }
        IdentitySizes sizes() const;
        std::vector<int64_t> global_ids() const;
        std::vector<TrackKey> members(int64_t global_id) const;
        //: The member list ITSELF, for the paths that only read it. `members()` above copies
        //: because it is the public accessor and two callers mutate the identity while
        //: walking it; the contest and the checks do not, and a copy per candidate per group
        //: is an allocation on the assign path.
        const std::vector<TrackKey>& member_list(int64_t global_id) const;
        //: The identity holding `key`, or -1. -1 rather than an optional because every caller
        //: here compares against a real id and `-1` is not one.
        int64_t owner_of(const TrackKey& key) const;
        int hits(const TrackKey& key) const;

      private:
        void observe(const std::vector<IdentityObservation>& observations);
        //: The pre-pass: empty, all-zero and mixed-width embeddings, refused before any state
        //: moves -- including two widths inside ONE instant, which is the case a virgin
        //: assigner has and #220's review found missing. Width is carried across instants
        //: too, so a second embedder shows up here rather than deep inside a similarity that
        //: has already assigned two groups. NOT `const`: `width_` is the identity space's
        //: embedder width, which changes this class's future answers, and marking the method
        //: `const` with a `mutable` member is what made "seed the width first" look illegal.
        void check_embeddings(const std::vector<IdentityObservation>& observations);
        //: Indices grouped by label, LARGEST GROUP FIRST, ties by first appearance. The
        //: reference's `reorderCluster`. Deterministic ordering is the point: the same input
        //: produces the same ids twice, which is the difference between a reproducible bug
        //: and a haunting.
        std::vector<std::vector<size_t>> ordered_groups(const std::vector<int>& labels) const;
        //: KEYS, not observations: the features are already in `features_`, normalised by
        //: `observe`, so taking the observations copied an embedding per track per instant --
        //: 750 heap copies of a real 2048-float vector at the design load, for a `.key`.
        void assign_group(const std::vector<TrackKey>& keys, std::set<int64_t>& matched);
        void resolve_between_identities(const TrackKey& key, int64_t owner, int64_t target,
                                        const std::vector<const std::vector<float>*>& overlap);
        //: `exclude` is a SET: it grows by one per group, and a linear scan per cluster key
        //: made this O(groups^2) over an instant of ~750 mostly-singleton clusters.
        std::vector<int64_t> candidate_ids(const std::vector<TrackKey>& keys,
                                           const std::set<int64_t>& exclude) const;
        int64_t select_by_oldest(const std::vector<int64_t>& candidates) const;
        //: This identity's existing track on `camera_id`, or absent. One identity holds at
        //: most one track per camera: it is one object, and a camera that sees it twice at one
        //: instant has a single-camera failure, not a cross-camera one.
        //: `ignoring` is the tracks already displaced this cluster. They are still in
        //: `members_` -- losers leave in a deferred pass -- and skipping them is what makes
        //: this the CURRENT holder rather than one that has already lost.
        const TrackKey* member_from_camera(int64_t global_id, const std::string& camera_id,
                                           const std::vector<TrackKey>& ignoring = {}) const;
        //: Cosine similarity against several, reduced by mean or max. Zero when there is
        //: nothing to compare against -- not an error and not a one: it makes appearance
        //: silent and lets the caller's other rules decide.
        static double similarity(const std::vector<float>& feature,
                                 const std::vector<const std::vector<float>*>& others,
                                 bool use_max);
        int64_t issue() { return counter_++; }
        void adopt(const TrackKey& key, int64_t global_id);
        void move(const TrackKey& key, int64_t source, int64_t destination);
        void place(const TrackKey& key, int64_t target, int64_t owner);
        void detach(const TrackKey& key, int64_t global_id);
        void forget(const TrackKey& key);
        void evict();

        Options options_;
        int64_t counter_ = 0;
        int64_t step_ = 0;
        //: The embedding width this identity space was built on, 0 until the first
        //: observation. One identity space is fed by one embedder.
        size_t width_ = 0;
        //: Identity -> its member tracks, in the order they joined. A vector and not a set
        //: because `member_from_camera` answers with the FIRST match and that has to be
        //: stable.
        std::map<int64_t, std::vector<TrackKey>> members_;
        std::map<TrackKey, int64_t> owner_;
        std::map<TrackKey, std::vector<float>> features_;
        std::map<TrackKey, int> hits_;
        std::map<TrackKey, int64_t> last_seen_;
    };

}  // namespace shipinfer::mtmc
