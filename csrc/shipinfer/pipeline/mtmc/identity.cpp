#include "shipinfer/pipeline/mtmc/identity.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace shipinfer::mtmc {

    namespace {

        std::vector<float> normalised(const std::vector<float>& embedding) {
            double sum = 0.0;
            for (const float value : embedding) sum += static_cast<double>(value) * value;
            const double norm = std::sqrt(sum);
            // A ZERO VECTOR IS REFUSED here as well as in `check_embeddings`, which is the
            // only caller that reaches this today: this is the last line of defence for a
            // future one that skips the pre-pass, and it is the reference's own rule --
            // measured, in `shipvision/types.py::_as_unit_vector`: "an all-zero embedding has
            // no direction ... Left alone it sits at cosine 0 from every gallery entry, which
            // is a plausible-looking answer to every query rather than an obvious failure." A
            // port that answered 0 here would diverge in a REFUSAL, which no golden can
            // catch: the reference raises before it answers anything to compare against.
            if (!(norm > 0.0)) {
                throw InferenceError(
                    "an all-zero track embedding has no direction, so it cannot be "
                    "normalised; at cosine 0 from everything it is a plausible answer to "
                    "every query rather than an obvious failure");
            }
            std::vector<float> out(embedding.size());
            for (size_t i = 0; i < embedding.size(); ++i) {
                out[i] = static_cast<float>(static_cast<double>(embedding[i]) / norm);
            }
            return out;
        }

        bool contains(const std::vector<TrackKey>& keys, const TrackKey& key) {
            return std::find(keys.begin(), keys.end(), key) != keys.end();
        }

    }  // namespace

    GlobalIdAssigner::GlobalIdAssigner(Options options) : options_(options) {
        if (options_.max_age < 1) {
            throw ConfigError(
                "max_age must be at least 1 instant; 0 would forget every track "
                "between consecutive frames, got " +
                std::to_string(options_.max_age));
        }
        if (options_.capacity < 1) {
            throw ConfigError("capacity must be positive, got " +
                              std::to_string(options_.capacity));
        }
        if (options_.max_tracks < 1) {
            throw ConfigError("max_tracks must be positive, got " +
                              std::to_string(options_.max_tracks));
        }
    }

    IdentitySizes GlobalIdAssigner::sizes() const {
        IdentitySizes out;
        out.identities = members_.size();
        out.tracks = owner_.size();
        out.features = features_.size();
        out.issued = counter_;
        out.step = step_;
        return out;
    }

    std::vector<int64_t> GlobalIdAssigner::global_ids() const {
        std::vector<int64_t> out;
        for (const auto& [global_id, _] : members_) out.push_back(global_id);
        return out;
    }

    std::vector<TrackKey> GlobalIdAssigner::members(int64_t global_id) const {
        const auto found = members_.find(global_id);
        return found == members_.end() ? std::vector<TrackKey>{} : found->second;
    }

    const std::vector<TrackKey>& GlobalIdAssigner::member_list(int64_t global_id) const {
        static const std::vector<TrackKey> none;
        const auto found = members_.find(global_id);
        return found == members_.end() ? none : found->second;
    }

    int64_t GlobalIdAssigner::owner_of(const TrackKey& key) const {
        const auto found = owner_.find(key);
        return found == owner_.end() ? -1 : found->second;
    }

    int GlobalIdAssigner::hits(const TrackKey& key) const {
        const auto found = hits_.find(key);
        return found == hits_.end() ? 0 : found->second;
    }

    void GlobalIdAssigner::reset() {
        // `width_` SURVIVES, like `counter_` and `step_`: an identity space is fed by one
        // embedder for the life of the process, so a post-reset chain with a different one is
        // refused rather than silently merged into the old space's history. #220's review
        // asked for this to be stated rather than inferred.
        members_.clear();
        owner_.clear();
        features_.clear();
        hits_.clear();
        last_seen_.clear();
    }

    void GlobalIdAssigner::check_embeddings(
        const std::vector<IdentityObservation>& observations) {
        // SEEDED FROM THE FIRST OBSERVATION, not only from previous instants. Guarding on
        // `width_ != 0` compared nothing inside a virgin assigner's first instant, so a chain
        // with two embedders merged two incomparable tracks into one global id and only then
        // began refusing -- and `gate.h` is explicit that a merge is the unrecoverable
        // direction. #220's review found it; the width is local to the pass now.
        size_t width = width_;
        for (const IdentityObservation& observation : observations) {
            const std::string who = observation.key.str();
            if (observation.embedding.empty()) {
                throw InferenceError(who +
                                     " has no embedding; cross-camera identity is decided on "
                                     "appearance, so an un-embedded track cannot be assigned");
            }
            if (width == 0) width = observation.embedding.size();
            if (observation.embedding.size() != width) {
                // HERE rather than in `similarity`, which is reached only after groups have
                // been assigned: a second embedder is a load-time misconfiguration and this is
                // the first place that can say so without having changed anything.
                throw InferenceError(
                    who + " carries a " + std::to_string(observation.embedding.size()) +
                    "-dimensional embedding and this identity space was built on " +
                    std::to_string(width) + "; one identity space is fed by one embedder");
            }
            double sum = 0.0;
            for (const float value : observation.embedding) {
                sum += static_cast<double>(value) * value;
            }
            if (!(sum > 0.0)) {
                throw InferenceError(who +
                                     ": an all-zero track embedding has no direction, so it "
                                     "cannot be normalised; at cosine 0 from everything it is "
                                     "a plausible answer to every query rather than an "
                                     "obvious failure");
            }
        }
        // LAST, so a refusal above leaves even this untouched: the instant is unapplied and
        // the caller may send it again.
        width_ = width;
    }

    void GlobalIdAssigner::observe(const std::vector<IdentityObservation>& observations) {
        for (const IdentityObservation& observation : observations) {
            // NO CHECKS HERE. `check_embeddings` owns them and runs before anything moves;
            // repeating them was two places encoding one rule, which is the drift its own
            // comment warns about. `normalised` keeps its refusal as the last line of defence
            // for a caller that is not `assign`, and says so at the line.
            const TrackKey& key = observation.key;
            features_[key] = normalised(observation.embedding);
            // CONSECUTIVE hits, which is what `select_by_oldest` reads as age: a track seen
            // at this instant and the one before continues its run, and any gap restarts it.
            const auto seen = last_seen_.find(key);
            const bool consecutive = seen != last_seen_.end() && seen->second == step_ - 1;
            hits_[key] = consecutive ? hits_[key] + 1 : 1;
            last_seen_[key] = step_;
        }
    }

    std::vector<std::vector<size_t>> GlobalIdAssigner::ordered_groups(
        const std::vector<int>& labels) const {
        // ONE PASS. `grouped` comes out in first-appearance order because a label appends a
        // group the first time it is seen, so the sort below needs no separate `first` table
        // -- and nothing re-scans `labels` per distinct label, which at ~750 tracks an
        // instant of mostly singleton clusters walked the whole vector ~750 times.
        std::map<int, size_t> where;
        std::vector<std::vector<size_t>> grouped;
        for (size_t index = 0; index < labels.size(); ++index) {
            const auto [at, fresh] = where.emplace(labels[index], grouped.size());
            if (fresh) grouped.emplace_back();
            grouped[at->second].push_back(index);
        }
        // Largest first, ties by FIRST APPEARANCE and not by label value. A STABLE sort over
        // groups that are already in first-appearance order gives both at once -- and `sort`
        // would not: libstdc++ is incidentally stable only below its insertion-sort
        // threshold, which `wide_tie_keeps_first_appearance` is deliberately wider than.
        std::stable_sort(grouped.begin(), grouped.end(),
                         [](const std::vector<size_t>& left, const std::vector<size_t>& right) {
                             return left.size() > right.size();
                         });
        return grouped;
    }

    double GlobalIdAssigner::similarity(const std::vector<float>& feature,
                                        const std::vector<const std::vector<float>*>& others,
                                        bool use_max) {
        if (others.empty()) return 0.0;
        double total = 0.0;
        double best = -1.0;
        for (const std::vector<float>* other : others) {
            if (other->size() != feature.size()) {
                // The reference gets numpy's ValueError here; truncating to the shorter
                // prefix answers a number that is not a cosine, because `observe` normalised
                // the FULL vector on both sides. One embedder per identity space is a
                // load-time fact, so a second width is a misconfiguration to name and refuse.
                throw InferenceError("cross-camera identity compares embeddings of " +
                                     std::to_string(feature.size()) + " and " +
                                     std::to_string(other->size()) +
                                     " dimensions; one identity space is fed by one embedder");
            }
            double dot = 0.0;
            for (size_t i = 0; i < feature.size(); ++i) {
                dot += static_cast<double>(feature[i]) * (*other)[i];
            }
            total += dot;
            best = std::max(best, dot);
        }
        // A DOT PRODUCT IS THE COSINE here because `observe` normalised both sides. Stated
        // rather than implied: an un-normalised feature reaching this would make every
        // comparison a magnitude contest and the ids would follow brightness.
        return use_max ? best : total / static_cast<double>(others.size());
    }

    void GlobalIdAssigner::adopt(const TrackKey& key, int64_t global_id) {
        owner_[key] = global_id;
        members_[global_id].push_back(key);
    }

    void GlobalIdAssigner::detach(const TrackKey& key, int64_t global_id) {
        const auto found = members_.find(global_id);
        if (found == members_.end()) return;
        std::vector<TrackKey>& keys = found->second;
        keys.erase(std::remove(keys.begin(), keys.end(), key), keys.end());
        // DROPPED AS SOON AS IT EMPTIES. The reference kept recently-created empty identities
        // on the theory that they might come back; nothing ever looked them up, so they were
        // a slow leak with a comment on it.
        if (keys.empty()) members_.erase(found);
    }

    void GlobalIdAssigner::move(const TrackKey& key, int64_t source, int64_t destination) {
        if (source == destination) return;
        detach(key, source);
        adopt(key, destination);
    }

    void GlobalIdAssigner::place(const TrackKey& key, int64_t target, int64_t owner) {
        if (owner < 0) {
            adopt(key, target);
        } else {
            move(key, owner, target);
        }
    }

    void GlobalIdAssigner::forget(const TrackKey& key) {
        const auto owned = owner_.find(key);
        if (owned != owner_.end()) {
            const int64_t global_id = owned->second;
            owner_.erase(owned);
            detach(key, global_id);
        }
        features_.erase(key);
        hits_.erase(key);
        last_seen_.erase(key);
    }

    const TrackKey* GlobalIdAssigner::member_from_camera(
        int64_t global_id, const std::string& camera_id,
        const std::vector<TrackKey>& ignoring) const {
        const auto found = members_.find(global_id);
        if (found == members_.end()) return nullptr;
        for (const TrackKey& member : found->second) {
            if (member.camera_id != camera_id) continue;
            if (contains(ignoring, member)) continue;
            return &member;
        }
        return nullptr;
    }

    std::vector<int64_t> GlobalIdAssigner::candidate_ids(
        const std::vector<TrackKey>& keys, const std::set<int64_t>& exclude) const {
        // COUNTED rather than scanned. The reference walks its whole global storage and
        // intersects each identity's member list with the cluster; counting how many of the
        // cluster's keys each id already owns is identical -- `owner_[k] == g` exactly when
        // `k` is in `members_[g]`, which `validate` enforces -- and is O(cluster) instead of
        // O(live identities).
        std::map<int64_t, size_t> counts;
        for (const TrackKey& key : keys) {
            const int64_t owner = owner_of(key);
            if (owner < 0) continue;
            if (exclude.count(owner) != 0) continue;
            ++counts[owner];
        }
        size_t best = 0;
        for (const auto& [global_id, count] : counts) best = std::max(best, count);
        std::vector<int64_t> out;
        if (best == 0) return out;
        // `counts` is a std::map, so this comes out sorted by id -- which is what makes a
        // downstream tie resolve the same way every run.
        for (const auto& [global_id, count] : counts) {
            if (count == best) out.push_back(global_id);
        }
        return out;
    }

    int64_t GlobalIdAssigner::select_by_oldest(const std::vector<int64_t>& candidates) const {
        // AGE IS CONSECUTIVE OBSERVATIONS, so "oldest" means "continuously confirmed the
        // longest" rather than "created first": an identity watched for two hundred frames
        // has more evidence than one created forty minutes ago and seen twice. Ties go to the
        // lowest id, which by construction is the one created first.
        if (candidates.size() == 1) return candidates.front();
        int64_t best_id = candidates.front();
        int best_age = -1;
        for (const int64_t global_id : candidates) {
            int age = 0;
            for (const TrackKey& member : member_list(global_id)) {
                age = std::max(age, hits(member));
            }
            if (age > best_age) {
                best_id = global_id;
                best_age = age;
            }
        }
        return best_id;
    }

    void GlobalIdAssigner::resolve_between_identities(
        const TrackKey& key, int64_t owner, int64_t target,
        const std::vector<const std::vector<float>*>& overlap) {
        // Its current identity has no track from this camera, so nothing is displaced either
        // way; the question is only which group this track looks more like. On the MAXIMUM
        // similarity and not the mean, because an identity's members are views from different
        // angles and the mean punishes an identity for holding a bad angle -- the normal state
        // of a large group.
        std::vector<const std::vector<float>*> rest;
        for (const TrackKey& member : member_list(owner)) {
            if (member == key) continue;
            const auto feature = features_.find(member);
            if (feature != features_.end()) rest.push_back(&feature->second);
        }
        const std::vector<float>& mine = features_.at(key);
        if (similarity(mine, overlap, true) > similarity(mine, rest, true)) {
            move(key, owner, target);
        }
    }

    void GlobalIdAssigner::assign_group(const std::vector<TrackKey>& keys,
                                        std::set<int64_t>& matched) {
        const std::vector<int64_t> candidates = candidate_ids(keys, matched);

        if (candidates.empty()) {
            std::vector<TrackKey> fresh;
            for (const TrackKey& key : keys) {
                if (owner_of(key) < 0) fresh.push_back(key);
            }
            if (fresh.empty()) {
                // Every member already has an id and no id wants this cluster: two known
                // identities this instant thinks are one. Splitting or merging them on one
                // frame's evidence is how identities oscillate, so nothing happens.
                return;
            }
            // ONE IDENTITY PER CAMERA, even for a brand-new cluster -- and the reference
            // splits it the same way: `same_camera_twice` in `golden/identity/basic.txt` is
            // `cam0#1=0 cam0#2=1`, and removing this guard fails the parity gate, not just
            // the unit test. Two same-camera keys reach one cluster when the matcher's
            // exclusion mask was bypassed; the answer is two identities, on both planes.
            const int64_t global_id = issue();
            std::vector<std::string> claimed;
            for (const TrackKey& key : fresh) {
                if (std::find(claimed.begin(), claimed.end(), key.camera_id) != claimed.end()) {
                    adopt(key, issue());
                    continue;
                }
                claimed.push_back(key.camera_id);
                adopt(key, global_id);
            }
            matched.insert(global_id);
            return;
        }

        const int64_t target = select_by_oldest(candidates);
        matched.insert(target);
        const std::vector<TrackKey> members_of_target = members(target);
        std::vector<const std::vector<float>*> overlap_features;
        std::vector<TrackKey> non_overlap;
        for (const TrackKey& key : keys) {
            if (contains(members_of_target, key)) {
                const auto feature = features_.find(key);
                if (feature != features_.end()) overlap_features.push_back(&feature->second);
            } else {
                non_overlap.push_back(key);
            }
        }
        std::vector<TrackKey> displaced;

        for (const TrackKey& key : non_overlap) {
            // AGAINST THE CURRENT HOLDER, not the one this cluster started with. A winner is
            // placed at once and the loser leaves in the deferred pass below, so a second
            // challenger from this camera used to contest a track that had already lost, win
            // on the same evidence, and be adopted alongside it -- both then stayed.
            const TrackKey* incumbent_ptr =
                member_from_camera(target, key.camera_id, displaced);
            const int64_t owner = owner_of(key);

            if (incumbent_ptr == nullptr) {
                if (owner < 0) {
                    adopt(key, target);
                } else if (owner != target) {
                    resolve_between_identities(key, owner, target, overlap_features);
                }
                continue;
            }
            // COPIED, because `place`/`forget` below mutate the member vector this points into.
            const TrackKey incumbent = *incumbent_ptr;
            // AND `overlap_features` HOLDS RAW POINTERS INTO `features_`, which `forget`
            // erases from a few lines down. Safe for one reason and only one: `forget` is
            // reached only when the incumbent was NOT observed this instant, so it is not in
            // `keys`, so its feature is not among the pointers. An edit that lets an observed
            // incumbent be forgotten makes this a use-after-free, and ASan will not see it
            // until a scenario contains that case -- so it is stated rather than left to be
            // rediscovered.

            const auto seen = last_seen_.find(incumbent);
            if (seen == last_seen_.end() || seen->second < step_) {
                // The camera's slot is held by a track nobody has seen this instant. A live
                // track outranks a stale one: the single-camera tracker replaced it, which is
                // what happens when a person is briefly occluded and comes back with a new id.
                forget(incumbent);
                place(key, target, owner);
                continue;
            }

            const double contest = similarity(features_.at(incumbent), overlap_features, false);
            const double challenge = similarity(features_.at(key), overlap_features, false);
            if (challenge > contest) {
                place(key, target, owner);
                displaced.push_back(incumbent);
            } else if (owner < 0) {
                // It lost and has no identity of its own: a real object this identity has no
                // room for, so it becomes its own identity rather than being published
                // unassigned.
                adopt(key, issue());
            }
        }

        // Losers move out only after the whole cluster is processed, so a chain of contests
        // within one cluster is judged against the state it started from.
        for (const TrackKey& key : displaced) {
            if (owner_of(key) == target) move(key, target, issue());
        }
    }

    void GlobalIdAssigner::evict() {
        // Runs on EVERY instant, including ones with no tracks at all -- which is the case
        // that matters: a camera group that goes quiet is exactly when a non-evicting
        // implementation stops evicting.
        const int64_t cutoff = step_ - options_.max_age;
        std::vector<TrackKey> stale;
        for (const auto& [key, seen] : last_seen_) {
            if (seen < cutoff) stale.push_back(key);
        }
        for (const TrackKey& key : stale) forget(key);

        if (owner_.size() > static_cast<size_t>(options_.max_tracks)) {
            std::vector<TrackKey> oldest;
            for (const auto& [key, _] : last_seen_) oldest.push_back(key);
            std::sort(oldest.begin(), oldest.end(), [&](const TrackKey& a, const TrackKey& b) {
                const int64_t sa = last_seen_.at(a);
                const int64_t sb = last_seen_.at(b);
                return sa != sb ? sa < sb : a < b;
            });
            const size_t excess = owner_.size() - static_cast<size_t>(options_.max_tracks);
            for (size_t i = 0; i < excess && i < oldest.size(); ++i) forget(oldest[i]);
        }

        if (members_.size() > static_cast<size_t>(options_.capacity)) {
            // An identity's recency is its most recently seen MEMBER's: an identity with one
            // live track is in use however old its others are. Ties evict the HIGHER id -- the
            // more recently created identity, with the least history behind it -- and, more
            // importantly, the same way every run.
            std::vector<int64_t> by_recency = global_ids();
            const auto recency = [&](int64_t global_id) {
                int64_t seen = -1;
                for (const TrackKey& member : members_.at(global_id)) {
                    const auto found = last_seen_.find(member);
                    if (found != last_seen_.end()) seen = std::max(seen, found->second);
                }
                return seen;
            };
            std::sort(by_recency.begin(), by_recency.end(), [&](int64_t a, int64_t b) {
                const int64_t ra = recency(a);
                const int64_t rb = recency(b);
                return ra != rb ? ra < rb : a > b;
            });
            const size_t excess = members_.size() - static_cast<size_t>(options_.capacity);
            for (size_t i = 0; i < excess && i < by_recency.size(); ++i) {
                for (const TrackKey& member : members(by_recency[i])) forget(member);
            }
        }
    }

    void GlobalIdAssigner::validate() const {
        for (const auto& [global_id, keys] : members_) {
            if (keys.empty()) {
                throw InferenceError("global id " + std::to_string(global_id) +
                                     " has no members; an empty identity should have been "
                                     "dropped when its last member left");
            }
            std::vector<std::string> cameras;
            for (size_t i = 0; i < keys.size(); ++i) {
                for (size_t j = i + 1; j < keys.size(); ++j) {
                    if (keys[i] == keys[j]) {
                        throw InferenceError("global id " + std::to_string(global_id) +
                                             " lists " + keys[i].str() + " twice");
                    }
                }
                if (std::find(cameras.begin(), cameras.end(), keys[i].camera_id) !=
                    cameras.end()) {
                    throw InferenceError(
                        "global id " + std::to_string(global_id) +
                        " holds two tracks from one camera (" + keys[i].camera_id +
                        "). It is one object; a camera seeing it twice at one instant is a "
                        "single-camera tracking failure, and every per-camera lookup here "
                        "would resolve to whichever came first");
                }
                cameras.push_back(keys[i].camera_id);
                if (owner_of(keys[i]) != global_id) {
                    throw InferenceError(keys[i].str() + " is a member of global id " +
                                         std::to_string(global_id) + " but its owner is " +
                                         std::to_string(owner_of(keys[i])) +
                                         "; the forward and reverse maps have desynchronised");
                }
            }
        }
        for (const auto& [key, global_id] : owner_) {
            if (!contains(member_list(global_id), key)) {
                throw InferenceError(key.str() + " claims global id " +
                                     std::to_string(global_id) + ", which does not list it");
            }
            if (features_.count(key) == 0 || last_seen_.count(key) == 0) {
                throw InferenceError(key.str() +
                                     " has an identity but no feature or no last-seen stamp; "
                                     "every path that adopts a track records both");
            }
        }
    }

    std::map<TrackKey, int64_t> GlobalIdAssigner::assign(
        const std::vector<IdentityObservation>& observations, const std::vector<int>& labels) {
        if (labels.size() != observations.size()) {
            throw ConfigError(std::to_string(labels.size()) + " labels for " +
                              std::to_string(observations.size()) +
                              " observations; these come from one clustering call and must "
                              "line up");
        }
        // BEFORE `++step_` and before `observe`, which is the whole point: this used to throw
        // on the second of five observations with the first already written and the step
        // advanced, and nothing could retry the instant.
        check_embeddings(observations);
        ++step_;
        observe(observations);

        std::set<int64_t> matched;
        std::vector<TrackKey> keys;
        for (const std::vector<size_t>& indices : ordered_groups(labels)) {
            keys.clear();
            keys.reserve(indices.size());
            for (const size_t index : indices) keys.push_back(observations[index].key);
            assign_group(keys, matched);
        }

        std::map<TrackKey, int64_t> result;
        for (const IdentityObservation& observation : observations) {
            int64_t owner = owner_of(observation.key);
            if (owner < 0) {
                // Unreachable through the branches above; kept because "every observation
                // leaves with an id" is a guarantee this function makes, and a guarantee that
                // depends on a case analysis being exhaustive should be enforced rather than
                // believed.
                owner = issue();
                adopt(observation.key, owner);
            }
            result[observation.key] = owner;
        }

        evict();
        if (options_.validate_every_step) validate();
        return result;
    }

}  // namespace shipinfer::mtmc
