// shipvision's cross-camera tracker behind the `ClusterTracker` seam — the LANE's only mtmc
// unit, and the only file here that includes the submodule.
//
// doc: long what this composes, and which half of the reference lives where
// WHAT IT COMPOSES, in the order `ClusterMTMCTracker.track` composes it: the gate drops tracks
// too small or too new to trust; the surviving embeddings become one `(n, n)` gram matrix; the
// gated matcher turns that into clusterable distances, vetoing pairs no single object could
// occupy; the clusterer turns distances into labels; and the assigner turns labels into
// identities that persist. Four of those five are the submodule's stateless passes. The gate
// and the assigner are STATEFUL, which is why they live in this tree -- `mtmc/frames.h` says
// so: "an identity map keyed on (camera, track) is Python's to own -- it is the stateful half".
//
// Nothing here reaches `core/platform.h`, which is what lets the lane's CI job build it with
// g++ alone. The stage that reads a `FrameState` is on the other side of that line.
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "shipinfer/pipeline/mtmc/cluster.h"
#include "shipinfer/pipeline/mtmc/gate.h"
#include "shipinfer/pipeline/mtmc/identity.h"
#include "shipvision/mtmc/clustering/agglomerative.h"
#include "shipvision/mtmc/frames.h"
#include "shipvision/mtmc/matchers/gated/matcher.h"

namespace shipinfer::mtmc {

    namespace {

        // Camera identity crosses as an INTEGER CODE, never as a name, and `frames.h` gives the
        // measurement: the reference compared camera strings pairwise to build the same-camera
        // exclusion, which at fifty cameras and fifteen tracks each is 560 000 string compares
        // per instant -- on its own more expensive than the clustering it feeds. Assigned here
        // by FIRST APPEARANCE, per call: nothing compares codes across calls, so any consistent
        // numbering describes the same mask.
        std::vector<shipvision::mtmc::Observation> as_observations(
            const std::vector<ClusterObservation>& admitted) {
            std::vector<std::string> codes;
            std::vector<shipvision::mtmc::Observation> out;
            out.reserve(admitted.size());
            for (const ClusterObservation& observation : admitted) {
                // ONE SEARCH. The index is what the code IS, so it is taken from the search
                // that decided whether to append rather than from a second identical one
                // (#221's review).
                const auto found =
                    std::find(codes.begin(), codes.end(), observation.key.camera_id);
                const size_t code = static_cast<size_t>(found - codes.begin());
                if (found == codes.end()) codes.push_back(observation.key.camera_id);
                shipvision::mtmc::Observation converted;
                converted.camera_code = static_cast<int>(code);
                for (int i = 0; i < 4; ++i) converted.box[i] = observation.box[i];
                converted.frame_width = observation.frame_width;
                converted.frame_height = observation.frame_height;
                out.push_back(converted);
            }
            return out;
        }

        // `(n, n)` appearance similarity, which is what the matcher takes: the embeddings
        // "reach here already multiplied into a gram matrix, for the reason
        // `matchers/appearance/matcher.h` gives". Normalised on the way in, so a dot product
        // IS the cosine -- the same normalisation `GlobalIdAssigner::observe` applies, and for
        // the same reason: an un-normalised feature would make every comparison a magnitude
        // contest and the ids would follow brightness.
        // ONE FLAT BUFFER, `float` accumulation, and only the symmetric half computed.
        // #221's review measured the first draft -- `vector<vector<float>>`, `double`
        // accumulation, the full n x n -- at 136.7 ms for n=750/d=512 against a 50 ms instant
        // budget, and `ids()` holds its lock across it. This shape is ~2.2x faster at that size
        // and the measured table is in the PR body; it is still a scalar loop, and
        // `MTMC-GRAM-WANTS-A-REAL-GEMM` carries the numbers and the two ways out (a BLAS
        // `ssyrk` behind this lane, or a bound on the admitted count per instant).
        std::vector<float> gram_of(const std::vector<ClusterObservation>& admitted) {
            const size_t n = admitted.size();
            const size_t dim = n == 0 ? 0 : admitted.front().embedding.size();
            std::vector<float> unit(n * dim);
            for (size_t i = 0; i < n; ++i) {
                const std::vector<float>& embedding = admitted[i].embedding;
                // REFUSED, both of them, for the reasons `mtmc/identity.cpp` gives at the same
                // two lines: the reference raises on an all-zero embedding
                // (`shipvision/types.py::_as_unit_vector`) because cosine 0 from everything is
                // a plausible answer to every query, and one identity space is fed by one
                // embedder, so a second width is a misconfiguration rather than a prefix to
                // average over. Here it would corrupt a whole (n, n) row rather than one score.
                if (embedding.size() != dim) {
                    throw InferenceError(
                        "cross-camera clustering was handed embeddings of " +
                        std::to_string(dim) + " and " + std::to_string(embedding.size()) +
                        " dimensions; one identity space is fed by one embedder");
                }
                float sum = 0.0f;
                for (const float value : embedding) sum += value * value;
                if (!(sum > 0.0f)) {
                    throw InferenceError(
                        admitted[i].key.str() +
                        ": an all-zero track embedding has no direction, so it cannot be "
                        "normalised; at cosine 0 from everything it is a plausible answer to "
                        "every query rather than an obvious failure");
                }
                const float inverse = 1.0f / std::sqrt(sum);
                for (size_t k = 0; k < dim; ++k) unit[i * dim + k] = embedding[k] * inverse;
            }
            std::vector<float> gram(n * n, 0.0f);
            for (size_t i = 0; i < n; ++i) {
                const float* row = unit.data() + i * dim;
                // FROM `i`, and the transpose written beside it: the matrix is symmetric, so
                // the upper triangle is the whole answer and the lower one is a store.
                for (size_t j = i; j < n; ++j) {
                    const float* other = unit.data() + j * dim;
                    float dot = 0.0f;
                    for (size_t k = 0; k < dim; ++k) dot += row[k] * other[k];
                    gram[i * n + j] = dot;
                    gram[j * n + i] = dot;
                }
            }
            return gram;
        }

        class ShipvisionCluster : public ClusterTracker {
          public:
            std::map<TrackKey, int64_t> ids(
                const std::vector<ClusterObservation>& instant) override {
                std::lock_guard<std::mutex> held(lock_);
                const std::vector<ClusterObservation> admitted = gate_.filter(instant);
                // COUNTED HERE, because the gate is the only place that knows how much of an
                // instant it kept -- and "no identities" has two causes that look identical
                // from outside: nothing admitted, or nothing that matched.
                note_instant(instant.size(), admitted.size());
                const std::vector<int> labels = cluster(admitted);
                std::vector<IdentityObservation> observations;
                observations.reserve(admitted.size());
                for (const ClusterObservation& observation : admitted) {
                    observations.push_back(
                        IdentityObservation{observation.key, observation.embedding});
                }
                const std::map<TrackKey, int64_t> assigned =
                    assigner_.assign(observations, labels);

                // EVERY observation answered, gated or not -- and a gated one with `-1`, which
                // is a different fact from "this track did not exist". `records.cpp` leaves the
                // field null for it; dropping the entry instead would make an unidentifiable
                // track indistinguishable from an absent one on the screen.
                std::map<TrackKey, int64_t> out;
                for (const ClusterObservation& observation : instant) {
                    const auto found = assigned.find(observation.key);
                    out[observation.key] =
                        found == assigned.end() ? kUnidentified : found->second;
                }
                return out;
            }

            IdentitySizes sizes() const override {
                std::lock_guard<std::mutex> held(lock_);
                return assigner_.sizes();
            }

          private:
            // Nothing to compare below two tracks, so the matcher and the clusterer are
            // SKIPPED rather than called with a degenerate input. Not an optimisation: a 1x1
            // distance matrix is a valid thing to hand a clusterer and it will refuse it, and
            // one visible track is the normal state of a quiet site.
            std::vector<int> cluster(const std::vector<ClusterObservation>& admitted) const {
                if (admitted.size() < 2) return std::vector<int>(admitted.size(), 0);
                const int n = static_cast<int>(admitted.size());
                const std::vector<float> gram = gram_of(admitted);
                const std::vector<shipvision::mtmc::Observation> observations =
                    as_observations(admitted);
                const std::vector<float> distances =
                    matcher_.build(gram.data(), observations.data(), n);
                // WIDENED, because the clusterer takes doubles and the matcher answers floats.
                // One conversion here rather than a second overload on either side.
                const std::vector<double> wide(distances.begin(), distances.end());
                return clusterer_.fit_predict(wide.data(), n);
            }

            mutable std::mutex lock_;
            ObservationGate gate_;
            // DEFAULT-CONSTRUCTED, and the spatial half falls open rather than being absent:
            // `matcher.h` states that the geometric veto is "True ... when the pair cannot be
            // judged at all. Falling open on 'cannot judge' is what lets an uncalibrated camera
            // keep taking part in cross-camera tracking on appearance alone". No chain in this
            // tree carries a homography yet, so every pair is unjudgeable and this composes to
            // appearance-only BY THE LIBRARY'S OWN DESIGN rather than by omission
            // (`MTMC-HAS-NO-GROUND-PLANE`).
            shipvision::mtmc::GatedMatcher matcher_;
            shipvision::mtmc::AgglomerativeClusterer clusterer_;
            GlobalIdAssigner assigner_;
        };

        // `impl: shipvision` in the chain, which is the name the plan carries.
        const ClusterRegistrar kShipvision("shipvision", [] {
            return std::make_shared<ShipvisionCluster>();
        });

    }  // namespace

}  // namespace shipinfer::mtmc
