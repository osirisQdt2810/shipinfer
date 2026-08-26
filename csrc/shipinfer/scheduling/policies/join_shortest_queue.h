// Send work to the instance with the fewest queued requests —
// `policies/join_shortest_queue.py`.
#pragma once

#include "shipinfer/scheduling/policies/base.h"

namespace shipinfer {

    // Optimal for a small, uniform pool. Its weakness at scale is herding: with many dispatcher
    // threads reading the same depths, they all pick the same "shortest" queue at the same
    // instant — which is exactly what PowerOfTwoChoicesPolicy exists to avoid.
    class JoinShortestQueuePolicy : public PlacementPolicy {
      public:
        Placeable* select(const std::vector<Placeable*>& candidates,
                          const PlacementRequest&) override {
            Placeable* best = candidates[0];
            size_t best_depth = best->depth();
            for (size_t i = 1; i < candidates.size(); ++i) {
                const size_t depth = candidates[i]->depth();
                if (depth < best_depth) {
                    best = candidates[i];
                    best_depth = depth;
                    if (depth == 0) break;  // cannot beat an idle instance
                }
            }
            return best;
        }
        std::string name() const override { return "join_shortest_queue"; }
        std::string describe() const override {
            return "shortest queue over all instances (optimal for small pools)";
        }
    };

}  // namespace shipinfer
