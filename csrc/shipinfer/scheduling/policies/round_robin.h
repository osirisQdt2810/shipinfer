// Strict rotation across instances — `policies/round_robin.py`.
#pragma once

#include <atomic>

#include "shipinfer/scheduling/policies/base.h"

namespace shipinfer {

    // Rotate through instances in order, ignoring load. The cheapest possible policy, and
    // correct only when every instance is equally fast and equally loaded — which is exactly
    // the assumption that produced the imbalance in the previous system. Kept as the baseline
    // the other policies are benchmarked against.
    class RoundRobinPolicy : public PlacementPolicy {
      public:
        Placeable* select(const std::vector<Placeable*>& candidates,
                          const PlacementRequest&) override {
            return candidates[counter_.fetch_add(1, std::memory_order_relaxed) %
                              candidates.size()];
        }
        std::string name() const override { return "round_robin"; }
        std::string describe() const override {
            return "rotate in order, load-blind (baseline)";
        }

      private:
        std::atomic<size_t> counter_{0};
    };

}  // namespace shipinfer
