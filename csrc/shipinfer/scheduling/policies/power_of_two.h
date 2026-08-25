// Two random probes, take the shorter queue — `policies/power_of_two.py`.
#pragma once

#include <mutex>
#include <random>

#include "shipinfer/scheduling/policies/base.h"

namespace shipinfer {

    // The classic result: two random probes capture almost all the benefit of full
    // join-shortest-queue while keeping the decision O(1) in pool size and avoiding the herding
    // that makes JSQ misbehave with many concurrent dispatchers. The default fallback inside
    // LocalityAwareSpilloverPolicy and SequenceAffinityPolicy.
    class PowerOfTwoChoicesPolicy : public PlacementPolicy {
      public:
        explicit PowerOfTwoChoicesPolicy(unsigned seed = std::random_device{}()) : rng_(seed) {}

        Placeable* select(const std::vector<Placeable*>& candidates,
                          const PlacementRequest&) override {
            const size_t n = candidates.size();
            if (n == 1) return candidates[0];
            size_t i, j;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                i = std::uniform_int_distribution<size_t>(0, n - 1)(rng_);
                j = std::uniform_int_distribution<size_t>(0, n - 2)(rng_);
            }
            if (j >= i) ++j;  // sample without replacement, no rejection loop
            Placeable* a = candidates[i];
            Placeable* b = candidates[j];
            return a->depth() <= b->depth() ? a : b;
        }
        std::string name() const override { return "power_of_two"; }
        std::string describe() const override {
            return "two random probes, shorter queue wins (scales, no herding)";
        }

      private:
        std::mutex mutex_;  // the generator is not thread-safe; dispatchers are many
        std::mt19937 rng_;
    };

}  // namespace shipinfer
