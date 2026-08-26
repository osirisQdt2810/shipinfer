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
        // Unseeded (the default): one generator per thread, no lock — as the fallback of
        // `locality_spillover` this runs on every spill from 48 dispatchers, and a process-wide
        // mutex there is a contended lock at design load. Seeded: the deterministic shared
        // generator the tests rely on, behind the mutex.
        PowerOfTwoChoicesPolicy() = default;
        explicit PowerOfTwoChoicesPolicy(unsigned seed) : seeded_(true), rng_(seed) {}

        Placeable* select(const std::vector<Placeable*>& candidates,
                          const PlacementRequest&) override {
            const size_t n = candidates.size();
            if (n == 1) return candidates[0];
            size_t i, j;
            if (seeded_) {
                std::lock_guard<std::mutex> lock(mutex_);
                i = std::uniform_int_distribution<size_t>(0, n - 1)(rng_);
                j = std::uniform_int_distribution<size_t>(0, n - 2)(rng_);
            } else {
                thread_local std::mt19937 local_rng{std::random_device{}()};
                i = std::uniform_int_distribution<size_t>(0, n - 1)(local_rng);
                j = std::uniform_int_distribution<size_t>(0, n - 2)(local_rng);
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
        bool seeded_ = false;
        std::mutex mutex_;  // the seeded generator is shared and not thread-safe
        std::mt19937 rng_{std::random_device{}()};
    };

}  // namespace shipinfer
