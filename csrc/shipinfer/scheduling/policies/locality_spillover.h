// Keep work on the GPU that already holds its data — until that stops being fastest.
// `policies/locality_spillover.py`.
#pragma once

#include <memory>

#include "shipinfer/scheduling/policies/base.h"
#include "shipinfer/scheduling/policies/power_of_two.h"

namespace shipinfer {

    // The project default, and the reason is bandwidth. A 1080p frame is ~6 MB, so moving it to
    // another GPU costs a D2H plus an H2D; a crop is a few KB and can go anywhere. So: if the
    // request is already resident on a candidate GPU *and* that GPU's queue is at or below
    // `spill_threshold`, run it there with no copy; otherwise defer to `fallback` and accept
    // the copy, because a queue that is genuinely backing up costs more than one transfer.
    // `spill_threshold=0` degenerates to pure load balancing; a very large value to pinning.
    class LocalityAwareSpilloverPolicy : public PlacementPolicy {
      public:
        explicit LocalityAwareSpilloverPolicy(
            int spill_threshold = 4, std::unique_ptr<PlacementPolicy> fallback = nullptr)
            : spill_threshold_(spill_threshold),
              fallback_(fallback ? std::move(fallback)
                                 : std::make_unique<PowerOfTwoChoicesPolicy>()) {
            if (spill_threshold < 0) {
                throw std::invalid_argument("locality_spillover: spill_threshold must be >= 0");
            }
        }

        Placeable* select(const std::vector<Placeable*>& candidates,
                          const PlacementRequest& request) override {
            if (request.resident_device.has_value() && request.resident_device->is_cuda()) {
                Placeable* local_best = nullptr;
                for (Placeable* candidate : candidates) {
                    if (candidate->device() == *request.resident_device &&
                        (local_best == nullptr || candidate->depth() < local_best->depth())) {
                        local_best = candidate;
                    }
                }
                if (local_best != nullptr &&
                    local_best->depth() <= static_cast<size_t>(spill_threshold_)) {
                    return local_best;
                }
            }
            return fallback_->select(candidates, request);
        }
        std::string name() const override { return "locality_spillover"; }
        std::string describe() const override {
            return "stay on the resident GPU while depth <= " +
                   std::to_string(spill_threshold_) + ", else " + fallback_->describe();
        }
        int spill_threshold() const { return spill_threshold_; }

      private:
        int spill_threshold_;
        std::unique_ptr<PlacementPolicy> fallback_;
    };

}  // namespace shipinfer
