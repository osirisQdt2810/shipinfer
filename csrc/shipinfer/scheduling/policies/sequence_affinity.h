// Sticky routing for stateful models — Triton's sequence batcher, as a policy.
// `policies/sequence_affinity.py`.
#pragma once

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "shipinfer/scheduling/policies/base.h"
#include "shipinfer/scheduling/policies/power_of_two.h"

namespace shipinfer {

    // Route every request of a sequence to the same instance. Some models carry state between
    // calls — a tracker's Kalman filter, a recurrent embedder — and are only correct if
    // consecutive requests for one sequence land on the instance that holds that state. The
    // sequence key is the camera id. Affinity is sticky, not permanent: when the affine
    // instance dies or leaves the ready set, the sequence is re-pinned through `fallback` —
    // refusing to re-pin would turn one dead GPU into permanently dropped cameras.
    class SequenceAffinityPolicy : public PlacementPolicy {
      public:
        explicit SequenceAffinityPolicy(std::unique_ptr<PlacementPolicy> fallback = nullptr,
                                        size_t max_sequences = 4096)
            : fallback_(fallback ? std::move(fallback)
                                 : std::make_unique<PowerOfTwoChoicesPolicy>()),
              max_sequences_(max_sequences) {}

        Placeable* select(const std::vector<Placeable*>& candidates,
                          const PlacementRequest& request) override {
            const std::string& key = request.camera_id;
            if (key.empty()) return fallback_->select(candidates, request);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto pinned = assigned_.find(key);
                if (pinned != assigned_.end() && pinned->second->is_ready() &&
                    std::find(candidates.begin(), candidates.end(), pinned->second) !=
                        candidates.end()) {
                    return pinned->second;
                }
            }
            Placeable* chosen = fallback_->select(candidates, request);
            std::lock_guard<std::mutex> lock(mutex_);
            if (assigned_.size() >= max_sequences_) assigned_.clear();
            assigned_[key] = chosen;
            return chosen;
        }
        // Release one sequence's pin — called when a camera disconnects.
        void forget(const std::string& key) {
            std::lock_guard<std::mutex> lock(mutex_);
            assigned_.erase(key);
        }
        std::string name() const override { return "sequence_affinity"; }
        std::string describe() const override {
            return "pin each camera to one instance (fallback: " + fallback_->describe() + ")";
        }

      private:
        std::unique_ptr<PlacementPolicy> fallback_;
        size_t max_sequences_;
        std::mutex mutex_;
        std::unordered_map<std::string, Placeable*> assigned_;
    };

}  // namespace shipinfer
