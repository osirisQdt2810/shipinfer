// The dispatcher: pick an instance, enqueue, and spill when the first choice is full.
// `scheduling/dispatcher.py`, seam for seam.
//
// It is intentionally small. All the intelligence is in the policy it holds; all the state is
// in the queues it enqueues into. What it owns is the *retry* behaviour, which is the part that
// turns a policy decision into a delivery guarantee: choose, try to enqueue, and on a full
// queue try the next-shortest instance once each before refusing. Without that loop, a policy
// that guesses slightly wrong under a burst turns a transient full queue into a dropped frame.
#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/scheduling/policies/base.h"
#include "shipinfer/scheduling/queues/base.h"

namespace shipinfer {

    struct DispatchResult {
        Placeable* instance = nullptr;
        int attempts = 0;
        bool spilled = false;
    };

    // Routes requests for **one model** across that model's instances.
    class Dispatcher {
      public:
        using OnSpill = std::function<void(Placeable* wanted, Placeable* actual)>;

        Dispatcher(std::string model_name, std::vector<Placeable*> instances,
                   std::unique_ptr<PlacementPolicy> policy, OnSpill on_spill = {})
            : model_name_(std::move(model_name)),
              instances_(std::move(instances)),
              policy_(std::move(policy)),
              on_spill_(std::move(on_spill)) {
            if (instances_.empty()) {
                throw ServerStateError("model '" + model_name_ +
                                       "' has no instances to dispatch to");
            }
            if (!policy_) throw ServerStateError("model '" + model_name_ + "' has no policy");
        }

        const PlacementPolicy& policy() const { return *policy_; }
        const std::vector<Placeable*>& instances() const { return instances_; }

        std::vector<Placeable*> ready_instances() const {
            std::vector<Placeable*> ready;
            for (Placeable* instance : instances_) {
                if (instance->is_ready()) ready.push_back(instance);
            }
            return ready;
        }

        // Ask the policy for one instance. Does not enqueue. Throws ServerStateError when
        // every instance is unavailable (loading, or failed).
        Placeable* select(const PlacementRequest& request) const {
            const std::vector<Placeable*> ready = ready_instances();
            if (ready.empty()) {
                throw ServerStateError("model '" + model_name_ + "' has no ready instance (" +
                                       std::to_string(instances_.size()) + " configured)");
            }
            return policy_->select(ready, request);
        }

        // Place a request on an instance, spilling to the next-shortest queue if needed.
        // `enqueue(instance)` hands the item to a chosen instance and answers with `PutStatus`;
        // it is injected rather than called as a method so this class keeps depending only on
        // the narrow `Placeable` contract, and so tests can drive it without a queue. Throws
        // QueueFullError only after every ready instance has refused — the honest signal that
        // the *pool* is saturated rather than one GPU — and ServerStateError when nothing is
        // ready.
        template <typename Enqueue>
        DispatchResult dispatch(const PlacementRequest& request, Enqueue enqueue) {
            const std::vector<Placeable*> ready = ready_instances();
            if (ready.empty()) {
                throw ServerStateError("model '" + model_name_ + "' has no ready instance");
            }
            Placeable* first = policy_->select(ready, request);
            if (enqueue(first) == PutStatus::Accepted) return DispatchResult{first, 1, false};
            // Spill: try the remaining instances shortest-queue-first. Sorting is acceptable
            // here because this path only runs when a queue is already full — rarely, and
            // never in the steady state.
            std::vector<Placeable*> remaining;
            for (Placeable* instance : ready) {
                if (instance != first) remaining.push_back(instance);
            }
            std::stable_sort(
                remaining.begin(), remaining.end(),
                [](Placeable* a, Placeable* b) { return a->depth() < b->depth(); });
            int attempt = 2;
            for (Placeable* candidate : remaining) {
                if (enqueue(candidate) == PutStatus::Accepted) {
                    if (on_spill_) on_spill_(first, candidate);
                    return DispatchResult{candidate, attempt, true};
                }
                ++attempt;
            }
            throw QueueFullError("model '" + model_name_ + "': every ready instance refused (" +
                                 std::to_string(ready.size()) +
                                 " tried); the pool is saturated");
        }

      private:
        std::string model_name_;
        std::vector<Placeable*> instances_;
        std::unique_ptr<PlacementPolicy> policy_;
        OnSpill on_spill_;
    };

}  // namespace shipinfer
