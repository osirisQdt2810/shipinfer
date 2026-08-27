#include "shipinfer/engine/model.h"

#include <algorithm>
#include <utility>

#include "shipinfer/core/types.h"

namespace shipinfer {

    Model::Model(std::string name, std::vector<std::unique_ptr<ModelInstance>> instances,
                 std::unique_ptr<PlacementPolicy> policy)
        : name_(std::move(name)), instances_(std::move(instances)) {
        std::vector<Placeable*> placeables;
        for (auto& instance : instances_) placeables.push_back(instance.get());
        dispatcher_ = std::make_unique<Dispatcher>(name_, placeables, std::move(policy));
    }

    Model::~Model() {
        stop();
    }

    void Model::start(std::chrono::milliseconds ready_timeout) {
        for (auto& instance : instances_) instance->start();
        for (auto& instance : instances_) {
            if (!instance->wait_ready(ready_timeout)) {
                std::string why = "did not become ready in time";
                if (instance->start_error()) {
                    try {
                        std::rethrow_exception(instance->start_error());
                    } catch (const std::exception& error) {
                        why = error.what();
                    }
                }
                throw BackendError("model " + name_ + ": instance " + instance->name() + " " +
                                   why);
            }
        }
    }

    void Model::stop() {
        for (auto& instance : instances_) instance->stop();
    }

    bool Model::is_ready() const {
        for (const auto& instance : instances_) {
            if (instance->is_ready()) return true;
        }
        return false;
    }

    int Model::max_batch() const {
        int smallest = 0;
        for (const auto& instance : instances_) {
            smallest = smallest == 0 ? instance->max_batch()
                                     : std::min(smallest, instance->max_batch());
        }
        return smallest;
    }

    size_t Model::total_depth() const {
        size_t total = 0;
        for (const auto& instance : instances_) total += instance->depth();
        return total;
    }

    std::future<InferenceResponse> Model::infer(InferenceRequest request) {
        WorkItem item(std::move(request));
        std::future<InferenceResponse> future = item.future();
        PlacementRequest placement{item.request().resident_device,
                                   item.request().tag.camera_id};
        try {
            dispatcher_->dispatch(placement, [&](Placeable* chosen) {
                // `put(T&&)` takes the item only on acceptance, so a refusal leaves it here for
                // the dispatcher's next candidate.
                return static_cast<ModelInstance*>(chosen)->enqueue(std::move(item));
            });
        } catch (...) {
            item.fail(std::current_exception());
        }
        return future;
    }

}  // namespace shipinfer
