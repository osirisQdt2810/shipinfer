// A servable model: its instances, its dispatcher, its window — `engine/model.py`.
#pragma once

#include <future>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/engine/instance.h"
#include "shipinfer/scheduling/dispatcher.h"
#include "shipinfer/scheduling/policies/base.h"

namespace shipinfer {

    class Model {
      public:
        Model(std::string name, std::vector<std::unique_ptr<ModelInstance>> instances,
              std::unique_ptr<PlacementPolicy> policy);
        ~Model();

        const std::string& name() const { return name_; }
        const std::vector<std::unique_ptr<ModelInstance>>& instances() const {
            return instances_;
        }
        const Dispatcher& dispatcher() const { return *dispatcher_; }

        void start(std::chrono::milliseconds ready_timeout = std::chrono::milliseconds(120000));
        void stop();
        bool is_ready() const;
        size_t total_depth() const;
        // The smallest engine batch across the instances — what a per-object stage chunks to.
        int max_batch() const;

        // Place the request and return the future its caller waits on. A request nothing will
        // take — every ready instance refused, or nothing is ready — comes back as a future
        // that already holds the error, so the caller has one path to wait on (the Python
        // `infer` raises instead; a future carrying the exception is the same fact in this
        // language).
        std::future<InferenceResponse> infer(InferenceRequest request);

      private:
        std::string name_;
        std::vector<std::unique_ptr<ModelInstance>> instances_;
        std::unique_ptr<Dispatcher> dispatcher_;
    };

}  // namespace shipinfer
