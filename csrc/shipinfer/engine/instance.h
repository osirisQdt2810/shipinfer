// One model instance: a backend copy pinned to one device, fed by its own bounded queue,
// drained by its own thread — `engine/instance.py`, seam for seam.
//
// The thread binds itself to its device once at start-up and never touches another (ADR-002).
// It takes a batch from the queue under the model's window, assembles it into the engine's
// input binding (the backend owns the copies), runs it, and scatters the output rows back to
// the futures the callers hold. A dead worker reads as dead: `is_ready()` drops before the
// queue is failed, so nothing new arrives between the two and finds a closed queue behind a
// ready instance. Everything it had accepted — the batch out of the queue and everything still
// in it — gets an answer, even if the answer is "cancelled".
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/backends/engine_api.h"
#include "shipinfer/engine/request.h"
#include "shipinfer/scheduling/policies/base.h"
#include "shipinfer/scheduling/queues/base.h"
#include "shipinfer/scheduling/queues/fair.h"

namespace shipinfer {

    struct InstanceStats {
        uint64_t batches = 0;
        uint64_t rows = 0;
        uint64_t requests = 0;
        uint64_t failed_batches = 0;
        double ewma_latency_us = 0.0;
    };

    class ModelInstance : public Placeable {
      public:
        // `bind_thread` runs first on the worker thread: `gpuSetDevice` for a real engine,
        // nothing for a test engine. Injected so the class knows no vendor API.
        ModelInstance(std::string name, std::unique_ptr<Engine> engine, BatchWindow window,
                      size_t queue_capacity, Overflow overflow = Overflow::Reject,
                      std::function<void(Device)> bind_thread = {});
        ~ModelInstance() override;
        ModelInstance(const ModelInstance&) = delete;
        ModelInstance& operator=(const ModelInstance&) = delete;

        const std::string& name() const { return name_; }

        // -- Placeable
        // -------------------------------------------------------------------------
        Device device() const override { return engine_->device(); }
        size_t depth() const override { return queue_.depth(); }
        double ewma_latency_us() const override { return ewma_latency_us_.load(); }
        bool is_ready() const override { return ready_.load(); }
        size_t capacity() const override { return queue_capacity_; }

        // -- lifecycle
        // -------------------------------------------------------------------------
        void start();
        // True once the worker is bound and serving, false on timeout or a failed start.
        bool wait_ready(std::chrono::milliseconds timeout);
        // Stop advertising, fail everything accepted and not yet answered, join the thread.
        void stop();
        std::exception_ptr start_error() const { return start_error_; }

        // -- the producer side
        // ----------------------------------------------------------------- Takes the item only
        // on acceptance; a refusal leaves it with the caller for the spill.
        PutStatus enqueue(WorkItem&& item);

        InstanceStats stats() const;
        int max_batch() const { return engine_->max_batch(); }

      private:
        void run();
        void execute_batch(std::vector<WorkItem>& items);
        void fail_batch(std::vector<WorkItem>& items, std::exception_ptr error);

        std::string name_;
        size_t queue_capacity_ = 0;
        std::unique_ptr<Engine> engine_;
        BatchWindow window_;
        std::function<void(Device)> bind_thread_;
        FairPriorityQueue<WorkItem> queue_;
        std::thread thread_;
        std::atomic<bool> running_{false};
        std::atomic<bool> ready_{false};
        std::atomic<double> ewma_latency_us_{0.0};
        std::exception_ptr start_error_;
        std::mutex settled_mutex_;
        std::condition_variable settled_;
        bool started_once_ = false;
        bool settled_flag_ = false;
        mutable std::mutex stats_mutex_;
        InstanceStats stats_;
    };

}  // namespace shipinfer
