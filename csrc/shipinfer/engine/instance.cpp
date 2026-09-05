#include "shipinfer/engine/instance.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "shipinfer/core/types.h"

namespace shipinfer {

    namespace {
        // Matches the Python instance: 0.1 on the newest batch.
        constexpr double kEwmaAlpha = 0.1;
    }  // namespace

    ModelInstance::ModelInstance(std::string name, std::unique_ptr<Engine> engine,
                                 BatchWindow window, size_t queue_capacity, Overflow overflow,
                                 std::function<void(Device)> bind_thread)
        : name_(std::move(name)),
          queue_capacity_(queue_capacity),
          engine_(std::move(engine)),
          window_(std::move(window)),
          bind_thread_(std::move(bind_thread)),
          // Dropped items are failed here with the same reasons the Python queue raises: a
          // queue that hands an item back with a reason is a queue that never loses one.
          queue_(
              name_, queue_capacity, overflow, 50, true,
              [this](WorkItem&& item, DropReason why) {
                  switch (why) {
                      case DropReason::Evicted:
                          // Evicted because the queue was full: the would-be depth is
                          // one past capacity, the way the Python queue reports it.
                          item.fail_with(QueueFullError(
                              "queue " + name_ + " evicted the request of the greediest camera",
                              queue_capacity_ + 1, queue_capacity_));
                          break;
                      case DropReason::Expired:
                          item.fail_with(RequestCancelledError(
                              "request deadline passed before execution"));
                          break;
                      case DropReason::Closed:
                          item.fail_with(
                              RequestCancelledError("instance " + name_ + " stopped"));
                          break;
                  }
              }) {
        if (!engine_) throw ConfigError("instance " + name_ + " has no engine");
        if (window_.max_batch_size > static_cast<size_t>(engine_->max_batch())) {
            throw ConfigError("instance " + name_ + ": the window's max_batch_size " +
                              std::to_string(window_.max_batch_size) +
                              " exceeds the engine's " + std::to_string(engine_->max_batch()));
        }
    }

    ModelInstance::~ModelInstance() {
        stop();
    }

    void ModelInstance::start() {
        if (started_once_) return;
        started_once_ = true;
        running_.store(true);
        thread_ = std::thread([this] { run(); });
    }

    bool ModelInstance::wait_ready(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(settled_mutex_);
        settled_.wait_for(lock, timeout, [this] { return settled_flag_; });
        return ready_.load();
    }

    PutStatus ModelInstance::enqueue(WorkItem&& item) {
        if (!ready_.load()) return PutStatus::Closed;
        item.request().received_ns =
            item.request().received_ns ? item.request().received_ns : monotonic_ns();
        return queue_.put(std::move(item));
    }

    void ModelInstance::stop() {
        // Ordered: stop advertising *before* failing the queue, so nothing new arrives between
        // the two and finds a closed queue with a ready instance in front of it.
        running_.store(false);
        ready_.store(false);
        queue_.close();  // fails everything still queued through the drop handler
        if (thread_.joinable()) thread_.join();
    }

    InstanceStats ModelInstance::stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        InstanceStats copy = stats_;
        copy.ewma_latency_us = ewma_latency_us_.load();
        return copy;
    }

    void ModelInstance::run() {
        try {
            if (bind_thread_) bind_thread_(engine_->device());
        } catch (...) {
            start_error_ = std::current_exception();
            queue_.close();
            std::lock_guard<std::mutex> lock(settled_mutex_);
            settled_flag_ = true;
            settled_.notify_all();
            return;
        }
        ready_.store(true);
        {
            std::lock_guard<std::mutex> lock(settled_mutex_);
            settled_flag_ = true;
        }
        settled_.notify_all();

        // The batch currently out of the queue. `close()` can only fail what is still *in* it,
        // and a batch that was already dequeued when the thread died would be stranded — its
        // futures never resolve and the caller holding one waits forever.
        std::vector<WorkItem> in_flight;
        try {
            while (running_.load()) {
                std::vector<WorkItem> items = queue_.get_batch(window_);
                if (items.empty()) {
                    if (queue_.is_closed()) break;
                    continue;
                }
                in_flight = std::move(items);
                execute_batch(in_flight);
                in_flight.clear();
            }
        } catch (...) {
            // A worker that dies must *read* as dead: readiness off before anything else.
            start_error_ = std::current_exception();
            ready_.store(false);
            running_.store(false);
        }
        ready_.store(false);
        // The batch that was out of the queue when the thread stopped, then anything still in
        // it. Both, so nothing this instance had accepted is left without an answer.
        for (WorkItem& item : in_flight) {
            item.fail_with(RequestCancelledError("instance " + name_ + " stopped"));
        }
        queue_.close();
    }

    void ModelInstance::fail_batch(std::vector<WorkItem>& items, std::exception_ptr error) {
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            ++stats_.failed_batches;
        }
        for (WorkItem& item : items) item.fail(error);
    }

    void ModelInstance::execute_batch(std::vector<WorkItem>& items) {
        const int64_t batched_ns = monotonic_ns();
        // Assemble: rows are copied into the engine's input binding at their span offsets. The
        // backend owns the copies (Triton counts them as compute_input; so does the Python
        // instance). An unassemblable batch is one bad *request*, not a broken instance: the
        // batch fails with the reason and the worker serves the next one.
        std::vector<std::pair<size_t, size_t>> spans;
        size_t offset = 0;
        try {
            const size_t width = engine_->input_row_elems();
            for (WorkItem& item : items) {
                const InferenceRequest& request = item.request();
                if (request.row_elems != width) {
                    throw BackendError("request for " + name_ + " has rows of " +
                                       std::to_string(request.row_elems) +
                                       " floats; the engine takes " + std::to_string(width));
                }
                spans.emplace_back(offset, offset + item.rows());
                offset += item.rows();
            }
            if (offset > static_cast<size_t>(engine_->max_batch())) {
                throw BackendError("assembled batch of " + std::to_string(offset) +
                                   " rows exceeds max_batch_size " +
                                   std::to_string(engine_->max_batch()) + " for " + name_);
            }
            for (size_t i = 0; i < items.size(); ++i) {
                const InferenceRequest& request = items[i].request();
                engine_->write_rows(spans[i].first, request.data, items[i].rows(),
                                    request.payload_device);
            }
        } catch (...) {
            fail_batch(items, std::current_exception());
            return;
        }

        const int64_t start_ns = monotonic_ns();
        try {
            engine_->execute(static_cast<int>(offset));
        } catch (...) {
            fail_batch(items, std::current_exception());
            return;
        }
        const int64_t end_ns = monotonic_ns();

        // Scatter: every output row returns to the request that produced it. Get it wrong and
        // two cameras' detections swap places, and nothing crashes. EVERY output, not just the
        // first: they are all `rows` long, so one span selects one request's slice of each,
        // and a segmentation engine's prototypes travel beside its detection rows.
        const size_t out_count = engine_->outputs();
        const int64_t completed_ns = monotonic_ns();
        for (size_t i = 0; i < items.size(); ++i) {
            WorkItem& item = items[i];
            InferenceResponse response;
            response.model_name = name_;
            response.tag = item.request().tag;
            response.rows = spans[i].second - spans[i].first;
            response.outputs.reserve(out_count);
            for (size_t o = 0; o < out_count; ++o) {
                const size_t width = engine_->output_row_elems(o);
                const float* out = engine_->output(o);
                OutputTensor tensor;
                tensor.name = engine_->output_name(o);
                tensor.row_elems = width;
                tensor.dims = engine_->output_dims(o);
                tensor.data.assign(out + spans[i].first * width, out + spans[i].second * width);
                response.outputs.push_back(std::move(tensor));
            }
            response.executed_on = engine_->device();
            response.timings.received_ns = item.request().received_ns;
            response.timings.queued_ns = item.enqueued_ns();
            response.timings.batched_ns = batched_ns;
            response.timings.compute_start_ns = start_ns;
            response.timings.compute_end_ns = end_ns;
            response.timings.completed_ns = completed_ns;
            item.complete(std::move(response));
        }
        const double latency_us = (end_ns - start_ns) / 1000.0;
        const double previous = ewma_latency_us_.load();
        ewma_latency_us_.store(
            previous == 0.0 ? latency_us : previous + kEwmaAlpha * (latency_us - previous));
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++stats_.batches;
        stats_.rows += offset;
        stats_.requests += items.size();
    }

}  // namespace shipinfer
