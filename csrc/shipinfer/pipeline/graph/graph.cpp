#include "shipinfer/pipeline/graph/graph.h"

#include <algorithm>
#include <cmath>

#include "shipinfer/core/platform.h"
#include "shipinfer/runtime/ops.h"

namespace shipinfer {
    namespace {

        // The detector's output is (300, 6): x1, y1, x2, y2, score, class — TensorRT's EfficientNMS
        // layout, which is what these plans are built with.
        constexpr int kDetStride = 6;

    }  // namespace

    ModelPool::ModelPool(std::string name, const std::string& plan, const std::vector<int>& devices,
                         int per_device)
        : name_(std::move(name)) {
        if (devices.empty()) throw ConfigError(name_ + " has no devices");
        if (per_device < 1) {
            throw ConfigError(name_ + " needs at least one instance per device, got " +
                              std::to_string(per_device));
        }
        for (int device : devices) {
            // One engine per device, shared by that device's instances: the weights are paid for
            // once per GPU rather than once per instance.
            auto engine = TrtEngine::load(plan, device);
            engines_.push_back(engine);
            for (int i = 0; i < per_device; ++i) {
                by_device_[device].push_back(instances_.size());
                instances_.push_back(std::make_unique<TrtInstance>(engine, device));
                locks_.push_back(std::make_unique<std::mutex>());
            }
        }
    }

    ModelPool::Lease ModelPool::lease(int device) {
        // Only this device's instances are candidates. Round-robin among them, then spin to the
        // first free one: with equal instances the next free is as good as the nominated one, and
        // waiting for a specific instance is how a queue forms behind a slow device.
        auto it = by_device_.find(device);
        if (it == by_device_.end() || it->second.empty()) {
            throw ConfigError(name_ + " has no instance on device " + std::to_string(device) +
                              "; a frame cannot be executed on a device its pixels are not on");
        }
        const auto& slots = it->second;
        // Counted around the whole acquisition, including the spin: a worker spinning for a free
        // instance is a worker waiting, and pretending otherwise is what hid the pegged pool.
        waiting_.fetch_add(1);
        struct Leaving {
            std::atomic<long long>* counter;
            ~Leaving() { counter->fetch_sub(1); }
        } leaving{&waiting_};

        for (size_t attempt = 0; attempt < slots.size() * 4; ++attempt) {
            const size_t slot = slots[next_.fetch_add(1) % slots.size()];
            if (locks_[slot]->try_lock()) {
                busy_.fetch_add(1);
                return Lease{instances_[slot].get(), slot};
            }
        }
        const size_t slot = slots[next_.fetch_add(1) % slots.size()];
        locks_[slot]->lock();
        busy_.fetch_add(1);
        return Lease{instances_[slot].get(), slot};
    }

    void ModelPool::release(const Lease& lease) {
        busy_.fetch_sub(1);
        locks_[lease.slot]->unlock();
    }

    std::map<int, uint64_t> ModelPool::per_device() const {
        std::map<int, uint64_t> counts;
        for (const auto& instance : instances_) {
            counts[instance->device()] += instance->executed();
        }
        return counts;
    }

    uint64_t ModelPool::executed() const {
        uint64_t total = 0;
        for (const auto& instance : instances_) total += instance->executed();
        return total;
    }

    PipelineGraph::PipelineGraph(const GraphConfig& config) : config_(config) {
        detector_ = std::make_unique<ModelPool>("ship_detector", config.detector_plan,
                                                config.devices, config.detector_instances);
        if (!config.segmenter_plan.empty()) {
            segmenter_ = std::make_unique<ModelPool>("ship_segmenter", config.segmenter_plan,
                                                     config.devices, config.segmenter_instances);
        }
        if (!config.embedder_plan.empty()) {
            embedder_ = std::make_unique<ModelPool>("person_embedder", config.embedder_plan,
                                                    config.devices, config.embedder_instances);
        }
        if (!config.ship_embedder_plan.empty()) {
            ship_embedder_ =
                std::make_unique<ModelPool>("ship_embedder", config.ship_embedder_plan,
                                            config.devices, config.ship_embedder_instances);
        }
    }

    PipelineGraph::~PipelineGraph() = default;

    std::vector<std::string> PipelineGraph::stage_names() const {
        std::vector<std::string> names{"detect", "crop"};
        if (segmenter_) names.push_back("ship_segmenter");
        if (embedder_) names.push_back("person_embedder");
        if (ship_embedder_) names.push_back("ship_embedder");
        return names;
    }

    size_t PipelineGraph::execute(std::vector<Work>& batch, int device,
                                  FrameCollector& collector) {
        if (batch.empty()) return 0;
        GPU_CHECK(gpuSetDevice(device));

        // -- detect, once, for the whole batch ------------------------------------------------
        // The plan is static, so the batch handed to `execute` is exactly `max_batch` rows. A
        // short batch is padded: the padding rows are letterboxed from the last real frame rather
        // than left uninitialised, because uninitialised device memory can contain NaNs and a NaN
        // through a detector produces warnings on every layer.
        const int rows = detector_->max_batch();
        // The caller drains a detector-sized batch, so this is an equality in practice. It is
        // asserted rather than assumed because a frame is one detector row *by convention*
        // (`FrameWork::rows()` returns 1), and anything past `rows` would be dropped here
        // without a word.
        if (batch.size() > static_cast<size_t>(rows)) {
            throw ConfigError("the graph was handed " + std::to_string(batch.size()) +
                              " frames for a detector built at batch " + std::to_string(rows) +
                              "; the drain must not exceed it");
        }
        const size_t real = std::min(batch.size(), static_cast<size_t>(rows));
        std::vector<LetterboxMap> maps(real);

        auto lease = detector_->lease(device);
        float* input = static_cast<float*>(lease.instance->input());
        const size_t stride = static_cast<size_t>(config_.detect_size) * config_.detect_size * 3;
        try {
            for (size_t i = 0; i < static_cast<size_t>(rows); ++i) {
                const Work& work = batch[std::min(i, real - 1)];
                const LetterboxMap map =
                    letterbox_into(work.image_device, work.state->height(), work.state->width(),
                                   input + i * stride, config_.detect_size, config_.detect_size,
                                   /*swap_rb=*/true, /*pad_value=*/0.5f, lease.instance->stream());
                if (i < real) maps[i] = map;
            }
            // No synchronisation. The kernels and `enqueueV3` are on the same stream, so the
            // stream orders them — and `execute` syncs that one stream at the end, which is what
            // makes the outputs readable. The first version launched on the default stream and
            // then called `gpuDeviceSynchronize`, which waits for *every* context on the device:
            // at four workers per GPU that serialised all of them behind each other.
            lease.instance->execute(rows);
        } catch (...) {
            detector_->release(lease);
            throw;
        }

        const float* out = lease.instance->output();
        const size_t per_row = lease.instance->output_rows();
        for (size_t i = 0; i < real; ++i) {
            std::vector<Detection> detections;
            const float* frame_out = out + i * per_row;
            const size_t candidates = per_row / kDetStride;
            for (size_t d = 0;
                 d < candidates && detections.size() < static_cast<size_t>(config_.max_objects);
                 ++d) {
                const float* row = frame_out + d * kDetStride;
                if (row[4] < config_.score_threshold) continue;
                Detection det;
                // Model space back to original pixels. Cropping in letterboxed coordinates is
                // where the off-by-a-pad-bar bugs live.
                det.x1 = (row[0] - static_cast<float>(maps[i].pad_x)) / maps[i].scale;
                det.y1 = (row[1] - static_cast<float>(maps[i].pad_y)) / maps[i].scale;
                det.x2 = (row[2] - static_cast<float>(maps[i].pad_x)) / maps[i].scale;
                det.y2 = (row[3] - static_cast<float>(maps[i].pad_y)) / maps[i].scale;
                det.score = row[4];
                det.class_id = static_cast<int>(row[5]);
                detections.push_back(det);
            }
            batch[i].state->set_detections(std::move(detections));
        }
        detector_->release(lease);

        for (size_t i = 0; i < real; ++i) {
            collector.deliver(batch[i].state->tag(), "detect");
            collector.deliver(batch[i].state->tag(), "crop");
        }

        // -- the per-object branches, per frame ----------------------------------------------
        //
        // Two passes, and the order is the fix. First every frame declares what will run, then
        // every frame runs it — with its own try/catch. The first version did both in one loop
        // and let a throw escape `execute`: the frames *after* the one that failed had never
        // reached `expect()`, so their expected set was still `{detect, crop}`, both delivered,
        // and the caller's `seal` reported seven batch-mates **Complete with missing=[]** while
        // their embedder had never run. A frame's failure is now its own: its unrun stages are
        // in `missing`, the others are unaffected, and the count comes back to the caller for
        // `frames_failed` — a batch-wide throw is kept for the detector, where it is true.
        struct Branches {
            std::vector<float> person_boxes, ship_boxes;
            std::vector<int> person_index, ship_index;
        };
        std::vector<Branches> branches(real);
        for (size_t i = 0; i < real; ++i) {
            FrameState& state = *batch[i].state;
            Branches& b = branches[i];
            for (const auto& det : state.detections()) {
                std::vector<float>* boxes = nullptr;
                std::vector<int>* index = nullptr;
                if (det.class_id == config_.person_class) {
                    boxes = &b.person_boxes;
                    index = &b.person_index;
                } else if (det.class_id == config_.ship_class) {
                    boxes = &b.ship_boxes;
                    index = &b.ship_index;
                }
                if (boxes == nullptr) continue;
                boxes->push_back(det.x1);
                boxes->push_back(det.y1);
                boxes->push_back(det.x2);
                boxes->push_back(det.y2);
                index->push_back(det.index);
            }
            // Declare what will actually run, *then* run it. A frame with no people does not
            // run the embedder, and that is a **skip** -- it has to be distinguishable in the
            // event from a stage that was expected and failed. Before this was wired, every
            // ship-only frame sealed Incomplete with `missing=["person_embedder"]` and every
            // person-only frame with the two ship stages: skipped, failed and timed out
            // collapsed into one.
            std::vector<std::string> will_run;
            if (embedder_ != nullptr && !b.person_index.empty()) {
                will_run.push_back("person_embedder");
            }
            if (segmenter_ != nullptr && !b.ship_index.empty()) {
                will_run.push_back("ship_segmenter");
            }
            if (ship_embedder_ != nullptr && !b.ship_index.empty()) {
                will_run.push_back("ship_embedder");
            }
            collector.expect(state.tag(), will_run);
        }

        size_t failed = 0;
        for (size_t i = 0; i < real; ++i) {
            FrameState& state = *batch[i].state;
            const Branches& b = branches[i];
            if (b.person_index.empty() && b.ship_index.empty()) continue;
            try {
                run_objects(embedder_.get(), state, batch[i].image_device, device, b.person_boxes,
                            b.person_index, config_.crop_h, config_.crop_w, "person_embedder",
                            collector);
                run_objects(segmenter_.get(), state, batch[i].image_device, device, b.ship_boxes,
                            b.ship_index, config_.detect_size, config_.detect_size,
                            "ship_segmenter", collector);
                run_objects(ship_embedder_.get(), state, batch[i].image_device, device,
                            b.ship_boxes, b.ship_index, config_.crop_h, config_.crop_w,
                            "ship_embedder", collector);
            } catch (const std::exception& error) {
                // This frame's stages that did not run stay undelivered, so `seal` reports it
                // Incomplete and names them. The batch-mates carry on.
                ++failed;
                if (on_frame_error_) on_frame_error_(state.tag(), error.what());
            }
        }
        return failed;
    }

    void PipelineGraph::run_objects(ModelPool* pool, FrameState& state, const uint8_t* image_device,
                                    int device, const std::vector<float>& boxes,
                                    const std::vector<int>& indices, int crop_h, int crop_w,
                                    const char* stage, FrameCollector& collector) {
        if (pool == nullptr || indices.empty()) return;
        const int limit = pool->max_batch();
        // Chunked to the engine's own batch, and padded up to it because these plans are static
        // too. Submitting a whole frame's crops as one request is what lost every crop in a
        // 25-person frame against a plan built at 16 — and 10-20 people per frame is the *normal*
        // case at this sizing, not an edge one.
        for (size_t start = 0; start < indices.size(); start += static_cast<size_t>(limit)) {
            const int count = static_cast<int>(std::min<size_t>(limit, indices.size() - start));
            auto lease = pool->lease(device);
            try {
                // The box list is padded to `limit` by repeating the last box, so the kernel
                // writes every row the static plan expects.
                std::vector<float> padded(static_cast<size_t>(limit) * 4);
                for (int i = 0; i < limit; ++i) {
                    const size_t source = start + static_cast<size_t>(std::min(i, count - 1));
                    std::copy_n(boxes.begin() + static_cast<long>(source * 4), 4,
                                padded.begin() + static_cast<long>(i) * 4);
                }
                // The instance's own scratch, not a fresh `gpuMalloc` per chunk: allocation
                // serialises on the driver and this runs once per object-batch per frame.
                void* box_device = lease.instance->scratch(padded.size() * sizeof(float));
                GPU_CHECK(gpuMemcpyAsync(box_device, padded.data(), padded.size() * sizeof(float),
                                         gpuMemcpyHostToDevice, lease.instance->stream()));
                crop_resize_into(image_device, state.height(), state.width(),
                                 static_cast<const float*>(box_device), limit,
                                 static_cast<float*>(lease.instance->input()), crop_h, crop_w,
                                 /*swap_rb=*/true, lease.instance->stream());
                lease.instance->execute(limit);
            } catch (...) {
                pool->release(lease);
                throw;
            }
            ObjectBatch out;
            out.name = std::string(stage) + "_out";
            out.width = static_cast<int>(lease.instance->output_rows());
            // Only the real rows are kept; the padding was there to satisfy the plan, not to be
            // reported as a result.
            out.data.assign(lease.instance->output(),
                            lease.instance->output() + static_cast<size_t>(count) * out.width);
            for (int i = 0; i < count; ++i) {
                out.object_indices.push_back(indices[start + static_cast<size_t>(i)]);
            }
            pool->release(lease);
            state.attach(std::move(out));
        }
        collector.deliver(state.tag(), stage);
    }

}  // namespace shipinfer
