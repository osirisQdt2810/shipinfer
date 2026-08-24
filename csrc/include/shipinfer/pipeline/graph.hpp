// The perception DAG and the worker pool that drives it.
//
// detect -> crop -> {segment, embed} is the shape, and the one thing that makes it a DAG rather
// than a chain is that the crop stage *changes cardinality*: one frame becomes N objects. Every
// other stage preserves it, and the graph refuses at start-up if a per-frame stage would be
// handed a per-object batch.
//
// A worker drives one frame through the whole graph and blocks on each model, so the number of
// frames in flight is the number of workers. That is a real difference from the baseline, whose
// inference threads each assemble their own batch from a shared queue, and it is why the worker
// count is a knob rather than a constant.
#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/pipeline/collector.hpp"
#include "shipinfer/pipeline/state.hpp"
#include "shipinfer/sched/fair_queue.hpp"
#include "shipinfer/trt/engine.hpp"

namespace shipinfer {

// A model plus its instances, one per (device, stream). Round-robin placement, because with
// equal instances on equal devices the cheapest correct policy is the best one — the fancier
// policies in `scheduling/policies/` exist for uneven fleets and are a Python-side concern
// until this plane needs them.
class ModelPool {
  public:
    ModelPool(std::string name, const std::string& plan, const std::vector<int>& devices,
              int per_device);

    // Blocks until an instance **on `device`** is free. Device-affine on purpose: a frame's
    // pixels live on the GPU they were copied to, and an execution context's device memory
    // belongs to the device that allocated it. Leasing across devices is a cross-device access
    // — ADR-002 forbids it, and it shows up as `an illegal memory access was encountered`
    // several frames later rather than at the call that caused it.
    struct Lease {
        TrtInstance* instance = nullptr;
        size_t slot = 0;
    };
    Lease lease(int device);
    void release(const Lease& lease);

    const std::string& name() const { return name_; }
    int max_batch() const { return instances_.front()->max_batch(); }
    size_t size() const { return instances_.size(); }
    TrtInstance& at(size_t index) { return *instances_[index]; }
    // model -> device -> executed. The per-device breakdown the balancing evidence needs.
    std::map<int, uint64_t> per_device() const;
    uint64_t executed() const;
    long long busy() const { return busy_.load(); }
    // Workers blocked in `lease` — the analogue of a model's queue depth, and the number the
    // occupancy log must carry.
    //
    // `busy()` alone cannot show pressure: this design has no queue in front of a pool, so a
    // fully committed pool reads as a *flat* `busy == size` and the analysis scores it
    // SUSTAINED. The first measurement had `ship_segmenter` sitting at exactly 4 with exactly
    // 4 instances — pegged, and invisible. The waiting count is where the queue actually is.
    long long waiting() const { return waiting_.load(); }

  private:
    std::string name_;
    std::vector<std::shared_ptr<TrtEngine>> engines_;
    std::vector<std::unique_ptr<TrtInstance>> instances_;
    std::vector<std::unique_ptr<std::mutex>> locks_;
    std::map<int, std::vector<size_t>> by_device_;
    std::atomic<size_t> next_{0};
    std::atomic<long long> busy_{0};
    std::atomic<long long> waiting_{0};
};

struct GraphConfig {
    std::string detector_plan;
    std::string segmenter_plan;
    std::string embedder_plan;
    //: Ships get their own embedder pool even though the plan is the same artefact as the
    //: person one. Separate pools, not a shared one: the Python repository declares
    //: `ship_embedder` and `person_embedder` as two models with their own instance counts and
    //: their own queues, and collapsing them here would change the concurrency the comparison
    //: is measuring.
    std::string ship_embedder_plan;
    std::vector<int> devices;
    int detector_instances = 2;
    int segmenter_instances = 1;
    int embedder_instances = 2;
    int ship_embedder_instances = 1;
    int detect_size = 640;
    int crop_h = 256;
    int crop_w = 128;
    float score_threshold = 0.25f;
    int max_objects = 64;
    // COCO: person is 0 and boat is 8. The wrong mapping crops every person into the ship
    // branch with every shape check still passing, which is exactly what happened once.
    int person_class = 0;
    int ship_class = 8;
};

class PerceptionGraph {
  public:
    explicit PerceptionGraph(const GraphConfig& config);
    ~PerceptionGraph();

    // Runs a **batch** of frames to completion on one device.
    //
    // Batched because these plans are *static*: `yolo26n_fp32.engine` is built at batch 8 and
    // `setInputShape` refuses any other batch outright — "Static dimension mismatch ... Set
    // dimensions are [1,3,640,640]. Expected dimensions are [8,3,640,640]". So a frame at a
    // time is not merely inefficient here, it does not run. The Python side hit the same wall
    // from the other direction, submitting a whole frame's crops against a plan built at 16.
    //
    // Every frame in `batch` must already be on `device`; the worker owns that, which is what
    // makes the lease device-affine and the access legal.
    struct Work {
        FrameState* state = nullptr;
        const uint8_t* image_device = nullptr;
    };
    void execute(std::vector<Work>& batch, int device, FrameCollector& collector);

    ModelPool& detector() { return *detector_; }
    ModelPool* segmenter() { return segmenter_.get(); }
    ModelPool* embedder() { return embedder_.get(); }
    ModelPool* ship_embedder() { return ship_embedder_.get(); }
    std::vector<std::string> stage_names() const;

  private:
    void run_objects(ModelPool* pool, FrameState& state, const uint8_t* image_device,
                     int device, const std::vector<float>& boxes,
                     const std::vector<int>& indices, int crop_h, int crop_w, const char* stage,
                     FrameCollector& collector);

    GraphConfig config_;
    std::unique_ptr<ModelPool> detector_;
    std::unique_ptr<ModelPool> segmenter_;
    std::unique_ptr<ModelPool> embedder_;
    std::unique_ptr<ModelPool> ship_embedder_;
};

}  // namespace shipinfer
