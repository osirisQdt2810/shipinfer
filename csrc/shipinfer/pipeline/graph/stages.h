// The three stages of the perception DAG over `Model::infer` —
// `graph/{detect,crop,objects}.py`.
//
// Each stage submits one request per frame (or per chunk) to a Model and blocks on the future.
// Waiting is not the throughput problem it looks like: each stage's *model* batches across
// every frame in flight, so while frame A is embedding, frame B is detecting; the concurrency
// comes from the worker pool, not from this call. That is the Python plane's shape, and it
// replaces the pool-lease graph the first C++ binary had.
#pragma once

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/core/buffers.h"
#include "shipinfer/core/platform.h"
#include "shipinfer/core/types.h"
#include "shipinfer/engine/model.h"
#include "shipinfer/pipeline/graph/plan_stages.h"
#include "shipinfer/pipeline/graph/stage.h"
#include "shipinfer/pipeline/graph/state.h"
#include "shipinfer/pipeline/mtmc/barrier.h"
#include "shipinfer/pipeline/mtmc/cluster.h"
#include "shipinfer/pipeline/tracking/associator.h"

namespace shipinfer {

    // The detector's output row: x1, y1, x2, y2, score, class — the yolo26 end-to-end layout.
    constexpr size_t kDetectionStride = 6;

    // Per-worker device scratch: a small pool of owned buffers per payload name. The first
    // version had one slot per name and reused it every frame, on the premise that a worker
    // waits on every stage's future before the next frame — which stops being true the moment
    // a stage times out: the abandoned request still sits in an instance's queue pointing at
    // the slot, and the next frame's crop overwrites it while the instance DMAs out of it.
    // So a buffer is handed out as a `shared_ptr` the request keeps, and is reused only once
    // nobody else holds it. Kernels are launched on the worker's own stream and the stream
    // is synchronised before a payload is handed to a model — the model copies on *its*
    // stream, and the two are only ordered by that synchronise.
    class WorkerScratch {
      public:
        explicit WorkerScratch(Device device);
        ~WorkerScratch();
        WorkerScratch(const WorkerScratch&) = delete;
        WorkerScratch& operator=(const WorkerScratch&) = delete;
        Device device() const { return device_; }
        gpuStream_t stream() const { return stream_; }
        // A device buffer of at least `bytes` nobody else holds, by name. The pool holds
        // exactly one reference to each buffer, so `use_count() == 1` means nobody else does —
        // that is the whole arbitration, and why the pool never hands out a copy of its own
        // handle. Reuses a released buffer, allocates while fewer than `kMaxHeldPerName` are
        // outstanding, and refuses beyond that: unbounded growth under timeouts would be a leak
        // wearing a pool's coat.
        std::shared_ptr<DeviceBuffer> acquire(const std::string& name, size_t bytes);
        size_t held(const std::string& name) const;
        static constexpr size_t kMaxHeldPerName = 16;
        // The boxes upload path: page-locked host staging and a device buffer, both grow-only.
        const float* upload_boxes(const std::vector<float>& boxes);
        void synchronise();

      private:
        Device device_;
        gpuStream_t stream_ = nullptr;
        std::map<std::string, std::vector<std::shared_ptr<DeviceBuffer>>> pools_;
        PinnedBuffer boxes_host_;
        DeviceBuffer boxes_device_;
    };

    // Shared by the two model-driven stages: build the request, submit, wait with a budget.
    class ModelStage : public Stage {
      public:
        ModelStage(std::string name, Model& model, std::chrono::milliseconds timeout,
                   std::vector<std::string> consumes, std::vector<std::string> needs,
                   std::vector<std::string> produces);
        Model& model() const { return model_; }

      protected:
        InferenceResponse infer(const FrameState& state, const float* data, size_t rows,
                                size_t row_elems, Device device,
                                std::shared_ptr<const void> keepalive = {});

      private:
        Model& model_;
        std::chrono::milliseconds timeout_;
    };

    // Letterbox one frame, run the detector, decode the boxes into frame pixels.
    class DetectStage : public ModelStage {
      public:
        DetectStage(std::string name, Model& detector, DetectConfig config,
                    WorkerScratch& scratch,
                    std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

      protected:
        size_t do_run(FrameState& state) override;

      private:
        DetectConfig config_;
        WorkerScratch& scratch_;
    };

    // Cut every detection out of the frame, once per configured crop set. A crop set with no
    // members is produced with zero rows rather than omitted — that is what makes conditional
    // execution work without a second mechanism.
    class CropStage : public Stage {
      public:
        CropStage(std::string name, std::vector<CropSpec> crops, int max_objects,
                  WorkerScratch& scratch);

      protected:
        size_t do_run(FrameState& state) override;

      private:
        std::vector<CropSpec> crops_;
        int max_objects_;
        WorkerScratch& scratch_;
    };

    // A model applied to every object of one frame at once, in chunks of the engine's batch.
    // Runs only when its source payload is non-empty (`needs`), so a frame with only people
    // never reaches the ship segmenter.
    // One response in, one row per crop out. The seam `PoolSegment._reduced` is on the Python
    // plane, and it exists for the same reason: a YOLO-seg engine emits detection rows and a
    // bank of mask prototypes, and one crop's mask is the two multiplied and reduced -- a fold
    // over two outputs that a per-row scatter cannot express. Returning the engine's first
    // output unchanged is the identity, which is what an embedder wants.
    //
    // Applied per CHUNK, before anything is appended, so the fold sees one response's rows
    // together. Folding after the join would read three chunks' answers as one.
    using ObjectCombine = std::function<OutputTensor(const InferenceResponse&)>;

    // One camera's tracker over one frame's detections, as an `ObjectBatch` of ids.
    //
    // NOT a `ModelStage`: no engine and no queue, so it runs on the worker's own thread like
    // `CropStage`. The associator is SHARED by every worker -- one tracker per camera is the
    // correctness constraint -- and it is an interface, not a tracker: `tracking/associator.h`
    // explains the two build lines that forbid this unit from including one.
    //
    // IT PICKS THE ROWS IT TRACKS, on `CropSpec`'s convention, because the Python element does
    // (`elements/track.py`: `selects_rows = True`, "exactly as a crop element picks the rows it
    // embeds"). A plane that tracked every row for a `classes: [ship]` slot would emit ids the
    // other never does AND different ids for the ships, since association and the per-camera
    // counter would have seen the people too.
    //
    // A DETECTION THE TRACKER DID NOT CONFIRM gets no row, which leaves its `track_id` null
    // rather than `-1`. That is what the Python plane emits for the same case, and the chain
    // file's own comment says why it happens: a detection between the publish threshold and
    // the tracker's own only ever CONTINUES a track.
    class TrackStage : public Stage {
      public:
        TrackStage(std::string name, std::string output, int class_id,
                   std::shared_ptr<tracking::Associator> associator);

      protected:
        size_t do_run(FrameState& state) override;

      private:
        std::string output_;
        int class_id_;
        std::shared_ptr<tracking::Associator> associator_;
        //: Scratch, not state: cleared at the top of every `do_run`. One Dag per worker
        //: (`bench.cpp`) is what makes a member safe here, and it is the same rule
        //: `CropStage` follows with `WorkerScratch` -- a per-frame vector on the dispatch
        //: path is an allocation a thousand times a second for nothing.
        std::vector<Detection> selected_;
    };

    // doc: long what this stage owns and what it deliberately does not
    // CROSS-CAMERA IDENTITY, and the stage owns none of it. Two components do the work -- the
    // barrier turns the chain's one-frame-at-a-time stream back into synchronised instants,
    // and the cluster tracker turns an instant into global ids -- and this class is the join:
    // it reads each detection's box, track id and embedding out of the frame's batches, hands
    // its camera's rows to the barrier, and scatters whatever comes back onto the rows the
    // detector produced.
    //
    // WHY IT READS THE TRACK ID AT ALL. Cross-camera identity is keyed by (camera, TRACK),
    // never by (camera, detection): a detection is one frame's observation and an identity has
    // to persist across frames, so an untracked row has nothing for the identity map to hold
    // on to. Such a row is passed over rather than published with somebody else's id.
    class MtmcStage : public Stage {
      public:
        MtmcStage(std::string name, std::string output, std::string track_source,
                  std::vector<std::string> embedding_sources,
                  std::shared_ptr<mtmc::InstantBarrier> barrier,
                  std::shared_ptr<mtmc::ClusterTracker> tracker);

      protected:
        size_t do_run(FrameState& state) override;

      private:
        std::string output_;
        //: The track stage's OUTPUT name, which is where the ids are (`stages.cpp`:
        //: `out.name = output_`), not the slot's own.
        std::string track_source_;
        //: Every embedder's output name. Several, because a chain embeds people and ships
        //: separately and one row is in exactly one of them.
        std::vector<std::string> embedding_sources_;
        std::shared_ptr<mtmc::InstantBarrier> barrier_;
        std::shared_ptr<mtmc::ClusterTracker> tracker_;
    };

    class ObjectStage : public ModelStage {
      public:
        ObjectStage(std::string name, Model& model, std::string source, std::string output,
                    std::chrono::milliseconds timeout = std::chrono::milliseconds(5000),
                    ObjectCombine combine = {});

      protected:
        size_t do_run(FrameState& state) override;

      private:
        std::string source_;
        std::string output_;
        ObjectCombine combine_;
    };

}  // namespace shipinfer
