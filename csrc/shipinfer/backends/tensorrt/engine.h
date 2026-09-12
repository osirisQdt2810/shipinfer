// TensorRT engine loading and execution — one instance, one stream, one device, for life.
//
// ADR-002 in C++: a `TrtInstance` is created on a device and never migrates. That is not a
// style preference. An execution context's device memory belongs to the context that
// allocated it, and using it from a thread bound elsewhere is a silent cross-device access
// that either fails at a confusing place or, worse, works slowly.
//
// WHAT THE PYTHON VERSION TAUGHT US, KEPT HERE
// -------------------------------------------
// * **Engine deserialisation is serialised process-wide.** In Python this was a hard
//   requirement: TensorRT holds an internal lock and calls back into the `ILogger`, which
//   needed the GIL, and six instance threads deadlocked start-up forever. There is no GIL
//   here, so the deadlock cannot recur — but deserialising two dozen plans concurrently
//   still thrashes host memory and the driver, so the lock stays for throughput rather than
//   for correctness, and this comment is here so nobody removes it thinking it was only
//   about Python.
// * **The engine's own shapes are the truth.** The config is a claim; `getTensorShape` is
//   the fact. A mismatch fails at load with both numbers named, because a plan built at
//   batch 16 fed a batch of 24 loses every request in it.
#pragma once

#include <NvInfer.h>

#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "shipinfer/backends/tensor_shape.h"
#include "shipinfer/core/buffers.h"
#include "shipinfer/core/platform.h"
#include "shipinfer/core/types.h"

namespace shipinfer {

    struct TensorSpec {
        std::string name;
        std::vector<int64_t> dims;  // without the batch dimension
        bool is_input = false;
        size_t element_size = 4;

        // Delegates, and does NOT clamp: `TrtEngine` refuses a non-positive per-row
        // dimension at load (`require_static_row`), because `row_bytes()` sizes the device
        // and host buffers from this and a `-1` clamped to 1 sized them for one element per
        // row where the engine writes many.
        size_t elements_per_row() const { return shipinfer::elements_per_row(dims); }
        size_t row_bytes() const { return elements_per_row() * element_size; }
    };

    // A deserialised plan, shared by every instance of the model. Immutable after load, so one
    // copy serves all of them and the VRAM cost of the weights is paid once per device.
    class TrtEngine {
      public:
        static std::shared_ptr<TrtEngine> load(const std::string& plan_path, int device);

        ~TrtEngine();
        TrtEngine(const TrtEngine&) = delete;
        TrtEngine& operator=(const TrtEngine&) = delete;

        nvinfer1::ICudaEngine* raw() const { return engine_; }
        int device() const { return device_; }
        int max_batch() const { return max_batch_; }
        // A static plan states its batch and refuses any other; a dynamic one carries -1 and
        // a profile. Read off the plan rather than assumed, so a dynamic plan keeps the batch
        // window's whole point — running the rows it was given — instead of a padded maximum.
        bool is_static() const { return static_batch_; }
        const std::vector<TensorSpec>& inputs() const { return inputs_; }
        const std::vector<TensorSpec>& outputs() const { return outputs_; }
        const std::string& path() const { return path_; }

      private:
        TrtEngine() = default;
        void introspect();

        std::string path_;
        int device_ = 0;
        int max_batch_ = 1;
        bool static_batch_ = true;
        nvinfer1::IRuntime* runtime_ = nullptr;
        nvinfer1::ICudaEngine* engine_ = nullptr;
        std::vector<TensorSpec> inputs_;
        std::vector<TensorSpec> outputs_;
    };

    // One execution context plus its own stream and its own I/O buffers. Buffers are allocated
    // **once**, at construction, sized for `max_batch`; a smaller batch uses a prefix. That is
    // ADR-008's precondition and it holds here for the same reason it held there: a reallocated
    // binding invalidates anything that captured its address.
    class TrtInstance {
      public:
        TrtInstance(std::shared_ptr<TrtEngine> engine, int device);
        ~TrtInstance();
        TrtInstance(const TrtInstance&) = delete;
        TrtInstance& operator=(const TrtInstance&) = delete;

        // doc: long what a fold is for, and the two things it changes about `execute`
        // A reduction over this instance's own DEVICE outputs, run on its stream after the
        // network and before anything is copied home. It writes `rows` floats to `areas`.
        //
        // WHY IT LIVES HERE AND NOT IN THE STAGE that consumes it: the output buffers are
        // overwritten by the next batch on this instance, and a stage's `combine` runs after
        // `execute` has returned and the instance is free again -- so folding there is a
        // use-after-overwrite race. Only the instance thread, still holding the batch, can
        // read those buffers safely.
        //
        // WHY IT IS A CALLBACK and not a kind of fold this class knows: `TrtInstance` is the
        // TensorRT seam and has no business knowing what a mask is. The composition root
        // builds the closure (`graph/mask_area_device.h`) and hands it over.
        using DeviceFold = std::function<void(const TrtInstance&, int rows, float* areas)>;

        // Attach one, and name the output it makes unnecessary. Call once, before `start`:
        // the buffers it needs are allocated here.
        //
        // `leave_on_device` is the output whose host copy is then never made -- the prototype
        // bank, 3.1 MB a crop, which is the whole point. `output(leave_on_device)` is not
        // readable afterwards, which is why the adapter above stops advertising it.
        void set_fold(DeviceFold fold, size_t leave_on_device);

        // Leave one output where the network wrote it, for a consumer that reads it on the
        // device. Call before `start`, like `set_fold`, because this is a per-MODEL decision:
        // a chain declares once which outputs a stage consumes on the device, and asking per
        // REQUEST would spend per-frame bytes on an answer that never changes.
        //
        // BY NAME, because which position an output occupies is the export's choice and not
        // the chain's -- the same argument `InferenceResponse::named` already carries. Refused
        // when the artefact has no such output, rather than silently keeping nothing.
        //
        // `set_fold` is one caller of this: the bank it reduces is an output nothing above
        // reads, which is the same fact stated for one output rather than any.
        void keep_on_device(const std::string& output_name);
        bool folds() const { return static_cast<bool>(fold_); }
        // The fold's answer on the host: `rows` floats, valid until the next `execute`.
        const float* fold_result() const { return fold_host_.as<float>(); }
        // An output where the network wrote it. The fold was the first caller; any output
        // named to `keep_on_device` is read this way too, which is why the comment that said
        // "nothing else needs this" is gone rather than qualified.
        const float* output_device(size_t index) const {
            return static_cast<const float*>(output_buffers_.at(index).get());
        }
        //: Whether `execute` skips this output's copy home -- so a reader knows that
        //: `output(index)` is a buffer nothing wrote and `output_device(index)` is the answer.
        bool keeps_on_device(size_t index) const { return kept_on_device_.count(index) != 0; }

        // Runs `rows` of already-preprocessed input that is *already on the device*, in this
        // instance's input buffer. Returns when the outputs are readable on the host.
        //
        // Device-in, host-out is deliberate: the caller writes straight into `input()` with a
        // CUDA kernel, so a frame's pixels never make a host round trip they do not need. The
        // outputs are small — 300x6 for the detector, 2048 floats for an embedder — and the
        // graph's next decision is made on the host, so they come back.
        void execute(int rows);

        // The instance's own stream. Exposed so preprocessing can be launched **on it**: a
        // kernel and the inference that consumes its output on the same stream are ordered by
        // the stream itself, and no synchronisation is needed at all. The first version
        // launched the kernels on the default stream and then called `gpuDeviceSynchronize`,
        // which is device-wide — every worker on that GPU stalled on every other worker's
        // kernels.
        gpuStream_t stream() const { return stream_; }

        // A scratch buffer on this instance's device, for the small per-call inputs the graph
        // needs to upload (a chunk's boxes). Persistent because `gpuMalloc` serialises on the
        // driver and this is the hot path.
        void* scratch(size_t bytes);

        void* input(size_t index = 0) const { return input_buffers_.at(index).get(); }
        const float* output(size_t index = 0) const {
            return host_outputs_.at(index).as<float>();
        }
        size_t output_rows(size_t index = 0) const;

        int max_batch() const { return engine_->max_batch(); }
        bool is_static() const { return engine_->is_static(); }
        int device() const { return device_; }
        uint64_t executed() const { return executed_; }
        uint64_t rows_executed() const { return rows_; }
        const TrtEngine& engine() const { return *engine_; }

      private:
        std::shared_ptr<TrtEngine> engine_;
        int device_ = 0;
        nvinfer1::IExecutionContext* context_ = nullptr;
        gpuStream_t stream_ = nullptr;
        std::vector<DeviceBuffer> input_buffers_;
        std::vector<DeviceBuffer> output_buffers_;
        std::vector<PinnedBuffer> host_outputs_;
        DeviceBuffer scratch_;
        DeviceFold fold_;
        //: Outputs whose host copy is never made -- the fold's bank, and anything a consumer
        //: asked for on the device. A SET rather than the one index this began as: the fold
        //: was the first case, not the only one, and two callers would have overwritten each
        //: other's single slot in silence.
        std::set<size_t> kept_on_device_;
        DeviceBuffer fold_device_;
        PinnedBuffer fold_host_;
        uint64_t executed_ = 0;
        uint64_t rows_ = 0;
    };

    // Process-wide, for the reason in the header comment. Exposed so a test can assert it
    // exists.
    std::mutex& engine_load_mutex();

}  // namespace shipinfer
