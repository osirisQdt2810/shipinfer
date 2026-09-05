#include "shipinfer/backends/tensorrt/engine.h"

#include <NvInferPlugin.h>

#include <fstream>
#include <iostream>
#include <numeric>

#include "shipinfer/core/buffers.h"
#include "shipinfer/core/platform.h"

namespace shipinfer {
    namespace {

        // TensorRT talks to us through this. Warnings and above only: an INFO-level logger at
        // twenty-four instances is thousands of lines of layer detail nobody reads, and on the
        // Python side it was also the thing that deadlocked start-up.
        class Logger : public nvinfer1::ILogger {
          public:
            void log(Severity severity, const char* msg) noexcept override {
                if (severity <= Severity::kWARNING) {
                    std::cerr << "[trt] " << msg << "\n";
                }
            }
        };

        Logger& logger() {
            static Logger instance;
            return instance;
        }

        size_t element_size(nvinfer1::DataType type) {
            switch (type) {
                case nvinfer1::DataType::kFLOAT:
                    return 4;
                case nvinfer1::DataType::kHALF:
                    return 2;
                case nvinfer1::DataType::kINT8:
                    return 1;
                case nvinfer1::DataType::kINT32:
                    return 4;
                case nvinfer1::DataType::kBOOL:
                    return 1;
                case nvinfer1::DataType::kUINT8:
                    return 1;
                default:
                    return 4;
            }
        }

    }  // namespace

    std::mutex& engine_load_mutex() {
        static std::mutex instance;
        return instance;
    }

    std::shared_ptr<TrtEngine> TrtEngine::load(const std::string& plan_path, int device) {
        // `new` rather than make_shared: the constructor is private, which is deliberate — a
        // half-built engine has a runtime and no engine and must not be observable.
        std::shared_ptr<TrtEngine> self(new TrtEngine());
        self->path_ = plan_path;
        self->device_ = device;

        GPU_CHECK(gpuSetDevice(device));

        std::ifstream file(plan_path, std::ios::binary | std::ios::ate);
        if (!file) {
            // The commonest way to reach this, by a distance: a resolved plan names an
            // artefact the repository does not hold yet. Engines are host-specific and built
            // on the target node (`model_repository/*/1/README.md`), so a fresh checkout has
            // a `config.yaml` and no `model.plan` -- and the operator meets that here, after
            // a container start, rather than when the plan was written.
            throw BackendError("cannot open plan " + plan_path +
                               ". An engine is built on the node that runs it: "
                               "`python scripts/build_engines.py` inside the container");
        }
        const std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> blob(static_cast<size_t>(size));
        if (!file.read(blob.data(), size)) throw BackendError("cannot read plan " + plan_path);

        {
            // See the header: kept for host-memory and driver contention, not for a GIL that
            // does not exist here.
            std::lock_guard<std::mutex> lock(engine_load_mutex());
            static bool plugins_ready = false;
            if (!plugins_ready) {
                initLibNvInferPlugins(&logger(), "");
                plugins_ready = true;
            }
            self->runtime_ = nvinfer1::createInferRuntime(logger());
            if (self->runtime_ == nullptr)
                throw BackendError("createInferRuntime returned null");
            self->engine_ = self->runtime_->deserializeCudaEngine(blob.data(), blob.size());
        }
        if (self->engine_ == nullptr) {
            throw BackendError(
                "deserializeCudaEngine returned null for " + plan_path +
                " — the plan is truncated, or was built for a different TensorRT "
                "version or compute capability");
        }
        self->introspect();
        return self;
    }

    void TrtEngine::introspect() {
        const int total = engine_->getNbIOTensors();
        for (int i = 0; i < total; ++i) {
            const char* name = engine_->getIOTensorName(i);
            const auto shape = engine_->getTensorShape(name);
            const bool is_input =
                engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;

            TensorSpec spec;
            spec.name = name;
            spec.is_input = is_input;
            spec.element_size = element_size(engine_->getTensorDataType(name));
            // Dimension 0 is the batch. A static plan states it; a dynamic one has -1 and the
            // profile's max is the truth.
            for (int d = 1; d < shape.nbDims; ++d) spec.dims.push_back(shape.d[d]);

            int batch = shape.nbDims > 0 ? static_cast<int>(shape.d[0]) : 1;
            if (is_input && batch < 0) static_batch_ = false;
            if (batch < 0) {
                const auto profile_max =
                    engine_->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
                batch = profile_max.nbDims > 0 ? static_cast<int>(profile_max.d[0]) : 1;
            }
            if (is_input) {
                // The engine's own batch, not the config's. A config that disagrees is a config
                // that will lose a whole batch's requests at run time.
                max_batch_ = std::max(max_batch_, batch);
                inputs_.push_back(std::move(spec));
            } else {
                outputs_.push_back(std::move(spec));
            }
        }
        if (inputs_.empty()) throw BackendError(path_ + " has no input tensors");
        if (outputs_.empty()) throw BackendError(path_ + " has no output tensors");
    }

    TrtEngine::~TrtEngine() {
        if (engine_ != nullptr) delete engine_;
        if (runtime_ != nullptr) delete runtime_;
    }

    TrtInstance::TrtInstance(std::shared_ptr<TrtEngine> engine, int device)
        : engine_(std::move(engine)), device_(device) {
        GPU_CHECK(gpuSetDevice(device_));
        context_ = engine_->raw()->createExecutionContext();
        if (context_ == nullptr) throw BackendError("createExecutionContext returned null");
        GPU_CHECK(gpuStreamCreate(&stream_));

        const int batch = engine_->max_batch();
        for (const auto& spec : engine_->inputs()) {
            input_buffers_.emplace_back(spec.row_bytes() * static_cast<size_t>(batch));
            context_->setTensorAddress(spec.name.c_str(), input_buffers_.back().get());
        }
        for (const auto& spec : engine_->outputs()) {
            const size_t bytes = spec.row_bytes() * static_cast<size_t>(batch);
            output_buffers_.emplace_back(bytes);
            host_outputs_.emplace_back(bytes);
            context_->setTensorAddress(spec.name.c_str(), output_buffers_.back().get());
        }
    }

    TrtInstance::~TrtInstance() {
        // Order matters: the context references the buffers, so it goes first. And the stream
        // is synchronised before anything is freed, because a free racing an in-flight kernel
        // is a crash somewhere else entirely.
        if (stream_ != nullptr) gpuStreamSynchronize(stream_);
        if (context_ != nullptr) delete context_;
        if (stream_ != nullptr) gpuStreamDestroy(stream_);
    }

    void* TrtInstance::scratch(size_t bytes) {
        if (scratch_.bytes() < bytes) {
            // Grown, never shrunk: the sizes here are bounded by `max_batch` so this settles
            // after the first few calls and then never allocates again.
            scratch_ = DeviceBuffer(bytes);
        }
        return scratch_.get();
    }

    size_t TrtInstance::output_rows(size_t index) const {
        return engine_->outputs().at(index).elements_per_row();
    }

    void TrtInstance::execute(int rows) {
        if (rows <= 0) return;
        if (rows > engine_->max_batch()) {
            throw BackendError(
                "assembled batch of " + std::to_string(rows) + " rows exceeds max_batch_size " +
                std::to_string(engine_->max_batch()) + " for " + engine_->path());
        }
        // No `gpuSetDevice`: the caller runs on a thread bound to this instance's device for
        // life (ADR-002) and the pool lease is device-affine; binding again per inference was
        // redundant against both.

        for (const auto& spec : engine_->inputs()) {
            nvinfer1::Dims dims{};
            dims.nbDims = static_cast<int>(spec.dims.size()) + 1;
            dims.d[0] = rows;
            for (size_t d = 0; d < spec.dims.size(); ++d) {
                dims.d[d + 1] = static_cast<int>(spec.dims[d]);
            }
            if (!context_->setInputShape(spec.name.c_str(), dims)) {
                throw BackendError("setInputShape failed for " + spec.name + " at " +
                                   std::to_string(rows) +
                                   " row(s) — a static plan refuses any "
                                   "batch but its own");
            }
        }
        if (!context_->enqueueV3(stream_)) throw BackendError("enqueueV3 failed");

        for (size_t i = 0; i < engine_->outputs().size(); ++i) {
            const auto& spec = engine_->outputs()[i];
            GPU_CHECK(gpuMemcpyAsync(host_outputs_[i].get(), output_buffers_[i].get(),
                                     spec.row_bytes() * static_cast<size_t>(rows),
                                     gpuMemcpyDeviceToHost, stream_));
        }
        GPU_CHECK(gpuStreamSynchronize(stream_));
        ++executed_;
        rows_ += static_cast<uint64_t>(rows);
    }

}  // namespace shipinfer
