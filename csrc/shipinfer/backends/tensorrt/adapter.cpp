#include "shipinfer/backends/tensorrt/adapter.h"

#include <string>

#include "shipinfer/core/platform.h"
#include "shipinfer/core/types.h"

namespace shipinfer {

    TrtEngineAdapter::TrtEngineAdapter(std::unique_ptr<TrtInstance> instance)
        : instance_(std::move(instance)) {}

    Device TrtEngineAdapter::device() const {
        return Device::cuda(instance_->device());
    }
    int TrtEngineAdapter::max_batch() const {
        return instance_->max_batch();
    }
    size_t TrtEngineAdapter::input_row_elems() const {
        return instance_->engine().inputs().front().elements_per_row();
    }
    size_t TrtEngineAdapter::output_row_elems() const {
        return instance_->output_rows();
    }

    void TrtEngineAdapter::write_rows(size_t row_offset, const float* src, size_t rows,
                                      Device src_device) {
        // ADR-002: one thread, one context, one GPU. A row that lives on another GPU is not
        // copied here — the placement policy's job is to keep it from arriving (locality) and
        // the service topology's job is to move it through the host ring (ledger T3). Refusing
        // beats a peer copy that works on one box and faults on the next.
        if (src_device.is_cuda() && src_device != device()) {
            throw BackendError("a batch row on " + src_device.str() +
                               " was handed to an instance on " + device().str() +
                               "; ADR-002 forbids the cross-device access");
        }
        float* input = static_cast<float*>(instance_->input());
        const size_t bytes = rows * input_row_elems() * sizeof(float);
        GPU_CHECK(gpuMemcpyAsync(
            input + row_offset * input_row_elems(), src, bytes,
            src_device.is_cuda() ? gpuMemcpyDeviceToDevice : gpuMemcpyHostToDevice,
            instance_->stream()));
    }

    void TrtEngineAdapter::execute(int rows) {
        instance_->execute(rows);
    }
    const float* TrtEngineAdapter::output() const {
        return instance_->output();
    }

}  // namespace shipinfer
