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
        float* input = static_cast<float*>(instance_->input());
        const size_t width = input_row_elems();
        const size_t bytes = rows * width * sizeof(float);
        float* dst = input + row_offset * width;
        if (src_device.is_cuda() && src_device != device()) {
            // A spill: the placement policy sent a row that lives on another GPU here, because
            // its home GPU's queue had backed up past the threshold. The Python plane copies in
            // that case (`.to(device)`), so this plane copies too — a peer copy on this stream,
            // which the driver stages through the host when the two devices cannot see each
            // other. ADR-002 forbids a kernel *accessing* another device's memory; a copy the
            // policy chose to pay for is the mechanism that keeps that rule true under load.
            GPU_CHECK(gpuMemcpyPeerAsync(dst, device().index, src, src_device.index, bytes,
                                         instance_->stream()));
            return;
        }
        GPU_CHECK(gpuMemcpyAsync(
            dst, src, bytes,
            src_device.is_cuda() ? gpuMemcpyDeviceToDevice : gpuMemcpyHostToDevice,
            instance_->stream()));
    }

    void TrtEngineAdapter::execute(int rows) {
        // These plans are static at their batch, and `setInputShape` refuses any other batch —
        // the first run of the ported plane sealed every frame Incomplete on exactly that. A
        // partial batch is padded to the plan's batch by repeating the last real row, so the
        // padding rows carry a real image rather than whatever the binding held before: a NaN
        // through a detector produces warnings on every layer. Only the real rows are read
        // back. The Python TensorRT backend pads the same way.
        const int plan_batch = instance_->max_batch();
        if (rows < plan_batch) {
            float* input = static_cast<float*>(instance_->input());
            const size_t width = input_row_elems();
            const float* last = input + static_cast<size_t>(rows - 1) * width;
            for (int r = rows; r < plan_batch; ++r) {
                GPU_CHECK(gpuMemcpyAsync(input + static_cast<size_t>(r) * width, last,
                                         width * sizeof(float), gpuMemcpyDeviceToDevice,
                                         instance_->stream()));
            }
        }
        instance_->execute(plan_batch);
    }
    const float* TrtEngineAdapter::output() const {
        return instance_->output();
    }

}  // namespace shipinfer
