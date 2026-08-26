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
            // its home GPU's queue had backed up past the threshold. ADR-002 is absolute about
            // how it travels — nothing in this codebase performs a cross-device memory access,
            // and a payload that must move between GPUs goes through host memory. So: D2H into
            // page-locked staging, then H2D into the binding, both on this instance's stream,
            // which orders them. Two PCIe copies is the price the ADR names; it is the same on
            // every topology, which is what the ADR bought, and the policy chose to pay it.
            const size_t needed = static_cast<size_t>(max_batch()) * width * sizeof(float);
            if (stage_.bytes() < needed) stage_ = PinnedBuffer(needed);
            float* stage = stage_.as<float>() + row_offset * width;
            GPU_CHECK(
                gpuMemcpyAsync(stage, src, bytes, gpuMemcpyDeviceToHost, instance_->stream()));
            GPU_CHECK(
                gpuMemcpyAsync(dst, stage, bytes, gpuMemcpyHostToDevice, instance_->stream()));
            return;
        }
        GPU_CHECK(gpuMemcpyAsync(
            dst, src, bytes,
            src_device.is_cuda() ? gpuMemcpyDeviceToDevice : gpuMemcpyHostToDevice,
            instance_->stream()));
    }

    void TrtEngineAdapter::execute(int rows) {
        // A static plan refuses any batch but its own — the first run of the ported plane
        // sealed every frame Incomplete on exactly that — so a partial batch is padded to the
        // plan's batch by repeating the last real row: the padding rows carry a real image
        // rather than whatever the binding held before, because a NaN through a detector
        // produces warnings on every layer. Only the real rows are read back. A dynamic plan
        // runs the rows it was given; padding it would discard the batch window's whole point.
        if (!instance_->is_static()) {
            instance_->execute(rows);
            return;
        }
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
