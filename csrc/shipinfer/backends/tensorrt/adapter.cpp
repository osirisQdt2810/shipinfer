#include "shipinfer/backends/tensorrt/adapter.h"

#include <string>

#include "shipinfer/core/platform.h"
#include "shipinfer/core/types.h"

namespace shipinfer {

    TrtEngineAdapter::TrtEngineAdapter(std::unique_ptr<TrtInstance> instance,
                                       std::optional<AdapterFold> fold)
        : instance_(std::move(instance)) {
        const size_t count = instance_->engine().outputs().size();
        for (size_t i = 0; i < count; ++i) {
            if (fold && i == fold->prototypes_index) continue;
            visible_.push_back(i);
        }
        if (!fold) return;
        fold_name_ = fold->name;
        instance_->set_fold(std::move(fold->fold), fold->prototypes_index);
    }

    bool TrtEngineAdapter::folds() const {
        return !fold_name_.empty();
    }

    Device TrtEngineAdapter::device() const {
        return Device::cuda(instance_->device());
    }
    int TrtEngineAdapter::max_batch() const {
        return instance_->max_batch();
    }
    size_t TrtEngineAdapter::input_row_elems() const {
        return instance_->engine().inputs().front().elements_per_row();
    }
    size_t TrtEngineAdapter::output_row_elems(size_t index) const {
        // GUARDED ON `folds()`, not on the index alone: with no fold attached
        // `visible_.size()` is one past the last valid index, and answering 1 there would
        // make an out-of-range question look like a sensible one.
        if (folds() && index == visible_.size()) return 1;  // the fold: one area a row
        return instance_->output_rows(visible_.at(index));
    }
    const float* TrtEngineAdapter::output_device(size_t index) const {
        // THE FOLD'S OWN ANSWER IS ALWAYS ON THE HOST: it is one float a row, copied home by
        // `execute` because that is the whole saving -- a bank reduced to a number that
        // travels. So the synthetic index past `visible_` is host-resident like any other.
        if (folds() && index == visible_.size()) return nullptr;
        const size_t real = visible_.at(index);
        if (!instance_->keeps_on_device(real)) return nullptr;
        return instance_->output_device(real);
    }

    size_t TrtEngineAdapter::outputs() const {
        return visible_.size() + (folds() ? 1 : 0);
    }
    std::vector<int64_t> TrtEngineAdapter::output_dims(size_t index) const {
        if (folds() && index == visible_.size()) return {1};
        return instance_->engine().outputs().at(visible_.at(index)).dims;
    }
    std::string TrtEngineAdapter::output_name(size_t index) const {
        // The ARTEFACT's own name, which is what a chain file's
        // `params: {segment: {prototypes: output1}}` names -- so the fold asks for the output
        // it needs rather than for a position the export controls. The fold's own name is the
        // one the chain gave it, which is how the stage finds it.
        if (folds() && index == visible_.size()) return fold_name_;
        return instance_->engine().outputs().at(visible_.at(index)).name;
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

    const float* TrtEngineAdapter::output(size_t index) const {
        if (folds() && index == visible_.size()) return instance_->fold_result();
        return instance_->output(visible_.at(index));
    }

}  // namespace shipinfer
