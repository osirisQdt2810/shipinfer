// `TrtInstance` behind the `Engine` contract, so `ModelInstance` does not know TensorRT.
#pragma once

#include <memory>

#include "shipinfer/backends/engine_api.h"
#include "shipinfer/backends/tensorrt/engine.h"

namespace shipinfer {

    class TrtEngineAdapter : public Engine {
      public:
        explicit TrtEngineAdapter(std::unique_ptr<TrtInstance> instance);
        Device device() const override;
        int max_batch() const override;
        size_t input_row_elems() const override;
        size_t output_row_elems() const override;
        void write_rows(size_t row_offset, const float* src, size_t rows,
                        Device src_device) override;
        void execute(int rows) override;
        const float* output() const override;
        TrtInstance& instance() { return *instance_; }

      private:
        std::unique_ptr<TrtInstance> instance_;
        // Page-locked staging for rows that arrive from another GPU. Sized for a whole batch
        // and indexed by row offset, so two spilled rows of one batch never share a region
        // and the stream's ordering is the only synchronisation needed.
        PinnedBuffer stage_;
    };

}  // namespace shipinfer
