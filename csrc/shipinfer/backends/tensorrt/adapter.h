// `TrtInstance` behind the `Engine` contract, so `ModelInstance` does not know TensorRT.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "shipinfer/backends/engine_api.h"
#include "shipinfer/backends/tensorrt/engine.h"

namespace shipinfer {

    // doc: long what attaching a fold changes about the outputs a consumer sees
    // One output computed on the device, in place of one the engine emits.
    //
    // The instance folds while it still owns the batch (`TrtInstance::set_fold`), and the
    // adapter then presents the answer as an ordinary output: the bank it reduced stops being
    // advertised, and a width-1 output named by the fold takes its place at the END of the
    // list. So `ModelInstance`'s scatter is unchanged -- every output is still `rows` long and
    // one span still selects one request's slice of each -- and the stage that used to fold on
    // the host reads a named output instead.
    struct AdapterFold {
        size_t prototypes_index = 0;
        std::string name;
        TrtInstance::DeviceFold fold;
    };

    class TrtEngineAdapter : public Engine {
      public:
        explicit TrtEngineAdapter(std::unique_ptr<TrtInstance> instance,
                                  std::optional<AdapterFold> fold = std::nullopt);
        Device device() const override;
        int max_batch() const override;
        size_t input_row_elems() const override;
        size_t output_row_elems(size_t index = 0) const override;
        size_t outputs() const override;
        std::string output_name(size_t index) const override;
        std::vector<int64_t> output_dims(size_t index) const override;
        void write_rows(size_t row_offset, const float* src, size_t rows,
                        Device src_device) override;
        void execute(int rows) override;
        const float* output(size_t index = 0) const override;
        TrtInstance& instance() { return *instance_; }
        //: Whether this adapter answers a folded output. For the stage that would otherwise
        //: fold on the host, and for a test.
        bool folds() const;

      private:
        //: The engine outputs this adapter still advertises, in order -- every one but the
        //: folded bank. Indices into the ENGINE's list; the fold sits one past the end.
        std::vector<size_t> visible_;
        std::string fold_name_;
        std::unique_ptr<TrtInstance> instance_;
        // Page-locked staging for rows that arrive from another GPU. Sized for a whole batch
        // and indexed by row offset, so two spilled rows of one batch never share a region
        // and the stream's ordering is the only synchronisation needed.
        PinnedBuffer stage_;
    };

}  // namespace shipinfer
