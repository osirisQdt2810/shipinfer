// The backend contract as the instance sees it — `backends/base.py`, seam 4 of CLAUDE.md.
//
// A backend receives an assembled batch and returns one. It does not decide what to batch,
// where to run, or when. Here that is four things: where it lives, how many rows it takes, how
// a row gets *into* its input binding (the backend owns the copies — Triton counts them as
// compute_input, and so does the Python instance), and running it. `TrtInstance` implements it
// through an adapter; the tests implement it with host memory and an identity "network", which
// is what lets the instance thread, the batching and the scatter be tested without a GPU.
//
// ONE INPUT, N OUTPUTS. This said "one output" until CSRC-SEGMENT-FOLD-MISSING, and that was
// the reason `InstanceMaskArea` had never been ported: a YOLO-seg engine emits detection rows
// AND a bank of mask prototypes, so a contract with one output has nowhere for the second to
// arrive. `TrtInstance` already read every output back (`host_outputs_` is a vector); only
// this contract and the adapter narrowed to the first. The row stays the unit the batcher and
// the scatter agree on: every output is `rows` long, so one row index selects one object's
// slice of all of them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "shipinfer/backends/tensor_shape.h"
#include "shipinfer/core/device.h"

namespace shipinfer {

    class Engine {
      public:
        virtual ~Engine() = default;
        virtual Device device() const = 0;
        virtual int max_batch() const = 0;
        // Floats per row, in and out. `index` selects one of `outputs()`; the default is the
        // first, which is the whole answer for the single-output models this began with.
        virtual size_t input_row_elems() const = 0;
        virtual size_t output_row_elems(size_t index = 0) const = 0;
        // How many outputs this engine has. Defaulted rather than pure: every double in the
        // tests has one, and a contract that made them all say so would be a contract paid
        // for by the implementations that do not need it. THE COST OF THAT DEFAULT: a real
        // backend that forgets to override it drops every output past the first SILENTLY --
        // there is no compile error, and a fold reading a second output would refuse by name
        // at run time rather than at load. `TrtEngineAdapter` is the only implementation that
        // answers anything but 1 today, so a second one starts by overriding this.
        virtual size_t outputs() const { return 1; }
        // The name the artefact gives an output, so a consumer that needs a SPECIFIC one --
        // the segmentation fold needs prototypes, not "the second" -- can ask for it by name
        // rather than by a position the export controls. Empty where a backend has no names.
        virtual std::string output_name(size_t index) const {
            (void)index;
            return "";
        }
        // One row's shape, without the batch dimension. `{output_row_elems(index)}` is the
        // honest default for a backend whose output is a flat row, which every double here is.
        virtual std::vector<int64_t> output_dims(size_t index) const {
            return {static_cast<int64_t>(output_row_elems(index))};
        }
        // Copy `rows` rows from `src` (on `src_device`) into the input binding at `row_offset`.
        // Ordered before `execute` by the backend's own stream; the caller never synchronises.
        virtual void write_rows(size_t row_offset, const float* src, size_t rows,
                                Device src_device) = 0;
        // Run `rows` rows. Returns when every `output()` is readable on the host.
        virtual void execute(int rows) = 0;
        virtual const float* output(size_t index = 0) const = 0;
    };

    // Every output's width and shape state the same row, checked once where the engine is
    // attached. The default `output_dims` derives one from the other and cannot disagree; an
    // override can, and `TrtEngineAdapter` reads the two from different places -- the width
    // from `TensorSpec::elements_per_row()`, the shape from the dimensions the plan declared.
    // A backend whose two answers differ has a batcher slicing rows at one width while a
    // consumer reads them at another, so this is a load-time refusal rather than a per-row
    // check on the dispatch path.
    inline void require_shapes_agree(const std::string& where, const Engine& engine) {
        for (size_t index = 0; index < engine.outputs(); ++index) {
            const std::vector<int64_t> dims = engine.output_dims(index);
            const std::string name = engine.output_name(index);
            require_static_row(name.empty() ? std::to_string(index) : name, dims);
            const size_t declared = engine.output_row_elems(index);
            if (elements_per_row(dims) == declared) continue;
            std::string shape;
            for (size_t i = 0; i < dims.size(); ++i) {
                shape += (i ? ", " : "") + std::to_string(dims[i]);
            }
            throw ConfigError(where + ": output '" +
                              (name.empty() ? std::to_string(index) : name) + "' is " +
                              std::to_string(declared) + " float(s) a row but has the shape (" +
                              shape + "), which is " + std::to_string(elements_per_row(dims)));
        }
    }

}  // namespace shipinfer
