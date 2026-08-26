// The backend contract as the instance sees it — `backends/base.py`, seam 4 of CLAUDE.md.
//
// A backend receives an assembled batch and returns one. It does not decide what to batch,
// where to run, or when. Here that is four things: where it lives, how many rows it takes, how
// a row gets *into* its input binding (the backend owns the copies — Triton counts them as
// compute_input, and so does the Python instance), and running it. `TrtInstance` implements it
// through an adapter; the tests implement it with host memory and an identity "network", which
// is what lets the instance thread, the batching and the scatter be tested without a GPU.
#pragma once

#include <cstddef>
#include <cstdint>

#include "shipinfer/core/device.h"

namespace shipinfer {

    class Engine {
      public:
        virtual ~Engine() = default;
        virtual Device device() const = 0;
        virtual int max_batch() const = 0;
        // Floats per row, in and out. One input and one output: the perception models here
        // have that shape, and the row is the unit the batcher and the scatter agree on.
        virtual size_t input_row_elems() const = 0;
        virtual size_t output_row_elems() const = 0;
        // Copy `rows` rows from `src` (on `src_device`) into the input binding at `row_offset`.
        // Ordered before `execute` by the backend's own stream; the caller never synchronises.
        virtual void write_rows(size_t row_offset, const float* src, size_t rows,
                                Device src_device) = 0;
        // Run `rows` rows. Returns when `output()` is readable on the host.
        virtual void execute(int rows) = 0;
        virtual const float* output() const = 0;
    };

}  // namespace shipinfer
