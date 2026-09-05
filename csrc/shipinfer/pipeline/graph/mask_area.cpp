#include "shipinfer/pipeline/graph/mask_area.h"

#include <cmath>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"

namespace shipinfer {

    namespace {

        // Columns before the mask coefficients in one detection row: `x1, y1, x2, y2, score,
        // class`, and the column of the confidence within them.
        constexpr size_t kPrefix = 6;
        constexpr size_t kScore = 4;

        std::string shape_of(const std::vector<int64_t>& dims) {
            std::string out = "(";
            for (size_t i = 0; i < dims.size(); ++i) {
                out += (i ? ", " : "") + std::to_string(dims[i]);
            }
            return out + ")";
        }

        void require_static(const std::vector<int64_t>& dims, const std::string& name) {
            for (int64_t dim : dims) {
                if (dim > 0) continue;
                throw BackendError("segmentation output '" + name + "' declares the shape " +
                                   shape_of(dims) +
                                   ", which holds a dynamic or empty dimension. A mask is "
                                   "counted in cells, and a cell count taken from a dimension "
                                   "the plan did not fix is arithmetic on a placeholder");
            }
        }

        const OutputTensor& require(const InferenceResponse& response,
                                    const std::string& name) {
            const OutputTensor* found = response.named(name);
            if (found == nullptr) {
                std::string got;
                for (const OutputTensor& output : response.outputs) {
                    got += (got.empty() ? "" : ", ") + output.name;
                }
                throw ConfigError("segmentation output '" + name + "' is missing (got: " + got +
                                  "); a detection-only engine has one output and a "
                                  "segmentation engine has two");
            }
            return *found;
        }

    }  // namespace

    OutputTensor mask_area(const InferenceResponse& response, const MaskAreaSpec& spec) {
        const OutputTensor& rows = require(response, spec.detections);
        const OutputTensor& protos = require(response, spec.prototypes);
        const size_t count = response.rows;

        // From the ARTEFACT's declared shapes, not solved from the flattened widths: a
        // `(candidates, 6 + coeffs)` row block and a `(coeffs, h, w)` bank are two shapes one
        // product cannot distinguish, and guessing between them would build a mask from the
        // wrong planes. This mirrors `InstanceMaskArea`'s `rows.ndim != 3` and `protos.ndim
        // != 4` refusals, which are the same statement about numpy shapes.
        // A DYNAMIC dimension first: `TensorSpec::elements_per_row` clamps a negative to 1
        // while `output_dims` reports the `-1` verbatim, so the two an `OutputTensor` carries
        // can disagree (`ENGINE-DIMS-CAN-DISAGREE-WITH-WIDTH`). This is the first consumer to
        // trust the shape, and a `-1` reaching `static_cast<size_t>` below is a plane count
        // of 18446744073709551615 in an otherwise sensible-looking message.
        require_static(rows.dims, spec.detections);
        require_static(protos.dims, spec.prototypes);
        if (rows.dims.size() != 2 || rows.dims[1] <= static_cast<int64_t>(kPrefix)) {
            throw BackendError("segmentation output '" + spec.detections +
                               "' must be (rows, 6 + coeffs) per crop, got " +
                               shape_of(rows.dims));
        }
        if (protos.dims.size() != 3) {
            throw BackendError("segmentation output '" + spec.prototypes +
                               "' must be (coeffs, h, w) per crop, got " +
                               shape_of(protos.dims));
        }
        const size_t candidates = static_cast<size_t>(rows.dims[0]);
        const size_t stride = static_cast<size_t>(rows.dims[1]);
        const size_t coefficients = stride - kPrefix;
        const size_t channels = static_cast<size_t>(protos.dims[0]);
        const size_t cells =
            static_cast<size_t>(protos.dims[1]) * static_cast<size_t>(protos.dims[2]);
        if (coefficients != channels) {
            throw BackendError(
                "segmentation engine emits " + std::to_string(coefficients) +
                " mask coefficient(s) per row but " + std::to_string(channels) +
                " prototype plane(s); one of the two outputs is not the one this stage was "
                "configured for, and combining them would build a mask from a truncated basis");
        }

        OutputTensor areas;
        areas.name = spec.name;
        areas.row_elems = 1;
        areas.data.assign(count, 0.f);
        // The logit at which a mask probability crosses `mask_threshold`, so the sigmoid over
        // 25 600 cells per object is never computed -- the same comparison, one `log`.
        const float cut = std::log(spec.mask_threshold / (1.f - spec.mask_threshold));
        const float cell_px = static_cast<float>(spec.crop_height) *
                              static_cast<float>(spec.crop_width) / static_cast<float>(cells);

        for (size_t crop = 0; crop < count; ++crop) {
            const float* candidate_rows = rows.row(crop);
            // The strongest row of this crop. The crop IS one object, so its instance is the
            // engine's best answer for it, not the union of everything it saw in it.
            size_t best = 0;
            for (size_t c = 1; c < candidates; ++c) {
                if (candidate_rows[c * stride + kScore] >
                    candidate_rows[best * stride + kScore]) {
                    best = c;
                }
            }
            const float* chosen = candidate_rows + best * stride;
            if (chosen[kScore] < spec.score_threshold) continue;  // found nothing: area 0

            const float* planes = protos.row(crop);
            size_t inside = 0;
            for (size_t cell = 0; cell < cells; ++cell) {
                float logit = 0.f;
                for (size_t m = 0; m < coefficients; ++m) {
                    logit += chosen[kPrefix + m] * planes[m * cells + cell];
                }
                if (logit >= cut) ++inside;
            }
            areas.data[crop] = static_cast<float>(inside) * cell_px;
        }
        return areas;
    }

}  // namespace shipinfer
