#include "shipinfer/pipeline/graph/mask_area_plan.h"

#include "shipinfer/core/types.h"

namespace shipinfer {

    namespace {

        // `mask_area.cpp`'s, and they have to stay these two numbers: the kernel reads a row's
        // score at column 4 and its mask coefficients from column 6.
        constexpr int kPrefix = 6;

        std::string shape_of(const std::vector<int64_t>& dims) {
            std::string out = "(";
            for (size_t i = 0; i < dims.size(); ++i) {
                out += (i ? ", " : "") + std::to_string(dims[i]);
            }
            return out + ")";
        }

        size_t index_of(const std::vector<DeclaredOutput>& outputs, const std::string& name) {
            for (size_t i = 0; i < outputs.size(); ++i) {
                if (outputs[i].first == name) return i;
            }
            std::string got;
            for (const DeclaredOutput& output : outputs) {
                got += (got.empty() ? "" : ", ") + output.first;
            }
            throw ConfigError("segmentation output '" + name + "' is missing (got: " + got +
                              "); a detection-only engine has one output and a segmentation "
                              "engine has two");
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

    }  // namespace

    MaskAreaPlan resolve_mask_area(const std::vector<DeclaredOutput>& outputs,
                                   const MaskAreaSpec& spec) {
        MaskAreaPlan plan;
        plan.detections_index = index_of(outputs, spec.detections);
        plan.prototypes_index = index_of(outputs, spec.prototypes);
        const std::vector<int64_t>& rows = outputs[plan.detections_index].second;
        const std::vector<int64_t>& protos = outputs[plan.prototypes_index].second;
        require_static(rows, spec.detections);
        require_static(protos, spec.prototypes);
        if (rows.size() != 2 || rows[1] <= kPrefix) {
            throw BackendError("segmentation output '" + spec.detections +
                               "' must be (rows, 6 + coeffs) per crop, got " + shape_of(rows));
        }
        if (protos.size() != 3) {
            throw BackendError("segmentation output '" + spec.prototypes +
                               "' must be (coeffs, h, w) per crop, got " + shape_of(protos));
        }
        plan.candidates = static_cast<int>(rows[0]);
        plan.stride = static_cast<int>(rows[1]);
        plan.prefix = kPrefix;
        plan.channels = static_cast<int>(protos[0]);
        plan.cells = static_cast<int>(protos[1]) * static_cast<int>(protos[2]);
        if (plan.stride - kPrefix != plan.channels) {
            throw BackendError(
                "segmentation engine emits " + std::to_string(plan.stride - kPrefix) +
                " mask coefficient(s) per row but " + std::to_string(plan.channels) +
                " prototype plane(s); one of the two outputs is not the one this stage was "
                "configured for, and combining them would build a mask from a truncated basis");
        }
        plan.score_threshold = spec.score_threshold;
        plan.mask_threshold = spec.mask_threshold;
        plan.crop_height = spec.crop_height;
        plan.crop_width = spec.crop_width;
        plan.name = spec.name;
        return plan;
    }

}  // namespace shipinfer
