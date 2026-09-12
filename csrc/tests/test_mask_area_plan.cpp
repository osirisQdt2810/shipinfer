// The fold's refusals, with two `(name, dims)` pairs and no driver.
//
// The offline half of `MASK-FOLD-BELONGS-ON-THE-DEVICE`: every line below is a statement about
// a CONFIGURATION -- an engine that is not the one a slot was configured for -- and none of
// them needs a GPU to decide. The kernel's own agreement with the readable fold is
// `test_mask_area_kernel.cpp`, which does need one.
#include <cstdio>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/graph/mask_area_plan.h"

namespace {

    using namespace shipinfer;

    int checks = 0;
    int failures = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (condition) return;
        ++failures;
        std::printf("FAIL: %s\n", what.c_str());
    }

    // The shipped `yolo26n-seg` shapes: 300 candidates of `6 + 32`, and 32 planes at 160x160.
    std::vector<DeclaredOutput> shipped() {
        return {{"output0", {300, 38}}, {"output1", {32, 160, 160}}};
    }

    MaskAreaSpec spec_of() {
        MaskAreaSpec spec;
        spec.crop_height = 640;
        spec.crop_width = 640;
        return spec;
    }

    void the_shipped_shapes_resolve() {
        const MaskAreaPlan plan = resolve_mask_area(shipped(), spec_of());

        check(plan.detections_index == 0, "the rows are the engine's first output");
        check(plan.prototypes_index == 1, "and the bank its second");
        check(plan.candidates == 300 && plan.stride == 38, "the row block, as declared");
        check(plan.channels == 32, "one coefficient a plane");
        check(plan.cells == 160 * 160, "and the plane's cells, multiplied out");
        check(plan.prefix == 6, "the columns before the coefficients");
        check(plan.name == "mask_area_px", "the fold's output carries the spec's name");
    }

    void the_names_are_the_artefacts_and_not_positions() {
        // Which slot an export puts its prototypes in is the export's choice, so a chain says
        // `params: {segment: {prototypes: ...}}` and the resolution follows the NAME.
        std::vector<DeclaredOutput> swapped{{"protos", {32, 160, 160}}, {"dets", {300, 38}}};
        MaskAreaSpec spec = spec_of();
        spec.detections = "dets";
        spec.prototypes = "protos";

        const MaskAreaPlan plan = resolve_mask_area(swapped, spec);

        check(plan.detections_index == 1 && plan.prototypes_index == 0,
              "positions follow the names, not the other way round");
    }

    void a_missing_output_names_what_was_there() {
        bool refused = false;
        std::string message;
        try {
            resolve_mask_area({{"output0", {300, 38}}}, spec_of());
        } catch (const ConfigError& error) {
            refused = true;
            message = error.what();
        }

        check(refused, "a detection-only engine cannot serve a segment slot");
        check(message.find("output1") != std::string::npos,
              "the missing name is in the message");
        check(message.find("got: output0") != std::string::npos, "and so is what was there");
    }

    void a_truncated_basis_is_refused() {
        // The defect this catches is silent: 16 coefficients against 32 planes builds a
        // plausible mask from half a basis, and nothing downstream can tell.
        std::vector<DeclaredOutput> mismatched{{"output0", {300, 22}},
                                               {"output1", {32, 160, 160}}};
        bool refused = false;
        try {
            resolve_mask_area(mismatched, spec_of());
        } catch (const BackendError& error) {
            refused = std::string(error.what()).find("truncated basis") != std::string::npos;
        }

        check(refused, "16 coefficients against 32 planes is not this slot's engine");
    }

    void a_dynamic_dimension_is_refused_rather_than_clamped() {
        std::vector<DeclaredOutput> dynamic{{"output0", {300, 38}}, {"output1", {32, -1, 160}}};
        bool refused = false;
        try {
            resolve_mask_area(dynamic, spec_of());
        } catch (const BackendError& error) {
            refused = std::string(error.what()).find("dynamic or empty") != std::string::npos;
        }

        check(refused, "a cell count from a dimension the plan did not fix is a placeholder");
    }

    void a_row_block_of_the_wrong_rank_is_refused() {
        std::vector<DeclaredOutput> flat{{"output0", {11400}}, {"output1", {32, 160, 160}}};
        bool refused = false;
        try {
            resolve_mask_area(flat, spec_of());
        } catch (const BackendError& error) {
            refused = std::string(error.what()).find("(rows, 6 + coeffs)") != std::string::npos;
        }

        check(refused, "a flattened row block and a row block are two shapes");
    }

}  // namespace

int main() {
    the_shipped_shapes_resolve();
    the_names_are_the_artefacts_and_not_positions();
    a_missing_output_names_what_was_there();
    a_truncated_basis_is_refused();
    a_dynamic_dimension_is_refused_rather_than_clamped();
    a_row_block_of_the_wrong_rank_is_refused();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
