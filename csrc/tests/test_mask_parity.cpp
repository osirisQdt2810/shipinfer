// The mask gate: this plane's fold against the areas the Python plane's produced.
//
// `test_record_parity` compares the two record BUILDERS on rows a scenario states -- and for
// `mask_area_px` a scenario states an already-reduced `(N, 1)`, so it can say nothing about
// the fold that produced it. That is exactly how `CSRC-SEGMENT-FOLD-MISSING` stayed invisible:
// this plane had no fold at all and published `output0[crop][0][0]`, a box coordinate, while
// the record gate went green. So this gate is one seam upstream: the scenario states what the
// ENGINE answered and each plane reduces it to one area per crop.
//
// A byte compare of the spellings, like every other gate here. The values are chosen so no
// summation order can move a cell across the threshold (`mask_scenario.py` says why).
//
// Offline: g++ alone, no CUDA, no TensorRT.

#include <cstdio>
#include <string>
#include <vector>

#include "shipinfer/core/events/json.h"
#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/graph/mask_area.h"
#include "tests/mask_scenario.h"
#include "tests/parity_files.h"

namespace {

    using namespace shipinfer;
    using namespace shipinfer::parity;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    // Listed rather than globbed: a golden that disappears has to fail this gate, and a
    // directory walk would call that "nothing to check" and pass.
    const std::vector<std::string> kScenarios = {"tiny_bank"};

    // Scenarios with NO golden, because what they describe is a response both planes must
    // REFUSE -- the half a byte compare cannot express.
    const std::vector<std::string> kRefused = {"truncated_basis"};

    std::string areas_text(const MaskScenario& scenario) {
        const OutputTensor areas = mask_area(response_of(scenario), spec_of(scenario));
        std::string out;
        for (float area : areas.data) {
            out += events::json_number(static_cast<double>(area)) + "\n";
        }
        return out;
    }

    void byte_compares_every_golden() {
        for (const std::string& name : kScenarios) {
            const MaskScenario scenario =
                load_mask_scenario(resolve("scenarios/masks/" + name + ".scn"));
            std::string expected;
            for (const std::string& line :
                 read_lines(resolve("golden/masks/" + name + ".txt"))) {
                expected += line + "\n";
            }
            const std::string got = areas_text(scenario);
            check(got == expected, name + ": the areas this plane folds are the golden's");
            if (got != expected) {
                std::printf("  want: %s\n  got : %s\n", expected.c_str(), got.c_str());
            }
        }
    }

    // What the byte compare cannot say. The Python half asserts the same three.
    void the_edges_the_golden_cannot_state() {
        const MaskScenario scenario =
            load_mask_scenario(resolve("scenarios/masks/tiny_bank.scn"));
        const OutputTensor areas = mask_area(response_of(scenario), spec_of(scenario));
        check(areas.name == "mask_area_px", "the fold names the quantity, not a model output");
        check(areas.row_elems == 1, "one number per crop, which is what a scatter needs");
        check(areas.data.size() == scenario.crops, "and one row per crop, not per candidate");

        // The floor is the refusal this fold exists for: crop 1 scores 0.2 and its plane is
        // entirely positive, so an unconditional argmax would report the WHOLE crop.
        check(areas.data.at(1) == 0.f, "a crop below the score floor reports no area");
        check(areas.data.at(3) == 128.f, "and a crop whose mask fills it reports the crop");

        MaskAreaSpec no_floor = spec_of(scenario);
        no_floor.score_threshold = 0.f;
        const OutputTensor unguarded = mask_area(response_of(scenario), no_floor);
        check(unguarded.data.at(1) == 128.f,
              "without the floor that same crop reports the largest area of the four, which "
              "is the plausible-and-wrong answer the threshold refuses");
    }

    bool refused(const MaskScenario& scenario) {
        try {
            mask_area(response_of(scenario), spec_of(scenario));
            return false;
        } catch (const std::exception&) {
            return true;
        }
    }

    void refuses_what_python_refuses() {
        for (const std::string& name : kRefused) {
            const MaskScenario scenario =
                load_mask_scenario(resolve("scenarios/masks/" + name + ".scn"));
            check(refused(scenario), name + ": both planes refuse this response");
        }
        // A missing output, which is a detection-only engine in a segment slot.
        MaskScenario scenario = load_mask_scenario(resolve("scenarios/masks/tiny_bank.scn"));
        InferenceResponse one_output = response_of(scenario);
        one_output.outputs.pop_back();
        bool named_it = false;
        try {
            mask_area(one_output, spec_of(scenario));
        } catch (const std::exception& error) {
            named_it = std::string(error.what()).find("output1") != std::string::npos;
        }
        check(named_it, "a missing output is refused BY NAME");

        // A DYNAMIC dimension. `TensorSpec::elements_per_row` clamps a negative to 1 while
        // `output_dims` reports it verbatim, so an `OutputTensor` can carry a width and a
        // shape that disagree -- and this is the first consumer to trust the shape.
        InferenceResponse dynamic = response_of(scenario);
        dynamic.outputs.at(1).dims = {-1, 4, 2};
        bool refused_dynamic = false;
        std::string what;
        try {
            mask_area(dynamic, spec_of(scenario));
        } catch (const std::exception& error) {
            refused_dynamic = true;
            what = error.what();
        }
        check(refused_dynamic, "a shape holding a dynamic dimension is refused");
        check(what.find("output1") != std::string::npos && what.find("-1") != std::string::npos,
              "naming the output and the shape, not folding -1 into a cell count");
        // The diagnostic that found the first version of this guard doing nothing: the `-1`
        // reached `static_cast<size_t>` and the message said "18446744073709551615 prototype
        // plane(s)", which is a truncated-basis refusal wearing a dynamic shape's clothes.
        if (!refused_dynamic || what.find("-1") == std::string::npos) {
            std::printf("  message was: %s\n", what.c_str());
        }
    }

}  // namespace

int main() {
    try {
        byte_compares_every_golden();
        the_edges_the_golden_cannot_state();
        refuses_what_python_refuses();
    } catch (const std::exception& error) {
        std::printf("FAIL: uncaught: %s\n", error.what());
        ++failures;
    }
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
