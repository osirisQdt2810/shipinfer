// The device fold against the host fold, through the real engine and the real adapter.
//
// `test_mask_area_kernel.cpp` proves the ARITHMETIC agrees on numbers a test made up. This
// proves the WIRING: the same batch through one adapter that folds on the device and one that
// does not, and the area the first advertises is the area the second's outputs fold to. What
// it catches is everything between the two -- the wrong output left on the device, the fold
// reading the next batch's buffers, an off-by-one in the output remapping, a name that does
// not reach the stage.
//
// Container tier, and it needs the repository's built engine: `model.plan` is per node
// (`model_repository/ship_segmenter/1/README.md`), so an absent one is a skip and not a
// failure -- the same stance `deploy/rootless/cpp.sh` takes.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "shipinfer/backends/tensorrt/adapter.h"
#include "shipinfer/backends/tensorrt/fold.h"
#include "shipinfer/core/platform.h"
#include "shipinfer/engine/request.h"
#include "shipinfer/pipeline/graph/mask_area.h"

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

    std::string plan_path() {
        const char* repository = std::getenv("SHIPINFER_TEST_REPOSITORY");
        const std::string root = repository != nullptr ? repository : "/work/model_repository";
        return root + "/ship_segmenter/1/model.plan";
    }

    MaskAreaSpec spec_of() {
        MaskAreaSpec spec;
        spec.crop_height = 640;
        spec.crop_width = 640;
        // NO SCORE FLOOR, deliberately. The input below is a pattern rather than a ship, so at
        // the production floor of 0.25 every crop is "found nothing" and both sides answer 0 --
        // a comparison a fold returning a constant zero would also pass. At 0 the argmax row is
        // always taken and the bank is really reduced, so the numbers below mean something.
        spec.score_threshold = 0.f;
        return spec;
    }

    // The same spec at a cut nothing passes. Paired with the one above it is what makes the
    // comparison mean something: a fold that never read the bank would answer the same number
    // at both cuts, and these two answers differ.
    MaskAreaSpec a_strict_spec() {
        MaskAreaSpec spec = spec_of();
        spec.mask_threshold = 0.999f;
        return spec;
    }

    // A deterministic batch, in the input's own shape. The pixels do not have to be a ship:
    // both arms see the SAME numbers, and what is under test is where the reduction ran.
    std::vector<float> a_batch(size_t rows, size_t width) {
        std::vector<float> input(rows * width);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>((i * 7919 + 13) % 251) / 251.0f;
        }
        return input;
    }

    // Everything one adapter answered, as a stage would read it.
    InferenceResponse response_of(Engine& engine, int rows) {
        InferenceResponse response;
        response.rows = static_cast<size_t>(rows);
        for (size_t o = 0; o < engine.outputs(); ++o) {
            OutputTensor tensor;
            tensor.name = engine.output_name(o);
            tensor.row_elems = engine.output_row_elems(o);
            tensor.dims = engine.output_dims(o);
            const float* base = engine.output(o);
            tensor.data.assign(base, base + tensor.row_elems * static_cast<size_t>(rows));
            response.outputs.push_back(std::move(tensor));
        }
        return response;
    }

    void a_kept_output_stops_being_copied_home_and_stops_being_advertised() {
        // `ENGINE-COPIES-EVERY-OUTPUT-HOME`. The fold proved ONE output can stop coming home;
        // this is the general door, and what it has to get right is that keeping also HIDES.
        // A host buffer nothing wrote is worse than no output at all, because a reader finds
        // it and gets the previous batch's numbers.
        std::shared_ptr<TrtEngine> engine;
        try {
            engine = TrtEngine::load(plan_path(), 0);
        } catch (const std::exception& error) {
            std::printf("SKIP: no engine at %s (%s)\n", plan_path().c_str(), error.what());
            return;
        }
        if (engine->outputs().size() < 2) {
            std::printf("SKIP: %s has one output\n", plan_path().c_str());
            return;
        }
        const int rows = engine->max_batch();
        const size_t width = engine->inputs().front().elements_per_row();
        const std::vector<float> input = a_batch(static_cast<size_t>(rows), width);
        // THE SECOND output: the first is the one every consumer reads, so a bug that kept
        // index 0 would be caught by everything else in this file.
        const std::string kept = engine->outputs()[1].name;

        TrtEngineAdapter home(std::make_unique<TrtInstance>(engine, 0));
        TrtEngineAdapter stays(std::make_unique<TrtInstance>(engine, 0));
        stays.keep_on_device(kept);
        for (Engine* adapter : {static_cast<Engine*>(&home), static_cast<Engine*>(&stays)}) {
            adapter->write_rows(0, input.data(), static_cast<size_t>(rows), Device::cpu());
            adapter->execute(rows);
        }

        check(home.outputs() == 2, "the ordinary adapter advertises both outputs");
        check(stays.outputs() == 1, "and the keeping one advertises the other alone");
        for (size_t o = 0; o < stays.outputs(); ++o) {
            check(stays.output_name(o) != kept,
                  "the kept output is not advertised under any index, so nothing can read a "
                  "host buffer this run never wrote");
        }
        // THE ONE THAT REMAINS IS UNCHANGED, which is what says the skip skipped only the
        // copy: same engine, same batch, same numbers as the adapter that copied both home.
        const size_t elems = home.output_row_elems(0) * static_cast<size_t>(rows);
        const float* expected = home.output(0);
        const float* got = stays.output(0);
        size_t differing = 0;
        for (size_t i = 0; i < elems; ++i) {
            if (expected[i] != got[i]) ++differing;
        }
        check(differing == 0,
              "every one of " + std::to_string(elems) +
                  " floats of the REMAINING output matches the run that copied both home");
    }

    void the_device_fold_answers_what_the_host_fold_answers() {
        std::shared_ptr<TrtEngine> engine;
        try {
            engine = TrtEngine::load(plan_path(), 0);
        } catch (const std::exception& error) {
            std::printf("SKIP: no engine at %s (%s)\n", plan_path().c_str(), error.what());
            return;
        }
        const int rows = engine->max_batch();
        const size_t width = engine->inputs().front().elements_per_row();
        const std::vector<float> input = a_batch(static_cast<size_t>(rows), width);

        TrtEngineAdapter plain(std::make_unique<TrtInstance>(engine, 0));
        TrtEngineAdapter folded(std::make_unique<TrtInstance>(engine, 0),
                                mask_area_fold(*engine, spec_of()));
        TrtEngineAdapter strict(std::make_unique<TrtInstance>(engine, 0),
                                mask_area_fold(*engine, a_strict_spec()));
        for (Engine* adapter : {static_cast<Engine*>(&plain), static_cast<Engine*>(&folded),
                                static_cast<Engine*>(&strict)}) {
            adapter->write_rows(0, input.data(), static_cast<size_t>(rows), Device::cpu());
            adapter->execute(rows);
        }

        const InferenceResponse plain_out = response_of(plain, rows);
        const OutputTensor expected = mask_area(plain_out, spec_of());
        const OutputTensor expected_strict = mask_area(plain_out, a_strict_spec());
        const InferenceResponse got = response_of(folded, rows);
        const InferenceResponse got_strict = response_of(strict, rows);

        check(plain.outputs() == 2, "the unfolded adapter advertises the engine's two outputs");
        check(folded.outputs() == 2,
              "and the folded one advertises the rows plus the area -- the bank is gone");
        const OutputTensor* areas = got.named(spec_of().name);
        const OutputTensor* strict_areas = got_strict.named(spec_of().name);
        check(areas != nullptr && strict_areas != nullptr,
              "the fold's answer is there under the name the chain gave it");
        check(got.named(spec_of().prototypes) == nullptr,
              "and the prototype bank is not, because it never came home");
        if (areas == nullptr || strict_areas == nullptr) return;

        check(areas->row_elems == 1, "one area a crop");
        check(areas->data.size() == static_cast<size_t>(rows), "one a row of the batch");
        size_t differing = 0;
        for (size_t i = 0; i < expected.data.size(); ++i) {
            // Both sides count cells and scale by the same constant, so the answers are equal
            // up to the order of one float multiply -- a relative epsilon, not an exact match.
            const float a = expected.data[i];
            const float b = areas->data[i];
            const float c = expected_strict.data[i];
            const float d = strict_areas->data[i];
            if (std::fabs(a - b) > 1e-3f * std::max(1.0f, std::fabs(a))) ++differing;
            if (std::fabs(c - d) > 1e-3f * std::max(1.0f, std::fabs(c))) ++differing;
        }
        check(differing == 0, "every crop's area agrees with the host fold's, at both cuts");
        // NON-VACUOUS: the two cuts have to disagree, or a fold that ignored the bank entirely
        // -- one that answered a constant -- would pass every line above.
        check(expected.data != expected_strict.data,
              "and the two cuts answer differently, so the bank is really being read");
        check(areas->data != strict_areas->data, "on the device side too");
        std::printf("fold wiring: %d crop(s), first area device %.3f vs host %.3f\n", rows,
                    areas->data.empty() ? 0.f : areas->data[0],
                    expected.data.empty() ? 0.f : expected.data[0]);
        std::printf("fold wiring: at the strict cut device %.3f vs host %.3f\n",
                    strict_areas->data.empty() ? 0.f : strict_areas->data[0],
                    expected_strict.data.empty() ? 0.f : expected_strict.data[0]);
    }

}  // namespace

int main() {
    the_device_fold_answers_what_the_host_fold_answers();
    a_kept_output_stops_being_copied_home_and_stops_being_advertised();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
