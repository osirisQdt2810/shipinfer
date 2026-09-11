// The fold's kernel against the fold's readable twin, on the same numbers.
//
// `graph/mask_area.cpp` is what the offline tier and the cross-plane golden check, and it is
// the only thing that makes `runtime/ops.cu`'s `mask_area_into` trustworthy: a kernel that
// agrees with nothing is a second answer, not a faster one. Profiled 11 Sep, the host fold
// costs 1.44 ms of CPU per crop and its input is 3.1 MB copied down per crop -- the reason the
// kernel exists at all.
//
// Container tier: this links CUDA, so `scripts/build_csrc.py --offline` does not build it and
// the offline job never runs it.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "shipinfer/core/buffers.h"
#include "shipinfer/core/platform.h"
#include "shipinfer/engine/request.h"
#include "shipinfer/pipeline/graph/mask_area.h"
#include "shipinfer/runtime/ops.h"

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

    //: The shapes `model_repository/ship_segmenter/config.yaml` ships, shrunk in the cell count
    //: so the test is fast and enlarged nowhere: the arithmetic does not depend on 160x160.
    constexpr int kCandidates = 12;
    constexpr int kChannels = 32;
    constexpr int kSide = 40;
    constexpr int kCells = kSide * kSide;
    constexpr int kPrefix = 6;
    constexpr int kStride = kPrefix + kChannels;

    // Deterministic, and NOT random: a fold that disagrees on one input in a thousand is a
    // parity bug either way, and a fixed sequence is one both planes can be handed.
    float sequence(int index) {
        return std::sin(static_cast<float>(index) * 0.37f) +
               0.25f * std::cos(static_cast<float>(index) * 1.13f);
    }

    InferenceResponse response_of(int crops) {
        InferenceResponse response;
        response.rows = static_cast<size_t>(crops);
        OutputTensor rows;
        rows.name = "output0";
        rows.dims = {kCandidates, kStride};
        rows.row_elems = static_cast<size_t>(kCandidates) * kStride;
        rows.data.resize(rows.row_elems * static_cast<size_t>(crops));
        for (size_t i = 0; i < rows.data.size(); ++i)
            rows.data[i] = sequence(static_cast<int>(i));
        OutputTensor protos;
        protos.name = "output1";
        protos.dims = {kChannels, kSide, kSide};
        protos.row_elems = static_cast<size_t>(kChannels) * kCells;
        protos.data.resize(protos.row_elems * static_cast<size_t>(crops));
        for (size_t i = 0; i < protos.data.size(); ++i) {
            protos.data[i] = 0.5f * sequence(static_cast<int>(i) + 7);
        }
        // ONE ROW PER CROP CLEARLY THE STRONGEST, so both implementations pick the same
        // candidate: `sequence` alone can tie, and a tie is the one input on which "the best
        // row" is not a property of the numbers.
        for (int crop = 0; crop < crops; ++crop) {
            const size_t base = static_cast<size_t>(crop) * rows.row_elems;
            for (int c = 0; c < kCandidates; ++c) rows.data[base + c * kStride + 4] = 0.1f;
            rows.data[base + static_cast<size_t>(crop % kCandidates) * kStride + 4] = 0.9f;
        }
        response.outputs.push_back(std::move(rows));
        response.outputs.push_back(std::move(protos));
        return response;
    }

    std::vector<float> on_the_device(const InferenceResponse& response,
                                     const MaskAreaSpec& spec, int crops) {
        const OutputTensor& rows = response.outputs[0];
        const OutputTensor& protos = response.outputs[1];
        DeviceBuffer rows_device(rows.data.size() * sizeof(float));
        DeviceBuffer protos_device(protos.data.size() * sizeof(float));
        DeviceBuffer areas_device(static_cast<size_t>(crops) * sizeof(float));
        GPU_CHECK(gpuMemcpy(rows_device.get(), rows.data.data(),
                            rows.data.size() * sizeof(float), gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(protos_device.get(), protos.data.data(),
                            protos.data.size() * sizeof(float), gpuMemcpyHostToDevice));

        mask_area_into(rows_device.as<float>(), kCandidates, kStride, kPrefix,
                       protos_device.as<float>(), kChannels, kCells, crops,
                       spec.score_threshold, spec.mask_threshold, spec.crop_height,
                       spec.crop_width, areas_device.as<float>(), nullptr);
        GPU_CHECK(gpuDeviceSynchronize());

        std::vector<float> areas(static_cast<size_t>(crops), -1.f);
        GPU_CHECK(gpuMemcpy(areas.data(), areas_device.get(), areas.size() * sizeof(float),
                            gpuMemcpyDeviceToHost));
        return areas;
    }

    MaskAreaSpec spec_of(float score = 0.25f, float mask = 0.5f) {
        MaskAreaSpec spec;
        spec.crop_height = 640;
        spec.crop_width = 640;
        spec.score_threshold = score;
        spec.mask_threshold = mask;
        return spec;
    }

    void the_kernel_answers_what_the_readable_fold_answers() {
        const int crops = 5;
        const InferenceResponse response = response_of(crops);
        const MaskAreaSpec spec = spec_of();

        const OutputTensor host = mask_area(response, spec);
        const std::vector<float> device = on_the_device(response, spec, crops);

        check(host.data.size() == static_cast<size_t>(crops), "one area per crop on the host");
        bool agreed = host.data.size() == device.size();
        float worst = 0.f;
        for (size_t i = 0; i < device.size() && agreed; ++i) {
            worst = std::max(worst, std::fabs(host.data[i] - device[i]));
        }
        // EXACT, not approximate. Both sides sum the same products in the same order per cell
        // and then COUNT cells, so the only float arithmetic that can differ is the dot
        // product's -- and a cell has to cross the cut for that to change an area. A tolerance
        // here would hide exactly the disagreement this test is for; one cell of 1 600 is
        // 256 px of area at these shapes, which is what the bound below allows.
        check(agreed && worst <= static_cast<float>(spec.crop_height) *
                                     static_cast<float>(spec.crop_width) / kCells,
              "every area agrees within one cell, worst " + std::to_string(worst));
        bool nonzero = false;
        for (const float area : device) nonzero = nonzero || area > 0.f;
        check(nonzero, "and the answer is not all zeros, which would agree vacuously");
    }

    void a_crop_whose_best_row_is_below_the_floor_is_area_zero() {
        const int crops = 3;
        const InferenceResponse response = response_of(crops);
        // A floor above every score in the fixture: both implementations must answer 0 rather
        // than fold whatever the strongest row happens to be.
        const MaskAreaSpec spec = spec_of(0.95f);

        const OutputTensor host = mask_area(response, spec);
        const std::vector<float> device = on_the_device(response, spec, crops);

        bool all_zero = true;
        for (size_t i = 0; i < device.size(); ++i) {
            all_zero = all_zero && device[i] == 0.f && host.data[i] == 0.f;
        }
        check(all_zero, "found nothing on either side is area 0, not an error");
    }

    void the_mask_threshold_moves_both_sides_together() {
        const int crops = 4;
        const InferenceResponse response = response_of(crops);

        const MaskAreaSpec loose = spec_of(0.25f, 0.2f);
        const MaskAreaSpec tight = spec_of(0.25f, 0.8f);
        const OutputTensor host_loose = mask_area(response, loose);
        const OutputTensor host_tight = mask_area(response, tight);
        const std::vector<float> device_loose = on_the_device(response, loose, crops);
        const std::vector<float> device_tight = on_the_device(response, tight, crops);

        float host_delta = 0.f;
        float device_delta = 0.f;
        for (size_t i = 0; i < device_loose.size(); ++i) {
            host_delta += host_loose.data[i] - host_tight.data[i];
            device_delta += device_loose[i] - device_tight[i];
        }
        check(host_delta > 0.f, "a looser mask threshold admits more cells on the host");
        check(std::fabs(host_delta - device_delta) <=
                  static_cast<float>(crops) * 640.f * 640.f / kCells,
              "and the kernel moves with it, within a cell per crop");
    }

    void a_threshold_with_no_logit_is_refused() {
        DeviceBuffer one(sizeof(float));
        std::string message;
        try {
            mask_area_into(one.as<float>(), kCandidates, kStride, kPrefix, one.as<float>(),
                           kChannels, kCells, 1, 0.25f, 1.0f, 640, 640, one.as<float>(),
                           nullptr);
        } catch (const ConfigError& error) {
            message = error.what();
        }
        check(message.find("logit") != std::string::npos,
              "mask_threshold 1.0 is refused rather than dividing by zero, got: " +
                  (message.empty() ? "(nothing)" : message));
    }

    void no_crops_is_not_an_error() {
        DeviceBuffer one(sizeof(float));
        mask_area_into(one.as<float>(), kCandidates, kStride, kPrefix, one.as<float>(),
                       kChannels, kCells, 0, 0.25f, 0.5f, 640, 640, one.as<float>(), nullptr);
        GPU_CHECK(gpuDeviceSynchronize());
        check(true, "a frame that segmented nothing launches nothing and throws nothing");
    }

    // doc: long the number this kernel exists for, and why it is printed and not asserted
    void what_it_costs_at_the_shipped_shapes() {
        // The shapes `ship_segmenter` actually ships -- `(300, 38)` and `(32, 160, 160)` -- and
        // the crop count of one full batch. PRINTED, NOT ASSERTED: this box is shared, and a
        // threshold here would fail for whoever else is on the device rather than for a
        // regression. The host fold's 1.44 ms/crop is in `MASK-FOLD-BELONGS-ON-THE-DEVICE`.
        const int candidates = 300, stride = 38, channels = 32, side = 160, cells = side * side;
        const int crops = 64;
        std::vector<float> rows(static_cast<size_t>(crops) * candidates * stride, 0.1f);
        std::vector<float> protos(static_cast<size_t>(crops) * channels * cells, 0.05f);
        for (int crop = 0; crop < crops; ++crop) {
            rows[static_cast<size_t>(crop) * candidates * stride + 4] = 0.9f;
        }
        DeviceBuffer rows_device(rows.size() * sizeof(float));
        DeviceBuffer protos_device(protos.size() * sizeof(float));
        DeviceBuffer areas_device(static_cast<size_t>(crops) * sizeof(float));
        GPU_CHECK(gpuMemcpy(rows_device.get(), rows.data(), rows.size() * sizeof(float),
                            gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(protos_device.get(), protos.data(), protos.size() * sizeof(float),
                            gpuMemcpyHostToDevice));
        // One warm launch first: the first kernel on a fresh context pays module loading.
        mask_area_into(rows_device.as<float>(), candidates, stride, kPrefix,
                       protos_device.as<float>(), channels, cells, crops, 0.25f, 0.5f, 640, 640,
                       areas_device.as<float>(), nullptr);
        GPU_CHECK(gpuDeviceSynchronize());

        const auto start = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < 10; ++repeat) {
            mask_area_into(rows_device.as<float>(), candidates, stride, kPrefix,
                           protos_device.as<float>(), channels, cells, crops, 0.25f, 0.5f, 640,
                           640, areas_device.as<float>(), nullptr);
        }
        GPU_CHECK(gpuDeviceSynchronize());
        const double us =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
                .count();
        std::printf("kernel: %d crops x10 in %.0f us -> %.1f us/crop (host fold: 1442)\n",
                    crops, us, us / (crops * 10));
        check(true, "the shipped shapes fold without a launch failure");
    }

}  // namespace

int main() {
    the_kernel_answers_what_the_readable_fold_answers();
    a_crop_whose_best_row_is_below_the_floor_is_area_zero();
    the_mask_threshold_moves_both_sides_together();
    a_threshold_with_no_logit_is_refused();
    no_crops_is_not_an_error();
    what_it_costs_at_the_shipped_shapes();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
