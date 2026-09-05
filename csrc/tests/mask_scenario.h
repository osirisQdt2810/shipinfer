// A segmentation response, from the same `.scn` the Python fold reads — `benchmarks/parity/
// mask_scenario.py`, directive for directive.
//
// The record scenario beside this one states already-reduced `(N, 1)` rows, so it compares
// two record builders and can say nothing about the fold that produced those rows. This one
// states what the ENGINE answered, and each plane reduces it.
#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/engine/request.h"
#include "shipinfer/pipeline/graph/mask_area.h"
#include "tests/parity_files.h"

namespace shipinfer::parity {

    struct MaskScenario {
        std::string name;
        int crop_height = 16;
        int crop_width = 8;
        int candidates = 2;
        int stride = 8;
        int coefficients = 2;
        int proto_height = 4;
        int proto_width = 2;
        float score_threshold = 0.25f;
        float mask_threshold = 0.5f;
        size_t crops = 0;
        // Flattened exactly as the emitter flattens them: `crops x candidates x stride` and
        // `crops x coefficients x h x w`, zero where the scenario is silent.
        std::vector<float> rows;
        std::vector<float> protos;
    };

    // The scenario as the response `ObjectStage` would have handed the fold.
    inline InferenceResponse response_of(const MaskScenario& scenario) {
        InferenceResponse response;
        response.model_name = "ship_segmenter";
        response.rows = scenario.crops;
        OutputTensor detections;
        detections.name = "output0";
        detections.row_elems =
            static_cast<size_t>(scenario.candidates) * static_cast<size_t>(scenario.stride);
        detections.dims = {scenario.candidates, scenario.stride};
        detections.data = scenario.rows;
        OutputTensor bank;
        bank.name = "output1";
        bank.row_elems = static_cast<size_t>(scenario.coefficients) *
                         static_cast<size_t>(scenario.proto_height) *
                         static_cast<size_t>(scenario.proto_width);
        bank.dims = {scenario.coefficients, scenario.proto_height, scenario.proto_width};
        bank.data = scenario.protos;
        response.outputs = {std::move(detections), std::move(bank)};
        return response;
    }

    inline MaskAreaSpec spec_of(const MaskScenario& scenario) {
        MaskAreaSpec spec;
        spec.crop_height = scenario.crop_height;
        spec.crop_width = scenario.crop_width;
        spec.score_threshold = scenario.score_threshold;
        spec.mask_threshold = scenario.mask_threshold;
        return spec;
    }

    inline MaskScenario load_mask_scenario(const std::string& path) {
        MaskScenario scenario;
        struct Row {
            int crop;
            int candidate;
            std::vector<float> values;
        };
        struct Plane {
            int crop;
            int coefficient;
            std::vector<float> values;
        };
        std::vector<Row> rows;
        std::vector<Plane> planes;
        int number = 0;
        for (const std::string& raw : read_lines_keeping_blanks(path)) {
            ++number;
            const std::string line = raw.substr(0, raw.find('#'));
            std::istringstream stream(line);
            std::vector<std::string> words;
            for (std::string word; stream >> word;) words.push_back(word);
            if (words.empty()) continue;
            const std::string where = path + ":" + std::to_string(number);
            const std::string& directive = words[0];
            if (directive == "scenario") {
                scenario.name = words.at(1);
            } else if (directive == "crop") {
                scenario.crop_height = std::stoi(words.at(1));
                scenario.crop_width = std::stoi(words.at(2));
            } else if (directive == "rows") {
                scenario.candidates = std::stoi(words.at(1));
                scenario.stride = std::stoi(words.at(2));
            } else if (directive == "protos") {
                scenario.coefficients = std::stoi(words.at(1));
                scenario.proto_height = std::stoi(words.at(2));
                scenario.proto_width = std::stoi(words.at(3));
            } else if (directive == "score_threshold") {
                scenario.score_threshold = std::stof(words.at(1));
            } else if (directive == "mask_threshold") {
                scenario.mask_threshold = std::stof(words.at(1));
            } else if (directive == "crops") {
                scenario.crops =
                    std::max(scenario.crops, static_cast<size_t>(std::stoi(words.at(1))));
            } else if (directive == "det" || directive == "plane") {
                const int crop = std::stoi(words.at(1));
                const int second = std::stoi(words.at(2));
                std::vector<float> values;
                for (size_t i = 3; i < words.size(); ++i) values.push_back(std::stof(words[i]));
                if (directive == "det") {
                    rows.push_back({crop, second, std::move(values)});
                } else {
                    planes.push_back({crop, second, std::move(values)});
                }
                scenario.crops = std::max(scenario.crops, static_cast<size_t>(crop) + 1);
            } else {
                throw ConfigError(where + ": unknown directive '" + directive + "'");
            }
        }
        if (scenario.name.empty()) throw ConfigError(path + ": no `scenario <name>` line");

        // Zero where the scenario is silent, which is what `arrays_of` does: a scenario says
        // what it is about, and the rest is the engine's own silence.
        const size_t stride = static_cast<size_t>(scenario.stride);
        const size_t cells = static_cast<size_t>(scenario.proto_height) *
                             static_cast<size_t>(scenario.proto_width);
        scenario.rows.assign(scenario.crops * static_cast<size_t>(scenario.candidates) * stride,
                             0.f);
        scenario.protos.assign(
            scenario.crops * static_cast<size_t>(scenario.coefficients) * cells, 0.f);
        constexpr size_t kPrefix = 6, kScore = 4;
        for (const Row& row : rows) {
            float* at = scenario.rows.data() + (static_cast<size_t>(row.crop) *
                                                    static_cast<size_t>(scenario.candidates) +
                                                static_cast<size_t>(row.candidate)) *
                                                   stride;
            at[kScore] = row.values.at(0);
            for (size_t i = 1; i < row.values.size(); ++i) at[kPrefix + i - 1] = row.values[i];
        }
        for (const Plane& plane : planes) {
            float* at =
                scenario.protos.data() +
                (static_cast<size_t>(plane.crop) * static_cast<size_t>(scenario.coefficients) +
                 static_cast<size_t>(plane.coefficient)) *
                    cells;
            for (size_t i = 0; i < plane.values.size(); ++i) at[i] = plane.values[i];
        }
        return scenario;
    }

}  // namespace shipinfer::parity
