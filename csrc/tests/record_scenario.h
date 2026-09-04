// One frame's STAGE OUTPUTS — `benchmarks/parity/record_scenario.py`, directive for
// directive. Line-oriented so this half needs no JSON parser and no YAML.
//
// The event scenario beside this one states finished records, so it compares the two JSON
// writers. This states what the graph leaves behind — detections by class ID, the per-object
// batches with their row indices, the label table and the field map — and each plane builds
// its own records from it. What is compared is `build_records`, on both planes.
#pragma once

#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/events/records.h"
#include "shipinfer/pipeline/graph/emission.h"
#include "tests/event_scenario.h"
#include "tests/parity_files.h"

namespace shipinfer::parity {

    struct RecordScenario {
        std::string name;
        std::string camera = "cam0";
        int64_t frame = 0;
        std::string source = "shard-0";
        int64_t width = 640;
        int64_t height = 480;
        double fps = 1.0;
        int64_t captured_ns = 0;
        int64_t captured_unix_ns = 0;
        int64_t emitted_unix_ns = 0;
        std::vector<std::string> missing;
        std::string reason = "complete";
        std::optional<FinishReason> finished;
        pipeline::events::ClassLabels labels;
        std::vector<Detection> detections;
        std::map<std::string, ObjectBatch> batches;
        //: In DECLARATION order, which is the field map's priority order -- a `std::map` would
        //: sort the field names and lose nothing, but the candidate order inside one field is
        //: the rule under test, so it is kept as written.
        std::vector<std::pair<std::string, std::vector<std::string>>> fields;
    };

    inline RecordScenario load_record_scenario(const std::string& path) {
        RecordScenario scenario;
        std::string last_batch;
        int last_width = 0;
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
            const auto want = [&](size_t count) {
                if (words.size() != count + 1) {
                    throw ConfigError(where + ": `" + directive + "` takes " +
                                      std::to_string(count) + " argument(s)");
                }
            };
            if (directive == "scenario") {
                want(1);
                scenario.name = words[1];
            } else if (directive == "camera") {
                want(1);
                scenario.camera = words[1];
            } else if (directive == "source") {
                want(1);
                scenario.source = words[1];
            } else if (directive == "reason") {
                want(1);
                scenario.reason = words[1];
            } else if (directive == "finished") {
                want(1);
                scenario.finished = detail::finish_reason_of(words[1], where);
            } else if (directive == "frame") {
                want(1);
                scenario.frame = std::stoll(words[1]);
            } else if (directive == "size") {
                want(2);
                scenario.width = std::stoll(words[1]);
                scenario.height = std::stoll(words[2]);
            } else if (directive == "fps") {
                want(1);
                scenario.fps = std::stod(words[1]);
            } else if (directive == "captured_ns") {
                want(1);
                scenario.captured_ns = std::stoll(words[1]);
            } else if (directive == "captured_unix_ns") {
                want(1);
                scenario.captured_unix_ns = std::stoll(words[1]);
            } else if (directive == "emitted_unix_ns") {
                want(1);
                scenario.emitted_unix_ns = std::stoll(words[1]);
            } else if (directive == "missing") {
                scenario.missing.assign(words.begin() + 1, words.end());
            } else if (directive == "label") {
                want(2);
                scenario.labels[std::stoi(words[1])] = words[2];
            } else if (directive == "det") {
                want(7);
                Detection detection;
                detection.index = std::stoi(words[1]);
                detection.class_id = std::stoi(words[2]);
                detection.score = std::stof(words[3]);
                detection.x1 = std::stof(words[4]);
                detection.y1 = std::stof(words[5]);
                detection.x2 = std::stof(words[6]);
                detection.y2 = std::stof(words[7]);
                scenario.detections.push_back(detection);
            } else if (directive == "batch") {
                want(2);
                last_batch = words[1];
                last_width = std::stoi(words[2]);
                ObjectBatch batch;
                batch.name = last_batch;
                batch.width = last_width;
                scenario.batches[last_batch] = batch;
            } else if (directive == "row") {
                if (last_batch.empty()) throw ConfigError(where + ": `row` before any `batch`");
                want(static_cast<size_t>(last_width) + 1);
                ObjectBatch& batch = scenario.batches[last_batch];
                batch.object_indices.push_back(std::stoi(words[1]));
                for (int i = 0; i < last_width; ++i) {
                    batch.data.push_back(std::stof(words[static_cast<size_t>(i) + 2]));
                }
            } else if (directive == "field") {
                if (words.size() < 3) {
                    throw ConfigError(where + ": expected `field <name> <batch>...`");
                }
                scenario.fields.emplace_back(
                    words[1], std::vector<std::string>(words.begin() + 2, words.end()));
            } else {
                throw ConfigError(where + ": unknown directive '" + directive + "'");
            }
        }
        if (scenario.name.empty()) throw ConfigError(path + ": no `scenario <name>` line");
        return scenario;
    }

}  // namespace shipinfer::parity
