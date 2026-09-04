// One perception event's inputs — `benchmarks/parity/event_scenario.py`, directive for
// directive. Line-oriented so this half needs no JSON parser and no YAML.
#pragma once

#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "shipinfer/core/events/schema.h"
#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/graph/emission.h"
#include "tests/parity_files.h"

namespace shipinfer::parity {

    struct EventScenario {
        std::string name;
        std::string camera = "cam0";
        int64_t frame = 0;
        std::string source = "shard-0";
        int64_t width = 0;
        int64_t height = 0;
        double fps = 0.0;
        int64_t captured_ns = 0;
        int64_t captured_unix_ns = 0;
        int64_t emitted_unix_ns = 0;
        std::vector<std::string> missing;
        std::string reason = "complete";
        //: Set when the scenario says `finished <reason>` rather than `reason <word>`, so the
        //: word comes from `to_string(FinishReason)` on this plane and from the collector's
        //: own constants on the other.
        std::optional<FinishReason> finished;
        std::vector<events::ObjectRecord> objects;
    };

    namespace detail {

        inline FinishReason finish_reason_of(const std::string& word,
                                             const std::string& where) {
            if (word == "complete") return FinishReason::Complete;
            if (word == "incomplete") return FinishReason::Incomplete;
            if (word == "timeout") return FinishReason::Timeout;
            if (word == "shutdown") return FinishReason::Shutdown;
            if (word == "evicted") return FinishReason::Evicted;
            throw ConfigError(where + ": '" + word + "' is not a FinishReason");
        }

        inline double number(const std::string& word, const std::string& where) {
            try {
                return std::stod(word);
            } catch (const std::exception&) {
                throw ConfigError(where + ": '" + word + "' is not a number");
            }
        }

        // `<det_id> <score> <x1> <y1> <x2> <y2>` then any of the keyword groups, in any
        // order -- the Python parser's shape, so a scenario that loads there loads here.
        inline events::ObjectRecord object(const std::string& class_name,
                                           std::vector<std::string> words,
                                           const std::string& where) {
            if (words.size() < 6) {
                throw ConfigError(where + ": " + class_name +
                                  " needs a det_id, a score and 4 bounds");
            }
            events::ObjectRecord record;
            record.class_name = class_name;
            record.det_id = words[0];
            record.score = number(words[1], where);
            for (int i = 0; i < 4; ++i) record.bbox[i] = number(words[2 + i], where);
            for (size_t i = 6; i < words.size();) {
                const std::string& key = words[i];
                if (key == "emb") {
                    for (size_t j = i + 1; j < words.size(); ++j) {
                        record.embedding.push_back(number(words[j], where));
                    }
                    break;
                }
                if (key == "track") {
                    record.track_id = static_cast<int64_t>(std::stoll(words.at(i + 1)));
                    record.track_state = words.at(i + 2);
                    i += 3;
                } else if (key == "global") {
                    record.global_id = static_cast<int64_t>(std::stoll(words.at(i + 1)));
                    i += 2;
                } else if (key == "ship") {
                    record.ship_id = static_cast<int64_t>(std::stoll(words.at(i + 1)));
                    record.similarity = number(words.at(i + 2), where);
                    i += 3;
                } else if (key == "mask") {
                    record.mask_area_px = number(words.at(i + 1), where);
                    i += 2;
                } else {
                    throw ConfigError(where + ": unknown object keyword '" + key + "'");
                }
            }
            return record;
        }

    }  // namespace detail

    inline EventScenario load_event_scenario(const std::string& path) {
        EventScenario scenario;
        bool named = false;
        int number = 0;
        for (const std::string& raw : read_lines_keeping_blanks(path)) {
            ++number;
            const std::string where = path + ":" + std::to_string(number);
            std::istringstream words(raw.substr(0, raw.find('#')));
            std::string directive;
            if (!(words >> directive)) continue;
            std::vector<std::string> rest;
            for (std::string word; words >> word;) rest.push_back(word);
            const auto first = [&rest, &where](const char* what) -> const std::string& {
                if (rest.empty()) throw ConfigError(where + ": " + what + " takes a value");
                return rest[0];
            };
            if (directive == "person" || directive == "ship") {
                scenario.objects.push_back(detail::object(directive, rest, where));
            } else if (directive == "scenario") {
                scenario.name = first("scenario");
                named = true;
            } else if (directive == "camera") {
                scenario.camera = first("camera");
            } else if (directive == "source") {
                scenario.source = first("source");
            } else if (directive == "reason") {
                scenario.reason = first("reason");
            } else if (directive == "finished") {
                // The ENUM, so this plane derives the word itself. `reason` states the string
                // and is echoed unchanged, which is why the gate could not see this plane
                // writing `failed` where Python writes `evicted`.
                scenario.finished = detail::finish_reason_of(first("finished"), where);
            } else if (directive == "frame") {
                scenario.frame = std::stoll(first("frame"));
            } else if (directive == "size") {
                scenario.width = std::stoll(first("size"));
                scenario.height = std::stoll(rest.at(1));
            } else if (directive == "fps") {
                scenario.fps = detail::number(first("fps"), where);
            } else if (directive == "captured_ns") {
                scenario.captured_ns = std::stoll(first("captured_ns"));
            } else if (directive == "captured_unix_ns") {
                scenario.captured_unix_ns = std::stoll(first("captured_unix_ns"));
            } else if (directive == "emitted_unix_ns") {
                scenario.emitted_unix_ns = std::stoll(first("emitted_unix_ns"));
            } else if (directive == "missing") {
                scenario.missing = rest;
            } else {
                throw ConfigError(where + ": unknown directive '" + directive + "'");
            }
        }
        if (!named) throw ConfigError(path + ": no scenario line");
        return scenario;
    }

}  // namespace shipinfer::parity
