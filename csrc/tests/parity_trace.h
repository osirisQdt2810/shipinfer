// The parity trace, written the way `benchmarks/parity/trace.py` writes it.
//
// Canonical by construction: fixed key order, integers only, printable ASCII with no quote
// and no backslash. That alphabet is the whole point — neither plane needs an escaper, so
// neither can grow one that disagrees with the other's, and this header carries no JSON
// library for a format that is forty lines wide (the ponytail principle refuses one).
//
// `parse_line` reads THIS format and no other. It is not a JSON parser and must never
// become one: it is sound only because the writer above it forbids escapes and nesting.
//
// Test support, not part of the tree under test: header-only, so it adds nothing to any link
// line and nothing to the include closure `scripts/build_csrc.py` walks.
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"

namespace shipinfer::parity {

    inline constexpr int kSchemaVersion = 1;

    // Every record kind, and the names of the numbers and words it carries — the same table
    // as `FIELDS` in `benchmarks/parity/trace.py` and as the one in `README.md`. Spelled
    // twice in two languages on purpose, and `test_parity_ingest.py` fails if they drift:
    // a field name is what a failing gate says, and a message naming "n[6]" is not one.
    struct FieldNames {
        std::vector<std::string> numbers;
        std::vector<std::string> text;
    };

    inline const std::map<std::string, FieldNames>& kFields() {
        static const std::map<std::string, FieldNames> table = {
            {"source_open", {{"attempt"}, {"outcome"}}},
            {"source_read", {{"index"}, {"outcome"}}},
            {"source_close", {{"index"}, {}}},
            {"frame", {{"frame_id"}, {}}},
            {"drop", {{}, {"reason"}}},
            {"retry", {{"attempt", "peek_us"}, {}}},
            {"state", {{}, {"from", "to"}}},
            {"health",
             {{"frames_read", "frames_published", "frames_dropped", "empty_reads", "connects",
               "connect_failures", "consecutive_failures"},
              {"state", "last_error"}}},
            {"stop", {{"abandoned"}, {}}},
            {"end", {{"cameras", "frames_read", "frames_published", "frames_dropped"}, {}}},
            // The queue seam. Fleet-level, every one of them, and the item's camera travels
            // in the words: a scheduling run is single-threaded with no clock in it, so WHICH
            // camera comes out next is the invariant rather than the nondeterminism, and only
            // the ungrouped sequence can compare that.
            {"qput", {{"rows", "depth"}, {"camera", "status"}}},
            {"qbatch", {{"items", "rows"}, {}}},
            {"qserved", {{"rows"}, {"camera"}}},
            {"qdrop", {{}, {"camera", "reason"}}},
            {"qstats",
             {{"accepted", "rejected", "evicted", "expired", "depth", "capacity"}, {}}},
            {"qcam", {{"depth", "rejected", "evicted", "expired"}, {"camera"}}},
        };
        return table;
    }

    // `trace.py`'s FLEET_KINDS. A set rather than a chain of `==`, so the two spellings can
    // be compared by a test instead of by whoever next adds a kind.
    inline const std::set<std::string>& kFleetKinds() {
        static const std::set<std::string> kinds = {"stop",    "end",   "qput",   "qbatch",
                                                    "qserved", "qdrop", "qstats", "qcam"};
        return kinds;
    }

    struct ParityRecord {
        std::string kind;
        std::string camera;
        std::vector<int64_t> numbers;
        std::vector<std::string> text;

        bool operator==(const ParityRecord& other) const {
            return kind == other.kind && camera == other.camera && numbers == other.numbers &&
                   text == other.text;
        }

        // `kind camera field=value …` — what a failing check prints.
        std::string render() const {
            const auto found = kFields().find(kind);
            std::string out = kind + " " + (camera.empty() ? "<fleet>" : camera);
            if (found == kFields().end()) return out;
            for (size_t i = 0; i < numbers.size() && i < found->second.numbers.size(); ++i) {
                out += " " + found->second.numbers[i] + "=" + std::to_string(numbers[i]);
            }
            for (size_t i = 0; i < text.size() && i < found->second.text.size(); ++i) {
                out += " " + found->second.text[i] + "='" + text[i] + "'";
            }
            return out;
        }
    };

    // Refuse anything the two writers would have to escape differently.
    inline const std::string& checked_word(const std::string& value, const std::string& field) {
        for (char c : value) {
            if (c < 0x20 || c > 0x7E || c == '"' || c == '\\') {
                throw ConfigError("parity trace " + field + " '" + value +
                                  "' carries a character the canonical writer refuses: "
                                  "records are printable ASCII without a quote or backslash");
            }
        }
        return value;
    }

    inline std::string to_line(const ParityRecord& record) {
        std::string line = "{\"kind\":\"" + checked_word(record.kind, "kind") +
                           "\",\"camera\":\"" + checked_word(record.camera, "camera") +
                           "\",\"n\":[";
        for (size_t i = 0; i < record.numbers.size(); ++i) {
            line += (i ? "," : "") + std::to_string(record.numbers[i]);
        }
        line += "],\"t\":[";
        for (size_t i = 0; i < record.text.size(); ++i) {
            line += std::string(i ? "," : "") + "\"" +
                    checked_word(record.text[i], record.kind + ".text") + "\"";
        }
        return line + "]}";
    }

    // Collects records from every actor thread and emits one deterministic file.
    //
    // Grouped by camera on output, fleet-level records last. That grouping is what makes a
    // line-by-line diff against the golden valid at all: which camera's thread reached its
    // first read first is scheduler nondeterminism and is never a parity property.
    class ParityTraceWriter {
      public:
        void header(const std::string& scenario, const std::string& plane) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!header_.empty()) throw ConfigError("parity trace header written twice");
            header_ = "{\"schema\":" + std::to_string(kSchemaVersion) + ",\"scenario\":\"" +
                      checked_word(scenario, "scenario") + "\",\"plane\":\"" +
                      checked_word(plane, "plane") + "\"}";
        }

        // Throws ConfigError on an unknown kind or the wrong number of fields for it: a plane
        // that emits the wrong shape fails where it wrote, not as an unreadable diff.
        void record(const std::string& kind, const std::string& camera,
                    std::vector<int64_t> numbers = {}, std::vector<std::string> text = {}) {
            const auto found = kFields().find(kind);
            if (found == kFields().end()) {
                throw ConfigError("unknown parity record kind '" + kind + "'");
            }
            const bool fleet = kFleetKinds().count(kind) == 1;
            if (fleet == !camera.empty()) {
                throw ConfigError("parity record '" + kind +
                                  "': fleet-level kinds carry no camera and every other kind "
                                  "must carry one; got '" +
                                  camera + "'");
            }
            if (numbers.size() != found->second.numbers.size() ||
                text.size() != found->second.text.size()) {
                throw ConfigError("parity record '" + kind + "' has the wrong field count");
            }
            // Checked here and not only at `to_line`, so a word the canonical form cannot
            // spell is refused where it was written rather than at the end of a whole run.
            checked_word(camera, "camera");
            for (const std::string& word : text) checked_word(word, kind + ".text");
            std::lock_guard<std::mutex> lock(mutex_);
            records_.push_back({kind, camera, std::move(numbers), std::move(text)});
        }

        std::vector<std::string> lines() const {
            std::lock_guard<std::mutex> lock(mutex_);
            if (header_.empty()) throw ConfigError("parity trace has no header");
            std::set<std::string> cameras;
            for (const ParityRecord& entry : records_) {
                if (!entry.camera.empty()) cameras.insert(entry.camera);
            }
            std::vector<std::string> out{header_};
            for (const std::string& camera : cameras) {
                for (const ParityRecord& entry : records_) {
                    if (entry.camera == camera) out.push_back(to_line(entry));
                }
            }
            for (const ParityRecord& entry : records_) {
                if (entry.camera.empty()) out.push_back(to_line(entry));
            }
            return out;
        }

      private:
        mutable std::mutex mutex_;
        std::string header_;
        std::vector<ParityRecord> records_;
    };

    // -- reading the golden -----------------------------------------------------------------

    inline std::string quoted_field(const std::string& line, const std::string& key) {
        const std::string needle = "\"" + key + "\":\"";
        const size_t at = line.find(needle);
        if (at == std::string::npos) throw ConfigError("no '" + key + "' in: " + line);
        const size_t start = at + needle.size();
        const size_t end = line.find('"', start);
        if (end == std::string::npos) throw ConfigError("unterminated '" + key + "': " + line);
        return line.substr(start, end - start);
    }

    inline ParityRecord parse_line(const std::string& line) {
        ParityRecord record;
        record.kind = quoted_field(line, "kind");
        record.camera = quoted_field(line, "camera");
        const size_t numbers_at = line.find("\"n\":[");
        const size_t text_at = line.find("\"t\":[");
        if (numbers_at == std::string::npos || text_at == std::string::npos) {
            throw ConfigError("no 'n' or 't' array in: " + line);
        }
        std::string field;
        for (size_t i = numbers_at + 5; i < line.size() && line[i] != ']'; ++i) {
            if (line[i] == ',') {
                record.numbers.push_back(std::stoll(field));
                field.clear();
            } else {
                field += line[i];
            }
        }
        if (!field.empty()) record.numbers.push_back(std::stoll(field));
        for (size_t i = text_at + 5; i < line.size() && line[i] != ']';) {
            if (line[i] != '"') {
                ++i;
                continue;
            }
            const size_t end = line.find('"', i + 1);
            if (end == std::string::npos) throw ConfigError("unterminated word in: " + line);
            record.text.push_back(line.substr(i + 1, end - i - 1));
            i = end + 1;
        }
        return record;
    }

}  // namespace shipinfer::parity
