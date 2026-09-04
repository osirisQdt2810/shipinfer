// The queue-scenario format — `benchmarks/parity/queue_scenario.py`, directive for directive.
//
// Line-oriented so this half needs no JSON parser, and refusals name the line: a scenario
// that loads in one plane and not the other is the worst outcome this harness has.
#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/scheduling/queues/base.h"
#include "tests/parity_files.h"

namespace shipinfer::parity {

    struct QueueOp {
        std::string verb;  // put | take | close
        std::string camera;
        size_t rows = 1;
        int priority = Priority::Normal;
        bool expired = false;
    };

    struct QueueScenario {
        std::string name;
        std::string queue;
        size_t capacity = 0;
        Overflow overflow = Overflow::Reject;
        size_t max_batch_size = 0;
        int64_t max_delay_us = 0;
        int records_min = 1;
        std::vector<QueueOp> ops;
    };

    inline int priority_from(const std::string& word, const std::string& where) {
        if (word == "tracking_critical") return Priority::TrackingCritical;
        if (word == "high") return Priority::High;
        if (word == "normal") return Priority::Normal;
        if (word == "background") return Priority::Background;
        throw ConfigError(where + ": unknown priority '" + word + "'");
    }

    inline Overflow overflow_from(const std::string& word, const std::string& where) {
        if (word == "reject") return Overflow::Reject;
        if (word == "block") return Overflow::Block;
        if (word == "drop_oldest") return Overflow::DropOldest;
        throw ConfigError(where + ": unknown overflow '" + word + "'");
    }

    inline QueueScenario load_queue_scenario(const std::string& path) {
        QueueScenario scenario;
        bool named = false, queued = false, sized = false, capped = false, flowed = false;
        int number = 0;
        for (const std::string& raw : read_lines_keeping_blanks(path)) {
            ++number;
            const std::string where = path + ":" + std::to_string(number);
            std::string line = raw.substr(0, raw.find('#'));
            std::istringstream words(line);
            std::string directive;
            if (!(words >> directive)) continue;
            std::string first;
            words >> first;
            if (directive == "scenario") {
                scenario.name = first;
                named = true;
            } else if (directive == "queue") {
                if (first != "fair" && first != "fifo")
                    throw ConfigError(where + ": unknown queue '" + first + "'");
                scenario.queue = first;
                queued = true;
            } else if (directive == "capacity") {
                scenario.capacity = static_cast<size_t>(std::stoul(first));
                capped = true;
            } else if (directive == "overflow") {
                scenario.overflow = overflow_from(first, where);
                flowed = true;
            } else if (directive == "max_batch_size") {
                scenario.max_batch_size = static_cast<size_t>(std::stoul(first));
                sized = true;
            } else if (directive == "max_delay_us") {
                scenario.max_delay_us = std::stoll(first);
            } else if (directive == "records_min") {
                scenario.records_min = std::stoi(first);
            } else if (directive == "put") {
                QueueOp op;
                op.verb = "put";
                if (first.empty()) throw ConfigError(where + ": put names a camera");
                op.camera = first;
                std::string word;
                if (words >> word) {
                    op.rows = static_cast<size_t>(std::stoul(word));
                    if (op.rows < 1) throw ConfigError(where + ": put rows must be >= 1");
                }
                if (words >> word) op.priority = priority_from(word, where);
                if (words >> word) {
                    if (word != "expired")
                        throw ConfigError(where + ": the fourth word is 'expired' or nothing");
                    op.expired = true;
                }
                scenario.ops.push_back(op);
            } else if (directive == "take" || directive == "close") {
                if (!first.empty())
                    throw ConfigError(where + ": " + directive + " takes no argument");
                scenario.ops.push_back(QueueOp{directive, "", 1, Priority::Normal, false});
            } else {
                throw ConfigError(where + ": unknown directive '" + directive + "'");
            }
        }
        if (!named || !queued || !capped || !flowed || !sized) {
            throw ConfigError(path +
                              ": needs scenario, queue, capacity, overflow and "
                              "max_batch_size lines");
        }
        if (scenario.ops.empty())
            throw ConfigError(path + ": no operations, so it would compare nothing");
        return scenario;
    }

}  // namespace shipinfer::parity
