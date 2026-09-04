// The queue-seam parity gate, C++ half: this plane against the golden the Python plane wrote.
//
// One ordered sequence, not one per camera. The ingest gate groups by camera because thread
// interleaving is nondeterministic; a queue run is single-threaded with no clock in it, so
// WHICH camera's item comes out next is the invariant and the whole trace is compared in
// order. There is no known-divergence register here and there is not meant to be one: the
// ingest register was emptied by converging the planes (P6-D), and a queue difference is a
// bug until somebody argues otherwise in `benchmarks/parity/known.py`.
//
// Offline: g++ alone, no CUDA, no GStreamer.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/scheduling/queues/base.h"
#include "shipinfer/scheduling/queues/fair.h"
#include "shipinfer/scheduling/queues/fifo.h"
#include "tests/parity_files.h"
#include "tests/parity_trace.h"
#include "tests/queue_scenario.h"

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

    // The item contract `queues/base.h` states, and nothing more: a parity run compares
    // bookkeeping, so the payload is a row count.
    struct QItem {
        std::string camera_id;
        size_t row_count = 1;
        int band = Priority::Normal;
        bool is_expired = false;

        std::string camera() const { return camera_id; }
        size_t rows() const { return row_count; }
        int priority() const { return band; }
        bool expired(int64_t) const { return is_expired; }
    };

    const char* to_word(PutStatus status) {
        switch (status) {
            case PutStatus::Accepted:
                return "accepted";
            case PutStatus::Rejected:
                return "rejected";
            case PutStatus::Closed:
                return "closed";
        }
        return "?";
    }

    // -- driving one scenario ---------------------------------------------------------------

    template <typename Queue>
    std::vector<std::string> drive(const QueueScenario& scenario, Queue& queue,
                                   std::vector<std::pair<QItem, DropReason>>& drops) {
        ParityTraceWriter writer;
        writer.header(scenario.name, "cpp");
        const BatchWindow window(scenario.max_batch_size, scenario.max_delay_us);
        size_t reported = 0;
        auto flush_drops = [&]() {
            for (; reported < drops.size(); ++reported) {
                writer.record(
                    "qdrop", "", {},
                    {drops[reported].first.camera(), to_string(drops[reported].second)});
            }
        };
        for (size_t index = 0; index < scenario.ops.size(); ++index) {
            const QueueOp& op = scenario.ops[index];
            if (op.verb == "put") {
                QItem item{op.camera, op.rows, op.priority, op.expired};
                const PutStatus status = queue.put(std::move(item));
                writer.record(
                    "qput", "",
                    {static_cast<int64_t>(op.rows), static_cast<int64_t>(queue.depth())},
                    {op.camera, to_word(status)});
                flush_drops();
            } else if (op.verb == "take") {
                if (queue.depth() == 0 && !queue.is_closed()) {
                    throw ConfigError(scenario.name + ": operation " +
                                      std::to_string(index + 1) +
                                      " takes from an open empty queue, where get_batch "
                                      "blocks by contract");
                }
                const std::vector<QItem> batch = queue.get_batch(window, 0);
                int64_t rows = 0;
                for (const QItem& item : batch) rows += static_cast<int64_t>(item.rows());
                writer.record("qbatch", "", {static_cast<int64_t>(batch.size()), rows});
                for (const QItem& item : batch) {
                    writer.record("qserved", "", {static_cast<int64_t>(item.rows())},
                                  {item.camera()});
                }
                flush_drops();
            } else {
                queue.close();
                flush_drops();
            }
        }
        const QueueStats stats = queue.stats();
        writer.record(
            "qstats", "",
            {static_cast<int64_t>(stats.accepted), static_cast<int64_t>(stats.rejected),
             static_cast<int64_t>(stats.evicted), static_cast<int64_t>(stats.expired),
             static_cast<int64_t>(stats.depth), static_cast<int64_t>(stats.capacity)});
        std::set<std::string> cameras;
        for (const QueueOp& op : scenario.ops) {
            if (!op.camera.empty()) cameras.insert(op.camera);
        }
        for (const std::string& camera : cameras) {
            auto at = [&camera](const std::map<std::string, uint64_t>& counts) -> int64_t {
                const auto found = counts.find(camera);
                return found == counts.end() ? 0 : static_cast<int64_t>(found->second);
            };
            writer.record("qcam", "",
                          {at(stats.depth_by_camera), at(stats.rejected_by_camera),
                           at(stats.evicted_by_camera), at(stats.expired_by_camera)},
                          {camera});
        }
        return writer.lines();
    }

    std::vector<std::string> run_scenario(const QueueScenario& scenario) {
        // Recorded in the order the queue drops them, which is the order the Python plane
        // emits: it reads its own `close()` return and refuses more than one non-close drop
        // per operation rather than comparing an order neither queue promises.
        std::vector<std::pair<QItem, DropReason>> drops;
        auto on_drop = [&drops](QItem&& item, DropReason reason) {
            drops.emplace_back(std::move(item), reason);
        };
        if (scenario.queue == "fifo") {
            FifoQueue<QItem> queue(scenario.name, scenario.capacity, scenario.overflow, 50,
                                   true, on_drop);
            return drive(scenario, queue, drops);
        }
        FairPriorityQueue<QItem> queue(scenario.name, scenario.capacity, scenario.overflow, 50,
                                       true, on_drop);
        return drive(scenario, queue, drops);
    }

    // -- the comparison -----------------------------------------------------------------------

    void test_this_plane_matches_the_golden(const std::string& name) {
        const QueueScenario scenario =
            load_queue_scenario(resolve("scenarios/queues/" + name + ".scn"));
        const std::vector<std::string> golden =
            read_lines(resolve("golden/queues/" + name + ".jsonl"));
        const std::vector<std::string> mine = run_scenario(scenario);
        check(static_cast<int>(mine.size()) - 1 >= scenario.records_min,
              name + ": produced at least the " + std::to_string(scenario.records_min) +
                  " record(s) it promised, got " + std::to_string(mine.size() - 1));
        check(golden.at(0).find("\"scenario\":\"" + name + "\"") != std::string::npos,
              name + ": the golden names this scenario");
        check(golden.at(0).find("\"plane\":\"python\"") != std::string::npos,
              name + ": the golden was emitted by the python plane: " + golden.at(0));
        auto records = [](const std::vector<std::string>& lines) {
            std::vector<ParityRecord> parsed;
            for (size_t i = 1; i < lines.size(); ++i) parsed.push_back(parse_line(lines[i]));
            return parsed;
        };
        const std::vector<ParityRecord> theirs = records(golden);
        const std::vector<ParityRecord> ours = records(mine);
        std::string problem;
        for (size_t i = 0; i < theirs.size() && i < ours.size() && problem.empty(); ++i) {
            const std::vector<std::string> differ = differing_fields(theirs[i], ours[i]);
            if (!differ.empty()) {
                std::string named;
                for (const std::string& field : differ) {
                    named += (named.empty() ? "" : ", ") + field;
                }
                problem = "record " + std::to_string(i) + " differs on " + named + ":\n  " +
                          theirs[i].render() + "\n  " + ours[i].render();
            }
        }
        if (problem.empty() && theirs.size() != ours.size()) {
            problem = "the golden has " + std::to_string(theirs.size()) + " record(s), this " +
                      "plane produced " + std::to_string(ours.size());
        }
        // ONE check per scenario, always counted, so a green run says how many scenarios it
        // actually compared rather than being indistinguishable from one that ran over none.
        check(problem.empty(),
              name + ": " + (problem.empty() ? "matches the golden" : problem));
    }

    // A scenario that loads in one plane and not the other is the worst outcome this harness
    // has, so both parsers are held to the same refusals -- see
    // `test_parity_queues.py::TestScenarioFormat`.
    void test_a_malformed_scenario_is_refused_naming_the_line() {
        const std::string path = "/tmp/shipinfer_queue_parity_probe.scn";
        {
            std::ofstream out(path);
            out << "scenario bad\nqueue fair\ncapacity 2\noverflow reject\n"
                << "max_batch_size 2\nput cam0\nsprint cam0\n";
        }
        std::string message;
        try {
            (void)load_queue_scenario(path);
        } catch (const ConfigError& error) {
            message = error.what();
        }
        check(message.find("unknown directive 'sprint'") != std::string::npos &&
                  message.find(":7") != std::string::npos,
              "a malformed scenario is refused, naming the directive AND the line: " + message);
        std::remove(path.c_str());
    }

    void test_a_missing_golden_is_a_failure_and_never_a_skip() {
        bool refused = false;
        try {
            (void)read_lines(resolve("golden/queues/no_such_scenario.jsonl"));
        } catch (const ConfigError& error) {
            refused = std::string(error.what()).find("cannot find") != std::string::npos;
        }
        check(refused, "a missing golden names every path it tried rather than skipping");
    }

}  // namespace

int main() {
    try {
        test_a_malformed_scenario_is_refused_naming_the_line();
        test_a_missing_golden_is_a_failure_and_never_a_skip();
        test_this_plane_matches_the_golden("fair_eviction");
        test_this_plane_matches_the_golden("reject_is_the_default");
        test_this_plane_matches_the_golden("priority_lanes");
        test_this_plane_matches_the_golden("expiry_on_take");
        test_this_plane_matches_the_golden("fifo_close_drains");
    } catch (const std::exception& error) {
        // A missing golden or an unreadable scenario is a HARD failure, never a skip: a gate
        // that fails open is worse than no gate, because it reads as evidence.
        std::fprintf(stderr, "FAIL: the queue parity harness could not run: %s\n",
                     error.what());
        ++failures;
    }
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
