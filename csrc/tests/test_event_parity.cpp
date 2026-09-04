// The perception-event gate: this plane's JSON against the line the Python plane wrote.
//
// A BYTE compare, and it has to be: the event is a wire format a deployed `motservice`
// parses, key order is part of the contract (`docs/design/event-schema.md`), and this plane
// writes JSON without ever parsing it -- vendoring a parser for one format is refused by the
// ponytail principle. So "the same event" means the same line, and the scenario restricts
// itself to numbers both languages spell identically (`event_scenario.py::FLOATS`).
//
// There is no known-divergence register here and there is not meant to be one: the ingest
// register was emptied by converging the planes (P6-D), the queue seam never needed one, and
// a difference in a wire format is a bug in one of the two writers.
//
// Offline: g++ alone, no CUDA, no GStreamer.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "shipinfer/core/events/json.h"
#include "shipinfer/core/events/schema.h"
#include "shipinfer/core/types.h"
#include "tests/event_scenario.h"
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

    // The scenario states both clocks, so what is compared is the SCHEMA -- key order, types,
    // null handling -- and not two wall clocks. `build()` is exercised by its own check below.
    events::PerceptionEvent event_of(const EventScenario& scenario) {
        events::PerceptionEvent event;
        event.camera_id = scenario.camera;
        event.frame_id = scenario.frame;
        event.source_id = scenario.source;
        event.objects = scenario.objects;
        event.img_width = scenario.width;
        event.img_height = scenario.height;
        event.img_fps = static_cast<int64_t>(std::llrint(scenario.fps));
        event.captured_unix_ns = scenario.captured_unix_ns;
        event.emitted_unix_ns = scenario.emitted_unix_ns;
        event.latency_us =
            scenario.captured_ns
                ? std::max<int64_t>(0, (scenario.emitted_unix_ns - scenario.captured_ns) / 1000)
                : 0;
        event.missing_stages = scenario.missing;
        // Derived HERE when the scenario named the enum, so this plane's own vocabulary is
        // what reaches the golden -- the whole point of the `finished` directive.
        event.reason = scenario.finished ? to_string(*scenario.finished) : scenario.reason;
        return event;
    }

    void test_this_plane_matches_the_golden(const std::string& name) {
        const EventScenario scenario =
            load_event_scenario(resolve("scenarios/events/" + name + ".scn"));
        const std::vector<std::string> golden =
            read_lines(resolve("golden/events/" + name + ".jsonl"));
        check(golden.size() == 1,
              name + ": the golden is one line, got " + std::to_string(golden.size()));
        const std::string mine = event_of(scenario).to_json();
        if (golden.size() == 1 && golden[0] != mine) {
            // The first differing column, because a 1200-byte diff of two JSON lines is
            // unreadable and the column is what says which key drifted.
            size_t at = 0;
            while (at < golden[0].size() && at < mine.size() && golden[0][at] == mine[at]) ++at;
            std::printf("FAIL: %s: differs at column %zu\n  python: ...%s\n  cpp   : ...%s\n",
                        name.c_str(), at, golden[0].substr(at > 40 ? at - 40 : 0, 90).c_str(),
                        mine.substr(at > 40 ? at - 40 : 0, 90).c_str());
            ++failures;
            ++checks;
            return;
        }
        check(true, name + ": byte-identical to the golden (" + std::to_string(mine.size()) +
                        " bytes)");
    }

    void test_build_stamps_the_clocks_it_is_given() {
        // `build()` is what production calls, so its two derived fields are checked here
        // rather than being left to a path the gate cannot compare.
        const events::PerceptionEvent event =
            events::build("cam0", 7, "shard-1", {}, 1920, 1080, 19.6, 1'000'000'000LL,
                          1'700'000'000'000'000'000LL, {"mtmc"}, "timeout", 3'000'000'000LL,
                          1'700'000'002'000'000'000LL);
        check(event.img_fps == 20, "19.6 fps rounds to 20, as `round(fps)` does");
        // The HALF-INTEGER boundary, which is the only place the two rounding rules differ
        // and the only place a real camera rate lands on: `round(12.5)` is 12 in Python
        // (half to even) and 13 under `llround` (half away from zero). 6.25 and 7.5 are
        // ordinary configured rates, so this is not a corner case.
        for (const auto& [rate, want] : std::vector<std::pair<double, int64_t>>{
                 {12.5, 12}, {18.5, 18}, {0.5, 0}, {2.5, 2}, {7.5, 8}}) {
            const events::PerceptionEvent at =
                events::build("cam0", 0, "s", {}, 0, 0, rate, 0, 0, {}, "complete", 1, 1);
            check(at.img_fps == want, "fps " + std::to_string(rate) + " -> " +
                                          std::to_string(at.img_fps) + ", python says " +
                                          std::to_string(want));
        }
        check(event.latency_us == 2'000'000,
              "latency is capture-to-emission on the MONOTONIC pair, in microseconds; got " +
                  std::to_string(event.latency_us));
        check(event.emitted_unix_ns == 1'700'000'002'000'000'000LL,
              "the wall-clock stamp is the one it was given");
        check(event.is_partial(), "a missing stage makes the event partial");
    }

    void test_a_malformed_scenario_is_refused_naming_the_line() {
        const std::string path = "/tmp/shipinfer_event_parity_probe.scn";
        {
            std::ofstream out(path);
            out << "scenario bad\ncamera cam0\nsprint cam0\n";
        }
        std::string message;
        try {
            (void)load_event_scenario(path);
        } catch (const ConfigError& error) {
            message = error.what();
        }
        check(message.find("unknown directive 'sprint'") != std::string::npos &&
                  message.find(":3") != std::string::npos,
              "a malformed scenario names the directive AND the line: " + message);
        std::remove(path.c_str());
    }

    void test_the_number_writer_spells_what_python_spells() {
        // The one place the two languages disagree by default: `std::to_chars` writes `1`
        // where Python writes `1.0`, and `std::to_string` writes `0.500000` for 0.5.
        const std::vector<std::pair<double, std::string>> cases = {
            {1.0, "1.0"},
            {0.5, "0.5"},
            {0.25, "0.25"},
            {0.1, "0.1"},
            {0.0, "0.0"},
            {-0.0, "-0.0"},
            {3.0, "3.0"},
            {123.456, "123.456"},
            {1e20, "1e+20"},
            // Both sides of both boundaries in Python's rule -- scientific only when the
            // decimal exponent is < -4 or >= 16. `to_chars` alone writes `1e+05` for the
            // first of these, which is where the two planes silently diverged.
            {100000.0, "100000.0"},
            {1e7, "10000000.0"},
            {1e15, "1000000000000000.0"},
            {1e16, "1e+16"},
            {1e-4, "0.0001"},
            {1e-5, "1e-05"},
            {1.5e16, "1.5e+16"},
            {123456789012345.0, "123456789012345.0"},
        };
        for (const auto& [value, spelling] : cases) {
            check(
                events::json_number(value) == spelling,
                "json_number -> '" + events::json_number(value) + "', want '" + spelling + "'");
        }
        // A bare `inf` or `nan` is not valid JSON, so one NaN out of an fp16 engine would make
        // a strict consumer reject the whole line rather than one field. Refused instead.
        for (const double hostile : {std::nan(""), HUGE_VAL, -HUGE_VAL}) {
            bool refused = false;
            try {
                (void)events::json_number(hostile);
            } catch (const ConfigError&) {
                refused = true;
            }
            check(refused, "a non-finite number is refused rather than written");
        }
        // `json.dumps` escapes every non-ASCII code point, so a Vietnamese camera id is
        // ordinary input rather than a reason to throw out of a worker thread's sink.
        check(events::json_string("c\xe1\xba\xa7u-01") == "\"c\\u1ea7u-01\"",
              "a non-ASCII camera id escapes as a \\uXXXX pair: " +
                  events::json_string("c\xe1\xba\xa7u-01"));
        check(events::json_string("a\nb") == "\"a\\nb\"", "a newline is a short escape");
    }

}  // namespace

int main() {
    try {
        test_the_number_writer_spells_what_python_spells();
        test_build_stamps_the_clocks_it_is_given();
        test_a_malformed_scenario_is_refused_naming_the_line();
        test_this_plane_matches_the_golden("mixed_frame");
        test_this_plane_matches_the_golden("empty_frame");
        test_this_plane_matches_the_golden("vietnamese_camera");
        test_this_plane_matches_the_golden("evicted_frame");
    } catch (const std::exception& error) {
        // A missing golden is a HARD failure, never a skip: a gate that fails open is worse
        // than no gate, because it reads as evidence.
        std::fprintf(stderr, "FAIL: the event parity harness could not run: %s\n",
                     error.what());
        ++failures;
    }
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
