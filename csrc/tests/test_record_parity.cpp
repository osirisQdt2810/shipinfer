// The record gate: this plane's `build_records` against the line the Python plane's built.
//
// `test_event_parity` compares the two JSON WRITERS on records a scenario states. This one
// compares the two BUILDERS -- the translation units production runs -- by taking one
// description of what the graph left behind and letting each plane turn it into records
// itself. P5-A-ALLOC's second half, and the seam it needed is the field map, which the
// resolved chain plan (ADR-020) now carries.
//
// A byte compare, for the event seam's reason: the event is a wire format a deployed
// `motservice` parses, and key order is part of the contract.
//
// Offline: g++ alone, no CUDA, no GStreamer.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/events/records.h"
#include "tests/parity_files.h"
#include "tests/record_scenario.h"

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
    const std::vector<std::string> kScenarios = {"scattered_frame"};

    // Scenarios with NO golden, because what they describe is a frame both planes must
    // REFUSE -- the half a byte compare cannot express.
    const std::vector<std::string> kRefused = {"contested_row_refused"};

    pipeline::events::FieldMap field_map_of(const RecordScenario& scenario) {
        pipeline::events::FieldMap fields;
        for (const auto& [name, candidates] : scenario.fields) fields[name] = candidates;
        return fields;
    }

    // Everything except the records themselves comes from the scenario, so what is compared
    // is the BUILDER and not two wall clocks -- the same split the event gate makes.
    events::PerceptionEvent event_from(const RecordScenario& scenario) {
        EmissionInputs inputs;
        inputs.tag.camera_id = scenario.camera;
        inputs.tag.frame_id = scenario.frame;
        inputs.width = static_cast<int>(scenario.width);
        inputs.height = static_cast<int>(scenario.height);
        inputs.fps = static_cast<float>(scenario.fps);
        inputs.detections = scenario.detections;
        inputs.batches = scenario.batches;

        events::PerceptionEvent event;
        event.camera_id = scenario.camera;
        event.frame_id = scenario.frame;
        event.source_id = scenario.source;
        event.objects =
            pipeline::events::build_records(inputs, scenario.labels, field_map_of(scenario));
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
        event.reason = scenario.finished ? to_string(*scenario.finished) : scenario.reason;
        return event;
    }

    void matches_the_golden(const std::string& name) {
        const RecordScenario scenario =
            load_record_scenario(resolve("scenarios/records/" + name + ".scn"));
        const std::vector<std::string> golden =
            read_lines(resolve("golden/records/" + name + ".jsonl"));
        check(golden.size() == 1,
              name + ": the golden is one line, got " + std::to_string(golden.size()));
        const std::string mine = event_from(scenario).to_json();
        if (golden.size() == 1 && golden[0] != mine) {
            // The first differing column: a 1200-byte diff of two JSON lines is unreadable,
            // and the column is what says which key drifted.
            size_t at = 0;
            while (at < golden[0].size() && at < mine.size() && golden[0][at] == mine[at]) ++at;
            std::printf("FAIL: %s: differs at column %zu\n  python: ...%s\n  cpp   : ...%s\n",
                        name.c_str(), at, golden[0].substr(at, 90).c_str(),
                        mine.substr(at, 90).c_str());
            ++failures;
        }
        ++checks;
    }

    // The refusal, which is the decision this seam settles: two batches covering one
    // detection is a chain-file error the project already had a message for
    // (`PoolEmbed._scatter`, `ChainWalk.inbound`), and picking one by declaration order
    // would attach an appearance vector to the wrong reasoning.
    void a_contested_row_is_refused(const std::string& name) {
        const RecordScenario scenario =
            load_record_scenario(resolve("scenarios/records/" + name + ".scn"));
        EmissionInputs inputs;
        inputs.tag.camera_id = scenario.camera;
        inputs.tag.frame_id = scenario.frame;
        inputs.detections = scenario.detections;
        inputs.batches = scenario.batches;

        std::string message;
        try {
            pipeline::events::build_records(inputs, scenario.labels, field_map_of(scenario));
        } catch (const InferenceError& error) {
            message = error.what();
        }
        check(!message.empty(), name + ": a contested row is refused, not picked");
        // The message has to name the row and the field, because an operator reading it has
        // to find the two slots in the chain file.
        check(message.find("row 0") != std::string::npos, name + ": it names the row");
        check(message.find("embedding") != std::string::npos, name + ": and the field");
        check(message.find("classes") != std::string::npos,
              name + ": and points at `params: classes:`, which is the fix");
    }

    // The parts of the builder a collision case does not reach, asserted rather than left to
    // the byte compare: an unnamed class, a row index past the detections, an empty batch.
    void the_edges_of_the_scatter() {
        const RecordScenario scenario =
            load_record_scenario(resolve("scenarios/records/scattered_frame.scn"));
        EmissionInputs inputs;
        inputs.tag.camera_id = scenario.camera;
        inputs.tag.frame_id = scenario.frame;
        inputs.detections = scenario.detections;
        inputs.batches = scenario.batches;
        const std::vector<events::ObjectRecord> records =
            pipeline::events::build_records(inputs, scenario.labels, field_map_of(scenario));

        check(records.size() == 3, "three detections, three records");
        check(records[0].class_name == "ship" && records[1].class_name == "person",
              "the label table resolves what it names");
        check(records[2].class_name == pipeline::events::kUnknownLabel,
              "and a class id it does not name is `unknown`, not silently relabelled");
        check(records[0].det_id == "cau-01_41_0", "`<camera>_<frame>_<index>`");
        check(records[0].mask_area_px && *records[0].mask_area_px == 100.0, "the mask area");
        check(records[0].ship_id && *records[0].ship_id == 4, "the gallery id");
        check(records[0].similarity && *records[0].similarity == 0.75, "the similarity");
        check(records[1].embedding.empty(),
              "an EMPTY batch fills nothing rather than becoming a null field");
        check(!records[2].mask_area_px.has_value(),
              "and a row no batch mentions carries no field at all");
    }

}  // namespace

int main() {
    try {
        for (const std::string& name : kScenarios) matches_the_golden(name);
        for (const std::string& name : kRefused) a_contested_row_is_refused(name);
        the_edges_of_the_scatter();
    } catch (const std::exception& error) {
        std::printf("FAIL: unexpected exception: %s\n", error.what());
        ++failures;
    }
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
