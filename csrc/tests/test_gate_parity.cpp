// The gate, held to the REFERENCE's answer rather than to my own expectations.
//
// `benchmarks/parity/scenarios/gate/basic.txt` is read by both planes; the golden beside it is
// what `shipvision.mtmc.gating.ObservationGate` answered. What this pins is the ORDER of the
// two gates -- height, then age over qualifying frames only -- which is the property a port
// gets wrong silently: swap them and a track banks age while unusably small, then enters the
// matrix on its first usable frame with the gate already satisfied.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/mtmc/gate.h"
#include "tests/parity_files.h"

namespace {

    using namespace shipinfer;
    using shipinfer::mtmc::ClusterObservation;
    using shipinfer::mtmc::ObservationGate;
    using shipinfer::mtmc::TrackKey;
    using shipinfer::parity::resolve;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    struct Scenario {
        std::string name;
        ObservationGate::Options options;
        std::vector<std::vector<ClusterObservation>> instants;
    };

    // `<camera>#<track>:<box_height_px>` -- the scenario file's header states the format.
    ClusterObservation parse_field(const std::string& field) {
        const size_t hash = field.find('#');
        const size_t colon = field.find(':', hash);
        if (hash == std::string::npos || colon == std::string::npos) {
            throw ConfigError("malformed observation '" + field + "'");
        }
        ClusterObservation out;
        out.key = TrackKey{field.substr(0, hash),
                           std::stoll(field.substr(hash + 1, colon - hash - 1))};
        out.embedding = {1.0f, 0.0f};
        out.box[0] = 0.0f;
        out.box[1] = 0.0f;
        out.box[2] = 40.0f;
        out.box[3] = std::stof(field.substr(colon + 1));
        out.frame_width = 1920;
        out.frame_height = 1080;
        return out;
    }

    std::vector<Scenario> read_scenarios(const std::string& path) {
        std::vector<Scenario> scenarios;
        for (const std::string& line : parity::read_lines(path, false)) {
            std::istringstream stream(line);
            std::string head;
            stream >> head;
            if (head == "scenario") {
                Scenario scenario;
                stream >> scenario.name >> scenario.options.min_hits >>
                    scenario.options.min_height_fraction;
                scenarios.push_back(std::move(scenario));
                continue;
            }
            if (head != "instant") throw ConfigError("unknown line '" + line + "'");
            if (scenarios.empty()) throw ConfigError("an `instant` before any `scenario`");
            std::vector<ClusterObservation> instant;
            std::string field;
            while (stream >> field) instant.push_back(parse_field(field));
            scenarios.back().instants.push_back(std::move(instant));
        }
        return scenarios;
    }

    void the_port_admits_what_the_reference_admitted() {
        const std::vector<Scenario> scenarios =
            read_scenarios(resolve("scenarios/gate/basic.txt"));
        const std::vector<std::string> golden =
            parity::read_lines(resolve("golden/gate/basic.txt"), false);

        std::vector<std::string> written;
        for (const Scenario& scenario : scenarios) {
            ObservationGate gate(scenario.options);
            written.push_back("scenario " + scenario.name);
            for (const std::vector<ClusterObservation>& instant : scenario.instants) {
                std::string names;
                for (const ClusterObservation& admitted : gate.filter(instant)) {
                    if (!names.empty()) names += " ";
                    names += admitted.key.str();
                }
                written.push_back(names.empty() ? "admitted" : "admitted " + names);
            }
            written.push_back("held " + std::to_string(gate.size()));
        }

        check(!scenarios.empty(), "the scenario file holds scenarios");
        check(written.size() == golden.size(),
              "the port answered " + std::to_string(written.size()) + " lines and the " +
                  "reference " + std::to_string(golden.size()));
        const size_t common = std::min(written.size(), golden.size());
        size_t agreed = 0;
        for (size_t i = 0; i < common; ++i) {
            if (written[i] == golden[i]) {
                ++agreed;
                continue;
            }
            check(false, "line " + std::to_string(i + 1) + " differs");
            std::printf("  reference: %s\n  port:      %s\n", golden[i].c_str(),
                        written[i].c_str());
        }
        check(agreed == common && common == golden.size(),
              "every line of the reference's answer is the port's answer too (" +
                  std::to_string(agreed) + "/" + std::to_string(golden.size()) + ")");
    }

    void the_golden_names_its_own_emitter() {
        const std::vector<std::string> all =
            parity::read_lines(resolve("golden/gate/basic.txt"), true);
        bool named = false;
        for (const std::string& line : all) {
            if (line.find("emit_parity_golden.py --kind gate") != std::string::npos)
                named = true;
        }

        check(named, "the golden carries the command that emits it");
    }

}  // namespace

int main() {
    try {
        the_port_admits_what_the_reference_admitted();
        the_golden_names_its_own_emitter();
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        ++failures;
        ++checks;
    }
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
