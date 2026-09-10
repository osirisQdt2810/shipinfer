// The identity port, held to the REFERENCE's answer rather than to my own expectations.
//
// `benchmarks/parity/scenarios/identity/basic.txt` is read by both planes;
// `benchmarks/parity/golden/identity/basic.txt` is what `shipvision.mtmc.identity` answered to
// it, emitted by `scripts/emit_parity_golden.py --kind identity`. This binary replays the
// scenarios through `GlobalIdAssigner` and compares line for line -- so a change to either
// implementation that moves one id fails here, which is the only thing that keeps a
// five-hundred-line port a port. `test_mtmc_identity.cpp` tests the same class against its
// own reasons; this one tests it against the other plane.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/mtmc/identity.h"
#include "tests/parity_files.h"

namespace {

    using namespace shipinfer;
    using shipinfer::mtmc::GlobalIdAssigner;
    using shipinfer::mtmc::IdentityObservation;
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

    std::vector<std::string> lines_of(const std::string& path, bool keep_comments) {
        std::ifstream file(path);
        if (!file) throw ConfigError("cannot read " + path);
        std::vector<std::string> out;
        std::string line;
        while (std::getline(file, line)) {
            while (!line.empty() && (line.back() == ' ' || line.back() == '\r'))
                line.pop_back();
            if (line.empty()) continue;
            if (!keep_comments && line[0] == '#') continue;
            out.push_back(line);
        }
        return out;
    }

    struct Instant {
        std::vector<int> labels;
        std::vector<IdentityObservation> observations;
    };

    struct Scenario {
        std::string name;
        GlobalIdAssigner::Options options;
        std::vector<Instant> instants;
    };

    // `<label>:<camera>#<track>:<x>,<y>` -- the scenario file's own header states the format.
    void parse_field(const std::string& field, Instant& instant) {
        const size_t first = field.find(':');
        const size_t second = field.find(':', first + 1);
        const size_t hash = field.find('#');
        const size_t comma = field.find(',', second);
        if (first == std::string::npos || second == std::string::npos ||
            hash == std::string::npos || comma == std::string::npos) {
            throw ConfigError("malformed observation '" + field + "'");
        }
        instant.labels.push_back(std::stoi(field.substr(0, first)));
        const std::string camera = field.substr(first + 1, hash - first - 1);
        const int64_t track = std::stoll(field.substr(hash + 1, second - hash - 1));
        const float x = std::stof(field.substr(second + 1, comma - second - 1));
        const float y = std::stof(field.substr(comma + 1));
        instant.observations.push_back(IdentityObservation{TrackKey{camera, track}, {x, y}});
    }

    std::vector<Scenario> read_scenarios(const std::string& path) {
        std::vector<Scenario> scenarios;
        for (const std::string& line : lines_of(path, false)) {
            std::istringstream stream(line);
            std::string head;
            stream >> head;
            if (head == "scenario") {
                Scenario scenario;
                stream >> scenario.name >> scenario.options.max_age >>
                    scenario.options.capacity >> scenario.options.max_tracks;
                // ON for the gate: the reference ran with it on, so a state the invariant
                // rejects would be a difference the ids might not show.
                scenario.options.validate_every_step = true;
                scenarios.push_back(std::move(scenario));
                continue;
            }
            if (head != "instant") throw ConfigError("unknown line '" + line + "'");
            if (scenarios.empty()) throw ConfigError("an `instant` before any `scenario`");
            Instant instant;
            std::string field;
            while (stream >> field) parse_field(field, instant);
            scenarios.back().instants.push_back(std::move(instant));
        }
        return scenarios;
    }

    void the_port_answers_what_the_reference_answered() {
        const std::vector<Scenario> scenarios =
            read_scenarios(resolve("scenarios/identity/basic.txt"));
        const std::vector<std::string> golden =
            lines_of(resolve("golden/identity/basic.txt"), false);

        std::vector<std::string> written;
        for (const Scenario& scenario : scenarios) {
            GlobalIdAssigner assigner(scenario.options);
            written.push_back("scenario " + scenario.name);
            for (const Instant& instant : scenario.instants) {
                const auto result = assigner.assign(instant.observations, instant.labels);
                std::string answers;
                for (const auto& [key, global_id] : result) {
                    if (!answers.empty()) answers += " ";
                    answers += key.str() + "=" + std::to_string(global_id);
                }
                written.push_back(answers.empty() ? "result" : "result " + answers);
            }
            written.push_back("issued " + std::to_string(assigner.issued()));
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
        // A golden nobody can regenerate is a golden that gets hand-edited the first time it
        // fails, which is how a gate becomes a rubber stamp.
        const std::vector<std::string> all =
            lines_of(resolve("golden/identity/basic.txt"), true);
        bool named = false;
        for (const std::string& line : all) {
            if (line.find("emit_parity_golden.py --kind identity") != std::string::npos) {
                named = true;
            }
        }

        check(named, "the golden carries the command that emits it");
    }

}  // namespace

int main() {
    try {
        the_port_answers_what_the_reference_answered();
        the_golden_names_its_own_emitter();
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        ++failures;
        ++checks;
    }
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
