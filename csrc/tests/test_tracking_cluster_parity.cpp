// The whole cross-camera composition, held to the REFERENCE's answer.
//
// `test_gate_parity.cpp` and `test_identity_parity.cpp` pin the two stateful halves; this pins
// that they COMPOSE the same way -- gate, gram, gated matcher, clusterer, assigner -- which is
// the part a port can get right in every piece and still get wrong between them.
//
// The LANE's test: it reaches the registered `shipvision` tracker, so it builds only where the
// submodule does. That is also why the two halves have goldens of their own -- those run in the
// offline tier where CI does not check the submodule out.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/mtmc/cluster.h"
#include "tests/parity_files.h"

namespace {

    using namespace shipinfer;
    using shipinfer::mtmc::ClusterObservation;
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
        int repeats = 1;
        std::vector<ClusterObservation> instant;
    };

    // `<camera>#<track>:<x>,<y>:<box_height_px>`
    ClusterObservation parse_field(const std::string& field) {
        const size_t hash = field.find('#');
        const size_t first = field.find(':', hash);
        const size_t comma = field.find(',', first);
        const size_t second = field.find(':', comma);
        if (hash == std::string::npos || first == std::string::npos ||
            comma == std::string::npos || second == std::string::npos) {
            throw ConfigError("malformed observation '" + field + "'");
        }
        ClusterObservation out;
        out.key = TrackKey{field.substr(0, hash),
                           std::stoll(field.substr(hash + 1, first - hash - 1))};
        out.embedding = {std::stof(field.substr(first + 1, comma - first - 1)),
                         std::stof(field.substr(comma + 1, second - comma - 1))};
        out.box[0] = 0.0f;
        out.box[1] = 0.0f;
        out.box[2] = 40.0f;
        out.box[3] = std::stof(field.substr(second + 1));
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
                stream >> scenario.name >> scenario.repeats;
                scenarios.push_back(std::move(scenario));
                continue;
            }
            if (head != "instant") throw ConfigError("unknown line '" + line + "'");
            if (scenarios.empty()) throw ConfigError("an `instant` before any `scenario`");
            std::string field;
            while (stream >> field) scenarios.back().instant.push_back(parse_field(field));
        }
        return scenarios;
    }

    void the_composition_answers_what_the_reference_answered() {
        const std::vector<Scenario> scenarios =
            read_scenarios(resolve("scenarios/cluster/basic.txt"));
        const std::vector<std::string> golden =
            parity::read_lines(resolve("golden/cluster/basic.txt"), false);

        std::vector<std::string> written;
        int slot = 0;
        for (const Scenario& scenario : scenarios) {
            // A SLOT PER SCENARIO, because `create_cluster_tracker` caches per (impl, slot) --
            // one identity space per group -- so sharing one would carry state between them.
            const auto tracker =
                mtmc::create_cluster_tracker("shipvision", "parity" + std::to_string(slot++));
            written.push_back("scenario " + scenario.name);
            for (int instant = 0; instant < scenario.repeats; ++instant) {
                std::vector<std::string> answers;
                for (const auto& [key, global_id] : tracker->ids(scenario.instant)) {
                    answers.push_back(key.str() + "=" + std::to_string(global_id));
                }
                // SORTED, because the golden is: the reference answers in its cluster's
                // flattened order and this one in key order, and neither order is a property
                // anybody downstream reads -- the mapping is.
                std::sort(answers.begin(), answers.end());
                std::string line;
                for (const std::string& answer : answers) {
                    if (!line.empty()) line += " ";
                    line += answer;
                }
                written.push_back("ids " + line);
            }
            written.push_back("identities " + std::to_string(tracker->sizes().identities));
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

    ClusterObservation looks(const std::string& camera, int64_t track,
                             std::vector<float> embedding) {
        ClusterObservation out = parse_field(camera + "#" + std::to_string(track) + ":1,0:300");
        out.embedding = std::move(embedding);
        return out;
    }

    // doc: long two refusals, why the golden cannot reach them, and what they stand between
    void the_two_refusals_the_golden_cannot_reach() {
        // `gram_of`'s guards are this PR's own content and no scenario can exercise them: every
        // observation in `scenarios/cluster/basic.txt` is a valid, equal-width, non-zero
        // 2-vector, and the emitter could not render one that is not -- the reference raises
        // before it answers. So they are driven through the seam here, at `n >= 2`, which is
        // the path `cluster()` does NOT short-circuit and therefore the one `identity.cpp`'s
        // equivalent refusals never see.
        //
        // The width check is not a niceness check: `embedding[k]` is `vector::operator[]`,
        // unchecked, and `dim` comes from `admitted.front()` -- so a 256-dim track in a 512-dim
        // group reads 256 floats past the end of its vector. #221's review priced it.
        const std::shared_ptr<mtmc::ClusterTracker> tracker =
            mtmc::create_cluster_tracker("shipvision", "refusals");
        // THREE INSTANTS, because the gate admits nothing until `min_hits`: the refusals live
        // behind it, so a one-instant test would pass on an empty admitted set.
        std::string zero_message;
        std::string width_message;
        for (int instant = 0; instant < 3; ++instant) {
            const bool last = instant == 2;
            try {
                tracker->ids({looks("cam0", 1, {1.0f, 0.0f}),
                              looks("cam1", 1,
                                    last ? std::vector<float>{0.0f, 0.0f}
                                         : std::vector<float>{1.0f, 0.0f})});
            } catch (const InferenceError& error) {
                zero_message = error.what();
            }
        }
        const std::shared_ptr<mtmc::ClusterTracker> second =
            mtmc::create_cluster_tracker("shipvision", "refusals-width");
        for (int instant = 0; instant < 3; ++instant) {
            const bool last = instant == 2;
            try {
                // THE LONG VECTOR FIRST, so `dim` is 3 and the SHORT one is the row the
                // normalisation loop indexes -- the direction the comment above prices, and
                // the one the first draft of this test did not cover (#221 round 3).
                second->ids({looks("cam0", 1, {1.0f, 0.0f, 0.0f}),
                             looks("cam1", 1,
                                   last ? std::vector<float>{1.0f, 0.0f}
                                        : std::vector<float>{1.0f, 0.0f, 0.0f})});
            } catch (const InferenceError& error) {
                width_message = error.what();
            }
        }

        check(zero_message.find("no direction") != std::string::npos,
              "an all-zero embedding is refused with the reference's own reason, got: " +
                  (zero_message.empty() ? "(nothing thrown)" : zero_message));
        check(zero_message.find("cam1#1") != std::string::npos,
              "and the refusal names the track rather than the condition");
        check(width_message.find("2 dimensions") != std::string::npos ||
                  width_message.find("and 2") != std::string::npos,
              "a second embedding width is refused, got: " +
                  (width_message.empty() ? "(nothing thrown)" : width_message));
        check(width_message.find("one embedder") != std::string::npos,
              "and says why one identity space cannot hold two widths");
    }

    void the_golden_names_its_own_emitter() {
        const std::vector<std::string> all =
            parity::read_lines(resolve("golden/cluster/basic.txt"), true);
        bool named = false;
        for (const std::string& line : all) {
            if (line.find("emit_parity_golden.py --kind cluster") != std::string::npos) {
                named = true;
            }
        }

        check(named, "the golden carries the command that emits it");
    }

}  // namespace

int main() {
    try {
        the_composition_answers_what_the_reference_answered();
        the_two_refusals_the_golden_cannot_reach();
        the_golden_names_its_own_emitter();
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        ++failures;
        ++checks;
    }
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
