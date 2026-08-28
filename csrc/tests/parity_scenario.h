// The scenario format, read the way `benchmarks/parity/scenario.py` reads it.
//
// Line-oriented and not JSON precisely so this file can exist: one directive per line,
// whitespace-separated, `#` to end of line is a comment. The normative description of the
// format lives in that Python module's docstring and in `benchmarks/parity/README.md`; this
// is the second implementation of it, and section A of `test_ingest_parity.cpp` is what says
// the two agree.
//
// Header-only test support: off every link line, invisible to the build's closure walker.
#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"

namespace shipinfer::parity {

    struct Outcome {
        std::string what;
        std::string detail;
    };

    struct CameraScript {
        std::string camera_id;
        bool enabled = true;
        std::vector<Outcome> opens;
        std::vector<Outcome> reads;
        std::vector<std::string> sinks;

        // The last entry of each list repeats for ever, so a short script drives an unbounded
        // loop and a long run needs no long file.
        const Outcome& open_at(size_t index) const {
            return opens[std::min(index, opens.size() - 1)];
        }
        const Outcome& read_at(size_t index) const {
            return reads[std::min(index, reads.size() - 1)];
        }
        std::string sink_at(size_t index) const {
            return sinks.empty() ? "accept" : sinks[std::min(index, sinks.size() - 1)];
        }
    };

    struct Scenario {
        std::string name;
        int records_min = -1;
        std::map<std::string, std::string> settings;
        std::vector<CameraScript> cameras;

        int int_setting(const std::string& key) const { return std::stoi(settings.at(key)); }
        double double_setting(const std::string& key) const {
            return std::stod(settings.at(key));
        }

        // The un-jittered delay attempt `attempt` would draw, in whole microseconds. The one
        // piece of backoff arithmetic both planes repeat: jitter draws from `mt19937_64` here
        // and from `random.Random` there, so the delays themselves are never comparable while
        // this sequence is identical from initial/factor/cap alone.
        int64_t peek_us(uint64_t attempt) const {
            const double initial = int_setting("reconnect_initial_ms") / 1000.0;
            const double cap = int_setting("reconnect_max_ms") / 1000.0;
            const double factor = double_setting("reconnect_factor");
            const double peek =
                std::min(cap, initial * std::pow(factor, static_cast<double>(attempt)));
            return static_cast<int64_t>(peek * 1e6 + 0.5);
        }
    };

    inline const std::set<std::string>& kIntSettings() {
        static const std::set<std::string> keys = {
            "empty_reads_before_reconnect", "empty_read_sleep_ms", "failures_before_unhealthy",
            "reconnect_initial_ms", "reconnect_max_ms"};
        return keys;
    }
    inline const std::set<std::string>& kDoubleSettings() {
        static const std::set<std::string> keys = {"reconnect_factor", "reconnect_jitter"};
        return keys;
    }

    inline ConfigError refuse(const std::string& path, int line, const std::string& why) {
        return ConfigError(path + ":" + std::to_string(line) + ": " + why);
    }

    inline bool one_of(const std::string& value, const std::vector<std::string>& allowed) {
        for (const std::string& candidate : allowed) {
            if (value == candidate) return true;
        }
        return false;
    }

    // Parse one `.scn` file. Throws ConfigError naming `path:line` for any malformed line: a
    // scenario that half-parses would drive half a run and be compared against a whole golden.
    inline Scenario load_scenario(const std::string& path) {
        std::ifstream file(path);
        if (!file) throw ConfigError("no parity scenario at " + path);
        Scenario scenario;
        std::string raw;
        for (int number = 1; std::getline(file, raw); ++number) {
            const std::string line = raw.substr(0, raw.find('#'));
            std::istringstream stream(line);
            std::vector<std::string> tokens;
            for (std::string token; stream >> token;) tokens.push_back(token);
            if (tokens.empty()) continue;
            const std::string& directive = tokens[0];
            const size_t arity = tokens.size() - 1;
            if (directive == "scenario") {
                if (!scenario.name.empty() || arity != 1) {
                    throw refuse(path, number, "expected exactly one `scenario <name>`, first");
                }
                scenario.name = tokens[1];
            } else if (directive == "records_min") {
                if (arity != 1) throw refuse(path, number, "expected `records_min <n>`");
                scenario.records_min = std::stoi(tokens[1]);
            } else if (kIntSettings().count(directive) || kDoubleSettings().count(directive)) {
                if (!scenario.cameras.empty()) {
                    throw refuse(path, number,
                                 "setting '" + directive + "' must precede any camera");
                }
                if (arity != 1) throw refuse(path, number, "expected `" + directive + " <v>`");
                scenario.settings[directive] = tokens[1];
            } else if (directive == "camera") {
                if (arity < 1 || arity > 2 || (arity == 2 && tokens[2] != "disabled")) {
                    throw refuse(path, number, "expected `camera <id> [disabled]`");
                }
                for (const CameraScript& seen : scenario.cameras) {
                    if (seen.camera_id == tokens[1]) {
                        throw refuse(path, number, "camera '" + tokens[1] + "' declared twice");
                    }
                }
                scenario.cameras.push_back({tokens[1], arity == 1, {}, {}, {}});
            } else if (directive == "open" || directive == "read" || directive == "sink") {
                if (scenario.cameras.empty()) {
                    throw refuse(path, number, "`" + directive + "` before any `camera <id>`");
                }
                const std::vector<std::string> allowed =
                    directive == "open" ? std::vector<std::string>{"ok", "SourceOpenError",
                                                                   "SourceUnavailableError"}
                    : directive == "read"
                        ? std::vector<std::string>{"frame", "empty", "exhaust",
                                                   "FrameDecodeError"}
                        : std::vector<std::string>{"accept", "full", "closed"};
                if (arity < 1 || !one_of(tokens[1], allowed)) {
                    throw refuse(path, number,
                                 "`" + directive +
                                     "` outcome is not one of the "
                                     "outcomes this directive has");
                }
                if (arity > (directive == "sink" ? 1u : 2u)) {
                    throw refuse(path, number,
                                 "`" + directive + "` takes an outcome and at most one detail");
                }
                CameraScript& camera = scenario.cameras.back();
                const std::string detail = arity > 1 ? tokens[2] : "";
                if (directive == "open") camera.opens.push_back({tokens[1], detail});
                if (directive == "read") camera.reads.push_back({tokens[1], detail});
                if (directive == "sink") camera.sinks.push_back(tokens[1]);
            } else {
                throw refuse(path, number, "unknown directive '" + directive + "'");
            }
        }
        if (scenario.name.empty()) throw ConfigError(path + ": no `scenario <name>` line");
        if (scenario.records_min < 0) throw ConfigError(path + ": no `records_min <n>` line");
        for (const std::string& key :
             {"empty_read_sleep_ms", "reconnect_initial_ms", "reconnect_max_ms",
              "reconnect_factor", "reconnect_jitter"}) {
            if (!scenario.settings.count(key)) {
                throw ConfigError(path + ": missing required setting '" + key + "'");
            }
        }
        for (const CameraScript& camera : scenario.cameras) {
            if (!camera.enabled) continue;
            if (camera.opens.empty() || camera.reads.empty()) {
                throw ConfigError(path + ": camera '" + camera.camera_id +
                                  "' needs at least one `open` and one `read`");
            }
            const bool ends = camera.reads.back().what == "exhaust" ||
                              camera.opens.back().what == "SourceUnavailableError" ||
                              one_of("closed", camera.sinks);
            if (!ends) {
                throw ConfigError(path + ": camera '" + camera.camera_id +
                                  "' never finishes -- the last entry of each list repeats for "
                                  "ever, so a script must end in `read exhaust`, `open "
                                  "SourceUnavailableError` or a `sink closed`");
            }
        }
        return scenario;
    }

}  // namespace shipinfer::parity
