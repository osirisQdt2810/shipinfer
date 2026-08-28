// The ingest parity gate: this plane, driven over the same scenario as the Python one, held
// to the same committed golden.
//
// The sync rule in CLAUDE.md says a change to a Python data-plane seam is not finished until
// this plane carries it. Nothing enforced that: the two `test_ingest` suites assert the same
// properties in two languages, which catches a property somebody removed and not a behaviour
// that quietly differs.
//
// A difference is either a bug or an entry in `benchmarks/parity/known.py`, cited on both
// sides with an open ledger line. `known_divergence` below is that register's other half —
// spelled twice, the way `omitted_lanes.h` mirrors `build_csrc.py`, and
// `test_parity_ingest.py` fails if they drift.
//
// Offline: g++ alone, no CUDA, no GStreamer, and it must not include
// `ingest/sources/{gstreamer,replay}.h`.

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/ingest/camera/health.h"
#include "shipinfer/ingest/config.h"
#include "shipinfer/ingest/manager.h"
#include "tests/parity_scenario.h"
#include "tests/parity_trace.h"
#include "tests/scripted_source.h"

namespace {

    using namespace shipinfer;
    using namespace shipinfer::parity;
    using namespace std::chrono_literals;

    int failures = 0;
    int checks = 0;

    //: How long the whole fleet may take to finish its script -- `_RUN_BUDGET_S` there.
    constexpr auto kRunBudget = 20s;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        }
    }

    // There is deliberately no `skip` here, where `test_ingest.cpp` has one: everything this
    // binary needs is a committed file in this repository. A missing golden is a HARD failure
    // (see `main`), because a gate that skips reads as evidence and is not.

    // -- where the scenarios and goldens are ------------------------------------------------

    // The binary is run from the repository root by CI and from `csrc/build` by hand, so the
    // root is searched rather than assumed — and a failure prints every path it tried, because
    // "no such file" with no path is the least useful sentence a gate can end on.
    std::string resolve(const std::string& relative) {
        std::vector<std::string> roots;
        const char* named = std::getenv("SHIPINFER_PARITY_GOLDEN");
        if (named) {
            // Authoritative, not merely preferred: an operator who names a directory and
            // silently gets a different one has been lied to by the gate itself.
            roots.emplace_back(named);
        } else {
            for (const char* root : {"benchmarks/parity", "../benchmarks/parity",
                                     "../../benchmarks/parity", "../../../benchmarks/parity"}) {
                roots.emplace_back(root);
            }
        }
        std::string tried;
        for (const std::string& root : roots) {
            const std::string path = root + "/" + relative;
            std::ifstream probe(path);
            if (probe) return path;
            tried += (tried.empty() ? "" : ", ") + path;
        }
        throw ConfigError("cannot find " + relative + "; tried " + tried +
                          ". Run this binary from the repository root, or set "
                          "SHIPINFER_PARITY_GOLDEN to the benchmarks/parity directory");
    }

    std::vector<std::string> read_lines(const std::string& path) {
        std::ifstream file(path);
        if (!file) throw ConfigError("cannot read " + path);
        std::vector<std::string> lines;
        for (std::string line; std::getline(file, line);) {
            if (!line.empty()) lines.push_back(line);
        }
        return lines;
    }

    // -- the register's other half ----------------------------------------------------------

    // Python's `str.isidentifier()`, which is what the other plane's entry is written
    // against: a leading digit is NOT an identifier there, so accepting one here would let
    // this plane excuse a difference the Python differ would report.
    bool is_identifier(const std::string& value) {
        if (value.empty() || std::isdigit(static_cast<unsigned char>(value[0]))) return false;
        for (char c : value) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') return false;
        }
        return true;
    }

    // `last_error_type_prefix`: the Python actor stores f"{type(error).__name__}: {error}"
    // (`ingest/camera/actor.py::_record_failure`); this plane stores redact_in(what())
    // (`ingest/camera/actor.cpp::record_failure`), which carries no type in front of it.
    bool only_the_exception_type_prefix(const ParityRecord& python, const ParityRecord& mine) {
        if (python.text.size() < 2 || mine.text.size() < 2) return false;
        const size_t at = python.text[1].find(": ");
        return at != std::string::npos && python.text[1].substr(at + 2) == mine.text[1] &&
               is_identifier(python.text[1].substr(0, at));
    }

    // `fatal_consecutive_failures`: the Python health reads `backoff.attempts`, and the fatal
    // SourceUnavailableError path never calls next_delay(), so it stays 0; this plane
    // increments `consecutive_failures_` inside record_failure, so it reads 1.
    //
    // Keyed on the error's own words and not on the counts alone, because 0-against-1 is a
    // shape a future unrelated divergence could also have. "is unavailable" is the message
    // BOTH planes build a SourceUnavailableError from, so this key survives the fix for
    // `last_error_type_prefix` — keying on the type name would tie the two entries together.
    bool fatal_open_charges_one_failure(const ParityRecord& python, const ParityRecord& mine) {
        if (python.numbers.size() < 7 || mine.numbers.size() < 7 || python.text.size() < 2) {
            return false;
        }
        return python.numbers[6] == 0 && mine.numbers[6] == 1 &&
               python.text[0] == "unhealthy" &&
               python.text[1].find("is unavailable") != std::string::npos;
    }

    // The register's other half, as a table rather than a chain of `if`s: `main` walks it to
    // fail when a registered divergence fired in NO scenario, and
    // `test_parity_ingest.py::TestKnownDivergences` regexes the ids out of this file and out
    // of `known_divergence` below, so a difference this plane excuses without an entry in
    // `benchmarks/parity/known.py` — or an entry there this plane does not honour — is red on
    // both sides. Every id here has citations on both sides and an open ledger line.
    struct KnownEntry {
        std::string id;
        std::string field;
        bool (*matches)(const ParityRecord& python, const ParityRecord& mine);
    };

    const std::vector<KnownEntry>& known_register() {
        static const std::vector<KnownEntry> entries = {
            {"last_error_type_prefix", "last_error", &only_the_exception_type_prefix},
            {"fatal_consecutive_failures", "consecutive_failures",
             &fatal_open_charges_one_failure},
        };
        return entries;
    }

    // The known-divergence id that accounts for this one field, or "" if none does.
    std::string known_divergence(const ParityRecord& python, const ParityRecord& mine,
                                 const std::string& field) {
        if (python.kind != "health") return "";
        for (const KnownEntry& entry : known_register()) {
            if (entry.field == field && entry.matches(python, mine)) return entry.id;
        }
        return "";
    }

    // Which registered ids actually fired, across every scenario in this run. A register whose
    // entries stop firing is a register rotting into a suppression list, so `main` checks it.
    std::set<std::string> fired;

    std::vector<std::string> differing_fields(const ParityRecord& left,
                                              const ParityRecord& right) {
        if (left.kind != right.kind) return {"kind"};
        if (left.camera != right.camera) return {"camera"};
        const auto entry = kFields().find(left.kind);
        if (entry == kFields().end())
            throw ConfigError("unknown record kind '" + left.kind + "'");
        const FieldNames& names = entry->second;
        std::vector<std::string> differ;
        for (size_t i = 0; i < names.numbers.size(); ++i) {
            if (i >= left.numbers.size() || i >= right.numbers.size() ||
                left.numbers[i] != right.numbers[i]) {
                differ.push_back(names.numbers[i]);
            }
        }
        for (size_t i = 0; i < names.text.size(); ++i) {
            if (i >= left.text.size() || i >= right.text.size() ||
                left.text[i] != right.text[i]) {
                differ.push_back(names.text[i]);
            }
        }
        return differ;
    }

    // -- driving one scenario over the real manager -----------------------------------------

    std::vector<std::string> run_scenario(const Scenario& scenario) {
        ParityTraceWriter writer;
        writer.header(scenario.name, "cpp");
        std::map<std::string, std::unique_ptr<CameraRecorder>> recorders;
        std::vector<IngestConfig> cameras;
        for (const CameraScript& script : scenario.cameras) {
            cameras.push_back(camera_config(scenario, script));
            if (script.enabled) {
                // The camera just resolved, so the recorder's mirrored backoff is built from
                // exactly the numbers the actor's own will be. Read in the constructor and
                // not held, so the vector reallocating later is harmless.
                recorders[script.camera_id] =
                    std::make_unique<CameraRecorder>(script, cameras.back(), writer);
            }
        }
        // The sink and the recorders outlive the manager, which references both and stops
        // its actors in its own destructor.
        RecordingSink sink(recorders);
        std::map<std::string, CameraHealth> healths;
        size_t abandoned = 0;
        {
            // The resolver is pointed at the manager after it is built and before it is
            // started, because the manager takes the factory that reads it.
            IngestManager* live = nullptr;
            auto resolve = [&live](const std::string& id) -> std::shared_ptr<CameraActor> {
                return live ? live->actor(id) : nullptr;
            };
            IngestManager manager(cameras, sink, scripted_factory(resolve, recorders));
            live = &manager;
            manager.start();
            std::map<std::string, std::shared_ptr<CameraActor>> actors;
            for (const auto& entry : recorders)
                actors[entry.first] = manager.actor(entry.first);
            // Every script ends on its own (an exhausted read, a fatal open, a closed
            // sink), so overrunning this means the run is STUCK. Mirrors the Python
            // driver's `ServerStateError`: a budget that expires silently would emit a
            // truncated trace and then compare it as if it were whole, which is a gate
            // reporting "17 record(s) against 31" for a hang.
            const auto deadline = std::chrono::steady_clock::now() + kRunBudget;
            for (;;) {
                std::vector<std::string> running;
                for (const auto& entry : actors) {
                    if (entry.second->is_running()) running.push_back(entry.first);
                }
                if (running.empty()) break;
                if (std::chrono::steady_clock::now() >= deadline) {
                    std::string names;
                    for (const std::string& name : running) {
                        names += (names.empty() ? "" : ", ") + name;
                    }
                    throw ServerStateError(
                        "parity run did not finish within 20s; still running: " + names);
                }
                std::this_thread::sleep_for(2ms);
            }
            for (const auto& entry : actors) healths[entry.first] = entry.second->health();
            abandoned = manager.stop(2000ms);
        }
        int64_t read = 0, published = 0, dropped = 0;
        for (const auto& entry : healths) {
            const CameraHealth& health = entry.second;
            writer.record("health", entry.first,
                          {static_cast<int64_t>(health.frames_read),
                           static_cast<int64_t>(health.frames_published),
                           static_cast<int64_t>(health.frames_dropped),
                           static_cast<int64_t>(health.empty_reads),
                           static_cast<int64_t>(health.connects),
                           static_cast<int64_t>(health.connect_failures),
                           static_cast<int64_t>(health.consecutive_failures)},
                          {to_string(health.state), health.last_error});
            read += static_cast<int64_t>(health.frames_read);
            published += static_cast<int64_t>(health.frames_published);
            dropped += static_cast<int64_t>(health.frames_dropped);
        }
        writer.record("stop", "", {static_cast<int64_t>(abandoned)});
        writer.record("end", "",
                      {static_cast<int64_t>(healths.size()), read, published, dropped});
        return writer.lines();
    }

    // -- section A: the scenario loader ------------------------------------------------------

    void write_file(const std::string& path, const std::string& body) {
        std::ofstream file(path);
        file << body;
    }

    void test_the_committed_scenarios_load() {
        for (const char* named : {"reconnect", "backpressure", "fatal_vs_retryable"}) {
            const std::string name = named;
            const Scenario scenario = load_scenario(resolve("scenarios/" + name + ".scn"));
            check(scenario.name == name, "scenario " + name + " names itself");
            check(scenario.records_min > 0, "scenario " + name + " declares a vacuity floor");
            check(!scenario.cameras.empty(), "scenario " + name + " has cameras");
        }
    }

    void test_a_malformed_scenario_is_refused_naming_the_line() {
        const std::string path = "/tmp/shipinfer_parity_malformed.scn";
        write_file(
            path,
            "scenario bad\nrecords_min 1\nempty_read_sleep_ms 0\nreconnect_initial_ms 2\n"
            "reconnect_max_ms 8\nreconnect_factor 2.0\nreconnect_jitter 0.0\n"
            "camera cam0\nopen ok\nread frame\nread exhaust\nwibble 3\n");
        try {
            load_scenario(path);
            check(false, "a malformed scenario line is refused");
        } catch (const ConfigError& refusal) {
            const std::string message = refusal.what();
            check(message.find(":12:") != std::string::npos,
                  "the refusal names the line: " + message);
            check(message.find("wibble") != std::string::npos,
                  "the refusal names the directive: " + message);
        }
        std::remove(path.c_str());
    }

    void test_a_camera_that_never_finishes_is_refused() {
        const std::string path = "/tmp/shipinfer_parity_endless.scn";
        write_file(path,
                   "scenario endless\nrecords_min 1\nempty_read_sleep_ms 0\n"
                   "reconnect_initial_ms 2\nreconnect_max_ms 8\nreconnect_factor 2.0\n"
                   "reconnect_jitter 0.0\ncamera cam0\nopen ok\nread frame\n");
        try {
            load_scenario(path);
            check(false, "a script whose last read repeats for ever is refused");
        } catch (const ConfigError& refusal) {
            check(std::string(refusal.what()).find("never finishes") != std::string::npos,
                  "the refusal says the script never finishes");
        }
        std::remove(path.c_str());
    }

    // -- section B: the writer's canonical form ----------------------------------------------

    void test_the_key_order_is_fixed_and_the_numbers_are_plain() {
        ParityTraceWriter writer;
        writer.header("shape", "cpp");
        writer.record("retry", "cam0", {3, 1234567});
        const std::vector<std::string> lines = writer.lines();
        check(lines.at(0) == "{\"schema\":1,\"scenario\":\"shape\",\"plane\":\"cpp\"}",
              "the header is canonical: " + lines.at(0));
        check(lines.at(1) ==
                  "{\"kind\":\"retry\",\"camera\":\"cam0\",\"n\":[3,1234567],\"t\":[]}",
              "a record is canonical, with no thousands separator: " + lines.at(1));
    }

    void test_a_word_the_canonical_writer_cannot_spell_is_refused() {
        ParityTraceWriter writer;
        writer.header("shape", "cpp");
        try {
            writer.record("drop", "cam0", {}, {"a\"quote"});
            check(false, "a quote in a record word is refused");
        } catch (const ConfigError&) {
            check(true, "a quote in a record word is refused");
        }
        try {
            writer.record("frame", "cam0", {1, 2});
            check(false, "the wrong field count is refused");
        } catch (const ConfigError&) {
            check(true, "the wrong field count is refused");
        }
    }

    void test_a_line_round_trips_through_the_reader() {
        const ParityRecord record{"health", "cam0", {1, 2, 3, 4, 5, 6, 7}, {"streaming", ""}};
        const ParityRecord back = parse_line(to_line(record));
        check(back == record, "a record survives to_line + parse_line: " + back.render());
    }

    // -- section C: this plane against the committed golden ----------------------------------

    // Per camera, in order, and the fleet's records as their own sequence — the comparison
    // rule `benchmarks/parity/diff.py` states. Cross-camera interleaving is scheduler
    // nondeterminism and is never compared, and a deleted record must not cascade into a
    // difference at every later index of every other camera.
    std::map<std::string, std::vector<ParityRecord>> by_camera(
        const std::vector<std::string>& lines) {
        std::map<std::string, std::vector<ParityRecord>> grouped;
        for (size_t i = 1; i < lines.size(); ++i) {
            ParityRecord record = parse_line(lines[i]);
            grouped[record.camera].push_back(std::move(record));
        }
        return grouped;
    }

    void test_this_plane_matches_the_golden(const std::string& name) {
        const Scenario scenario = load_scenario(resolve("scenarios/" + name + ".scn"));
        const std::vector<std::string> golden =
            read_lines(resolve("golden/" + name + ".jsonl"));
        const std::vector<std::string> mine = run_scenario(scenario);
        check(static_cast<int>(mine.size()) - 1 >= scenario.records_min,
              name + ": produced at least the " + std::to_string(scenario.records_min) +
                  " record(s) it promised, got " + std::to_string(mine.size() - 1));
        check(golden.at(0).find("\"scenario\":\"" + name + "\"") != std::string::npos,
              name + ": the golden names this scenario");
        check(golden.at(0).find("\"schema\":1") != std::string::npos,
              name + ": the golden is this schema");
        // The register is asymmetric — `known_divergence` is given the Python record first —
        // so a golden emitted by some other plane would have the two sides the wrong way
        // round and quietly excuse the mirror image of each entry.
        check(golden.at(0).find("\"plane\":\"python\"") != std::string::npos,
              name + ": the golden was emitted by the python plane: " + golden.at(0));
        const auto theirs = by_camera(golden);
        const auto ours = by_camera(mine);
        std::set<std::string> cameras;
        for (const auto& entry : theirs) cameras.insert(entry.first);
        for (const auto& entry : ours) cameras.insert(entry.first);
        std::set<std::string> accepted;
        check(!cameras.empty(), name + ": the golden groups at least one camera");
        for (const std::string& camera : cameras) {
            static const std::vector<ParityRecord> kNone;
            const std::vector<ParityRecord>& left =
                theirs.count(camera) ? theirs.at(camera) : kNone;
            const std::vector<ParityRecord>& right =
                ours.count(camera) ? ours.at(camera) : kNone;
            const std::string who = camera.empty() ? "<fleet>" : camera;
            // ONE check per camera, always counted, so a green run says how many records it
            // actually compared. A comparison that contributes no checks when it passes is
            // indistinguishable from one that ran over nothing.
            std::string problem;
            for (size_t i = 0; i < left.size() && i < right.size() && problem.empty(); ++i) {
                std::string unexplained;
                for (const std::string& field : differing_fields(left[i], right[i])) {
                    const std::string entry = known_divergence(left[i], right[i], field);
                    if (entry.empty()) {
                        unexplained += (unexplained.empty() ? "" : ", ") + field;
                    } else {
                        accepted.insert(camera + "." + field + " = " + entry);
                        fired.insert(entry);
                    }
                }
                if (unexplained.empty()) continue;
                problem = "record " + std::to_string(i) + " differs on " + unexplained +
                          "\n  golden: " + left[i].render() +
                          "\n  here:   " + right[i].render();
            }
            if (problem.empty() && left.size() != right.size()) {
                const size_t at = std::min(left.size(), right.size());
                problem = std::to_string(left.size()) + " golden record(s) against " +
                          std::to_string(right.size()) + " here; first extra is " +
                          (at < left.size() ? left[at] : right[at]).render();
            }
            check(problem.empty(),
                  name + ": " + who + ": " +
                      (problem.empty() ? std::to_string(left.size()) + " record(s) match"
                                       : problem));
        }
        for (const std::string& entry : accepted) {
            std::printf("KNOWN: %s: %s\n", name.c_str(), entry.c_str());
        }
    }

    // -- section D: the register is still a register -----------------------------------------

    // A registered divergence that fires in NO scenario is the register rotting into a
    // permanent suppression: either the divergence was fixed — in which case the entry goes,
    // on BOTH sides, which is the whole point of banning `xfail` — or the case that reproduced
    // it was lost. Neither is visible from the diff alone, because a fix at this plane's CALL
    // SITE leaves the entry's own citation untouched and simply stops printing `KNOWN:`.
    void test_the_register_is_still_a_register() {
        for (const KnownEntry& entry : known_register()) {
            check(fired.count(entry.id) == 1,
                  "known divergence " + entry.id +
                      " fired in no scenario: either it is fixed (delete it here AND in "
                      "benchmarks/parity/known.py, with its ledger line) or the case that "
                      "reproduced it is gone");
        }
        // The other direction: an id excused by an `if` written past the table. The Python
        // half fails on that too (the two id sets must be equal), and this makes the binary
        // itself say so rather than printing a KNOWN line nobody reads.
        for (const std::string& id : fired) {
            bool registered = false;
            for (const KnownEntry& entry : known_register()) registered |= entry.id == id;
            check(registered, "'" + id +
                                  "' was excused past known_register(): every id "
                                  "known_divergence returns is an entry in the table, and "
                                  "every entry in the table has one in known.py");
        }
    }

}  // namespace

int main() {
    try {
        test_the_committed_scenarios_load();
        test_a_malformed_scenario_is_refused_naming_the_line();
        test_a_camera_that_never_finishes_is_refused();

        test_the_key_order_is_fixed_and_the_numbers_are_plain();
        test_a_word_the_canonical_writer_cannot_spell_is_refused();
        test_a_line_round_trips_through_the_reader();

        test_this_plane_matches_the_golden("reconnect");
        test_this_plane_matches_the_golden("backpressure");
        test_this_plane_matches_the_golden("fatal_vs_retryable");

        // Last, deliberately: it reads what the three comparisons above accumulated.
        test_the_register_is_still_a_register();
    } catch (const std::exception& error) {
        // A missing golden or an unreadable scenario is a HARD failure, never a skip: a gate
        // that fails open is worse than no gate, because it reads as evidence.
        std::fprintf(stderr, "FAIL: the parity harness could not run: %s\n", error.what());
        ++failures;
    }
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
