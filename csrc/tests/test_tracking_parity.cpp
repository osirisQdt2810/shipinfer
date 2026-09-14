// One camera's per-ROW track ids, beside the other plane's.
//
// THE CASE `tracker_options` NEVER HAD. Both planes run the same ByteTrack over the same
// boxes; what differs is how each maps the tracker's answer back onto the detection ROWS --
// `Track::last_match` here, an IoU re-derivation against the FILTERED box there. The 13 Sep
// ruling (`CSRC-TRACKER-ATTRIBUTION`) settled that this lane must not grow the re-derivation;
// what it could not say is how far apart the two mappings actually put the ids, because no
// tracking scenario existed. This prints both streams so the difference is a number.
//
// A COUNT OF ZERO IS ONLY WORTH READING IF THE CASE REACHES THE EDGE, so the scenarios are
// chosen to sit on it and this binary asserts they still do -- `aspect_edge` keeps its track
// at the narrowest width that can, `aspect_break` is a tenth of a pixel past and re-mints.
// Without that pair a submodule bump moves the edge and "0 differing" stays green on a case
// that probes nothing, which is the whole failure mode a register full of goldens invites.
//
// The LANE's test: it reaches `shipvision::mot`, so it builds only where the submodule does.
#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/tracking/shard.h"
#include "tests/parity_files.h"

namespace {

    using namespace shipinfer;
    using shipinfer::parity::read_lines;
    using shipinfer::parity::resolve;

    // What these scenarios differed by when the case was built. See the ratchet in main.
    constexpr int MEASURED = 0;

    int failures = 0;
    int checks = 0;
    int differing = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    struct Scenario {
        std::string name;
        std::vector<std::vector<Detection>> frames;
    };

    std::vector<Scenario> load(const std::string& path) {
        std::vector<Scenario> out;
        for (const std::string& raw : read_lines(path, /*keep_comments=*/false)) {
            std::istringstream words(raw);
            std::string head;
            if (!(words >> head)) continue;
            if (head == "scenario") {
                Scenario made;
                words >> made.name;
                out.push_back(std::move(made));
            } else if (head == "frame") {
                out.back().frames.emplace_back();
            } else if (head == "box") {
                Detection det;
                words >> det.x1 >> det.y1 >> det.x2 >> det.y2 >> det.score;
                det.index = static_cast<int>(out.back().frames.back().size());
                out.back().frames.back().push_back(det);
            } else {
                throw ConfigError(path + ": unknown line '" + raw + "'");
            }
        }
        return out;
    }

    // NORMALISED PER SCENARIO, the way the other plane's driver does it and for the same
    // reason: a track id comes from a process-wide counter, so the literal number says
    // nothing across runs. What the seam is about is which ROW carries an identity, and
    // whether it is the same one as the frame before.
    std::string render(const std::vector<Scenario>& scenarios) {
        std::string out;
        for (const Scenario& scenario : scenarios) {
            shipvision::mot::ByteTrackTracker::Options options;
            options.min_hits = 1;
            options.max_age = 30;
            TrackerShard shard(options);
            std::map<int, int> seen;
            out += "scenario " + scenario.name + "\n";
            for (size_t frame = 0; frame < scenario.frames.size(); ++frame) {
                const TrackUpdate answer =
                    shard.update("cam0", static_cast<int64_t>(frame), scenario.frames[frame]);
                out += "frame " + std::to_string(frame) + " ids";
                for (size_t row = 0; row < answer.ids.size(); ++row) {
                    if (answer.ids[row] < 0) {
                        out += " -";
                        continue;
                    }
                    const auto found = seen.find(answer.ids[row]);
                    const int label =
                        found != seen.end()
                            ? found->second
                            : (seen[answer.ids[row]] = static_cast<int>(seen.size()) + 1);
                    out += " " + std::to_string(label);
                }
                out += "\n";
            }
        }
        return out;
    }

    // The last `frame N ids ...` line of one scenario, as its id string. The bracket checks
    // read this rather than a frame number, so adding frames to a scenario cannot silently
    // move which frame they test.
    std::string last_ids(const std::string& rendered, const std::string& scenario) {
        std::istringstream lines(rendered);
        std::string line, found;
        bool inside = false;
        while (std::getline(lines, line)) {
            if (line.rfind("scenario ", 0) == 0) {
                inside = line.substr(9) == scenario;
                continue;
            }
            const size_t ids = line.find(" ids");
            if (inside && ids != std::string::npos) found = line.substr(ids + 4);
        }
        if (!found.empty() && found[0] == ' ') found.erase(0, 1);
        return found;
    }

}  // namespace

int main() {
    std::string mine;
    std::string theirs;
    try {
        mine = render(load(resolve("scenarios/tracking/attribution.scn")));
        for (const std::string& line :
             read_lines(resolve("golden/tracking/attribution.txt"), /*keep_comments=*/false)) {
            theirs += line + "\n";
        }
    } catch (const std::exception& error) {
        // NOT A SKIP. This binary is built only where the submodule is, so nothing here is
        // optional: a throw means the lane is broken, and reporting that as success is how a
        // green suite stops meaning anything. It was a skip, and that was the bug.
        std::printf("FAIL: the tracking lane could not run (%s)\n", error.what());
        std::printf("1 checks, 1 failure(s)\n");
        return 1;
    }

    // THE CASE STILL SITS ON THE EDGE, read off this plane's own render. `1` is the first
    // track of the scenario, so "still 1" means continued and anything else means re-minted.
    check(last_ids(mine, "steady") == "1", "the control scenario still holds one track");
    check(last_ids(mine, "aspect_edge") == "1",
          "aspect_edge still CONTINUES its track -- the narrowest width that can. If this "
          "fails the tracker's gate moved and the scenario no longer reaches the edge");
    const std::string broke = last_ids(mine, "aspect_break");
    check(!broke.empty() && broke != "1",
          "aspect_break still RE-MINTS -- one tenth of a pixel past the edge. With this and "
          "the check above, the pair brackets the gate and the count below means something");

    // NOT A BYTE GATE, and that is the point. `tracker_options` is a REGISTERED divergence, so
    // a test that failed on a difference would be asserting the planes agree where the
    // register says they do not. It prints the difference instead, per line, and the count is
    // what `SHIPVISION-TRACK-LAST-MATCH` asks for before anyone builds the fix.
    std::istringstream ours(mine), reference(theirs);
    std::string a, b;
    size_t line = 0;
    while (std::getline(ours, a) && std::getline(reference, b)) {
        ++line;
        if (a == b) continue;
        ++differing;
        std::printf("DIFFERS line %zu: cpp %-28s py %s\n", line, a.c_str(), b.c_str());
    }
    check(line > 0, "the golden and this run both produced lines to compare");
    check(!std::getline(ours, a) && !std::getline(reference, b),
          "the two streams have the same number of lines, so the scenarios match");

    // A RATCHET ON THE MEASUREMENT, not an assertion that the planes agree. MEASURED is what
    // these scenarios produced on 14 Sep; the register still says the two rules CAN differ,
    // and if one day they do here that is the news `SHIPVISION-TRACK-LAST-MATCH` is waiting
    // for -- so it is loud rather than a number nobody re-reads.
    check(differing <= MEASURED,
          "the two planes now differ on a case where they did not. That is not a regression "
          "to paper over: `tracker_options` has become REACHABLE at the defaults, so re-read "
          "the entry in known.py, record the new count, and price the fix");
    std::printf("tracking parity: %zu line(s) compared, %d differing (measured %d)\n", line,
                differing, MEASURED);
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
