// Which external build lanes this binary was compiled WITHOUT, and the source names they own.
//
// WHY A BINARY HAS TO KNOW THIS
// -----------------------------
// `scripts/build_csrc.py` compiles a unit that reaches outside this tree only when that unit's
// `pkg-config` lane is part of the build (`EXTERNAL`, `--with-external`). A unit that is not
// compiled is a unit whose file-scope `SourceRegistrar` never runs, so the registry does not
// contain that source and `create_source("gstreamer")` used to answer **"unknown video
// source"** — for a name that is spelled correctly, ships in this tree, and is documented in
// `ingest/sources/gstreamer.h`. `ingest/registry.h` argues at length that "not registered" and
// "not installed" are different problems with different fixes; a third one, *not in this
// build*, was arriving disguised as the first (#46 round 2). The build script is the only thing
// that knows which lanes it left out, so it bakes the list into every unit it compiles:
// `-DSHIPINFER_OMITTED_LANES="gstreamer,opencv"` (a comma list, possibly empty).
//
// WHY THE TABLE BELOW IS NOT A CLOSURE VIOLATION
// ----------------------------------------------
// The invariant at the top of `ingest/registry.cpp` is about **includes**: no unit under
// `ingest/` may include `sources/replay.h` or `sources/gstreamer.h`, because the build script
// follows a header to the `.cpp` beside it, so one such include would drag CUDA, OpenCV or
// `libgstreamer-1.0` into the closure of the whole ingest plane and the offline C++ tier would
// stop building. **This header includes neither, and no other header outside `core/`.** What it
// names is the *registered names* those units choose — the same strings an operator puts in a
// camera's `source` field — as string literals, which pull in nothing at all. The registry
// itself still knows the name of no source: it is handed this answer, and only on the path
// where it was about to say "unknown".
//
// WHY THE MESSAGE DOES NOT BLAME pkg-config
// -----------------------------------------
// Two kinds of build omit a lane and only one of them is a fault: a **full** build whose `-dev`
// package is missing (there the script prints a WARNING naming the package and the install
// hint), and **`--offline` without `--with-external <lane>`**, which is the offline tier
// working exactly as designed — g++ alone, nobody's `-dev` package, by ADR-001. A refusal that
// blamed `pkg-config` would therefore be wrong about half the builds it printed in, so it
// states the build fact and points at the flag that changes it.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace shipinfer {

    // The comma-separated external lanes `scripts/build_csrc.py` left out of this binary.
    //
    // Empty when the macro is undefined, which is any binary that script did not build: a
    // hand-rolled `g++` line keeps the old bare "unknown video source" rather than gaining a
    // wrong claim about a lane nobody configured.
    inline std::string omitted_lanes() {
#ifdef SHIPINFER_OMITTED_LANES
        return SHIPINFER_OMITTED_LANES;
#else
        return "";
#endif
    }

    // The omitted lane that would have registered `name`, or empty when there is none.
    //
    // `name` is the raw string a camera's config asked for — canonical name or alias, because
    // `SourceRegistry::canonical` is the only caller and it arrives here precisely because it
    // could resolve neither.
    inline std::string omitted_lane_of_source(const std::string& name) {
        // Lane -> every name the sources in that lane register, canonical and alias alike. One
        // entry per lane in `EXTERNAL`; the two tables are checked against each other by
        // `tests/test_build_csrc.py`, because a lane renamed on one side only would make this
        // whole mechanism go quiet — which is the old behaviour, silently restored.
        static const std::vector<std::pair<std::string, std::vector<std::string>>> kTable = {
            {"gstreamer", {"gstreamer", "gst"}},
            {"opencv", {"replay", "file", "video"}},
        };
        // Bracketed with commas at both ends so a lane name is matched whole: without it,
        // "opencv" would match a lane list containing "opencv-contrib" one day.
        const std::string omitted = "," + omitted_lanes() + ",";
        for (const auto& [lane, names] : kTable) {
            if (omitted.find("," + lane + ",") == std::string::npos) continue;
            for (const std::string& registered : names) {
                if (registered == name) return lane;
            }
        }
        return "";
    }

}  // namespace shipinfer
