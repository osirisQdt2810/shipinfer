// One URI per camera, read from a file the control plane wrote.
//
// WHY A FILE AND NOT A TEMPLATE
//
// `cli/bench.cpp` gave every even camera `--person-frames` verbatim, which is right for
// `replay` (a folder of JPEGs, shared by design) and wrong for every real source: fifty RTSP
// cameras need fifty URIs. The obvious fix is to format `<base>/cam<N>` here -- and that is
// the mistake `benchmarks/harness/shipinfer.py::_rtsp_cameras` documents having made: "the
// server owns the path layout, and a benchmark that guessed it would fail as a connection
// refusal minutes into a run rather than as a mistake at start-up". `scripts/rtsp_serve.py`
// already prints its own URIs, so they are read rather than reconstructed.
//
// The Python plane splits the fleet half on a person server and half on a ship server,
// because the content mix decides the crop fan-out and therefore the downstream load. That
// split lives in whoever writes the file, so both planes get it from one place.
#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"

namespace shipinfer {

    // Read `cameras` URIs, in order, one per line. Blank lines and `#` comments are skipped so
    // a generated file may explain itself.
    //
    // Throws ConfigError when the file is unreadable or names FEWER cameras than the run
    // wants: a short list silently reused would point several cameras at one stream and
    // measure a fleet that does not exist. More lines than cameras is fine -- a 50-camera
    // file serves a 8-camera smoke run, which is how the same file gets reused.
    inline std::vector<std::string> read_camera_uris(const std::string& path, size_t cameras) {
        std::ifstream file(path);
        if (!file) throw ConfigError("cannot read the camera URI list at " + path);
        std::vector<std::string> uris;
        std::string line;
        while (std::getline(file, line)) {
            const size_t hash = line.find('#');
            if (hash != std::string::npos) line.erase(hash);
            const size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) continue;
            const size_t last = line.find_last_not_of(" \t\r\n");
            uris.push_back(line.substr(first, last - first + 1));
        }
        if (uris.size() < cameras) {
            throw ConfigError("the camera URI list at " + path + " names " +
                              std::to_string(uris.size()) + " camera(s) and this run wants " +
                              std::to_string(cameras) +
                              "; a short list would point several cameras at one stream and "
                              "measure a fleet that does not exist");
        }
        return uris;
    }

}  // namespace shipinfer
