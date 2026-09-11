// Where a parity gate finds the scenarios and goldens it is held to.
//
// Shared by every parity binary: the search order is part of the contract with CI and with
// whoever runs one by hand, and two copies of it would drift into two contracts.
#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"

namespace shipinfer::parity {

    // The binary is run from the repository root by CI and from `csrc/build` by hand, so the
    // root is searched rather than assumed — and a failure prints every path it tried, because
    // "no such file" with no path is the least useful sentence a gate can end on.
    inline std::string resolve(const std::string& relative) {
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

    // Blank lines kept, because a scenario parser counts them to name the line of a refusal.
    inline std::vector<std::string> read_lines_keeping_blanks(const std::string& path) {
        std::ifstream file(path);
        if (!file) throw ConfigError("cannot read " + path);
        std::vector<std::string> lines;
        for (std::string line; std::getline(file, line);) lines.push_back(line);
        return lines;
    }

    // Trailing whitespace trimmed, blank lines dropped, and `#` comments dropped unless the
    // caller wants them. THREE parity binaries had each written this for themselves --
    // identity, gate and now cluster -- which is the rule-of-three this header exists to stop:
    // a reading contract in three places is three contracts one edit apart.
    inline std::vector<std::string> read_lines(const std::string& path, bool keep_comments) {
        std::ifstream file(path);
        if (!file) throw ConfigError("cannot read " + path);
        std::vector<std::string> lines;
        for (std::string line; std::getline(file, line);) {
            while (!line.empty() && (line.back() == ' ' || line.back() == '\r'))
                line.pop_back();
            if (line.empty()) continue;
            if (!keep_comments && line[0] == '#') continue;
            lines.push_back(line);
        }
        return lines;
    }

    // DELEGATES, so there is one contract and not two overloads of one name with different
    // trimming -- #221's review caught that the new two-argument form trimmed trailing
    // whitespace and this one did not, while five binaries call this one.
    inline std::vector<std::string> read_lines(const std::string& path) {
        return read_lines(path, true);
    }

}  // namespace shipinfer::parity
