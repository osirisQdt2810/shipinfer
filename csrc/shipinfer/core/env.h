#pragma once

// One reading of an environment flag, so two callers cannot disagree about what "set" means.
//
// `docker run -e VAR` with VAR unset on the host passes it through as EMPTY rather than not at
// all, which is how `deploy/rootless/*.sh` forward a knob. So an empty value has to read as
// "not asked for" -- otherwise every container run flips whatever the flag controls.
//
// Here and not in `platform.h`: the parse is pure config, and `--offline` refuses any unit
// whose closure reaches `platform.h`, which would leave it untested on a machine with no
// driver.

#include <cstdlib>
#include <string>
#include <string_view>

#include "shipinfer/core/types.h"

namespace shipinfer {

    /// Whether ``name`` asks for something: set, non-empty, and not ``0``.
    inline bool env_flag(std::string_view name) {
        const char* value = std::getenv(std::string(name).c_str());
        return value != nullptr && *value != '\0' && std::string_view(value) != "0";
    }

    /// The same reading for a knob that is ON unless refused: only ``0`` turns it off.
    ///
    /// Empty still means "not asked", so it takes the default rather than the refusal --
    /// which for a default-on knob is ON. That is the whole reason this is a second function
    /// and not `!env_flag`: `!env_flag` would read a container's empty pass-through as OFF and
    /// silently disable the knob on every containerised run.
    inline bool env_flag_unless_refused(std::string_view name) {
        const char* value = std::getenv(std::string(name).c_str());
        return value == nullptr || *value == '\0' || std::string_view(value) != "0";
    }

    /// A switch spelled ``on``/``off``, read as ``envs.py`` reads one: trimmed, and unset or
    /// blank is ``fallback``. Anything else throws ``ConfigError`` naming the variable, so a
    /// typo cannot pick a side.
    inline bool env_on_off(std::string_view name, bool fallback) {
        const char* value = std::getenv(std::string(name).c_str());
        std::string_view spelled = value == nullptr ? std::string_view() : value;
        const size_t first = spelled.find_first_not_of(" \t\n\r");
        if (first == std::string_view::npos) return fallback;
        spelled = spelled.substr(first, spelled.find_last_not_of(" \t\n\r") - first + 1);
        if (spelled == "on") return true;
        if (spelled == "off") return false;
        throw ConfigError(std::string(name) + " is '" + std::string(spelled) +
                          "'; it takes on or off");
    }
}  // namespace shipinfer
