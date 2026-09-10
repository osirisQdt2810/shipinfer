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

namespace shipinfer {

    /// Whether ``name`` asks for something: set, non-empty, and not ``0``.
    inline bool env_flag(std::string_view name) {
        const char* value = std::getenv(std::string(name).c_str());
        return value != nullptr && *value != '\0' && std::string_view(value) != "0";
    }
}  // namespace shipinfer
