#include "shipinfer/core/options.h"

#include "shipinfer/core/types.h"

namespace shipinfer {

    int option_int(const KeywordOptions& options, const std::string& key, int fallback) {
        auto it = options.find(key);
        if (it == options.end()) return fallback;
        try {
            return std::stoi(it->second);
        } catch (const std::exception&) {
            throw ConfigError("placement policy option " + key + " must be an integer, got '" +
                              it->second + "'");
        }
    }

    void refuse_unknown_options(const std::string& policy, const KeywordOptions& options,
                                const std::vector<std::string>& accepted) {
        for (const auto& [key, _] : options) {
            bool known = false;
            for (const std::string& name : accepted) known = known || name == key;
            if (!known) {
                throw ConfigError(policy + ": unknown option '" + key +
                                  "' — the constructor does not take it");
            }
        }
    }

}  // namespace shipinfer
