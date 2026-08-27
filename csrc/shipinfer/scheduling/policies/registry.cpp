#include "shipinfer/scheduling/policies/registry.h"

#include <sstream>

#include "shipinfer/core/types.h"

namespace shipinfer {

    void PolicyRegistry::add(const std::string& name, const std::vector<std::string>& aliases,
                             const std::string& description, PolicyFactory factory) {
        if (entries_.count(name) != 0 || by_alias_.count(name) != 0) {
            throw ConfigError("placement policy " + name + " is registered twice");
        }
        entries_[name] = Entry{description, std::move(factory)};
        for (const std::string& alias : aliases) {
            if (entries_.count(alias) != 0 || by_alias_.count(alias) != 0) {
                throw ConfigError("placement policy alias " + alias + " is already taken");
            }
            by_alias_[alias] = name;
        }
    }

    std::string PolicyRegistry::canonical(const std::string& name) const {
        if (entries_.count(name) != 0) return name;
        auto alias = by_alias_.find(name);
        if (alias != by_alias_.end()) return alias->second;
        std::ostringstream known;
        for (const auto& [n, _] : entries_) known << (known.tellp() > 0 ? ", " : "") << n;
        throw ConfigError("unknown placement policy '" + name +
                          "'; known policies: " + known.str());
    }

    bool PolicyRegistry::contains(const std::string& name) const {
        return entries_.count(name) != 0 || by_alias_.count(name) != 0;
    }

    std::unique_ptr<PlacementPolicy> PolicyRegistry::build(const std::string& name,
                                                           const PolicyOptions& options) const {
        return entries_.at(canonical(name)).factory(options);
    }

    std::vector<std::string> PolicyRegistry::names() const {
        std::vector<std::string> out;
        for (const auto& [name, _] : entries_) out.push_back(name);
        return out;
    }

    std::vector<std::pair<std::string, std::string>> PolicyRegistry::describe() const {
        std::vector<std::pair<std::string, std::string>> out;
        for (const auto& [name, entry] : entries_) out.emplace_back(name, entry.description);
        return out;
    }

    PolicyRegistry& POLICIES() {
        static PolicyRegistry registry;
        return registry;
    }

    std::unique_ptr<PlacementPolicy> build_policy(const std::string& name,
                                                  const PolicyOptions& options) {
        return POLICIES().build(name, options);
    }

}  // namespace shipinfer
