// `POLICIES`, the placement-policy registry — `scheduling/policies/registry.py`.
//
// A new policy is a new file plus one `PolicyRegistrar` at the bottom of it; nothing else in
// the tree changes. That is the registry rule the Python plane runs on (CLAUDE.md, seam 1),
// and the reason a deployment picks a policy by *name* in its settings rather than by a branch.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/scheduling/policies/base.h"

namespace shipinfer {

    // Constructor keywords, as the settings tree carries them: `placement_policy_options`.
    using PolicyOptions = std::map<std::string, std::string>;
    using PolicyFactory = std::function<std::unique_ptr<PlacementPolicy>(const PolicyOptions&)>;

    class PolicyRegistry {
      public:
        void add(const std::string& name, const std::vector<std::string>& aliases,
                 const std::string& description, PolicyFactory factory);
        // The canonical name for a name or alias; throws `ConfigError` naming the alternatives.
        std::string canonical(const std::string& name) const;
        std::unique_ptr<PlacementPolicy> build(const std::string& name,
                                               const PolicyOptions& options = {}) const;
        std::vector<std::string> names() const;
        std::vector<std::pair<std::string, std::string>> describe() const;
        bool contains(const std::string& name) const;

      private:
        struct Entry {
            std::string description;
            PolicyFactory factory;
        };
        std::map<std::string, Entry> entries_;
        std::map<std::string, std::string> by_alias_;
    };

    // Function-local static: every translation unit's registrar can run before `main`, in any
    // order, and still find the one registry.
    PolicyRegistry& POLICIES();

    std::unique_ptr<PlacementPolicy> build_policy(const std::string& name,
                                                  const PolicyOptions& options = {});

    // One of these at the bottom of each policy file is the `@POLICIES.register(...)`.
    struct PolicyRegistrar {
        PolicyRegistrar(const std::string& name, const std::vector<std::string>& aliases,
                        const std::string& description, PolicyFactory factory) {
            POLICIES().add(name, aliases, description, std::move(factory));
        }
    };

    // Option parsing shared by the policies: a keyword the constructor does not take is a
    // configuration error, as the Python constructors' TypeError becomes one.
    int option_int(const PolicyOptions& options, const std::string& key, int fallback);
    void refuse_unknown_options(const std::string& policy, const PolicyOptions& options,
                                const std::vector<std::string>& accepted);

}  // namespace shipinfer
