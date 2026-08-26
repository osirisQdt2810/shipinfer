// Is this process inside a container? Mirrors `runtime/containment.py`, signal for signal.
//
// The rule (CLAUDE.md, "Where commands run") is that anything touching an accelerator runs
// inside a container, and the Python side enforces it in the process that would do the work.
// The bench binary used to be the one way around it: run directly, it passed both enforcement
// points. So it asks here, before it opens a device.
//
// A container is established by *agreement* between independent signals, two of three: the
// marker file (`/.dockerenv` or `/run/.containerenv` — a file anyone can `touch`, so never
// alone), pid 1's cgroup naming a container runtime, and the root filesystem being an overlay.
// Pid 1 rather than self because `--pid=host` puts self in the host's cgroup from inside a
// container, which is the configuration `deploy/rootless/` uses. `SHIPINFER_IN_CONTAINER=1`
// forces the answer for an image that shows none of the signals.
#pragma once

#include <optional>
#include <string>

namespace shipinfer::runtime {

    struct Containment {
        bool marker = false;
        bool cgroup = false;
        bool overlay_root = false;
        std::optional<std::string> forced;

        int signals() const { return int(marker) + int(cgroup) + int(overlay_root); }
        bool in_container() const;
        std::string describe() const;
    };

    // Pure: classify from the texts the detector would read, so the rule is testable without
    // a container. `cgroup_text` is `/proc/1/cgroup`; `mountinfo_text` is
    // `/proc/self/mountinfo`.
    Containment classify(bool marker_present, const std::string& cgroup_text,
                         const std::string& mountinfo_text, const char* forced);

    // Reads the real files and the environment.
    Containment detect();

    // Throws `ConfigError` naming the rule and the override unless this process is in a
    // container or `SHIPINFER_ALLOW_HOST_RUN=1` is set — in which case it says so on stderr,
    // because the report has to say the override was used.
    void require_container(const char* what);

    inline constexpr const char* kAllowHostRunEnv = "SHIPINFER_ALLOW_HOST_RUN";
    inline constexpr const char* kForceEnv = "SHIPINFER_IN_CONTAINER";

}  // namespace shipinfer::runtime
