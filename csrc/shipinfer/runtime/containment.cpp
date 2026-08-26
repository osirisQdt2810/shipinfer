#include "shipinfer/runtime/containment.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "shipinfer/core/types.h"

namespace shipinfer::runtime {

    namespace {

        std::string read_all(const char* path) {
            std::ifstream in(path);
            if (!in) return {};
            std::ostringstream out;
            out << in.rdbuf();
            return out.str();
        }

        bool file_exists(const char* path) {
            std::ifstream in(path);
            return static_cast<bool>(in);
        }

        bool cgroup_names_a_runtime(const std::string& text) {
            for (const char* needle :
                 {"docker", "containerd", "kubepods", "libpod", "podman", "lxc"}) {
                if (text.find(needle) != std::string::npos) return true;
            }
            return false;
        }

        // `/proc/self/mountinfo`: the root mount's filesystem type is the first field after
        // the " - " separator; the mount point is the fifth field before it.
        bool root_is_overlay(const std::string& text) {
            std::istringstream lines(text);
            std::string line;
            while (std::getline(lines, line)) {
                const auto sep = line.find(" - ");
                if (sep == std::string::npos) continue;
                std::istringstream head(line.substr(0, sep));
                std::string field, mount_point;
                for (int i = 0; i < 5 && head >> field; ++i) mount_point = field;
                std::istringstream tail(line.substr(sep + 3));
                std::string fs_type;
                tail >> fs_type;
                if (mount_point == "/") return fs_type == "overlay" || fs_type == "overlayfs";
            }
            return false;
        }

    }  // namespace

    bool Containment::in_container() const {
        // Mirrors `containment.py`: only "1" and "0" assert anything; any other value falls
        // through to the signals, so `SHIPINFER_IN_CONTAINER=true` inside a real container
        // does not refuse here while the Python gate lets it run.
        if (forced.has_value() && *forced == "1") return true;
        if (forced.has_value() && *forced == "0") return false;
        // Two of three. One is not enough: the marker is a file, and `touch /.dockerenv` once
        // let a host run certify itself.
        return signals() >= 2;
    }

    std::string Containment::describe() const {
        std::string out = std::string("marker=") + (marker ? "true" : "false") +
                          " cgroup=" + (cgroup ? "true" : "false") +
                          " overlay_root=" + (overlay_root ? "true" : "false");
        if (forced.has_value()) out += " forced=" + *forced;
        return out;
    }

    Containment classify(bool marker_present, const std::string& cgroup_text,
                         const std::string& mountinfo_text, const char* forced) {
        Containment c;
        c.marker = marker_present;
        c.cgroup = cgroup_names_a_runtime(cgroup_text);
        c.overlay_root = root_is_overlay(mountinfo_text);
        if (forced != nullptr) c.forced = std::string(forced);
        return c;
    }

    Containment detect() {
        const bool marker = file_exists("/.dockerenv") || file_exists("/run/.containerenv");
        return classify(marker, read_all("/proc/1/cgroup"), read_all("/proc/self/mountinfo"),
                        std::getenv(kForceEnv));
    }

    void require_container(const char* what) {
        const Containment c = detect();
        if (c.in_container()) return;
        const char* allow = std::getenv(kAllowHostRunEnv);
        if (allow != nullptr && std::string(allow) == "1") {
            std::fprintf(stderr,
                         "%s: running on the host under %s=1 (%s) — a host number is not a "
                         "production number; say so in the report\n",
                         what, kAllowHostRunEnv, c.describe().c_str());
            return;
        }
        throw ConfigError(std::string(what) + ": refused outside a container (" + c.describe() +
                          "). The rule: anything touching an accelerator runs inside a "
                          "container — use deploy/rootless/cpp.sh or bench.sh. If the operator "
                          "has agreed to a host run, set " +
                          kAllowHostRunEnv + "=1 and say so in the report.");
    }

}  // namespace shipinfer::runtime
