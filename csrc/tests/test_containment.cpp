// The container agreement, classified from texts — no container needed to test the rule.
#include <cstdio>
#include <string>

#include "shipinfer/runtime/containment.h"

namespace {

    int failures = 0;
    int checks = 0;

    void check(bool ok, const char* what) {
        ++checks;
        if (!ok) {
            ++failures;
            std::printf("FAIL: %s\n", what);
        }
    }

    const std::string kHostCgroup = "0::/init.scope\n";
    const std::string kDockerCgroup = "0::/system.slice/docker-3f1c2a.scope\n";
    const std::string kHostMounts =
        "22 1 8:2 / / rw,relatime shared:1 - ext4 /dev/sda2 rw\n"
        "23 22 0:21 / /proc rw,nosuid - proc proc rw\n";
    const std::string kOverlayMounts =
        "500 400 0:60 / / rw,relatime - overlay overlay rw,lowerdir=/a,upperdir=/b\n"
        "501 500 0:61 / /proc rw,nosuid - proc proc rw\n";

}  // namespace

int main() {
    using shipinfer::runtime::classify;

    check(!classify(true, kHostCgroup, kHostMounts, nullptr).in_container(),
          "the marker alone is a file anyone can touch");
    check(!classify(false, kDockerCgroup, kHostMounts, nullptr).in_container(),
          "the cgroup alone is not enough either");
    check(classify(true, kDockerCgroup, kHostMounts, nullptr).in_container(),
          "marker + cgroup is the agreement");
    check(classify(false, kDockerCgroup, kOverlayMounts, nullptr).in_container(),
          "cgroup + overlay root is the agreement (--pid=host hides the marker's friend)");
    check(classify(true, kHostCgroup, kOverlayMounts, nullptr).in_container(),
          "marker + overlay root is the agreement");
    check(!classify(false, kHostCgroup, kHostMounts, nullptr).in_container(), "a plain host");
    check(classify(false, kHostCgroup, kHostMounts, "1").in_container(),
          "SHIPINFER_IN_CONTAINER=1 forces the answer");
    check(!classify(true, kDockerCgroup, kOverlayMounts, "0").in_container(),
          "SHIPINFER_IN_CONTAINER=0 forces the other answer");
    check(classify(true, kDockerCgroup, kHostMounts, "true").in_container(),
          "any other value is ignored: the signals decide (container)");
    check(!classify(false, kHostCgroup, kHostMounts, "true").in_container(),
          "any other value is ignored: the signals decide (host)");
    for (const char* needle : {"containerd", "kubepods", "libpod", "podman", "lxc"}) {
        check(
            classify(false, std::string("0::/x/") + needle + "/y", kHostMounts, nullptr).cgroup,
            "every container runtime name is a cgroup signal");
    }
    check(!classify(false, kHostCgroup, "garbage without separator\n", nullptr).overlay_root,
          "an unparseable mountinfo is not an overlay root");
    const auto described = classify(true, kDockerCgroup, kHostMounts, nullptr).describe();
    check(described.find("marker=true") != std::string::npos &&
              described.find("cgroup=true") != std::string::npos &&
              described.find("overlay_root=false") != std::string::npos,
          "describe() names all three signals");

    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
