// `core/env.h`'s `env_flag`, read here through the knob it exists for: whether a device's
// synchronisation blocks or spins.
//
// Measured (`THE-INSTANCE-THREADS-SPIN-ON-cudaStreamSynchronize`): with it on, the
// model-instance threads' host CPU HALVES -- 632 -> 279 CPU-s over two interleaved pairs at
// the design load -- and events rise ~15%, because the spin was starving the pipeline
// workers. It is off by default because it trades wake-up latency for host CPU, and the host
// is only the wall at this load.
//
// Offline: g++ alone, no CUDA, no TensorRT, no device. Only the ENV PARSE is testable here --
// the flag itself needs a driver, and the A/B that justifies it is in the ledger.

#include <cstdio>
#include <cstdlib>
#include <string>

#include "shipinfer/core/env.h"

namespace {

    using namespace shipinfer;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    void with_env(const char* value, bool expected, const std::string& what) {
        if (value == nullptr) {
            unsetenv("SHIPINFER_CUDA_BLOCKING_SYNC");
        } else {
            setenv("SHIPINFER_CUDA_BLOCKING_SYNC", value, 1);
        }
        check(env_flag("SHIPINFER_CUDA_BLOCKING_SYNC") == expected, what);
    }

    void unset_means_spin() {
        // The default has to be the current behaviour: this changes a device-wide scheduling
        // decision, and a run that did not ask for it must measure what it measured before.
        with_env(nullptr, false, "unset -> off");
    }

    void the_docker_e_flag_shape_is_off() {
        // `docker run -e VAR` with VAR unset on the host passes it through as EMPTY rather
        // than not at all, which is how `deploy/rootless/cpp.sh` forwards it. An empty value
        // must therefore read as "not asked for" -- otherwise every container run would flip
        // a scheduling flag nobody set.
        with_env("", false, "empty (docker -e with nothing set) -> off");
    }

    void zero_means_off() {
        // So a wrapper can pass 0 rather than having to unset the variable.
        with_env("0", false, "0 -> off");
    }

    void anything_else_means_on() {
        with_env("1", true, "1 -> on");
        with_env("yes", true, "yes -> on");
        with_env("true", true, "true -> on");
    }

}  // namespace

int main() {
    unset_means_spin();
    the_docker_e_flag_shape_is_off();
    zero_means_off();
    anything_else_means_on();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
