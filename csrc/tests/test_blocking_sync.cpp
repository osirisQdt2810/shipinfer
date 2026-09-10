// `core/env.h`'s two readings of one variable, through the knob they exist for: whether a
// device's synchronisation blocks or spins.
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

    void unset_means_off_for_the_plain_rule() {
        // `env_flag` is the ON-ONLY rule and stays that way: every other knob reads it, and a
        // knob nobody set must not turn on. The blocking-sync knob no longer uses it -- see
        // the default-on cases below.
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

    void with_default_on(const char* value, bool expected, const std::string& what) {
        if (value == nullptr) {
            unsetenv("SHIPINFER_CUDA_BLOCKING_SYNC");
        } else {
            setenv("SHIPINFER_CUDA_BLOCKING_SYNC", value, 1);
        }
        check(env_flag_unless_refused("SHIPINFER_CUDA_BLOCKING_SYNC") == expected, what);
    }

    void the_default_on_rule_is_refused_only_by_zero() {
        // The knob the two-load measurement flipped: ON unless refused. Only `0` refuses, and
        // EMPTY does not -- which is why this is a second function rather than `!env_flag`:
        // `docker run -e VAR` forwards an unset host variable as empty, so reading empty as a
        // refusal would disable a default-on knob on every containerised run.
        with_default_on(nullptr, true, "unset -> on (the default)");
        with_default_on("", true, "empty (docker -e with nothing set) -> on (the default)");
        with_default_on("0", false, "0 -> off, the only refusal");
        with_default_on("1", true, "1 -> on");
        with_default_on("no", true, "anything that is not 0 -> on");
    }

    void the_two_rules_disagree_exactly_where_they_should() {
        // Both readings of one variable, so the difference is a test rather than a comment:
        // they agree on `0` and on any non-empty non-zero value, and differ on absent/empty.
        with_env(nullptr, false, "plain: unset -> off");
        with_default_on(nullptr, true, "default-on: unset -> on");
        with_env("0", false, "plain: 0 -> off");
        with_default_on("0", false, "default-on: 0 -> off");
        with_env("1", true, "plain: 1 -> on");
        with_default_on("1", true, "default-on: 1 -> on");
    }

}  // namespace

int main() {
    unset_means_off_for_the_plain_rule();
    the_docker_e_flag_shape_is_off();
    zero_means_off();
    anything_else_means_on();
    the_default_on_rule_is_refused_only_by_zero();
    the_two_rules_disagree_exactly_where_they_should();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
