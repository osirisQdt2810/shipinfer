// `core/env.h`'s on/off reading, through the switch it exists for: `SHIPINFER_CUDA_GRAPHS`,
// which both planes read with one spelling (`src/shipinfer/envs.py`). Offline: g++ alone.

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

    void set(const char* value) {
        if (value == nullptr) {
            unsetenv("SHIPINFER_CUDA_GRAPHS");
        } else {
            setenv("SHIPINFER_CUDA_GRAPHS", value, 1);
        }
    }

    void unset_and_empty_keep_the_default() {
        // Empty is what `docker run -e VAR` forwards when VAR is unset on the host.
        for (const char* value : {static_cast<const char*>(nullptr), "", "  "}) {
            set(value);
            check(env_on_off("SHIPINFER_CUDA_GRAPHS", true),
                  "unset, empty or blank keeps a default of on");
            check(!env_on_off("SHIPINFER_CUDA_GRAPHS", false),
                  "and a default of off, so neither reading is baked in");
        }
    }

    void on_and_off_win_over_the_default() {
        set("off");
        check(!env_on_off("SHIPINFER_CUDA_GRAPHS", true), "off turns a default-on switch off");
        set("on");
        check(env_on_off("SHIPINFER_CUDA_GRAPHS", false), "on turns a default-off switch on");
        set(" on ");
        check(env_on_off("SHIPINFER_CUDA_GRAPHS", false), "trimmed first, as `envs.py` trims");
    }

    void anything_else_is_refused_by_name() {
        // `1` and `0` included: `env_flag` reads those, and this switch does not, so a reader
        // who guessed the other spelling must hear so rather than get the default.
        for (const char* value : {"1", "0", "true", "OFF"}) {
            set(value);
            std::string message;
            try {
                (void)env_on_off("SHIPINFER_CUDA_GRAPHS", true);
            } catch (const ConfigError& error) {
                message = error.what();
            }
            check(message.find("SHIPINFER_CUDA_GRAPHS") != std::string::npos &&
                      message.find("on or off") != std::string::npos,
                  std::string("'") + value + "' is refused, naming the variable: " +
                      (message.empty() ? "(no throw)" : message));
        }
        set(nullptr);
    }

}  // namespace

int main() {
    unset_and_empty_keep_the_default();
    on_and_off_win_over_the_default();
    anything_else_is_refused_by_name();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
