// `core/thread_name.h`: every thread this plane starts says what it is.
//
// The Python plane names all six of its threads; this one named none, so `top -H` and
// `/proc/<tid>/comm` showed fifty-odd rows called `bench`. `NOT-GPU-BOUND-AT-FIVE-GPUS`
// measured the wall as host CPU and left "which threads spend it" open, and per-thread
// accounting reads names -- so this is the prerequisite, not a cosmetic.
//
// Offline: g++ alone, no CUDA, no TensorRT.

#include <pthread.h>

#include <cstdio>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/core/thread_name.h"

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

    /// What the kernel will report for this thread, which is the only reader that matters.
    std::string as_the_kernel_sees_it() {
        char buffer[32] = {};
        pthread_getname_np(pthread_self(), buffer, sizeof(buffer));
        return std::string(buffer);
    }

    std::string named_on_its_own_thread(const std::string& name) {
        std::string seen;
        std::thread worker([&]() {
            name_this_thread(name);
            seen = as_the_kernel_sees_it();
        });
        worker.join();
        return seen;
    }

    void a_short_name_arrives_whole() {
        check(named_on_its_own_thread("sweeper") == "sweeper", "sweeper survives");
        check(named_on_its_own_thread("sampler") == "sampler", "sampler survives");
        check(named_on_its_own_thread("pipe-12") == "pipe-12", "a worker's index survives");
        check(named_on_its_own_thread("cam-cam-000") == "cam-cam-000", "a camera id survives");
    }

    void a_long_name_keeps_its_head() {
        // Fifteen usable bytes, and the class prefix plus the discriminator's start is what
        // a reader needs: `mdl-ship_detect` and `mdl-ship_segmen` are two readable rows.
        // Keeping the TAIL instead would give `-ship_detector` and drop the class.
        check(named_on_its_own_thread(thread_name("mdl", "ship_detector")) == "mdl-ship_detect",
              "a long model name is truncated from the end");
        check(
            named_on_its_own_thread(thread_name("mdl", "ship_segmenter")) == "mdl-ship_segmen",
            "and the next one is still distinguishable from it");
        check(
            named_on_its_own_thread(thread_name("mdl", "person_embedder")) == "mdl-person_embe",
            "as is the third");
    }

    void the_budget_is_the_kernels_and_not_a_guess() {
        // 16 bytes including the NUL: `pthread_setname_np` returns ERANGE for anything
        // longer and the name would silently not be set at all.
        const std::string too_long(kThreadNameMax + 1, 'x');
        check(pthread_setname_np(pthread_self(), too_long.c_str()) != 0,
              "one byte over the cap is refused by the kernel");
        check(named_on_its_own_thread(too_long).size() == kThreadNameMax,
              "so the helper must truncate rather than pass it through");
    }

    void every_name_fits_the_budget() {
        // The five call sites' widest realistic arguments, so a new long model name shows up
        // here rather than as a silently unnamed thread. The model one goes through
        // `instance_thread_label` because that is what the runtime calls: `bench.cpp` builds
        // `ModelInstance` with `model:device:index`, not with the bare model name.
        for (const std::string& name :
             {std::string("sweeper"), std::string("sampler"), thread_name("pipe", "127"),
              thread_name("cam", "camera-0049"),
              instance_thread_label("person_embedder:7:1")}) {
            check(named_on_its_own_thread(name).size() <= kThreadNameMax, name + " fits");
            check(!named_on_its_own_thread(name).empty(), name + " is actually set");
        }
    }

    void every_instance_on_every_device_gets_its_own_name() {
        // THE FAILURE THE FIRST DRAFT SHIPPED. `instance.cpp` named the worker
        // `thread_name("mdl", name_)`, and `name_` is `model:device:index` -- so
        // `mdl-ship_detector:0:0` and `mdl-ship_detector:4:0` both became `mdl-ship_detect`
        // and a five-GPU run's twenty instance threads shared four names. The device index
        // was the one thing truncation ate, and "which device's instances are hot" is the
        // question `NOT-GPU-BOUND-AT-FIVE-GPUS` left open.
        //
        // Enumerated through the REAL function over the real repository's four models, eight
        // devices and two instances each, because the first draft's test truncated
        // `mdl-<model>` -- a string the runtime never builds -- and so passed straight
        // through the defect it claimed to guard.
        const std::vector<std::string> models{"ship_detector", "ship_segmenter",
                                              "ship_embedder", "person_embedder"};
        std::set<std::string> seen;
        size_t built = 0;
        for (const std::string& model : models) {
            for (int device = 0; device < 8; ++device) {
                for (int index = 0; index < 2; ++index) {
                    const std::string name =
                        model + ":" + std::to_string(device) + ":" + std::to_string(index);
                    const std::string label = instance_thread_label(name);
                    check(label.size() <= kThreadNameMax, name + " -> " + label + " fits");
                    check(named_on_its_own_thread(label) == label,
                          name + " survives the kernel whole");
                    seen.insert(label);
                    ++built;
                }
            }
        }
        check(seen.size() == built, "every (model, device, instance) has its own name: " +
                                        std::to_string(seen.size()) + " of " +
                                        std::to_string(built));
    }

    void a_name_without_a_device_keeps_the_ordinary_scheme() {
        // `test_engine.cpp` builds instances called `m:0`, and other callers may pass a bare
        // model name; neither may lose its class prefix.
        check(instance_thread_label("m:0") == "m0-m", instance_thread_label("m:0"));
        check(instance_thread_label("ship_detector") == "mdl-ship_detect",
              instance_thread_label("ship_detector"));
    }

    void the_main_thread_is_left_alone() {
        // Naming a thread is the thread's own business: the helper takes `pthread_self()`, so
        // a worker cannot rename the process's main thread by accident.
        const std::string before = as_the_kernel_sees_it();
        named_on_its_own_thread("pipe-0");
        check(as_the_kernel_sees_it() == before, "the caller's own name is untouched");
    }

}  // namespace

int main() {
    a_short_name_arrives_whole();
    a_long_name_keeps_its_head();
    the_budget_is_the_kernels_and_not_a_guess();
    every_name_fits_the_budget();
    every_instance_on_every_device_gets_its_own_name();
    a_name_without_a_device_keeps_the_ordinary_scheme();
    the_main_thread_is_left_alone();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
