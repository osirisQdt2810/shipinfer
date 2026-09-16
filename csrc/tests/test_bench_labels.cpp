// `obs/device_labels.h`: what a run calls each device, and when it refuses to guess.
//
// This exists for the reason `test_bench_models.cpp` does. The mapping lived in
// `cli/bench.cpp`'s anonymous namespace -- a `main()` translation unit no gate can link -- and
// #294 round 1 got it WRONG: labels were applied positionally, so `6,3,2` printed host 2's
// counters under `6:`, an inverted `per_device` table in the one file whose value is cross-run
// comparison, and nothing in the tree failed.
//
// Offline: g++ alone, no CUDA, no TensorRT.

#include <cstdio>
#include <string>
#include <vector>

#include "shipinfer/obs/device_labels.h"

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

    void no_labels_means_every_device_keeps_its_own_number() {
        // Every run that narrows nothing takes this path, so it has to be the identity.
        const std::vector<int> ordinals{1, 3, 4, 6};
        const std::vector<int> none{};
        for (int device : ordinals)
            check(label_of(ordinals, none, device) == device, "unlabelled device keeps its id");
        check(labels_refusal(ordinals, none).empty(), "and no labels is not a refusal");
    }

    void a_label_is_read_by_position_in_the_pair() {
        // The narrowed shape: ordinals 0..N-1, labels the host ids in the SAME order. Sorting
        // is the shell's job (`deploy/rootless/_narrow.sh`); this asserts the pairing only.
        const std::vector<int> ordinals{0, 1, 2};
        const std::vector<int> labels{2, 3, 6};
        check(label_of(ordinals, labels, 0) == 2, "ordinal 0 reports as host 2");
        check(label_of(ordinals, labels, 1) == 3, "ordinal 1 reports as host 3");
        check(label_of(ordinals, labels, 2) == 6, "ordinal 2 reports as host 6");
    }

    void a_device_outside_the_run_is_reported_as_itself() {
        // An error path can print a device this run never opened; mangling it into whatever
        // label happens to sit at that position would be worse than reporting the number.
        const std::vector<int> ordinals{0, 1};
        const std::vector<int> labels{4, 5};
        check(label_of(ordinals, labels, 7) == 7, "an unknown device is not relabelled");
    }

    void a_short_label_list_is_refused_and_says_both_sizes() {
        // THE DANGEROUS CASE: a short list would relabel the leading devices and leave the
        // rest reporting ordinals -- an output a reader cannot tell from a correct one.
        const std::vector<int> ordinals{0, 1, 2};
        const std::string refusal = labels_refusal(ordinals, {2, 3});
        check(!refusal.empty(), "two labels for three devices is refused");
        check(refusal.find("has 2 entries") != std::string::npos, "and names the label count");
        check(refusal.find("--gpu-ids has 3") != std::string::npos, "and the device count");
    }

    void a_long_label_list_is_refused_too() {
        check(!labels_refusal({0, 1}, {2, 3, 6}).empty(), "three labels for two devices");
    }

    void one_label_per_device_is_accepted() {
        check(labels_refusal({0, 1, 2}, {2, 3, 6}).empty(), "a label per device is fine");
    }

}  // namespace

int main() {
    try {
        no_labels_means_every_device_keeps_its_own_number();
        a_label_is_read_by_position_in_the_pair();
        a_device_outside_the_run_is_reported_as_itself();
        a_short_label_list_is_refused_and_says_both_sizes();
        a_long_label_list_is_refused_too();
        one_label_per_device_is_accepted();
    } catch (const std::exception& error) {
        std::printf("FAIL: uncaught: %s\n", error.what());
        ++failures;
    }
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
