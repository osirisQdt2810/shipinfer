// `core/percentile.h`, because an off-by-one in a rank misreports the tail it exists to show.
//
// The reassembly wait is now printed as p50/p95/p99 (`cli/bench.cpp`), and the whole reason it
// is percentiles and not a mean is the tail. A rank that is one out at n=100 reports the 98th
// as the 99th, which is invisible in a report and wrong in an argument.
//
// Offline: g++ alone, no CUDA, no device, no allocation past the vectors below.

#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "shipinfer/core/percentile.h"

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

    std::vector<uint32_t> one_to(uint32_t last) {
        std::vector<uint32_t> samples(last);
        std::iota(samples.begin(), samples.end(), 1u);
        return samples;
    }

    void an_empty_sample_reads_zero() {
        std::vector<uint32_t> none;
        check(percentile(none, 0.5) == 0, "empty p50 is 0");
        check(percentile(none, 0.99) == 0, "empty p99 is 0");
    }

    void one_sample_is_every_percentile() {
        std::vector<uint32_t> one{42};
        check(percentile(one, 0.0) == 42, "n=1 p0");
        check(percentile(one, 0.5) == 42, "n=1 p50");
        check(percentile(one, 1.0) == 42, "n=1 p100");
    }

    void the_ranks_are_the_rounded_rank_over_1_to_100() {
        // rank = round(p * 99) over 1..100, so the value is rank + 1. NOT classic nearest-rank,
        // which would give 50 for p50; the header says which convention and why.
        auto samples = one_to(100);
        check(percentile(samples, 0.50) == 51, "p50 of 1..100 is 51, not 50");
        samples = one_to(100);
        check(percentile(samples, 0.95) == 95, "p95 of 1..100 is 95");
        samples = one_to(100);
        check(percentile(samples, 0.99) == 99, "p99 of 1..100 is 99");
        samples = one_to(100);
        check(percentile(samples, 1.0) == 100, "p100 of 1..100 is the max");
        samples = one_to(100);
        check(percentile(samples, 0.0) == 1, "p0 of 1..100 is the min");
    }

    void the_order_of_the_input_does_not_matter() {
        std::vector<uint32_t> ascending{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<uint32_t> descending{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
        check(percentile(ascending, 0.9) == percentile(descending, 0.9), "p90 is order-free");
    }

    void a_fraction_past_the_end_cannot_index_past_it() {
        // Not a caller this file has, and a crash would be the wrong answer to a typo.
        auto samples = one_to(10);
        check(percentile(samples, 1.5) == 10, "p150 clamps to the max");
        samples = one_to(10);
        check(percentile(samples, -0.5) == 1, "a negative fraction clamps to the min");
    }

    void one_outlier_in_a_hundred_is_the_max_and_not_the_p99() {
        // The reason `report_latency` prints `max` beside the three ranks. Nearest-rank p99 of
        // 100 samples is the 99th of them, so a SINGLE slow frame never reaches it -- and a
        // reader who has only p99 would call that run clean.
        std::vector<uint32_t> samples(99, 1000);
        samples.push_back(900000);
        auto copy = samples;
        check(percentile(copy, 0.50) == 1000, "p50 ignores the outlier");
        copy = samples;
        check(percentile(copy, 0.99) == 1000, "p99 of 100 is the 99th, not the worst");
        copy = samples;
        check(percentile(copy, 1.0) == 900000, "only p100/max sees it");
    }

    void five_slow_in_a_hundred_do_reach_the_p99() {
        // And the complement, so the test above is a property of the rank rather than of the
        // outlier being large: five slow frames in a hundred move p99 and leave p95 alone.
        std::vector<uint32_t> samples(95, 1000);
        samples.insert(samples.end(), 5, 900000);
        auto copy = samples;
        check(percentile(copy, 0.95) == 1000, "p95 is still the fast frames");
        copy = samples;
        check(percentile(copy, 0.99) == 900000, "p99 is the slow tail");
    }

}  // namespace

int main() {
    an_empty_sample_reads_zero();
    one_sample_is_every_percentile();
    the_ranks_are_the_rounded_rank_over_1_to_100();
    the_order_of_the_input_does_not_matter();
    a_fraction_past_the_end_cannot_index_past_it();
    one_outlier_in_a_hundred_is_the_max_and_not_the_p99();
    five_slow_in_a_hundred_do_reach_the_p99();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
