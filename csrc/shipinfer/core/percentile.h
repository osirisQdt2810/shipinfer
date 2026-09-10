#pragma once

// Nearest-rank percentiles over a sample vector, for reporting a tail rather than a mean.
//
// A mean hides exactly what a 50-camera fleet is judged on, and CLAUDE.md's sizing section
// says so: the bottleneck "is not raw throughput, it is (a) load balance and (b) end-to-end
// latency". An EWMA is not a percentile either -- `InstanceStats::ewma_latency_us` already
// existed and cannot answer "what did the worst 1% wait".
//
// Here and not in `platform.h`: pure arithmetic, and `--offline` refuses any unit whose
// closure reaches that header, which would leave this untested where it is cheapest to test.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace shipinfer {

    /// Nearest-rank percentile of ``samples``, which this REORDERS. Empty reads as zero.
    ///
    /// `nth_element` and not a sort: linear, and the caller wants four ranks out of a vector
    /// it is finished with. `fraction` is clamped, so a caller cannot index past the end.
    inline uint32_t percentile(std::vector<uint32_t>& samples, double fraction) {
        if (samples.empty()) return 0;
        const double bounded = std::clamp(fraction, 0.0, 1.0);
        const auto last = static_cast<double>(samples.size() - 1);
        const auto rank = static_cast<std::size_t>(bounded * last + 0.5);
        const auto at = samples.begin() + static_cast<std::ptrdiff_t>(rank);
        std::nth_element(samples.begin(), at, samples.end());
        return *at;
    }

}  // namespace shipinfer
