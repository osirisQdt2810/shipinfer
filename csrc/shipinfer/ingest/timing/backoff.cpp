#include "shipinfer/ingest/timing/backoff.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "shipinfer/core/types.h"

namespace shipinfer {

    ExponentialBackoff::ExponentialBackoff(double initial_s, double cap_s, double factor,
                                           double jitter, uint64_t seed)
        : initial_s_(initial_s), cap_s_(cap_s), factor_(factor), jitter_(jitter) {
        if (!(initial_s > 0.0)) throw ConfigError("backoff initial_s must be > 0");
        if (cap_s < initial_s) throw ConfigError("backoff cap_s must be >= initial_s");
        if (factor <= 1.0) throw ConfigError("backoff factor must be > 1");
        if (jitter < 0.0 || jitter >= 1.0)
            throw ConfigError("backoff jitter must be in [0, 1)");
        ceiling_ =
            cap_s <= initial_s
                ? 0
                : static_cast<int>(std::ceil(std::log(cap_s / initial_s) / std::log(factor)));
        // Seeded from entropy *plus* the clock: fifty cameras constructed in one loop must not
        // draw the same sequence, which is the whole point of jittering them.
        rng_.seed(seed != 0
                      ? seed
                      : (static_cast<uint64_t>(std::random_device{}()) << 32) ^
                            static_cast<uint64_t>(
                                std::chrono::steady_clock::now().time_since_epoch().count()));
    }

    double ExponentialBackoff::peek() const {
        if (attempts_ >= ceiling_) return cap_s_;
        return std::min(cap_s_, initial_s_ * std::pow(factor_, attempts_));
    }

    double ExponentialBackoff::next_delay() {
        const double base = peek();
        ++attempts_;
        if (jitter_ <= 0.0) return base;
        // Uniform in [0, 1), then subtractive: the result is in [(1 - jitter) * base, base],
        // which is why the cap is a bound the delay can reach but never exceed.
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        return base * (1.0 - jitter_ * unit(rng_));
    }

}  // namespace shipinfer
