// Exponential backoff with jitter and a cap — `ingest/timing/backoff.py`.
//
// The reference implementation retried a dead camera every twenty seconds, forever, from a
// shared monitor thread. Two things go wrong with that. A camera that blipped for 200 ms stays
// dark for twenty seconds; and fifty cameras behind one switch retry in lockstep, so the switch
// coming back up is met with fifty simultaneous RTSP DESCRIBEs, which is how a recovery turns
// into a second outage.
//
// Exponential growth fixes the first, jitter fixes the second, and the cap keeps a camera that
// has been down all night from waiting hours to notice it is back.
#pragma once

#include <cstdint>
#include <random>

namespace shipinfer {

    // Successive retry delays: `initial`, `initial * factor`, ... capped at `cap`.
    //
    // Pure and hardware-free, so the delay *sequence* is asserted in the offline tier rather
    // than inferred from a log. That matters: "it retried" is easy to observe and says nothing,
    // while "it retried at 0.5 s, 1 s, 2 s, 4 s, 8 s, 16 s, 30 s, 30 s" is the actual policy.
    class ExponentialBackoff {
      public:
        // `jitter` is the fraction of each delay removed at random, in [0, 1). 0.2 means the
        // returned delay is uniform in [0.8 d, d] — subtractive rather than additive, so the
        // cap is a real bound.
        //
        // `seed` of 0 seeds from the platform's entropy: fifty cameras in one process must not
        // draw the same sequence, which would defeat the whole purpose of jittering. A non-zero
        // seed is for a test that wants a repeatable draw.
        //
        // Throws ConfigError on any input that would make the backoff not back off.
        explicit ExponentialBackoff(double initial_s = 0.5, double cap_s = 30.0,
                                    double factor = 2.0, double jitter = 0.2,
                                    uint64_t seed = 0);

        // Delays handed out since the last `reset` — the consecutive-failure count.
        int attempts() const { return attempts_; }

        // The next delay's un-jittered value, without consuming an attempt.
        double peek() const;

        // The next delay, in seconds, and advance the sequence.
        double next_delay();

        // Back to the first delay. Called the moment a *frame* arrives — never on a successful
        // connect: an RTSP source that accepts a connection and then delivers nothing is the
        // most common real failure mode of a camera fleet, and treating "opened" as "healthy"
        // is precisely how it stays invisible.
        void reset() { attempts_ = 0; }

        double initial_s() const { return initial_s_; }
        double cap_s() const { return cap_s_; }

      private:
        double initial_s_;
        double cap_s_;
        double factor_;
        double jitter_;
        // The attempt at which the cap is reached, precomputed so `factor ^ attempts` is never
        // evaluated for a large exponent. This is not hypothetical: a camera down overnight
        // reaches attempt ~1000 at 30 s apiece, `std::pow(2.0, 1000)` is `inf`, and `min(cap,
        // inf)` happens to be right only by luck — one sign change away from an infinite delay
        // on the one camera that most needs to still be trying.
        int ceiling_;
        int attempts_ = 0;
        std::mt19937_64 rng_;
    };

}  // namespace shipinfer
