#pragma once

// The Python plane names every thread it starts; this plane named none, so `top -H`, a
// backtrace and `/proc/<tid>/comm` showed fifty-odd identical rows called `bench`. That
// matters more here than it looks: `NOT-GPU-BOUND-AT-FIVE-GPUS` measured the wall as host
// CPU and left "which threads spend it" unanswered, and per-thread accounting reads names.

#include <pthread.h>

#include <string>
#include <string_view>

namespace shipinfer {

    /// Linux caps a thread name at 16 bytes *including* the NUL, so fifteen are usable.
    inline constexpr std::size_t kThreadNameMax = 15;

    /// Name the calling thread. Must run ON that thread: `pthread_self()`.
    ///
    /// Truncated from the END, which is the opposite of the obvious choice and the reason is
    /// the scheme: a short class prefix (`mdl-`, `cam-`, `pipe-`) then the discriminator, so
    /// the head is what distinguishes. `mdl-ship_detect`, `mdl-ship_segmen` and
    /// `mdl-person_embe` are three readable rows; keeping the tail instead would give
    /// `-ship_detector` and drop the class. A name is for a human and for accounting, never
    /// for identity -- the tid is that -- so a collision between two long names is a
    /// legibility cost and nothing more.
    inline void name_this_thread(std::string_view name) {
        const std::string_view kept = name.substr(0, kThreadNameMax);
        pthread_setname_np(pthread_self(), std::string(kept).c_str());
    }

    /// `<prefix>-<discriminator>`, the scheme both planes use, fitted to the budget above.
    inline std::string thread_name(std::string_view prefix, std::string_view discriminator) {
        return std::string(prefix) + "-" + std::string(discriminator);
    }

    /// The label for a model instance, whose name is `model[:device[:index]]`.
    ///
    /// THE DEVICE GOES IN THE HEAD, which inverts the rule above, and the inversion is the
    /// point: at this one site the head is the SHARED part. `mdl-ship_detector:0:0` and
    /// `mdl-ship_detector:4:0` both truncate to `mdl-ship_detect`, so a five-GPU run's twenty
    /// instance threads would share four names -- and "which DEVICE's instances are hot" is
    /// exactly the question the accounting exists to answer. `m0.0-ship_detec` keeps all
    /// three. A name with no `:` at all keeps the ordinary scheme.
    ///
    /// Returns the label already cut to the budget, unlike `thread_name`: what has to be
    /// distinct is the string the kernel ends up holding, so that is the string a caller and
    /// a test should be comparing. `name_this_thread` truncating again is then a no-op.
    inline std::string instance_thread_label(std::string_view name) {
        const std::size_t first = name.find(':');
        if (first == std::string_view::npos) {
            return thread_name("mdl", name).substr(0, kThreadNameMax);
        }
        std::string tail(name.substr(first + 1));
        for (char& character : tail) {
            if (character == ':') character = '.';
        }
        return thread_name("m" + tail, name.substr(0, first)).substr(0, kThreadNameMax);
    }
}  // namespace shipinfer
