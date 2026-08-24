// Bounded, camera-fair reassembly: every frame that is opened is reported exactly once.
//
// THE INVARIANT
// -------------
// A frame enters, some stages deliver, and eventually it leaves — complete, timed out, evicted
// or shut down, but it leaves. The previous generation had no such guarantee: a frame whose
// last stage never answered simply stopped existing, and the operator saw it as a camera that
// "intermittently misses".
//
// The timeout is swept on a timer rather than armed per frame: one wake-up every
// `sweep_interval_ms` for the whole fleet against a thousand timers a second. The cost is that
// the timeout is accurate to within one interval, which against a 1500 ms budget is noise.
//
// CAPTURE UNDER THE LOCK, BUILD OUTSIDE IT
// ----------------------------------------
// The emitter must not read a `FrameState` whose worker is still running (ADR-002), so the
// capture happens while the lock is held. It must also not do the *expensive* half there: the
// Python version built every record inside this mutex and that was 770 us per frame on the one
// lock every worker takes on every stage. So `sweep` captures, releases, and only then emits.
#pragma once

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "shipinfer/pipeline/state.hpp"

namespace shipinfer {

enum class FinishReason { Complete, Incomplete, Timeout, Shutdown, Evicted };

const char* to_string(FinishReason reason);

struct FrameResult {
    EmissionInputs inputs;
    std::vector<std::string> delivered;
    std::vector<std::string> missing;
    FinishReason reason = FinishReason::Complete;
    int64_t waited_us = 0;
};

class FrameCollector {
  public:
    using Emit = std::function<void(FrameResult&&)>;

    FrameCollector(Emit emit, size_t capacity, int timeout_ms);

    // False when the buffer is full and nothing could be evicted, or when a frame with this
    // tag is already in flight. The caller must fail its own frame in that case — returning
    // false and letting it vanish is the bug this whole class exists to prevent.
    bool open(const std::shared_ptr<FrameState>& state, const std::vector<std::string>& expected);
    void expect(const FrameTag& tag, const std::vector<std::string>& stages);
    void deliver(const FrameTag& tag, const std::string& stage);
    void seal(const FrameTag& tag);
    int sweep();
    int drain();

    size_t pending() const;
    std::map<std::string, size_t> pending_by_camera() const;
    uint64_t reported() const;
    uint64_t evicted() const;
    uint64_t timed_out() const;

  private:
    struct Pending {
        std::shared_ptr<FrameState> state;
        std::set<std::string> expected;
        std::set<std::string> delivered;
        int64_t opened_ns = 0;

        bool complete() const { return delivered.size() >= expected.size(); }
    };

    FrameResult finish_locked(Pending& frame, FinishReason reason, int64_t now_ns) const;
    bool evict_locked();

    Emit emit_;
    size_t capacity_;
    int64_t timeout_ns_;
    mutable std::mutex mutex_;
    std::map<std::string, Pending> pending_;
    std::map<std::string, size_t> per_camera_;
    uint64_t reported_ = 0;
    uint64_t evicted_ = 0;
    uint64_t timed_out_ = 0;
};

}  // namespace shipinfer
