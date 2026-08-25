#include "shipinfer/pipeline/reassembly/collector.h"

#include <algorithm>

namespace shipinfer {
    namespace {

        int64_t now_ns() {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        }

    }  // namespace

    const char* to_string(FinishReason reason) {
        switch (reason) {
            case FinishReason::Complete:
                return "complete";
            case FinishReason::Incomplete:
                return "incomplete";
            case FinishReason::Timeout:
                return "timeout";
            case FinishReason::Shutdown:
                return "shutdown";
            case FinishReason::Evicted:
                return "evicted";
        }
        return "unknown";
    }

    FrameCollector::FrameCollector(Emit emit, size_t capacity, int timeout_ms)
        : emit_(std::move(emit)),
          capacity_(capacity),
          timeout_ns_(static_cast<int64_t>(timeout_ms) * 1000000) {}

    bool FrameCollector::open(const std::shared_ptr<FrameState>& state,
                              const std::vector<std::string>& expected) {
        const std::string key = state->tag().key();
        std::unique_lock<std::mutex> lock(mutex_);
        if (pending_.count(key) != 0) {
            // A duplicate tag. Refusing the newcomer rather than clobbering the entry is what
            // keeps the first frame's caller from waiting on something nobody will resolve.
            return false;
        }
        std::optional<FrameResult> evicted;
        if (pending_.size() >= capacity_) {
            evicted = evict_locked(now_ns());
            if (!evicted.has_value()) return false;
        }

        Pending frame;
        frame.state = state;
        frame.expected.insert(expected.begin(), expected.end());
        frame.opened_ns = now_ns();
        pending_.emplace(key, std::move(frame));
        ++per_camera_[state->tag().camera_id];
        lock.unlock();

        // Emitted, and outside the lock. The first version destroyed the frame here with no
        // event at all and still incremented `reported_`, so `collector_reported` over-counted
        // by the eviction count and an operator had no per-camera number to point at — which
        // is the entire diagnosis ADR-005 exists to make possible.
        if (evicted.has_value()) emit_(std::move(*evicted));
        return true;
    }

    void FrameCollector::expect(const FrameTag& tag, const std::vector<std::string>& stages) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pending_.find(tag.key());
        if (it == pending_.end()) return;
        it->second.expected.insert(stages.begin(), stages.end());
    }

    void FrameCollector::deliver(const FrameTag& tag, const std::string& stage) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pending_.find(tag.key());
        if (it == pending_.end()) return;  // already finished; a late answer is not an error
        it->second.delivered.insert(stage);
    }

    void FrameCollector::seal(const FrameTag& tag) {
        FrameResult result;
        bool ready = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = pending_.find(tag.key());
            if (it == pending_.end()) return;
            const bool complete = it->second.complete();
            result = finish_locked(it->second,
                                   complete ? FinishReason::Complete : FinishReason::Incomplete,
                                   now_ns());
            auto camera = it->second.state->tag().camera_id;
            pending_.erase(it);
            if (--per_camera_[camera] == 0) per_camera_.erase(camera);
            ++reported_;
            ready = true;
        }
        // Outside the lock. See the header: this is where the per-object work happens and it
        // must not be serialised on the mutex every worker takes on every stage.
        if (ready) emit_(std::move(result));
    }

    int FrameCollector::sweep() {
        std::vector<FrameResult> ready;
        const int64_t at = now_ns();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto it = pending_.begin(); it != pending_.end();) {
                if (at - it->second.opened_ns < timeout_ns_) {
                    ++it;
                    continue;
                }
                ready.push_back(finish_locked(it->second, FinishReason::Timeout, at));
                const auto camera = it->second.state->tag().camera_id;
                it = pending_.erase(it);
                if (--per_camera_[camera] == 0) per_camera_.erase(camera);
                ++reported_;
                ++timed_out_;
            }
        }
        for (auto& result : ready) emit_(std::move(result));
        return static_cast<int>(ready.size());
    }

    int FrameCollector::drain() {
        std::vector<FrameResult> ready;
        const int64_t at = now_ns();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& [key, frame] : pending_) {
                ready.push_back(finish_locked(frame, FinishReason::Shutdown, at));
                ++reported_;
            }
            pending_.clear();
            per_camera_.clear();
        }
        // A shutdown that silently discards four hundred half-finished frames is not an orderly
        // shutdown.
        for (auto& result : ready) emit_(std::move(result));
        return static_cast<int>(ready.size());
    }

    FrameResult FrameCollector::finish_locked(Pending& frame, FinishReason reason,
                                              int64_t at) const {
        FrameResult result;
        result.reason = reason;
        result.waited_us = (at - frame.opened_ns) / 1000;
        result.inputs = frame.state->capture();
        result.delivered.assign(frame.delivered.begin(), frame.delivered.end());
        for (const auto& stage : frame.expected) {
            if (frame.delivered.count(stage) == 0) result.missing.push_back(stage);
        }
        return result;
    }

    std::optional<FrameResult> FrameCollector::evict_locked(int64_t at) {
        // ADR-005: the camera holding the most frames loses one, never whoever is oldest.
        // That inversion is the bug the previous generation had — a crowded camera filled the
        // buffer and pushed out a quiet camera's work.
        if (per_camera_.empty()) return std::nullopt;
        const auto greediest =
            std::max_element(per_camera_.begin(), per_camera_.end(),
                             [](const auto& a, const auto& b) { return a.second < b.second; });

        for (auto it = pending_.begin(); it != pending_.end(); ++it) {
            if (it->second.state->tag().camera_id != greediest->first) continue;
            FrameResult result = finish_locked(it->second, FinishReason::Evicted, at);
            const auto camera = greediest->first;
            pending_.erase(it);
            if (--per_camera_[camera] == 0) per_camera_.erase(camera);
            ++evicted_;
            ++evicted_by_camera_[camera];
            ++reported_;
            return result;
        }
        return std::nullopt;
    }

    std::map<std::string, uint64_t> FrameCollector::evicted_by_camera() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return evicted_by_camera_;
    }

    size_t FrameCollector::pending() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_.size();
    }

    std::map<std::string, size_t> FrameCollector::pending_by_camera() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return per_camera_;
    }

    uint64_t FrameCollector::reported() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return reported_;
    }

    uint64_t FrameCollector::evicted() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return evicted_;
    }

    uint64_t FrameCollector::timed_out() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return timed_out_;
    }

}  // namespace shipinfer
