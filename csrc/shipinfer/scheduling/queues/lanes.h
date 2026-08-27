// The per-key FIFO lane used by the fair queue — `scheduling/queues/lanes.py`, line for line.
//
// One priority level: per-key FIFOs plus a round-robin cursor over active keys. Both `push` and
// `pop` are O(1). The obvious alternative — scan every camera on each tick and take one — is
// O(cameras) per request, which at 50 cameras and 15 000 requests/s is exactly the kind of
// quiet waste this project exists to remove. (The first C++ queue did the scan; this is the
// port.)
#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace shipinfer {

    template <typename T>
    class Lane {
      public:
        void push(T item) {
            const std::string& key = item.camera();
            auto it = by_key_.find(key);
            if (it == by_key_.end()) {
                it = by_key_.emplace(key, std::deque<T>{}).first;
                order_.push_back(key);
                seen_.push_back(key);
            }
            it->second.push_back(std::move(item));
            ++size_;
        }

        // The item `peek()` showed. The key goes to the back of the rotation if it still has
        // work: that is the round-robin.
        T pop() {
            const std::string key = std::move(order_.front());
            order_.pop_front();
            auto it = by_key_.find(key);
            T item = std::move(it->second.front());
            it->second.pop_front();
            if (!it->second.empty()) {
                order_.push_back(key);
            } else {
                by_key_.erase(it);
                forget(key);
            }
            --size_;
            return item;
        }

        // The item `pop()` would return, without removing it. Needed because a batch is
        // bounded in *rows* and an item carries however many rows its request does, so the
        // drain has to see an item's size before committing to it — popping first and pushing
        // back would send it to the back of its own key's FIFO and reorder a camera's frames.
        const T* peek() const {
            if (order_.empty()) return nullptr;
            return &by_key_.find(order_.front())->second.front();
        }

        // Drop the **oldest** request of whichever key is hogging this lane. Deliberately not
        // "the globally oldest": the request that has waited longest is usually the victim of
        // a flood, not its cause; penalising the loudest camera is what keeps a quiet camera's
        // frames alive. Ties go to the key that was first *seen* — Python scans `by_key` in
        // insertion order (`max()` over a dict), which `pop()` never reorders. The rotation
        // deque does get reordered by every `pop()`, so scanning it gave a different victim
        // after a single serve (review of #19, round 3); `seen_` is the insertion order kept
        // for exactly this scan, so the two planes evict the same item on the same trace.
        std::optional<T> evict_from_longest() {
            if (by_key_.empty()) return std::nullopt;
            std::string worst;
            size_t deepest = 0;
            for (const std::string& key : seen_) {
                const size_t depth = by_key_.find(key)->second.size();
                if (depth > deepest) {
                    deepest = depth;
                    worst = key;
                }
            }
            auto it = by_key_.find(worst);
            T item = std::move(it->second.front());
            it->second.pop_front();
            --size_;
            if (it->second.empty()) {
                by_key_.erase(it);
                for (auto o = order_.begin(); o != order_.end(); ++o) {
                    if (*o == worst) {
                        order_.erase(o);
                        break;
                    }
                }
                forget(worst);
            }
            return item;
        }

        // How many requests each key is holding in this lane, accumulated into a caller's
        // map so one walk covers all four priority lanes (`scheduling/queues/lanes.py`
        // returns a fresh dict; the queue sums them the same way). O(keys), and only ever
        // called from `stats()` — never per frame. A running per-key counter maintained by
        // `push`/`pop` would put bookkeeping on the hot path to serve a snapshot an operator
        // reads every few seconds.
        void add_depths(std::map<std::string, uint64_t>& out) const {
            for (const auto& entry : by_key_) {
                out[entry.first] += static_cast<uint64_t>(entry.second.size());
            }
        }

        std::vector<T> drain() {
            std::vector<T> items;
            items.reserve(size_);
            // In rotation order, so a drained lane reads the way it would have been served.
            for (const std::string& key : order_) {
                auto& bucket = by_key_.find(key)->second;
                for (T& item : bucket) items.push_back(std::move(item));
            }
            by_key_.clear();
            order_.clear();
            seen_.clear();
            size_ = 0;
            return items;
        }

        size_t size() const { return size_; }
        bool empty() const { return size_ == 0; }

      private:
        std::unordered_map<std::string, std::deque<T>> by_key_;
        std::deque<std::string> order_;
        // Keys in first-seen order, the order Python's dict scan uses; a key leaves when its
        // bucket empties and re-enters at the back when it pushes again, as a dict key would.
        std::vector<std::string> seen_;

        void forget(const std::string& key) {
            for (auto it = seen_.begin(); it != seen_.end(); ++it) {
                if (*it == key) {
                    seen_.erase(it);
                    return;
                }
            }
        }
        size_t size_ = 0;
    };

}  // namespace shipinfer
