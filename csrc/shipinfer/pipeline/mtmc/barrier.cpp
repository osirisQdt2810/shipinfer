#include "shipinfer/pipeline/mtmc/barrier.h"

#include <algorithm>
#include <chrono>

namespace shipinfer::mtmc {

    namespace {

        double steady_seconds() {
            using namespace std::chrono;
            return duration_cast<duration<double>>(steady_clock::now().time_since_epoch())
                .count();
        }

        // Whether `live` is a subset of the cameras that reported into `bucket`.
        bool every_live_reported(const std::set<std::string>& live,
                                 const std::map<std::string, double>& reported) {
            for (const std::string& camera : live) {
                if (reported.count(camera) == 0) return false;
            }
            return true;
        }

    }  // namespace

    WaiterBudget::WaiterBudget(int permits) : permits_(permits) {
        if (permits < 0) {
            throw ConfigError("a waiter budget cannot have " + std::to_string(permits) +
                              " permits; 0 means 'never wait', which is what a single-worker "
                              "runner gets");
        }
    }

    int WaiterBudget::held() const {
        std::lock_guard<std::mutex> guard(lock_);
        return held_;
    }

    uint64_t WaiterBudget::over_released() const {
        std::lock_guard<std::mutex> guard(lock_);
        return over_released_;
    }

    bool WaiterBudget::acquire() {
        std::lock_guard<std::mutex> guard(lock_);
        if (held_ >= permits_) return false;
        ++held_;
        return true;
    }

    void WaiterBudget::release() {
        if (!release_held()) {
            throw ServerStateError(
                "a waiter budget was released more times than it was acquired; the permit "
                "count is now meaningless and the never-starve guard with it");
        }
    }

    bool WaiterBudget::release_held() noexcept {
        std::lock_guard<std::mutex> guard(lock_);
        if (held_ <= 0) {
            // SATURATED, and counted, because the ONE caller on this path is a destructor.
            // A `noexcept` destructor that threw would `std::terminate` the shard with no
            // `ServerStateError` anywhere in the logs to say why -- and this budget is
            // process-wide by design, so one stray `release()` from anywhere else is enough
            // to drive `held_` to zero while a worker is parked. The public `release()` keeps
            // the typed refusal for callers that can receive it.
            ++over_released_;
            return false;
        }
        --held_;
        return true;
    }

    InstantBarrier::InstantBarrier(Options options, std::shared_ptr<WaiterBudget> budget,
                                   Clock clock, OnEvent on_event)
        : window_s_(options.sync_window_s),
          workers_(options.workers),
          configured_max_instants_(options.max_instants),
          max_instants_(options.max_instants.value_or(kDefaultMaxInstants)),
          budget_(budget ? std::move(budget)
                         : std::make_shared<WaiterBudget>(std::max(0, options.workers - 1))),
          clock_(clock ? std::move(clock) : Clock(steady_seconds)),
          on_event_(std::move(on_event)),
          // Four windows of history, floored at eight: long enough that a frame arriving a
          // few instants late is called LATE rather than opening a new instant, short enough
          // that the map is not a leak. It follows the bound, so it grows with the fleet too
          // -- `refresh_live` recomputes both together.
          recent_limit_(static_cast<size_t>(std::max(8, max_instants_ * 4))) {
        if (!(window_s_ > 0.0)) {
            throw ConfigError("sync_window_s must be positive, got " +
                              std::to_string(window_s_) +
                              "; an instant with no width admits one camera and refuses the "
                              "rest of its group as late");
        }
        if (max_instants_ < 1) {
            // `max_instants_`, not the optional: the two are the same number whenever this
            // branch is reachable, and a future floor of 0 would make the dereference a crash
            // inside an error path.
            throw ConfigError("max_instants must be at least 1, got " +
                              std::to_string(max_instants_) +
                              "; zero open instants means every frame evicts itself");
        }
    }

    int InstantBarrier::waiters() const {
        std::lock_guard<std::mutex> guard(lock_);
        return waiters_;
    }

    std::set<std::string> InstantBarrier::live() const {
        std::lock_guard<std::mutex> guard(lock_);
        return live_;
    }

    size_t InstantBarrier::open_instants() const {
        std::lock_guard<std::mutex> guard(lock_);
        return buckets_.size();
    }

    std::map<std::string, uint64_t> InstantBarrier::instant_stats() const {
        std::lock_guard<std::mutex> guard(lock_);
        return instant_counts_;
    }

    void InstantBarrier::note_arrival_lag_us(uint32_t lag_us, bool negative) {
        std::lock_guard<std::mutex> held(lock_);
        if (negative) ++lag_negative_;
        if (arrival_lag_us_.size() < kMaxLagSamples) {
            arrival_lag_us_.push_back(lag_us);
            return;
        }
        // WRAPS rather than stops. Keeping the first N froze the distribution on the warm-up
        // of any run longer than the ring, while the frame percentiles printed beside it
        // covered the whole run -- two numbers over different windows, with nothing saying so.
        arrival_lag_us_[lag_next_] = lag_us;
        lag_next_ = (lag_next_ + 1) % kMaxLagSamples;
        ++lag_overwritten_;
    }

    std::vector<uint32_t> InstantBarrier::arrival_lag_us() const {
        std::lock_guard<std::mutex> held(lock_);
        return arrival_lag_us_;  // copied: `percentile` reorders what it is given
    }

    uint64_t InstantBarrier::lag_samples_overwritten() const {
        std::lock_guard<std::mutex> held(lock_);
        return lag_overwritten_;
    }

    uint64_t InstantBarrier::lag_samples_negative() const {
        std::lock_guard<std::mutex> held(lock_);
        return lag_negative_;
    }

    InstantBarrier::InstantSizes InstantBarrier::instant_sizes() const {
        std::lock_guard<std::mutex> guard(lock_);
        return {cameras_held_, instants_ended_, cameras_held_max_};
    }

    std::map<std::string, uint64_t> InstantBarrier::frame_stats() const {
        std::lock_guard<std::mutex> guard(lock_);
        return frame_counts_;
    }

    std::set<std::string> InstantBarrier::silent_cameras() const {
        std::lock_guard<std::mutex> guard(lock_);
        // ANNOUNCED MINUS SEEN. No `hooked_` check: with no lifecycle wiring `announced_` is
        // empty, so the difference is empty and nothing can be silent by construction -- a
        // branch for that would state the same fact twice.
        std::set<std::string> silent;
        for (const std::string& camera : announced_) {
            if (seen_.count(camera) == 0) silent.insert(camera);
        }
        return silent;
    }

    void InstantBarrier::refresh_live() {
        live_ = hooked_ ? announced_ : seen_;
        // THE BOUND FOLLOWS THE FLEET unless the chain named one: every camera keeps one
        // instant open and seals more as it advances, so a bound below the fleet's size
        // evicts buckets the group is still filling rather than a clock that is wrong.
        //
        // SEEN *UNION* ANNOUNCED, and NOT `live_`: the live set answers who must report for an
        // instant to be complete, which is a roster decision; the bound answers how many
        // instants can legitimately be open, which is a fact about traffic. A shard whose
        // roster names four cameras and whose fleet sends fifty has fifty cameras opening
        // instants, and a bound of eight would evict on every one of them.
        if (configured_max_instants_) return;
        std::set<std::string> fleet = announced_;
        fleet.insert(seen_.begin(), seen_.end());
        max_instants_ = std::max<int>(kDefaultMaxInstants, static_cast<int>(fleet.size()));
        // AND NEVER UNDER WHAT IS OPEN: a drain recomputes downward, so the next `open()`
        // would evict buckets the SURVIVING cameras are still filling. `+ 1` because `evict`
        // is `>=`. A snapshot taken when the fleet changes, not a ratchet: it falls with the
        // map as those buckets retire on their own deadlines.
        max_instants_ = std::max<int>(max_instants_, static_cast<int>(buckets_.size()) + 1);
        recent_limit_ = static_cast<size_t>(std::max(8, max_instants_ * 4));
    }

    int InstantBarrier::max_instants() const {
        std::lock_guard<std::mutex> guard(lock_);
        return max_instants_;
    }

    void InstantBarrier::camera_added(const std::string& camera_id) {
        std::lock_guard<std::mutex> guard(lock_);
        // ANNOUNCED WINS from the first announcement onward. Before any hook fires the live
        // set is what has been seen, so a chain with no lifecycle wiring still closes on
        // evidence; after one, a camera that has never sent a frame is still waited for,
        // which is the whole point of the hook.
        hooked_ = true;
        announced_.insert(camera_id);
        refresh_live();
    }

    void InstantBarrier::drop_camera(const std::string& camera_id) {
        std::lock_guard<std::mutex> guard(lock_);
        announced_.erase(camera_id);
        seen_.erase(camera_id);
        newest_capture_.erase(camera_id);
        backward_run_.erase(camera_id);
        refresh_live();
        // doc: long why a bucket with NO waiters is left alone here
        // SEALED, NOT CLOSED, and by the lifecycle thread: dropping the last missing camera
        // completes every open instant, and this thread has no association function -- so it
        // marks them and a waiter does the work. `seal` explains why that split matters.
        //
        // AND ONLY A BUCKET SOMEBODY IS WAITING ON, which `topology/barrier.py` guards the
        // same way ("A bucket with no waiters is left alone: every frame in it was already
        // emitted by the never-starve guard, so nobody is owed an answer"). Two things go
        // wrong without it, and the first is not a metric: a sealed bucket is skipped by
        // `match`, so it stops accepting entries -- and on a shard where the never-starve
        // guard is active, the frames that would have COMPLETED it open a fresh instant and
        // come back unanswered instead. A camera outage would then cost every open instant
        // its association. The second is that such a bucket is never associated, so `retire`
        // counts it under `complete` -- a health report claiming a good instant for work that
        // did not happen.
        for (auto& [instant, bucket] : buckets_) {
            if (bucket->waiters == 0 || bucket->ready || bucket->done) continue;
            if (every_live_reported(live_, bucket->reported)) {
                seal(*bucket, kClosedComplete);
            }
        }
    }

    std::shared_ptr<InstantBarrier::Bucket> InstantBarrier::match(double capture_s) {
        // NEWEST FIRST, because when two open buckets could both take a capture the later one
        // is the instant the group is currently filling; joining the older would put this
        // frame in an instant its own camera may already have contributed to.
        for (auto it = buckets_.rbegin(); it != buckets_.rend(); ++it) {
            Bucket& bucket = *it->second;
            if (bucket.ready || bucket.done) continue;
            const double spread =
                std::max(bucket.last, capture_s) - std::min(bucket.first, capture_s);
            if (spread < window_s_) return it->second;
        }
        return nullptr;
    }

    int64_t InstantBarrier::late_instant(double capture_s) const {
        for (auto it = recent_.rbegin(); it != recent_.rend(); ++it) {
            const auto& [first, last] = it->second;
            if (first <= capture_s && capture_s <= last) return it->first;
        }
        return 0;
    }

    std::shared_ptr<InstantBarrier::Bucket> InstantBarrier::open(double capture_s, double now) {
        evict();
        ++next_instant_;
        auto bucket = std::make_shared<Bucket>();
        bucket->instant = next_instant_;
        bucket->deadline = now + window_s_;
        bucket->first = capture_s;
        bucket->last = capture_s;
        buckets_[bucket->instant] = bucket;
        return bucket;
    }

    void InstantBarrier::seal(Bucket& bucket, const std::string& reason) {
        // doc: long why sealing is not closing, and who is allowed to run the association
        // SEALED rather than closed on the spot because the thread that seals is not a member
        // of this instant: if it ran the association and the callback threw, the exception
        // would fail ITS frame -- a frame from a different instant -- while the instant that
        // actually failed was somebody else's. A waiter is a worker holding the callback and
        // owns the answer it is waiting for, so it is the right thread to run it. A sealed
        // bucket with no waiters simply expires: every frame in it was emitted by the
        // never-starve guard and nobody is owed an answer.
        bucket.ready = true;
        bucket.ready_reason = reason;
        if (bucket.waiters > 0) cond_.notify_all();
    }

    InstantOutcome InstantBarrier::missed(const std::string& reason, int64_t instant) {
        ++frame_counts_[reason];
        InstantOutcome outcome;
        outcome.reason = reason;
        outcome.instant = instant;
        return outcome;
    }

    void InstantBarrier::event(const std::string& reason) {
        ++instant_counts_[reason];
        if (on_event_) on_event_(reason);
    }

    void InstantBarrier::remember(const Bucket& bucket) {
        // EVERY ended bucket passes here -- closed, evicted, expired or shut down -- which is
        // why the tally lives in this function and not in `close`.
        const uint64_t held = static_cast<uint64_t>(bucket.reported.size());
        cameras_held_ += held;
        ++instants_ended_;
        cameras_held_max_ = std::max(cameras_held_max_, held);
        recent_[bucket.instant] = {bucket.first, bucket.last};
        while (recent_.size() > recent_limit_) recent_.erase(recent_.begin());
    }

    InstantOutcome InstantBarrier::close(const std::shared_ptr<Bucket>& bucket,
                                         const std::string& reason,
                                         const Association& associate) {
        // OUT OF THE MAP BEFORE THE CALLBACK RUNS, so a frame arriving for this instant while
        // the association is in flight is late rather than a member of a bucket that is
        // already being consumed. The `shared_ptr` is what makes that safe with waiters still
        // asleep on it.
        buckets_.erase(bucket->instant);
        remember(*bucket);
        Results results;
        try {
            results = associate(bucket->entries);
        } catch (...) {
            bucket->results = nullptr;
            bucket->associated = false;
            bucket->reason = kDroppedFailed;
            bucket->done = true;
            event(kDroppedFailed);
            cond_.notify_all();
            throw;
        }
        bucket->results = std::move(results);
        bucket->associated = true;
        bucket->reason = reason;
        bucket->done = true;
        event(reason);
        cond_.notify_all();
        InstantOutcome outcome;
        outcome.reason = reason;
        outcome.results = bucket->results;
        outcome.associated = true;
        outcome.instant = bucket->instant;
        return outcome;
    }

    void InstantBarrier::retire(double now) {
        // doc: long why an expired bucket with no waiters is discarded rather than associated
        // Such a bucket holds only frames the never-starve guard already emitted, so no
        // association would have a reader -- running one would cost a tracker call and a
        // global-id assignment for nothing, and NOT running one would leave the bucket to be
        // evicted later and read as clock skew. It is discarded and counted under the reason
        // it was SEALED with, because "a camera moved on" is what happened to that instant and
        // `kClosedAdvanced` is the share an operator reads before touching `sync_window_ms`.
        if (buckets_.empty()) return;
        // The front bucket has the earliest deadline of all -- every deadline is one window
        // after its bucket opened and the map is ordered by instant id -- so if it has not
        // expired, none has. This runs on every submit and almost always stops here.
        if (buckets_.begin()->second->deadline > now) return;
        std::vector<int64_t> stale;
        for (const auto& [instant, bucket] : buckets_) {
            if (bucket->waiters == 0 && bucket->deadline <= now) stale.push_back(instant);
        }
        for (const int64_t instant : stale) {
            const auto found = buckets_.find(instant);
            if (found == buckets_.end()) continue;
            const std::shared_ptr<Bucket> bucket = found->second;
            buckets_.erase(found);
            remember(*bucket);
            bucket->results = nullptr;
            bucket->associated = false;
            bucket->reason =
                bucket->ready ? bucket->ready_reason : std::string(kDroppedExpired);
            bucket->done = true;
            event(bucket->reason);
        }
        if (!stale.empty()) cond_.notify_all();
    }

    void InstantBarrier::evict() {
        // OLDEST BY INSTANT ID, which is the order buckets were opened and therefore also the
        // order their deadlines expire. Evicting by capture time instead would let a single
        // camera with a stale clock push out the instant the rest of the group is actively
        // filling, which is the opposite of what eviction is for.
        while (buckets_.size() >= static_cast<size_t>(max_instants_)) {
            const std::shared_ptr<Bucket> bucket = buckets_.begin()->second;
            buckets_.erase(buckets_.begin());
            remember(*bucket);
            bucket->results = nullptr;
            bucket->associated = false;
            bucket->reason = kDroppedEvicted;
            bucket->done = true;
            event(kDroppedEvicted);
            cond_.notify_all();
        }
    }

    size_t InstantBarrier::close_all(const std::string& reason) {
        std::lock_guard<std::mutex> guard(lock_);
        closed_ = true;
        const size_t resolved = buckets_.size();
        while (!buckets_.empty()) {
            const std::shared_ptr<Bucket> bucket = buckets_.begin()->second;
            buckets_.erase(buckets_.begin());
            remember(*bucket);
            bucket->results = nullptr;
            bucket->associated = false;
            bucket->reason = reason;
            bucket->done = true;
            event(reason);
        }
        cond_.notify_all();
        return resolved;
    }

    InstantOutcome InstantBarrier::submit(const std::string& camera_id, double capture_s,
                                          std::shared_ptr<void> payload,
                                          const Association& associate) {
        std::unique_lock<std::mutex> guard(lock_);
        if (closed_) return missed(kDroppedShutdown, 0);
        const double now = clock_();
        retire(now);
        if (seen_.insert(camera_id).second) refresh_live();

        // BEFORE ANY BUCKET IS TOUCHED, and against this camera's OWN newest stamp: a step
        // backwards past the window matches no open bucket and no resolved span, so it would
        // open an instant in the past and wait out a whole window for cameras whose clocks did
        // not step. Refused and counted instead.
        bool adopted = false;
        const auto newest = newest_capture_.find(camera_id);
        if (newest != newest_capture_.end() && capture_s < newest->second - window_s_) {
            int& run = backward_run_[camera_id];
            if (run < kBackwardRefusalsBeforeAdopting) {
                ++run;
                return missed(kMissedBackward, 0);
            }
            // ADOPTED HERE, where the decision is made, and not below where the bucket is
            // resolved: `late` and `duplicate` return above that, so a frame that was decided
            // to be an adoption would otherwise be measured against the reference it replaced
            // on the very next frame.
            newest->second = capture_s;
            backward_run_.erase(camera_id);
            adopted = true;
        }

        std::shared_ptr<Bucket> bucket = match(capture_s);
        if (!bucket) {
            const int64_t late = late_instant(capture_s);
            if (late != 0) return missed(kMissedLate, late);
            bucket = open(capture_s, now);
        } else {
            const auto reported = bucket->reported.find(camera_id);
            if (reported != bucket->reported.end()) {
                if (capture_s <= reported->second) {
                    return missed(kMissedDuplicate, bucket->instant);
                }
                // This camera has moved on, so that instant has every frame it will ever get
                // from it. Seal it -- a waiter closes it -- and put this frame in the instant
                // that starts here.
                seal(*bucket, kClosedAdvanced);
                bucket = open(capture_s, now);
            }
        }

        backward_run_.erase(camera_id);
        // FIND, not `operator[]`: the subscript default-constructs to 0.0 and `std::max` then
        // keeps it, so a camera's FIRST frame at a negative stamp recorded 0.0 here and the
        // real value on the Python plane -- the mirror image of the falsy-zero that plane's
        // comment warns about, and `test_barrier.py` already submits at -0.4.
        const auto recorded = newest_capture_.find(camera_id);
        if (recorded == newest_capture_.end()) {
            newest_capture_.emplace(camera_id, capture_s);
        } else if (adopted) {
            recorded->second = capture_s;
        } else {
            recorded->second = std::max(recorded->second, capture_s);
        }
        bucket->reported[camera_id] = capture_s;
        bucket->entries.push_back(InstantEntry{camera_id, std::move(payload)});
        bucket->first = std::min(bucket->first, capture_s);
        bucket->last = std::max(bucket->last, capture_s);

        if (every_live_reported(live_, bucket->reported)) {
            return close(bucket, kClosedComplete, associate);
        }
        // The never-starve guard, counted process-wide: two barriers each admitting
        // `workers - 1` waiters would park every worker between them.
        if (!budget_->acquire()) return missed(kMissedWouldStarve, bucket->instant);

        // PAIRED STRUCTURALLY, the way Python's `try`/`finally` is: the permit and both
        // waiter counts come back on every exit from the wait, including a throw.
        struct Waiting {
            InstantBarrier& barrier;
            Bucket& bucket;
            ~Waiting() {
                --barrier.waiters_;
                --bucket.waiters;
                // `release_held`, not `release`: this destructor is noexcept, and the typed
                // refusal would be a `std::terminate` rather than an error.
                barrier.budget_->release_held();
            }
        };
        {
            ++waiters_;
            ++bucket->waiters;
            const Waiting held{*this, *bucket};
            // Re-derived from `clock_` on every wake rather than computed once: a notify_all
            // for a DIFFERENT bucket wakes every waiter here anyway, and re-deriving is what
            // puts expiry on the injected clock. One `wait_for` with the whole window would
            // hand the decision back to real time whatever `clock_` says.
            while (!(bucket->done || bucket->ready)) {
                const double remaining = bucket->deadline - clock_();
                if (remaining <= 0.0) break;
                cond_.wait_for(guard, std::chrono::duration<double>(remaining));
            }
        }
        if (bucket->done) {
            InstantOutcome outcome;
            outcome.reason = bucket->reason;
            outcome.results = bucket->results;
            outcome.associated = bucket->associated;
            outcome.instant = bucket->instant;
            return outcome;
        }
        const auto still = buckets_.find(bucket->instant);
        if (still != buckets_.end() && still->second == bucket) {
            // Either somebody sealed it while we slept or the window ran out. Both mean: this
            // thread is the closer.
            return close(bucket,
                         bucket->ready ? bucket->ready_reason : std::string(kClosedWindow),
                         associate);
        }
        // Removed from the map without being marked done. No path does that today; a refusal
        // is still better than a hang if one is ever added.
        return missed(kDroppedExpired, bucket->instant);
    }

}  // namespace shipinfer::mtmc
