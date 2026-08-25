// The scheduling seam, tested the way the Python plane tests it.
//
// Every check here has a namesake in `tests/scheduling/` — `test_queue_fairness.py`,
// `test_batch_window.py`, `test_batch_rows.py` — and asserts the same thing, so a divergence
// between the planes shows up as one green file and one red one before the parity harness
// (ledger P6) ever runs a trace. No device, no GPU: the queue is pure logic on both sides.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/scheduling/queues/base.h"
#include "shipinfer/scheduling/queues/fair.h"
#include "shipinfer/scheduling/queues/fifo.h"
#include "shipinfer/scheduling/queues/lanes.h"

namespace {

    using namespace shipinfer;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        }
    }

    // The item under test: what `WorkItem` shows the queue, and nothing else.
    struct Item {
        std::string cam;
        int id = 0;
        size_t rows_ = 1;
        int prio = Priority::Normal;
        int64_t deadline_ns = 0;

        std::string camera() const { return cam; }
        size_t rows() const { return rows_; }
        int priority() const { return prio; }
        bool expired(int64_t now_ns) const { return deadline_ns != 0 && now_ns > deadline_ns; }
    };

    using Clock = std::chrono::steady_clock;
    int64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count();
    }
    double ms_since(Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    std::vector<std::string> cameras_of(const std::vector<Item>& batch) {
        std::vector<std::string> out;
        for (const Item& item : batch) out.push_back(item.cam);
        return out;
    }

    // -- TestFairDrain ---------------------------------------------------------------------

    void test_round_robin_across_cameras() {
        FairPriorityQueue<Item> queue("q", 64);
        for (int i = 0; i < 3; ++i) queue.put(Item{"busy", i});
        queue.put(Item{"quiet", 100});
        const auto batch = queue.get_batch(BatchWindow(4));
        check(cameras_of(batch) == std::vector<std::string>{"busy", "quiet", "busy", "busy"},
              "a quiet camera's frame is second, not fourth");
    }

    void test_fifo_is_not_fair() {
        FifoQueue<Item> queue("q", 64);
        for (int i = 0; i < 3; ++i) queue.put(Item{"busy", i});
        queue.put(Item{"quiet", 100});
        const auto batch = queue.get_batch(BatchWindow(4));
        check(cameras_of(batch) == std::vector<std::string>{"busy", "busy", "busy", "quiet"},
              "FIFO serves the flood first — the control the fair queue is compared against");
    }

    void test_priority_beats_fairness() {
        FairPriorityQueue<Item> queue("q", 64);
        queue.put(Item{"a", 1, 1, Priority::Normal});
        queue.put(Item{"b", 2, 1, Priority::Normal});
        queue.put(Item{"c", 3, 1, Priority::TrackingCritical});
        const auto batch = queue.get_batch(BatchWindow(2));
        check(batch.size() == 2 && batch[0].cam == "c",
              "a TRACKING_CRITICAL request leads the batch whatever camera it is from");
        check(batch[1].cam == "a", "then the normal lane, in rotation order");
    }

    // -- TestOverflowPolicies --------------------------------------------------------------

    void test_reject_when_full_reports_depth() {
        FairPriorityQueue<Item> queue("q", 2, Overflow::Reject);
        check(queue.put(Item{"a", 1}) == PutStatus::Accepted, "first accepted");
        check(queue.put(Item{"a", 2}) == PutStatus::Accepted, "second accepted");
        check(queue.put(Item{"b", 3}) == PutStatus::Rejected, "the third is refused, loudly");
        const QueueStats stats = queue.stats();
        check(stats.depth == 2 && stats.capacity == 2, "depth and capacity are what the refusal saw");
        check(stats.rejected == 1 && stats.rejected_by_camera.at("b") == 1,
              "the refusal is charged to the camera that was refused");
    }

    void test_drop_oldest_evicts_the_greediest_camera() {
        std::vector<std::pair<int, DropReason>> dropped;
        FairPriorityQueue<Item> queue("q", 4, Overflow::DropOldest, 50, true,
                                      [&](Item&& item, DropReason why) {
                                          dropped.emplace_back(item.id, why);
                                      });
        queue.put(Item{"greedy", 1});
        queue.put(Item{"greedy", 2});
        queue.put(Item{"greedy", 3});
        queue.put(Item{"quiet", 10});
        check(queue.put(Item{"another", 20}) == PutStatus::Accepted, "the newcomer is admitted");
        check(dropped.size() == 1 && dropped[0].second == DropReason::Evicted, "exactly one eviction");
        // The **oldest** of the greediest camera — never the quiet camera's frame, and not the
        // greedy camera's newest (which is what the first C++ queue dropped).
        check(dropped[0].first == 1, "the greedy camera's oldest frame is the one sacrificed");
        const QueueStats stats = queue.stats();
        check(stats.evicted == 1 && stats.evicted_by_camera.at("greedy") == 1,
              "the eviction is charged to the greedy camera");
        check(queue.depth() == 4, "depth is back at capacity");
    }

    void test_block_policy_waits_for_space() {
        FairPriorityQueue<Item> queue("q", 1, Overflow::Block, /*block_timeout_ms=*/500);
        queue.put(Item{"a", 1});
        std::atomic<bool> accepted{false};
        std::thread producer([&] { accepted.store(queue.put(Item{"a", 2}) == PutStatus::Accepted); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        check(!accepted.load(), "the producer is still blocked while the queue is full");
        const auto batch = queue.get_batch(BatchWindow(1));
        producer.join();
        check(batch.size() == 1 && accepted.load(), "it is admitted once a slot frees");
        check(queue.depth() == 1, "the second item is now queued");
    }

    void test_the_producer_wakes_when_the_slot_frees_not_when_the_deadline_expires() {
        FairPriorityQueue<Item> queue("q", 1, Overflow::Block, /*block_timeout_ms=*/500);
        queue.put(Item{"a", 1});
        Clock::time_point freed;
        std::atomic<double> waited_ms{0.0};
        std::thread producer([&] {
            const auto start = Clock::now();
            queue.put(Item{"a", 2});
            waited_ms.store(ms_since(start));
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        // The row-budget exit is the common one under load; the producer must wake on it.
        (void)queue.get_batch(BatchWindow(1));
        producer.join();
        check(waited_ms.load() < 250.0,
              "the producer woke when the slot freed (" + std::to_string(waited_ms.load()) +
                  " ms), not at the 500 ms deadline");
    }

    // -- TestQueueLifecycle ----------------------------------------------------------------

    void test_expired_requests_are_dropped_before_execution() {
        std::vector<std::pair<int, DropReason>> dropped;
        FairPriorityQueue<Item> queue("q", 8, Overflow::Reject, 50, true,
                                      [&](Item&& item, DropReason why) {
                                          dropped.emplace_back(item.id, why);
                                      });
        queue.put(Item{"a", 1, 1, Priority::Normal, now_ns() - 1});  // already past its deadline
        queue.put(Item{"a", 2});
        const auto batch = queue.get_batch(BatchWindow(8));
        check(batch.size() == 1 && batch[0].id == 2, "the live request is executed");
        check(dropped.size() == 1 && dropped[0].first == 1 && dropped[0].second == DropReason::Expired,
              "the expired one is handed back as expired, before execution");
        check(queue.stats().expired == 1, "and counted");
    }

    void test_close_fails_everything_still_queued() {
        std::vector<int> closed_ids;
        FairPriorityQueue<Item> queue("q", 8, Overflow::Reject, 50, true,
                                      [&](Item&& item, DropReason why) {
                                          if (why == DropReason::Closed) closed_ids.push_back(item.id);
                                      });
        queue.put(Item{"a", 1});
        queue.put(Item{"b", 2});
        queue.put(Item{"a", 3});
        (void)queue.close();
        check(closed_ids.size() == 3, "every queued request is failed on close, none silently");
        check(queue.put(Item{"c", 4}) == PutStatus::Closed, "a closed queue refuses new work by name");
        check(queue.get_batch(BatchWindow(4)).empty(), "and the consumer learns to exit");
        check(queue.depth() == 0, "depth is zero after close");
    }

    // -- TestDelayWindow -------------------------------------------------------------------

    void test_returns_immediately_when_batch_is_already_full() {
        FairPriorityQueue<Item> queue("q", 64);
        for (int i = 0; i < 4; ++i) queue.put(Item{"a", i});
        const auto start = Clock::now();
        const auto batch = queue.get_batch(BatchWindow(4, /*max_delay_us=*/200000));
        check(batch.size() == 4, "a full batch");
        check(ms_since(start) < 50.0, "returned without waiting out the window");
    }

    void test_waits_for_the_window_then_sends_a_partial_batch() {
        FairPriorityQueue<Item> queue("q", 64);
        queue.put(Item{"a", 1});
        const auto start = Clock::now();
        const auto batch = queue.get_batch(BatchWindow(8, /*max_delay_us=*/40000));
        const double waited = ms_since(start);
        check(batch.size() == 1, "the partial batch is sent");
        check(waited >= 30.0, "after the window (" + std::to_string(waited) + " ms)");
    }

    void test_late_arrivals_join_the_batch() {
        FairPriorityQueue<Item> queue("q", 64);
        queue.put(Item{"a", 1});
        std::thread late([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            queue.put(Item{"b", 2});
        });
        const auto batch = queue.get_batch(BatchWindow(8, /*max_delay_us=*/200000));
        late.join();
        check(batch.size() == 2, "the request that arrived during the window is in the batch");
    }

    void test_preferred_size_short_circuits_the_wait() {
        FairPriorityQueue<Item> queue("q", 64);
        queue.put(Item{"a", 1});
        queue.put(Item{"b", 2});
        const auto start = Clock::now();
        const auto batch = queue.get_batch(BatchWindow(8, /*max_delay_us=*/200000, {2}));
        check(batch.size() == 2, "a preferred size is taken as soon as it is reached");
        check(ms_since(start) < 50.0, "without waiting out the window");
    }

    void test_window_rejects_impossible_preferred_sizes() {
        bool refused = false;
        try {
            BatchWindow window(4, 0, {8});
            (void)window;
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check(refused, "a preferred size above max_batch_size is a configuration error");
    }

    // -- TestTheRowBudgetIsRespected -------------------------------------------------------

    void test_multi_row_requests_do_not_overfill_the_batch() {
        FairPriorityQueue<Item> queue("q", 64);
        for (int i = 0; i < 4; ++i) queue.put(Item{"cam" + std::to_string(i), i, 6});
        const auto batch = queue.get_batch(BatchWindow(16));
        size_t rows = 0;
        for (const Item& item : batch) rows += item.rows();
        check(batch.size() == 2 && rows == 12, "two 6-row requests fit a budget of 16; a third would not");
        check(queue.depth() == 2, "the remainder stays queued");
    }

    void test_a_request_larger_than_the_budget_is_still_dequeued() {
        FairPriorityQueue<Item> queue("q", 64);
        queue.put(Item{"a", 1, 24});
        queue.put(Item{"b", 2, 1});
        const auto first = queue.get_batch(BatchWindow(16));
        check(first.size() == 1 && first[0].id == 1, "the oversized request comes out alone");
        const auto second = queue.get_batch(BatchWindow(16));
        check(second.size() == 1 && second[0].id == 2, "and does not stall what is behind it");
    }

    // -- Lane ------------------------------------------------------------------------------

    void test_a_tie_between_greedy_cameras_goes_to_the_one_that_entered_first() {
        Lane<Item> lane;
        lane.push(Item{"first", 1});
        lane.push(Item{"second", 2});
        lane.push(Item{"first", 3});
        lane.push(Item{"second", 4});
        const std::optional<Item> victim = lane.evict_from_longest();
        check(victim.has_value() && victim->cam == "first" && victim->id == 1,
              "deterministic tie-break, and the oldest of that camera — the same answer Python gives");
        check(lane.size() == 3, "one item gone");
    }

}  // namespace

int main() {
    test_round_robin_across_cameras();
    test_fifo_is_not_fair();
    test_priority_beats_fairness();

    test_reject_when_full_reports_depth();
    test_drop_oldest_evicts_the_greediest_camera();
    test_block_policy_waits_for_space();
    test_the_producer_wakes_when_the_slot_frees_not_when_the_deadline_expires();

    test_expired_requests_are_dropped_before_execution();
    test_close_fails_everything_still_queued();

    test_returns_immediately_when_batch_is_already_full();
    test_waits_for_the_window_then_sends_a_partial_batch();
    test_late_arrivals_join_the_batch();
    test_preferred_size_short_circuits_the_wait();
    test_window_rejects_impossible_preferred_sizes();

    test_multi_row_requests_do_not_overfill_the_batch();
    test_a_request_larger_than_the_budget_is_still_dequeued();

    test_a_tie_between_greedy_cameras_goes_to_the_one_that_entered_first();

    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
