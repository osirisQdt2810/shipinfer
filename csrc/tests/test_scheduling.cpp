// The scheduling seam, tested the way the Python plane tests it.
//
// Every check here has a namesake in `tests/scheduling/` — `test_queue_fairness.py`,
// `test_batch_window.py`, `test_batch_rows.py` — and asserts the same thing, so a divergence
// between the planes shows up as one green file and one red one before the parity harness
// (ledger P6) ever runs a trace. No device, no GPU: the queue is pure logic on both sides.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/scheduling/dispatcher.h"
#include "shipinfer/scheduling/policies/join_shortest_queue.h"
#include "shipinfer/scheduling/policies/locality_spillover.h"
#include "shipinfer/scheduling/policies/power_of_two.h"
#include "shipinfer/scheduling/policies/registry.h"
#include "shipinfer/scheduling/policies/round_robin.h"
#include "shipinfer/scheduling/policies/sequence_affinity.h"
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
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   Clock::now().time_since_epoch())
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
        check(stats.depth == 2 && stats.capacity == 2,
              "depth and capacity are what the refusal saw");
        check(stats.rejected == 1 && stats.rejected_by_camera.at("b") == 1,
              "the refusal is charged to the camera that was refused");
    }

    void test_drop_oldest_evicts_the_greediest_camera() {
        std::vector<std::pair<int, DropReason>> dropped;
        FairPriorityQueue<Item> queue(
            "q", 4, Overflow::DropOldest, 50, true,
            [&](Item&& item, DropReason why) { dropped.emplace_back(item.id, why); });
        queue.put(Item{"greedy", 1});
        queue.put(Item{"greedy", 2});
        queue.put(Item{"greedy", 3});
        queue.put(Item{"quiet", 10});
        check(queue.put(Item{"another", 20}) == PutStatus::Accepted,
              "the newcomer is admitted");
        check(dropped.size() == 1 && dropped[0].second == DropReason::Evicted,
              "exactly one eviction");
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
        std::thread producer(
            [&] { accepted.store(queue.put(Item{"a", 2}) == PutStatus::Accepted); });
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
        check(waited_ms.load() < 250.0, "the producer woke when the slot freed (" +
                                            std::to_string(waited_ms.load()) +
                                            " ms), not at the 500 ms deadline");
    }

    // -- TestQueueLifecycle ----------------------------------------------------------------

    void test_expired_requests_are_dropped_before_execution() {
        std::vector<std::pair<int, DropReason>> dropped;
        FairPriorityQueue<Item> queue(
            "q", 8, Overflow::Reject, 50, true,
            [&](Item&& item, DropReason why) { dropped.emplace_back(item.id, why); });
        queue.put(
            Item{"a", 1, 1, Priority::Normal, now_ns() - 1});  // already past its deadline
        queue.put(Item{"a", 2});
        const auto batch = queue.get_batch(BatchWindow(8));
        check(batch.size() == 1 && batch[0].id == 2, "the live request is executed");
        check(dropped.size() == 1 && dropped[0].first == 1 &&
                  dropped[0].second == DropReason::Expired,
              "the expired one is handed back as expired, before execution");
        check(queue.stats().expired == 1, "and counted");
    }

    // -- TestPerCameraAttribution ----------------------------------------------------------
    // Who paid for each drop. The totals already say a queue refused, evicted or expired
    // work; they cannot say whose, and that is the exact question the inherited bug hid.
    // Namesakes of `TestPerCameraAttribution` in `tests/scheduling/test_queue_fairness.py`.

    void test_eviction_is_charged_to_the_greedy_camera_alone() {
        FairPriorityQueue<Item> queue("q", 4, Overflow::DropOldest);
        for (int i = 0; i < 3; ++i) queue.put(Item{"loud", i});
        queue.put(Item{"quiet", 10});
        queue.put(Item{"loud", 99});
        const QueueStats stats = queue.stats();
        check(stats.evicted == 1 && stats.evicted_by_camera.size() == 1 &&
                  stats.evicted_by_camera.at("loud") == 1,
              "the flood pays for its own flood, and nobody else is named");
        check(stats.depth_by_camera.at("quiet") == 1, "the quiet camera's frame is untouched");
    }

    void test_expiry_names_only_the_camera_that_was_late() {
        FairPriorityQueue<Item> queue("q", 8);
        queue.put(Item{"late", 1, 1, Priority::Normal, now_ns() - 1});
        queue.put(Item{"ontime", 2});
        (void)queue.get_batch(BatchWindow(8));
        const QueueStats stats = queue.stats();
        check(stats.expired == 1 && stats.expired_by_camera.size() == 1 &&
                  stats.expired_by_camera.at("late") == 1,
              "only the camera whose deadline passed is charged for the expiry");
    }

    void test_depth_by_camera_sums_to_depth_across_priority_bands() {
        // A camera with work in two lanes is one camera; a breakdown that does not add up to
        // `depth` is worse than none.
        FairPriorityQueue<Item> queue("q", 32);
        for (int i = 0; i < 3; ++i) queue.put(Item{"cam_a", i});
        queue.put(Item{"cam_a", 9, 1, Priority::TrackingCritical});
        for (int i = 0; i < 2; ++i) queue.put(Item{"cam_b", i, 1, Priority::Background});
        const QueueStats stats = queue.stats();
        check(stats.depth_by_camera.size() == 2 && stats.depth_by_camera.at("cam_a") == 4 &&
                  stats.depth_by_camera.at("cam_b") == 2,
              "one entry per camera, summed over every lane it has work in");
        size_t total = 0;
        for (const auto& entry : stats.depth_by_camera) total += entry.second;
        check(total == stats.depth && stats.depth == 6, "the breakdown adds up to the depth");
    }

    void test_close_does_not_charge_anybody() {
        // Shutdown loss is not a per-camera fault; the runner's `items_queue_closed` owns it.
        FairPriorityQueue<Item> queue("q", 8);
        for (int i = 0; i < 3; ++i) queue.put(Item{"cam_a", i});
        (void)queue.close();
        const QueueStats stats = queue.stats();
        check(stats.evicted_by_camera.empty() && stats.expired_by_camera.empty() &&
                  stats.rejected_by_camera.empty() && stats.depth_by_camera.empty(),
              "an orderly stop must not read like a flood");
    }

    void test_fifo_attributes_the_same_four_outcomes() {
        // The fairness-blind control reports the same maps — a comparison where only one side
        // can name a victim is not a comparison. What differs is *who* gets named.
        FifoQueue<Item> queue("q", 4, Overflow::DropOldest);
        queue.put(Item{"quiet", 10});  // oldest, and blameless
        for (int i = 0; i < 3; ++i) queue.put(Item{"loud", i});
        queue.put(Item{"loud", 99});
        const QueueStats evicting = queue.stats();
        check(evicting.evicted_by_camera.size() == 1 &&
                  evicting.evicted_by_camera.at("quiet") == 1,
              "FIFO sacrifices the blameless head — the inherited bug, now visible");
        check(evicting.depth_by_camera.size() == 1 && evicting.depth_by_camera.at("loud") == 4,
              "and the whole queue is now the loud camera's");

        FifoQueue<Item> refusing("q", 2, Overflow::Reject);
        refusing.put(Item{"late", 1, 1, Priority::Normal, now_ns() - 1});
        refusing.put(Item{"ontime", 2});
        check(refusing.put(Item{"newcomer", 3}) == PutStatus::Rejected, "the third is refused");
        (void)refusing.get_batch(BatchWindow(8));
        const QueueStats stats = refusing.stats();
        check(stats.rejected_by_camera.at("newcomer") == 1, "the refusal names the newcomer");
        check(stats.expired_by_camera.size() == 1 && stats.expired_by_camera.at("late") == 1,
              "and the expiry names the camera that was late");
    }

    void test_stats_hands_out_copies_not_the_live_maps() {
        // `stats()` returns by value; mutating the snapshot must not reach the queue. The
        // Python plane pins the same property on `as_dict()`, which is what /v2/statistics
        // serialises.
        FairPriorityQueue<Item> queue("q", 1, Overflow::Reject);
        queue.put(Item{"cam_a", 1});
        check(queue.put(Item{"cam_b", 2}) == PutStatus::Rejected, "the second is refused");
        QueueStats snapshot = queue.stats();
        snapshot.rejected_by_camera["cam_b"] = 999;
        snapshot.rejected_by_camera["ghost"] = 1;
        snapshot.depth_by_camera.clear();
        const QueueStats fresh = queue.stats();
        check(fresh.rejected_by_camera.size() == 1 && fresh.rejected_by_camera.at("cam_b") == 1,
              "the queue's own counters are untouched by an edited snapshot");
        check(fresh.depth_by_camera.at("cam_a") == 1, "and so is the depth breakdown");
    }

    void test_close_fails_everything_still_queued() {
        std::vector<int> closed_ids;
        FairPriorityQueue<Item> queue("q", 8, Overflow::Reject, 50, true,
                                      [&](Item&& item, DropReason why) {
                                          if (why == DropReason::Closed)
                                              closed_ids.push_back(item.id);
                                      });
        queue.put(Item{"a", 1});
        queue.put(Item{"b", 2});
        queue.put(Item{"a", 3});
        (void)queue.close();
        check(closed_ids.size() == 3, "every queued request is failed on close, none silently");
        check(queue.put(Item{"c", 4}) == PutStatus::Closed,
              "a closed queue refuses new work by name");
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
        check(batch.size() == 2 && rows == 12,
              "two 6-row requests fit a budget of 16; a third would not");
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
              "deterministic tie-break, and the oldest of that camera — the same answer Python "
              "gives");
        check(lane.size() == 3, "one item gone");
    }

    void test_a_serve_does_not_change_who_is_evicted_next() {
        // The trace from review round 3: push A, A, B; serve one (A's oldest); now A and B
        // are tied at one each. Python scans `by_key` in insertion order and evicts A's
        // remaining request; a scan of the *rotation* — reordered by the serve — evicted B's.
        Lane<Item> lane;
        lane.push(Item{"first", 1});
        lane.push(Item{"first", 2});
        lane.push(Item{"second", 3});
        const Item served = lane.pop();
        check(served.cam == "first" && served.id == 1, "round-robin serves first's oldest");
        const std::optional<Item> victim = lane.evict_from_longest();
        check(victim.has_value() && victim->cam == "first" && victim->id == 2,
              "the tie goes to the key seen first, not to the key at the front of the "
              "rotation — the same item Python evicts on this trace");
        check(lane.size() == 1, "second's request survives");
        // A key that empties and returns re-enters at the back, as a dict key would.
        lane.push(Item{"first", 4});
        lane.push(Item{"first", 5});
        lane.push(Item{"second", 6});
        const std::optional<Item> next = lane.evict_from_longest();
        check(next.has_value() && next->cam == "second" && next->id == 3,
              "second (seen earlier, tied at two) is the greediest by insertion order");
    }

    // -- the policies (tests/scheduling/test_policies.py) -------------------------------------

    // A `Placeable` with no machinery behind it. The policies are given exactly four attributes
    // by contract, so a four-field struct is a *complete* test double — which is itself
    // evidence the contract is narrow enough.
    struct FakeInstance : Placeable {
        Device dev;
        size_t depth_ = 0;
        bool ready = true;
        FakeInstance(Device d, size_t depth = 0, bool is_ready = true)
            : dev(d), depth_(depth), ready(is_ready) {}
        Device device() const override { return dev; }
        size_t depth() const override { return depth_; }
        double ewma_latency_us() const override { return 0.0; }
        bool is_ready() const override { return ready; }
    };

    std::vector<Placeable*> pointers(std::vector<FakeInstance>& instances) {
        std::vector<Placeable*> out;
        for (FakeInstance& i : instances) out.push_back(&i);
        return out;
    }

    void test_round_robin_rotates() {
        std::vector<FakeInstance> pool{FakeInstance(Device::cuda(0)),
                                       FakeInstance(Device::cuda(1)),
                                       FakeInstance(Device::cuda(2))};
        auto candidates = pointers(pool);
        RoundRobinPolicy policy;
        std::vector<int> picked;
        for (int i = 0; i < 6; ++i)
            picked.push_back(policy.select(candidates, {})->device().index);
        check(picked == std::vector<int>{0, 1, 2, 0, 1, 2}, "round robin rotates in order");
    }

    void test_join_shortest_queue_picks_the_shortest() {
        std::vector<FakeInstance> pool{FakeInstance(Device::cuda(0), 5),
                                       FakeInstance(Device::cuda(1), 2),
                                       FakeInstance(Device::cuda(2), 9)};
        auto candidates = pointers(pool);
        JoinShortestQueuePolicy policy;
        check(policy.select(candidates, {})->device().index == 1,
              "jsq picks the shortest queue");
    }

    void test_power_of_two_never_picks_the_same_instance_twice() {
        // With two candidates the two probes must be the two instances, so the shorter one wins
        // every single time — a with-replacement sampler would sometimes probe one twice and
        // return the longer queue.
        std::vector<FakeInstance> pool{FakeInstance(Device::cuda(0), 10),
                                       FakeInstance(Device::cuda(1), 1)};
        auto candidates = pointers(pool);
        PowerOfTwoChoicesPolicy policy(/*seed=*/7);
        bool always_shorter = true;
        for (int i = 0; i < 200; ++i)
            always_shorter &= policy.select(candidates, {})->device().index == 1;
        check(always_shorter, "two probes without replacement always find the shorter of two");
    }

    void test_locality_keeps_work_on_the_resident_gpu() {
        std::vector<FakeInstance> pool{FakeInstance(Device::cuda(0), 3),
                                       FakeInstance(Device::cuda(1), 0)};
        auto candidates = pointers(pool);
        LocalityAwareSpilloverPolicy policy(/*spill_threshold=*/4);
        PlacementRequest request{Device::cuda(0), "cam"};
        check(policy.select(candidates, request)->device().index == 0,
              "a resident GPU at or under the threshold keeps the work, even with an idle "
              "neighbour");
    }

    void test_locality_spills_once_the_resident_gpu_backs_up() {
        std::vector<FakeInstance> pool{FakeInstance(Device::cuda(0), 9),
                                       FakeInstance(Device::cuda(1), 0)};
        auto candidates = pointers(pool);
        LocalityAwareSpilloverPolicy policy(/*spill_threshold=*/4,
                                            std::make_unique<JoinShortestQueuePolicy>());
        PlacementRequest request{Device::cuda(0), "cam"};
        check(policy.select(candidates, request)->device().index == 1,
              "past the threshold the copy is cheaper than the wait");
    }

    void test_locality_falls_back_when_there_is_no_hint() {
        std::vector<FakeInstance> pool{FakeInstance(Device::cuda(0), 9),
                                       FakeInstance(Device::cuda(1), 0)};
        auto candidates = pointers(pool);
        LocalityAwareSpilloverPolicy policy(4, std::make_unique<JoinShortestQueuePolicy>());
        check(policy.select(candidates, PlacementRequest{})->device().index == 1,
              "no resident device: the fallback decides");
    }

    void test_sequence_affinity_pins_a_camera() {
        std::vector<FakeInstance> pool{FakeInstance(Device::cuda(0)),
                                       FakeInstance(Device::cuda(1)),
                                       FakeInstance(Device::cuda(2))};
        auto candidates = pointers(pool);
        SequenceAffinityPolicy policy(std::make_unique<RoundRobinPolicy>());
        PlacementRequest cam3{std::nullopt, "cam3"};
        Placeable* first = policy.select(candidates, cam3);
        bool sticky = true;
        for (int i = 0; i < 5; ++i) sticky &= policy.select(candidates, cam3) == first;
        check(sticky, "every request of a camera lands on the same instance");
        check(policy.select(candidates, PlacementRequest{std::nullopt, "cam4"}) != first ||
                  candidates.size() == 1,
              "another camera is placed by the fallback, not by cam3's pin");
    }

    void test_sequence_affinity_repins_when_the_instance_dies() {
        std::vector<FakeInstance> pool{FakeInstance(Device::cuda(0)),
                                       FakeInstance(Device::cuda(1))};
        auto candidates = pointers(pool);
        SequenceAffinityPolicy policy(std::make_unique<RoundRobinPolicy>());
        PlacementRequest cam{std::nullopt, "cam"};
        Placeable* pinned = policy.select(candidates, cam);
        static_cast<FakeInstance*>(pinned)->ready = false;
        std::vector<Placeable*> still_ready;
        for (Placeable* p : candidates) {
            if (p->is_ready()) still_ready.push_back(p);
        }
        Placeable* repinned = policy.select(still_ready, cam);
        check(repinned != pinned && repinned->is_ready(),
              "a dead instance's camera is re-pinned, not dropped");
        check(policy.select(still_ready, cam) == repinned, "and the new pin sticks");
    }

    void test_every_policy_is_registered_and_buildable() {
        const std::set<std::string> want{"round_robin", "join_shortest_queue", "power_of_two",
                                         "locality_spillover", "sequence_affinity"};
        std::set<std::string> have;
        for (const std::string& name : POLICIES().names()) have.insert(name);
        check(std::includes(have.begin(), have.end(), want.begin(), want.end()),
              "the five policies the Python plane registers are registered here");
        bool all_describe = true;
        for (const std::string& name : POLICIES().names()) {
            all_describe &= !build_policy(name)->describe().empty();
        }
        check(all_describe, "every policy builds by name and describes itself");
        check(build_policy("jsq")->name() == "join_shortest_queue" &&
                  build_policy("locality")->name() == "locality_spillover" &&
                  build_policy("sticky")->name() == "sequence_affinity",
              "the Python aliases resolve here too");
        check(static_cast<LocalityAwareSpilloverPolicy&>(
                  *build_policy("locality_spillover", {{"spill_threshold", "9"}}))
                      .spill_threshold() == 9,
              "options reach the constructor");
    }

    void test_unknown_policy_names_its_alternatives() {
        bool refused = false;
        std::string message;
        try {
            (void)build_policy("nonexistent");
        } catch (const ConfigError& error) {
            refused = true;
            message = error.what();
        }
        check(refused && message.find("round_robin") != std::string::npos,
              "an unknown policy is refused with the known names in the message");
        refused = false;
        try {
            (void)build_policy("round_robin", {{"spill_threshold", "3"}});
        } catch (const ConfigError&) {
            refused = true;
        }
        check(refused, "an option the constructor does not take is a configuration error");
    }

    // -- the dispatcher (tests/scheduling/test_dispatcher.py) ---------------------------------

    struct Pool {
        std::vector<FakeInstance> instances;
        std::set<Placeable*> full;
        std::vector<Placeable*> landed;
        PutStatus enqueue(Placeable* instance) {
            if (full.count(instance)) return PutStatus::Rejected;
            landed.push_back(instance);
            return PutStatus::Accepted;
        }
    };

    void test_dispatch_places_on_the_policy_choice() {
        Pool pool{{FakeInstance(Device::cuda(0), 5), FakeInstance(Device::cuda(1), 1)}};
        Dispatcher dispatcher("m", pointers(pool.instances),
                              std::make_unique<JoinShortestQueuePolicy>());
        const DispatchResult result =
            dispatcher.dispatch({}, [&](Placeable* i) { return pool.enqueue(i); });
        check(result.instance->device().index == 1 && result.attempts == 1 && !result.spilled,
              "the item lands where the policy said, first try");
    }

    void test_dispatch_skips_instances_that_are_not_ready() {
        Pool pool{{FakeInstance(Device::cuda(0), 0, /*ready=*/false),
                   FakeInstance(Device::cuda(1), 7)}};
        Dispatcher dispatcher("m", pointers(pool.instances),
                              std::make_unique<JoinShortestQueuePolicy>());
        const DispatchResult result =
            dispatcher.dispatch({}, [&](Placeable* i) { return pool.enqueue(i); });
        check(result.instance->device().index == 1,
              "an instance that is not ready is never a candidate");
    }

    void test_dispatch_spills_when_the_first_choice_is_full() {
        Pool pool{{FakeInstance(Device::cuda(0), 0), FakeInstance(Device::cuda(1), 3),
                   FakeInstance(Device::cuda(2), 1)}};
        auto candidates = pointers(pool.instances);
        pool.full.insert(candidates[0]);  // the policy's choice refuses
        std::vector<std::pair<int, int>> spills;
        Dispatcher dispatcher("m", candidates, std::make_unique<JoinShortestQueuePolicy>(),
                              [&](Placeable* wanted, Placeable* actual) {
                                  spills.emplace_back(wanted->device().index,
                                                      actual->device().index);
                              });
        const DispatchResult result =
            dispatcher.dispatch({}, [&](Placeable* i) { return pool.enqueue(i); });
        check(result.spilled && result.attempts == 2 && result.instance->device().index == 2,
              "spills to the next-shortest queue, once");
        check(spills == std::vector<std::pair<int, int>>{{0, 2}},
              "the spill is reported as wanted -> actual");
    }

    void test_dispatch_raises_only_when_the_whole_pool_is_saturated() {
        Pool pool{{FakeInstance(Device::cuda(0)), FakeInstance(Device::cuda(1))}};
        auto candidates = pointers(pool.instances);
        pool.full.insert(candidates[0]);
        pool.full.insert(candidates[1]);
        Dispatcher dispatcher("m", candidates, std::make_unique<RoundRobinPolicy>());
        bool refused = false;
        try {
            (void)dispatcher.dispatch({}, [&](Placeable* i) { return pool.enqueue(i); });
        } catch (const QueueFullError&) {
            refused = true;
        }
        check(refused,
              "every instance refused: the pool is saturated, and the caller hears that");
    }

    void test_dispatch_raises_when_nothing_is_ready() {
        Pool pool{{FakeInstance(Device::cuda(0), 0, false)}};
        Dispatcher dispatcher("m", pointers(pool.instances),
                              std::make_unique<RoundRobinPolicy>());
        bool refused = false;
        try {
            (void)dispatcher.dispatch({}, [&](Placeable* i) { return pool.enqueue(i); });
        } catch (const ServerStateError&) {
            refused = true;
        }
        check(refused, "nothing ready is a server-state error, not a full queue");
    }

    void test_dispatcher_refuses_to_exist_without_instances() {
        bool refused = false;
        try {
            Dispatcher dispatcher("m", {}, std::make_unique<RoundRobinPolicy>());
        } catch (const ServerStateError&) {
            refused = true;
        }
        check(refused, "a dispatcher over no instances is refused at construction");
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

    test_eviction_is_charged_to_the_greedy_camera_alone();
    test_expiry_names_only_the_camera_that_was_late();
    test_depth_by_camera_sums_to_depth_across_priority_bands();
    test_close_does_not_charge_anybody();
    test_fifo_attributes_the_same_four_outcomes();
    test_stats_hands_out_copies_not_the_live_maps();

    test_returns_immediately_when_batch_is_already_full();
    test_waits_for_the_window_then_sends_a_partial_batch();
    test_late_arrivals_join_the_batch();
    test_preferred_size_short_circuits_the_wait();
    test_window_rejects_impossible_preferred_sizes();

    test_multi_row_requests_do_not_overfill_the_batch();
    test_a_request_larger_than_the_budget_is_still_dequeued();

    test_a_tie_between_greedy_cameras_goes_to_the_one_that_entered_first();
    test_a_serve_does_not_change_who_is_evicted_next();

    test_round_robin_rotates();
    test_join_shortest_queue_picks_the_shortest();
    test_power_of_two_never_picks_the_same_instance_twice();
    test_locality_keeps_work_on_the_resident_gpu();
    test_locality_spills_once_the_resident_gpu_backs_up();
    test_locality_falls_back_when_there_is_no_hint();
    test_sequence_affinity_pins_a_camera();
    test_sequence_affinity_repins_when_the_instance_dies();
    test_every_policy_is_registered_and_buildable();
    test_unknown_policy_names_its_alternatives();

    test_dispatch_places_on_the_policy_choice();
    test_dispatch_skips_instances_that_are_not_ready();
    test_dispatch_spills_when_the_first_choice_is_full();
    test_dispatch_raises_only_when_the_whole_pool_is_saturated();
    test_dispatch_raises_when_nothing_is_ready();
    test_dispatcher_refuses_to_exist_without_instances();

    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
