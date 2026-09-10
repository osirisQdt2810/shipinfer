// The instant barrier's synchronisation properties, with strings and a callback.
//
// The point of the barrier being pure is that this file needs no tracker, no submodule and no
// device: what is under test is who waits, who closes, who is told to give up, and what a
// frame gets back. `topology/barrier.py`'s own suite makes the same argument on the other
// plane, and the two files are deliberately the same list of decisions.
#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/pipeline/mtmc/barrier.h"

namespace {

    using namespace shipinfer;
    using shipinfer::mtmc::Association;
    using shipinfer::mtmc::InstantBarrier;
    using shipinfer::mtmc::InstantEntry;
    using shipinfer::mtmc::InstantOutcome;
    using shipinfer::mtmc::Results;
    using shipinfer::mtmc::WaiterBudget;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    // The payload and the answer are both strings: the barrier must not care.
    std::shared_ptr<void> payload_of(const std::string& text) {
        return std::make_shared<std::string>(text);
    }

    std::string read(const std::shared_ptr<void>& payload) {
        return *std::static_pointer_cast<std::string>(payload);
    }

    // Joins every camera's payload, so a caller can see WHICH entries the association was
    // given -- which is the only thing about the callback that matters here.
    Results join(const std::vector<InstantEntry>& entries) {
        std::string joined;
        for (const InstantEntry& entry : entries) {
            if (!joined.empty()) joined += ",";
            joined += entry.camera_id + ":" + read(entry.payload);
        }
        return std::make_shared<const std::string>(joined);
    }

    std::string answer_of(const InstantOutcome& outcome) {
        if (!outcome.associated || !outcome.results) return "";
        return *std::static_pointer_cast<const std::string>(outcome.results);
    }

    const Association kJoin = join;

    // A clock the test moves by hand, so the window is testable without sleeping.
    class Clock {
      public:
        double now() const { return now_.load(); }
        void advance(double seconds) { now_.store(now_.load() + seconds); }
        InstantBarrier::Clock fn() {
            return [this] { return now_.load(); };
        }

      private:
        std::atomic<double> now_{1000.0};
    };

    InstantBarrier::Options options(double window_s, int workers, int max_instants = 8) {
        InstantBarrier::Options built;
        built.sync_window_s = window_s;
        built.workers = workers;
        built.max_instants = max_instants;
        return built;
    }

    // ---------------------------------------------------------------- construction

    void a_zero_window_is_refused() {
        bool refused = false;
        try {
            InstantBarrier barrier(options(0.0, 2));
        } catch (const ConfigError& error) {
            refused = std::string(error.what()).find("sync_window_s must be positive") !=
                      std::string::npos;
        }

        check(refused,
              "a zero window is a ConfigError: it admits one camera and refuses the "
              "rest of the group as late");
    }

    void a_barrier_with_no_room_for_an_instant_is_refused() {
        bool refused = false;
        try {
            InstantBarrier barrier(options(0.06, 2, /*max_instants=*/0));
        } catch (const ConfigError& error) {
            refused = std::string(error.what()).find("max_instants") != std::string::npos;
        }

        check(refused, "zero open instants means every frame evicts itself");
    }

    void a_negative_budget_is_refused_and_zero_is_not() {
        bool refused = false;
        try {
            WaiterBudget budget(-1);
        } catch (const ConfigError&) {
            refused = true;
        }
        WaiterBudget none(0);

        check(refused, "a negative permit count is refused");
        check(!none.acquire(), "and 0 permits means never wait, which is legitimate");
    }

    void over_releasing_a_budget_is_a_typed_refusal() {
        WaiterBudget budget(1);
        check(budget.acquire(), "one permit is available");
        budget.release();
        bool refused = false;
        try {
            budget.release();
        } catch (const ServerStateError& error) {
            refused = std::string(error.what()).find("more times than it was acquired") !=
                      std::string::npos;
        }

        check(refused, "a silent decrement past zero would hand out permits that do not exist");
        check(budget.held() == 0, "and the count is still honest");
    }

    // ------------------------------------------------- the instant is anchored, not gridded

    void a_group_of_one_closes_on_its_own_frame() {
        InstantBarrier barrier(options(0.06, 1));

        const InstantOutcome outcome = barrier.submit("cam0", 100.0, payload_of("a"), kJoin);

        check(outcome.reason == mtmc::kClosedComplete, "one camera is a complete group");
        check(outcome.associated, "and it was associated");
        check(answer_of(outcome) == "cam0:a", "over its own entry");
    }

    void two_captures_inside_one_window_are_one_instant() {
        // FROZEN CLOCK, so the parked waiter cannot time out before the second frame arrives.
        // With the real clock this asserted that 60 ms of window outlasts a thread handoff --
        // true on an idle box, false under a sanitiser, and not what the test is about.
        Clock clock;
        InstantBarrier barrier(options(0.06, 4), nullptr, clock.fn());
        barrier.camera_added("cam0");
        barrier.camera_added("cam1");

        // cam0 waits -- but with workers=4 the budget has 3 permits, so it parks. Run it on a
        // thread and let cam1 complete the instant.
        InstantOutcome first;
        std::thread waiter(
            [&] { first = barrier.submit("cam0", 100.000, payload_of("a"), kJoin); });
        while (barrier.waiters() == 0) std::this_thread::yield();
        const InstantOutcome second = barrier.submit("cam1", 100.055, payload_of("b"), kJoin);
        waiter.join();

        check(second.reason == mtmc::kClosedComplete, "the second frame completed the instant");
        check(first.instant == second.instant, "both frames are in ONE instant");
        check(answer_of(first) == answer_of(second), "and both read the same answer");
        check(answer_of(second) == "cam0:a,cam1:b", "which covers both cameras");
    }

    void two_captures_more_than_a_window_apart_are_two_instants() {
        Clock clock;
        InstantBarrier barrier(options(0.06, 1), nullptr, clock.fn());

        const InstantOutcome first = barrier.submit("cam0", 100.0, payload_of("a"), kJoin);
        const InstantOutcome second = barrier.submit("cam0", 100.5, payload_of("b"), kJoin);

        check(first.instant != second.instant, "half a second apart is two instants");
        check(second.reason == mtmc::kClosedComplete, "and the second one closed on its own");
    }

    void a_camera_never_collides_with_itself_at_any_window() {
        // The absolute-grid failure: a window WIDER than the frame period used to put two
        // consecutive frames of one camera in one bucket, and each was refused an answer.
        for (const double window : {0.02, 0.06, 0.2, 1.0}) {
            Clock clock;
            InstantBarrier barrier(options(window, 1), nullptr, clock.fn());
            int duplicates = 0;
            for (int frame = 0; frame < 40; ++frame) {
                const double capture = 100.0 + 0.05 * frame;  // 20 fps, no jitter
                const InstantOutcome outcome =
                    barrier.submit("cam0", capture, payload_of("x"), kJoin);
                if (outcome.reason == mtmc::kMissedDuplicate) ++duplicates;
            }
            check(duplicates == 0, "no self-collision at window " + std::to_string(window));
        }
    }

    void a_second_frame_from_a_camera_in_the_bucket_closes_it() {
        Clock clock;
        InstantBarrier barrier(options(0.06, 4), nullptr, clock.fn());
        barrier.camera_added("cam0");
        barrier.camera_added("cam1");

        InstantOutcome first;
        InstantOutcome advanced;
        std::thread waiter(
            [&] { first = barrier.submit("cam0", 100.0, payload_of("a"), kJoin); });
        while (barrier.waiters() == 0) std::this_thread::yield();
        // cam0's NEXT frame: that instant has every frame it will ever get from cam0. On a
        // thread of its own because it opens an instant of its own and parks on it -- and with
        // a frozen injected clock a parked waiter never times out, which is what makes the
        // window testable and would hang the test if this ran on the main thread.
        std::thread mover(
            [&] { advanced = barrier.submit("cam0", 100.05, payload_of("a2"), kJoin); });
        waiter.join();
        while (barrier.waiters() == 0) std::this_thread::yield();
        clock.advance(1.0);
        mover.join();

        check(first.reason == mtmc::kClosedAdvanced,
              "the parked frame's instant closed because its own camera moved on");
        check(first.associated && answer_of(first) == "cam0:a",
              "over the one entry it had, and it still got an answer");
        check(advanced.instant != first.instant, "the later frame opened the next instant");
        check(advanced.reason == mtmc::kClosedWindow,
              "and closed on its own window once the clock moved");
    }

    void a_repeat_of_a_capture_already_in_the_bucket_is_a_duplicate() {
        InstantBarrier barrier(options(0.06, 4));
        barrier.camera_added("cam0");
        barrier.camera_added("cam1");

        InstantOutcome parked;
        std::thread waiter(
            [&] { parked = barrier.submit("cam0", 100.0, payload_of("a"), kJoin); });
        while (barrier.waiters() == 0) std::this_thread::yield();
        const InstantOutcome repeat = barrier.submit("cam0", 100.0, payload_of("a"), kJoin);

        check(repeat.reason == mtmc::kMissedDuplicate, "the same capture twice is a duplicate");
        check(!repeat.associated, "and gets no answer");
        barrier.close_all();
        waiter.join();
    }

    void another_camera_running_ahead_does_not_make_the_next_frame_a_duplicate() {
        // `last` is the maximum over the GROUP, so testing a repeat against it would refuse
        // cam0's genuinely next frame whenever cam1 has pushed the span past it: a@100.000,
        // b@100.055, then a@100.050 -- 50 ms after a's own frame, a duplicate of nothing.
        InstantBarrier barrier(options(0.06, 8));
        barrier.camera_added("cam0");
        barrier.camera_added("cam1");
        barrier.camera_added("cam2");

        InstantOutcome a;
        InstantOutcome b;
        std::thread first([&] { a = barrier.submit("cam0", 100.000, payload_of("a"), kJoin); });
        while (barrier.waiters() < 1) std::this_thread::yield();
        std::thread second(
            [&] { b = barrier.submit("cam1", 100.055, payload_of("b"), kJoin); });
        while (barrier.waiters() < 2) std::this_thread::yield();
        const InstantOutcome next = barrier.submit("cam0", 100.050, payload_of("a2"), kJoin);
        first.join();
        second.join();

        check(next.reason != mtmc::kMissedDuplicate,
              "cam0's own next frame is not a duplicate because cam1 ran ahead");
        check(a.reason == mtmc::kClosedAdvanced, "it sealed the instant cam0 was already in");
    }

    // ------------------------------------------------------------- the answer and the scatter

    void the_answer_is_handed_back_whole_and_never_indexed() {
        // A group's association answers for every camera in one flattened list, so the
        // barrier publishes what it was given and each caller reads its own entry out.
        InstantBarrier barrier(options(0.06, 4));
        barrier.camera_added("cam0");
        barrier.camera_added("cam1");

        InstantOutcome parked;
        std::thread waiter(
            [&] { parked = barrier.submit("cam0", 100.0, payload_of("a"), kJoin); });
        while (barrier.waiters() == 0) std::this_thread::yield();
        const InstantOutcome closer = barrier.submit("cam1", 100.01, payload_of("b"), kJoin);
        waiter.join();

        check(parked.results == closer.results,
              "both frames hold the SAME answer object, not two copies");
        check(answer_of(parked).find("cam0:a") != std::string::npos &&
                  answer_of(parked).find("cam1:b") != std::string::npos,
              "and it covers the whole group");
    }

    void an_empty_answer_is_not_a_missed_frame() {
        // ADR-005: never an empty result to mean failure. An instant in which nothing was
        // visible on any camera legitimately returns an empty answer.
        InstantBarrier barrier(options(0.06, 1));
        const Association nothing = [](const std::vector<InstantEntry>&) -> Results {
            return std::make_shared<const std::string>("");
        };

        const InstantOutcome outcome = barrier.submit("cam0", 100.0, payload_of("a"), nothing);

        check(outcome.associated, "an association ran");
        check(answer_of(outcome).empty(), "and answered nothing, which is not the same fact");
    }

    // ------------------------------------------------------------------- the window and time

    void a_missing_camera_costs_one_window_and_the_association_still_runs() {
        Clock clock;
        InstantBarrier barrier(options(0.06, 4), nullptr, clock.fn());
        barrier.camera_added("cam0");
        barrier.camera_added("cam-absent");

        InstantOutcome outcome;
        std::thread waiter(
            [&] { outcome = barrier.submit("cam0", 100.0, payload_of("a"), kJoin); });
        while (barrier.waiters() == 0) std::this_thread::yield();
        clock.advance(0.07);
        // The waiter re-derives `remaining` from the injected clock on every wake, so a
        // notify is all it needs to notice the window is gone.
        barrier.camera_added("cam-absent");
        waiter.join();

        check(outcome.reason == mtmc::kClosedWindow,
              "the window ran out with a camera missing");
        check(outcome.associated, "and a partial instant is worth more than no instant");
        check(answer_of(outcome) == "cam0:a", "over whoever did report");
    }

    void an_instant_does_not_expire_until_the_injected_clock_says_so() {
        Clock clock;
        InstantBarrier barrier(options(0.06, 1), nullptr, clock.fn());
        barrier.submit("cam0", 100.0, payload_of("a"), kJoin);
        barrier.camera_added("cam0");
        barrier.camera_added("cam1");
        InstantOutcome outcome;
        std::thread waiter(
            [&] { outcome = barrier.submit("cam0", 200.0, payload_of("b"), kJoin); });

        // workers=1 means zero permits, so this frame never parks at all.
        waiter.join();

        check(outcome.reason == mtmc::kMissedWouldStarve,
              "a single-worker runner never waits: the guard fires immediately");
        check(barrier.open_instants() == 1, "and the instant stays open for the group");
    }

    // -------------------------------------------------------------------------- late frames

    void a_frame_for_a_closed_instant_is_told_it_is_late() {
        InstantBarrier barrier(options(0.06, 1));
        const InstantOutcome closed = barrier.submit("cam0", 100.0, payload_of("a"), kJoin);

        const InstantOutcome late = barrier.submit("cam1", 100.0, payload_of("b"), kJoin);

        check(late.reason == mtmc::kMissedLate, "a frame inside a resolved instant is late");
        check(late.instant == closed.instant, "and it names the instant it missed");
        check(!late.associated,
              "never retro-fitted: a second answer would contradict the first");
    }

    void a_late_frame_does_not_re_open_its_instant() {
        InstantBarrier barrier(options(0.06, 1));
        barrier.submit("cam0", 100.0, payload_of("a"), kJoin);

        barrier.submit("cam1", 100.0, payload_of("b"), kJoin);

        check(barrier.open_instants() == 0, "the late frame opened nothing");
        check(barrier.frame_stats()[mtmc::kMissedLate] == 1, "and was counted as a frame miss");
        check(barrier.instant_stats().count(mtmc::kMissedLate) == 0,
              "a frame-level miss is not an instant-level event");
    }

    // --------------------------------------------------------------------------- boundedness

    void the_oldest_instant_is_evicted_and_counted() {
        Clock clock;
        InstantBarrier barrier(options(0.06, 1, /*max_instants=*/3), nullptr, clock.fn());
        barrier.camera_added("cam0");
        barrier.camera_added("cam-absent");

        std::vector<int64_t> instants;
        for (int frame = 0; frame < 5; ++frame) {
            // Each capture is its own instant (a second apart) and none can complete, so the
            // map fills and then evicts.
            instants.push_back(
                barrier.submit("cam0", 100.0 + frame, payload_of("x"), kJoin).instant);
        }

        check(barrier.open_instants() <= 3, "the map is bounded by max_instants");
        check(barrier.instant_stats()[mtmc::kDroppedEvicted] >= 1,
              "and the eviction is counted where an operator reads clock skew");
        check(instants.front() != instants.back(), "each capture opened its own instant");
    }

    void an_evicted_instant_releases_the_frames_waiting_on_it() {
        // Every instant here has to STAY open for the map to fill, which is why the clock is
        // frozen and all three frames are on threads: a frame that closed on its own window
        // would leave room and nothing would ever be evicted.
        Clock clock;
        InstantBarrier barrier(options(0.06, 8, /*max_instants=*/2), nullptr, clock.fn());
        barrier.camera_added("cam0");
        barrier.camera_added("cam-absent");

        InstantOutcome first;
        InstantOutcome second;
        InstantOutcome third;
        std::thread one([&] { first = barrier.submit("cam0", 100.0, payload_of("a"), kJoin); });
        while (barrier.waiters() < 1) std::this_thread::yield();
        std::thread two(
            [&] { second = barrier.submit("cam0", 200.0, payload_of("b"), kJoin); });
        while (barrier.waiters() < 2) std::this_thread::yield();
        // The third instant does not fit, so opening it evicts the one open longest.
        std::thread three(
            [&] { third = barrier.submit("cam0", 300.0, payload_of("c"), kJoin); });
        one.join();
        clock.advance(1.0);
        two.join();
        three.join();

        check(first.reason == mtmc::kDroppedEvicted, "the parked frame was released, not hung");
        check(!first.associated, "with no answer, which is the honest gap");
        check(second.reason == mtmc::kClosedWindow && third.reason == mtmc::kClosedWindow,
              "and the two that still fit closed on their own windows");
    }

    // ------------------------------------------------------------------------- never starves

    void at_most_workers_minus_one_ever_wait() {
        for (const int workers : {1, 2, 4, 8}) {
            // Half a second, so a parked waiter releases itself on its own window and this
            // test never has to race a `close_all` against a thread that has not submitted
            // yet -- which is how the starve count picked up a `shutdown` and went flaky.
            InstantBarrier barrier(options(0.5, workers));
            barrier.camera_added("cam-absent");
            for (int camera = 0; camera < 16; ++camera) {
                barrier.camera_added("cam" + std::to_string(camera));
            }
            std::atomic<int> starved{0};
            std::vector<std::thread> threads;
            std::atomic<int> peak{0};
            for (int camera = 0; camera < 16; ++camera) {
                threads.emplace_back([&, camera] {
                    const InstantOutcome outcome = barrier.submit(
                        "cam" + std::to_string(camera), 100.0, payload_of("x"), kJoin);
                    if (outcome.reason == mtmc::kMissedWouldStarve) ++starved;
                });
            }
            for (int probe = 0; probe < 20000; ++probe) {
                const int seen = barrier.waiters();
                if (seen > peak.load()) peak.store(seen);
                std::this_thread::yield();
            }
            for (std::thread& thread : threads) thread.join();

            check(peak.load() <= workers - 1, "at most workers-1 parked at " +
                                                  std::to_string(workers) + " workers, saw " +
                                                  std::to_string(peak.load()));
            check(workers > 1 || starved.load() == 16,
                  "a single-worker runner never waits at all, saw " +
                      std::to_string(starved.load()) + " starved of 16");
        }
    }

    void a_starved_frame_still_contributes_its_payload_to_the_instant() {
        // The guard fires AFTER the entry is in the bucket: dropping it instead would degrade
        // the instant for every camera that did wait.
        InstantBarrier barrier(options(60.0, 1));  // zero permits: nobody may wait
        barrier.camera_added("cam0");
        barrier.camera_added("cam1");

        const InstantOutcome starved = barrier.submit("cam0", 100.0, payload_of("a"), kJoin);
        const InstantOutcome closer = barrier.submit("cam1", 100.01, payload_of("b"), kJoin);

        check(starved.reason == mtmc::kMissedWouldStarve, "the first frame was not parked");
        check(!starved.associated, "so it carries a gap");
        check(answer_of(closer) == "cam0:a,cam1:b",
              "and its tracks still took part in the group's association");
    }

    // ----------------------------------------------------------------------- the live camera
    // set

    void a_barrier_nobody_announced_learns_its_group_from_traffic() {
        InstantBarrier barrier(options(0.06, 4));

        // No `camera_added` at all: the live set is what has been seen, so the first frame of
        // a one-camera group completes on its own evidence.
        const InstantOutcome first = barrier.submit("cam0", 100.0, payload_of("a"), kJoin);

        check(first.reason == mtmc::kClosedComplete, "a chain with no lifecycle wiring closes");
        check(barrier.live() == std::set<std::string>{"cam0"}, "on what it has seen");
    }

    void the_first_announcement_latches_the_hooks_on_for_good() {
        InstantBarrier barrier(options(0.06, 4));
        barrier.submit("cam-seen", 100.0, payload_of("a"), kJoin);
        barrier.camera_added("cam-announced");

        check(barrier.live() == std::set<std::string>{"cam-announced"},
              "once anything announces, ANNOUNCED is the live set and a merely-seen camera "
              "no longer holds instants open");
    }

    void a_removed_camera_no_longer_holds_an_instant_open() {
        InstantBarrier barrier(options(60.0, 4));
        barrier.camera_added("cam0");
        barrier.camera_added("cam-going");

        InstantOutcome parked;
        std::thread waiter(
            [&] { parked = barrier.submit("cam0", 100.0, payload_of("a"), kJoin); });
        while (barrier.waiters() == 0) std::this_thread::yield();
        barrier.drop_camera("cam-going");
        waiter.join();

        check(parked.reason == mtmc::kClosedComplete,
              "dropping the last missing camera completes the instant on evidence");
        check(parked.associated && answer_of(parked) == "cam0:a", "and the waiter ran it");
    }

    void dropping_a_camera_runs_no_association_on_the_lifecycle_thread() {
        // The lifecycle thread must return promptly and must not run a tracker, so it SEALS.
        InstantBarrier barrier(options(60.0, 4));
        barrier.camera_added("cam0");
        barrier.camera_added("cam-going");
        std::atomic<int> associations{0};
        const Association counting = [&](const std::vector<InstantEntry>& entries) -> Results {
            ++associations;
            return join(entries);
        };

        InstantOutcome parked;
        std::thread waiter(
            [&] { parked = barrier.submit("cam0", 100.0, payload_of("a"), counting); });
        while (barrier.waiters() == 0) std::this_thread::yield();
        barrier.drop_camera("cam-going");
        waiter.join();

        check(associations.load() == 1, "exactly one association ran");
        check(parked.associated, "and the WAITER is the thread that ran it");
    }

    // ------------------------------------------------------------------ retirement and
    // shutdown

    void dropping_a_camera_with_nobody_waiting_leaves_the_bucket_alone() {
        // doc: long what actually distinguishes the two behaviours, and what does not
        // The Python test this port had missed (`test_barrier.py`'s own name). Its two
        // assertions -- one open instant, no `complete` counted -- hold under EITHER
        // behaviour, because sealing neither removes the bucket from the map nor counts an
        // event; I wrote them first and a mutation walked straight through them. What
        // separates the two is what `retire` does with the bucket LATER: a sealed one keeps
        // its sealing reason, so an instant whose association never ran is counted
        // `complete`, which `barrier.h` documents as "the association ran over the whole
        // group". Guarded, it is counted `expired`, which is what happened to it.
        Clock clock;
        InstantBarrier barrier(options(0.06, /*workers=*/1, /*max_instants=*/4), nullptr,
                               clock.fn());
        barrier.camera_added("cam-a");
        barrier.camera_added("cam-b");
        const InstantOutcome starved = barrier.submit("cam-a", 10.0, payload_of("p"), kJoin);
        check(starved.reason == mtmc::kMissedWouldStarve, "nobody is waiting on that instant");

        // Dropping cam-b leaves the bucket COMPLETE on the live set, which is exactly when
        // the unguarded version seals it.
        barrier.drop_camera("cam-b");
        check(barrier.open_instants() == 1, "still open either way -- sealing does not remove");

        clock.advance(1.0);
        // From a camera the live set does NOT hold, so this submit retires the old bucket
        // without completing an instant of its own -- otherwise it counts the `complete` the
        // assertion below is looking for and the test measures itself.
        barrier.submit("cam-x", 500.0, payload_of("q"), kJoin);

        const auto stats = barrier.instant_stats();
        check(stats.count(mtmc::kClosedComplete) == 0,
              "nothing counted `complete` for an association that never ran");
        const auto expired = stats.find(mtmc::kDroppedExpired);
        check(expired != stats.end() && expired->second == 1,
              "the abandoned instant is counted as what it was: expired");
    }

    void a_sealed_bucket_would_stop_accepting_frames_that_could_still_join() {
        // The half that is not a metric. `match` skips a sealed bucket, so a frame that could
        // still have joined opens a fresh instant instead -- and on a shard where the
        // never-starve guard is active that frame comes back unanswered. Reachable because
        // `live_` is the ANNOUNCED set once anything announces, while `match` consults no such
        // thing: a camera whose traffic beats its announcement can still join.
        Clock clock;
        InstantBarrier barrier(options(0.06, /*workers=*/1, /*max_instants=*/4), nullptr,
                               clock.fn());
        barrier.camera_added("cam-a");
        barrier.camera_added("cam-b");
        const InstantOutcome first = barrier.submit("cam-a", 10.00, payload_of("a"), kJoin);
        check(first.reason == mtmc::kMissedWouldStarve, "nobody waiting");

        barrier.drop_camera("cam-b");
        // An unannounced camera's frame, inside the same window.
        const InstantOutcome joiner = barrier.submit("cam-late", 10.01, payload_of("l"), kJoin);

        check(joiner.instant == first.instant,
              "it joined the instant the drop left open rather than opening a second one");
        // AND THE ASSOCIATION RAN OVER BOTH, which is what joining is for: sealed, this frame
        // would have opened instant 2, found cam-a unreported there, and starved with no
        // association at all.
        check(joiner.reason == mtmc::kClosedComplete, "and completed it");
        check(answer_of(joiner) == "cam-a:a,cam-late:l", "over both cameras' entries");
    }

    void an_over_released_budget_is_counted_rather_than_fatal() {
        // The one caller on the destructor path cannot receive an exception: a `noexcept`
        // destructor that threw would `std::terminate` the shard with nothing in the logs.
        WaiterBudget budget(1);
        check(budget.acquire(), "a permit is available");
        budget.release();

        check(!budget.release_held(), "an over-release answers false rather than throwing");
        check(budget.over_released() == 1, "and is counted, so the guard can be distrusted");
        check(budget.held() == 0, "with the count saturated rather than negative");

        bool typed = false;
        try {
            budget.release();
        } catch (const ServerStateError&) {
            typed = true;
        }
        check(typed, "while the public release keeps its typed refusal");
    }

    void a_bucket_past_its_deadline_with_no_waiters_is_discarded() {
        Clock clock;
        InstantBarrier barrier(options(0.06, 1), nullptr, clock.fn());
        barrier.camera_added("cam0");
        barrier.camera_added("cam-absent");
        barrier.submit("cam0", 100.0, payload_of("a"), kJoin);  // starved, nobody waiting
        check(barrier.open_instants() == 1, "the instant is open");

        clock.advance(0.07);
        barrier.submit("cam0", 500.0, payload_of("b"), kJoin);  // any submit retires

        check(barrier.instant_stats()[mtmc::kDroppedExpired] == 1,
              "an abandoned instant is discarded rather than associated for nobody");
    }

    void a_sealed_bucket_nobody_waited_on_keeps_the_reason_it_was_sealed_with() {
        // "A camera moved on" is what happened to that instant, and `advanced` is the share
        // an operator reads before touching sync_window_ms -- counting it as expired would
        // read that share as zero in exactly the starved shard where it matters.
        Clock clock;
        InstantBarrier barrier(options(0.06, 1), nullptr, clock.fn());
        barrier.camera_added("cam0");
        barrier.camera_added("cam-absent");
        barrier.submit("cam0", 100.00, payload_of("a"), kJoin);
        barrier.submit("cam0", 100.02, payload_of("b"), kJoin);  // seals the first, advanced

        clock.advance(0.07);
        barrier.submit("cam0", 500.0, payload_of("c"), kJoin);

        check(barrier.instant_stats()[mtmc::kClosedAdvanced] >= 1,
              "the sealed instant is counted under `advanced`, not `expired`");
    }

    void a_stray_release_under_a_parked_waiter_does_not_kill_the_shard() {
        // The scenario the budget's public-ness makes reachable: it is process-wide BY DESIGN
        // -- that is its reason to exist -- so one stray `release()` from anywhere drives
        // `held_` to zero while a worker is parked in `submit`. That worker's scope guard then
        // releases an empty budget, and a `noexcept` destructor that threw would take down the
        // shard with no `ServerStateError` anywhere in the logs to say why.
        InstantBarrier barrier(options(0.5, 8));
        barrier.camera_added("cam0");
        barrier.camera_added("cam-absent");
        InstantOutcome parked;
        std::thread waiter(
            [&] { parked = barrier.submit("cam0", 100.0, payload_of("a"), kJoin); });
        while (barrier.waiters() == 0) std::this_thread::yield();

        barrier.budget().release();  // the stray one: held_ was 1, now 0
        waiter.join();               // the guard's destructor runs on an empty budget

        check(parked.reason == mtmc::kClosedWindow,
              "the parked worker returned on its own window rather than aborting");
        check(barrier.budget().over_released() == 1,
              "and the inconsistency is counted, so the guard can be distrusted");
    }

    void a_shutdown_releases_the_parked_workers_at_once() {
        InstantBarrier barrier(options(60.0, 8));
        barrier.camera_added("cam0");
        barrier.camera_added("cam1");
        barrier.camera_added("cam-absent");

        InstantOutcome one;
        InstantOutcome two;
        std::thread first([&] { one = barrier.submit("cam0", 100.0, payload_of("a"), kJoin); });
        std::thread second(
            [&] { two = barrier.submit("cam1", 100.01, payload_of("b"), kJoin); });
        while (barrier.waiters() < 2) std::this_thread::yield();
        const size_t resolved = barrier.close_all();
        first.join();
        second.join();

        check(resolved == 1, "one open instant was resolved");
        check(one.reason == mtmc::kDroppedShutdown && two.reason == mtmc::kDroppedShutdown,
              "both parked workers were released at once");
        check(!one.associated && !two.associated, "with an honest gap rather than an answer");
    }

    void a_submit_after_close_all_is_refused_rather_than_parked() {
        InstantBarrier barrier(options(60.0, 8));
        barrier.close_all();
        barrier.close_all();  // idempotent

        const InstantOutcome outcome = barrier.submit("cam0", 100.0, payload_of("a"), kJoin);

        check(outcome.reason == mtmc::kDroppedShutdown, "refused, not parked");
        check(barrier.frame_stats()[mtmc::kDroppedShutdown] == 1,
              "and counted as a FRAME miss, which is why the two families are apart");
    }

    void a_failed_association_releases_the_waiters_and_the_closer_gets_the_exception() {
        InstantBarrier barrier(options(60.0, 8));
        barrier.camera_added("cam0");
        barrier.camera_added("cam1");
        const Association throwing = [](const std::vector<InstantEntry>&) -> Results {
            throw InferenceError("the tracker refused this instant");
        };

        InstantOutcome parked;
        std::thread waiter(
            [&] { parked = barrier.submit("cam0", 100.0, payload_of("a"), throwing); });
        while (barrier.waiters() == 0) std::this_thread::yield();
        bool threw = false;
        try {
            barrier.submit("cam1", 100.01, payload_of("b"), throwing);
        } catch (const InferenceError&) {
            threw = true;
        }
        waiter.join();

        check(threw, "the closing thread gets the exception -- one frame carries the failure");
        check(parked.reason == mtmc::kDroppedFailed, "and the rest carry a gap");
        check(!parked.associated, "with no result");
        check(barrier.instant_stats()[mtmc::kDroppedFailed] == 1, "counted once, per instant");
    }

    void one_event_per_instant_and_not_one_per_frame() {
        std::vector<std::string> events;
        InstantBarrier barrier(options(0.06, 4), nullptr, {},
                               [&](const std::string& reason) { events.push_back(reason); });
        barrier.camera_added("cam0");
        barrier.camera_added("cam1");

        InstantOutcome parked;
        std::thread waiter(
            [&] { parked = barrier.submit("cam0", 100.0, payload_of("a"), kJoin); });
        while (barrier.waiters() == 0) std::this_thread::yield();
        barrier.submit("cam1", 100.01, payload_of("b"), kJoin);
        waiter.join();

        check(events.size() == 1, "two frames, one instant, ONE event");
        check(!events.empty() && events[0] == mtmc::kClosedComplete, "naming how it closed");
    }

    void every_frame_of_a_group_gets_the_same_answer_or_an_honest_gap() {
        // The property the whole file exists for, under real threads: a frame either takes
        // part in its group's association and reads the group's answer, or it is told why not.
        const int cameras = 6;
        // Frozen, for the reason above: what is under test is that one instant produces one
        // answer, not that six thread starts fit inside a window.
        Clock clock;
        InstantBarrier barrier(options(0.06, cameras + 1), nullptr, clock.fn());
        for (int camera = 0; camera < cameras; ++camera) {
            barrier.camera_added("cam" + std::to_string(camera));
        }
        std::vector<InstantOutcome> outcomes(cameras);
        std::vector<std::thread> threads;
        for (int camera = 0; camera < cameras; ++camera) {
            threads.emplace_back([&, camera] {
                outcomes[camera] =
                    barrier.submit("cam" + std::to_string(camera), 100.0 + 0.001 * camera,
                                   payload_of(std::to_string(camera)), kJoin);
            });
        }
        for (std::thread& thread : threads) thread.join();

        std::string agreed;
        int answered = 0;
        bool disagreed = false;
        for (const InstantOutcome& outcome : outcomes) {
            if (!outcome.associated) {
                // Every gap is named, never silent.
                const bool named = outcome.reason == mtmc::kMissedWouldStarve ||
                                   outcome.reason == mtmc::kMissedLate ||
                                   outcome.reason == mtmc::kMissedDuplicate ||
                                   outcome.reason == mtmc::kDroppedEvicted ||
                                   outcome.reason == mtmc::kDroppedExpired;
                check(named,
                      "an unanswered frame carries a reason, got '" + outcome.reason + "'");
                continue;
            }
            ++answered;
            if (agreed.empty()) {
                agreed = answer_of(outcome);
            } else if (answer_of(outcome) != agreed) {
                disagreed = true;
            }
        }

        check(answered >= 1, "at least one frame got the group's answer");
        check(!disagreed, "and every answered frame of one instant read the SAME answer");
    }

}  // namespace

int main() {
    a_zero_window_is_refused();
    a_barrier_with_no_room_for_an_instant_is_refused();
    a_negative_budget_is_refused_and_zero_is_not();
    over_releasing_a_budget_is_a_typed_refusal();
    a_group_of_one_closes_on_its_own_frame();
    two_captures_inside_one_window_are_one_instant();
    two_captures_more_than_a_window_apart_are_two_instants();
    a_camera_never_collides_with_itself_at_any_window();
    a_second_frame_from_a_camera_in_the_bucket_closes_it();
    a_repeat_of_a_capture_already_in_the_bucket_is_a_duplicate();
    another_camera_running_ahead_does_not_make_the_next_frame_a_duplicate();
    the_answer_is_handed_back_whole_and_never_indexed();
    an_empty_answer_is_not_a_missed_frame();
    a_missing_camera_costs_one_window_and_the_association_still_runs();
    an_instant_does_not_expire_until_the_injected_clock_says_so();
    a_frame_for_a_closed_instant_is_told_it_is_late();
    a_late_frame_does_not_re_open_its_instant();
    the_oldest_instant_is_evicted_and_counted();
    an_evicted_instant_releases_the_frames_waiting_on_it();
    at_most_workers_minus_one_ever_wait();
    a_starved_frame_still_contributes_its_payload_to_the_instant();
    a_barrier_nobody_announced_learns_its_group_from_traffic();
    the_first_announcement_latches_the_hooks_on_for_good();
    a_removed_camera_no_longer_holds_an_instant_open();
    dropping_a_camera_runs_no_association_on_the_lifecycle_thread();
    dropping_a_camera_with_nobody_waiting_leaves_the_bucket_alone();
    a_sealed_bucket_would_stop_accepting_frames_that_could_still_join();
    an_over_released_budget_is_counted_rather_than_fatal();
    a_bucket_past_its_deadline_with_no_waiters_is_discarded();
    a_sealed_bucket_nobody_waited_on_keeps_the_reason_it_was_sealed_with();
    a_stray_release_under_a_parked_waiter_does_not_kill_the_shard();
    a_shutdown_releases_the_parked_workers_at_once();
    a_submit_after_close_all_is_refused_rather_than_parked();
    a_failed_association_releases_the_waiters_and_the_closer_gets_the_exception();
    one_event_per_instant_and_not_one_per_frame();
    every_frame_of_a_group_gets_the_same_answer_or_an_honest_gap();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
