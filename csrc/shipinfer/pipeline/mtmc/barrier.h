// The instant barrier: per-camera frames, turned back into synchronised instants.
//
// The C++ twin of `topology/barrier.py`, and pure for the same reason: it buckets OPAQUE
// payloads by capture instant, decides when a bucket is complete, picks the thread that runs
// the association and scatters the answer back -- so every property worth testing about
// cross-camera synchronisation is testable with strings and a callback, with no tracker and no
// external lane anywhere near it. The mtmc stage gives the payloads meaning.
//
// doc: long the three decisions this file exists to hold, each of them measured
// WHY THE INSTANT IS ANCHORED AND NOT A GRID. `floor(capture_s / window)` is wrong at every
// setting. A window WIDER than one frame period puts two consecutive frames of one camera in
// one bucket once every `window / period` frames -- one in six at 20 fps and 60 ms, each
// refused an answer, with genlocked cameras and no jitter. A window NARROWER than the period
// fixes that and breaks the other half: free-running cameras spread captures across a whole
// period, so no absolute cell holds the group. The two constraints are incompatible, which is
// the proof that no absolute window is correct. So an instant is anchored by its FIRST
// ARRIVAL, and a camera reporting a second, later frame is the signal that the instant it was
// in has all the evidence it will get.
//
// WHY THE BARRIER MUST NEVER BLOCK THE LAST WORKER. The walk is synchronous: a worker waiting
// inside a stage is a worker not draining its lane. If every worker parks waiting for cameras
// whose frames are still QUEUED, no bucket can complete on evidence and the deployment has
// converted itself into a fixed window of latency per frame with a stalled queue behind it. A
// `WaiterBudget` caps the waiters and the frame that would take the last permit is emitted
// immediately with an honest gap.
//
// WHY THE RESULT IS OPAQUE AND NEVER INDEXED. A group's association answers for every camera
// in one flattened list, and camera A's three results and camera B's one come back as four
// whose positions mean nothing to either frame. The barrier publishes whatever the association
// returned and never looks inside it -- each caller reads its own entries out by a key it
// chose. Scattering by list position is the classic reassembly bug (ADR-002's tag rule, one
// layer up), and it produces a plausible answer rather than an error.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "shipinfer/core/types.h"

namespace shipinfer::mtmc {

    //: Every camera of the live set reported. The association ran over the whole group.
    inline constexpr const char* kClosedComplete = "complete";
    //: The window ran out with a camera still missing. The association still ran over whoever
    //: did report -- a partial instant is worth more than no instant, and MTMC over a subset
    //: of a group is exactly what a camera outage looks like.
    inline constexpr const char* kClosedWindow = "window";
    //: A camera already in the bucket reported its NEXT frame, so no further evidence for this
    //: instant can arrive from it. A rising share is the number to read before touching
    //: `sync_window_ms`: one camera is running ahead of its group by less than a window.
    inline constexpr const char* kClosedAdvanced = "advanced";
    //: Pushed out by `max_instants` newer ones -- the group's clocks disagree by more than the
    //: window, or one camera is far ahead.
    inline constexpr const char* kDroppedEvicted = "evicted";
    //: Past its deadline with nobody waiting: every frame in it had already been emitted by
    //: the never-starve guard, so there was no answer for anyone to receive.
    inline constexpr const char* kDroppedExpired = "expired";
    //: `close_all` resolved it because the stage is closing. Also the FRAME-level reason a
    //: submit after that gets, which is why the two families are counted apart.
    inline constexpr const char* kDroppedShutdown = "shutdown";
    //: The association itself threw. Waiters are released with no result and the closing
    //: thread gets the exception -- one frame carries the failure, the rest carry a gap.
    inline constexpr const char* kDroppedFailed = "failed";

    //: The frame's instant had already closed when it arrived. Counted, never retro-fitted:
    //: re-running an association to add one late camera would issue a second, contradictory
    //: set of results for objects already published under the first.
    inline constexpr const char* kMissedLate = "late";
    //: One camera offered the same capture instant twice. A cross-camera cluster refuses that
    //: (they are two instants), so the second is emitted with a gap rather than allowed to
    //: make the group's same-camera exclusion mask leak. A LATER capture from a camera already
    //: in the bucket is not this -- it is the next instant, and it closes the open one.
    inline constexpr const char* kMissedDuplicate = "duplicate";
    //: This camera's CAPTURE clock went backwards by more than an instant is wide. Measured
    //: against that camera's own newest stamp, so a camera whose clock merely sits behind the
    //: rest of the group never trips it -- what this catches is a STEP, which NTP can deliver
    //: at any moment and which would otherwise open an instant in the past that no other
    //: camera will ever join. Counted and refused rather than mis-bucketed silently
    //: (`MTMC-INSTANTS-NEED-A-SHARED-MONOTONIC-CLOCK`). ONE FRAME PER STEP -- see
    //: `kBackwardRefusalsBeforeAdopting`.
    inline constexpr const char* kMissedBackward = "backward";
    // doc: long why a step is adopted rather than refused for its whole length
    //: How many frames a camera is refused before its offered stamp BECOMES the reference.
    //:
    //: A step is a new, persistent time base and not a transient anomaly. Every camera on a
    //: shard shares one `CLOCK_REALTIME` (`ingest/frame/tag.py` stamps `time.time_ns()` at
    //: decode), so an NTP step moves the whole fleet at once: a reference that could not be
    //: adopted would refuse EVERY frame of EVERY camera for the length of the step -- a total
    //: outage in place of the single re-anchoring an anchored instant already absorbs. One
    //: refusal is what the counter needs to make the step visible, and it is what stops one
    //: stray past stamp from opening an instant nobody joins. The same rule recovers a camera
    //: whose stamp jumped into the FUTURE, which is a camera reboot on the DeepStream path
    //: where the stamp is the camera's own RTCP clock rather than this box's.
    inline constexpr int kBackwardRefusalsBeforeAdopting = 1;
    //: Waiting would have parked the last worker. The frame's payload is already in its bucket
    //: when the guard fires, so its tracks still take part in the association -- only the
    //: ANSWER is not delivered to this frame. Dropping the entry instead would degrade the
    //: instant for every camera that did wait.
    inline constexpr const char* kMissedWouldStarve = "would_starve";

    //: How wide an instant is, in milliseconds. A PROPOSAL, not a measurement: with the
    //: anchored instant it is no longer constrained from below by the frame period, it has to
    //: be at least the group's arrival spread, and 60 ms is a comfortable margin over the
    //: ~1 ms genlock skew of a wired group. `topology/barrier.py`'s own default.
    inline constexpr double kDefaultSyncWindowMs = 60.0;
    // doc: long why the bound is a floor and the fleet is the number, with the measurement
    //: The FLOOR on how many instants may be open before the oldest is evicted. Eight is half
    //: a second at the default window: enough to absorb a camera a few frames behind, far too
    //: few to hide a clock that is minutes out -- and, on its own, far too few for a fleet.
    //: Every camera holds an instant of its own open -- the one its last frame landed in --
    //: plus whatever it has SEALED by moving on and the window has not yet retired. So the
    //: number legitimately open scales with the FLEET, and a bound below that evicts buckets
    //: the group is still filling rather than the stale clock eviction exists for. Hence the
    //: floor and the cameras this barrier has seen or been told about, unless the chain names
    //: a number. MEASURED 11 Sep at the design load (50 cameras x 20 fps,
    //: `benchmarks/RESULTS.md`): at 8 the run evicted 20-28% of its instants and admitted
    //: 97-545 observations; from 16 up it evicts NOTHING and admits ~2 200. The fleet is 50
    //: there, three times the knee.
    inline constexpr int kDefaultMaxInstants = 8;

    //: How many arrival-lag samples one barrier keeps -- a RING, so a long run reports its
    //: recent tail rather than freezing on the first few minutes. 200k is ~800 KB and about
    //: three minutes of a 50-camera fleet; past it the oldest sample is overwritten and the
    //: count of overwrites is kept, so a saturated barrier does not read like a quiet one.
    inline constexpr size_t kMaxLagSamples = 200000;

    // One camera's contribution to an instant: who, and whatever the caller put in.
    //
    // `payload` is `shared_ptr<void>` on purpose. The barrier is pure and must not learn what
    // a camera's tracks are -- that is the stage's vocabulary, and keeping it out of here is
    // what lets the synchronisation properties be tested with strings.
    struct InstantEntry {
        std::string camera_id;
        std::shared_ptr<void> payload;
    };

    // Whatever the association produced, unexamined. The caller downcasts it.
    using Results = std::shared_ptr<const void>;
    using Association = std::function<Results(const std::vector<InstantEntry>&)>;

    // What a submitted frame got back.
    //
    // `associated` is a flag and not `results != nullptr`, because an instant in which nothing
    // was visible on any camera legitimately returns an EMPTY answer, and that is a different
    // fact from "this frame missed its instant" (ADR-005: never an empty result to mean
    // failure). `instant` is the barrier's monotonic id, which is also its eviction order; 0
    // for a frame that never reached a bucket.
    struct InstantOutcome {
        std::string reason;
        Results results;
        bool associated = false;
        int64_t instant = 0;
    };

    // How many workers may be parked inside a barrier at once, PER PROCESS.
    //
    // doc: long why the budget is process-wide, with the measurement that says so
    // The never-starve guard counts permits here rather than waiters in one barrier, and that
    // is the whole reason this class exists. A chain may hold two `mtmc` slots -- two camera
    // groups is a supported configuration -- and with a per-barrier count, barrier A admits
    // `workers - 1` waiters while barrier B, which sees zero, admits the last one. Every worker
    // is then parked and neither barrier can close on evidence. One budget for the process,
    // built by the runner with `workers - 1` permits, makes the invariant hold for any number
    // of them -- and makes their coverage INTERFERE: permits are handed out first-come, so a
    // shard running two barriers needs the SUM of its groups' sizes in workers rather than the
    // larger. Measured on the Python twin, two 8-camera groups: 100% coverage each at 16
    // workers, 73%/52% at 9.
    class WaiterBudget {
      public:
        // `permits` 0 is legitimate and means "never wait" -- a single-worker runner, or one
        // that did not say how many workers it has.
        explicit WaiterBudget(int permits);

        int permits() const { return permits_; }
        int held() const;

        // Take a permit if there is one. NEVER BLOCKS: a barrier calls this holding its own
        // condition lock, so blocking here would be a lock-ordering hazard between two
        // barriers. Returns whether the caller may wait.
        bool acquire();

        // Give one back. Throws `ServerStateError` on more releases than acquires: a silent
        // decrement past zero would hand out permits that do not exist.
        void release();

        // The same, for a caller that CANNOT receive an exception -- `submit`'s scope guard is
        // a destructor, and a `noexcept` one that threw would `std::terminate` the shard with
        // nothing in the logs. Returns whether a permit was actually held; an over-release is
        // saturated at zero and counted (`over_released`) rather than thrown or ignored.
        bool release_held() noexcept;

        //: How many releases arrived with no permit held. Nonzero means the never-starve
        //: guard's count is no longer trustworthy, which is worth reporting rather than
        //: aborting for.
        uint64_t over_released() const;

      private:
        int permits_;
        mutable std::mutex lock_;
        int held_ = 0;
        uint64_t over_released_ = 0;
    };

    // NAMESPACE SCOPE and not nested, because a nested struct's default member initialisers
    // are not available while the enclosing class is still being defined -- so `Options
    // options = {}` on the constructor below would not compile.
    struct BarrierOptions {
        double sync_window_s = kDefaultSyncWindowMs / 1000.0;
        //: The runner's worker count. The budget gets `workers - 1` permits, so a
        //: single-worker runner never waits.
        int workers = 1;
        //: UNSET follows the fleet: `max(kDefaultMaxInstants, live cameras)`, recomputed as
        //: cameras arrive. A number here is EXACT and overrides that, both ways -- which is
        //: what makes eviction testable at all, and what lets an operator who has measured
        //: their own spread say so.
        std::optional<int> max_instants;
    };

    class InstantBarrier {
      public:
        using Clock = std::function<double()>;
        using OnEvent = std::function<void(const std::string&)>;
        using Options = BarrierOptions;

        // `budget` is shared so two barriers in one process share one never-starve guard;
        // passing none builds a private one from `options.workers`. `clock` is injected so the
        // window is testable without sleeping.
        explicit InstantBarrier(Options options = {},
                                std::shared_ptr<WaiterBudget> budget = nullptr,
                                Clock clock = {}, OnEvent on_event = {});

        double window_s() const { return window_s_; }
        int workers() const { return workers_; }
        WaiterBudget& budget() const { return *budget_; }
        int waiters() const;
        std::set<std::string> live() const;
        size_t open_instants() const;
        //: The bound in force now -- the chain's number, or the live set against the floor.
        int max_instants() const;
        std::map<std::string, uint64_t> instant_stats() const;
        //: How much of the fleet an instant actually held when it ended: the total cameras
        //: over every instant that left the map, how many instants that was, and the largest
        //: one. The number to read before touching `sync_window_ms`: a group whose instants
        //: hold two cameras of fifty is not synchronised, whatever its reasons say, and no
        //: cross-camera association can form in an instant that holds one.
        struct InstantSizes {
            uint64_t cameras = 0;
            uint64_t instants = 0;
            uint64_t largest = 0;
        };
        InstantSizes instant_sizes() const;
        std::map<std::string, uint64_t> frame_stats() const;

        // doc: long why the lag is handed in rather than taken, and what it answers
        //: HOW LATE A FRAME REACHES THIS BARRIER: microseconds between the stamp a frame was
        //: CAPTURED at and the moment it was submitted. `late` says a frame missed its instant
        //: and `window` says an instant ran out of time; neither says by how much, so neither
        //: can tell a window that is too narrow from a chain that is too slow to reach it.
        //:
        //: HANDED IN, not measured here, and that is the whole reason this is two methods. The
        //: capture stamp is a WALL time and this barrier's own clock is deliberately steady
        //: (`clock_` defaults to `steady_seconds`, and ADR says why instants key on the wall
        //: stamp while deadlines do not), so subtracting one from the other here would be
        //: arithmetic across two clocks. The mtmc stage holds both and is the only place that
        //: legitimately can.
        //:
        //: A RING, because a 24/7 server is not a benchmark and the OLDEST samples are the
        //: least useful: past `kMaxLagSamples` the next sample overwrites the oldest, so a
        //: ten-minute run reports its recent tail instead of its warm-up. Head truncation was
        //: the first version and it froze the distribution at whatever the first three minutes
        //: held -- while the frame percentiles printed beside it covered the whole run, so the
        //: two numbers a reader compares would have spanned different windows.
        //: `negative` says this sample was clamped: the frame arrived before its capture
        //: stamp, so what it measures is the two clocks disagreeing and not a fast chain.
        void note_arrival_lag_us(uint32_t lag_us, bool negative = false,
                                 bool saturated = false);
        std::vector<uint32_t> arrival_lag_us() const;
        //: How many samples have been overwritten. Non-zero says the ring wrapped, so the
        //: percentiles above describe the last `kMaxLagSamples` frames and not the run.
        uint64_t lag_samples_overwritten() const;
        //: And how many arrived BEFORE they were captured, clamped to zero by the caller.
        //: Non-zero says the server's wall clock and the source's disagree -- an NTP step or
        //: a source stamping ahead -- and that the percentiles are pulled toward zero by it.
        //: Counted rather than inferred: `backward` cannot see this. It compares a camera's
        //: capture stamp against that camera's OWN history, both in the capture domain, so a
        //: server clock that steps leaves every camera monotonic among itself and `backward`
        //: at zero while every lag clamps to 0 -- a barrier every frame appears to reach
        //: instantly, which is the inversion this whole measurement exists to prevent.
        uint64_t lag_samples_negative() const;

        //: How many samples hit the `uint32_t` ceiling (~71.6 minutes) and were clamped down
        //: to it. The other plane clamps at the same value with the same counter -- without a
        //: ceiling there it turned one input into two different histograms. Non-zero at all is
        //: a stalled source rather than a latency figure, and flattening it silently would
        //: pull the p99 toward a number no deployment ever produced.
        uint64_t lag_samples_saturated() const;

        //: Declared cameras that have never sent a frame. EMPTY is the healthy answer, and a
        //: non-empty one is a configuration fault that is otherwise silent: announced cameras
        //: win over seen ones, so a roster naming cameras this fleet does not have makes
        //: `complete` unreachable and every instant closes on the window or on a newer one.
        //: Measured 11 Sep: `topology/ship_person_cpu.yaml` declares `cam-01 ... cam-04` and
        //: every bench fleet is `cam00 ... cam11`, so not ONE instant closed complete in six
        //: runs -- 421 of 905 did once the roster named the run's own cameras.
        std::set<std::string> silent_cameras() const;

        // The lifecycle half: which cameras the group is waiting for. Announced cameras win
        // over merely-seen ones the moment anything announces, the way the Python element's
        // `camera_added` hook does.
        void camera_added(const std::string& camera_id);
        void drop_camera(const std::string& camera_id);

        // Put one camera's frame into its instant and come back with the group's answer.
        //
        // `capture_s` is when the frame was CAPTURED, not when it arrived. `associate` is
        // called by whichever thread closes the bucket, with every entry in it, under this
        // barrier's lock; if it throws, every waiter is released with `kDroppedFailed` and the
        // exception reaches the closing thread.
        InstantOutcome submit(const std::string& camera_id, double capture_s,
                              std::shared_ptr<void> payload, const Association& associate);

        // Resolve every open instant and refuse every later submit. Idempotent. Without it a
        // worker parked in `submit` would hold shutdown for the rest of its window.
        size_t close_all(const std::string& reason = kDroppedShutdown);

      private:
        struct Bucket {
            int64_t instant = 0;
            double deadline = 0.0;
            //: The CAPTURE span of what has joined so far, which is what makes the instant
            //: anchored rather than gridded.
            double first = 0.0;
            double last = 0.0;
            std::vector<InstantEntry> entries;
            //: Camera -> the capture THAT CAMERA put in. A map and not a set because the
            //: duplicate test is per camera: `last` is the maximum over the group, so testing
            //: a repeat against it would refuse a camera's genuinely next frame whenever
            //: another camera has already pushed the span past it.
            std::map<std::string, double> reported;
            int waiters = 0;
            //: Set by the thread that resolved this bucket. A waiter wakes, sees it, returns.
            bool done = false;
            //: "Somebody sealed this bucket but had no association function in hand." A
            //: sealed bucket accepts no further entries and a waiter -- which is a worker
            //: holding the callback -- does the work.
            bool ready = false;
            std::string ready_reason = kClosedComplete;
            std::string reason;
            Results results;
            bool associated = false;
        };

        void refresh_live();
        std::shared_ptr<Bucket> match(double capture_s);
        //: The resolved instant this capture belongs inside, or 0. Tested against the closed
        //: instant's ACTUAL span, not the window it was allowed: a genlocked group closes
        //: instants whose span is a single point, and testing against the full window would
        //: call every subsequent frame of that camera late -- the absolute grid's failure.
        int64_t late_instant(double capture_s) const;
        std::shared_ptr<Bucket> open(double capture_s, double now);
        void seal(Bucket& bucket, const std::string& reason);
        InstantOutcome missed(const std::string& reason, int64_t instant);
        void event(const std::string& reason);
        void remember(const Bucket& bucket);
        InstantOutcome close(const std::shared_ptr<Bucket>& bucket, const std::string& reason,
                             const Association& associate);
        void retire(double now);
        void evict();

        double window_s_;
        int workers_;
        //: The chain's, if it named one. Empty means `refresh_live` owns `max_instants_`.
        std::optional<int> configured_max_instants_;
        int max_instants_;
        std::shared_ptr<WaiterBudget> budget_;
        Clock clock_;
        //: The arrival lag samples, and how many did not fit. Guarded by `lock_` like every
        //: other counter here.
        std::vector<uint32_t> arrival_lag_us_;
        //: Where the next sample goes once the ring is full, and how often it has wrapped.
        size_t lag_next_ = 0;
        uint64_t lag_overwritten_ = 0;
        uint64_t lag_negative_ = 0;
        uint64_t lag_saturated_ = 0;
        OnEvent on_event_;

        mutable std::mutex lock_;
        std::condition_variable cond_;
        // doc: long why the buckets are shared_ptr and not unique_ptr
        //: Keyed by instant id, which `std::map` keeps in ascending order -- so the FRONT
        //: bucket has the earliest deadline of all (every deadline is one window after its
        //: bucket opened), which is what lets `retire` answer in O(1) when nothing is stale.
        //:
        //: SHARED and not unique, which is the one place this port cannot follow the Python
        //: line for line. A bucket leaves the map before its association runs, and again when
        //: it is evicted or shut down -- and in every one of those cases a waiter may still be
        //: asleep holding it. Python's refcount keeps such a bucket alive until the last waiter
        //: has read its answer; a `unique_ptr` erased from the map would free it under them.
        std::map<int64_t, std::shared_ptr<Bucket>> buckets_;
        int64_t next_instant_ = 0;
        //: Resolved instants' capture spans, so a frame inside one is late rather than new.
        std::map<int64_t, std::pair<double, double>> recent_;
        size_t recent_limit_;
        //: `instant_sizes`, tallied in `remember` -- the one place every ended bucket passes.
        uint64_t cameras_held_ = 0;
        uint64_t instants_ended_ = 0;
        uint64_t cameras_held_max_ = 0;
        //: Each camera's newest capture stamp, and how many frames in a row it has refused
        //: against it. Both bounded by the fleet and erased with the camera.
        std::map<std::string, double> newest_capture_;
        std::map<std::string, int> backward_run_;
        std::set<std::string> announced_;
        std::set<std::string> seen_;
        bool hooked_ = false;
        std::set<std::string> live_;
        int waiters_ = 0;
        bool closed_ = false;
        std::map<std::string, uint64_t> instant_counts_;
        std::map<std::string, uint64_t> frame_counts_;
    };

}  // namespace shipinfer::mtmc
