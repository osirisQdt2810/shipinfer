// The ingest plane, tested the way `tests/ingest/` tests the Python one: with a source that
// has no camera, no network and no decoder.
//
// Every check here has a namesake on the Python side — `test_backoff.py`,
// `test_source_base.py`, `test_camera_actor.py`, `test_manager.py` — and asserts the same
// thing, so a divergence between the planes shows up as one green file and one red one. No
// device, no OpenCV: this binary is part of the offline C++ tier and **must not include
// `ingest/sources/replay.h`**, which is the one unit in this package that reaches the driver's
// headers (see the note at the top of `ingest/registry.cpp`).

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/core/redact.h"
#include "shipinfer/core/stop_signal.h"
#include "shipinfer/core/types.h"
#include "shipinfer/ingest/base.h"
#include "shipinfer/ingest/camera/actor.h"
#include "shipinfer/ingest/camera/health.h"
#include "shipinfer/ingest/config.h"
#include "shipinfer/ingest/frame.h"
#include "shipinfer/ingest/manager.h"
#include "shipinfer/ingest/registry.h"
#include "shipinfer/ingest/sink.h"
#include "shipinfer/ingest/timing/backoff.h"
#include "shipinfer/ingest/timing/pacing.h"

namespace {

    using namespace shipinfer;
    using namespace std::chrono_literals;

    int failures = 0;
    int checks = 0;
    int skips = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        }
    }

    // A test that cannot run must say so and be counted — a run that prints "N checks, 0
    // failure(s)" and reads as green is a test that fails open.
    void skip(const std::string& why) {
        ++skips;
        std::fprintf(stderr, "SKIP: %s\n", why.c_str());
    }

    using Clock = std::chrono::steady_clock;
    double ms_since(Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    // -- the scripted source ----------------------------------------------------------------

    // What one camera's decode path is told to do, call by call. Held by the test and shared
    // with every source the actor builds, so the counters survive the rebuild a reconnect does
    // — which is exactly the property under test in several of these.
    struct FakeScript {
        std::atomic<int> builds{0};
        std::atomic<int> opens{0};
        std::atomic<int> reads{0};
        std::atomic<int> closes{0};
        std::atomic<bool> exhausted{false};

        // Called with the 0-based call index. Throw to fail the call.
        std::function<void(int)> on_open;
        // 1 = a frame, 0 = "not yet". Throw to break the stream.
        std::function<int(int)> on_read;
        std::function<void(int)> on_close;

        int height = 4;
        int width = 6;
        double fps = 0.0;
        std::vector<uint8_t> pixels = std::vector<uint8_t>(4 * 6 * 3, 7);
    };

    class FakeSource : public FrameSource {
      public:
        FakeSource(const IngestConfig& config, FrameCounter& counter, StopSignal& stop,
                   FakeScript& script)
            : FrameSource(config, counter, stop), script_(script) {}

        bool is_exhausted() const override { return script_.exhausted.load(); }

      protected:
        void do_open() override {
            const int index = script_.opens.fetch_add(1);
            if (script_.on_open) script_.on_open(index);
            set_format(script_.height, script_.width, script_.fps);
        }

        std::optional<HostFrame> do_read() override {
            const int index = script_.reads.fetch_add(1);
            if (script_.on_read && script_.on_read(index) == 0) return std::nullopt;
            HostFrame image;
            image.pixels = script_.pixels.data();
            image.height = script_.height;
            image.width = script_.width;
            return image;
        }

        void do_close() override {
            const int index = script_.closes.fetch_add(1);
            if (script_.on_close) script_.on_close(index);
        }

      private:
        FakeScript& script_;
    };

    // The registered fake, for the registry checks. It shares one process-wide script because
    // the registry's factory signature carries nothing a test could hang a per-test script on —
    // which is the point: a source built by name is built the way production builds it.
    FakeScript& registry_script() {
        static FakeScript script;
        return script;
    }

    const SourceRegistrar kFake("fake", {"double", "scripted"},
                                "a scripted source for the offline tier",
                                [](const IngestConfig& config, FrameCounter& counter,
                                   StopSignal& stop) -> std::unique_ptr<FrameSource> {
                                    return std::make_unique<FakeSource>(config, counter, stop,
                                                                        registry_script());
                                });

    SourceFactory scripted(FakeScript& script) {
        return [&script](const IngestConfig& config, FrameCounter& counter,
                         StopSignal& stop) -> std::unique_ptr<FrameSource> {
            script.builds.fetch_add(1);
            return std::make_unique<FakeSource>(config, counter, stop, script);
        };
    }

    IngestConfig a_camera(const std::string& id) {
        IngestConfig config;
        config.camera_id = id;
        config.uri = "fake://" + id + "/stream";
        config.source = "fake";
        return config;
    }

    // -- the injectable wait ----------------------------------------------------------------

    // Records each delay the actor asked to wait out and then parks it there until the test
    // releases it. Lock-step, so a check reads the state the actor is *in* rather than one it
    // has already left — and the delay *sequence* is asserted directly instead of inferred from
    // a log, which is the whole reason the wait is injectable.
    class SteppingWait {
      public:
        bool operator()(double seconds) {
            std::unique_lock<std::mutex> lock(mutex_);
            delays_.push_back(seconds);
            arrived_.notify_all();
            released_.wait(lock, [this] { return open_ || permits_ > 0; });
            if (!open_) --permits_;
            return false;  // never "cut short by a stop": the tests decide when to release
        }

        // Block until at least `n` delays have been recorded.
        bool arrived(size_t n, std::chrono::milliseconds budget = 3000ms) {
            std::unique_lock<std::mutex> lock(mutex_);
            return arrived_.wait_for(lock, budget, [&] { return delays_.size() >= n; });
        }

        void step(size_t n = 1) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                permits_ += n;
            }
            released_.notify_all();
        }

        // Stop parking altogether — called before a stop, so the actor can reach its exit.
        void open() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                open_ = true;
            }
            released_.notify_all();
        }

        std::vector<double> delays() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return delays_;
        }

      private:
        mutable std::mutex mutex_;
        std::condition_variable arrived_;
        std::condition_variable released_;
        std::vector<double> delays_;
        size_t permits_ = 0;
        bool open_ = false;
    };

    // Signal first, then unpark, then join: releasing the gate before the signal lets the actor
    // spin through iterations the test did not ask for.
    void finish(CameraActor& actor, SteppingWait& gate) {
        actor.request_stop();
        gate.open();
        actor.stop(2000ms);
    }

    // A one-shot latch, for parking a scripted hook where a test needs the actor held.
    class Latch {
      public:
        void wait() {
            std::unique_lock<std::mutex> lock(mutex_);
            opened_.wait(lock, [this] { return open_; });
        }
        void open() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                open_ = true;
            }
            opened_.notify_all();
        }

      private:
        std::mutex mutex_;
        std::condition_variable opened_;
        bool open_ = false;
    };

    // -- the sinks --------------------------------------------------------------------------

    // Refuses the first `refuse_first` frames with QueueFullError, then accepts.
    class RefusingSink : public FrameSink {
      public:
        explicit RefusingSink(uint64_t refuse_first) : refuse_first_(refuse_first) {}

        void put(Frame&& frame) override {
            std::lock_guard<std::mutex> lock(mutex_);
            ++offered_;
            if (offered_ <= refuse_first_) {
                ++refused_by_camera_[frame.tag.camera_id];
                throw QueueFullError("sink is full", 4096, 4096);
            }
            ++accepted_;
        }

        uint64_t accepted() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return accepted_;
        }
        std::map<std::string, uint64_t> refused_by_camera() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return refused_by_camera_;
        }

      private:
        mutable std::mutex mutex_;
        uint64_t refuse_first_;
        uint64_t offered_ = 0;
        uint64_t accepted_ = 0;
        std::map<std::string, uint64_t> refused_by_camera_;
    };

    class ClosingSink : public FrameSink {
      public:
        void put(Frame&&) override {
            offered_.fetch_add(1);
            throw RequestCancelledError("the consumer is gone");
        }
        uint64_t offered() const { return offered_.load(); }

      private:
        std::atomic<uint64_t> offered_{0};
    };

    // Throws something the actor has no case for. Its safety net is what must catch it.
    class UntypedSink : public FrameSink {
      public:
        explicit UntypedSink(uint64_t throw_first) : throw_first_(throw_first) {}
        void put(Frame&&) override {
            if (offered_.fetch_add(1) < throw_first_) {
                throw std::logic_error("a bug in a sink nobody wrote a case for");
            }
            accepted_.fetch_add(1);
        }
        uint64_t accepted() const { return accepted_.load(); }

      private:
        uint64_t throw_first_;
        std::atomic<uint64_t> offered_{0};
        std::atomic<uint64_t> accepted_{0};
    };

    // =====================================================================================
    // A. StopSignal
    // =====================================================================================

    void test_stop_signal() {
        StopSignal signal;
        const auto start = Clock::now();
        check(!signal.wait_for(0.05) && ms_since(start) >= 40.0,
              "an unset signal waits out its timeout and reports nothing to do");

        signal.set();
        const auto immediate = Clock::now();
        check(signal.wait_for(30.0) && ms_since(immediate) < 5.0,
              "set before the wait: the answer is immediate, not in thirty seconds");
        check(signal.is_set(),
              "and it is sticky — a thread that was not yet waiting still sees it");
        signal.clear();
        check(!signal.is_set(), "clear() puts it back");

        StopSignal late;
        std::atomic<double> waited_ms{-1.0};
        std::thread waiter([&] {
            const auto begin = Clock::now();
            (void)late.wait_for(30.0);
            waited_ms.store(ms_since(begin));
        });
        std::this_thread::sleep_for(20ms);
        late.set();
        waiter.join();
        check(waited_ms.load() >= 0.0 && waited_ms.load() < 200.0,
              "a cross-thread set wakes a 30 s wait at once (" +
                  std::to_string(waited_ms.load()) +
                  " ms) — the reason a reconnect delay is "
                  "not a sleep");

        StopSignal both;
        std::atomic<int> woke{0};
        std::thread a([&] {
            both.wait();
            woke.fetch_add(1);
        });
        std::thread b([&] {
            both.wait();
            woke.fetch_add(1);
        });
        std::this_thread::sleep_for(10ms);
        both.set();
        a.join();
        b.join();
        check(woke.load() == 2, "every waiter wakes, not just the first");
    }

    // =====================================================================================
    // B. ExponentialBackoff
    // =====================================================================================

    void test_backoff_sequence() {
        ExponentialBackoff plain(0.5, 30.0, 2.0, 0.0);
        check(plain.peek() == 0.5, "the first delay is the configured floor");
        check(plain.attempts() == 0, "and peeking does not consume an attempt");
        const std::vector<double> want{0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0};
        std::vector<double> got;
        for (size_t i = 0; i < want.size(); ++i) got.push_back(plain.next_delay());
        check(got == want,
              "the policy is a sequence, and this is it: 0.5, 1, 2, 4, 8, 16, 30, 30");
        check(plain.attempts() == static_cast<int>(want.size()), "attempts counts the delays");
        plain.reset();
        check(plain.attempts() == 0 && plain.next_delay() == 0.5,
              "reset goes back to the floor — what a *frame* does, never a connect");
    }

    void test_backoff_jitter_and_cap() {
        ExponentialBackoff backoff(0.5, 30.0, 2.0, 0.2, /*seed=*/7);
        bool in_band = true;
        bool under_cap = true;
        for (int i = 0; i < 200; ++i) {
            const double base = backoff.peek();
            const double drawn = backoff.next_delay();
            in_band &= drawn >= 0.8 * base - 1e-12 && drawn <= base + 1e-12;
            under_cap &= drawn <= 30.0 + 1e-12;
        }
        check(in_band, "every draw is inside [0.8 d, d] — subtractive jitter");
        check(under_cap, "so the cap is a real bound, which additive jitter would not give");

        ExponentialBackoff at_cap(0.5, 30.0, 2.0, 0.2, /*seed=*/11);
        for (int i = 0; i < 10; ++i) (void)at_cap.next_delay();  // walk it up to the ceiling
        std::vector<double> draws;
        for (int i = 0; i < 50; ++i) draws.push_back(at_cap.next_delay());
        bool all_same = true;
        for (double d : draws) all_same &= d == draws.front();
        check(!all_same,
              "draws differ: fifty cameras behind one switch must not retry in lockstep");
    }

    void test_backoff_survives_the_night() {
        ExponentialBackoff backoff(0.5, 30.0, 2.0, 0.0);
        double last = 0.0;
        for (int i = 0; i < 2000; ++i) last = backoff.next_delay();
        check(std::isfinite(last) && last == 30.0,
              "attempt 2000 is still the cap, not an overflow — the camera down all night is "
              "the one that most needs to still be trying");
    }

    void test_backoff_refuses_nonsense() {
        auto refused = [](double initial, double cap, double factor, double jitter) {
            try {
                ExponentialBackoff backoff(initial, cap, factor, jitter);
                (void)backoff;
            } catch (const ConfigError&) {
                return true;
            }
            return false;
        };
        check(refused(0.0, 30.0, 2.0, 0.2), "a zero floor is a configuration error");
        check(refused(1.0, 0.5, 2.0, 0.2), "a cap below the floor is a configuration error");
        check(refused(0.5, 30.0, 1.0, 0.2), "a factor of 1 does not back off");
        check(refused(0.5, 30.0, 2.0, 1.0), "jitter of 1 could return a zero delay");
    }

    // =====================================================================================
    // C. DeadlinePacer
    // =====================================================================================

    void test_pacer() {
        DeadlinePacer off(0.0);
        bool asked = false;
        check(!off.wait([&](double) {
            asked = true;
            return false;
        }),
              "fps <= 0 disables pacing: a live camera sets its own rate");
        check(!asked && !off.enabled(), "and nothing is ever asked to sleep");

        // Deadlines accumulate: the requested delay grows because the schedule is absolute,
        // not because each wait is longer. A `sleep(1/fps)` loop would ask for the same value
        // every time and deliver 17 fps for a 20 fps target.
        DeadlinePacer paced(100.0);
        paced.reset();
        std::vector<double> asked_for;
        auto record = [&](double seconds) {
            asked_for.push_back(seconds);
            return false;  // never actually sleep
        };
        for (int i = 0; i < 3; ++i) (void)paced.wait(record);
        check(asked_for.size() == 3 && asked_for[1] > asked_for[0] &&
                  asked_for[2] > asked_for[1] && asked_for[2] < 0.035,
              "the deadline accumulates: 10 ms, 20 ms, 30 ms out, because nothing slept");

        DeadlinePacer late(1000.0);
        late.reset();
        std::this_thread::sleep_for(30ms);  // 30 periods behind
        int slept = 0;
        auto count = [&](double) {
            ++slept;
            return false;
        };
        (void)late.wait(count);
        check(
            late.behind() == 1 && slept == 0,
            "lateness is absorbed and counted once, not repaid as thirty back-to-back frames");
        (void)late.wait(count);
        check(late.behind() == 1 && slept == 1,
              "and the schedule restarts from now, so the next frame is paced again");

        DeadlinePacer interrupted(10.0);
        interrupted.reset();
        check(interrupted.wait([](double) { return true; }),
              "an interrupted wait says so, and the caller produces no frame");
        double after = 0.0;
        (void)interrupted.wait([&](double seconds) {
            after = seconds;
            return false;
        });
        check(after > 0.05 && after <= 0.1 + 1e-9,
              "and the deadline did not advance across it, so no frame's worth of schedule was "
              "skipped");
    }

    // =====================================================================================
    // D. FrameCounter and the two clocks
    // =====================================================================================

    void test_frame_counter() {
        FrameCounter counter("cam7", 100);
        HostFrame image;
        std::vector<uint8_t> pixels(12, 3);
        image.pixels = pixels.data();
        image.height = 2;
        image.width = 2;

        const Frame first = counter.stamp(image);
        const Frame second = counter.stamp(image);
        check(first.tag.frame_id == 100 && second.tag.frame_id == 101,
              "ids run consecutively from first_frame_id");
        check(first.tag.key() == "cam7:100", "the reassembly key is camera:frame");
        check(counter.stamped() == 2 && counter.next_frame_id() == 102,
              "and the counter says where the sequence has reached");

        // The counter belongs to the ACTOR, so a rebuilt source keeps counting. A counter that
        // died with its source would hand a downstream tracker a second frame 100.
        StopSignal stop;
        IngestConfig config = a_camera("cam7");
        FakeScript script;
        script.on_read = [](int) { return 1; };
        int64_t after_rebuild = 0;
        {
            FakeSource source(config, counter, stop, script);
            source.open();
            after_rebuild = source.read()->tag.frame_id;
            source.close();
        }
        {
            FakeSource rebuilt(config, counter, stop, script);
            rebuilt.open();
            check(after_rebuild == 102 && rebuilt.read()->tag.frame_id == 103,
                  "a source rebuilt by a reconnect keeps counting — no second frame 102");
            rebuilt.close();
        }

        const int64_t steady = monotonic_ns();
        const int64_t wall = unix_ns();
        const Frame stamped = counter.stamp(image);
        check(std::llabs(stamped.tag.captured_ns - steady) < 50'000'000LL,
              "captured_ns is the STEADY clock, read at decode");
        check(stamped.tag.captured_unix_ns > 1'000'000'000'000'000'000LL &&
                  std::llabs(stamped.tag.captured_unix_ns - wall) < 2'000'000'000LL,
              "captured_unix_ns is the WALL clock, read at decode");
        check(
            stamped.tag.captured_unix_ns - stamped.tag.captured_ns > 100'000'000'000'000'000LL,
            "the two are orders of magnitude apart — which is why stamping wall time into "
            "captured_ns would put every deadline ~54 years out and expire nothing");
    }

    // =====================================================================================
    // E. The FrameSource template method
    // =====================================================================================

    void test_source_contract() {
        IngestConfig config = a_camera("cam0");
        StopSignal stop;

        {
            FrameCounter counter("cam0");
            FakeScript script;
            FakeSource source(config, counter, stop, script);
            bool refused = false;
            try {
                (void)source.read();
            } catch (const SourceOpenError&) {
                refused = true;
            }
            check(refused, "read() before open() is an error, not an empty result");
            check(!source.is_exhausted() && !source.hwaccel(),
                  "a backend that cannot decode on a GPU resolves hwaccel to false however the "
                  "camera is configured");

            source.close();
            check(script.closes.load() == 0, "close() before open() touches nothing");
            source.open();
            source.open();
            check(script.opens.load() == 1, "open() is idempotent");
            check(source.is_open() && source.height() == 4 && source.width() == 6,
                  "set_format reports what the stream negotiated");
            source.close();
            source.close();
            check(script.closes.load() == 1, "and so is close()");
        }

        {
            FrameCounter counter("cam0");
            FakeScript script;
            script.on_open = [](int) { throw SourceOpenError("cam0", "fake://x", "refused"); };
            FakeSource source(config, counter, stop, script);
            bool refused = false;
            try {
                source.open();
            } catch (const SourceOpenError&) {
                refused = true;
            }
            check(refused && !source.is_open(), "a throwing do_open leaves the source closed");
            check(script.closes.load() == 1,
                  "and gets exactly one unwind, so a half-open stream does not leak a socket");
        }

        {
            FrameCounter counter("cam0");
            FakeScript script;
            script.on_open = [](int) { throw SourceOpenError("cam0", "fake://x", "refused"); };
            script.on_close = [](int) { throw std::runtime_error("close blew up too"); };
            FakeSource source(config, counter, stop, script);
            std::string message;
            try {
                source.open();
            } catch (const std::exception& error) {
                message = error.what();
            }
            check(message.find("refused") != std::string::npos,
                  "the ORIGINAL failure propagates even when the unwind throws too — the "
                  "diagnosis is the open, not the cleanup");
        }

        {
            FrameCounter counter("cam0", 5);
            FakeScript script;
            script.on_read = [](int index) { return index == 0 ? 0 : 1; };
            FakeSource source(config, counter, stop, script);
            source.open();
            check(!source.read().has_value() && counter.next_frame_id() == 5,
                  "'not yet' consumes no frame id");
            const std::optional<Frame> frame = source.read();
            check(
                frame.has_value() && frame->tag.frame_id == 5 && frame->tag.camera_id == "cam0",
                "the frame that does arrive gets the next id");
            check(frame->tag.captured_ns > 0 && frame->tag.captured_unix_ns > 0 &&
                      frame->image.bytes() == 4 * 6 * 3,
                  "stamped with both clocks, and the image comes through intact");
            source.close();
        }

        {
            FrameCounter wrong("someone_else");
            FakeScript script;
            bool refused = false;
            try {
                FakeSource source(config, wrong, stop, script);
                (void)source;
            } catch (const ConfigError&) {
                refused = true;
            }
            check(refused,
                  "a counter belonging to another camera is refused at construction, not "
                  "discovered as a duplicate tag downstream");
        }
    }

    // =====================================================================================
    // F. The source registry
    // =====================================================================================

    void test_registry() {
        check(SOURCES().contains("fake") && SOURCES().contains("scripted"),
              "a file-scope registrar is the whole registration");
        check(SOURCES().canonical("double") == "fake",
              "an alias resolves to the canonical name");

        IngestConfig config = a_camera("cam0");
        config.source = "double";
        FrameCounter counter("cam0");
        StopSignal stop;
        std::unique_ptr<FrameSource> built = create_source(config, counter, stop);
        check(built != nullptr && built->camera_id() == "cam0",
              "and building by that alias constructs the registered source");

        std::string message;
        try {
            IngestConfig unknown = a_camera("cam0");
            unknown.source = "gstremaer";
            (void)create_source(unknown, counter, stop);
        } catch (const ConfigError& error) {
            message = error.what();
        }
        check(message.find("gstremaer") != std::string::npos &&
                  message.find("fake") != std::string::npos,
              "an unknown name is refused with the known ones listed");

        bool refused = false;
        try {
            SOURCES().add("fake", {}, "again", nullptr);
        } catch (const ConfigError&) {
            refused = true;
        }
        check(refused,
              "a name registered twice is a configuration error, not a silent replace");

        std::string unnamed;
        try {
            IngestConfig empty = a_camera("cam0");
            empty.source.clear();
            (void)create_source(empty, counter, stop);
        } catch (const ConfigError& error) {
            unnamed = error.what();
        }
        check(unnamed.find("source") != std::string::npos,
              "an empty source names the knob rather than guessing a backend");

        // The one honest gap in this tier: `replay` registers itself from a unit that reaches
        // the driver's headers, so an offline binary does not link it and its registry contains
        // no real source at all. That is the offline-closure invariant working, not a hole —
        // and it is reported rather than papered over.
        if (SOURCES().contains("replay")) {
            check(SOURCES().canonical("file") == "replay",
                  "the replay source's aliases resolve where it is linked");
        } else {
            skip(
                "no real video source in this binary: replay lives in a CUDA-facing unit that "
                "the offline build does not compile (see ingest/registry.cpp)");
        }
    }

    // =====================================================================================
    // G. The actor: connecting, backing off, and what resets what
    // =====================================================================================

    void test_actor_backoff_sequence() {
        FakeScript script;
        script.on_open = [](int) { throw SourceOpenError("cam0", "fake://x", "refused"); };
        CountingSink sink;
        SteppingWait gate;
        CameraActor actor(a_camera("cam0"), sink, scripted(script),
                          [&gate](double seconds) { return gate(seconds); });
        actor.start();
        check(gate.arrived(1), "the actor reached its first reconnect delay");
        gate.step();
        check(gate.arrived(2), "and its second");
        const std::vector<double> delays = gate.delays();
        check(delays.size() >= 2 && delays[0] >= 0.4 && delays[0] <= 0.5,
              "the first retry is the jittered floor, in [0.4, 0.5]");
        check(delays.size() >= 2 && delays[1] >= 0.8 && delays[1] <= 1.0,
              "and the second has doubled, in [0.8, 1.0]");
        finish(actor, gate);
    }

    void test_a_connect_does_not_clear_a_failure_but_a_frame_does() {
        // The policy that is not the obvious one. A source that accepts a connection and then
        // delivers nothing is the most common real failure of a camera fleet.
        std::atomic<bool> deliver{false};
        FakeScript script;
        script.on_open = [](int index) {
            if (index == 0) throw SourceOpenError("cam0", "fake://x", "refused once");
        };
        script.on_read = [&deliver](int) { return deliver.load() ? 1 : 0; };
        CountingSink sink;
        SteppingWait gate;
        IngestConfig config = a_camera("cam0");
        config.empty_reads_before_reconnect =
            1000;  // isolate the connect from the empty budget
        CameraActor actor(config, sink, scripted(script),
                          [&gate](double seconds) { return gate(seconds); });
        actor.start();
        check(gate.arrived(1), "the first connect failed and parked in its backoff");
        gate.step();
        check(gate.arrived(2), "the second connect succeeded and the source read nothing yet");
        check(actor.health().connects == 1 && actor.health().consecutive_failures == 1,
              "a successful CONNECT does not clear the failure count");
        deliver.store(true);
        gate.step(4);
        // The empty-read sleep is what parks the actor, so a frame has landed by the time the
        // gate has been stepped past it.
        for (int i = 0; i < 100 && actor.health().frames_read == 0; ++i) {
            gate.step();
            std::this_thread::sleep_for(2ms);
        }
        check(actor.health().frames_read >= 1 && actor.health().consecutive_failures == 0,
              "a FRAME does");
        finish(actor, gate);
    }

    void test_a_missing_decode_runtime_is_fatal() {
        FakeScript script;
        script.on_open = [](int) {
            throw SourceUnavailableError("fake", "install the runtime");
        };
        CountingSink sink;
        CameraActor actor(a_camera("cam0"), sink, scripted(script));
        actor.start();
        for (int i = 0; i < 200 && actor.is_running(); ++i) std::this_thread::sleep_for(5ms);
        check(!actor.is_running(),
              "the actor gives up: no amount of retrying installs a library");
        check(script.builds.load() == 1,
              "and the factory was called exactly once — not hammered behind a reconnect loop");
        check(actor.state() == CameraState::Unhealthy, "it reports UNHEALTHY");
        actor.stop();
        check(
            actor.state() == CameraState::Unhealthy,
            "and stays UNHEALTHY after stop(), because 'install the runtime' and 'an operator "
            "removed it' are different answers");
    }

    void test_degraded_then_unhealthy_and_sticky() {
        Latch held;
        FakeScript script;
        script.on_open = [&held](int index) {
            if (index == 3) held.wait();  // park the fourth attempt mid-connect
            throw SourceOpenError("cam0", "fake://x", "refused");
        };
        CountingSink sink;
        SteppingWait gate;
        CameraActor actor(a_camera("cam0"), sink, scripted(script),
                          [&gate](double seconds) { return gate(seconds); });
        actor.start();
        check(gate.arrived(1) && actor.state() == CameraState::Degraded,
              "one failure is DEGRADED — recoverable, and not worth paging on");
        gate.step();
        check(gate.arrived(2) && actor.state() == CameraState::Degraded, "two still is");
        gate.step();
        check(gate.arrived(3) && actor.state() == CameraState::Unhealthy,
              "the third consecutive failure is UNHEALTHY (failures_before_unhealthy = 3)");
        gate.step();
        // The actor is now inside its fourth connect, past `mark_connecting` and before the
        // failure: the state it reports here is what a dashboard sampling it would see.
        for (int i = 0; i < 200 && script.opens.load() < 4; ++i)
            std::this_thread::sleep_for(2ms);
        check(actor.state() == CameraState::Unhealthy,
              "and it is sticky across a retry: a camera flapping UNHEALTHY/CONNECTING every "
              "30 s is a dashboard that reports whichever it happened to catch");
        held.open();
        finish(actor, gate);
    }

    void test_a_broken_read_rebuilds_the_source() {
        FakeScript script;
        script.on_read = [](int index) -> int {
            if (index == 0) throw FrameDecodeError("cam0", "the stream ended");
            return 1;
        };
        CountingSink sink;
        SteppingWait gate;
        CameraActor actor(a_camera("cam0"), sink, scripted(script),
                          [&gate](double seconds) { return gate(seconds); });
        actor.start();
        check(gate.arrived(1), "a throwing read backs off like a failed connect");
        check(script.closes.load() == 1,
              "the broken source is closed, not leaked — one bad frame is not a fleet");
        gate.step();
        for (int i = 0; i < 200 && script.builds.load() < 2; ++i)
            std::this_thread::sleep_for(2ms);
        check(script.builds.load() == 2, "and a fresh one is built by the factory");
        check(actor.health().frames_read >= 1, "after which frames arrive again");
        finish(actor, gate);
    }

    void test_an_untyped_sink_exception_does_not_kill_the_camera() {
        FakeScript script;
        script.on_read = [](int) { return 1; };
        UntypedSink sink(/*throw_first=*/1);
        SteppingWait gate;
        CameraActor actor(a_camera("cam0"), sink, scripted(script),
                          [&gate](double seconds) { return gate(seconds); });
        actor.start();
        check(
            gate.arrived(1),
            "a sink that throws something nobody wrote a case for is caught by the safety net");
        check(actor.health().connect_failures == 1, "counted");
        gate.step();
        for (int i = 0; i < 200 && sink.accepted() == 0; ++i) {
            gate.step();
            std::this_thread::sleep_for(2ms);
        }
        check(sink.accepted() >= 1 && actor.is_running(),
              "and the camera keeps going: a bug in a sink degrades one camera, it does not "
              "leave the fleet quietly 49 cameras wide");
        finish(actor, gate);
    }

    // =====================================================================================
    // H. Empty reads and exhaustion
    // =====================================================================================

    void test_the_empty_read_budget() {
        FakeScript script;
        script.on_read = [](int) { return 0; };  // a source that never delivers
        CountingSink sink;
        SteppingWait gate;
        IngestConfig config = a_camera("cam0");  // budget 5, sleep 5 ms
        CameraActor actor(config, sink, scripted(script),
                          [&gate](double seconds) { return gate(seconds); });
        actor.start();
        check(gate.arrived(1), "the first empty read parks in the anti-spin sleep");
        gate.step(3);
        check(gate.arrived(4), "four empty reads inside the budget of five");
        const std::vector<double> waits = gate.delays();
        bool exactly_the_sleep = waits.size() >= 4;
        for (size_t i = 0; i < 4 && i < waits.size(); ++i) {
            exactly_the_sleep &= std::abs(waits[i] - 0.005) < 1e-9;
        }
        check(exactly_the_sleep,
              "and each waits exactly empty_read_sleep_ms — the knob that stops a source "
              "returning nothing from spinning a core at 100%");
        check(actor.health().empty_reads == 4 && actor.state() == CameraState::Degraded,
              "four empty reads, counted, and the camera reads DEGRADED");
        check(script.opens.load() == 1 && actor.health().connect_failures == 0,
              "with no reconnect yet: an RTSP stream between frames is not a fault");
        gate.step();
        check(gate.arrived(5), "the fifth empty read is the one that reconnects");
        check(gate.delays()[4] > 0.1 && actor.health().connect_failures == 1,
              "it backs off rather than sleeping 5 ms again, and counts the failure");
        check(script.closes.load() == 1, "tearing the silent source down on the way");
        finish(actor, gate);
    }

    void test_a_frame_restarts_the_empty_run() {
        std::atomic<int> delivered{0};
        FakeScript script;
        // Empty, empty, empty, a frame, then empty forever: without the reset, read 8 would be
        // the fifth consecutive empty and reconnect.
        script.on_read = [&delivered](int index) {
            if (index == 3) {
                delivered.fetch_add(1);
                return 1;
            }
            return 0;
        };
        CountingSink sink;
        SteppingWait gate;
        CameraActor actor(a_camera("cam0"), sink, scripted(script),
                          [&gate](double seconds) { return gate(seconds); });
        actor.start();
        check(gate.arrived(1), "the actor is running");
        gate.step(7);
        for (int i = 0; i < 200 && script.reads.load() < 8; ++i)
            std::this_thread::sleep_for(2ms);
        check(delivered.load() == 1 && script.opens.load() == 1,
              "eight reads with one frame among them: the frame restarted the run, so no "
              "reconnect");
        finish(actor, gate);
    }

    void test_an_exhausted_source_finishes() {
        FakeScript script;
        script.on_read = [&script](int index) {
            if (index == 0) return 1;
            script.exhausted.store(true);
            return 0;
        };
        CountingSink sink;
        CameraActor actor(a_camera("cam0"), sink, scripted(script));
        actor.start();
        for (int i = 0; i < 400 && actor.is_running(); ++i) std::this_thread::sleep_for(5ms);
        check(!actor.is_running(),
              "a finite source finishing is how a bench terminates itself");
        check(actor.state() == CameraState::Exhausted && script.opens.load() == 1,
              "EXHAUSTED, and short-circuited before the empty budget could reconnect it");
        check(sink.total() == 1, "having delivered exactly what it had");
        actor.stop();
        check(
            actor.state() == CameraState::Exhausted,
            "and EXHAUSTED survives stop(): a finished file is not a camera somebody removed");
    }

    // =====================================================================================
    // I. What the sink says, and what the actor does about it
    // =====================================================================================

    void test_a_full_sink_is_a_drop_charged_to_this_camera() {
        FakeScript script;
        script.on_read = [](int) { return 1; };
        RefusingSink sink(/*refuse_first=*/1);
        CameraActor actor(a_camera("cam0"), sink, scripted(script));
        actor.start();
        for (int i = 0; i < 400 && sink.accepted() == 0; ++i) std::this_thread::sleep_for(5ms);
        actor.stop();
        const CameraHealth health = actor.health();
        check(health.frames_dropped == 1,
              "a refused frame is one drop, and the camera carries on to the next");
        check(health.frames_published >= 1 && sink.accepted() >= 1,
              "the frame after the refusal is published");
        check(health.frames_read == health.frames_published + health.frames_dropped,
              "frames_read counts the refused one too: it is offered load, not accepted load");
        check(sink.refused_by_camera().at("cam0") == 1,
              "and the drop is charged to THIS camera — the attribution ADR-005 exists for");
        check(health.drop_ratio() > 0.0 && health.drop_ratio() <= 1.0,
              "which is what makes drop_ratio a number an operator can act on");
    }

    void test_a_closed_sink_finishes_the_actor() {
        FakeScript script;
        script.on_read = [](int) { return 1; };
        ClosingSink sink;
        CameraActor actor(a_camera("cam0"), sink, scripted(script));
        actor.start();
        for (int i = 0; i < 400 && actor.is_running(); ++i) std::this_thread::sleep_for(5ms);
        check(!actor.is_running() && sink.offered() == 1,
              "the consumer is gone, so the actor finishes instead of logging one line per "
              "frame for as long as the process lives");
        actor.stop();
        check(actor.state() == CameraState::Stopped, "and reports STOPPED");
    }

    void test_an_accepting_sink_publishes_everything() {
        FakeScript script;
        script.on_read = [](int) { return 1; };
        CountingSink sink;
        CameraActor actor(a_camera("cam0"), sink, scripted(script));
        actor.start();
        for (int i = 0; i < 400 && sink.total() < 20; ++i) std::this_thread::sleep_for(2ms);
        actor.stop();
        const CameraHealth health = actor.health();
        check(health.frames_read == health.frames_published && health.frames_dropped == 0,
              "nothing refused, nothing dropped: published == read");
        check(sink.counts().at("cam0") == health.frames_published,
              "and the sink's own per-camera count agrees with the camera's");
    }

    // =====================================================================================
    // J. Lifecycle
    // =====================================================================================

    void test_actor_lifecycle() {
        {
            FakeScript script;
            script.on_read = [](int) { return 0; };
            CountingSink sink;
            CameraActor actor(a_camera("cam0"), sink, scripted(script));
            actor.start();
            bool refused = false;
            try {
                actor.start();
            } catch (const ServerStateError&) {
                refused = true;
            }
            check(refused,
                  "an actor is never restarted: the manager builds a fresh one, which has an "
                  "opinion about where its frame counter stands");
            actor.stop();
            check(actor.state() == CameraState::Stopped, "a clean stop reports STOPPED");
            actor.stop();
            check(!actor.is_running(), "and stopping twice is a no-op, not a hang");
        }
        {
            FakeScript script;
            CountingSink sink;
            CameraActor actor(a_camera("cam0"), sink, scripted(script));
            actor.stop();
            check(!actor.is_running() && script.builds.load() == 0,
                  "stopping an actor that never started is a no-op too — shutdown paths call "
                  "this from more than one place and neither may hang");
        }
    }

    void test_stop_interrupts_a_thirty_second_backoff() {
        FakeScript script;
        script.on_open = [](int) { throw SourceOpenError("cam0", "fake://x", "refused"); };
        CountingSink sink;
        IngestConfig config = a_camera("cam0");
        config.reconnect_initial_ms = 30000;  // straight to the cap
        config.reconnect_max_ms = 30000;
        // The default wait: on the stop signal, not on the clock. That is the whole check.
        CameraActor actor(config, sink, scripted(script));
        actor.start();
        for (int i = 0; i < 400 && script.opens.load() == 0; ++i)
            std::this_thread::sleep_for(5ms);
        std::this_thread::sleep_for(20ms);  // let it get into the backoff
        const auto start = Clock::now();
        actor.stop(2000ms);
        const double waited = ms_since(start);
        check(waited < 200.0,
              "stop() lands inside a 30 s backoff in " + std::to_string(waited) +
                  " ms — a sleeping actor is one that gets abandoned holding a decoder");
        check(!actor.is_running(), "and the thread is gone, not detached");
    }

    void test_health_is_safe_to_read_from_any_thread() {
        FakeScript script;
        script.on_read = [](int) { return 1; };
        CountingSink sink;
        CameraActor actor(a_camera("cam0"), sink, scripted(script));
        actor.start();
        std::atomic<int> mismatched{0};
        std::vector<std::thread> readers;
        for (int t = 0; t < 4; ++t) {
            readers.emplace_back([&] {
                for (int i = 0; i < 500; ++i) {
                    const CameraHealth health = actor.health();
                    if (health.camera_id != "cam0" ||
                        health.frames_read < health.frames_published) {
                        mismatched.fetch_add(1);
                    }
                }
            });
        }
        for (std::thread& reader : readers) reader.join();
        actor.stop();
        check(mismatched.load() == 0,
              "a snapshot is built under the actor's lock, so no reader ever sees half of an "
              "update");
    }

    // =====================================================================================
    // K. Health and the fps window
    // =====================================================================================

    void test_health_reports_a_recent_rate() {
        CameraHealth idle;
        idle.camera_id = "cam0";
        check(idle.fps == 0.0 && !idle.is_healthy() && idle.drop_ratio() == 0.0,
              "a camera with no frames reports no rate and no drops");
        CameraHealth losing;
        losing.state = CameraState::Streaming;
        losing.frames_read = 10;
        losing.frames_dropped = 2;
        check(losing.drop_ratio() == 0.2 && losing.is_healthy(),
              "drop_ratio is drops over offered load, and STREAMING is healthy");
        check(is_healthy(CameraState::Connecting) && !is_healthy(CameraState::Unhealthy) &&
                  !is_healthy(CameraState::Degraded),
              "CONNECTING is healthy (a camera two seconds into start-up is not a fault); "
              "DEGRADED and UNHEALTHY are not");
        check(std::string(to_string(CameraState::Exhausted)) == "exhausted" &&
                  std::string(to_string(CameraState::Idle)) == "idle",
              "the state names match the Python enum's values, so one health payload reads the "
              "same from either plane");

        // A partial window: the actor answers before the first full 2 s has closed, because
        // reporting 0 fps for the first two seconds of every camera makes a start-up look like
        // an outage.
        std::atomic<bool> stop_reading{false};
        FakeScript script;
        script.on_read = [&stop_reading](int) {
            std::this_thread::sleep_for(10ms);  // ~100 frames a second offered
            if (stop_reading.load()) throw FrameDecodeError("cam0", "cut off");
            return 1;
        };
        CountingSink sink;
        CameraActor actor(a_camera("cam0"), sink, scripted(script));
        actor.start();
        std::this_thread::sleep_for(600ms);
        const double fps = actor.health().fps;
        check(fps > 30.0 && fps < 300.0,
              "the partial-window estimate is near the offered rate (" + std::to_string(fps) +
                  " fps against ~100 offered)");
        stop_reading.store(true);
        for (int i = 0; i < 200 && actor.health().connect_failures == 0; ++i) {
            std::this_thread::sleep_for(5ms);
        }
        check(actor.health().fps == 0.0,
              "a failure zeroes it: a camera that has just failed is not still delivering at "
              "whatever rate it last managed");
        actor.stop();
    }

    // =====================================================================================
    // L. The manager
    // =====================================================================================

    // Builds a fleet whose sources are scripted per camera.
    struct Fleet {
        std::map<std::string, std::unique_ptr<FakeScript>> scripts;
        std::atomic<int> built{0};

        FakeScript& script(const std::string& camera_id) {
            auto found = scripts.find(camera_id);
            if (found == scripts.end()) {
                found = scripts.emplace(camera_id, std::make_unique<FakeScript>()).first;
            }
            return *found->second;
        }

        SourceFactory factory() {
            return [this](const IngestConfig& config, FrameCounter& counter,
                          StopSignal& stop) -> std::unique_ptr<FrameSource> {
                built.fetch_add(1);
                return std::make_unique<FakeSource>(config, counter, stop,
                                                    script(config.camera_id));
            };
        }
    };

    void test_the_manager_validates_before_it_starts_a_thread() {
        Fleet fleet;
        CountingSink sink;
        std::vector<IngestConfig> cameras{a_camera("cam0"), a_camera("cam1"), a_camera("cam0")};
        IngestManager manager(cameras, sink, fleet.factory());
        std::string message;
        try {
            manager.start();
        } catch (const ConfigError& error) {
            message = error.what();
        }
        check(message.find("cam0") != std::string::npos,
              "a duplicate camera id is refused, and named");
        check(fleet.built.load() == 0 && manager.size() == 0,
              "before any thread exists — a mistyped database is a start-up failure, not fifty "
              "actors failing one at a time");
    }

    void test_a_disabled_camera_is_not_in_the_fleet() {
        Fleet fleet;
        CountingSink sink;
        IngestConfig off = a_camera("cam1");
        off.enabled = false;
        for (auto& camera : {a_camera("cam0"), off}) fleet.script(camera.camera_id);
        fleet.script("cam0").on_read = [](int) { return 1; };
        IngestManager manager({a_camera("cam0"), off}, sink, fleet.factory());
        check(manager.configured_cameras().size() == 1,
              "a camera can stay in the database and out of the fleet");
        manager.start();
        manager.start();
        check(manager.size() == 1 && manager.contains("cam0") && !manager.contains("cam1"),
              "start() is idempotent and runs only what is enabled");
        manager.stop();
    }

    void test_adding_and_removing_a_camera_while_the_fleet_runs() {
        Fleet fleet;
        CountingSink sink;
        fleet.script("cam0").on_read = [](int) { return 1; };
        fleet.script("cam1").on_read = [](int) { return 1; };
        IngestManager manager({a_camera("cam0")}, sink, fleet.factory());
        manager.start();
        bool refused = false;
        try {
            manager.add_camera(a_camera("cam0"));
        } catch (const ConfigError&) {
            refused = true;
        }
        check(refused,
              "adding a camera that is already running is refused: two threads on one stream "
              "means two counters producing duplicate tags");
        manager.add_camera(a_camera("cam1"));
        check(manager.size() == 2, "a camera is onboarded without restarting the fleet");
        manager.remove_camera("cam1");
        check(manager.size() == 1 && !manager.contains("cam1"), "and removed again");
        std::string message;
        try {
            manager.remove_camera("cam9");
        } catch (const ConfigError& error) {
            message = error.what();
        }
        check(message.find("cam9") != std::string::npos &&
                  message.find("cam0") != std::string::npos,
              "removing a camera that is not there names what IS running, so a typo in an "
              "operator's call gets an answer instead of a silent no-op");
        manager.stop();
        check(manager.size() == 0, "and stop() forgets the fleet");
    }

    void test_stop_signals_everybody_before_it_joins_anybody() {
        // Eight cameras, each parked inside a read that takes a second and cannot be
        // interrupted. Signal-then-join costs one read; join-then-signal costs eight, which is
        // the shutdown this two-pass exists to fix.
        Fleet fleet;
        CountingSink sink;
        std::vector<IngestConfig> cameras;
        for (int c = 0; c < 8; ++c) {
            const std::string id = "cam" + std::to_string(c);
            cameras.push_back(a_camera(id));
            fleet.script(id).on_read = [](int) {
                std::this_thread::sleep_for(1000ms);
                return 1;
            };
        }
        IngestManager manager(cameras, sink, fleet.factory());
        manager.start();
        std::this_thread::sleep_for(150ms);  // all eight are inside a read
        const auto start = Clock::now();
        manager.stop(5000ms);
        const double waited = ms_since(start);
        check(waited < 4000.0, "eight cameras stop in " + std::to_string(waited) +
                                   " ms — one read timeout, not eight");
        check(manager.size() == 0, "and none of them is left running");
    }

    void test_wait_ready_names_the_silent_cameras() {
        Fleet fleet;
        CountingSink sink;
        std::vector<IngestConfig> cameras;
        for (int c = 0; c < 5; ++c) {
            const std::string id = "cam" + std::to_string(c);
            cameras.push_back(a_camera(id));
            // cam1 and cam3 accept a connection and then deliver nothing — the failure a
            // fleet database typo actually produces.
            const bool silent = (c == 1 || c == 3);
            fleet.script(id).on_read = [silent](int) { return silent ? 0 : 1; };
        }
        IngestManager manager(cameras, sink, fleet.factory());
        manager.start();
        std::vector<std::string> named;
        double reported = 0.0;
        try {
            manager.wait_ready(500ms, 10ms);
        } catch (const CameraUnavailableError& error) {
            named = error.cameras;
            reported = error.timeout_s;
        }
        check(named == std::vector<std::string>{"cam1", "cam3"},
              "wait_ready names exactly the cameras that produced nothing — a deploy against a "
              "mistyped database fails at start-up instead of looking healthy");
        check(reported > 0.4 && reported < 0.6, "and says how long it waited");

        const IngestSummary summary = manager.summary();
        check(summary.cameras == 5 && summary.streaming == 3 && !summary.is_healthy(),
              "the fleet summary is strict: three of five streaming is not healthy");
        check(summary.frames_read >= 3 && summary.frames_dropped == 0 &&
                  summary.frames_published <= summary.frames_read &&
                  summary.frames_read - summary.frames_published <= summary.cameras,
              "and it aggregates what every camera read and published — nothing dropped, and "
              "the only gap is the at-most-one frame per camera still in flight");
        manager.stop();
    }

    void test_wait_ready_returns_once_every_camera_delivers() {
        Fleet fleet;
        CountingSink sink;
        std::vector<IngestConfig> cameras;
        for (int c = 0; c < 4; ++c) {
            const std::string id = "cam" + std::to_string(c);
            cameras.push_back(a_camera(id));
            fleet.script(id).on_read = [](int) { return 1; };
        }
        IngestManager manager(cameras, sink, fleet.factory());
        manager.start();
        bool ready = true;
        try {
            manager.wait_ready(3000ms, 5ms);
        } catch (const CameraUnavailableError&) {
            ready = false;
        }
        check(ready, "wait_ready returns as soon as every camera has delivered a frame");
        const std::map<std::string, CameraHealth> health = manager.health();
        check(health.size() == 4 && health.at("cam2").frames_read >= 1,
              "and health() is one snapshot per camera, keyed by id");
        manager.stop();
    }

    // =====================================================================================
    // M. The counting sink, and what is left after a stop
    // =====================================================================================

    void test_counting_sink_and_a_quiet_shutdown() {
        Fleet fleet;
        CountingSink sink;
        std::vector<IngestConfig> cameras{a_camera("cam0"), a_camera("cam1")};
        for (const IngestConfig& camera : cameras) {
            fleet.script(camera.camera_id).on_read = [](int) { return 1; };
        }
        IngestManager manager(cameras, sink, fleet.factory());
        manager.start();
        // Waited for per camera, not on the total: one fast camera reaching forty frames says
        // nothing about the other, and a test that stops there is a test of one camera.
        for (int i = 0; i < 1000; ++i) {
            const std::map<std::string, uint64_t> counts = sink.counts();
            if (counts.size() == 2 && counts.at("cam0") >= 20 && counts.at("cam1") >= 20) break;
            std::this_thread::sleep_for(2ms);
        }
        manager.stop();
        const std::map<std::string, uint64_t> counts = sink.counts();
        check(counts.size() == 2 && counts.at("cam0") >= 20 && counts.at("cam1") >= 20,
              "the counting sink keeps one integer per camera, which is what a fairness "
              "assertion reads");
        uint64_t summed = 0;
        for (const auto& [id, count] : counts) summed += count;
        check(summed == sink.total(), "and the per-camera counts add up to the total");
        check(manager.size() == 0 && !manager.contains("cam0"),
              "and after stop() no actor is left running");
    }

    // =====================================================================================
    // L2. Redaction — no credential reaches a log, an error or the health API
    // =====================================================================================
    // `tests/ingest/test_redaction.py`, ported (#33 round 2: the one file with no tests was
    // the one that failed open).

    const std::string kSecret = "s3cr3t-fleet-password";
    const std::string kUri = "rtsp://admin:" + kSecret + "@10.0.0.5/stream";

    bool contains(const std::string& text, const std::string& needle) {
        return text.find(needle) != std::string::npos;
    }

    void test_redact_uri_masks_the_password_and_nothing_else() {
        const std::string out = redact_uri(kUri);
        check(!contains(out, kSecret) && out == "rtsp://admin:***@10.0.0.5/stream",
              "the password is replaced and the rest survives: " + out);
        check(contains(out, "admin"), "the username is kept — it is not the secret");
        check(
            redact_uri("rtsp://u:a@h/s") == redact_uri("rtsp://u:aaaaaaaaaaaaaaaaaaaaaaa@h/s"),
            "the mask does not leak the length");
        for (const std::string quiet :
             {std::string("rtsp://10.0.0.5/stream"), std::string("/data/frames"),
              std::string("file:///data/clip.mp4"), std::string("")}) {
            check(redact_uri(quiet) == quiet, "a uri with no password is untouched: " + quiet);
        }
        check(redact_uri("rtsp://admin:" + kSecret + "@10.0.0.5:8554/s") ==
                  "rtsp://admin:***@10.0.0.5:8554/s",
              "a port survives");
        check(redact_uri("rtsp://admin:secret@") == "<unparseable uri>",
              "a credential with nothing after it is not echoed — fail closed");
    }

    void test_redaction_never_throws_on_hostile_input() {
        // It runs inside error construction and logging; throwing there turns a diagnostic
        // into a second failure on the path that is already failing.
        for (const std::string hostile :
             {std::string("://"), std::string("rtsp://["), std::string("%%%"),
              std::string("rtsp://u:p@"), std::string(1, '\0')}) {
            bool threw = false;
            try {
                (void)redact_uri(hostile);
                (void)redact_in(hostile);
            } catch (...) {
                threw = true;
            }
            check(!threw, "hostile input must not raise");
        }
    }

    void test_the_passwords_that_break_the_easy_parse() {
        // The three shapes that made the first Python implementation fail open. All three
        // reach the health endpoint via SourceOpenError -> last_error, forever, on retry.
        const std::string kSlash = "rtsp://admin:pa/ss@10.0.0.5/stream";
        const std::string kAt = "rtsp://admin:p@ss123@10.0.0.5/stream";
        const std::string kBoth = "rtsp://admin:Ab/c@123@cam.local:554/h264";
        for (const std::string& uri : {kSlash, kAt, kBoth}) {
            const std::string redacted = redact_uri(uri);
            check(!contains(redacted, "pa/ss") && !contains(redacted, "ss123") &&
                      !contains(redacted, "Ab/c") && contains(redacted, "***"),
                  "no fragment of the password survives: " + redacted);
            const std::string host = uri.substr(uri.rfind('@') + 1);
            check(contains(redacted, "admin") &&
                      contains(redacted, host.substr(0, host.find('/'))),
                  "the host survives so the line is still diagnostic: " + redacted);
            for (const std::string prefix :
                 {std::string("[Errno 111] Connection refused: '"),
                  std::string("could not set property \"location\" to \"")}) {
                const std::string redacted_in = redact_in(prefix + uri + "'");
                check(!contains(redacted_in, "pa/ss") && !contains(redacted_in, "ss123") &&
                          !contains(redacted_in, "Ab/c") && contains(redacted_in, "***"),
                      "the same holds embedded in a decoder message: " + redacted_in);
            }
        }
    }

    void test_an_embedded_uri_is_redacted_in_place() {
        const std::string description =
            "rtspsrc location=" + kUri + " latency=200 ! rtph264depay ! appsink";
        const std::string out = redact_in(description);
        check(!contains(out, kSecret) && contains(out, "latency=200"),
              "only the password is replaced: " + out);
    }

    void test_a_scheme_behind_a_numeric_prefix_still_redacts() {
        // #33 round 2, the fail-open: the scheme walk-back consumed the digits and dots of
        // "2.rtsp" and then gave up because '2' is not alpha — Python anchors on the first
        // alpha of the run and redacts. The two planes must fail in the same direction.
        const std::string leaky =
            "could not set property \"location\" to \"2.rtsp://admin:" + kSecret +
            "@10.0.0.5/stream\"";
        const std::string out = redact_in(leaky);
        check(!contains(out, kSecret) && contains(out, "***") && contains(out, "10.0.0.5"),
              "a numeric prefix glued to the scheme must not leak the password: " + out);
        for (const std::string prefix :
             {std::string("+"), std::string("-"), std::string(".")}) {
            const std::string glued = prefix + "rtsp://u:pw@h/s";
            check(!contains(redact_in(glued), "pw"),
                  "a '" + prefix + "' prefix must not leak either");
        }
        check(redact_in("2://u:pw@h") == "2://u:pw@h",
              "an all-digit token before :// is not a scheme and stays untouched");
    }

    void test_config_bounds_match_the_python_plane() {
        // The five bounds pydantic enforces that the struct did not (#33 round 2).
        struct Case {
            const char* what;
            std::function<void(IngestConfig&)> mutate;
        };
        const std::vector<Case> cases = {
            {"width and height must be >= 16",
             [](IngestConfig& c) {
                 c.width = 8;
                 c.height = 8;
             }},
            {"fps must be >= 0", [](IngestConfig& c) { c.fps = -1.0; }},
            {"first_frame_id must be >= 0", [](IngestConfig& c) { c.first_frame_id = -1; }},
            {"latency_ms must be >= 0", [](IngestConfig& c) { c.latency_ms = -1; }},
            {"empty_read_sleep_ms must be >= 0",
             [](IngestConfig& c) { c.empty_read_sleep_ms = -1; }},
        };
        for (const Case& one : cases) {
            IngestConfig config = a_camera("cam0");
            one.mutate(config);
            bool named = false;
            try {
                config.validate();
            } catch (const ConfigError& error) {
                named = contains(error.what(), "cam0") && contains(error.what(), one.what);
            }
            check(named, std::string("refused, naming the camera and the rule: ") + one.what);
        }
    }

    // =====================================================================================
    // M. The manager's lifecycle races (#33 round 1)
    // =====================================================================================

    void test_a_directly_built_actor_names_the_camera_in_its_refusal() {
        FakeScript script;
        CountingSink sink;
        IngestConfig config = a_camera("cam7");
        config.reconnect_factor = 1.0;
        bool named = false;
        try {
            CameraActor actor(config, sink, scripted(script));
            check(false, "a reconnect_factor of 1.0 must refuse at construction");
        } catch (const ConfigError& error) {
            named = std::string(error.what()).find("cam7") != std::string::npos;
        }
        check(named,
              "and the refusal names the camera — the backoff's own anonymous \"factor must "
              "be > 1\" from a fifty-camera fleet is a search where this is an answer");
    }

    void test_a_camera_added_during_stop_never_keeps_running() {
        // `add_camera` starts its actor outside the lock, so a concurrent stop() can strip
        // the map in the window — and the stop request it sent was aimed at a thread that
        // did not exist yet, which start() then cleared. The invariant under every
        // interleaving: a successful add is a *tracked* camera. The failure this pins was
        // worse than a leak: a camera running behind a manager that had forgotten it, with
        // no shutdown left that could ever reach it.
        int orphans = 0;
        for (int round = 0; round < 100; ++round) {
            FakeScript script;
            script.on_read = [](int) { return 1; };
            CountingSink sink;
            IngestManager manager({}, sink, scripted(script));
            std::shared_ptr<CameraActor> added;
            std::thread adder([&] {
                try {
                    added = manager.add_camera(a_camera("cam0"));
                } catch (const ServerStateError&) {
                    // the documented refusal: the fleet forgot the camera mid-add
                }
            });
            if (round % 2 == 0) std::this_thread::yield();
            manager.stop(2000ms);
            adder.join();
            // The orphan this pins is a camera RUNNING behind a manager that forgot it.
            // `added && !contains` alone is not that: on a slow machine the add completes
            // first and the stop then legitimately empties the map with the camera cleanly
            // stopped — 12 of 100 rounds on CI's two-core runner, which is how this
            // assertion's first version failed on main (#34's first run).
            if (added && !manager.contains("cam0") && added->is_running()) ++orphans;
            manager.stop(2000ms);
        }
        check(orphans == 0,
              "100 add-vs-stop races produced " + std::to_string(orphans) +
                  " camera(s) RUNNING behind a manager that forgot them (want 0: a "
                  "successful add is tracked or stopped, never running untracked)");
    }

    void test_the_managers_death_leaks_the_abandoned_rather_than_freeing_them() {
        // The containment invariant, at the moment it is needed: an abandoned actor's
        // detached thread is still standing on the actor's members when the manager dies.
        // Static locals because the leaked thread also touches the script and the sink
        // after this function would otherwise have destroyed them.
        static FakeScript script;
        static CountingSink sink;
        static std::promise<void> gate;
        static std::shared_future<void> opened = gate.get_future().share();
        script.on_read = [](int index) {
            if (index == 0) opened.wait();
            return 1;
        };
        {
            IngestManager manager({}, sink, scripted(script));
            manager.add_camera(a_camera("cam0"));
            for (int i = 0; i < 400 && script.reads.load() == 0; ++i)
                std::this_thread::sleep_for(5ms);
            check(script.reads.load() >= 1, "the camera is parked inside its decode read");
            check(manager.stop(100ms) == 1, "stop() reports the abandonment to its caller");
            check(manager.size() == 0, "the abandoned camera is forgotten by the fleet");
        }  // ~IngestManager — the leak under test: ~vector must NOT free the actor
        gate.set_value();
        bool resumed = false;
        for (int i = 0; i < 600; ++i) {
            if (sink.total() >= 1 || script.closes.load() >= 1) {
                resumed = true;
                break;
            }
            std::this_thread::sleep_for(5ms);
        }
        check(resumed,
              "the detached thread resumed AFTER the manager died and still published its "
              "frame and closed its source — i.e. the actor it stands on was leaked alive, "
              "not freed under it");
        // Give the thread a beat to leave run() before the harness moves on.
        for (int i = 0; i < 600 && script.closes.load() == 0; ++i)
            std::this_thread::sleep_for(5ms);
    }

    void test_a_refused_add_pays_the_abandonment_debt() {
        // #33 round 3: the deadly interleaving is a stop() landing between add_camera's map
        // insert and its start() — the stop signal is aimed at a thread that does not exist
        // yet, start() then clears it, and the re-check has to stop an actor whose do_open
        // is already blocked. If that stop has to DETACH, the throw must not drop the last
        // reference; the actor is parked with the others the destructor deliberately leaks.
        // The window is ~100 ns wide and unreachable by hammering (400 ASan rounds in
        // review landed zero hits), so the manager exposes it as a seam instead.
        static FakeScript script;
        static CountingSink sink;
        static std::promise<void> gate;
        static std::shared_future<void> opened = gate.get_future().share();
        // The lifetime witness (#35 round 1): a weak_ptr taken while the camera is still
        // tracked. Without it the plain -O2 build cannot tell the fix from its absence —
        // the freed actor's memory is not reused before the gate opens, so the detached
        // thread's use-after-free LOOKS like the correct behaviour unless ASan is watching.
        // Expired after the refusal == the throw dropped the last reference; alive == the
        // refusal parked it on abandoned_.
        static std::weak_ptr<CameraActor> parked;
        // The re-check's grace, timed from where it starts (#35 round 1: timing from before
        // add_camera would include this test's own poll loop in the hook below).
        static Clock::time_point recheck_began;
        script.on_open = [](int) { opened.wait(); };
        script.on_read = [](int) { return 1; };

        class StopsInTheWindow : public IngestManager {
          public:
            using IngestManager::IngestManager;

          protected:
            void between_publish_and_start() override {
                // Still tracked here — the stop below is what forgets it.
                parked = actor("cam0");
                // The concurrent stop(): strips the map, signals the not-yet-existing
                // thread, parks nothing (the actor is not joinable yet), reports 0.
                stop(0ms);
            }
            void between_start_and_recheck() override {
                // The other half of the interleaving: the fresh thread must be INSIDE its
                // blocked do_open before the re-check's stop request lands, or it exits
                // cleanly at its first signal check and the safe sub-case is all that runs.
                for (int i = 0; i < 600 && script.opens.load() == 0; ++i)
                    std::this_thread::sleep_for(5ms);
                recheck_began = Clock::now();
            }
        };

        bool refused = false;
        {
            StopsInTheWindow manager({}, sink, scripted(script));
            try {
                manager.add_camera(a_camera("cam0"));
            } catch (const ServerStateError&) {
                refused = true;
            }
            check(refused, "the add is refused, not returned as a camera nobody tracks");
            check(ms_since(recheck_began) < 2000.0,
                  "and the refusal used the short re-check grace, not the full shutdown one");
            check(manager.size() == 0, "the fleet holds nothing");
            check(!parked.expired(),
                  "the actor outlived the refusal: parked on abandoned_, not dropped by the "
                  "throw");
        }  // ~IngestManager — must leak the parked actor, not free it under its thread
        check(!parked.expired(),
              "and it outlived the manager too — the deliberate leak covers the parked");
        gate.set_value();
        bool resumed = false;
        for (int i = 0; i < 600; ++i) {
            if (script.closes.load() >= 1) {
                resumed = true;
                break;
            }
            std::this_thread::sleep_for(5ms);
        }
        check(resumed,
              "the detached thread resumed AFTER the refusal and the manager's death, and "
              "closed its source — the actor it stands on was parked and leaked, not freed "
              "by the throw");
    }

    void test_stop_charges_one_deadline_to_the_fleet_not_one_per_camera() {
        // Five cameras all hung in a decoder read: the 300 ms budget is the *fleet's*, so
        // the shutdown costs one deadline, not five in sequence — the header's "one read
        // timeout rather than fifty" made literal.
        static FakeScript script;
        static CountingSink sink;
        static std::promise<void> gate;
        static std::shared_future<void> opened = gate.get_future().share();
        script.on_read = [](int) {
            opened.wait();
            return 1;
        };
        std::vector<IngestConfig> cameras;
        for (int i = 0; i < 5; ++i) cameras.push_back(a_camera("cam" + std::to_string(i)));
        {
            IngestManager manager(cameras, sink, scripted(script));
            manager.start();
            for (int i = 0; i < 400 && script.reads.load() < 5; ++i)
                std::this_thread::sleep_for(5ms);
            check(script.reads.load() >= 5, "all five cameras are parked inside a read");
            const auto start = Clock::now();
            const size_t abandoned = manager.stop(300ms);
            const double waited = ms_since(start);
            check(abandoned == 5, "all five are reported abandoned, not silently detached");
            check(waited < 1200.0, "five hung cameras cost one 300 ms deadline (" +
                                       std::to_string(waited) +
                                       " ms), not five in sequence (1500+ ms)");
        }
        gate.set_value();
        for (int i = 0; i < 600 && script.closes.load() < 5; ++i)
            std::this_thread::sleep_for(5ms);
    }

}  // namespace

int main() {
    test_stop_signal();

    test_backoff_sequence();
    test_backoff_jitter_and_cap();
    test_backoff_survives_the_night();
    test_backoff_refuses_nonsense();

    test_pacer();

    test_frame_counter();

    test_source_contract();
    test_registry();

    test_actor_backoff_sequence();
    test_a_connect_does_not_clear_a_failure_but_a_frame_does();
    test_a_missing_decode_runtime_is_fatal();
    test_degraded_then_unhealthy_and_sticky();
    test_a_broken_read_rebuilds_the_source();
    test_an_untyped_sink_exception_does_not_kill_the_camera();

    test_the_empty_read_budget();
    test_a_frame_restarts_the_empty_run();
    test_an_exhausted_source_finishes();

    test_a_full_sink_is_a_drop_charged_to_this_camera();
    test_a_closed_sink_finishes_the_actor();
    test_an_accepting_sink_publishes_everything();

    test_actor_lifecycle();
    test_stop_interrupts_a_thirty_second_backoff();
    test_health_is_safe_to_read_from_any_thread();

    test_health_reports_a_recent_rate();

    test_the_manager_validates_before_it_starts_a_thread();
    test_a_disabled_camera_is_not_in_the_fleet();
    test_adding_and_removing_a_camera_while_the_fleet_runs();
    test_stop_signals_everybody_before_it_joins_anybody();
    test_wait_ready_names_the_silent_cameras();
    test_wait_ready_returns_once_every_camera_delivers();

    test_counting_sink_and_a_quiet_shutdown();

    test_redact_uri_masks_the_password_and_nothing_else();
    test_redaction_never_throws_on_hostile_input();
    test_the_passwords_that_break_the_easy_parse();
    test_an_embedded_uri_is_redacted_in_place();
    test_a_scheme_behind_a_numeric_prefix_still_redacts();
    test_config_bounds_match_the_python_plane();

    test_a_directly_built_actor_names_the_camera_in_its_refusal();
    test_a_camera_added_during_stop_never_keeps_running();
    test_the_managers_death_leaks_the_abandoned_rather_than_freeing_them();
    test_a_refused_add_pays_the_abandonment_debt();
    test_stop_charges_one_deadline_to_the_fleet_not_one_per_camera();

    std::printf("%d checks, %d failure(s), %d skipped\n", checks, failures, skips);
    return failures == 0 ? 0 : 1;
}
