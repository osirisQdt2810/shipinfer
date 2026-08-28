// The instrument: a decode path with no camera, no network and no decoder — only a script.
//
// The counterpart of `benchmarks/parity/drive_python.py`, and the one thing in this harness
// that is written twice. That duplication is its own risk, so every call emits a `source_*`
// record: the two scripts drifting apart shows up in the source-event stream before it can
// be mistaken for a divergence in the actor.
//
// Everything is observed from the **actor's own thread**, at the source and sink calls, so
// the trace records that thread's program order rather than a poll racing it. NOT registered
// in `SOURCES()`: it is injected as a `SourceFactory`, the way `test_ingest.cpp`'s own fake
// already is, so no binary's registry gains a source production would not have.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
#include "tests/parity_scenario.h"
#include "tests/parity_trace.h"

namespace shipinfer::parity {

    // One camera's trace, and the counters that turn its script into call outcomes.
    //
    // Touched only by that camera's actor thread (the driver reads it once, at the end),
    // which is why nothing here but `exhausted` is atomic: one actor per camera for the
    // actor's whole life is what ADR-002 buys.
    class CameraRecorder {
      public:
        CameraRecorder(CameraScript script, const Scenario& scenario, ParityTraceWriter& writer)
            : script_(std::move(script)), scenario_(scenario), writer_(writer) {}

        const CameraScript& script() const { return script_; }
        bool exhausted() const { return exhausted_.load(); }
        void exhaust() { exhausted_.store(true); }
        void adopt(std::shared_ptr<CameraActor> actor) { actor_ = std::move(actor); }

        void emit(const std::string& kind, std::vector<int64_t> numbers = {},
                  std::vector<std::string> text = {}) {
            writer_.record(kind, script_.camera_id, std::move(numbers), std::move(text));
        }

        size_t step(const std::string& what) { return counts_[what]++; }

        // Emit whatever the actor has done to itself since the last call boundary. Called
        // first at every source and sink hook, so a state change and the retries that caused
        // it land before the record of the call that observed them.
        void observe() {
            if (!actor_) return;
            const CameraHealth health = actor_->health();
            const std::string state = to_string(health.state);
            if (state != state_) {
                emit("state", {}, {state_, state});
                state_ = state;
            }
            while (health.consecutive_failures > failures_) {
                emit("retry", {static_cast<int64_t>(failures_), scenario_.peek_us(failures_)});
                ++failures_;
            }
            failures_ = std::min(failures_, health.consecutive_failures);
        }

      private:
        CameraScript script_;
        const Scenario& scenario_;
        ParityTraceWriter& writer_;
        std::shared_ptr<CameraActor> actor_;
        std::atomic<bool> exhausted_{false};
        std::map<std::string, size_t> counts_;
        std::string state_ = "idle";
        uint64_t failures_ = 0;
    };

    class ScriptedSource : public FrameSource {
      public:
        ScriptedSource(const IngestConfig& config, FrameCounter& counter, StopSignal& stop,
                       CameraRecorder& recorder)
            : FrameSource(config, counter, stop), recorder_(recorder) {}

        bool is_exhausted() const override { return recorder_.exhausted(); }

      protected:
        void do_open() override {
            recorder_.observe();
            const size_t index = recorder_.step("open");
            const Outcome& outcome = recorder_.script().open_at(index);
            recorder_.emit("source_open", {static_cast<int64_t>(index)}, {outcome.what});
            if (outcome.what == "SourceOpenError") {
                throw SourceOpenError(camera_id(), config().uri, outcome.detail);
            }
            if (outcome.what == "SourceUnavailableError") {
                throw SourceUnavailableError("scripted", outcome.detail);
            }
            set_format(kHeight, kWidth, 0.0);
        }

        std::optional<HostFrame> do_read() override {
            recorder_.observe();
            const size_t index = recorder_.step("read");
            const Outcome& outcome = recorder_.script().read_at(index);
            recorder_.emit("source_read", {static_cast<int64_t>(index)}, {outcome.what});
            if (outcome.what == "FrameDecodeError") {
                throw FrameDecodeError(camera_id(), outcome.detail);
            }
            if (outcome.what == "exhaust") recorder_.exhaust();
            if (outcome.what != "frame") return std::nullopt;
            HostFrame image;
            image.pixels = pixels().data();
            image.height = kHeight;
            image.width = kWidth;
            return image;
        }

        void do_close() override {
            recorder_.observe();
            recorder_.emit("source_close", {static_cast<int64_t>(recorder_.step("close"))});
        }

      private:
        // Every scripted frame is this size, and no pixel is ever looked at: the parity
        // property is which frames were produced and what happened to them.
        static constexpr int kHeight = 4;
        static constexpr int kWidth = 6;
        static const std::vector<uint8_t>& pixels() {
            static const std::vector<uint8_t> buffer(kHeight * kWidth * 3, 0);
            return buffer;
        }

        CameraRecorder& recorder_;
    };

    // The consumer, scripted per camera: accept, refuse, or say the consumer has gone.
    //
    // Records the frame it was offered and the refusal it answered with. What the *actor*
    // then charged to the camera is not read here at all — it is in the final `health`
    // record, so a plane that forgot to count a drop diverges there rather than being
    // covered for.
    class RecordingSink : public FrameSink {
      public:
        explicit RecordingSink(
            std::map<std::string, std::unique_ptr<CameraRecorder>>& by_camera)
            : by_camera_(by_camera) {}

        void put(Frame&& frame) override {
            CameraRecorder& recorder = *by_camera_.at(frame.tag.camera_id);
            recorder.observe();
            recorder.emit("frame", {frame.tag.frame_id});
            const std::string outcome = recorder.script().sink_at(recorder.step("sink"));
            if (outcome == "full") {
                recorder.emit("drop", {}, {"sink_full"});
                throw QueueFullError("parity:" + frame.tag.camera_id, 1, 1);
            }
            if (outcome == "closed") {
                recorder.emit("drop", {}, {"sink_closed"});
                throw RequestCancelledError("parity sink for " + frame.tag.camera_id +
                                            " is closed");
            }
        }

      private:
        std::map<std::string, std::unique_ptr<CameraRecorder>>& by_camera_;
    };

    // The camera this scenario declares, as the resolved record this plane takes.
    inline IngestConfig camera_config(const Scenario& scenario, const CameraScript& script) {
        IngestConfig config;
        config.camera_id = script.camera_id;
        config.uri = "scripted://" + script.camera_id + "/stream";
        config.source = "scripted";
        config.enabled = script.enabled;
        config.empty_read_sleep_ms = scenario.int_setting("empty_read_sleep_ms");
        config.reconnect_initial_ms = scenario.int_setting("reconnect_initial_ms");
        config.reconnect_max_ms = scenario.int_setting("reconnect_max_ms");
        config.reconnect_factor = scenario.double_setting("reconnect_factor");
        config.reconnect_jitter = scenario.double_setting("reconnect_jitter");
        if (scenario.settings.count("empty_reads_before_reconnect")) {
            config.empty_reads_before_reconnect =
                scenario.int_setting("empty_reads_before_reconnect");
        }
        if (scenario.settings.count("failures_before_unhealthy")) {
            config.failures_before_unhealthy =
                scenario.int_setting("failures_before_unhealthy");
        }
        return config;
    }

    // Builds the scripted source for whichever camera the actor is, and hands the recorder
    // the actor itself — on the actor's own thread and before its first hook. The manager
    // publishes an actor into its map before it starts the thread, while the driver only
    // learns of it after `start()` returns: a window in which the first state change would
    // have gone unobserved on some runs and not on others.
    //
    // `resolve` rather than an `IngestManager&` because the manager is constructed WITH this
    // factory, so the factory cannot hold it yet — the caller points the resolver at the
    // manager after construction and before `start()`, which is the only window that matters.
    using ActorResolver = std::function<std::shared_ptr<CameraActor>(const std::string&)>;

    inline SourceFactory scripted_factory(
        ActorResolver resolve,
        std::map<std::string, std::unique_ptr<CameraRecorder>>& by_camera) {
        return [resolve = std::move(resolve), &by_camera](
                   const IngestConfig& config, FrameCounter& counter,
                   StopSignal& stop) -> std::unique_ptr<FrameSource> {
            CameraRecorder& recorder = *by_camera.at(config.camera_id);
            try {
                if (std::shared_ptr<CameraActor> actor = resolve(config.camera_id)) {
                    recorder.adopt(std::move(actor));
                }
            } catch (const ConfigError&) {  // already forgotten: keep whatever we had
            }
            return std::make_unique<ScriptedSource>(config, counter, stop, recorder);
        };
    }

}  // namespace shipinfer::parity
