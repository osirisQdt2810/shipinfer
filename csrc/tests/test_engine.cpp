// The instance thread and the model over its dispatcher, tested the way `tests/engine/` tests
// the Python ones: with an engine that has no device. `IdentityEngine` keeps its bindings in
// host memory and copies input rows to output rows, so a wrong span shows up as the wrong
// caller's numbers — which is the one thing the scatter must never get wrong.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/backends/engine_api.h"
#include "shipinfer/core/types.h"
#include "shipinfer/engine/model.h"
#include "shipinfer/engine/request.h"
#include "shipinfer/scheduling/policies/join_shortest_queue.h"
#include "shipinfer/scheduling/policies/round_robin.h"

namespace {

    using namespace shipinfer;
    using namespace std::chrono_literals;

    int failures = 0;
    int checks = 0;
    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        }
    }

    // Host memory, identity network, optional failure injection and per-execute latency.
    class IdentityEngine : public Engine {
      public:
        IdentityEngine(Device device, int max_batch, size_t width,
                       std::chrono::milliseconds latency = 0ms)
            : device_(device),
              max_batch_(max_batch),
              width_(width),
              latency_(latency),
              input_(static_cast<size_t>(max_batch) * width, 0.f),
              output_(static_cast<size_t>(max_batch) * width, 0.f) {}
        Device device() const override { return device_; }
        int max_batch() const override { return max_batch_; }
        size_t input_row_elems() const override { return width_; }
        size_t output_row_elems(size_t = 0) const override { return width_; }
        void write_rows(size_t row_offset, const float* src, size_t rows, Device) override {
            std::memcpy(input_.data() + row_offset * width_, src,
                        rows * width_ * sizeof(float));
        }
        void execute(int rows) override {
            ++executes;
            last_rows = rows;
            if (fail_next) {
                fail_next = false;
                throw BackendError("injected engine failure");
            }
            if (latency_.count() > 0) std::this_thread::sleep_for(latency_);
            std::memcpy(output_.data(), input_.data(),
                        static_cast<size_t>(rows) * width_ * sizeof(float));
        }
        const float* output(size_t = 0) const override { return output_.data(); }

        std::atomic<int> executes{0};
        std::atomic<int> last_rows{0};
        std::atomic<bool> fail_next{false};

      private:
        Device device_;
        int max_batch_;
        size_t width_;
        std::chrono::milliseconds latency_;
        std::vector<float> input_;
        std::vector<float> output_;
    };

    // A backend with TWO outputs, so the widened contract is exercised rather than only
    // compiled. The second is the first negated, which tells "every output was scattered"
    // from "the first one twice" -- and both are `rows` long, which is the contract's own
    // rule: one row index selects one object's slice of all of them.
    class TwoOutputEngine : public IdentityEngine {
      public:
        TwoOutputEngine(Device device, int max_batch, size_t width)
            : IdentityEngine(device, max_batch, width), width_(width) {}
        size_t outputs() const override { return 2; }
        std::string output_name(size_t index) const override {
            return index == 0 ? "rows" : "negated";
        }
        std::vector<int64_t> output_dims(size_t index) const override {
            (void)index;
            return {static_cast<int64_t>(width_ / 2), 2};
        }
        const float* output(size_t index) const override {
            const float* first = IdentityEngine::output(0);
            if (index == 0) return first;
            negated_.assign(first, first + width_ * static_cast<size_t>(max_batch()));
            for (float& value : negated_) value = -value;
            return negated_.data();
        }

      private:
        size_t width_;
        mutable std::vector<float> negated_;
    };

    InferenceRequest a_request(const std::string& camera, int64_t frame,
                               const std::vector<float>& payload, size_t width) {
        InferenceRequest request;
        request.model_name = "m";
        request.tag = FrameTag{camera, frame, 0};
        request.data = payload.data();
        request.rows = payload.size() / width;
        request.row_elems = width;
        return request;
    }

    // -- the instance
    // ----------------------------------------------------------------------------

    void test_every_output_is_scattered_and_named() {
        // The widened contract (CSRC-SEGMENT-FOLD-MISSING): a segmentation engine answers
        // detection rows AND a prototype bank, so a response that carried one output had
        // nowhere for the second to arrive and the mask fold could not be written at all.
        auto engine = std::make_unique<TwoOutputEngine>(Device::cuda(0), 4, 2);
        ModelInstance instance("m:0", std::move(engine), BatchWindow(4, 0), 16);
        instance.start();
        check(instance.wait_ready(2s), "the instance becomes ready");
        const std::vector<float> payload{1.f, 2.f, 3.f, 4.f};
        WorkItem item(a_request("cam", 7, payload, 2));
        auto future = item.future();
        check(instance.enqueue(std::move(item)) == PutStatus::Accepted, "accepted");
        const InferenceResponse response = future.get();

        check(response.outputs.size() == 2, "both outputs come back");
        check(response.first().data == payload, "the first is the answer it always was");
        check(response.named("negated") != nullptr, "and the second is reachable BY NAME");
        check(response.named("negated")->data == std::vector<float>{-1.f, -2.f, -3.f, -4.f},
              "with its own values, not a second copy of the first");
        check(response.named("nosuch") == nullptr, "an output the engine has not is null");
        check(response.row_elems() == 2 && response.row(1)[0] == 3.f,
              "and the single-output accessors still mean the first output");
        check(response.first().dims == std::vector<int64_t>{1, 2},
              "the artefact's own per-row shape travels, which a flat width cannot carry");
        instance.stop();
    }

    void test_an_instance_answers_with_the_rows_it_was_given() {
        auto engine = std::make_unique<IdentityEngine>(Device::cuda(0), 4, 2);
        ModelInstance instance("m:0", std::move(engine), BatchWindow(4, 0), 16);
        instance.start();
        check(instance.wait_ready(2s), "the instance becomes ready");
        std::vector<float> payload{1.f, 2.f, 3.f, 4.f};  // two rows of two
        WorkItem item(a_request("cam", 7, payload, 2));
        auto future = item.future();
        check(instance.enqueue(std::move(item)) == PutStatus::Accepted, "accepted");
        const InferenceResponse response = future.get();
        check(response.rows == 2 && response.first().data == payload,
              "the identity engine returns the same rows");
        check(response.tag.camera_id == "cam" && response.tag.frame_id == 7,
              "the tag travels unchanged");
        check(response.executed_on == Device::cuda(0),
              "the response names the device it ran on");
        check(response.timings.completed_ns >= response.timings.batched_ns &&
                  response.timings.batched_ns >= response.timings.queued_ns,
              "timings are ordered queued <= batched <= completed");
        instance.stop();
    }

    void test_two_requests_are_batched_and_scattered_to_their_own_callers() {
        // A window of 10 ms: both arrive inside it and leave in one execute; the scatter must
        // hand each caller its own rows.
        auto engine_ptr = std::make_unique<IdentityEngine>(Device::cuda(0), 8, 1);
        IdentityEngine* engine = engine_ptr.get();
        ModelInstance instance("m:0", std::move(engine_ptr), BatchWindow(8, 50000), 16);
        instance.start();
        instance.wait_ready(2s);
        std::vector<float> a{10.f, 11.f, 12.f};
        std::vector<float> b{20.f};
        WorkItem ia(a_request("a", 1, a, 1));
        WorkItem ib(a_request("b", 1, b, 1));
        auto fa = ia.future();
        auto fb = ib.future();
        instance.enqueue(std::move(ia));
        instance.enqueue(std::move(ib));
        const InferenceResponse ra = fa.get();
        const InferenceResponse rb = fb.get();
        check(ra.first().data == a && rb.first().data == b,
              "each caller gets exactly its own rows back");
        check(engine->executes.load() == 1 && engine->last_rows.load() == 4,
              "one execute of four rows: the two requests were batched");
        instance.stop();
    }

    void test_a_failing_engine_fails_the_batch_and_the_instance_serves_the_next() {
        auto engine_ptr = std::make_unique<IdentityEngine>(Device::cuda(0), 4, 1);
        IdentityEngine* engine = engine_ptr.get();
        ModelInstance instance("m:0", std::move(engine_ptr), BatchWindow(4, 0), 16);
        instance.start();
        instance.wait_ready(2s);
        engine->fail_next = true;
        std::vector<float> one{1.f};
        WorkItem doomed(a_request("a", 1, one, 1));
        auto fd = doomed.future();
        instance.enqueue(std::move(doomed));
        bool failed = false;
        try {
            fd.get();
        } catch (const BackendError&) {
            failed = true;
        }
        check(failed, "the batch's caller hears the engine's error");
        check(instance.is_ready(), "one bad batch is not a dead instance");
        WorkItem next(a_request("a", 2, one, 1));
        auto fn = next.future();
        instance.enqueue(std::move(next));
        check(fn.get().first().data == one, "the next batch is served");
        check(instance.stats().failed_batches == 1, "and the failure was counted");
        instance.stop();
    }

    void test_a_request_wider_than_the_engine_is_refused_not_misread() {
        auto engine = std::make_unique<IdentityEngine>(Device::cuda(0), 4, 2);
        ModelInstance instance("m:0", std::move(engine), BatchWindow(4, 0), 16);
        instance.start();
        instance.wait_ready(2s);
        std::vector<float> three{1.f, 2.f, 3.f};
        WorkItem item(a_request("a", 1, three, 3));  // rows of three into an engine of two
        auto future = item.future();
        instance.enqueue(std::move(item));
        bool refused = false;
        try {
            future.get();
        } catch (const BackendError&) {
            refused = true;
        }
        check(refused,
              "a row shape the engine does not take is an error, not a silent misalignment");
        instance.stop();
    }

    void test_stop_fails_everything_queued_and_in_flight() {
        // A slow engine holds one batch in flight while a second waits in the queue; stop()
        // must answer both — cancelled — rather than strand either future.
        auto engine_ptr = std::make_unique<IdentityEngine>(Device::cuda(0), 1, 1, 300ms);
        ModelInstance instance("m:0", std::move(engine_ptr), BatchWindow(1, 0), 16);
        instance.start();
        instance.wait_ready(2s);
        std::vector<float> one{1.f};
        WorkItem first(a_request("a", 1, one, 1));
        WorkItem second(a_request("a", 2, one, 1));
        auto f1 = first.future();
        auto f2 = second.future();
        instance.enqueue(std::move(first));
        instance.enqueue(std::move(second));
        std::this_thread::sleep_for(50ms);  // the first is in flight, the second queued
        instance.stop();
        int answered = 0;
        for (auto* f : {&f1, &f2}) {
            try {
                (void)f->get();
                ++answered;  // the in-flight one may finish before the stop lands; that is an
                             // answer too
            } catch (const RequestCancelledError&) {
                ++answered;
            }
        }
        check(answered == 2, "every accepted request is answered on stop, none stranded");
        check(!instance.is_ready(), "a stopped instance is not ready");
        check(instance.enqueue(WorkItem(a_request("a", 3, one, 1))) == PutStatus::Closed,
              "a stopped instance refuses new work");
    }

    void test_a_bind_that_fails_is_a_failed_start_not_a_hang() {
        auto engine = std::make_unique<IdentityEngine>(Device::cuda(3), 4, 1);
        ModelInstance instance("m:3", std::move(engine), BatchWindow(4, 0), 16,
                               Overflow::Reject,
                               [](Device) { throw BackendError("no such device"); });
        instance.start();
        check(!instance.wait_ready(2s), "a failed bind is reported, not waited out");
        check(!instance.is_ready() && instance.start_error() != nullptr,
              "and the error is kept");
        instance.stop();
    }

    // -- the model
    // -------------------------------------------------------------------------------

    std::unique_ptr<Model> two_instance_model(std::vector<IdentityEngine*>& engines) {
        std::vector<std::unique_ptr<ModelInstance>> instances;
        for (int d = 0; d < 2; ++d) {
            auto engine = std::make_unique<IdentityEngine>(Device::cuda(d), 4, 1);
            engines.push_back(engine.get());
            instances.push_back(std::make_unique<ModelInstance>(
                "m:" + std::to_string(d), std::move(engine), BatchWindow(4, 0), 2));
        }
        return std::make_unique<Model>("m", std::move(instances),
                                       std::make_unique<JoinShortestQueuePolicy>());
    }

    void test_the_model_places_by_policy_and_answers() {
        std::vector<IdentityEngine*> engines;
        auto model = two_instance_model(engines);
        model->start(2s);
        check(model->is_ready(), "the model is ready when an instance is");
        std::vector<float> one{5.f};
        auto future = model->infer(a_request("cam", 1, one, 1));
        const InferenceResponse response = future.get();
        check(response.first().data == one, "the answer comes back through the model");
        check(
            response.executed_on == Device::cuda(0) || response.executed_on == Device::cuda(1),
            "on one of the model's devices");
        model->stop();
    }

    void test_a_request_nothing_will_take_comes_back_as_a_failed_future() {
        std::vector<IdentityEngine*> engines;
        auto model = two_instance_model(engines);
        // Not started: no instance is ready.
        std::vector<float> one{5.f};
        auto future = model->infer(a_request("cam", 1, one, 1));
        bool failed = false;
        try {
            future.get();
        } catch (const ServerStateError&) {
            failed = true;
        }
        check(failed,
              "no ready instance: the future carries ServerStateError, nobody waits forever");
    }

    void test_the_model_spills_when_the_policy_choice_is_full() {
        // Queues of capacity 2 with a slow engine: JSQ picks the shorter, and once both are
        // full the caller hears QueueFullError rather than being dropped.
        std::vector<std::unique_ptr<ModelInstance>> instances;
        for (int d = 0; d < 2; ++d) {
            auto engine = std::make_unique<IdentityEngine>(Device::cuda(d), 1, 1, 200ms);
            instances.push_back(std::make_unique<ModelInstance>(
                "m:" + std::to_string(d), std::move(engine), BatchWindow(1, 0), 1));
        }
        Model model("m", std::move(instances), std::make_unique<RoundRobinPolicy>());
        model.start(2s);
        std::vector<float> one{1.f};
        std::vector<std::future<InferenceResponse>> futures;
        for (int i = 0; i < 6; ++i) futures.push_back(model.infer(a_request("cam", i, one, 1)));
        int refused = 0, answered = 0;
        for (auto& f : futures) {
            try {
                (void)f.get();
                ++answered;
            } catch (const QueueFullError&) {
                ++refused;
            }
        }
        check(refused >= 1,
              "once every instance's queue is full the caller hears QueueFullError");
        check(answered >= 2, "and the ones that fit were served");
        model.stop();
    }

    // -- the two clocks a frame carries
    // ---------------------------------------------------

    void test_a_deadline_built_from_the_steady_stamp_expires() {
        // The arithmetic every stage will do once frames carry a budget: deadline = when the
        // frame was captured + how long it is worth spending on it.
        FrameTag tag;
        tag.camera_id = "cam";
        tag.captured_ns = monotonic_ns();
        tag.captured_unix_ns = unix_ns();

        InferenceRequest request;
        request.tag = tag;
        request.deadline_ns = tag.captured_ns + 50 * 1000 * 1000;  // 50 ms
        check(!request.is_expired(monotonic_ns()),
              "a 50 ms budget on a frame stamped now has not expired now");
        check(request.is_expired(tag.captured_ns + 60 * 1000 * 1000),
              "and has expired 60 ms later — the deadline is built from the STEADY stamp");
    }

    void test_the_same_deadline_from_the_wall_stamp_would_never_expire() {
        // The latent bug this pins. A wall-clock stamp is ~1.7e18 ns and the monotonic clock a
        // deadline is compared against is ~1e13, so `captured_unix_ns + budget` lands roughly
        // 54 years out and NOTHING would ever expire — the queue would happily execute frames
        // an hour late, and no counter would say so. Two fields, two names, one of them
        // documented as never being deadline arithmetic.
        FrameTag tag;
        tag.captured_ns = monotonic_ns();
        tag.captured_unix_ns = unix_ns();

        InferenceRequest wrong;
        wrong.tag = tag;
        wrong.deadline_ns = tag.captured_unix_ns + 50 * 1000 * 1000;
        check(!wrong.is_expired(monotonic_ns()) &&
                  !wrong.is_expired(tag.captured_ns + 86400LL * 365 * 1000000000LL),
              "a deadline built from captured_unix_ns is still unexpired a year later, which "
              "is why the two clocks are two fields");
    }

}  // namespace

int main() {
    test_an_instance_answers_with_the_rows_it_was_given();
    test_every_output_is_scattered_and_named();
    test_two_requests_are_batched_and_scattered_to_their_own_callers();
    test_a_failing_engine_fails_the_batch_and_the_instance_serves_the_next();
    test_a_request_wider_than_the_engine_is_refused_not_misread();
    test_stop_fails_everything_queued_and_in_flight();
    test_a_bind_that_fails_is_a_failed_start_not_a_hang();
    test_the_model_places_by_policy_and_answers();
    test_a_request_nothing_will_take_comes_back_as_a_failed_future();
    test_the_model_spills_when_the_policy_choice_is_full();
    test_a_deadline_built_from_the_steady_stamp_expires();
    test_the_same_deadline_from_the_wall_stamp_would_never_expire();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
