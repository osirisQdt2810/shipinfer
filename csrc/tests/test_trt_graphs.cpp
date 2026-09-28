// A replayed CUDA graph against the plain enqueue, through the real engine and the real
// adapter.
//
// Two instances of one plan on one profile, used in turn: one enqueues, one replays the graphs
// `prepare` captured. Every output must match to the bit, at captured sizes and at one that is
// not, and a second batch at the same size must answer ITS input -- a graph that recorded a
// stale copy would repeat the first. Container tier; no engine is a skip, as in
// `test_fold_wiring.cpp`.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/backends/tensorrt/adapter.h"
#include "shipinfer/core/platform.h"

namespace {

    using namespace shipinfer;

    int checks = 0;
    int failures = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (condition) return;
        ++failures;
        std::printf("FAIL: %s\n", what.c_str());
    }

    std::string plan_path() {
        const char* repository = std::getenv("SHIPINFER_TEST_REPOSITORY");
        const std::string root = repository != nullptr ? repository : "/work/model_repository";
        return root + "/ship_detector/1/model.plan";
    }

    // Deterministic and different per `seed`, so two batches at one size cannot agree by luck.
    std::vector<float> a_batch(size_t rows, size_t width, size_t seed) {
        std::vector<float> input(rows * width);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = static_cast<float>((i * 7919 + 13 + seed * 104729) % 251) / 251.0f;
        return input;
    }

    // How many floats of every advertised output differ, over `rows` rows.
    size_t differing(const Engine& a, const Engine& b, int rows) {
        size_t count = 0;
        for (size_t o = 0; o < a.outputs(); ++o) {
            const size_t elems = a.output_row_elems(o) * static_cast<size_t>(rows);
            for (size_t i = 0; i < elems; ++i) {
                if (a.output(o)[i] != b.output(o)[i]) ++count;
            }
        }
        return count;
    }

    void a_replay_answers_what_the_enqueue_answers() {
        std::shared_ptr<TrtEngine> engine;
        try {
            engine = TrtEngine::load(plan_path(), 0);
        } catch (const std::exception& error) {
            std::printf("SKIP: no engine at %s (%s)\n", plan_path().c_str(), error.what());
            return;
        }
        const int max = engine->max_batch();
        // A static plan runs one size, padded; a dynamic one leaves 2 uncaptured on purpose.
        const std::vector<int> captured =
            engine->is_static() ? std::vector<int>{max} : std::vector<int>{1, 3, max};
        const std::vector<int> run =
            engine->is_static() ? std::vector<int>{max} : std::vector<int>{1, 2, 3, max};

        auto graphed_instance = std::make_unique<TrtInstance>(engine, 0);
        TrtInstance* graphs = graphed_instance.get();
        graphs->capture_graphs(captured);
        TrtEngineAdapter plain(std::make_unique<TrtInstance>(engine, 0));
        TrtEngineAdapter graphed(std::move(graphed_instance));
        graphed.prepare();
        check(graphs->graph_captures() == captured.size() && graphs->graph_failures() == 0,
              "prepare captured every size asked for: " +
                  std::to_string(graphs->graph_captures()) + " of " +
                  std::to_string(captured.size()) + ", " +
                  std::to_string(graphs->graph_failures()) + " failed");

        const size_t width = engine->inputs().front().elements_per_row();
        uint64_t replays = 0;
        for (int rows : run) {
            for (size_t seed : {1u, 2u}) {
                const std::vector<float> input =
                    a_batch(static_cast<size_t>(rows), width, seed);
                for (Engine* adapter :
                     {static_cast<Engine*>(&plain), static_cast<Engine*>(&graphed)}) {
                    adapter->write_rows(0, input.data(), static_cast<size_t>(rows),
                                        Device::cpu());
                    adapter->execute(rows);
                }
                const bool is_captured =
                    std::find(captured.begin(), captured.end(), rows) != captured.end();
                if (is_captured) ++replays;
                check(differing(plain, graphed, rows) == 0,
                      std::to_string(rows) + " row(s), batch " + std::to_string(seed) + ", " +
                          (is_captured ? "replayed" : "enqueued") +
                          ": every output matches the plain enqueue to the bit");
            }
        }
        check(graphs->graph_replays() == replays,
              "a captured size replays and an uncaptured one does not: " +
                  std::to_string(graphs->graph_replays()) + " replays, " +
                  std::to_string(replays) + " expected");
        check(graphs->launch_ns() > 0, "and the launch time is counted");
    }

    void a_size_outside_the_plan_is_refused() {
        std::shared_ptr<TrtEngine> engine;
        try {
            engine = TrtEngine::load(plan_path(), 0);
        } catch (const std::exception& error) {
            std::printf("SKIP: no engine at %s (%s)\n", plan_path().c_str(), error.what());
            return;
        }
        TrtInstance instance(engine, 0);
        for (int size : {0, engine->max_batch() + 1}) {
            std::string message;
            try {
                instance.capture_graphs({size});
            } catch (const BackendError& error) {
                message = error.what();
            }
            check(message.find("outside [1, " + std::to_string(engine->max_batch()) + "]") !=
                      std::string::npos,
                  "size " + std::to_string(size) + " is refused with the plan's range: " +
                      (message.empty() ? "(no throw)" : message));
        }
    }

}  // namespace

int main() {
    GPU_CHECK(gpuSetDevice(0));
    a_replay_answers_what_the_enqueue_answers();
    a_size_outside_the_plan_is_refused();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
