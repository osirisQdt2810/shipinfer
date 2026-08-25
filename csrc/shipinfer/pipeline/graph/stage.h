// The stage contract — `pipeline/graph/stage.py`: a stage declares what it consumes, requires
// and produces, and the graph decides from the frame's state whether it runs, in declared
// order.
//
// A stage that raises does **not** end the frame: its outputs never become available, so its
// branch is skipped and every other branch continues (the graph's business, not the stage's).
// A `ModelStage` submits one request to a `Model` and blocks on the future; the concurrency
// comes from the worker pool and the model's batching across every frame in flight, not from
// this call.
#pragma once

#include <chrono>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include "shipinfer/pipeline/graph/state.h"

namespace shipinfer {

    enum class StageStatus { Ran, Skipped, Failed };

    inline const char* to_string(StageStatus status) {
        switch (status) {
            case StageStatus::Ran:
                return "ran";
            case StageStatus::Skipped:
                return "skipped";
            case StageStatus::Failed:
                return "failed";
        }
        return "?";
    }

    struct StageOutcome {
        std::string stage;
        StageStatus status = StageStatus::Skipped;
        size_t rows = 0;
        std::string error;  // empty unless Failed
        int64_t elapsed_us = 0;
        bool ran() const { return status == StageStatus::Ran; }
    };

    class Stage {
      public:
        Stage(std::string name, std::vector<std::string> consumes,
              std::vector<std::string> requires, std::vector<std::string> produces)
            : name_(std::move(name)),
              consumes_(std::move(consumes)),
              requires_(std::move(requires)),
              produces_(std::move(produces)) {}
        virtual ~Stage() = default;

        const std::string& name() const { return name_; }
        // Names read at all (the graph releases them after the last reader).
        const std::vector<std::string>& consumes() const { return consumes_; }
        // Names that must be present *and non-empty* for the stage to run — the conditional
        // branch: no ship crops, no ship segmenter, and that is a skip, not a failure.
        const std::vector<std::string>& requires() const { return requires_; }
        const std::vector<std::string>& produces() const { return produces_; }

        // Template method: times the work, turns an exception into a Failed outcome carrying
        // its message, never lets one escape into the graph.
        StageOutcome run(FrameState& state) {
            const auto start = std::chrono::steady_clock::now();
            StageOutcome outcome;
            outcome.stage = name_;
            try {
                outcome.rows = do_run(state);
                outcome.status = StageStatus::Ran;
            } catch (const std::exception& error) {
                outcome.status = StageStatus::Failed;
                outcome.error = error.what();
            }
            outcome.elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
            return outcome;
        }

      protected:
        // Do the work and mutate `state`. Returns the number of rows produced.
        virtual size_t do_run(FrameState& state) = 0;

      private:
        std::string name_;
        std::vector<std::string> consumes_;
        std::vector<std::string> requires_;
        std::vector<std::string> produces_;
    };

}  // namespace shipinfer
