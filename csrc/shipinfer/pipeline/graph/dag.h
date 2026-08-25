// The perception graph as a sequence of stages — `pipeline/graph/graph.py`.
//
// Run every stage that can run, in declared order, and report each one. A stage is *runnable*
// when everything it consumes is available and everything it requires is non-empty — decided
// from the frame's state right now, never speculatively: a frame with three ships and no
// people must not announce a person embedder that is never called, or reassembly would wait for
// it until the frame timed out. A stage that fails does not end the frame; its branch is
// skipped and the emitted event names what was lost.
#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/pipeline/graph/stage.h"
#include "shipinfer/pipeline/graph/state.h"
#include "shipinfer/pipeline/reassembly/collector.h"

namespace shipinfer {

    // Told what will run and what happened, stage by stage — `StageObserver`.
    class StageObserver {
      public:
        virtual ~StageObserver() = default;
        virtual void planned(const std::vector<std::string>& stages) = 0;
        virtual void finished(const StageOutcome& outcome) = 0;
    };

    // The collector as an observer: what is planned becomes expected, what ran is delivered,
    // what was skipped is neither, what failed stays missing — three distinguishable events.
    class CollectorObserver : public StageObserver {
      public:
        CollectorObserver(FrameCollector& collector, FrameTag tag)
            : collector_(collector), tag_(std::move(tag)) {}
        void planned(const std::vector<std::string>& stages) override {
            collector_.also_expect(tag_, stages);
        }
        void finished(const StageOutcome& outcome) override {
            if (outcome.ran()) collector_.deliver(tag_, outcome.stage);
        }

      private:
        FrameCollector& collector_;
        FrameTag tag_;
    };

    class Dag {
      public:
        void add(std::unique_ptr<Stage> stage) { stages_.push_back(std::move(stage)); }
        size_t size() const { return stages_.size(); }
        std::vector<std::string> stage_names() const {
            std::vector<std::string> names;
            for (const auto& stage : stages_) names.push_back(stage->name());
            return names;
        }

        // Stages whose inputs are present and non-empty right now, and not yet done.
        std::vector<std::string> runnable(const FrameState& state,
                                          const std::vector<std::string>& done) const {
            const std::vector<std::string> available = state.available();
            const std::vector<std::string> non_empty = state.non_empty();
            auto has = [](const std::vector<std::string>& names, const std::string& name) {
                return std::find(names.begin(), names.end(), name) != names.end();
            };
            std::vector<std::string> ready;
            for (const auto& stage : stages_) {
                if (has(done, stage->name())) continue;
                bool ok = true;
                for (const std::string& name : stage->consumes())
                    ok = ok && has(available, name);
                for (const std::string& name : stage->requires())
                    ok = ok && has(non_empty, name);
                if (ok) ready.push_back(stage->name());
            }
            return ready;
        }

        std::vector<StageOutcome> execute(FrameState& state, StageObserver& observer) {
            std::vector<StageOutcome> outcomes;
            std::vector<std::string> done;
            for (const auto& stage : stages_) {
                const std::vector<std::string> ready = runnable(state, done);
                if (!ready.empty()) observer.planned(ready);
                StageOutcome outcome;
                if (std::find(ready.begin(), ready.end(), stage->name()) != ready.end()) {
                    outcome = stage->run(state);
                } else {
                    outcome.stage = stage->name();
                    outcome.status = StageStatus::Skipped;
                }
                done.push_back(stage->name());
                observer.finished(outcome);
                outcomes.push_back(std::move(outcome));
            }
            return outcomes;
        }

      private:
        std::vector<std::unique_ptr<Stage>> stages_;
    };

}  // namespace shipinfer
