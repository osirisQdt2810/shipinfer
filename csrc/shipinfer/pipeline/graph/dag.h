// The perception graph as a sequence of stages — `pipeline/graph/graph.py`.
//
// Run every stage that can run, in declared order, and report each one. A stage is *runnable*
// when everything it consumes is available and everything it needs is non-empty — decided
// from the frame's state right now, never speculatively: a frame with three ships and no
// people must not announce a person embedder that is never called, or reassembly would wait for
// it until the frame timed out. A stage that fails does not end the frame; its branch is
// skipped and the emitted event names what was lost.
#pragma once

#include <algorithm>
#include <memory>
#include <optional>
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
                for (const std::string& name : stage->needs()) ok = ok && has(non_empty, name);
                if (ok) ready.push_back(stage->name());
            }
            return ready;
        }

        // In declared order, one stage at a time -- except a run of `overlaps()` stages none of
        // which reads another's output: all of them are begun before any is finished, so the
        // frame waits for the slowest of their models rather than for their sum.
        std::vector<StageOutcome> execute(FrameState& state, StageObserver& observer) {
            std::vector<StageOutcome> outcomes;
            std::vector<std::string> done;
            for (size_t i = 0; i < stages_.size();) {
                const std::vector<std::string> ready = runnable(state, done);
                if (!ready.empty()) observer.planned(ready);
                const size_t end = wave_end(i);
                std::vector<std::optional<StageOutcome>> begun(end - i);
                for (size_t k = i; k < end && end - i > 1; ++k) {
                    if (is_in(ready, stages_[k]->name()))
                        begun[k - i] = stages_[k]->begin(state);
                }
                for (size_t k = i; k < end; ++k) {
                    const Stage& stage = *stages_[k];
                    StageOutcome outcome;
                    if (!is_in(ready, stage.name())) {
                        outcome.stage = stage.name();
                        outcome.status = StageStatus::Skipped;
                    } else if (end - i == 1) {
                        outcome = stages_[k]->run(state);
                    } else if (!begun[k - i]->ran()) {
                        outcome = *begun[k - i];  // failed to begin: nothing to finish
                    } else {
                        outcome = stages_[k]->finish(state);
                    }
                    done.push_back(stage.name());
                    observer.finished(outcome);
                    outcomes.push_back(std::move(outcome));
                }
                i = end;
            }
            return outcomes;
        }

      private:
        static bool is_in(const std::vector<std::string>& names, const std::string& name) {
            return std::find(names.begin(), names.end(), name) != names.end();
        }

        // One past the last stage that may overlap `stages_[first]`: consecutive `overlaps()`
        // stages, cut before the first that consumes what an earlier one in the run produces.
        size_t wave_end(size_t first) const {
            size_t end = first + 1;
            if (!stages_[first]->overlaps()) return end;
            std::vector<std::string> produced = stages_[first]->produces();
            for (; end < stages_.size() && stages_[end]->overlaps(); ++end) {
                for (const std::string& name : stages_[end]->consumes()) {
                    if (is_in(produced, name)) return end;
                }
                const auto& more = stages_[end]->produces();
                produced.insert(produced.end(), more.begin(), more.end());
            }
            return end;
        }

        std::vector<std::unique_ptr<Stage>> stages_;
    };

}  // namespace shipinfer
