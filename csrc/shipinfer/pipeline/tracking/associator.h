// What a tracker does, with nothing of a tracker in it: detections in, ids out.
//
// TWO LINES CROSS HERE AND THEY ARE NOT THE SAME LINE.
//   * `3rdparty/shipvision` is an EXTERNAL LANE, so a unit reaching it is compiled only when
//     that lane is in the build (`scripts/build_csrc.py`). A header that reaches it drags the
//     lane into every includer -- measured, as a build without the submodule that dropped
//     `stages.o` and `from_plan.o` and would not link.
//   * `core/platform.h` is the CUDA line, and `--offline` refuses any unit whose closure
//     reaches it. `graph/state.h` does, because a frame can hold a device surface.
//
// A tracking stage sits on the CUDA side (it reads `FrameState`) and a tracker sits on the lane
// side, so they cannot be the same unit: the lane's CI job builds `--offline`, g++ alone. This
// interface is the seam -- `Detection` and ints, from `core/types.h`, which crosses neither
// line. The stage lives in `graph/stages.h` where stages live; the tracker lives in the lane's
// own unit and registers itself here.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"

namespace shipinfer::tracking {

    // One camera's identities over time. `ids` is parallel to `detections`, `-1` where a
    // detection matched no confirmed track -- the caller needs to put an id back on a row.
    //
    // `ids` MAY THROW `InferenceError` for one frame: a tracker is stateful and ordered, so a
    // frame that does not advance its camera's stream is refused rather than replayed. That is
    // a routine outcome and not a fault -- the caller publishes the frame with no ids and
    // calls `note_untracked`, which is what `topology/elements/track.py` does with the same
    // refusal (`_untracked`, counted, never an exception out of the element).
    class Associator {
      public:
        virtual ~Associator() = default;
        virtual std::vector<int> ids(const std::string& camera_id, int64_t frame_id,
                                     const std::vector<Detection>& detections) = 0;

        // Frames the caller published with no ids because `ids` refused them. The count lives
        // HERE and not on the stage because one associator is shared by every worker for one
        // slot (`create_associator`), so this is the run's answer rather than one thread's.
        void note_untracked() { untracked_.fetch_add(1, std::memory_order_relaxed); }
        uint64_t untracked_frames() const { return untracked_.load(std::memory_order_relaxed); }

      private:
        std::atomic<uint64_t> untracked_{0};
    };

    // doc: long what the chain states about a tracker, and why absent is not a default here
    //: WHAT THE CHAIN SAID ABOUT THIS TRACKER. Absent means "the chain did not say", which is
    //: the lane's own default and NOT a number chosen here -- two defaults for one knob is how
    //: they drift, and `ClusterOptions` next door carries its two the same way.
    //:
    //: `options` is left as strings on purpose. The plan is a line format both planes read,
    //: the lane owns the key table, and parsing a value into the wrong type here would move
    //: that ownership into the plan reader -- where an unknown key could only be dropped.
    //: `bytetrack.cpp` converts and REFUSES a key it does not have, which is what
    //: `TrackerShard` does at `open()` on the other plane.
    //: NO `attribution_iou`, and its absence is the decision rather than an omission: that
    //: knob maps a tracker's answers back onto detection ROWS, and this plane has no such
    //: step -- `TrackerShard::update` returns ids per detection already. A plan line nothing
    //: reads is the same trap as a reader that drops what it does not know, so the remaining
    //: divergence stays named in `benchmarks/parity/known.py` instead.
    struct TrackerOptions {
        std::optional<int64_t> regression_reset;
        std::map<std::string, std::string> options;
        //: The chain's `params: algorithm:`. Empty means the chain did not say, and then the
        //: lane's own default stands -- which is the ONE case where the two planes agree
        //: without carrying anything, because both default to ByteTrack.
        std::string algorithm;

        bool operator==(const TrackerOptions& other) const {
            return regression_reset == other.regression_reset && options == other.options &&
                   algorithm == other.algorithm;
        }
    };

    using AssociatorFactory = std::function<std::shared_ptr<Associator>(const TrackerOptions&)>;

    class AssociatorRegistry {
      public:
        void add(const std::string& impl, AssociatorFactory factory);
        bool has(const std::string& impl) const;
        std::shared_ptr<Associator> create(const std::string& impl,
                                           const TrackerOptions& options = {}) const;
        std::vector<std::string> names() const;

      private:
        std::map<std::string, AssociatorFactory> entries_;
    };

    // Function-local static: every unit's registrar can run before `main`, in any order.
    AssociatorRegistry& ASSOCIATORS();

    // The associator for `impl` on `slot`, or a refusal that says whether the name is unknown
    // or the LANE is absent -- two problems with two fixes, the distinction
    // `ingest/registry.h` was rewritten to make.
    //
    // ONE PER (impl, slot), CACHED FOR THE PROCESS, and both halves of that key are load
    // bearing. Shared across SLOTS is wrong: a tracker is keyed by camera, so two slots with
    // disjoint selections over one camera would take turns refusing each other's frames and
    // age each set out on the other's update -- the Python plane builds a `TrackerShard` per
    // ELEMENT INSTANCE (`track.py::_do_open`) and this is that shape. Fresh per CALLER is
    // wrong too: `bench.cpp` builds one Dag per worker, so that would be a tracker per thread
    // for one camera, which is the identity split `shard.h` exists to prevent.
    //
    // REFUSED WHEN TWO CALLERS DISAGREE about one (impl, slot)'s options, for the reason the
    // cache exists at all: the first caller's tracker is the one every later caller gets, so
    // a second set of options would be silently ignored and one slot would run a
    // configuration no chain states.
    std::shared_ptr<Associator> create_associator(const std::string& impl,
                                                  const std::string& slot,
                                                  const TrackerOptions& options = {});

    // What `create_associator` has built so far, so a run can report per-slot counters
    // without reaching into a worker's Dag for a stage.
    struct MadeAssociator {
        std::string impl;
        std::string slot;
        std::shared_ptr<Associator> associator;
    };
    std::vector<MadeAssociator> made_associators();

    // One of these at the bottom of a tracker's unit is the `@ASSOCIATORS.register(...)`.
    struct AssociatorRegistrar {
        AssociatorRegistrar(const std::string& impl, AssociatorFactory factory) {
            ASSOCIATORS().add(impl, std::move(factory));
        }
    };

}  // namespace shipinfer::tracking
