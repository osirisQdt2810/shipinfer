// doc: long why this left `cli/bench.cpp`, which is the reason it can be tested at all
// ONE FRAME'S WORTH OF SCHEDULED WORK, and the sink that makes it.
//
// Lifted out of `cli/bench.cpp` so it can be tested. It was an anonymous-namespace type in a
// composition root, which was fine while `put` was six lines; it now decides which of two pixel
// representations a frame carries, copies a decoder surface out of its pool, and refuses a
// shape that cannot work -- three things worth a gate each.
//
// It stays in `pipeline/` and not in `ingest/`: mapping a frame onto a unit of SCHEDULED work
// is dispatch policy, and the same code has to undo the mapping when results are reassembled.
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "shipinfer/core/types.h"
#include "shipinfer/ingest/frame.h"
#include "shipinfer/ingest/sink.h"
#include "shipinfer/pipeline/graph/state.h"
#include "shipinfer/pipeline/surface_intake.h"
#include "shipinfer/scheduling/queues/fair.h"

namespace shipinfer {

    // One frame's worth of work as it sits in the fair queue.
    //
    // The pixels stay on the **host** here, and the worker copies them to its own device. The
    // first version had the camera thread copy to a device chosen by camera index, and then any
    // worker could pick up any frame — a frame on device 1 executed by an instance on device 0,
    // which is a cross-device access. It does not fail at the call that caused it; it surfaces
    // as `an illegal memory access was encountered` inside an unrelated `gpuDeviceSynchronize`
    // several frames later, which is why ADR-002 makes the rule structural rather than
    // advisory.
    //
    // Keeping one fleet-wide fair queue matters more than saving the copy: fairness across
    // cameras is the thing this project exists to get right, and a queue per device would make
    // it fair only within a device.
    struct FrameWork {
        FrameTag tag;
        std::shared_ptr<FrameState> state;
        HostFrame frame;  // the library owns the pixels and outlives every frame in flight
        //: The other representation, already copied out of the decoder's pool by
        //: `SurfaceIntake` so no queued frame holds a decode slot. Exactly one of these two is
        //: populated, which is a property of the camera and not of the frame.
        DeviceSurface surface;
        int device = -1;  // where `surface` lives; -1 for a host frame, which any worker takes

        size_t rows() const { return 1; }
        std::string camera() const { return tag.camera_id; }
        // A frame into the detector is NORMAL priority with no deadline — what the Python
        // pipeline submits. The lanes above and below exist for the requests that will use
        // them.
        int priority() const { return Priority::Normal; }
        bool expired(int64_t) const { return false; }
    };

    // The bridge from the ingest plane to the fair queue: a `FrameSink` that turns a tagged
    // frame into one queue entry.
    //
    // It is *here*, in the application, rather than in `ingest/` — mapping a frame onto a unit
    // of scheduled work is dispatch policy, and the same code has to undo the mapping when the
    // results are reassembled. Refusal is thrown rather than returned because the actor is the
    // only component that knows whose frame it is and can charge the drop to that camera
    // (ADR-005).
    class QueueSink : public FrameSink {
      public:
        // `devices` is the GPU list this process drives, in `--devices` order. The SET and not
        // the count, because what makes a device frame unusable is not how many GPUs a process
        // has -- it is whether the frame's GPU is one of them.
        QueueSink(FairPriorityQueue<FrameWork>& queue, size_t pooled, std::vector<int> devices)
            : queue_(queue), pooled_(pooled), devices_(std::move(devices)) {}

        void put(Frame&& frame) override {
            FrameWork work;
            work.tag = frame.tag;
            if (frame.on_device()) {
                // doc: long the predicate this needs, and the one it had
                // A DEVICE FRAME MUST BE ON A GPU THIS PROCESS DRIVES, and it is ADR-004 rather
                // than a shortcut: a frame stays where it was decoded, so a worker on another
                // GPU cannot take it, while this bench's ONE fleet-wide queue is what lets any
                // worker take any frame (the property that makes it fair across cameras rather
                // than within a device). One process, one GPU, for the device path; the
                // deployment shape already resolves it -- `--runner fleet` is one shard per
                // GPU.
                //
                // THE SET, NOT THE COUNT. The first version refused on `devices > 1`, which
                // cannot see the case it exists for: `--devices 3 --source nvdec` on this box
                // (an ordinary choice when gpu0 is busy) has every camera decoding on gpu0
                // because nothing sets their `device` option, `devices == 1` so the sink
                // accepts, and the worker bound to gpu3 then throws per frame, forever -- the
                // exact outcome this refusal exists to replace with one health line.
                if (std::find(devices_.begin(), devices_.end(), frame.device.device) ==
                    devices_.end()) {
                    std::string given;
                    for (int device : devices_) {
                        given += (given.empty() ? "" : ",") + std::to_string(device);
                    }
                    throw ConfigError(
                        "camera '" + frame.tag.camera_id + "': its frames are decoded on gpu" +
                        std::to_string(frame.device.device) +
                        ", which this process does not drive (it has " + given +
                        ") -- a frame stays where it was decoded (ADR-004). Point the camera's "
                        "`device` option at one of those, or run one process per GPU "
                        "(`--runner fleet`)");
                }
                // COPIED OUT HERE, on the actor's thread, and the decode slot given back
                // before this frame is queued -- `pipeline/surface_intake.h` argues why at
                // length. A surface that travelled through the queue would exhaust a pool of
                // two output surfaces at the third frame, under exactly the load the queue
                // exists to serve.
                work.device = frame.device.device;
                work.surface =
                    SurfaceIntake::take(intake(work.device), frame.device, work.tag.camera_id);
                // RELEASED HERE, explicitly, and not left to a destructor: `put` takes
                // `Frame&&`
                // -- an rvalue REFERENCE -- so the caller's frame outlives this call and its
                // surface would be held until the actor's next read. That is one extra slot out
                // of a pool of two, per camera, for as long as a queue holds frames, which is
                // exactly the shortage the copy above exists to remove.
                frame.device.owner.reset();
                work.state = std::make_shared<FrameState>(work.tag, frame.device.height,
                                                          frame.device.width, 0.0f);
                work.state->set_surface(work.surface, work.device);
            } else {
                work.frame = std::move(frame.image);
                work.state = std::make_shared<FrameState>(work.tag, work.frame.height,
                                                          work.frame.width, 0.0f);
            }
            const PutStatus status = queue_.put(std::move(work));
            if (status == PutStatus::Rejected) {
                const QueueStats stats = queue_.stats();
                throw QueueFullError("pipeline queue is full", stats.depth, stats.capacity);
            }
            if (status == PutStatus::Closed) {
                throw RequestCancelledError("pipeline queue is closed");
            }
        }

      private:
        // One intake per DEVICE rather than per camera: every camera on a GPU shares its free
        // lists, which is far fewer idle buffers for the same steady state (the lists are keyed
        // by size, so a mixed-resolution fleet still gets pool hits). Created on first use,
        // because a device source may be configured for any GPU.
        //
        // A `shared_ptr` because a frame's surface holds one: the pool must outlive the sink on
        // the unwind path, and `pipeline/surface_intake.h` states why at length.
        std::shared_ptr<SurfaceIntake> intake(int device) {
            std::lock_guard<std::mutex> lock(intakes_mutex_);
            std::shared_ptr<SurfaceIntake>& slot = intakes_[device];
            if (!slot) slot = std::make_shared<SurfaceIntake>(device, pooled_);
            return slot;
        }

      public:
        // Idle buffers held across every device and size. For the occupancy log, so a run
        // reports whether the pool is doing anything rather than leaving it to an argument.
        size_t pooled_buffers() const {
            std::lock_guard<std::mutex> lock(intakes_mutex_);
            size_t total = 0;
            for (const auto& [device, intake] : intakes_) total += intake->pooled();
            return total;
        }

      private:
        FairPriorityQueue<FrameWork>& queue_;
        //: IDLE buffers kept per device per size. Small on purpose: a buffer in flight is out
        //: of the pool, so this bounds only how many are simultaneously unused -- which in
        //: steady state is the difference between returns and takes, a handful. It was the
        //: queue's capacity (256) when this landed, which is the bound on IN-FLIGHT buffers and
        //: therefore never reached: nothing was ever freed and ~800 MB of idle NV12 per device
        //: stayed for the life of the run. `pipeline_pool_size` in the occupancy log is how a
        //: run says which of us is right.
        size_t pooled_;
        //: The GPUs this process drives, in `--devices` order. A device frame must be on one
        //: of them; see `put`.
        std::vector<int> devices_;
        mutable std::mutex intakes_mutex_;
        std::map<int, std::shared_ptr<SurfaceIntake>> intakes_;
    };

}  // namespace shipinfer
