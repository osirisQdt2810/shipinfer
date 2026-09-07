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
        QueueSink(FairPriorityQueue<FrameWork>& queue, size_t pooled, size_t devices)
            : queue_(queue), pooled_(pooled), devices_(devices) {}

        void put(Frame&& frame) override {
            FrameWork work;
            work.tag = frame.tag;
            if (frame.on_device()) {
                // ONE PROCESS, ONE GPU, for a device frame -- and it is ADR-004 rather than a
                // shortcut: a frame stays where it was decoded, so a worker on another GPU
                // cannot take it, and this bench's ONE fleet-wide fair queue is what lets any
                // worker take any frame (which is the property that makes it fair across
                // cameras rather than within a device). The deployment shape already resolves
                // it: `--runner fleet` is one shard process per GPU, so a camera is decoded and
                // processed on the same one. Refused HERE, in the sink, so `CameraActor` stops
                // the camera with the reason in health instead of failing every frame.
                if (devices_ > 1) {
                    throw ConfigError(
                        "camera '" + frame.tag.camera_id +
                        "': a device frame cannot be scheduled across " +
                        std::to_string(devices_) +
                        " GPUs from one process -- the frame stays where it was decoded "
                        "(ADR-004) and this queue is fleet-wide. Use `--runner fleet` (one "
                        "process per GPU) for the multi-GPU shape, or give this bench a "
                        "single `--devices`");
                }
                // COPIED OUT HERE, on the actor's thread, and the decode slot given back
                // before this frame is queued -- `pipeline/surface_intake.h` argues why at
                // length. A surface that travelled through the queue would exhaust a pool of
                // two output surfaces at the third frame, under exactly the load the queue
                // exists to serve.
                work.device = frame.device.device;
                work.surface = intake(work.device).take(frame.device);
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
        // One intake per DEVICE rather than per camera: fifty cameras of one resolution share
        // one free list, which is fifty times fewer idle buffers for the same steady state.
        // Created on first use, because a device source may be configured for any GPU.
        SurfaceIntake& intake(int device) {
            std::lock_guard<std::mutex> lock(intakes_mutex_);
            std::unique_ptr<SurfaceIntake>& slot = intakes_[device];
            if (!slot) slot = std::make_unique<SurfaceIntake>(device, pooled_);
            return *slot;
        }

        FairPriorityQueue<FrameWork>& queue_;
        //: Idle buffers kept per device. The queue's capacity is the bound that matters: a
        //: frame waiting in it holds one of these, so a pool that size never allocates in
        //: steady state and nothing beyond it is worth holding.
        size_t pooled_;
        //: How many GPUs this process drives. A device frame needs exactly one; see `put`.
        size_t devices_;
        std::mutex intakes_mutex_;
        std::map<int, std::unique_ptr<SurfaceIntake>> intakes_;
    };

}  // namespace shipinfer
