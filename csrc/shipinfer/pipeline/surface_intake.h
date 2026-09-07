// doc: long the copy is the design decision here, and it needs its alternatives stated
// A DECODER'S SURFACE IN, A SURFACE THE PIPELINE OWNS OUT — and the decoder's slot back at
// once.
//
// WHY A COPY IS THE RIGHT ANSWER HERE, AND NOT A COMPROMISE
// --------------------------------------------------------
// NVDEC hands out a mapped surface from a per-camera pool of two output surfaces. The fair
// queue's whole purpose is to HOLD frames while a batch assembles, so a surface travelling
// through it would exhaust that pool at the third frame -- `cuvidMapVideoFrame` fails, the
// camera reconnect-loops, and it happens under exactly the load the queue exists to serve.
// Raising the pool to the queue's depth is per-camera VRAM times fifty cameras; shortening the
// queue gives up the fairness this project exists to get right.
//
// So the hand-off copies, which is also what DeepStream does between the decoder's NVMM pool
// and `nvvideoconvert`'s (V86: the reference implementation's shape is the default). It stays
// V156's route -- ~3 MB DEVICE-TO-DEVICE for a 1080p NV12 frame, against ~6 MB down *and* 6 MB
// up for the host round trip the route exists to remove -- and the decoder's pool goes back to
// being bounded by decode depth rather than by queue depth.
//
// What comes out is TIGHT -- `uv_offset` is `stride * height`, exactly one luma plane. That is
// also what a `cuvidMapVideoFrame` surface reports (#159 measured it: the coded height sizes
// the decode surfaces, not the mapped output one), so the copy preserves the number rather
// than changing it, and no consumer can tell the two producers apart.
#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "shipinfer/core/buffers.h"
#include "shipinfer/ingest/frame.h"
#include "shipinfer/pipeline/graph/state.h"

namespace shipinfer {

    // doc: long the thread contract and the lifetime contract, both of which shipped wrong
    // ONE GPU'S intake, shared by every camera decoding on it -- so `take` is called
    // concurrently by every one of those actor threads, and everything shared is under
    // `mutex_`. (It said "one camera's intake, not thread-safe to call" when this landed, which
    // was true of neither and an invitation to add an unguarded field.) `gpuSetDevice` is
    // per-thread and each caller states it, so no CUDA state is shared either.
    //
    // HELD THROUGH A `shared_ptr`, and that is a contract rather than a convenience: a frame's
    // surface keeps its pool alive. The first version captured `this` in the deleter and argued
    // that the sink outlives every frame -- which is false on the unwind path the whole of
    // `core/join_on_unwind.h` exists for: the sink is declared AFTER the guard that stops the
    // workers, so a throw between `manager.start()` and `queue.close()` destroys the pool while
    // worker threads still hold surfaces, and every deleter then locks a destroyed mutex. There
    // is no cycle to worry about: the free list holds buffers, never surfaces.
    // doc: long why this is not `WorkerScratch`, which is the other device-buffer pool here
    // WHY NOT `WorkerScratch` (`pipeline/graph/stages.h`), which also reuses `DeviceBuffer`s
    // under a cap: it is single-threaded by design (one worker owns one, ADR-002), it keys by
    // NAME rather than by size, and it THROWS past its cap. An ingest-side pool can accept none
    // of the three -- every camera on a GPU shares this one, a mixed-resolution fleet needs the
    // size to be the key, and a burst must free a buffer rather than fail a frame. Same
    // primitive, different contract; a shared base would be one class with two behaviours.
    class SurfaceIntake {
      public:
        // `device` is the GPU the decoder is on, and the one every buffer is allocated on: a
        // frame stays where it was decoded (ADR-004).
        //
        // `max_pooled` is per SIZE, because that is what a bucket is: a fleet of mixed
        // resolutions holds one cap's worth of each rather than one cap between them.
        explicit SurfaceIntake(int device, size_t max_pooled = 8);

        // `image` copied into a pooled buffer. The returned surface owns that buffer and
        // returns it to the pool when the last reference goes -- and holds `self` so the pool
        // outlives it. The CALLER still holds `image.owner`, and dropping it is what gives the
        // decode slot back.
        //
        // `self` must be the `shared_ptr` that owns `*this`; the static overload below is the
        // reason this is not a plain member. `camera` names the camera in a refusal.
        //
        // Throws `ConfigError` if `image` is not a usable NV12 surface, and the typed CUDA
        // error if the copy fails -- never a surface with fewer bytes than it claims.
        static DeviceSurface take(const std::shared_ptr<SurfaceIntake>& self,
                                  const DeviceImage& image, const std::string& camera);

        // How many buffers are sitting in the free lists, across sizes. For the gate that
        // proves reuse: an allocation per frame at a thousand frames a second is what this
        // class avoids.
        size_t pooled() const;

      private:
        // Returned by the deleter, into ITS OWN SIZE'S bucket. A cap per bucket, because a
        // stalled consumer must not turn into unbounded VRAM -- and keyed by size because one
        // shared `bytes_` made a mixed-resolution fleet retire the whole list on every
        // alternating frame: a `cudaMalloc` plus a `cudaFree` per frame, inside the mutex every
        // camera on the GPU contends for, which is precisely what this class exists to prevent.
        void give_back(std::unique_ptr<DeviceBuffer> buffer);

        int device_;
        size_t max_pooled_;
        mutable std::mutex mutex_;
        //: Size -> the buffers of that size waiting to be reused, so a fleet of two
        //: resolutions gets pool hits for both instead of retiring each other's.
        //:
        //: NOTHING RETIRES A STALE BUCKET, and that is a decision rather than an omission: a
        //: camera that reconnects at a new resolution leaves its old size pooled for the run.
        //: Bounded -- `max_pooled` x distinct sizes x frame bytes, so ~93 MB per stale 1080p
        //: bucket at the derived cap -- and a source refuses a resolution change mid-stream,
        //: so a size only appears across a reconnect. A retirement rule needs a
        //: least-recently-taken clock to avoid dropping a bucket a camera still wants, which
        //: is more machinery than a bounded, stated cost is worth until a run shows otherwise.
        std::map<size_t, std::vector<std::unique_ptr<DeviceBuffer>>> free_;
    };

}  // namespace shipinfer
