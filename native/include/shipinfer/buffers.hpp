// Persistent, growable device and pinned-host scratch.
//
// The first version of this extension allocated and freed its scratch on every
// call, and it was no faster than the pure-torch path. The kernel was never the
// problem: device allocation and free are *synchronising* calls, and a pageable
// copy runs at roughly half the bandwidth of a pinned one
// because the driver stages it through an internal bounce buffer. Between them
// they cost more than the work.
//
// So the scratch lives as long as the `ImageOps` object and grows
// monotonically. That is the same bet torch's caching allocator makes, made
// here because this extension does not link torch.

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "shipinfer/platform.hpp"

namespace shipinfer {

  /// Device scratch that grows to the high-water mark and is never handed back.
  class DeviceScratch {
    public:
      DeviceScratch() = default;
      ~DeviceScratch() { release(); }
      DeviceScratch(const DeviceScratch&) = delete;
      DeviceScratch& operator=(const DeviceScratch&) = delete;

      /// A pointer to at least `bytes`, reallocating only when the request grows.
      void* reserve(size_t bytes) {
        if (bytes <= capacity_)
          return ptr_;
        release();
        // Over-allocate by a quarter so a slowly growing batch does not reallocate
        // every call.
        const size_t target = bytes + bytes / 4;
        check(gpuMalloc(&ptr_, target), "gpuMalloc (device scratch)");
        capacity_ = target;
        return ptr_;
      }

      size_t capacity() const { return capacity_; }

      void release() {
        if (ptr_ != nullptr) {
          gpuFree(ptr_);
          ptr_ = nullptr;
          capacity_ = 0;
        }
      }

    private:
      void* ptr_ = nullptr;
      size_t capacity_ = 0;
  };

  /// Page-locked host scratch, same growth policy.
  ///
  /// Worth the trouble for one reason: an async copy from pageable memory
  /// silently degrades to a synchronous copy at about half the bandwidth. Staging
  /// through pinned memory is what makes the upload both faster and genuinely
  /// asynchronous.
  class PinnedScratch {
    public:
      PinnedScratch() = default;
      ~PinnedScratch() { release(); }
      PinnedScratch(const PinnedScratch&) = delete;
      PinnedScratch& operator=(const PinnedScratch&) = delete;

      unsigned char* reserve(size_t bytes) {
        if (bytes <= capacity_)
          return ptr_;
        release();
        const size_t target = bytes + bytes / 4;
        check(gpuHostAlloc(reinterpret_cast<void**>(&ptr_), target, gpuHostAllocDefault),
              "gpuHostAlloc");
        capacity_ = target;
        return ptr_;
      }

      size_t capacity() const { return capacity_; }

      void release() {
        if (ptr_ != nullptr) {
          gpuHostFree(ptr_);
          ptr_ = nullptr;
          capacity_ = 0;
        }
      }

    private:
      unsigned char* ptr_ = nullptr;
      size_t capacity_ = 0;
  };

  /// One batch's worth of staging: pinned host bytes, the device copy of them, and the
  /// descriptor table the kernel reads.
  ///
  /// Rotating several of these is what makes the asynchronous path actually asynchronous.
  /// With a single set, the next call's host memcpy overwrites the pinned buffer while the
  /// previous call's DMA is still reading it, and its upload overwrites the device frames
  /// while the previous kernel is still sampling them. Both races produce plausible output
  /// — and are invisible to any benchmark that submits the same image twice.
  ///
  /// The event is what makes reuse safe without a blanket synchronise: it is recorded after
  /// the kernel, and a slot waits on its own event before being written again. With enough
  /// slots that wait has almost always already been satisfied, so the cost is a completed
  /// event query rather than a stall.
  class StagingSlot {
    public:
      StagingSlot() {
        // Timing disabled: this event exists to order work, and the timing machinery
        // costs a measurable amount on every record.
        check(gpuEventCreateWithFlags(&done_, gpuEventDisableTiming), "gpuEventCreate");
      }
      ~StagingSlot() {
        if (done_ != nullptr)
          gpuEventDestroy(done_);
      }
      StagingSlot(const StagingSlot&) = delete;
      StagingSlot& operator=(const StagingSlot&) = delete;

      /// Block until the work that last used this slot has finished.
      void wait() {
        if (recorded_)
          check(gpuEventSynchronize(done_), "gpuEventSynchronize");
      }

      /// Mark everything issued on `stream` so far as the work this slot is waiting on.
      void record(gpuStream_t stream) {
        check(gpuEventRecord(done_, stream), "gpuEventRecord");
        recorded_ = true;
      }

      DeviceScratch& frames() { return frames_; }
      DeviceScratch& views() { return views_; }
      PinnedScratch& pinned() { return pinned_; }

      size_t bytes() const {
        return frames_.capacity() + views_.capacity() + pinned_.capacity();
      }

    private:
      DeviceScratch frames_;
      DeviceScratch views_;
      PinnedScratch pinned_;
      gpuEvent_t done_ = nullptr;
      bool recorded_ = false;
  };

  /// A fixed rotation of staging slots.
  ///
  /// Three by default: one being filled on the host, one in flight, one being consumed by
  /// the kernel. Two would work and leave no slack; more costs pinned memory for a queue
  /// depth the scheduler does not actually reach.
  class StagingRing {
    public:
      explicit StagingRing(size_t slots = 3) : slots_(slots) {
        for (size_t i = 0; i < slots; ++i)
          ring_.push_back(std::make_unique<StagingSlot>());
      }

      /// The next slot, already waited on and safe to overwrite.
      StagingSlot& acquire() {
        StagingSlot& slot = *ring_[cursor_];
        cursor_ = (cursor_ + 1) % slots_;
        slot.wait();
        return slot;
      }

      size_t bytes() const {
        size_t total = 0;
        for (const auto& slot : ring_)
          total += slot->bytes();
        return total;
      }

    private:
      std::vector<std::unique_ptr<StagingSlot>> ring_;
      size_t slots_;
      size_t cursor_ = 0;
  };

} // namespace shipinfer
