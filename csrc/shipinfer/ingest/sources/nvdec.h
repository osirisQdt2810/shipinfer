// One RTSP camera decoded by NVDEC straight into VRAM — V156's route, and the only source that
// answers `do_read_device`.
//
// `rtsp -> nv12 -> tren vram het -> xu ly tren vram toan bo`. GStreamer carries the H.264
// BITSTREAM off the socket (`build_pipeline` with `bitstream = true`: depay, parse, appsink, no
// decoder) and NVDEC does the decode through `libnvcuvid`. An access unit is a few KB where a
// decoded 1080p BGR frame is ~6 MB, so the host per-frame cost collapses — measured at 36x on
// the committed 2K fixture, before NV12 halves the decoded side again.
//
// WHY NOT GSTREAMER'S OWN CUDA MEMORY
// -----------------------------------
// `memory:CUDAMemory` needs `gst-plugins-bad`'s CUDA library, which became a public pkg-config
// module in GStreamer **1.22**; this image is 1.20, and installing
// `libgstreamer-plugins-bad1.0-dev` there gives neither `gstreamer-cuda-1.0` nor `gst/cuda`
// (MEASURED, 7 Sep — it is what `PHASE-D-NV12` had been waiting on). `libnvcuvid` needs no such
// package: it is a DRIVER library, and nv-codec-headers' *dynlink* variants `dlopen` it at run
// time, so this unit has no build-time driver dependency and a box without one fails at load
// with a message rather than at link. That is `runtime/native.py`'s arrangement (ADR-003).
//
// THIS UNIT IS AN EXTERNAL LANE, like `sources/gstreamer.*` and `sources/replay.*`
// --------------------------------------------------------------------------------
// `scripts/build_csrc.py` follows a header to the `.cpp` beside it, so one `#include` of this
// header from anywhere else under `ingest/` would drag `libnvcuvid`'s headers and GStreamer
// into the closure of the whole plane. **Nothing under `ingest/` may include this except the
// `.cpp` beside it** — the same rule `registry.cpp` states for the other two, now guarding
// three units. `--with-external nvdec` opts it back in.
//
// THE ERROR TAXONOMY
// ------------------
//   pipeline will not parse / no appsink / PLAYING refused    -> SourceOpenError (retryable)
//   EOS, a bus ERROR, a cuvid decode failure                  -> FrameDecodeError (reconnect)
//   libnvcuvid absent, or no CUDA device                      -> SourceUnavailableError (fatal)
//
// **EOS ON A CAMERA IS A FAULT**, as it is for the GStreamer source: a live stream that ends
// has broken, so this never overrides `is_exhausted()`.
#pragma once

#include <memory>
#include <optional>
#include <string>

#include "shipinfer/ingest/base.h"
#include "shipinfer/ingest/frame.h"

namespace shipinfer {

    // One RTSP camera, decoded by NVDEC into NV12 surfaces that never leave the device.
    //
    // Knobs from the camera's `options`, matching the GStreamer source's spelling so an
    // operator moving a camera between the two does not relearn them:
    //
    //   max_buffers  appsink queue depth for the BITSTREAM, default 2. Access units are small,
    //                so this is about latency rather than memory: a deep queue means the
    //                decoder works on stale pictures.
    //   surfaces     decode surfaces cuvid allocates, default 4. Its own floor is what the
    //                stream's DPB needs; more buys pipelining and costs VRAM per camera, which
    //                at fifty cameras is the number that matters.
    //   device       which GPU decodes, default 0. One actor thread owns this source and its
    //                context for the source's whole life (ADR-002).
    //
    // ONE MAPPED SURFACE AT A TIME, and it is a contract rather than a limitation. A frame's
    // `owner` holds a slot out of a pool; the decoder is created with two OUTPUT surfaces, so a
    // consumer may hold one while the next is decoded and mapped, and holding two while asking
    // for a third is what `cuvidMapVideoFrame failed` means. Measured while writing the gate:
    // one output surface delivered exactly one frame, two delivered two, and a consumer that
    // releases each frame before the next read runs indefinitely.
    class NvdecSource : public FrameSource {
      public:
        using FrameSource::FrameSource;

        // Closes defensively: `~FrameSource` cannot dispatch to `do_close`, and a source
        // dropped without one would leave a decoder, an RTSP socket and — the expensive part —
        // a surface pool alive for the life of the process.
        ~NvdecSource() override;

        bool supports_hwaccel() const override { return true; }

        // The `gst-launch-1.0` line carrying the bitstream, credentials intact; empty before
        // `open()`. Redact it where it becomes text a human reads.
        const std::string& pipeline_description() const { return description_; }

      protected:
        void do_open() override;
        // NOTHING, always. This source decodes into VRAM for its whole life, and `read()`
        // latches that on the first frame (`FrameCounter::latch_where`). Pure in the base
        // class on purpose, so the choice is the compiler's to enforce rather than a comment's.
        std::optional<HostFrame> do_read() override { return std::nullopt; }
        std::optional<DeviceImage> do_read_device() override;
        void do_close() override;

      private:
        struct Decoder;
        std::unique_ptr<Decoder> decoder_;
        std::string description_;
    };

}  // namespace shipinfer
