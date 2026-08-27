// One RTSP camera, decoded by a GStreamer pipeline into BGR frames — the gst-linked half of
// `ingest/sources/gstreamer.py`. The pure half (the pipeline strings and the element choices)
// is the sibling header `sources/gstreamer_pipeline.h`, which anything may include.
//
// **THIS UNIT IS THE INGEST PLANE'S ONE GSTREAMER-LINKED FILE**, and like `sources/replay.*`
// that position is a build fact rather than a preference. `scripts/build_csrc.py` follows a
// header to the `.cpp` beside it, so a single `#include "shipinfer/ingest/sources/gstreamer.h"`
// from anywhere else under `ingest/` would drag `libgstreamer-1.0` into the closure of the
// whole ingest plane, and the offline C++ tier — `g++` alone, with nobody's `-dev` package
// installed — would stop building. **Nothing under `ingest/` may include this header except the
// `.cpp` beside it**; see the note at the top of `ingest/registry.cpp`, which now guards two
// such units. The build script's `--with-external gstreamer` is what opts this one unit back
// into an otherwise offline build, in the one container that has both the headers and the
// plugins.
//
// NO GSTREAMER TYPE IS NAMED HERE, AND THE CHOICE IS PIMPL RATHER THAN `void*`
// ---------------------------------------------------------------------------
// The graph is a forward-declared `Graph` behind a `unique_ptr`, so this header parses on a
// machine with no GStreamer at all. That is defence in depth, not the primary guard: the
// closure walker above is what actually keeps the plane offline-buildable, and an accidental
// include would still pull this unit onto the link line. What pimpl buys is that the accident
// fails as a *link* decision the build script can see and report, instead of as `gst/gst.h: No
// such file or directory` in a translation unit that has nothing to do with cameras. `void*`
// members would hide the same two pointers with the same effect and cost every use in the
// `.cpp` a cast, which is exactly where a `GstElement*` gets `gst_object_unref`'d as a
// `GstSample*` one day.
//
// THE ERROR TAXONOMY, WHICH IS THE POINT OF HAVING FOUR TYPES
// ----------------------------------------------------------
//   pipeline will not parse / no appsink / PLAYING refused / did not start in time
//                                                          -> SourceOpenError   (retryable)
//   EOS, a bus ERROR, an unmappable buffer                  -> FrameDecodeError  (reconnect)
//   no decoder or converter plugin installed                -> SourceUnavailableError (fatal)
//
// **EOS ON A CAMERA IS A FAULT.** A live stream that ends has broken, so this source never
// overrides `is_exhausted()`: end-of-stream raises `FrameDecodeError` and the actor reconnects.
// Only a finite source (`replay` with `loop=false`) is ever exhausted, and conflating the two
// would make a dropped camera look like a completed job.
#pragma once

#include <memory>
#include <optional>
#include <string>

#include "shipinfer/ingest/base.h"
#include "shipinfer/ingest/frame.h"

namespace shipinfer {

    // One RTSP camera, over a `gst-launch`-compatible pipeline built by `build_pipeline`.
    //
    // Hardware decode is a *preference*, not a requirement: the decoder is chosen by probing
    // the plugin registry (`select_decoder`), so this runs on a DeepStream host, on a desktop
    // with `gst-plugins-bad`'s nvcodec, and on a plain Ubuntu box with neither.
    //
    // Two knobs come from the camera's `options`, because this plane has no environment layer
    // yet (`ingest/config.h` says so) while the Python plane reads the same two from
    // `SHIPINFER_GST_DECODER` and `SHIPINFER_GST_APPSINK_MAX_BUFFERS`:
    //
    //   decoder      force one element instead of probing — the escape hatch for a box where
    //                the probe picks a decoder that is installed but broken.
    //   max_buffers  appsink queue depth, default 2. A deep decoder queue converts a throughput
    //                problem into a latency problem and hides it.
    class GStreamerSource : public FrameSource {
      public:
        using FrameSource::FrameSource;

        // Closes defensively. `~FrameSource` cannot dispatch to `do_close` (see `base.h`), so a
        // source dropped without a `close()` would otherwise leave a decoder thread and an RTSP
        // socket alive for the life of the process. Declared here and defined in the `.cpp`
        // because that is where `Graph` is a complete type.
        ~GStreamerSource() override;

        bool supports_hwaccel() const override { return true; }

        // The exact `gst-launch-1.0` line in use, credentials intact; empty before `open()`.
        // **Redact it at the point it becomes text a human reads** — see the note in the
        // `.cpp`.
        //
        // The parity of `GStreamerSource.pipeline_description` on the Python side, and the hook
        // P5's logging needs. Deliberately unasserted here, because it cannot be: reaching it
        // needs the concrete type, and `tests/test_ingest.cpp` may not include this header (the
        // closure invariant above). The string it returns is `build_pipeline`'s, and *that* is
        // pinned character by character in the offline tier.
        const std::string& pipeline_description() const { return description_; }

      protected:
        void do_open() override;
        std::optional<HostFrame> do_read() override;
        void do_close() override;

      private:
        // Best effort, from the appsink's sink pad: the pad may carry no caps yet even in
        // PLAYING, in which case the first decoded frame fills them in. Reporting the
        // *negotiated* size rather than the requested one is what makes a silently ignored
        // `width`/`height` visible.
        void negotiate_from_appsink();
        // Ask the bus why a read came back empty. A timeout means "keep waiting"; an EOS or an
        // ERROR means this stream is over, and throws.
        void raise_if_stream_ended();

        struct Graph;
        std::unique_ptr<Graph> graph_;
        std::string description_;
    };

}  // namespace shipinfer
