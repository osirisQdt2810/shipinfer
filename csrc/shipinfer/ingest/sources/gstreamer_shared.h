// WHAT BOTH GSTREAMER-LINKED SOURCES NEED, and neither owns.
//
// `sources/gstreamer.cpp` carries frames off the socket and decodes them; `sources/nvdec.cpp`
// carries the bitstream off the same socket and lets NVDEC decode it. Two units, one library,
// and two things that must be identical in both or the second copy drifts:
//
//   `initialise_gstreamer()`     `gst_init` exactly once per process, whoever asks first.
//   `raise_if_stream_ended()`    quiet, or over? Only the bus knows.
//
// The file arrived as `gstreamer_bus.h` for the second of those alone. The first was found by
// RUNNING the bench over RTSP with `--source nvdec`: nothing had initialised GStreamer, and
// `gst_parse_launch` came back with eighteen `gst_is_initialized()` assertions and a segfault.
// `test_ingest` had never caught it because the GStreamer section ran first in the same binary
// and initialised the library for it -- which is why that order is now reversed and says so.
//
// **THIS HEADER INCLUDES GSTREAMER, WHICH ITS SIBLING `gstreamer_pipeline.h` DELIBERATELY DOES
// NOT.** So it is includable only from a translation unit that already declares an external
// lane carrying `gstreamer-1.0` — today `sources/gstreamer.cpp` and `sources/nvdec.cpp`, and
// `tests/test_build_csrc.py::TestOnlyGstLaneUnitsReachTheBus` derives that set from the lane
// table rather than listing it. The closure walker in `scripts/build_csrc.py` attributes lanes
// to `.cpp` units, so it cannot see a header's own dependency; an offline unit that included
// this would fail with `gst/gst.h: No such file` instead of the lane machinery's own message,
// which is why the guard is a test and not a comment.
#pragma once

#include <gst/gst.h>

#include <cstdlib>
#include <mutex>
#include <string>

#include "shipinfer/core/types.h"

namespace shipinfer {

    // doc: long the once-flag, the proxy resolver, and why `gst_init_check` and not `gst_init`
    // Initialise GStreamer exactly once per process, whoever asks first.
    //
    // The Python loader (`runtime/gstreamer.py`) holds a lock around the *import* as well,
    // because PyGObject resolves `gi.repository` members lazily and a concurrent first touch
    // has come back as "'GLib' object has no attribute 'Idle'". C++ has no such race -- the
    // library is bound at link time -- but `gst_init` must still happen exactly once, and fifty
    // camera threads reach this at start-up together.
    //
    // `gst_init_check` rather than `gst_init`: `gst_init` terminates the process when it fails,
    // and a server must not `exit(1)` because a plugin registry was unwritable. A throw out of
    // the lambda leaves the flag unset, so the next camera legitimately retries.
    inline void initialise_gstreamer() {
        static std::once_flag once;
        std::call_once(once, [] {
            // doc: long the libproxy crash this one line prevents, which killed a whole run
            // `rtspsrc` asks GIO for a proxy resolver before it connects, and GIO's default on
            // a desktop-less system is libproxy, which throws a C++
            // `std::runtime_error("Unable to read configuration")` when it finds no GSettings
            // or D-Bus to read. Uncaught across the C boundary that is `terminate` for the
            // whole process, which is how the first containerised RTSP run died with fifty
            // cameras connected and zero frames decoded. GIO's documented override selects its
            // no-op resolver instead. The trailing `0` is `overwrite=false` -- the exact
            // `os.environ.setdefault` of `runtime/gstreamer.py`, so an operator who has
            // configured a real proxy keeps it.
            ::setenv("GIO_USE_PROXY_RESOLVER", "dummy", 0);
            GError* error = nullptr;
            if (gst_init_check(nullptr, nullptr, &error) == FALSE) {
                const std::string reason = (error != nullptr && error->message != nullptr)
                                               ? error->message
                                               : "(no message)";
                if (error != nullptr) g_error_free(error);
                // Fatal on purpose: a GStreamer that cannot initialise will not start working
                // on its own, so the actor must not spend a reconnect budget on it. Note what
                // is *not* here -- the Python plane's "PyGObject is not importable" case. These
                // units link against `libgstreamer-1.0`, so a missing runtime is a link
                // failure, not something a camera can discover at connect time. A missing
                // *plugin* is still discoverable, and `select_decoder` / `select_converter`
                // raise the same error for it.
                throw SourceUnavailableError("gstreamer", "gst_init failed: " + reason);
            }
        });
    }

    // Throws `FrameDecodeError` if `pipeline`'s bus is holding an ERROR or an EOS, else
    // returns. Pops ONE message: the actor reconnects on the first, so draining the rest would
    // only discard the context a later read could report.
    //
    // EOS on a camera is a fault, not an end — a live stream that stops has broken — so this
    // raises for it too, and neither source overrides `is_exhausted()`.
    inline void raise_if_stream_ended(const std::string& camera_id, GstElement* pipeline) {
        GstBus* bus = pipeline != nullptr ? gst_element_get_bus(pipeline) : nullptr;
        if (bus == nullptr) return;
        GstMessage* message = gst_bus_pop_filtered(
            bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
        gst_object_unref(bus);
        if (message == nullptr) return;

        std::string reason = "end of stream";
        if (GST_MESSAGE_TYPE(message) != GST_MESSAGE_EOS) {
            GError* error = nullptr;
            gchar* debug = nullptr;
            gst_message_parse_error(message, &error, &debug);
            reason = (error != nullptr && error->message != nullptr) ? error->message
                                                                     : "(no message)";
            // The debug string is where GStreamer puts the element that failed and the file and
            // line it failed at, which is the half of the message worth having.
            reason += " (" + std::string(debug != nullptr ? debug : "") + ")";
            if (error != nullptr) g_error_free(error);
            g_free(debug);
        }
        gst_message_unref(message);
        throw FrameDecodeError(camera_id, reason);
    }

}  // namespace shipinfer
