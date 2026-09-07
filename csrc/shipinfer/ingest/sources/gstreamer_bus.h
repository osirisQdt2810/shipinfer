// "Has this stream ended, or is it merely quiet?" — the one question a pull timeout cannot
// answer, asked of the pipeline's bus.
//
// Shared by `sources/gstreamer.cpp` and `sources/nvdec.cpp`. Both reach the same place — a
// `try_pull_sample` that returned nothing — and both have to tell a camera that has gone quiet
// (keep waiting, the actor's empty-read budget decides) from one that has broken (reconnect,
// now, with the reason). Getting that wrong in one of two copies is how a switch flap comes
// out as "5 consecutive empty reads": #156 round 1 was exactly that.
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

#include <string>

#include "shipinfer/core/types.h"

namespace shipinfer {

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
