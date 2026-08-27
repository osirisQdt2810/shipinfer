// The `gst-launch`-compatible pipeline description for one camera, and the element choices
// behind it — the PURE half of `ingest/sources/gstreamer.py`, ported.
//
// **The pipeline is built as a string and logged.** That is the single most useful debugging
// decision available here: an operator with a camera that will not connect can paste the
// logged line into `gst-launch-1.0`, add `-v`, and watch the negotiation fail for themselves —
// without this server, without a Python interpreter, and without asking anyone. Building the
// same graph element by element through `gst_element_factory_make` is more "correct" and
// produces nothing a human can reproduce by hand.
//
// **NO GSTREAMER IN THIS HEADER, AND THAT IS THE WHOLE POINT.** Every string below is a pure
// function of its arguments, so the exact line an operator would paste is assertable with no
// GStreamer installed, no camera and no network — which is the property
// `tests/ingest/test_sources_gstreamer.py` has on the Python side, kept here for the offline
// C++ tier. The unit that links against `libgstreamer-1.0` and implements the `FrameSource` is
// a separate one, and by the offline-closure invariant at the top of `ingest/registry.cpp` it
// must stay unreachable from anything an offline binary compiles. This header is that unit's
// *sibling*, not its header: anything may include this, including a test binary built with g++
// alone.
//
// The elements are **probed, not assumed**. `nvv4l2decoder` exists on a DeepStream install and
// nowhere else; `nvh264dec` comes with `gst-plugins-bad`'s nvcodec; a plain Ubuntu box has
// neither. The reference implementation hard-coded `avdec_h264` — software decode for fifty 4K
// streams — and then commented the whole pipeline out in favour of `cv2.VideoCapture`, which is
// the same software decode with less control. Probing is why this one can use the video engine
// when it is there and still start when it is not. The probe is *injected*
// (`ElementAvailable`), so the selection order is testable here too while the plugin-registry
// lookup that implements it stays in the gst-linked unit.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"

namespace shipinfer {

    // The name given to the appsink in the generated pipeline, so the source can find that
    // element again once GStreamer has parsed the string back into a graph.
    inline constexpr const char* kAppsinkName = "shipinfer_sink";

    // Is this element installed on this host? Normally a probe of the GStreamer plugin
    // registry; a set's membership in a test.
    using ElementAvailable = std::function<bool(const std::string&)>;

    // Private to this header. Nested under `detail` rather than added to it, because
    // `core/redact.h` also contributes to `shipinfer::detail` and two headers defining one
    // `inline` name (`converters`, `bracketed`) is an ODR clash the linker will not report.
    namespace detail::gst {

        // RTP depayloader and parser per codec. `auto` has neither: it uses `decodebin`.
        // Only ever reached with a codec `build_pipeline` has already validated.
        inline std::string depay_for(const std::string& codec) {
            return codec == "h265" ? "rtph265depay" : "rtph264depay";
        }
        inline std::string parse_for(const std::string& codec) {
            return codec == "h265" ? "h265parse" : "h264parse";
        }

        // Hardware decoders in preference order. `nvv4l2decoder` first because on a DeepStream
        // host it is the one that keeps frames in NVMM memory; the nvcodec elements are the
        // desktop/`gst-plugins-bad` equivalent. An unrecognised codec has no candidates rather
        // than a wrong one — `select_decoder` then reports an install problem naming what it
        // tried, which is the same thing the Python `dict.get(codec, ())` does.
        inline std::vector<std::string> hw_decoders(const std::string& codec) {
            if (codec == "h264") return {"nvv4l2decoder", "nvh264dec"};
            if (codec == "h265") return {"nvv4l2decoder", "nvh265dec"};
            return {};
        }
        inline std::vector<std::string> sw_decoders(const std::string& codec) {
            if (codec == "h264") return {"avdec_h264"};
            if (codec == "h265") return {"avdec_h265"};
            return {};
        }

        // NVMM/CUDA memory straight out of a hardware decoder; `videoconvert` is the portable
        // system-memory fallback and is always present.
        inline std::vector<std::string> converters() {
            return {"nvvideoconvert", "nvvidconv", "videoconvert"};
        }

        // Decoders that can output GL memory and will open a GL display to do it. See the
        // filter in `build_pipeline`.
        inline bool is_gl_capable(const std::string& element) {
            return element == "nvh264dec" || element == "nvh265dec";
        }

        // `[a, b, c]`. The Python messages render `list(candidates)`, whose repr quotes every
        // element; unquoted here to match the bracketed lists this plane's other messages use
        // (`ingest/config.h`'s codec refusal). The names are what an operator greps for in
        // `gst-inspect-1.0`, and quotes get in the way of that.
        inline std::string bracketed(const std::vector<std::string>& names) {
            std::string out = "[";
            for (size_t i = 0; i < names.size(); ++i) {
                if (i != 0) out += ", ";
                out += names[i];
            }
            return out + "]";
        }

    }  // namespace detail::gst

    // Pick the best installed decoder element for `codec`.
    //
    // `hwaccel` tries the hardware decoders first. `codec` is `h264` or `h265`; `auto` never
    // reaches here, because `decodebin` does its own selection by plugin rank.
    //
    // Throws `SourceUnavailableError` when no decoder for this codec is installed at all,
    // hardware or software. That is an install problem, not a camera problem: it will never fix
    // itself, so the actor must not spend its reconnect budget on it (`core/types.h`).
    inline std::string select_decoder(const std::string& codec, bool hwaccel,
                                      const ElementAvailable& available) {
        std::vector<std::string> candidates;
        if (hwaccel) {
            const std::vector<std::string> hardware = detail::gst::hw_decoders(codec);
            candidates.insert(candidates.end(), hardware.begin(), hardware.end());
        }
        const std::vector<std::string> software = detail::gst::sw_decoders(codec);
        candidates.insert(candidates.end(), software.begin(), software.end());
        for (const std::string& element : candidates) {
            if (available(element)) return element;
        }
        throw SourceUnavailableError("gstreamer",
                                     "no " + codec + " decoder found (tried " +
                                         detail::gst::bracketed(candidates) +
                                         "); install gstreamer1.0-libav for software decode, "
                                         "or the nvcodec/DeepStream plugins");
    }

    // Pick a colour converter, preferring the ones that can read decoder memory.
    inline std::string select_converter(const ElementAvailable& available) {
        for (const std::string& element : detail::gst::converters()) {
            if (available(element)) return element;
        }
        throw SourceUnavailableError(
            "gstreamer", "none of " + detail::gst::bracketed(detail::gst::converters()) +
                             " is installed; install gstreamer1.0-plugins-base "
                             "for videoconvert");
    }

    // What `build_pipeline` needs besides the URI.
    //
    // A struct rather than nine parameters because the Python original takes them keyword-only
    // and for the same reason: `width` and `max_buffers` are both ints, and a positional call
    // that swapped two of these would build a pipeline that silently negotiates the wrong thing
    // rather than failing to compile. Every default is the Python default, field for field.
    struct PipelineOptions {
        // `h264`, `h265`, or `auto` for a `decodebin` that negotiates the codec at connect
        // time.
        std::string codec = "h264";
        // `rtspsrc`'s jitter buffer. A direct latency cost, so keep it small.
        int latency_ms = 200;
        // `tcp` (default), `udp`, or `auto` to omit the property and let `rtspsrc` choose.
        std::string transport = "tcp";
        // The decoder element name, from `select_decoder`. Empty falls back to the software
        // decoder for `codec`, so a hand-written call cannot silently pair an H.265 stream with
        // an H.264 decoder — a pipeline that never links.
        std::string decoder;
        // The colour converter element name, from `select_converter`.
        std::string converter = "videoconvert";
        // Scale in the pipeline instead of on the host. Both or neither. 0 keeps the native
        // resolution — the sentinel `IngestConfig` already uses for the same pair, where the
        // Python plane spells it `None`.
        int width = 0;
        int height = 0;
        // appsink queue depth. `drop=true` plus a depth of 2 means the newest frame wins, which
        // for live perception is the only sane policy: a 5-second-old frame is not worth a GPU.
        int max_buffers = 2;
        std::string appsink_name = kAppsinkName;
    };

    // The one line an operator pastes into `gst-launch-1.0`.
    //
    // Pure, which is what makes the exact strings assertable in the offline tier. The element
    // *choices* are made by `select_decoder` / `select_converter`, which need a plugin registry
    // and therefore cannot be.
    //
    // Throws `ConfigError` on an unknown codec, so a typo in a camera's config fails at
    // start-up instead of producing a pipeline that never negotiates, and on half a scale,
    // because one of `width`/`height` alone has no meaning downstream.
    inline std::string build_pipeline(const std::string& uri, const PipelineOptions& options) {
        const std::string& codec = options.codec;
        if (codec != "h264" && codec != "h265" && codec != "auto") {
            throw ConfigError("unsupported codec '" + codec + "'; expected one of " +
                              detail::gst::bracketed({"auto", "h264", "h265"}));
        }
        if ((options.width == 0) != (options.height == 0)) {
            throw ConfigError("width and height must be given together, or neither");
        }

        std::string source =
            "rtspsrc location=" + uri + " latency=" + std::to_string(options.latency_ms);
        // `protocols=auto` is not a GStreamer value. "Let `rtspsrc` decide" *is* the property
        // being absent, so `auto` emits nothing rather than a value that fails to parse.
        if (options.transport == "tcp" || options.transport == "udp") {
            source += " protocols=" + options.transport;
        }

        std::string decode;
        bool force_system_memory = false;
        if (codec == "auto") {
            // `decodebin` picks the decoder by plugin rank, so it will use NVDEC when the
            // nvcodec/DeepStream plugins are installed and fall back on its own when they are
            // not. The cost is that we no longer know which decoder ran.
            decode = "decodebin";
            force_system_memory = true;
        } else {
            const std::string element = options.decoder.empty()
                                            ? detail::gst::sw_decoders(codec).front()
                                            : options.decoder;
            decode = detail::gst::depay_for(codec) + " ! " + detail::gst::parse_for(codec) +
                     " ! " + element;
            force_system_memory = detail::gst::is_gl_capable(element);
        }
        if (force_system_memory) {
            // System memory, stated. nvcodec's `nvh264dec` / `nvh265dec` can output GL memory,
            // and when downstream leaves the choice open they create a GL display first — which
            // a headless container does not have: `gst_gl_display_gbm_new: could not find or
            // open DRM device`, then a segfault, on the first RTSP benchmark run. A
            // `video/x-raw` filter with **no memory feature** makes the decoder negotiate plain
            // system memory and never touch GL. Deliberately not applied to the DeepStream pair
            // (`nvv4l2decoder` + `nvvideoconvert`), whose NVMM hand-off is the entire reason
            // for choosing them.
            decode += " ! video/x-raw";
        }

        const std::string caps = "video/x-raw,format=BGR";
        const std::string scale = options.width != 0
                                      ? "videoscale ! " + caps +
                                            ",width=" + std::to_string(options.width) +
                                            ",height=" + std::to_string(options.height)
                                      : caps;

        return source + " ! " + decode + " ! " + options.converter + " ! " + scale +
               " ! appsink name=" + options.appsink_name +
               " emit-signals=false sync=false drop=true max-buffers=" +
               std::to_string(options.max_buffers);
    }

}  // namespace shipinfer
