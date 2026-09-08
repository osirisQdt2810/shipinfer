#include "shipinfer/ingest/sources/gstreamer.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#include "shipinfer/core/options.h"
#include "shipinfer/core/types.h"
#include "shipinfer/ingest/registry.h"
#include "shipinfer/ingest/sources/gstreamer_pipeline.h"
#include "shipinfer/ingest/sources/gstreamer_shared.h"

namespace shipinfer {

    // The two GStreamer objects this source owns, out of the header so no other translation
    // unit can name a `Gst*` type by accident (see the pimpl note in `gstreamer.h`).
    struct GStreamerSource::Graph {
        GstElement* pipeline = nullptr;
        // Our own ref from `gst_bin_get_by_name`, released in `do_close`.
        GstAppSink* appsink = nullptr;
    };

    namespace {

        // The knobs this source takes from a camera's `options`. Anything else is refused
        // rather than ignored: a deployment that runs for months with a knob wired to nothing
        // is the failure this check exists to prevent.
        const std::vector<std::string> kAcceptedOptions = {"decoder", "max_buffers"};

        // What every refusal from this source names. Fifty cameras in one log means a message
        // that does not say which camera is a message an operator cannot act on — the same
        // subject `replay`'s option refusal carries.
        std::string subject_for(const std::string& camera_id) {
            return "camera '" + camera_id + "': gstreamer";
        }

        // `%g`, which is the format the Python message uses (`{self.open_timeout_s:g}`): "10"
        // rather than the "10.000000" `std::to_string(double)` would put in front of an
        // operator.
        //: How long ONE pull may block, so the loop checks the stop signal at least this
        //: often. Not the read timeout: that is still the deadline for the whole read.
        constexpr int kPullSliceMs = 100;

        std::string seconds_text(double seconds) {
            char buffer[32];
            std::snprintf(buffer, sizeof(buffer), "%g", seconds);
            return buffer;
        }

        // Is this element installed on this host? The probe behind `select_decoder` and
        // `select_converter`, which take it injected so the *selection order* stays testable
        // with no GStreamer at all (`gstreamer_pipeline.h`).
        bool element_installed(const std::string& element) {
            GstElementFactory* factory = gst_element_factory_find(element.c_str());
            if (factory == nullptr) return false;
            // `find` hands back a ref. Fifty cameras probing three elements each leaks 150
            // factory refs without this, once per reconnect, forever.
            gst_object_unref(factory);
            return true;
        }

    }  // namespace

    GStreamerSource::~GStreamerSource() {
        try {
            do_close();
        } catch (...) {  // NOLINT — a destructor must not propagate; `close()` already ran
        }
    }

    // -- lifecycle --------------------------------------------------------------------------

    void GStreamerSource::do_open() {
        // Before GStreamer is touched at all: a typo in a camera's options is a configuration
        // problem, and initialising a plugin registry to discover it is work nobody asked for.
        refuse_unknown_options(subject_for(camera_id()), config().options, kAcceptedOptions);

        initialise_gstreamer();

        const ElementAvailable available = element_installed;

        PipelineOptions options;
        options.codec = config().codec;
        options.latency_ms = config().latency_ms;
        options.transport = config().transport;
        options.converter = select_converter(available);
        options.width = config().width;
        options.height = config().height;
        options.max_buffers =
            option_int(subject_for(camera_id()), config().options, "max_buffers", 2);
        if (options.max_buffers < 1) {
            // The Python plane's _positive_int, mirrored (#46 round 1): max-buffers=0 is
            // GstAppSink's documented "unlimited", and with nothing to drop against the
            // appsink queues every decoded frame — ~124 MB/s of unbounded growth per 1080p
            // camera whenever the consumer falls behind, ending in an OOM kill with no
            // typed event anywhere. The opposite of ADR-005, refused at start-up.
            throw ConfigError(subject_for(camera_id()) +
                              ": max_buffers must be >= 1 (0 is GStreamer's 'unlimited', "
                              "which is an unbounded decoder queue)");
        }

        const auto override_it = config().options.find("decoder");
        if (config().codec == "auto") {
            // `decodebin` names its own decoder by plugin rank at connect time, so there is
            // nothing to select and nothing to override — `build_pipeline` ignores the field in
            // that branch. The override losing to `auto` is the Python order (`_do_open`),
            // kept: an operator who wants a specific element also has to say which codec it
            // decodes.
            options.decoder.clear();
        } else if (override_it != config().options.end()) {
            options.decoder = override_it->second;
        } else {
            // `config().codec` is `auto`, `h264` or `h265` — `IngestConfig::validate()` refused
            // anything else before this camera had a thread, so there is deliberately no second
            // codec check here.
            options.decoder = select_decoder(config().codec, hwaccel(), available);
        }

        // NOT LOGGED, and that is a gap rather than a decision: this plane has no logging
        // framework yet, so there is nowhere for the one line an operator most wants — the
        // resolved pipeline, pasteable into `gst-launch-1.0`. `pipeline_description()` exposes
        // it for whoever gets one first. **When P5 adds logging, the description goes through
        // `redact_in` on the way out**, exactly as `sources/gstreamer.py:276` does: the string
        // embeds `location=<uri>`, and a fleet shares one credential across every camera, so
        // one unredacted log line is the whole fleet's password. Redaction belongs at the site
        // where text leaves the process — never inside `build_pipeline`, whose output has to be
        // the real thing `gst_parse_launch` is handed.
        description_ = build_pipeline(config().uri, options);

        // Owned from the moment it exists. The base's `open()` unwinds a throwing `do_open`
        // with exactly one best-effort `do_close` (`base.h`), and a pipeline that is not on the
        // object yet when the next check throws is a leaked decoder thread and a leaked socket.
        graph_ = std::make_unique<Graph>();

        GError* error = nullptr;
        GstElement* pipeline = gst_parse_launch(description_.c_str(), &error);
        graph_->pipeline = pipeline;
        if (pipeline == nullptr || error != nullptr) {
            // `gst_parse_launch` can hand back a *partially* constructed bin with the error set
            // (that is `GST_PARSE_FLAG_FATAL_ERRORS` being off), and a half-built graph never
            // negotiates. PyGObject raises on the same condition, so both planes refuse it.
            std::string reason = "pipeline would not parse";
            if (error != nullptr && error->message != nullptr) {
                // Passed through raw: `SourceOpenError`'s constructor redacts `reason` itself
                // (`core/types.h`), and redacting twice here would mask a `***` that is already
                // a mask.
                reason += std::string(": ") + error->message;
            }
            if (error != nullptr) g_error_free(error);
            throw SourceOpenError(camera_id(), config().uri, reason);
        }

        GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), kAppsinkName);
        if (sink == nullptr) {
            // Only reachable if `build_pipeline` and this lookup disagree about the name, which
            // is why they read it from the same constant.
            throw SourceOpenError(
                camera_id(), config().uri,
                std::string("pipeline has no appsink '") + kAppsinkName + "'");
        }
        graph_->appsink = GST_APP_SINK(sink);

        if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            throw SourceOpenError(camera_id(), config().uri,
                                  "pipeline refused to enter PLAYING");
        }
        // PLAYING is asynchronous: `rtspsrc` has not even sent DESCRIBE yet. Block until the
        // state change actually completes, so a wrong URI or a bad credential is a failed
        // `open()` — counted, backed off, visible in health — instead of a stream that silently
        // never delivers and looks like a slow camera.
        GstState state = GST_STATE_NULL;
        GstState pending = GST_STATE_NULL;
        const GstStateChangeReturn changed = gst_element_get_state(
            pipeline, &state, &pending,
            static_cast<GstClockTime>(config().open_timeout_s() * GST_SECOND));
        if (changed != GST_STATE_CHANGE_SUCCESS) {
            throw SourceOpenError(camera_id(), config().uri,
                                  "stream did not start within " +
                                      seconds_text(config().open_timeout_s()) + "s (" +
                                      gst_element_state_change_return_get_name(changed) + ")");
        }
        negotiate_from_appsink();
    }

    void GStreamerSource::negotiate_from_appsink() {
        GstPad* pad = gst_element_get_static_pad(GST_ELEMENT(graph_->appsink), "sink");
        if (pad == nullptr) return;
        GstCaps* caps = gst_pad_get_current_caps(pad);
        gst_object_unref(pad);
        if (caps == nullptr) return;
        if (gst_caps_get_size(caps) != 0) {
            // Borrowed from `caps`, so every read happens before the unref below.
            const GstStructure* structure = gst_caps_get_structure(caps, 0);
            gint height = 0;
            gint width = 0;
            const gboolean ok_h = gst_structure_get_int(structure, "height", &height);
            const gboolean ok_w = gst_structure_get_int(structure, "width", &width);
            double fps = config().fps;
            gint numerator = 0;
            gint denominator = 0;
            if (gst_structure_get_fraction(structure, "framerate", &numerator, &denominator) &&
                denominator != 0) {
                fps = static_cast<double>(numerator) / static_cast<double>(denominator);
            }
            if (ok_h && ok_w) set_format(height, width, fps);
        }
        gst_caps_unref(caps);
    }

    // -- reading ----------------------------------------------------------------------------

    std::optional<HostFrame> GStreamerSource::do_read() {
        // No fallback to the `try-pull-sample` *signal* here. The Python plane needs one
        // because the appsink's methods exist only when the GstApp typelib has been loaded;
        // this unit links `gstreamer-app-1.0` directly, so the call is either there at link
        // time or the build failed.
        // doc: long the pull is SLICED so a stop is noticed, which the design load demanded
        // THE READ TIMEOUT IS SPENT IN SLICES, and the stop signal is checked between them. One
        // pull of `read_timeout_s` (5 s by default) is simpler and was what this did -- but the
        // actor only learns of a stop when `read()` RETURNS, so a fleet whose stop budget is
        // shorter abandons the camera. Measured at the design load: 43 of 50 cameras "did not
        // stop within 0ms" and `bench` exited without unwinding. The DEADLINE is unchanged; the
        // slice only decides how often the loop comes up for air. `sources/nvdec.cpp` does the
        // same for the same reason, and the two must not differ about it.
        GstSample* sample = nullptr;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(std::max(1, config().read_timeout_ms));
        while (sample == nullptr) {
            if (stop().is_set()) return std::nullopt;
            const auto left = deadline - std::chrono::steady_clock::now();
            if (left <= std::chrono::steady_clock::duration::zero()) break;
            // A SLICE OF ZERO NANOSECONDS IS THE DEADLINE, not a pull. Without this the tail
            // of every timed-out read is a busy spin: the pull returns at once, `left` stays
            // positive, and the loop turns until the clock catches up. The Python twin had the
            // same shape and its test is what found it.
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(left).count();
            const int64_t slice = std::min<int64_t>(ns, kPullSliceMs * 1000000LL);
            if (slice <= 0) break;
            sample =
                gst_app_sink_try_pull_sample(graph_->appsink, static_cast<GstClockTime>(slice));
            // THE BUS IS ASKED PER SLICE, and this is what the first version of the slicing got
            // wrong: it asked only after the whole deadline had run out, so an EOS -- which the
            // single full-timeout pull used to report on its FIRST null return -- took the
            // entire `read_timeout_ms` to notice, spinning through fifty slices to get there.
            // Slower to detect AND busier while detecting, on the run whose own body names the
            // CPU as the contended resource. `nvdec.cpp` asks inside its own loop for exactly
            // this reason, and the comment above claims the two do not differ about it.
            if (sample == nullptr) raise_if_stream_ended(camera_id(), graph_->pipeline);
        }
        if (sample == nullptr) {
            // The pure-timeout case: the loop asked the bus after every null slice and it was
            // empty each time, so this camera is quiet rather than over and the actor's
            // empty-read budget is what decides.
            return std::nullopt;
        }

        GstCaps* caps = gst_sample_get_caps(sample);  // borrowed from the sample
        gint width = 0;
        gint height = 0;
        if (caps != nullptr && gst_caps_get_size(caps) != 0) {
            const GstStructure* structure = gst_caps_get_structure(caps, 0);
            gst_structure_get_int(structure, "width", &width);
            gst_structure_get_int(structure, "height", &height);
        }
        if (width <= 0 || height <= 0) {
            // Typed rather than tolerated. The Python plane reaches `stride = ((None * 3) + 3)`
            // here and dies of a `TypeError` the actor cannot classify; returning a zero-sized
            // frame instead would put an "image" of no bytes onto the queue, which is worse
            // than either.
            gst_sample_unref(sample);
            throw FrameDecodeError(camera_id(), "sample carries no video size");
        }
        if (height != this->height() || width != this->width()) {
            // Mid-stream re-negotiation. An RTSP camera whose profile changes while connected
            // sends new caps rather than a new connection, and a source still reporting the old
            // size would letterbox against the wrong numbers from here on.
            set_format(height, width, fps() != 0.0 ? fps() : config().fps);
        }

        // GStreamer pads each row of raw video to a multiple of 4 bytes. For 3-byte BGR that
        // only matters at widths not divisible by 4 — which is exactly the case a naive reshape
        // gets wrong, and only for some cameras.
        const size_t stride = static_cast<size_t>(((width * 3) + 3) & ~3);
        const size_t row = static_cast<size_t>(width) * 3;
        GstBuffer* buffer = gst_sample_get_buffer(sample);  // borrowed from the sample
        // Checked *before* the map, so the failure path below has one owned handle to release
        // instead of two.
        if (buffer == nullptr ||
            gst_buffer_get_size(buffer) < stride * static_cast<size_t>(height)) {
            gst_sample_unref(sample);
            throw FrameDecodeError(camera_id(), "frame buffer is shorter than its own caps");
        }
        GstMapInfo info;
        if (gst_buffer_map(buffer, &info, GST_MAP_READ) == FALSE) {
            gst_sample_unref(sample);
            throw FrameDecodeError(camera_id(), "could not map the frame buffer");
        }

        // THE COPY IS NOT OPTIONAL. `unmap` returns the buffer to the decoder's pool, which
        // will overwrite it while a zero-copy view is still being read downstream — a bug that
        // produces plausible-looking frames and is invisible to a test that submits the same
        // image twice (`sources/gstreamer.py:356-360`). The allocation per frame is what the
        // Python plane's `image.copy()` also pays; a frame pool, if one is ever wanted,
        // replaces this line and nothing else.
        auto pixels = std::make_shared<std::vector<uint8_t>>(row * static_cast<size_t>(height));
        if (stride == row) {
            std::memcpy(pixels->data(), info.data, pixels->size());
        } else {
            for (int y = 0; y < height; ++y) {
                std::memcpy(pixels->data() + static_cast<size_t>(y) * row,
                            info.data + static_cast<size_t>(y) * stride, row);
            }
        }
        gst_buffer_unmap(buffer, &info);
        gst_sample_unref(sample);

        HostFrame frame;
        frame.pixels = pixels->data();
        frame.height = height;
        frame.width = width;
        // The owner *is* the copy: nothing else keeps these pages alive once the sample is
        // gone.
        frame.owner = std::move(pixels);
        return frame;
    }

    void GStreamerSource::do_close() {
        // Tolerates a partial open: `do_open` throwing anywhere after `make_unique` gets
        // exactly one `do_close`, and it may be holding a pipeline with no appsink or neither.
        if (!graph_) return;
        if (graph_->pipeline != nullptr) {
            // NULL first, so the decoder threads are joined and the RTSP session is torn down
            // before the last ref goes.
            gst_element_set_state(graph_->pipeline, GST_STATE_NULL);
        }
        if (graph_->appsink != nullptr) {
            gst_object_unref(graph_->appsink);
            graph_->appsink = nullptr;
        }
        if (graph_->pipeline != nullptr) {
            gst_object_unref(graph_->pipeline);
            graph_->pipeline = nullptr;
        }
        graph_.reset();
    }

    namespace {

        const SourceRegistrar kRegistrar(
            "gstreamer", {"gst"},
            "one RTSP camera, decoded by a GStreamer pipeline into BGR frames",
            [](const IngestConfig& config, FrameCounter& counter,
               StopSignal& stop) -> std::unique_ptr<FrameSource> {
                return std::make_unique<GStreamerSource>(config, counter, stop);
            });

    }  // namespace

}  // namespace shipinfer
