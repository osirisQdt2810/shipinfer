#include "shipinfer/ingest/sources/gstreamer.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

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
        std::string seconds_text(double seconds) {
            char buffer[32];
            std::snprintf(buffer, sizeof(buffer), "%g", seconds);
            return buffer;
        }

        // Initialise GStreamer exactly once per process, whoever asks first.
        //
        // The Python loader (`runtime/gstreamer.py`) holds a lock around the *import* as well,
        // because PyGObject resolves `gi.repository` members lazily and a concurrent first
        // touch has come back as "'GLib' object has no attribute 'Idle'". C++ has no such race
        // — the library is bound at link time — but `gst_init` must still happen exactly once,
        // and fifty camera threads reach this at start-up together.
        //
        // `gst_init_check` rather than `gst_init`: `gst_init` terminates the process when it
        // fails, and a server must not `exit(1)` because a plugin registry was unwritable. A
        // throw out of the lambda leaves the flag unset, so the next camera legitimately
        // retries.
        void initialise_gstreamer() {
            static std::once_flag once;
            std::call_once(once, [] {
                // `rtspsrc` asks GIO for a proxy resolver before it connects, and GIO's default
                // on a desktop-less system is libproxy, which throws a C++
                // `std::runtime_error("Unable to read configuration")` when it finds no
                // GSettings or D-Bus to read. Uncaught across the C boundary that is
                // `terminate` for the whole process, which is how the first containerised RTSP
                // run died with fifty cameras connected and zero frames decoded. GIO's
                // documented override selects its no-op resolver instead. The trailing `0` is
                // `overwrite=false` — the exact `os.environ.setdefault` of
                // `runtime/gstreamer.py:57`, so an operator who has configured a real proxy
                // keeps it.
                ::setenv("GIO_USE_PROXY_RESOLVER", "dummy", 0);
                GError* error = nullptr;
                if (gst_init_check(nullptr, nullptr, &error) == FALSE) {
                    const std::string reason = (error != nullptr && error->message != nullptr)
                                                   ? error->message
                                                   : "(no message)";
                    if (error != nullptr) g_error_free(error);
                    // Fatal on purpose: a GStreamer that cannot initialise will not start
                    // working on its own, so the actor must not spend a reconnect budget on it.
                    // Note what is *not* here — the Python plane's "PyGObject is not
                    // importable" case. This unit links against `libgstreamer-1.0`, so a
                    // missing runtime is a link failure, not something a camera can discover at
                    // connect time. A missing *plugin* is still discoverable, and
                    // `select_decoder` / `select_converter` raise the same error for it.
                    throw SourceUnavailableError("gstreamer", "gst_init failed: " + reason);
                }
            });
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
        GstSample* sample = gst_app_sink_try_pull_sample(
            graph_->appsink, static_cast<GstClockTime>(config().read_timeout_s() * GST_SECOND));
        if (sample == nullptr) {
            // Nothing within the timeout. Distinguish "quiet" from "over" by asking the bus: an
            // EOS or an ERROR means reconnect, a timeout means keep waiting.
            raise_if_stream_ended();
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

    void GStreamerSource::raise_if_stream_ended() {
        GstBus* bus = gst_element_get_bus(graph_->pipeline);
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
        throw FrameDecodeError(camera_id(), reason);
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
