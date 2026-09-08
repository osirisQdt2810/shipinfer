#include "shipinfer/ingest/sources/nvdec.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "shipinfer/core/options.h"
#include "shipinfer/core/types.h"
#include "shipinfer/ingest/registry.h"
#include "shipinfer/ingest/sources/gstreamer_pipeline.h"
#include "shipinfer/ingest/sources/gstreamer_shared.h"

// The dynlink variants: these declare the API and `dlopen` `libnvcuvid.so` at run time, so this
// unit links against no driver library and a box without one fails at load with a message.
//
// ORDER MATTERS, and defining the guard by hand does not work: `FFNV_DYNLINK_CUDA_H` is
// `dynlink_cuda.h`'s OWN include guard, so predefining it turns that include into a no-op and
// takes `CUresult`, `CUdeviceptr` and half of `CuvidFunctions` with it (`dynlink_loader.h`
// gates the CUDA half on the same macro, which the header sets for itself).
#include <ffnvcodec/dynlink_cuda.h>
#include <ffnvcodec/dynlink_cuviddec.h>
#include <ffnvcodec/dynlink_loader.h>
#include <ffnvcodec/dynlink_nvcuvid.h>

namespace shipinfer {

    namespace {

        //: Surfaces cuvid allocates. Its floor is the stream's DPB; more buys pipelining and
        //: costs VRAM per camera, which at fifty cameras is the number that matters.
        constexpr int kDefaultSurfaces = 4;
        //: The ceiling, which is about a typo rather than about the hardware: an NV12 1080p
        //: surface is ~3 MB, so 64 is ~200 MB for one camera and fifty of those is the box.
        constexpr int kMaxSurfaces = 64;
        //: How long ONE pull may block, so the loop checks the stop signal at least this
        //: often. Not the read timeout: that is the deadline across access units.
        constexpr int kPullSliceMs = 100;

        void unref_sample(GstSample* sample) {
            if (sample != nullptr) gst_sample_unref(sample);
        }

        // EVERYTHING A MAPPED SURFACE STILL NEEDS, and nothing else. Shared, because a frame
        // outlives the source a reconnect replaced -- `frame.h` says so for the host twin, and
        // a device surface is worse: releasing one calls back into cuvid, so a decoder
        // destroyed while a worker still holds a frame is a use-after-free in whichever thread
        // happens to drop it. The last reference destroys the decoder, whoever holds it.
        //
        // Both entry points PUSH THE CONTEXT themselves, which is what makes a release correct
        // from any thread: the push/pop pair is a per-thread stack, so the actor thread's own
        // push (held for the source's life, for the parse path) is unaffected by a second one.
        struct Session {
            CudaFunctions* cuda = nullptr;
            CuvidFunctions* cuvid = nullptr;
            CUdevice device = 0;
            CUcontext context = nullptr;
            CUvideoctxlock lock = nullptr;
            CUvideodecoder decoder = nullptr;
            //: Where cuvid's POST-PROCESSING runs. Left at 0 it is the legacy default stream,
            //: and every stream in `csrc/` is `gpuStreamCreate`'s -- which is BLOCKING, so each
            //: mapped frame made all 23 worker streams on that GPU wait. See `map_next`.
            CUstream output = nullptr;

            ~Session() {
                const bool pushed = on_context();
                if (output != nullptr) cuda->cuStreamDestroy(output);
                if (decoder != nullptr) cuvid->cuvidDestroyDecoder(decoder);
                if (lock != nullptr) cuvid->cuvidCtxLockDestroy(lock);
                if (pushed) off_context();
                // The PRIMARY context is released, not destroyed: it is shared with the runtime
                // API the rest of this plane uses, which is the whole reason for retaining that
                // one rather than creating a private context whose surfaces
                // `nv12_letterbox_into` could not read.
                if (context != nullptr) cuda->cuDevicePrimaryCtxRelease(device);
                if (cuvid != nullptr) cuvid_free_functions(&cuvid);
                if (cuda != nullptr) cuda_free_functions(&cuda);
            }

            void unmap(CUdeviceptr frame) {
                const bool pushed = on_context();
                cuvid->cuvidUnmapVideoFrame(decoder, frame);
                if (pushed) off_context();
            }

          private:
            bool on_context() {
                return context != nullptr && cuda->cuCtxPushCurrent(context) == CUDA_SUCCESS;
            }
            void off_context() {
                CUcontext popped = nullptr;
                cuda->cuCtxPopCurrent(&popped);
            }
        };

    }  // namespace

    // Everything that needs a GStreamer or a cuvid type. Behind a pimpl so `nvdec.h` parses on
    // a machine with neither -- see that header on why the closure walker is the real guard and
    // this is defence in depth.
    struct NvdecSource::Decoder {
        // -- the bitstream half ----------------------------------------------------------
        GstElement* pipeline = nullptr;
        GstAppSink* sink = nullptr;

        // -- the decode half, all of it OWNED BY `session` -------------------------------
        //: `cuda` and `cuvid` are NON-OWNING aliases into it, because every call site reads
        //: better as `d.cuvid->cuvidX` than `d.session->cuvid->cuvidX`. Valid for exactly as
        //: long as `session` is held, which is until the end of `close()`.
        std::shared_ptr<Session> session = std::make_shared<Session>();
        CudaFunctions* cuda = nullptr;
        CuvidFunctions* cuvid = nullptr;
        int device_index = 0;
        bool context_pushed = false;
        CUvideoparser parser = nullptr;

        //: What the SEQUENCE callback reported. The OUTPUT surface's extent, which is the
        //: display rect because nothing here resizes -- see `map_next` on why the coded height
        //: is not this and is not kept.
        int display_height = 0;
        int display_width = 0;
        int surfaces = kDefaultSurfaces;

        //: EVERY displayable picture, oldest first -- not one slot. One `cuvidParseVideoData`
        //: can display more than one picture (a reordering flush, or an access unit carrying
        //: two), and a single slot silently destroyed all but the last: a B-frame camera then
        //: delivered fewer frames than its fps with `frames_read` and `frames_dropped` both
        //: looking healthy (#156 round 1). A display info is an INDEX, not a mapped surface,
        //: and cuvid cannot hand out more than `surfaces` of them before this returns.
        std::deque<CUVIDPARSERDISPINFO> ready;
        //: Raised by a callback, thrown by `feed_one_access_unit`. A callback that throws
        //: unwinds through cuvid's C frames, which is undefined; so it records and the caller
        //: raises. `fatal` picks the type -- a stream this decoder can never handle must stop
        //: the camera rather than spend its reconnect budget forever.
        std::string failure;
        bool fatal = false;

        ~Decoder() { close(); }

        void close();

        // STATIC MEMBERS rather than free functions: `Decoder` is private to `NvdecSource`, so
        // a free callback cannot name it. cuvid invokes these SYNCHRONOUSLY from
        // `cuvidParseVideoData`, so they run on the actor's own thread -- which is what lets
        // this whole decoder be lock-free and keeps every CUDA call on one thread (ADR-002).
        static int CUDAAPI on_sequence(void* user, CUVIDEOFORMAT* format);
        static int CUDAAPI on_decode(void* user, CUVIDPICPARAMS* picture);
        static int CUDAAPI on_display(void* user, CUVIDPARSERDISPINFO* info);
    };

    int CUDAAPI NvdecSource::Decoder::on_sequence(void* user, CUVIDEOFORMAT* format) {
        auto* self = static_cast<NvdecSource::Decoder*>(user);
        Session& session = *self->session;
        if (session.decoder != nullptr) {
            // A resolution change mid-stream. Refused rather than reconfigured: the chain
            // above was built for one extent (the crop elements resize to a fixed size and
            // the mask fold counts cells in a fixed bank), so a new one is a reconnect.
            self->failure = "the stream changed resolution mid-flight";
            return 0;
        }
        if (format->bit_depth_luma_minus8 != 0) {
            // FATAL, because it will never fix itself. `cudaVideoSurfaceFormat_NV12` is
            // 8-bit; a 10-bit stream needs `P016`, and every consumer of `DeviceImage` is
            // written for NV12. Refusing here stops the camera with the reason, where
            // `cuvidCreateDecoder`'s own failure read as "refused this stream" and the actor
            // reconnected on it forever.
            self->fatal = true;
            self->failure = "the stream is " +
                            std::to_string(format->bit_depth_luma_minus8 + 8) +
                            "-bit; this source decodes 8-bit NV12 only (P016 is unimplemented)";
            return 0;
        }
        // THE STREAM'S OWN DPB, as a FLOOR under the knob and never a ceiling over it.
        // `min_num_decode_surfaces` is the one field cuvid fills in to say "this many or
        // decoding is wrong", and nobody knows a camera's reference depth from the outside: an
        // H.264 main stream with three references and three B-frames needs six, so a default of
        // four made `cuvidCreateDecoder` refuse -- and refuse RETRYABLY, which is the
        // reconnect-forever shape the 10-bit branch above exists to prevent. Worse when the
        // create tolerates the low count: the PARSER stays capped, hands a picture index back
        // while it is still a reference, and the output is corrupted with `frames_read`
        // climbing and nothing red anywhere.
        const unsigned needed =
            std::max(static_cast<unsigned>(self->surfaces),
                     static_cast<unsigned>(format->min_num_decode_surfaces));
        self->surfaces = static_cast<int>(needed);  // `on_display`'s bound reads this
        CUVIDDECODECREATEINFO create{};
        create.CodecType = format->codec;
        create.ChromaFormat = format->chroma_format;
        create.OutputFormat = cudaVideoSurfaceFormat_NV12;
        create.bitDepthMinus8 = format->bit_depth_luma_minus8;
        create.DeinterlaceMode = format->progressive_sequence
                                     ? cudaVideoDeinterlaceMode_Weave
                                     : cudaVideoDeinterlaceMode_Adaptive;
        // TWO, not one, and the second is not spare capacity: a consumer holds a mapped
        // surface (the `owner` keepalive) while the next picture is decoded and mapped, so
        // one output surface makes the SECOND frame's `cuvidMapVideoFrame` fail. Observed
        // exactly that -- one surface delivered, then "cuvidMapVideoFrame failed".
        create.ulNumOutputSurfaces = 2;
        create.ulCreationFlags = cudaVideoCreate_PreferCUVID;
        create.ulNumDecodeSurfaces = needed;
        create.vidLock = session.lock;
        create.ulWidth = format->coded_width;
        create.ulHeight = format->coded_height;
        create.ulMaxWidth = format->coded_width;
        create.ulMaxHeight = format->coded_height;
        // The DISPLAY rect is what the camera sends; the coded extent is that rounded up.
        create.display_area.left = static_cast<short>(format->display_area.left);
        create.display_area.top = static_cast<short>(format->display_area.top);
        create.display_area.right = static_cast<short>(format->display_area.right);
        create.display_area.bottom = static_cast<short>(format->display_area.bottom);
        create.ulTargetWidth = format->display_area.right - format->display_area.left;
        create.ulTargetHeight = format->display_area.bottom - format->display_area.top;

        self->display_width = static_cast<int>(create.ulTargetWidth);
        self->display_height = static_cast<int>(create.ulTargetHeight);

        if (self->cuvid->cuvidCreateDecoder(&session.decoder, &create) != CUDA_SUCCESS) {
            self->failure = "cuvidCreateDecoder refused this stream";
            return 0;
        }
        // `> 1` OVERRIDES THE PARSER'S `ulMaxNumDecodeSurfaces`, which is what this return
        // value is for -- 0 is fail and 1 is "succeeded, keep yours". NVIDIA's own
        // `NvDecoder::HandleVideoSequence` creates the parser with 1 and returns
        // `min_num_decode_surfaces` here for exactly this reason; returning 1 leaves the parser
        // cycling through however many indices the knob happened to say.
        return static_cast<int>(needed);
    }

    int CUDAAPI NvdecSource::Decoder::on_decode(void* user, CUVIDPICPARAMS* picture) {
        auto* self = static_cast<NvdecSource::Decoder*>(user);
        CUvideodecoder decoder = self->session->decoder;
        if (decoder == nullptr) return 0;
        if (self->cuvid->cuvidDecodePicture(decoder, picture) != CUDA_SUCCESS) {
            self->failure = "cuvidDecodePicture failed";
            return 0;
        }
        return 1;
    }

    int CUDAAPI NvdecSource::Decoder::on_display(void* user, CUVIDPARSERDISPINFO* info) {
        auto* self = static_cast<NvdecSource::Decoder*>(user);
        if (self->ready.size() >= static_cast<size_t>(self->surfaces)) {
            // Unreachable: cuvid stops decoding when the surface pool is exhausted, so it
            // cannot display more pictures than there are surfaces before this parse returns.
            // Recorded rather than asserted, because a callback must not throw and dropping it
            // silently is the shape this deque exists to remove.
            self->failure = "cuvid displayed more pictures than the surface pool holds";
            return 0;
        }
        self->ready.push_back(*info);
        return 1;
    }

    void NvdecSource::Decoder::close() {
        if (pipeline != nullptr) {
            // NULL first, so the decoder threads are joined and the RTSP session is torn down
            // before the last ref goes.
            gst_element_set_state(pipeline, GST_STATE_NULL);
        }
        if (sink != nullptr) {
            // OUR OWN REF, from `gst_bin_get_by_name`, which is (transfer full): the pipeline's
            // ref is not this one. Leaving it took a `GstAppSink`, its pad, its caps and its
            // queued access units on every reconnect -- unbounded growth in a process whose
            // whole point is to stay up (#156 round 1).
            gst_object_unref(sink);
            sink = nullptr;
        }
        if (pipeline != nullptr) {
            gst_object_unref(pipeline);
            pipeline = nullptr;
        }
        if (parser != nullptr && cuvid != nullptr) {
            cuvid->cuvidDestroyVideoParser(parser);
            parser = nullptr;
        }
        // Popped here, not in `~Session`: this push belongs to the ACTOR thread, and the
        // session may well be destroyed later on whichever thread drops the last surface.
        if (context_pushed && cuda != nullptr) {
            CUcontext popped = nullptr;
            cuda->cuCtxPopCurrent(&popped);
            context_pushed = false;
        }
        // The decoder, the context lock and the function tables go with the session -- NOW if
        // no surface is still mapped, and otherwise when the last one is released.
        cuda = nullptr;
        cuvid = nullptr;
        session.reset();
    }

    NvdecSource::~NvdecSource() {
        NvdecSource::do_close();
    }

    void NvdecSource::do_close() {
        decoder_.reset();
    }

    void NvdecSource::do_open() {
        auto decoder = std::make_unique<Decoder>();
        Session& session = *decoder->session;
        decoder->surfaces =
            option_int(config().camera_id, config().options, "surfaces", kDefaultSurfaces);
        // A FLOOR, so the range is stated at both ends: 1 is "let the stream decide" (the
        // sequence callback raises it to `min_num_decode_surfaces`), and the ceiling is there
        // because `surfaces: 400` is a typo that costs VRAM per camera and nothing else.
        if (decoder->surfaces < 1 || decoder->surfaces > kMaxSurfaces) {
            throw ConfigError("camera '" + config().camera_id + "': surfaces must be 1.." +
                              std::to_string(kMaxSurfaces) +
                              " (the stream's own DPB is the floor; more buys pipelining and "
                              "costs VRAM per camera)");
        }
        // REFUSED, not silently ignored. This source IS the hardware decoder, so
        // `hwaccel: false` on it is a contradiction rather than a preference -- and honouring
        // it by falling back would hand the graph a host frame from a source whose whole
        // contract is that it never produces one. The sibling is the software path.
        if (!config().hwaccel) {
            throw ConfigError("camera '" + config().camera_id +
                              "': hwaccel is false on an 'nvdec' camera, which decodes on the "
                              "video engine by definition; use the 'gstreamer' source for a "
                              "software or a negotiated decoder");
        }

        // -- the driver, dlopen'd ---------------------------------------------------------
        if (cuda_load_functions(&session.cuda, nullptr) != 0 || session.cuda == nullptr) {
            throw SourceUnavailableError("nvdec",
                                         "libcuda could not be loaded; this source needs an "
                                         "NVIDIA driver on the machine that runs it");
        }
        if (cuvid_load_functions(&session.cuvid, nullptr) != 0 || session.cuvid == nullptr) {
            throw SourceUnavailableError("nvdec",
                                         "libnvcuvid could not be loaded; it ships with the "
                                         "NVIDIA driver, not with CUDA");
        }
        decoder->cuda = session.cuda;
        decoder->cuvid = session.cuvid;
        if (session.cuda->cuInit(0) != CUDA_SUCCESS) {
            throw SourceUnavailableError("nvdec", "cuInit failed");
        }
        decoder->device_index = option_int(config().camera_id, config().options, "device", 0);
        if (session.cuda->cuDeviceGet(&session.device, decoder->device_index) != CUDA_SUCCESS ||
            session.cuda->cuDevicePrimaryCtxRetain(&session.context, session.device) !=
                CUDA_SUCCESS) {
            throw SourceUnavailableError(
                "nvdec", "no CUDA device " + std::to_string(decoder->device_index));
        }
        // CURRENT ON THIS THREAD, and it has to be: `cuvidCreateDecoder` and
        // `cuvidMapVideoFrame` operate on the calling thread's context, and retaining a
        // context does not make it current. Without this the sequence callback got as far as
        // `cuvidCreateDecoder` and it "refused this stream" -- a message about the stream for
        // a fault in the caller, which is why the failure read as a codec problem.
        //
        // PUSHED once and popped in `close()`, rather than around every call: one actor thread
        // owns this source for its whole life (ADR-002), so the context is on that thread's
        // stack for exactly as long as the source exists and no other thread is touched.
        // `cuCtxPushCurrent` and not `cuCtxSetCurrent` because the dynlink loader's
        // `CudaFunctions` carries the push/pop pair and not the setter.
        if (session.cuda->cuCtxPushCurrent(session.context) != CUDA_SUCCESS) {
            throw SourceUnavailableError("nvdec", "could not make CUDA device " +
                                                      std::to_string(decoder->device_index) +
                                                      "'s context current");
        }
        decoder->context_pushed = true;
        // doc: long the stream cuvid post-processes on, and the barrier stream 0 was
        // ITS OWN STREAM, NOT THE DEFAULT ONE. `cuvidMapVideoFrame`'s post-processing runs on
        // `CUVIDPROCPARAMS::output_stream`, and 0 means the LEGACY DEFAULT stream -- which
        // every blocking stream in this process waits for. Every stream in `csrc/` comes from
        // `gpuStreamCreate` (`core/platform.h`) and is therefore blocking, so each mapped frame
        // was a device-wide barrier against all 23 worker streams on that GPU: at ~195 frames a
        // second per GPU, 195 barriers a second, none of them ours.
        //
        // NON-BLOCKING, deliberately: a blocking stream here would keep the barrier in the
        // other direction. Nothing else reads these bytes without first waiting on this stream
        // -- `SurfaceIntake` copies out of it on its own stream and synchronises before the
        // frame is queued, which is the one consumer there is.
        if (session.cuda->cuStreamCreate(&session.output, CU_STREAM_NON_BLOCKING) !=
            CUDA_SUCCESS) {
            throw SourceUnavailableError("nvdec",
                                         "could not create the decoder's output "
                                         "stream");
        }
        // A lock, because cuvid's own decode thread touches the context too: the parser
        // callbacks are synchronous but the decoder's internal engine is not, and the docs
        // require a `vidLock` wherever a context is shared. Cheap, and skipping it is the kind
        // of race that shows as one corrupted frame an hour.
        if (session.cuvid->cuvidCtxLockCreate(&session.lock, session.context) != CUDA_SUCCESS) {
            throw SourceUnavailableError("nvdec", "cuvidCtxLockCreate failed");
        }

        // -- the parser -------------------------------------------------------------------
        CUVIDPARSERPARAMS params{};
        params.CodecType = config().codec == "h265" ? cudaVideoCodec_HEVC : cudaVideoCodec_H264;
        params.ulMaxNumDecodeSurfaces = static_cast<unsigned>(decoder->surfaces);
        params.ulMaxDisplayDelay = 0;  // lowest latency: display as soon as decoded
        params.pUserData = decoder.get();
        params.pfnSequenceCallback = Decoder::on_sequence;
        params.pfnDecodePicture = Decoder::on_decode;
        params.pfnDisplayPicture = Decoder::on_display;
        if (session.cuvid->cuvidCreateVideoParser(&decoder->parser, &params) != CUDA_SUCCESS) {
            throw SourceOpenError(config().camera_id, config().uri,
                                  "cuvidCreateVideoParser failed");
        }

        // -- the bitstream ----------------------------------------------------------------
        // FIRST, because `gst_parse_launch` below is the first GStreamer call this unit makes
        // and nothing else in the process need have made one. Missing until the bench ran with
        // `--source nvdec`: eighteen `gst_is_initialized()` assertions and a segfault, in a
        // binary whose `test_ingest` had always passed because its GStreamer section ran first
        // and initialised the library on this unit's behalf.
        initialise_gstreamer();
        PipelineOptions options;
        options.bitstream = true;
        options.codec = config().codec.empty() ? "h264" : config().codec;
        options.latency_ms = config().latency_ms;
        options.transport = config().transport.empty() ? "tcp" : config().transport;
        options.max_buffers =
            option_int(config().camera_id, config().options, "max_buffers", 2);
        description_ = build_pipeline(config().uri, options);

        GError* error = nullptr;
        decoder->pipeline = gst_parse_launch(description_.c_str(), &error);
        if (decoder->pipeline == nullptr) {
            const std::string reason = error != nullptr ? error->message : "unknown";
            if (error != nullptr) g_error_free(error);
            throw SourceOpenError(config().camera_id, config().uri,
                                  "pipeline will not parse: " + reason);
        }
        if (error != nullptr) g_error_free(error);
        GstElement* sink =
            gst_bin_get_by_name(GST_BIN(decoder->pipeline), options.appsink_name.c_str());
        if (sink == nullptr) {
            throw SourceOpenError(config().camera_id, config().uri,
                                  "the pipeline has no appsink named " + options.appsink_name);
        }
        decoder->sink = GST_APP_SINK(sink);
        if (gst_element_set_state(decoder->pipeline, GST_STATE_PLAYING) ==
            GST_STATE_CHANGE_FAILURE) {
            throw SourceOpenError(config().camera_id, config().uri, "PLAYING was refused");
        }
        // PLAYING is asynchronous: `rtspsrc` has not even sent DESCRIBE yet, so ASYNC says
        // nothing about the URI, the credential or the route. Block until the state change
        // completes, so a stale password is a failed `open()` -- counted, backed off, visible
        // in health -- instead of a camera the fleet reports up that never delivers a frame.
        // The sibling does exactly this (`sources/gstreamer.cpp`) and for the same reason.
        GstState state = GST_STATE_NULL;
        GstState pending = GST_STATE_NULL;
        const GstStateChangeReturn changed = gst_element_get_state(
            decoder->pipeline, &state, &pending,
            static_cast<GstClockTime>(config().open_timeout_ms) * GST_MSECOND);
        if (changed != GST_STATE_CHANGE_SUCCESS) {
            throw SourceOpenError(config().camera_id, config().uri,
                                  "stream did not start within " +
                                      std::to_string(config().open_timeout_ms) + "ms (" +
                                      gst_element_state_change_return_get_name(changed) + ")");
        }

        // The extent is not known until the first sequence header, which arrives with the first
        // access unit. Reported as zero here and filled in by `do_read_device`, the way the
        // GStreamer source reports what the appsink negotiated rather than what was asked for.
        set_format(0, 0, config().fps);
        decoder_ = std::move(decoder);
    }

    std::optional<DeviceImage> NvdecSource::do_read_device() {
        Decoder& d = *decoder_;
        // ONE DEADLINE ACROSS SEVERAL ACCESS UNITS, which is where this parts company with the
        // GStreamer source. There one pull is one frame, so `read_timeout_ms` can be the pull
        // timeout. Here a frame is N access units -- SPS/PPS/SEI carry no picture, and a
        // reordering stream displays nothing until the first flush -- so bounding the PULL
        // instead of the read would answer "nothing yet" in microseconds and spend the actor's
        // `empty_reads_before_reconnect` budget (5) in well under a second, reconnecting before
        // the first IDR ever arrived and blaming the network for it (#156 round 1).
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(config().read_timeout_ms);
        while (d.ready.empty()) {
            // THE STOP SIGNAL IS CHECKED EVERY PASS, not only between reads. A read may spend
            // its whole `read_timeout_ms` gathering access units, and the actor only learns of
            // a stop when `read()` returns -- so a fleet whose stop budget is shorter than that
            // abandons every camera and `bench` exits without unwinding. Measured: eight
            // cameras, "did not stop within 0ms", no summary printed at all.
            if (stop().is_set()) return std::nullopt;
            const auto left = deadline - std::chrono::steady_clock::now();
            if (left <= std::chrono::steady_clock::duration::zero()) {
                return std::nullopt;  // quiet, not over: an empty read, which the actor counts
            }
            // ONE PULL IS CAPPED so the stop check above is reached promptly: without it a
            // quiet camera blocks for the whole remaining deadline in a single pull and the
            // check runs once. The DEADLINE is still what bounds the read -- this only decides
            // how often the loop comes back up for air.
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(left).count();
            feed_one_access_unit(
                d, std::min<uint64_t>(static_cast<uint64_t>(ns), kPullSliceMs * 1000000ull));
        }
        return map_next(d);
    }

    void NvdecSource::feed_one_access_unit(Decoder& d, uint64_t timeout_ns) {
        GstSample* raw =
            gst_app_sink_try_pull_sample(d.sink, static_cast<GstClockTime>(timeout_ns));
        if (raw == nullptr) {
            // Nothing left of the deadline. Distinguish "quiet" from "over" by asking the bus:
            // an EOS or an ERROR means reconnect NOW, with the element and line GStreamer put
            // in the debug string, rather than the wrong diagnosis the empty-read budget gives
            // it five reads later.
            raise_if_stream_ended(config().camera_id, d.pipeline);
            return;
        }
        std::unique_ptr<GstSample, void (*)(GstSample*)> sample(raw, unref_sample);

        GstBuffer* buffer = gst_sample_get_buffer(sample.get());
        GstMapInfo map;
        if (buffer == nullptr || !gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            throw FrameDecodeError(config().camera_id, "an access unit could not be mapped");
        }
        CUVIDSOURCEDATAPACKET packet{};
        packet.payload = map.data;
        packet.payload_size = static_cast<unsigned long>(map.size);
        packet.flags = CUVID_PKT_TIMESTAMP;
        packet.timestamp = static_cast<CUvideotimestamp>(GST_BUFFER_PTS(buffer));
        const CUresult parsed = d.cuvid->cuvidParseVideoData(d.parser, &packet);
        gst_buffer_unmap(buffer, &map);

        // A callback records rather than throws -- unwinding through cuvid's C frames is
        // undefined -- so the refusal surfaces here, once, with the callback's own words. A
        // `ConfigError` is fatal for the camera (`CameraActor::pump`), which is what a stream
        // this decoder can never handle needs; everything else is a reconnect.
        if (!d.failure.empty()) {
            const std::string reason = std::move(d.failure);
            d.failure.clear();
            if (d.fatal) throw ConfigError("camera '" + config().camera_id + "': " + reason);
            throw FrameDecodeError(config().camera_id, reason);
        }
        if (parsed != CUDA_SUCCESS) {
            throw FrameDecodeError(config().camera_id, "cuvidParseVideoData refused a packet");
        }
    }

    DeviceImage NvdecSource::map_next(Decoder& d) {
        const CUVIDPARSERDISPINFO picture = d.ready.front();
        d.ready.pop_front();

        CUdeviceptr frame = 0;
        unsigned pitch = 0;
        CUVIDPROCPARAMS proc{};
        proc.progressive_frame = picture.progressive_frame;
        proc.second_field = picture.repeat_first_field + 1;
        proc.top_field_first = picture.top_field_first;
        proc.unpaired_field = picture.repeat_first_field < 0;
        proc.output_stream = d.session->output;  // not stream 0; see `do_open`
        if (d.cuvid->cuvidMapVideoFrame(d.session->decoder, picture.picture_index, &frame,
                                        &pitch, &proc) != CUDA_SUCCESS) {
            throw FrameDecodeError(config().camera_id, "cuvidMapVideoFrame failed");
        }

        if (height() != d.display_height || width() != d.display_width) {
            set_format(d.display_height, d.display_width, config().fps);
        }

        DeviceImage image;
        image.nv12 = reinterpret_cast<const void*>(frame);
        image.height = d.display_height;
        image.width = d.display_width;
        image.pitch = static_cast<int>(pitch);
        // doc: long this unit's own measurement, which is where the rule was settled
        // THE OUTPUT SURFACE'S HEIGHT, which is `ulTargetHeight` -- the display extent, because
        // nothing here resizes. #156 used the CODED height; probing both out of a real mapped
        // surface is what settled it, and the rule now lives once in `ingest/frame.h`:
        //
        //   PROBE pitch=2048 display=1920x1080 coded_h=1088
        //         at_coded=cudaErrorInvalidValue  at_display=cudaSuccess
        image.uv_offset = static_cast<size_t>(pitch) * static_cast<size_t>(d.display_height);
        image.device = d.device_index;
        // Unmaps on release, and holds the DECODER alive to do it. NVDEC hands out a slot from
        // a pool of `surfaces`, so holding this is what stops the next picture overwriting
        // these pixels -- dropping it early does not free them, it lets them change under a
        // worker. The `Session` reference is the other half: a frame outlives the source a
        // reconnect replaced (`frame.h`), and without it `cuvidDestroyDecoder` could run first
        // and the release would call through a freed function table. `unmap` pushes the
        // context itself, so any thread may be the one that drops the last reference.
        std::shared_ptr<Session> session = d.session;
        image.owner = std::shared_ptr<const void>(
            reinterpret_cast<const void*>(frame),
            [session, frame](const void*) { session->unmap(frame); });
        return image;
    }

    namespace {

        const SourceRegistrar kRegistrar(
            "nvdec", {"nvv12", "cuvid"},
            "one RTSP camera, decoded by NVDEC into NV12 surfaces that never leave the device",
            [](const IngestConfig& config, FrameCounter& counter,
               StopSignal& stop) -> std::unique_ptr<FrameSource> {
                return std::make_unique<NvdecSource>(config, counter, stop);
            },
            // DEVICE FRAMES, declared: the only source that answers from `do_read_device`, and
            // the fact a caller needs before any camera connects (`ingest/registry.h`).
            /*device_frames=*/true);

    }  // namespace

}  // namespace shipinfer
