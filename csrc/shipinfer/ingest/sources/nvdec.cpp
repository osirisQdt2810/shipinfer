#include "shipinfer/ingest/sources/nvdec.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include "shipinfer/core/options.h"
#include "shipinfer/core/types.h"
#include "shipinfer/ingest/registry.h"
#include "shipinfer/ingest/sources/gstreamer_pipeline.h"

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
        //: How long a read waits for one access unit. Short, because a camera that has gone
        //: quiet must come back as "nothing yet" rather than as a stall (`base.h`).
        constexpr int kPullTimeoutMs = 100;

        void unref_sample(GstSample* sample) {
            if (sample != nullptr) gst_sample_unref(sample);
        }

    }  // namespace

    // Everything that needs a GStreamer or a cuvid type. Behind a pimpl so `nvdec.h` parses on
    // a machine with neither -- see that header on why the closure walker is the real guard and
    // this is defence in depth.
    struct NvdecSource::Decoder {
        // -- the bitstream half ----------------------------------------------------------
        GstElement* pipeline = nullptr;
        GstAppSink* sink = nullptr;

        // -- the decode half -------------------------------------------------------------
        CudaFunctions* cuda = nullptr;
        CuvidFunctions* cuvid = nullptr;
        CUcontext context = nullptr;
        CUdevice device = 0;
        int device_index = 0;
        bool context_pushed = false;
        CUvideoctxlock lock = nullptr;
        CUvideoparser parser = nullptr;
        CUvideodecoder decoder = nullptr;

        //: What the SEQUENCE callback reported, which is the only place the coded height is
        //: stated. `uv_offset` is `pitch * coded_height`, and nothing else knows it: the
        //: display rect is smaller, and the pitch says nothing about the padding.
        int coded_height = 0;
        int display_height = 0;
        int display_width = 0;
        int surfaces = kDefaultSurfaces;

        //: ONE picture, newest wins -- the appsink's own `drop=true max-buffers=2` policy one
        //: stage along, and for the same reason: a five-second-old frame is not worth a GPU.
        //: Not a queue, because a queue here holds decode surfaces out of a pool of four.
        std::optional<CUVIDPARSERDISPINFO> ready;
        uint64_t dropped = 0;
        //: Raised by a callback, thrown by `do_read_device`. A callback that throws unwinds
        //: through cuvid's C frames, which is undefined; so it records and the caller raises.
        std::string failure;

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
        if (self->decoder != nullptr) {
            // A resolution change mid-stream. Refused rather than reconfigured: the chain
            // above was built for one extent (the crop elements resize to a fixed size and
            // the mask fold counts cells in a fixed bank), so a new one is a reconnect.
            self->failure = "the stream changed resolution mid-flight";
            return 0;
        }
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
        create.ulNumDecodeSurfaces = static_cast<unsigned>(self->surfaces);
        create.vidLock = self->lock;
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
        // KEPT, because it is the only statement of where the chroma plane begins. The
        // output surface is `ulTargetHeight` tall in the display sense and padded to the
        // coded height in memory: `uv_offset = pitch * coded_height`, and getting that
        // wrong reads luma rows as chroma (`runtime/ops.h`).
        self->coded_height = static_cast<int>(format->coded_height);

        if (self->cuvid->cuvidCreateDecoder(&self->decoder, &create) != CUDA_SUCCESS) {
            self->failure = "cuvidCreateDecoder refused this stream";
            return 0;
        }
        return 1;
    }

    int CUDAAPI NvdecSource::Decoder::on_decode(void* user, CUVIDPICPARAMS* picture) {
        auto* self = static_cast<NvdecSource::Decoder*>(user);
        if (self->decoder == nullptr) return 0;
        if (self->cuvid->cuvidDecodePicture(self->decoder, picture) != CUDA_SUCCESS) {
            self->failure = "cuvidDecodePicture failed";
            return 0;
        }
        return 1;
    }

    int CUDAAPI NvdecSource::Decoder::on_display(void* user, CUVIDPARSERDISPINFO* info) {
        auto* self = static_cast<NvdecSource::Decoder*>(user);
        if (self->ready) ++self->dropped;  // newest wins
        self->ready = *info;
        return 1;
    }

    void NvdecSource::Decoder::close() {
        if (pipeline != nullptr) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
            gst_object_unref(pipeline);
            pipeline = nullptr;
            sink = nullptr;  // owned by the pipeline
        }
        if (parser != nullptr && cuvid != nullptr) {
            cuvid->cuvidDestroyVideoParser(parser);
            parser = nullptr;
        }
        if (decoder != nullptr && cuvid != nullptr) {
            cuvid->cuvidDestroyDecoder(decoder);
            decoder = nullptr;
        }
        if (lock != nullptr && cuvid != nullptr) {
            cuvid->cuvidCtxLockDestroy(lock);
            lock = nullptr;
        }
        // The PRIMARY context is released, not destroyed: it is shared with the runtime API the
        // rest of this plane uses, which is the whole reason for retaining that one rather than
        // creating a private context whose surfaces `nv12_letterbox_into` could not read.
        if (context_pushed && cuda != nullptr) {
            CUcontext popped = nullptr;
            cuda->cuCtxPopCurrent(&popped);
            context_pushed = false;
        }
        if (context != nullptr && cuda != nullptr) {
            cuda->cuDevicePrimaryCtxRelease(device);
            context = nullptr;
        }
        if (cuvid != nullptr) cuvid_free_functions(&cuvid);
        if (cuda != nullptr) cuda_free_functions(&cuda);
    }

    NvdecSource::~NvdecSource() {
        NvdecSource::do_close();
    }

    void NvdecSource::do_close() {
        decoder_.reset();
    }

    void NvdecSource::do_open() {
        auto decoder = std::make_unique<Decoder>();
        decoder->surfaces =
            option_int(config().camera_id, config().options, "surfaces", kDefaultSurfaces);
        if (decoder->surfaces < 1) {
            throw ConfigError(
                "camera '" + config().camera_id +
                "': surfaces must be >= 1 (cuvid needs at least the stream's DPB)");
        }

        // -- the driver, dlopen'd ---------------------------------------------------------
        if (cuda_load_functions(&decoder->cuda, nullptr) != 0 || decoder->cuda == nullptr) {
            throw SourceUnavailableError("nvdec",
                                         "libcuda could not be loaded; this source needs an "
                                         "NVIDIA driver on the machine that runs it");
        }
        if (cuvid_load_functions(&decoder->cuvid, nullptr) != 0 || decoder->cuvid == nullptr) {
            throw SourceUnavailableError("nvdec",
                                         "libnvcuvid could not be loaded; it ships with the "
                                         "NVIDIA driver, not with CUDA");
        }
        if (decoder->cuda->cuInit(0) != CUDA_SUCCESS) {
            throw SourceUnavailableError("nvdec", "cuInit failed");
        }
        decoder->device_index = option_int(config().camera_id, config().options, "device", 0);
        if (decoder->cuda->cuDeviceGet(&decoder->device, decoder->device_index) !=
                CUDA_SUCCESS ||
            decoder->cuda->cuDevicePrimaryCtxRetain(&decoder->context, decoder->device) !=
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
        if (decoder->cuda->cuCtxPushCurrent(decoder->context) != CUDA_SUCCESS) {
            throw SourceUnavailableError("nvdec", "could not make CUDA device " +
                                                      std::to_string(decoder->device_index) +
                                                      "'s context current");
        }
        decoder->context_pushed = true;
        // A lock, because cuvid's own decode thread touches the context too: the parser
        // callbacks are synchronous but the decoder's internal engine is not, and the docs
        // require a `vidLock` wherever a context is shared. Cheap, and skipping it is the kind
        // of race that shows as one corrupted frame an hour.
        if (decoder->cuvid->cuvidCtxLockCreate(&decoder->lock, decoder->context) !=
            CUDA_SUCCESS) {
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
        if (decoder->cuvid->cuvidCreateVideoParser(&decoder->parser, &params) != CUDA_SUCCESS) {
            throw SourceOpenError(config().camera_id, config().uri,
                                  "cuvidCreateVideoParser failed");
        }

        // -- the bitstream ----------------------------------------------------------------
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

        // The extent is not known until the first sequence header, which arrives with the first
        // access unit. Reported as zero here and filled in by `do_read_device`, the way the
        // GStreamer source reports what the appsink negotiated rather than what was asked for.
        set_format(0, 0, config().fps);
        decoder_ = std::move(decoder);
    }

    std::optional<DeviceImage> NvdecSource::do_read_device() {
        Decoder& d = *decoder_;
        GstSample* raw = gst_app_sink_try_pull_sample(
            d.sink, static_cast<GstClockTime>(kPullTimeoutMs) * GST_MSECOND);
        if (raw == nullptr) {
            if (gst_app_sink_is_eos(d.sink)) {
                throw FrameDecodeError(config().camera_id,
                                       "the stream ended; a live camera that stops has broken");
            }
            return std::nullopt;  // nothing yet, which consumes no frame id
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
        // undefined -- so the refusal surfaces here, once, with the callback's own words.
        if (!d.failure.empty()) {
            const std::string reason = std::move(d.failure);
            d.failure.clear();
            throw FrameDecodeError(config().camera_id, reason);
        }
        if (parsed != CUDA_SUCCESS) {
            throw FrameDecodeError(config().camera_id, "cuvidParseVideoData refused a packet");
        }
        if (!d.ready) return std::nullopt;  // decoded nothing displayable yet: still "not yet"

        CUVIDPARSERDISPINFO picture = *d.ready;
        d.ready.reset();

        CUdeviceptr frame = 0;
        unsigned pitch = 0;
        CUVIDPROCPARAMS proc{};
        proc.progressive_frame = picture.progressive_frame;
        proc.second_field = picture.repeat_first_field + 1;
        proc.top_field_first = picture.top_field_first;
        proc.unpaired_field = picture.repeat_first_field < 0;
        if (d.cuvid->cuvidMapVideoFrame(d.decoder, picture.picture_index, &frame, &pitch,
                                        &proc) != CUDA_SUCCESS) {
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
        // THE CODED HEIGHT, not the display one. The surface is padded, so the chroma plane
        // begins past the last displayed luma row -- 1088 for 1080p. `pitch * height` here
        // would read the last eight luma rows as chroma on every frame, and
        // `nv12_letterbox_into`'s own guard cannot catch it because that value satisfies it
        // exactly (#153 round 4, #155).
        image.uv_offset = static_cast<size_t>(pitch) * static_cast<size_t>(d.coded_height);
        image.device = d.device_index;
        // Unmaps on release. NVDEC hands out a slot from a pool of `surfaces`, so holding this
        // is what stops the next picture overwriting these pixels -- dropping it early does not
        // free them, it lets them change under a worker.
        CuvidFunctions* cuvid = d.cuvid;
        CUvideodecoder handle = d.decoder;
        image.owner = std::shared_ptr<const void>(
            reinterpret_cast<const void*>(frame), [cuvid, handle, frame](const void*) {
                cuvid->cuvidUnmapVideoFrame(handle, frame);
            });
        return image;
    }

    namespace {

        const SourceRegistrar kRegistrar(
            "nvdec", {"nvv12", "cuvid"},
            "one RTSP camera, decoded by NVDEC into NV12 surfaces that never leave the device",
            [](const IngestConfig& config, FrameCounter& counter,
               StopSignal& stop) -> std::unique_ptr<FrameSource> {
                return std::make_unique<NvdecSource>(config, counter, stop);
            });

    }  // namespace

}  // namespace shipinfer
