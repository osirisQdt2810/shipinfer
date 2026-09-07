// `ingest/frame.h`'s DeviceImage and the one-hook latch in `ingest/base.cpp`.
//
// V156's route is `rtsp -> nv12 -> tren vram het -> xu ly tren vram toan bo`, so a frame has
// to be able to reference VRAM. The carrier is the first half of that and it is testable with
// no device at all: a device POINTER is an address, and nothing here dereferences one.
//
// Offline: g++ alone, no CUDA, no TensorRT.

#include <cstdio>
#include <memory>
#include <optional>
#include <string>

#include "shipinfer/core/types.h"
#include "shipinfer/ingest/base.h"
#include "shipinfer/ingest/frame.h"

namespace {

    using namespace shipinfer;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    //: A plausible NVDEC surface: 1080p, padded to a 256-byte pitch, on device 3.
    DeviceImage a_surface(std::shared_ptr<const void> owner = nullptr) {
        DeviceImage image;
        image.nv12 = reinterpret_cast<const void*>(0xdeadbeef000ULL);
        image.height = 1080;
        image.width = 1920;
        image.pitch = 2048;
        image.device = 3;
        image.owner = std::move(owner);
        return image;
    }

    IngestConfig a_camera(const std::string& id) {
        IngestConfig config;
        config.camera_id = id;
        config.uri = "rtsp://127.0.0.1:8554/cam0";
        config.source = "fake";
        config.fps = 20.0;
        return config;
    }

    // A source that answers from whichever hook it was told to, so the latch can be provoked.
    class SwitchingSource : public FrameSource {
      public:
        SwitchingSource(IngestConfig config, FrameCounter& counter, StopSignal& stop)
            : FrameSource(std::move(config), counter, stop) {}

        bool next_is_device = true;
        //: Leave the device index at its default, the way a decoder that filled everything
        //: else and forgot this one would. `-1` is what `DeviceImage` starts with.
        bool forget_device_index = false;
        std::vector<uint8_t> pixels{1, 2, 3};

      protected:
        void do_open() override { set_format(1080, 1920, 20.0); }
        void do_close() override {}

        std::optional<DeviceImage> do_read_device() override {
            if (!next_is_device) return std::nullopt;
            DeviceImage image = a_surface();
            if (forget_device_index) image.device = -1;
            return image;
        }
        std::optional<HostFrame> do_read() override {
            HostFrame frame;
            frame.pixels = pixels.data();
            frame.height = 1;
            frame.width = 1;
            frame.owner = std::shared_ptr<const void>(pixels.data(), [](const void*) {});
            return frame;
        }
    };

    void a_device_image_needs_no_cuda_to_carry() {
        const DeviceImage image = a_surface();

        check(!image.empty(), "a full surface is not empty");
        check(image.pitch > image.width, "the pitch is padded past the width, as NVDEC's is");
        check(image.device == 3, "and it remembers which device it belongs to");
    }

    void an_incomplete_surface_is_empty_rather_than_trusted() {
        DeviceImage no_pointer = a_surface();
        no_pointer.nv12 = nullptr;
        DeviceImage no_size = a_surface();
        no_size.height = 0;
        // The one that is easy to get wrong: a pitch BELOW the width cannot describe a row, so
        // it is a mis-filled surface rather than a tight one. Reading it walks off each row.
        DeviceImage short_pitch = a_surface();
        short_pitch.pitch = 1024;

        check(no_pointer.empty(), "no pointer, no image");
        check(no_size.empty(), "no height, no image");
        check(short_pitch.empty(), "a pitch under the width cannot hold a row");
        DeviceImage no_device = a_surface();
        no_device.device = -1;  // the field's own default
        check(no_device.empty(),
              "and no device index, which a consumer would compare against its own bound GPU");
        DeviceImage tight = a_surface();
        tight.pitch = tight.width;
        check(!tight.empty(), "but pitch == width is legal: an unpadded surface");
    }

    void the_stamp_is_the_same_key_wherever_the_pixels_are() {
        FrameCounter counter("cam07");
        const Frame device_frame = counter.stamp(a_surface());
        HostFrame host;
        std::vector<uint8_t> pixels{9};
        host.pixels = pixels.data();
        host.height = host.width = 1;
        const Frame host_frame = counter.stamp(host);

        check(device_frame.tag.camera_id == "cam07" && device_frame.tag.frame_id == 0,
              "a device frame is tagged like any other");
        check(device_frame.tag.captured_ns > 0 && device_frame.tag.captured_unix_ns > 0,
              "and both clocks are read at decode, not later");
        check(host_frame.tag.frame_id == 1,
              "ONE counter for both, because the (camera, frame) key does not care where the "
              "pixels are");
        check(device_frame.on_device() && device_frame.image.empty(),
              "a device frame carries no host image");
        check(!host_frame.on_device() && !host_frame.image.empty(), "and the reverse");
    }

    void the_owner_is_what_unmaps_the_surface() {
        // NVDEC hands out a slot from a small pool and reuses it the moment it is released, so
        // dropping the keepalive early does not free the pixels -- it lets the next frame
        // overwrite them, which surfaces as an unrelated frame's contents several frames on.
        int released = 0;
        {
            std::shared_ptr<const void> owner(&released, [](const void* p) {
                ++*const_cast<int*>(static_cast<const int*>(p));
            });
            const Frame frame = FrameCounter("cam0").stamp(a_surface(owner));
            check(released == 0, "held while the frame is alive");
            check(frame.device.owner != nullptr, "and the frame is what holds it");
        }

        check(released == 1, "and unmapped exactly once when the last reference goes");
    }

    void a_source_answers_from_ONE_hook_for_its_whole_life() {
        FrameCounter counter("cam0");
        StopSignal stop;
        SwitchingSource source(a_camera("cam0"), counter, stop);
        source.open();

        const std::optional<Frame> first = source.read();
        check(first && first->on_device(), "the first frame came off the device");

        source.next_is_device = false;
        bool refused = false;
        std::string message;
        try {
            (void)source.read();
        } catch (const ConfigError& error) {
            refused = true;
            message = error.what();
        }

        check(refused, "changing hook is refused, not accepted silently");
        check(message.find("do_read after this camera had answered from do_read_device") !=
                  std::string::npos,
              "and the message names BOTH answers: " + message);
        check(message.find("cam0") != std::string::npos, "and the camera");
    }

    void an_incomplete_surface_is_REFUSED_and_not_laundered_into_a_host_frame() {
        // #153 round 3, and it is the sharpest of the three: `empty()` was defensive
        // DOCUMENTATION -- nothing on the read path consulted it. `Frame::on_device()` is
        // defined as `!device.empty()`, so an engaged-but-invalid surface was reclassified as
        // a HOST frame with a null pixel pointer, and `Frame`'s own "exactly one is populated"
        // became ZERO. Downstream that reads as healthy all the way to a 0x0 letterbox.
        FrameCounter counter("cam9");
        StopSignal stop;
        SwitchingSource source(a_camera("cam9"), counter, stop);
        source.forget_device_index = true;  // everything else filled, as a real decoder would
        source.open();

        bool refused = false;
        std::string message;
        try {
            (void)source.read();
        } catch (const ConfigError& error) {
            refused = true;
            message = error.what();
        }

        check(refused, "an incomplete surface is refused, not delivered as a host frame");
        check(message.find("incomplete device surface") != std::string::npos,
              "and says what is missing: " + message);
        check(message.find("cam9") != std::string::npos, "and which camera");
        check(!counter.where_latched(),
              "and it did NOT latch: a refused read must not decide where this camera reads "
              "from, or the rebuilt source is judged against a frame that never counted");
    }

    void the_latch_survives_a_RECONNECT_which_is_its_whole_motivation() {
        // #153 round 1: the latch was on the SOURCE, and a source is a per-connection object --
        // its own header says the reconnect state lives outside it. So the plausible failure
        // was the one it could not catch: the stream hiccups, the actor throws the source away,
        // the new one finds the hardware decoder busy and falls back to software, and the graph
        // moves onto the host path with nothing said. Two successive sources, ONE counter,
        // exactly as `CameraActor` rebuilds through `factory_(config_, counter_, stop_)`.
        FrameCounter counter("cam0");
        StopSignal stop;
        {
            SwitchingSource first(a_camera("cam0"), counter, stop);
            first.open();
            const std::optional<Frame> frame = first.read();
            check(frame && frame->on_device(), "connection 1 decoded into VRAM");
        }
        check(counter.where_latched() && counter.reads_device(),
              "and the CAMERA remembers that, not the source that is now destroyed");

        SwitchingSource second(a_camera("cam0"), counter, stop);
        second.next_is_device = false;  // the reconnect fell back to software decode
        second.open();
        bool refused = false;
        std::string message;
        try {
            (void)second.read();
        } catch (const ConfigError& error) {
            refused = true;
            message = error.what();
        }

        check(refused, "the rebuilt source is refused, which is the case that matters");
        check(message.find("software decode") != std::string::npos,
              "and the message names the plausible cause: " + message);
    }

    void the_frame_id_still_advances_across_the_rebuild() {
        // The latch lives beside the frame id now, so this is worth pinning together: one
        // counter per camera for its whole life is what ADR-002 relies on.
        FrameCounter counter("cam2");
        StopSignal stop;
        {
            SwitchingSource first(a_camera("cam2"), counter, stop);
            first.open();
            (void)first.read();
        }
        SwitchingSource second(a_camera("cam2"), counter, stop);
        second.open();
        const std::optional<Frame> frame = second.read();

        check(frame && frame->tag.frame_id == 1,
              "a reconnect does not restart the frame id at zero");
    }

    void a_host_only_source_never_touches_the_device_hook() {
        // `do_read_device` defaults to nothing, so every source that existed before this keeps
        // working with no edit -- which is the point of defaulting it rather than making it
        // pure.
        FrameCounter counter("cam1");
        StopSignal stop;
        SwitchingSource source(a_camera("cam1"), counter, stop);
        source.next_is_device = false;
        source.open();

        const std::optional<Frame> first = source.read();
        const std::optional<Frame> second = source.read();

        check(first && !first->on_device(), "a host source answers on the host hook");
        check(second && !second->on_device(), "and keeps doing so, with no refusal");
    }

}  // namespace

int main() {
    a_device_image_needs_no_cuda_to_carry();
    an_incomplete_surface_is_empty_rather_than_trusted();
    the_stamp_is_the_same_key_wherever_the_pixels_are();
    the_owner_is_what_unmaps_the_surface();
    an_incomplete_surface_is_REFUSED_and_not_laundered_into_a_host_frame();
    a_source_answers_from_ONE_hook_for_its_whole_life();
    the_latch_survives_a_RECONNECT_which_is_its_whole_motivation();
    the_frame_id_still_advances_across_the_rebuild();
    a_host_only_source_never_touches_the_device_hook();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
