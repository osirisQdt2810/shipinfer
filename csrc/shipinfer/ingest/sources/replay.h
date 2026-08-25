// One stateful source per camera, paced to a target fps.
//
// ADR-011 in C++: a camera owns its own thread and its own state — its frame counter, its
// pacing debt, its health. It pushes into a sink and has no queue of its own, because a queue
// per camera plus a queue for the fleet is two places for the same frame to wait.
//
// PACING ABSORBS LATENESS RATHER THAN CATCHING UP
// -----------------------------------------------
// If a camera falls behind, it does *not* burst to make up the deficit. A burst turns a
// transient stall into a spike that the fleet's queue then has to absorb, and at 50 cameras
// those spikes align. So a late frame is simply late, and the shortfall is *counted* — which
// is what makes `frames_read` an honest measure of offered load rather than a restatement of
// the configured target.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/core/types.h"

namespace shipinfer {

    // A decoded frame on the host: uint8 HWC BGR. The replay source decodes once per file and
    // then serves from memory, because decoding the same JPEG a thousand times measures
    // libjpeg.
    struct HostFrame {
        const uint8_t* pixels = nullptr;
        int height = 0;
        int width = 0;
    };

    class ReplaySource {
      public:
        // Loads every .jpg/.png under `folder`, decoded once. Throws SourceError if empty,
        // because a benchmark that silently offers zero frames is the failure this project
        // keeps finding.
        explicit ReplaySource(const std::string& folder, int limit = 0);

        ~ReplaySource();

        size_t size() const { return frames_.size(); }
        HostFrame at(size_t index) const;
        // Whether every image is page-locked. False means the run is still correct and its
        // host->device copies take the slow path — worth printing, not worth failing over.
        // True when every image's pages are registered. One refused registration used to flip a
        // single flag and the destructor then unregistered nothing, leaving the pages that had
        // registered locked; each image records its own outcome now.
        bool pinned() const;

      private:
        struct Image {
            std::vector<uint8_t> pixels;
            int height = 0;
            int width = 0;
        };
        std::vector<Image> frames_;
        std::vector<char> registered_;  // per image, 1 when gpuHostRegister succeeded
    };

    class CameraActor {
      public:
        // `publish` returns false when the sink refused the frame; the camera counts that as a
        // drop against *itself*, which is the attribution ADR-005 is about.
        using Publish = std::function<bool(const FrameTag&, HostFrame)>;

        CameraActor(std::string camera_id, std::shared_ptr<ReplaySource> library, double fps,
                    Publish publish);
        ~CameraActor();

        void start();
        void stop();

        uint64_t read() const { return read_.load(); }
        uint64_t dropped() const { return dropped_.load(); }
        const std::string& id() const { return id_; }

      private:
        void run();

        std::string id_;
        std::shared_ptr<ReplaySource> library_;
        double fps_;
        Publish publish_;
        std::thread thread_;
        std::atomic<bool> stopping_{false};
        std::atomic<uint64_t> read_{0};
        std::atomic<uint64_t> dropped_{0};
    };

}  // namespace shipinfer
