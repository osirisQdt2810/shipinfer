#include "shipinfer/ingest/sources/replay.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "shipinfer/core/platform.h"

namespace shipinfer {
    namespace {

        int64_t unix_ns() {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                .count();
        }

    }  // namespace

    ReplaySource::ReplaySource(const std::string& folder, int limit) {
        namespace fs = std::filesystem;
        if (!fs::is_directory(folder)) {
            throw SourceError("frame folder is not a directory: " + folder);
        }
        std::vector<std::string> paths;
        for (const auto& entry : fs::directory_iterator(folder)) {
            if (!entry.is_regular_file()) continue;
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                paths.push_back(entry.path().string());
            }
        }
        // Sorted, so two runs replay the same frames in the same order and a difference between
        // them is a difference in the system rather than in the input.
        std::sort(paths.begin(), paths.end());
        if (limit > 0 && paths.size() > static_cast<size_t>(limit)) {
            paths.resize(static_cast<size_t>(limit));
        }
        for (const auto& path : paths) {
            cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
            if (image.empty()) {
                // Counted and named: 900 undecodable files out of 1000 must not silently become
                // a 100-frame library that replays ten times faster than the real footage.
                ++undecodable_;
                std::cerr << "replay: cannot decode " << path << "\n";
                continue;
            }
            if (!image.isContinuous()) image = image.clone();
            Image decoded;
            decoded.height = image.rows;
            decoded.width = image.cols;
            decoded.pixels.assign(
                image.data, image.data + static_cast<size_t>(image.total()) * image.elemSize());
            frames_.push_back(std::move(decoded));
        }
        // Registered as pinned. Every frame is copied host->device once per replay, and a
        // *pageable* source forces the driver to stage the copy through its own bounded buffer
        // and serialise it against every other copy on the context: at 350 img/s x 6 MB that is
        // 2 GB/s of copies taking the slow path. `gpuHostRegister` on the decoded library is
        // one call at start-up and makes those copies DMA straight out of these pages.
        //
        // Failure is not fatal: registration can be refused (a locked-memory rlimit, a kernel
        // that will not pin that much), and the correct response is a slower run rather than no
        // run.
        registered_.assign(frames_.size(), 0);
        for (size_t i = 0; i < frames_.size(); ++i) {
            auto& image = frames_[i];
            if (gpuHostRegister(image.pixels.data(), image.pixels.size(),
                                gpuHostRegisterDefault) == gpuSuccess) {
                registered_[i] = 1;
            } else {
                gpuGetLastError();  // cleared, so the next real error is not misattributed
            }
        }

        if (frames_.empty()) {
            throw SourceError(
                "no decodable images under " + folder +
                " — a run that offers zero frames is not a slower measurement, it is "
                "a different experiment");
        }
    }

    ReplaySource::~ReplaySource() {
        // Only what registered is unregistered — per image, so one refusal does not leak the
        // rest.
        for (size_t i = 0; i < frames_.size(); ++i) {
            if (registered_[i]) gpuHostUnregister(frames_[i].pixels.data());
        }
    }

    bool ReplaySource::pinned() const {
        for (char r : registered_) {
            if (!r) return false;
        }
        return !frames_.empty();
    }

    HostFrame ReplaySource::at(size_t index) const {
        const Image& image = frames_[index % frames_.size()];
        HostFrame frame;
        frame.pixels = image.pixels.data();
        frame.height = image.height;
        frame.width = image.width;
        return frame;
    }

    CameraActor::CameraActor(std::string camera_id, std::shared_ptr<ReplaySource> library,
                             double fps, Publish publish)
        : id_(std::move(camera_id)),
          library_(std::move(library)),
          fps_(fps),
          publish_(std::move(publish)) {}

    CameraActor::~CameraActor() {
        stop();
    }

    void CameraActor::start() {
        if (thread_.joinable()) return;
        thread_ = std::thread([this] { run(); });
    }

    void CameraActor::stop() {
        stopping_.store(true);
        if (thread_.joinable()) thread_.join();
    }

    void CameraActor::run() {
        using clock = std::chrono::steady_clock;
        const auto period = std::chrono::duration<double>(1.0 / std::max(1e-6, fps_));
        auto next = clock::now();
        int64_t frame_id = 0;

        while (!stopping_.load()) {
            FrameTag tag;
            tag.camera_id = id_;
            tag.frame_id = frame_id++;
            tag.captured_ns = unix_ns();

            const HostFrame frame = library_->at(static_cast<size_t>(tag.frame_id));
            read_.fetch_add(1);
            // `publish_` allocates (a FrameState, a lane entry); a std::bad_alloc escaping this
            // thread would call std::terminate and lose the run with no counters — the same
            // failure the worker threads were guarded against. A frame that could not be
            // published is a dropped frame, counted, and this camera keeps going.
            try {
                if (!publish_(tag, frame)) dropped_.fetch_add(1);
            } catch (const std::exception& error) {
                dropped_.fetch_add(1);
                static std::atomic<int> shouted{0};
                if (shouted.fetch_add(1) < 5) {
                    std::cerr << "camera " << id_ << " could not publish frame " << tag.frame_id
                              << ": " << error.what() << "\n";
                }
            }

            next += std::chrono::duration_cast<clock::duration>(period);
            const auto now = clock::now();
            if (next > now) {
                std::this_thread::sleep_for(next - now);
            } else {
                // Behind. Absorb rather than catch up — see the header. Resetting `next` to now
                // is what makes the deficit show up in `read_` as a lower offered rate instead
                // of becoming a burst the fleet's queue has to eat.
                next = now;
            }
        }
    }

}  // namespace shipinfer
