#include "shipinfer/ingest/sources/replay.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "shipinfer/core/platform.h"
#include "shipinfer/core/types.h"
#include "shipinfer/ingest/registry.h"

namespace shipinfer {
    namespace {

        // Used when the camera config says nothing. 25 rather than 20 so an accidental default
        // is visible in a report instead of looking like the real fleet rate.
        constexpr double kFallbackFps = 25.0;

        // The decoded libraries currently in use, keyed by folder and limit. Weak, so the last
        // source to let go is what frees ~62 MB of decoded frames and unregisters their pages;
        // a strong cache here would hold every folder any camera ever replayed for the life of
        // the process.
        std::mutex& library_mutex() {
            static std::mutex mutex;
            return mutex;
        }
        std::map<std::string, std::weak_ptr<const ReplayLibrary>>& library_cache() {
            static std::map<std::string, std::weak_ptr<const ReplayLibrary>> cache;
            return cache;
        }

    }  // namespace

    // -- the library ------------------------------------------------------------------------

    ReplayLibrary::ReplayLibrary(const std::string& folder, int limit) {
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

    ReplayLibrary::~ReplayLibrary() {
        // Only what registered is unregistered — per image, so one refusal does not leak the
        // rest.
        for (size_t i = 0; i < frames_.size(); ++i) {
            if (registered_[i]) gpuHostUnregister(frames_[i].pixels.data());
        }
    }

    std::shared_ptr<const ReplayLibrary> ReplayLibrary::acquire(const std::string& folder,
                                                                int limit) {
        const std::string key = folder + "\x1f" + std::to_string(limit);
        std::lock_guard<std::mutex> lock(library_mutex());
        auto& cache = library_cache();
        auto found = cache.find(key);
        if (found != cache.end()) {
            if (std::shared_ptr<const ReplayLibrary> live = found->second.lock()) return live;
            cache.erase(found);
        }
        // `new` rather than `make_shared` because the constructor is private and this is the
        // only thing allowed to call it: a second decode of the same folder is the cost this
        // whole class exists to avoid.
        std::shared_ptr<const ReplayLibrary> library(new ReplayLibrary(folder, limit));
        cache[key] = library;
        return library;
    }

    bool ReplayLibrary::pinned() const {
        for (char r : registered_) {
            if (!r) return false;
        }
        return !frames_.empty();
    }

    HostFrame ReplayLibrary::at(size_t index) const {
        const Image& image = frames_[index % frames_.size()];
        HostFrame frame;
        frame.pixels = image.pixels.data();
        frame.height = image.height;
        frame.width = image.width;
        return frame;
    }

    // -- the source -------------------------------------------------------------------------

    void ReplaySource::do_open() {
        int limit = 0;
        auto option = config().options.find("limit");
        if (option != config().options.end()) {
            try {
                limit = std::stoi(option->second);
            } catch (const std::exception&) {
                throw ConfigError("camera '" + camera_id() + "': replay option limit must be " +
                                  "an integer, got '" + option->second + "'");
            }
        }
        try {
            library_ = ReplayLibrary::acquire(config().uri, limit);
        } catch (const SourceError& error) {
            // A path an operator can fix is **retryable**, so it is a SourceOpenError.
            // SourceUnavailableError would mean a missing decode runtime, which the actor
            // treats as fatal and stops retrying — the wrong answer for a typo'd folder that
            // somebody is about to create.
            throw SourceOpenError(camera_id(), config().uri, error.what());
        }
        const HostFrame probe = library_->at(0);
        // The decoded size, not `config.width/height`: this source does not resize, so
        // reporting a size it does not deliver would send the letterbox scaling twice.
        set_format(probe.height, probe.width, config().fps > 0.0 ? config().fps : kFallbackFps);
        index_ = 0;
        exhausted_ = false;
        pacer_ = DeadlinePacer(fps());
        pacer_.reset();
    }

    std::optional<HostFrame> ReplaySource::do_read() {
        if (exhausted_) return std::nullopt;
        const double budget = config().read_timeout_s();
        const bool interrupted = pacer_.wait([this, budget](double due_s) {
            // Capped at one read timeout, because the actor's contract is that a read answers
            // within one — a source that blocked for a 40 s frame period would make a stop
            // request take 40 s to land. The documented surprise: below `1 / read_timeout_s`
            // fps a replay source reports empty reads and the actor eventually reconnects it.
            // That is harmless (the library is cached, so the reopen is free) and visible in
            // `empty_reads` rather than silent. The pacer does not advance on an interrupted
            // wait, so the frame schedule survives it.
            if (due_s > budget) {
                (void)stop().wait_for(budget);
                return true;
            }
            return stop().wait_for(due_s);
        });
        if (interrupted) return std::nullopt;

        if (index_ >= library_->size()) {
            if (!config().loop) {
                exhausted_ = true;
                return std::nullopt;
            }
            index_ = 0;
        }
        HostFrame image = library_->at(index_);
        ++index_;
        // The keepalive: a reconnect that replaces this source must not free the pages a worker
        // is still DMAing out of, and this handle is what makes the library outlive the source.
        image.owner = library_;
        return image;
    }

    void ReplaySource::do_close() {
        library_.reset();
        index_ = 0;
    }

    namespace {

        const SourceRegistrar kRegistrar(
            "replay", {"file", "video"},
            "a directory of decoded frames, paced at the camera's fps — no camera, no network",
            [](const IngestConfig& config, FrameCounter& counter,
               StopSignal& stop) -> std::unique_ptr<FrameSource> {
                return std::make_unique<ReplaySource>(config, counter, stop);
            });

    }  // namespace

}  // namespace shipinfer
