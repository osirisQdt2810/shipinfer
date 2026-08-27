#include "shipinfer/ingest/manager.h"

#include <algorithm>
#include <set>
#include <thread>
#include <utility>

#include "shipinfer/core/types.h"

namespace shipinfer {

    IngestManager::IngestManager(std::vector<IngestConfig> cameras, FrameSink& sink,
                                 SourceFactory factory)
        : cameras_(std::move(cameras)), sink_(sink), factory_(std::move(factory)) {}

    IngestManager::~IngestManager() {
        stop();
    }

    std::vector<IngestConfig> IngestManager::configured_cameras() const {
        std::set<std::string> seen;
        std::vector<IngestConfig> enabled;
        for (const IngestConfig& camera : cameras_) {
            camera.validate();
            if (!seen.insert(camera.camera_id).second) {
                throw ConfigError("camera '" + camera.camera_id +
                                  "' is declared twice in the fleet");
            }
            if (camera.enabled) enabled.push_back(camera);
        }
        return enabled;
    }

    void IngestManager::start() {
        // Resolved and validated first, so a bad fleet starts no threads at all rather than
        // leaving half of them running behind the exception.
        const std::vector<IngestConfig> cameras = configured_cameras();
        {
            // Checked and set in one critical section: two callers racing here would otherwise
            // both pass the check and the second would then be refused, one camera at a time,
            // by `add_camera`.
            std::lock_guard<std::mutex> lock(mutex_);
            if (started_) return;
            started_ = true;
        }
        for (const IngestConfig& camera : cameras) add_camera(camera);
    }

    void IngestManager::stop(std::chrono::milliseconds timeout) {
        std::vector<std::unique_ptr<CameraActor>> actors;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& [id, actor] : actors_) actors.push_back(std::move(actor));
            actors_.clear();
        }
        // Pass one: signal everybody. Pass two: wait for everybody. See the header for the
        // shutdown this ordering fixed.
        for (const auto& actor : actors) actor->request_stop();
        for (auto& actor : actors) {
            if (!actor->stop(timeout)) {
                std::lock_guard<std::mutex> lock(mutex_);
                abandoned_.push_back(std::move(actor));
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        started_ = false;
    }

    CameraActor& IngestManager::add_camera(const IngestConfig& config) {
        config.validate();
        CameraActor* actor = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (actors_.count(config.camera_id) != 0) {
                throw ConfigError("camera '" + config.camera_id +
                                  "' is already running; remove it before adding it again");
            }
            auto owned = std::make_unique<CameraActor>(config, sink_, factory_);
            actor = owned.get();
            actors_[config.camera_id] = std::move(owned);
        }
        // Started outside the lock: `start()` spawns a thread, and holding the fleet's lock
        // across a thread launch makes every health read wait on the scheduler.
        actor->start();
        return *actor;
    }

    void IngestManager::remove_camera(const std::string& camera_id,
                                      std::chrono::milliseconds timeout) {
        std::unique_ptr<CameraActor> actor;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto found = actors_.find(camera_id);
            if (found == actors_.end()) {
                std::string running;
                for (const auto& [id, _] : actors_) {
                    running += (running.empty() ? "" : ", ") + id;
                }
                throw ConfigError("camera '" + camera_id + "' is not running; running: " +
                                  (running.empty() ? "(none)" : running));
            }
            actor = std::move(found->second);
            actors_.erase(found);
        }
        if (!actor->stop(timeout)) {
            std::lock_guard<std::mutex> lock(mutex_);
            abandoned_.push_back(std::move(actor));
        }
    }

    CameraActor& IngestManager::actor(const std::string& camera_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = actors_.find(camera_id);
        if (found == actors_.end()) {
            std::string running;
            for (const auto& [id, _] : actors_) running += (running.empty() ? "" : ", ") + id;
            throw ConfigError("camera '" + camera_id + "' is not running; running: " +
                              (running.empty() ? "(none)" : running));
        }
        return *found->second;
    }

    std::vector<std::string> IngestManager::camera_ids() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> ids;
        for (const auto& [id, _] : actors_) ids.push_back(id);
        return ids;
    }

    size_t IngestManager::size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return actors_.size();
    }

    bool IngestManager::contains(const std::string& camera_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return actors_.count(camera_id) != 0;
    }

    std::vector<CameraHealth> IngestManager::snapshot() const {
        // The actors are collected under the lock and read outside it: `CameraActor::health`
        // takes the actor's own lock, and holding the fleet's lock across fifty of those would
        // serialise a health endpoint against every camera in the fleet.
        std::vector<CameraActor*> actors;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& [id, actor] : actors_) actors.push_back(actor.get());
        }
        std::vector<CameraHealth> out;
        out.reserve(actors.size());
        for (CameraActor* actor : actors) out.push_back(actor->health());
        return out;
    }

    std::map<std::string, CameraHealth> IngestManager::health() const {
        std::map<std::string, CameraHealth> out;
        for (CameraHealth& one : snapshot()) {
            const std::string id = one.camera_id;
            out.emplace(id, std::move(one));
        }
        return out;
    }

    IngestSummary IngestManager::summary() const {
        const std::vector<CameraHealth> snap = snapshot();
        IngestSummary summary;
        summary.cameras = snap.size();
        for (const CameraHealth& one : snap) {
            if (one.state == CameraState::Streaming) ++summary.streaming;
            if (one.state == CameraState::Unhealthy) ++summary.unhealthy;
            summary.total_fps += one.fps;
            summary.frames_read += one.frames_read;
            summary.frames_published += one.frames_published;
            summary.frames_dropped += one.frames_dropped;
        }
        return summary;
    }

    void IngestManager::wait_ready(std::chrono::milliseconds timeout,
                                   std::chrono::milliseconds poll) const {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (true) {
            std::vector<std::string> pending;
            for (const CameraHealth& one : snapshot()) {
                if (one.frames_read == 0) pending.push_back(one.camera_id);
            }
            if (pending.empty()) return;
            if (std::chrono::steady_clock::now() >= deadline) {
                std::sort(pending.begin(), pending.end());
                throw CameraUnavailableError(std::move(pending),
                                             std::chrono::duration<double>(timeout).count());
            }
            std::this_thread::sleep_for(poll);
        }
    }

}  // namespace shipinfer
