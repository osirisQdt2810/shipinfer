#include "shipinfer/ingest/manager.h"

#include <algorithm>
#include <set>
#include <thread>
#include <utility>

#include "shipinfer/core/types.h"

namespace shipinfer {

    namespace {

        // The re-check's stop grace (#35): a freshly started actor that got its stop signal
        // before its first wait is gone in microseconds; one that is not is inside a blocked
        // `do_open` (whose budget, 10 s by default, outlives any stop grace) and will have
        // to be detached. Long enough to tell those apart, short enough not to stall the
        // error path of an API call for the full shutdown grace.
        constexpr std::chrono::milliseconds kRecheckStopGrace{250};

    }  // namespace

    IngestManager::IngestManager(std::vector<IngestConfig> cameras, FrameSink& sink,
                                 SourceFactory factory)
        : cameras_(std::move(cameras)), sink_(sink), factory_(std::move(factory)) {}

    IngestManager::~IngestManager() {
        stop();
        // Deliberate leak, and the point of `abandoned_` (see the header): each of these holds
        // a detached thread that is still standing on the actor's members. Letting `~vector`
        // free them here would be a use-after-free at the exact moment the containment is
        // needed — so the references are parked on the heap instead, keeping the refcount
        // pinned for the life of the process.
        for (auto& actor : abandoned_) {
            new std::shared_ptr<CameraActor>(std::move(actor));
        }
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

    size_t IngestManager::stop(std::chrono::milliseconds timeout) {
        std::vector<std::shared_ptr<CameraActor>> actors;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& [id, actor] : actors_) actors.push_back(std::move(actor));
            actors_.clear();
        }
        // Pass one: signal everybody. Pass two: wait for everybody. See the header for the
        // shutdown this ordering fixed.
        //
        // One deadline for the whole fleet, not one per actor: the timeout exists for the
        // camera that genuinely hangs, and charging it per actor would turn one stuck decoder
        // into fifty consecutive waits — the 250 s shutdown the header promises not to have.
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        for (const auto& actor : actors) actor->request_stop();
        size_t abandoned = 0;
        for (auto& actor : actors) {
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now());
            if (!actor->stop(std::max(remaining, std::chrono::milliseconds(0)))) {
                ++abandoned;
                std::lock_guard<std::mutex> lock(mutex_);
                abandoned_.push_back(std::move(actor));
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        started_ = false;
        return abandoned;
    }

    std::shared_ptr<CameraActor> IngestManager::add_camera(const IngestConfig& config) {
        config.validate();
        std::shared_ptr<CameraActor> actor;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (actors_.count(config.camera_id) != 0) {
                throw ConfigError("camera '" + config.camera_id +
                                  "' is already running; remove it before adding it again");
            }
            actor = std::make_shared<CameraActor>(config, sink_, factory_);
            actors_[config.camera_id] = actor;
        }
        // Started outside the lock: `start()` spawns a thread, and holding the fleet's lock
        // across a thread launch makes every health read wait on the scheduler. The local
        // `shared_ptr` is what makes that safe — a concurrent `stop()` can strip the map in
        // this window, and without it the actor would be freed under our feet.
        between_publish_and_start();
        actor->start();
        between_start_and_recheck();
        {
            // The re-check. If the fleet forgot this camera while it was starting — a
            // `stop()` or `remove_camera` landed in the window — its stop request was aimed
            // at a thread that did not exist yet (`start()` clears the signal), so honour it
            // here: stop the actor we just started and say what happened, rather than return
            // a camera that keeps running after the manager has forgotten it.
            std::unique_lock<std::mutex> lock(mutex_);
            auto found = actors_.find(config.camera_id);
            if (found == actors_.end() || found->second != actor) {
                lock.unlock();
                if (!actor->stop(kRecheckStopGrace)) {
                    // The abandonment debt (#33 round 3): the detached thread is still
                    // standing on this actor, and the throw below drops our last reference —
                    // park it with the others `~IngestManager` deliberately leaks, exactly
                    // as `stop()` and `remove_camera` do.
                    std::lock_guard<std::mutex> parked(mutex_);
                    abandoned_.push_back(std::move(actor));
                }
                throw ServerStateError("camera '" + config.camera_id +
                                       "' was removed while it was starting; the fleet is "
                                       "stopping or the camera was removed — add it again "
                                       "once the manager is running");
            }
        }
        return actor;
    }

    void IngestManager::remove_camera(const std::string& camera_id,
                                      std::chrono::milliseconds timeout) {
        std::shared_ptr<CameraActor> actor;
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

    std::shared_ptr<CameraActor> IngestManager::actor(const std::string& camera_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = actors_.find(camera_id);
        if (found == actors_.end()) {
            std::string running;
            for (const auto& [id, _] : actors_) running += (running.empty() ? "" : ", ") + id;
            throw ConfigError("camera '" + camera_id + "' is not running; running: " +
                              (running.empty() ? "(none)" : running));
        }
        return found->second;
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
        // serialise a health endpoint against every camera in the fleet. The copies are
        // `shared_ptr`, not raw pointers, so a `remove_camera` that lands mid-read cannot free
        // an actor this loop is still asking about.
        std::vector<std::shared_ptr<CameraActor>> actors;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& [id, actor] : actors_) actors.push_back(actor);
        }
        std::vector<CameraHealth> out;
        out.reserve(actors.size());
        for (const auto& actor : actors) out.push_back(actor->health());
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
