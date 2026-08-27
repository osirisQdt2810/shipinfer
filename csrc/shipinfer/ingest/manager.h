// Runs the fleet: start, stop, add and remove cameras while the server is live —
// `ingest/manager.py`.
//
// Adding a camera at runtime is not a luxury. The reference service exposed it over REST for a
// reason: a fifty-camera site gains and loses cameras during commissioning, and restarting the
// whole perception tier to onboard one of them means restarting the trackers too, which loses
// every tracklet on every other camera. So the manager owns actor lifecycle, and an actor is
// cheap — one thread, one decoder, no GPU state.
//
// There is exactly one actor per camera id and one thread per actor, and neither is ever
// recycled. A stopped camera that comes back gets a *new* actor, because reusing one would mean
// deciding what its frame counter should now say, and the only safe answer to that is "keep
// counting" — which a new actor with a preserved `first_frame_id` does explicitly rather than
// by accident.
#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "shipinfer/ingest/camera/actor.h"
#include "shipinfer/ingest/camera/health.h"
#include "shipinfer/ingest/config.h"
#include "shipinfer/ingest/registry.h"
#include "shipinfer/ingest/sink.h"

namespace shipinfer {

    class IngestManager {
      public:
        // `sink` is where every camera publishes — one sink for the fleet, because the fairness
        // that matters is *between* cameras and can only be arbitrated somewhere they meet. See
        // `ingest/sink.h` for why that arbitration is the consumer's job and not this
        // package's (ADR-005).
        //
        // `factory` overrides source construction for every actor: the seam a test uses to run
        // the whole manager against a fake camera.
        IngestManager(std::vector<IngestConfig> cameras, FrameSink& sink,
                      SourceFactory factory = {});
        virtual ~IngestManager();

        IngestManager(const IngestManager&) = delete;
        IngestManager& operator=(const IngestManager&) = delete;

        // The cameras `start()` will run, validated and duplicate-checked **before a single
        // thread exists**, so a mistyped configuration is a start-up failure rather than fifty
        // actors failing one at a time. Disabled cameras are filtered out.
        //
        // Throws ConfigError naming the duplicate, or whatever `IngestConfig::validate` throws.
        std::vector<IngestConfig> configured_cameras() const;

        // Start an actor for every enabled camera. Idempotent.
        void start();

        // Stop every actor. Idempotent, and safe before `start()`.
        //
        // SIGNAL, THEN JOIN — two passes, and the order is the whole method. Stop requests go
        // to *all* actors first and only then are they joined, so shutting down fifty cameras
        // costs one read timeout rather than fifty. The first pass is what `request_stop`
        // exists for: a one-pass `stop(0)` joined for zero seconds and `join(0)` returns
        // immediately with the thread still alive, so every clean shutdown logged "did not
        // stop; abandoning the thread" once per camera and marked each one STOPPED while it was
        // still reading and publishing. Fifty false alarms per shutdown is how a real abandoned
        // thread stops being noticed.
        //
        // The deadline is the fleet's, so later actors may be handed a remaining budget of
        // zero — that is not mistreatment: the signal pass already reached them at t0, so
        // one that is still running at the deadline would have missed a per-actor budget
        // too, and one that has finished joins instantly on zero.
        //
        // Returns how many actors had to be *abandoned* — detached past the deadline, still
        // holding their sink and source references. 0 is the clean shutdown. A non-zero return
        // is the caller's cue that references it lent to the fleet (the sink above all) must
        // now outlive this manager; `bench` exits without unwinding for exactly that reason.
        size_t stop(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

        // Start one camera. Throws ConfigError if a camera with this id is already running:
        // silently replacing it would leave two threads pulling one stream and two frame
        // counters producing duplicate tags.
        //
        // Safe against a concurrent `stop()`/`remove_camera`: the actor is started outside the
        // lock (see `actors_`), then the map is re-checked — if the fleet forgot the camera in
        // the window, the freshly started actor is stopped here and ServerStateError says so,
        // rather than leaving a camera running that no shutdown will ever reach.
        //
        // Returns shared ownership, not a reference: `remove_camera` on another thread can
        // erase the map's copy at any moment, and a reference into that map would dangle. The
        // caller's `shared_ptr` keeps the actor alive for as long as the caller looks at it —
        // the same rule `snapshot()` applies internally.
        std::shared_ptr<CameraActor> add_camera(const IngestConfig& config);

        // Stop and forget one camera. Throws ConfigError naming what *is* running, because a
        // typo in an operator's API call deserves an answer rather than a silent no-op.
        void remove_camera(const std::string& camera_id,
                           std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

        // Shared ownership for the same reason as `add_camera` — see there.
        std::shared_ptr<CameraActor> actor(const std::string& camera_id) const;
        std::vector<std::string> camera_ids() const;
        size_t size() const;
        bool contains(const std::string& camera_id) const;

        std::map<std::string, CameraHealth> health() const;
        IngestSummary summary() const;

        // Block until every camera has delivered a frame. What turns a mistyped camera database
        // into a failed deploy rather than a server that looks healthy and produces no
        // detections.
        //
        // Throws CameraUnavailableError naming every camera that produced nothing in time.
        void wait_ready(std::chrono::milliseconds timeout = std::chrono::milliseconds(30000),
                        std::chrono::milliseconds poll = std::chrono::milliseconds(50)) const;

      protected:
        // The two halves of the ~100 ns window in `add_camera`, made places a test can
        // stand — the same job the injectable wait does for the actor's own sleeps. A
        // concurrent `stop()` in the first half strips the map and signals a thread that
        // does not exist yet; the second half is where that thread must reach its (blocked)
        // `do_open` before the re-check's own stop request lands, which is what forces the
        // re-check to detach and pay the abandonment debt. That interleaving is otherwise
        // unreachable on purpose-built hardware (#33 round 3 hammered 400 ASan rounds
        // without landing in it once).
        virtual void between_publish_and_start() {}
        virtual void between_start_and_recheck() {}

      private:
        std::vector<CameraHealth> snapshot() const;

        std::vector<IngestConfig> cameras_;
        FrameSink& sink_;
        SourceFactory factory_;
        mutable std::mutex mutex_;
        // `shared_ptr`, not `unique_ptr`, and the difference is lifetime under concurrency:
        // `add_camera` starts the actor outside the lock (a thread launch under the fleet's
        // lock would serialise every health read against the scheduler), and `snapshot` reads
        // `health()` outside it for the same reason — so a concurrent `stop`/`remove_camera`
        // can strip the map while another thread still holds the actor. The map drops its
        // reference; the thread's own reference keeps the actor alive until it is done.
        std::map<std::string, std::shared_ptr<CameraActor>> actors_;
        // Actors whose thread had to be detached. They are kept — not destroyed — because the
        // detached thread is still using `this`, and freeing it under a live thread is a
        // use-after-free on the shutdown path. It is a deliberate, bounded leak: an abandoned
        // thread is already a bug being contained, and the containment must not be worse than
        // the bug. The destructor makes the leak literal — it releases these rather than
        // letting `~vector` free memory a live thread is standing on.
        std::vector<std::shared_ptr<CameraActor>> abandoned_;
        bool started_ = false;
    };

}  // namespace shipinfer
