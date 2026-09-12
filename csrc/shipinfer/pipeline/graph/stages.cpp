#include "shipinfer/pipeline/graph/stages.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>

#include "shipinfer/core/buffers.h"
#include "shipinfer/pipeline/graph/pixels.h"
#include "shipinfer/runtime/ops.h"

namespace shipinfer {

    // -- WorkerScratch
    // -------------------------------------------------------------------------

    WorkerScratch::WorkerScratch(Device device) : device_(device) {
        GPU_CHECK(gpuStreamCreate(&stream_));
    }

    WorkerScratch::~WorkerScratch() {
        if (stream_ != nullptr) gpuStreamDestroy(stream_);
    }

    std::shared_ptr<DeviceBuffer> WorkerScratch::acquire(const std::string& name,
                                                         size_t bytes) {
        auto& pool = pools_[name];
        for (auto& buffer : pool) {
            if (buffer.use_count() != 1) continue;  // a request still points into it
            if (buffer->bytes() < bytes) buffer = std::make_shared<DeviceBuffer>(bytes);
            return buffer;
        }
        if (pool.size() >= kMaxHeldPerName) {
            throw ServerStateError("worker scratch '" + name +
                                   "': " + std::to_string(pool.size()) +
                                   " payloads are still held by requests that have not "
                                   "completed (timed out or still queued); refusing to "
                                   "allocate more rather than grow without bound");
        }
        pool.push_back(std::make_shared<DeviceBuffer>(bytes));
        return pool.back();
    }

    size_t WorkerScratch::held(const std::string& name) const {
        const auto it = pools_.find(name);
        if (it == pools_.end()) return 0;
        size_t held = 0;
        for (const auto& buffer : it->second) held += buffer.use_count() != 1 ? 1 : 0;
        return held;
    }

    const float* WorkerScratch::upload_boxes(const std::vector<float>& boxes) {
        const size_t bytes = boxes.size() * sizeof(float);
        if (boxes_host_.bytes() < bytes) boxes_host_ = PinnedBuffer(bytes);
        if (boxes_device_.bytes() < bytes) boxes_device_ = DeviceBuffer(bytes);
        std::memcpy(boxes_host_.get(), boxes.data(), bytes);
        GPU_CHECK(gpuMemcpyAsync(boxes_device_.get(), boxes_host_.get(), bytes,
                                 gpuMemcpyHostToDevice, stream_));
        return boxes_device_.as<float>();
    }

    void WorkerScratch::synchronise() {
        GPU_CHECK(gpuStreamSynchronize(stream_));
    }

    // -- ModelStage
    // ----------------------------------------------------------------------------

    ModelStage::ModelStage(std::string name, Model& model, std::chrono::milliseconds timeout,
                           std::vector<std::string> consumes, std::vector<std::string> needs,
                           std::vector<std::string> produces)
        : Stage(std::move(name), std::move(consumes), std::move(needs), std::move(produces)),
          model_(model),
          timeout_(timeout) {
        if (timeout_.count() <= 0)
            throw ConfigError("stage " + this->name() + ": timeout must be > 0");
    }

    InferenceResponse ModelStage::infer(const FrameState& state, const float* data, size_t rows,
                                        size_t row_elems, Device device,
                                        std::shared_ptr<const void> keepalive) {
        // The request carries the frame's tag **unchanged** (ADR-002): batching, spillover to
        // another GPU and out-of-order completion are all fine because reassembly keys on the
        // tag rather than on arrival order.
        InferenceRequest request;
        request.model_name = model_.name();
        request.tag = state.tag();
        request.priority = state.priority();
        request.deadline_ns = state.deadline_ns();
        request.resident_device = device;
        request.data = data;
        request.rows = rows;
        request.row_elems = row_elems;
        request.payload_device = device;
        request.keepalive = std::move(keepalive);
        std::future<InferenceResponse> future = model_.infer(std::move(request));
        if (future.wait_for(timeout_) != std::future_status::ready) {
            // Bounded so a wedged instance costs one frame and one worker for this long rather
            // than forever; the stage fails and the event names it as missing.
            throw RequestTimeoutError(
                "stage " + name() + ": model " + model_.name() + " did not answer within " +
                std::to_string(timeout_.count()) + " ms for " + state.tag().key());
        }
        return future.get();
    }

    // -- DetectStage
    // ---------------------------------------------------------------------------

    DetectStage::DetectStage(std::string name, Model& detector, DetectConfig config,
                             WorkerScratch& scratch, std::chrono::milliseconds timeout)
        : ModelStage(std::move(name), detector, timeout, {FRAME_INPUT}, {FRAME_INPUT},
                     {DETECTIONS}),
          config_(config),
          scratch_(scratch) {}

    size_t DetectStage::do_run(FrameState& state) {
        const size_t row_elems = static_cast<size_t>(3) * config_.size * config_.size;
        std::shared_ptr<DeviceBuffer> owner =
            scratch_.acquire("letterbox", row_elems * sizeof(float));
        float* input = owner->as<float>();
        // Whichever representation this camera's decoder produces -- `graph/pixels.h` is the
        // one place that asks, and the geometry it returns is the same either way.
        const LetterboxMap map =
            letterbox_frame(state, input, config_.size, config_.size, /*swap_rb=*/true,
                            config_.pad_value, scratch_.stream());
        scratch_.synchronise();
        // Stored on the state, not recomputed downstream: the decode must undo exactly the
        // transform that was applied.
        state.set_letterbox(map.scale, map.pad_x, map.pad_y);

        const InferenceResponse response =
            infer(state, input, 1, row_elems, scratch_.device(), owner);
        if (response.row_elems() % kDetectionStride != 0) {
            throw BackendError("stage " + name() + ": the detector's row of " +
                               std::to_string(response.row_elems()) + " floats is not " +
                               std::to_string(kDetectionStride) + " per candidate");
        }
        // Model space back to original pixels. Cropping in letterboxed coordinates is where the
        // off-by-a-pad-bar bugs live.
        std::vector<Detection> detections;
        const size_t candidates = response.row_elems() / kDetectionStride;
        for (size_t d = 0;
             d < candidates && detections.size() < static_cast<size_t>(config_.max_objects);
             ++d) {
            const float* row = response.row(0) + d * kDetectionStride;
            if (row[4] < config_.score_threshold) continue;
            Detection det;
            det.x1 = (row[0] - static_cast<float>(map.pad_x)) / map.scale;
            det.y1 = (row[1] - static_cast<float>(map.pad_y)) / map.scale;
            det.x2 = (row[2] - static_cast<float>(map.pad_x)) / map.scale;
            det.y2 = (row[3] - static_cast<float>(map.pad_y)) / map.scale;
            det.score = row[4];
            det.class_id = static_cast<int>(row[5]);
            detections.push_back(det);
        }
        const size_t count = detections.size();
        state.set_detections(std::move(detections));
        state.set_detected(true);
        return count;
    }

    // -- CropStage
    // -----------------------------------------------------------------------------

    CropStage::CropStage(std::string name, std::vector<CropSpec> crops, int max_objects,
                         WorkerScratch& scratch)
        : Stage(std::move(name), {DETECTIONS, FRAME_INPUT}, {DETECTIONS},
                [&crops] {
                    std::vector<std::string> names;
                    for (const CropSpec& spec : crops) names.push_back(spec.name);
                    return names;
                }()),
          crops_(std::move(crops)),
          max_objects_(max_objects),
          scratch_(scratch) {
        if (crops_.empty())
            throw ConfigError("stage " + this->name() + " must declare at least one crop set");
    }

    size_t CropStage::do_run(FrameState& state) {
        size_t total = 0;
        for (const CropSpec& spec : crops_) {
            std::vector<float> boxes;
            std::vector<int> indices;
            for (const Detection& det : state.detections()) {
                // A negative id is EVERY row, which is what a crop element with no
                // `classes:` means on the Python plane ("Default: every row",
                // `elements/pool.py`). Nothing constructed one before the plan did.
                // `!= kAnyClass` and NOT `>= 0`: `kNoClass` is negative too, so `>= 0`
                // skipped nothing and a DECLARED EMPTY selection cropped every row -- the
                // opposite of what it means, and "at an embedder a doubled GPU bill".
                if (spec.class_id != CropSpec::kAnyClass && det.class_id != spec.class_id) {
                    continue;
                }
                boxes.insert(boxes.end(), {det.x1, det.y1, det.x2, det.y2});
                indices.push_back(det.index);
            }
            DevicePayload payload;
            payload.name = spec.name;
            payload.class_name = spec.class_name;
            payload.device = scratch_.device();
            payload.row_elems = static_cast<size_t>(3) * spec.height * spec.width;
            if (indices.empty()) {
                // Zero rows rather than a missing entry: the name exists, so the graph is
                // valid, and it is empty, so no stage requiring it is planned.
                state.attach_payload(std::move(payload));
                continue;
            }
            const int count = static_cast<int>(indices.size());
            // doc: long the ceiling is the plan's now, so the bound is stated in its terms
            // Sized for what was detected, rounded up in steps of eight so the pool reuses a
            // buffer across frames of similar crowds — NOT for `max_objects_` every time.
            // The worst case is `max_objects_ * 3 * h * w * 4 B` per buffer per worker, and
            // `max_objects_` comes from the plan now rather than from `DetectConfig`'s old
            // 64: at 64 and 640x640 that is 315 MB, and the production chain's
            // `max_detections 300` makes the same crop set ~1.5 GB — against
            // `WorkerScratch::kMaxHeldPerName == 16`, and the pool may hold a second one
            // under a timeout. Which is why this sizes for the frame and not for the cap.
            const int rows = std::min(max_objects_, ((count + 7) / 8) * 8);
            std::shared_ptr<DeviceBuffer> owner =
                scratch_.acquire(spec.name, static_cast<size_t>(std::max(count, rows)) *
                                                payload.row_elems * sizeof(float));
            float* dst = owner->as<float>();
            const float* boxes_device = scratch_.upload_boxes(boxes);
            crop_frame(state, boxes_device, count, dst, spec.height, spec.width,
                       /*swap_rb=*/true, scratch_.stream());
            scratch_.synchronise();
            payload.data = dst;
            payload.owner = std::move(owner);
            payload.rows = static_cast<size_t>(count);
            payload.object_indices = std::move(indices);
            payload.boxes = std::move(boxes);
            state.attach_payload(std::move(payload));
            total += static_cast<size_t>(count);
        }
        return total;
    }

    // -- ObjectStage
    // ---------------------------------------------------------------------------

    ObjectStage::ObjectStage(std::string name, Model& model, std::string source,
                             std::string output, std::chrono::milliseconds timeout,
                             ObjectCombine combine)
        : ModelStage(std::move(name), model, timeout, {source}, {source}, {output}),
          source_(std::move(source)),
          output_(std::move(output)),
          combine_(std::move(combine)) {}

    TrackStage::TrackStage(std::string name, std::string output, int class_id,
                           std::shared_ptr<tracking::Associator> associator)
        // doc: long two events this stage has to tell apart, and what needing DETECTIONS cost
        // CONSUMES the detections and does NOT need them. `needs` is "present AND NON-EMPTY",
        // so needing DETECTIONS skipped the stage on every frame the detector answered with
        // ZERO boxes -- and an unadvanced tracker does not age, so a ship that left frame
        // stays lost-but-alive with a Kalman prediction where it was and the next object near
        // that box is published with its id. `consumes` gates on `available()` (`detected_`),
        // so the stage still skips the frame the detector never answered for -- `track.py`'s
        // `detections is None` arm -- and now runs on the zero-box frame, which is that
        // file's other arm: "an empty `Detections` still advances the tracker below, because
        // ageing is how a track dies".
        : Stage(std::move(name), {DETECTIONS}, {}, {output}),
          output_(std::move(output)),
          class_id_(class_id),
          associator_(std::move(associator)) {}

    size_t TrackStage::do_run(FrameState& state) {
        // THE SELECTED ROWS ONLY, and the filtered copy keeps `(*selected)[i].index` -- the
        // associator's answer is parallel to what it was GIVEN, so the ids still land on the
        // detector's own indices rather than on the selection's.
        //
        // NO COPY at all for a slot that selects every row, which is the common shape: the
        // associator takes a const reference and the frame's own vector is already the answer.
        // A selected slot reuses `selected_` rather than allocating one per frame.
        //
        // `!= kAnyClass` and not `>= 0`, the same reading `CropStage` makes: `kNoClass` is
        // negative too, so `>= 0` would treat a declared EMPTY selection as every row -- the
        // opposite of what it means.
        const std::vector<Detection>* selected = &state.detections();
        if (class_id_ != CropSpec::kAnyClass) {
            selected_.clear();
            for (const Detection& det : state.detections()) {
                if (det.class_id == class_id_) selected_.push_back(det);
            }
            selected = &selected_;
        }
        ObjectBatch batch;
        batch.name = output_;
        batch.width = 1;
        // CAUGHT HERE, and not left to `Stage::run`. A tracker refuses a frame that does not
        // advance its camera's stream, and a pool of workers reorders frames routinely -- so
        // this is an ordinary outcome. Letting it out would make the stage FAIL, which never
        // delivers the slot to the collector, so the frame the graph promised sits in
        // `pending_` until `sweep()` retires it as a TIMEOUT: published a reassembly window
        // late and counted in the fault channel, for a frame whose only problem is that it
        // has no ids. `track.py` catches the same refusal and returns `_untracked(item)`.
        std::vector<int> ids;
        try {
            ids = associator_->ids(state.tag().camera_id, state.tag().frame_id, *selected);
        } catch (const InferenceError&) {
            // NOT swallowed: counted on the associator, which is shared per slot, so the run
            // can report it. Only this type -- a ConfigError from a misconfigured tracker is
            // a fault and must still fail the stage.
            associator_->note_untracked();
            ids.clear();
        }
        for (size_t row = 0; row < ids.size() && row < selected->size(); ++row) {
            // A row per CONFIRMED id only. `-1` is "matched no track", and an absent row is how
            // `events/records.cpp` leaves `track_id` null for that object.
            if (ids[row] < 0) continue;
            batch.object_indices.push_back((*selected)[row].index);
            // FLOAT, because `ObjectBatch::data` is one: exact for ids below 2^24, and a
            // monotonic per-camera counter reaches that after ~16.7M tracks. The Python plane
            // carries the same ids as `int64`, so the two planes differ THERE and the ledger
            // holds the item; nothing here can widen the carrier alone.
            batch.data.push_back(static_cast<float>(ids[row]));
        }
        const size_t rows = batch.rows();
        // Attached even when EMPTY, like a crop payload: the name exists, so a reader that
        // joins on it finds an answer rather than a missing key.
        state.attach(std::move(batch));
        return rows;
    }

    MtmcStage::MtmcStage(std::string name, std::string output, std::string track_source,
                         std::vector<std::string> embedding_sources,
                         std::shared_ptr<mtmc::InstantBarrier> barrier,
                         std::shared_ptr<mtmc::ClusterTracker> tracker,
                         std::vector<std::string> roster)
        // CONSUMES the track ids and does NOT need them, for the reason `TrackStage` does not
        // need the detections: a camera with nothing to report still has to REPORT, or the
        // instant it belongs to waits for it until the window runs out and every other camera
        // in the group is answered late. `needs` is "present and non-empty".
        : Stage(std::move(name), {track_source}, {}, {output}),
          output_(std::move(output)),
          track_source_(std::move(track_source)),
          embedding_sources_(std::move(embedding_sources)),
          barrier_(std::move(barrier)),
          roster_(roster.begin(), roster.end()),
          tracker_(std::move(tracker)) {}

    namespace {

        //: One detection's embedding, from whichever embedder holds that row. `nullptr` when
        //: no embedder ran for it -- a person row in a chain whose ship embedder answered, or
        //: a row every embedder's crop selection passed over.
        const float* embedding_of(const FrameState& state,
                                  const std::vector<std::string>& sources, int index,
                                  int& width) {
            for (const std::string& source : sources) {
                const ObjectBatch* batch = state.batch(source);
                if (batch == nullptr) continue;
                for (size_t row = 0; row < batch->rows(); ++row) {
                    if (batch->object_indices[row] != index) continue;
                    width = batch->width;
                    return batch->row(row);
                }
            }
            return nullptr;
        }

    }  // namespace

    size_t MtmcStage::do_run(FrameState& state) {
        // NOT THIS GROUP'S CAMERA. Published with a null global id and never submitted: this
        // group's barrier must not wait on it, and its identity space must not hold it. An
        // empty batch under this stage's own name is the same answer a row the gate refused
        // gets, so a reader that joins on the name still finds one.
        if (!roster_.empty() && roster_.count(state.tag().camera_id) == 0) {
            ObjectBatch skipped;
            skipped.name = output_;
            state.attach(std::move(skipped));
            return 0;
        }
        // WHAT THIS CAMERA CAN CONTRIBUTE: a row needs a TRACK id (identity is keyed by
        // (camera, track), so an untracked row has nothing to hold on to) and an EMBEDDING
        // (cross-camera identity is decided on appearance). A row missing either is passed
        // over here and published with a null global id, which is a different fact from
        // "this row does not exist".
        std::vector<mtmc::ClusterObservation> mine;
        const ObjectBatch* tracks = state.batch(track_source_);
        if (tracks != nullptr) {
            for (size_t row = 0; row < tracks->rows(); ++row) {
                const int index = tracks->object_indices[row];
                if (index < 0 || static_cast<size_t>(index) >= state.detections().size()) {
                    continue;
                }
                int width = 0;
                const float* embedding = embedding_of(state, embedding_sources_, index, width);
                if (embedding == nullptr || width <= 0) continue;
                const Detection& detection = state.detections()[static_cast<size_t>(index)];
                mtmc::ClusterObservation observation;
                observation.key = mtmc::TrackKey{state.tag().camera_id,
                                                 static_cast<int64_t>(tracks->row(row)[0])};
                observation.embedding.assign(embedding, embedding + width);
                observation.box[0] = detection.x1;
                observation.box[1] = detection.y1;
                observation.box[2] = detection.x2;
                observation.box[3] = detection.y2;
                observation.box_index = index;
                observation.frame_width = state.width();
                observation.frame_height = state.height();
                mine.push_back(std::move(observation));
            }
        }

        // THE INSTANT, not this camera. `barrier.h` states the constraint: a cross-camera
        // tracker consumes every camera of a group at one synchronised instant and refuses
        // anything less, because one camera at a time turns cross-camera association into
        // within-camera deduplication. The callback below runs on whichever worker closes the
        // bucket, over EVERY camera's entries.
        // MOVED, not copied. Only `key` and `box_index` are read back after `submit`, so the
        // scatter keeps those two and the embeddings travel once -- a copy here was 2048
        // floats per row per frame on the dispatch path (#222's review).
        std::vector<std::pair<mtmc::TrackKey, int>> scatter;
        scatter.reserve(mine.size());
        for (const mtmc::ClusterObservation& observation : mine) {
            scatter.emplace_back(observation.key, observation.box_index);
        }
        const auto payload =
            std::make_shared<std::vector<mtmc::ClusterObservation>>(std::move(mine));
        const mtmc::Association associate =
            [this](const std::vector<mtmc::InstantEntry>& entries) -> mtmc::Results {
            std::vector<mtmc::ClusterObservation> instant;
            // THE ROSTER COMES FROM THE ENTRIES, not from the observations. One entry per
            // camera that reported, so a camera whose frame held no tracks is here with an
            // empty payload -- and the gate needs it: that camera WAS in this instant, and a
            // streak of its tracks has to break rather than be carried across.
            std::vector<std::string> cameras;
            cameras.reserve(entries.size());
            for (const mtmc::InstantEntry& entry : entries) {
                cameras.push_back(entry.camera_id);
                const auto& camera =
                    *std::static_pointer_cast<std::vector<mtmc::ClusterObservation>>(
                        entry.payload);
                instant.insert(instant.end(), camera.begin(), camera.end());
            }
            return std::make_shared<const std::map<mtmc::TrackKey, int64_t>>(
                tracker_->ids(instant, cameras));
        };
        // SECONDS FROM THE CAPTURE (WALL) STAMP, because the other plane keys the same
        // barrier on `item.context.captured_unix_ns` and two planes bucketing one clip into
        // different instants is two different sets of global ids -- the sync rule's whole
        // subject, and #222's review caught the divergence. The steady stamp was the first
        // choice for a real reason (NTP can step the wall clock, including backwards, and a
        // stepped frame lands in the wrong instant rather than merely late) but it is also
        // PER PROCESS, so a fleet's shards could never share an instant with it. The NTP risk
        // is `MTMC-INSTANTS-NEED-A-SHARED-MONOTONIC-CLOCK`.
        //
        // REFUSED AT ZERO, the way the Python element's validator is: a source that never
        // stamps would otherwise put every camera's every frame into ONE instant that closes
        // once and makes everything after it late for the life of the process -- which reads
        // as clock skew and is a wiring fault.
        if (state.tag().captured_unix_ns <= 0) {
            throw ConfigError("stage " + name() + ": frame " +
                              std::to_string(state.tag().frame_id) + " of camera '" +
                              state.tag().camera_id +
                              "' carries no capture stamp, and an instant is keyed on when a "
                              "frame was captured; the source must set captured_unix_ns");
        }
        const double capture_s = static_cast<double>(state.tag().captured_unix_ns) / 1e9;
        // HOW LATE THIS FRAME IS, measured HERE because this is the only place that holds both
        // stamps on one clock: the capture stamp is a wall time and the barrier's own clock is
        // deliberately steady. `late` counts frames that missed their instant; this says by how
        // much, which is what separates a window that is too narrow from a chain too slow.
        const auto arrival_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count();
        const int64_t lag_ns = arrival_ns - state.tag().captured_unix_ns;
        // CLAMPED AT ZERO AND COUNTED. A frame arriving before its own capture stamp is the
        // two clocks disagreeing -- an NTP step on this shard, or a source stamping ahead --
        // and `backward` cannot see it: that compares a camera's stamps against its OWN
        // history, so a stepped server clock leaves it at zero while every lag reads 0.
        barrier_->note_arrival_lag_us(
            static_cast<uint32_t>(std::min<int64_t>(std::max<int64_t>(lag_ns, 0) / 1000,
                                                    std::numeric_limits<uint32_t>::max())),
            lag_ns < 0);
        // CAUGHT AT THE SUBMIT CALL SITE, which is where the other plane catches it
        // (`elements/mtmc.py`) -- and that matters twice over. ONE CAMERA'S FAULT COSTS THE
        // GROUP'S INSTANT AND NOT THE CLOSING FRAME: `barrier.h` says a throwing association
        // releases every waiter with `kDroppedFailed` and rethrows here, so without this the
        // frame that happened to close the bucket would fail its stage and be retired as a
        // reassembly TIMEOUT -- #215's lesson one level up. And catching HERE rather than
        // inside the association keeps the two planes' LEDGERS the same: the barrier records
        // the instant as failed for its waiters rather than as `complete`, which is what
        // #222's review found diverging when the catch was one level down. A `ConfigError` is
        // still fatal: a misconfigured tracker is not data.
        mtmc::InstantOutcome outcome;
        try {
            outcome = barrier_->submit(state.tag().camera_id, capture_s, payload, associate);
        } catch (const InferenceError&) {
            tracker_->note_refused();
            outcome = mtmc::InstantOutcome{mtmc::kDroppedFailed, nullptr, false, 0};
        }

        ObjectBatch batch;
        batch.name = output_;
        batch.width = 1;
        if (outcome.associated && outcome.results) {
            const auto& ids =
                *std::static_pointer_cast<const std::map<mtmc::TrackKey, int64_t>>(
                    outcome.results);
            // KEYED, never positional. The association answers for the whole group in one
            // map, so camera A's three results and camera B's one mean nothing to each other
            // by position -- and each frame reads its OWN entries out by the key it chose.
            // Scattering by list position is the classic reassembly bug one layer up.
            for (const auto& [key, box_index] : scatter) {
                const auto found = ids.find(key);
                // `kUnidentified` IS NOT AN ID. The gate admits nothing for a track that is
                // too small or too new and the seam answers `-1` for it, which the event
                // schema means as a null `global_id` -- publishing it as a row would put -1
                // on the screen as an identity.
                if (found == ids.end() || found->second == mtmc::kUnidentified) continue;
                batch.object_indices.push_back(box_index);
                batch.data.push_back(static_cast<float>(found->second));
            }
        }
        const size_t rows = batch.rows();
        // Attached even when EMPTY, like every other per-object payload: the name exists, so
        // a reader that joins on it finds an answer rather than a missing key -- and an
        // instant this frame missed is a gap the event records as a null id.
        state.attach(std::move(batch));
        return rows;
    }

    size_t ObjectStage::do_run(FrameState& state) {
        const DevicePayload* payload = state.payload(source_);
        if (payload == nullptr)
            throw ConfigError("stage " + name() + ": no payload named " + source_);
        // Chunked to the engine's own batch — the plans are static, and submitting a whole
        // frame's crops as one request is what lost every crop in a 25-person frame against a
        // plan built at 16. One ObjectBatch, grown per chunk, attached once.
        const size_t limit = static_cast<size_t>(std::max(1, model().max_batch()));
        ObjectBatch out;
        out.name = output_;
        for (size_t start = 0; start < payload->rows; start += limit) {
            const size_t count = std::min(limit, payload->rows - start);
            const InferenceResponse response =
                infer(state, payload->data + start * payload->row_elems, count,
                      payload->row_elems, payload->device, payload->owner);
            if (response.rows != count) {
                throw BackendError("stage " + name() + ": model " + model().name() +
                                   " returned " + std::to_string(response.rows) +
                                   " row(s) for " + std::to_string(count) + " object(s)");
            }
            // Per chunk and before the append, which is the whole point of the seam: the fold
            // reads one response's outputs together, and what is scattered is already one row
            // per crop. `PoolSegment` applies `_reduced` at exactly this point.
            const OutputTensor rows_out = combine_ ? combine_(response) : response.first();
            if (rows_out.data.size() != count * rows_out.row_elems) {
                throw BackendError("stage " + name() + ": the fold answered " +
                                   std::to_string(rows_out.data.size()) + " float(s) for " +
                                   std::to_string(count) + " crop(s) of " +
                                   std::to_string(rows_out.row_elems) +
                                   "; a scatter needs "
                                   "one row per crop");
            }
            out.append(rows_out.data.data(), static_cast<int>(count),
                       static_cast<int>(rows_out.row_elems), payload->object_indices, start);
        }
        state.attach(std::move(out));
        return payload->rows;
    }

}  // namespace shipinfer
