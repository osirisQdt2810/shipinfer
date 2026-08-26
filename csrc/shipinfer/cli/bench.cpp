// The measurable binary: ingest -> fair queue -> perception graph -> reassembly -> sink.
//
// It writes the same buffer-occupancy JSONL the Python driver and the baseline binary write, so
// `benchmarks/harness/analysis.py` scores all three with one implementation. That is the whole
// point of the file format being boring.
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/backends/tensorrt/adapter.h"
#include "shipinfer/backends/tensorrt/engine.h"
#include "shipinfer/core/buffers.h"
#include "shipinfer/core/platform.h"
#include "shipinfer/ingest/sources/replay.h"
#include "shipinfer/obs/sampler.h"
#include "shipinfer/pipeline/graph/dag.h"
#include "shipinfer/pipeline/graph/shapes.h"
#include "shipinfer/pipeline/graph/stages.h"
#include "shipinfer/pipeline/reassembly/collector.h"
#include "shipinfer/runtime/containment.h"
#include "shipinfer/scheduling/policies/registry.h"
#include "shipinfer/scheduling/queues/fair.h"
#include "shipinfer/server/model.h"

using namespace shipinfer;

namespace {

    // One frame's worth of work as it sits in the fair queue.
    //
    // The pixels stay on the **host** here, and the worker copies them to its own device. The
    // first version had the camera thread copy to a device chosen by camera index, and then any
    // worker could pick up any frame — a frame on device 1 executed by an instance on device 0,
    // which is a cross-device access. It does not fail at the call that caused it; it surfaces
    // as `an illegal memory access was encountered` inside an unrelated `gpuDeviceSynchronize`
    // several frames later, which is why ADR-002 makes the rule structural rather than
    // advisory.
    //
    // Keeping one fleet-wide fair queue matters more than saving the copy: fairness across
    // cameras is the thing this project exists to get right, and a queue per device would make
    // it fair only within a device.
    struct FrameWork {
        FrameTag tag;
        std::shared_ptr<FrameState> state;
        HostFrame frame;  // the library owns the pixels and outlives every frame in flight

        size_t rows() const { return 1; }
        std::string camera() const { return tag.camera_id; }
        // A frame into the detector is NORMAL priority with no deadline — what the Python
        // pipeline submits. The lanes above and below exist for the requests that will use
        // them.
        int priority() const { return Priority::Normal; }
        bool expired(int64_t) const { return false; }
    };

    struct Options {
        std::string person_frames;
        std::string ship_frames;
        std::string det_plan;
        std::string seg_plan;
        std::string emb_plan;
        std::string ship_emb_plan;
        std::string log_path = "buffers.jsonl";
        std::vector<int> devices{0};
        int cameras = 12;
        double fps = 10.0;
        double seconds = 40.0;
        int workers = 32;
        int queue_capacity = 65536;
        // The detector's batch window, `dynamic_batching.max_queue_delay_us` in the demo
        // repository's config: once one frame is in, the queue waits this long for a batch to
        // fill.
        int batch_delay_us = 2000;
        // The placement policy for every model, by name — the Python plane's default.
        std::string policy = "locality_spillover";
        int stage_timeout_ms = 5000;
        int reassembly_capacity = 1024;
        int reassembly_timeout_ms = 1500;
        int det_instances = 2;
        int seg_instances = 1;
        int emb_instances = 2;
        int ship_emb_instances = 1;
        double sample_interval_s = 1.0;
    };

    std::vector<int> parse_ints(const std::string& csv) {
        std::vector<int> out;
        std::stringstream stream(csv);
        std::string item;
        while (std::getline(stream, item, ',')) {
            if (!item.empty()) out.push_back(std::stoi(item));
        }
        return out;
    }

    Options parse(int argc, char** argv) {
        Options options;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw ConfigError("missing value for " + flag);
                return argv[++i];
            };
            if (flag == "--person-frames")
                options.person_frames = next();
            else if (flag == "--ship-frames")
                options.ship_frames = next();
            else if (flag == "--det-engine")
                options.det_plan = next();
            else if (flag == "--seg-engine")
                options.seg_plan = next();
            else if (flag == "--emb-engine")
                options.emb_plan = next();
            else if (flag == "--ship-emb-engine")
                options.ship_emb_plan = next();
            else if (flag == "--log-jsonl")
                options.log_path = next();
            else if (flag == "--gpu-ids")
                options.devices = parse_ints(next());
            else if (flag == "--cameras")
                options.cameras = std::stoi(next());
            else if (flag == "--fps")
                options.fps = std::stod(next());
            else if (flag == "--seconds")
                options.seconds = std::stod(next());
            else if (flag == "--workers")
                options.workers = std::stoi(next());
            else if (flag == "--queue-capacity")
                options.queue_capacity = std::stoi(next());
            else if (flag == "--batch-delay-us")
                options.batch_delay_us = std::stoi(next());
            else if (flag == "--policy")
                options.policy = next();
            else if (flag == "--stage-timeout-ms")
                options.stage_timeout_ms = std::stoi(next());
            else if (flag == "--det-instances")
                options.det_instances = std::stoi(next());
            else if (flag == "--seg-instances")
                options.seg_instances = std::stoi(next());
            else if (flag == "--emb-instances")
                options.emb_instances = std::stoi(next());
            else if (flag == "--ship-emb-instances")
                options.ship_emb_instances = std::stoi(next());
            else if (flag == "--sample-interval")
                options.sample_interval_s = std::stod(next());
            else
                throw ConfigError("unknown flag " + flag);
        }
        if (options.person_frames.empty() || options.det_plan.empty()) {
            throw ConfigError("--person-frames and --det-engine are required");
        }
        return options;
    }

    std::string meta_json(const Options& options, const std::vector<std::string>& stages) {
        std::ostringstream out;
        out << "{\"meta\": {\"system\": \"cpp\", \"config\": {";
        out << "\"cameras\": " << options.cameras;
        out << ", \"fps\": " << options.fps;
        out << ", \"seconds\": " << options.seconds;
        out << ", \"workers\": " << options.workers;
        out << ", \"buffer_capacity\": " << options.queue_capacity;
        out << ", \"batch_delay_us\": " << options.batch_delay_us;
        out << ", \"policy\": \"" << options.policy << "\"";
        out << ", \"gpus\": [";
        for (size_t i = 0; i < options.devices.size(); ++i) {
            out << (i ? ", " : "") << options.devices[i];
        }
        out << "]}, \"stages\": [";
        for (size_t i = 0; i < stages.size(); ++i)
            out << (i ? ", " : "") << "\"" << stages[i] << "\"";
        out << "], \"note\": \"C++ data plane; tracking and fused kernels are NOT in this "
               "measurement\"}}";
        return out.str();
    }

}  // namespace

int main(int argc, char** argv) {
    try {
        // Before any device is opened: the rule is enforced in the process that would do
        // the work, and this binary run directly used to be the one spelling that passed
        // both gates. Inside the try, so a host run reports and exits 1 like every other
        // failure instead of terminating.
        shipinfer::runtime::require_container("csrc bench");
        const Options options = parse(argc, argv);

        // -- the models: one Model per plan, one instance per (device x count), each behind the
        // Engine contract — the Python plane's shape (server/model.py) replacing the pool
        // graph.
        std::cerr << "loading engines...\n";
        const auto load_start = std::chrono::steady_clock::now();
        struct Spec {
            std::string name;
            std::string plan;
            int per_device;
            std::vector<int64_t> fed_row;
        };
        const int64_t d = 640;
        const std::vector<Spec> specs{
            {"ship_detector", options.det_plan, options.det_instances, {3, d, d}},
            {"ship_segmenter", options.seg_plan, options.seg_instances, {3, d, d}},
            {"person_embedder", options.emb_plan, options.emb_instances, {3, 256, 128}},
            {"ship_embedder", options.ship_emb_plan, options.ship_emb_instances, {3, 256, 128}},
        };
        std::map<std::string, std::unique_ptr<Model>> models;
        for (const Spec& spec : specs) {
            if (spec.plan.empty()) continue;
            std::vector<std::unique_ptr<ModelInstance>> instances;
            for (int device : options.devices) {
                // One engine per device, shared by that device's instances: the weights are
                // paid for once per GPU rather than once per instance.
                auto engine = TrtEngine::load(spec.plan, device);
                if (device == options.devices.front()) {
                    // Once per model: which `execute()` branch this plan takes. A static plan
                    // is padded to its batch; a dynamic one runs the rows it was given.
                    std::printf("engine %s: max_batch %d, %s plan\n", spec.name.c_str(),
                                engine->max_batch(),
                                engine->is_static() ? "static" : "dynamic");
                }
                // The config is a claim about the plan; the plan is the fact (review of #15).
                if (engine->inputs().empty())
                    throw ConfigError(spec.name + ": the plan declares no input");
                expect_input_row(engine->inputs().front(), spec.fed_row, spec.name);
                for (const TensorSpec& t : engine->inputs()) expect_float32(t, spec.name);
                for (const TensorSpec& t : engine->outputs()) expect_float32(t, spec.name);
                for (int i = 0; i < spec.per_device; ++i) {
                    auto adapter = std::make_unique<TrtEngineAdapter>(
                        std::make_unique<TrtInstance>(engine, device));
                    const BatchWindow window(static_cast<size_t>(engine->max_batch()),
                                             options.batch_delay_us);
                    instances.push_back(std::make_unique<ModelInstance>(
                        spec.name + ":" + std::to_string(device) + ":" + std::to_string(i),
                        std::move(adapter), window, static_cast<size_t>(options.queue_capacity),
                        Overflow::Reject,
                        [](Device dev) { GPU_CHECK(gpuSetDevice(dev.index)); }));
                }
            }
            models[spec.name] = std::make_unique<Model>(spec.name, std::move(instances),
                                                        build_policy(options.policy));
        }
        if (models.count("ship_detector") == 0) throw ConfigError("--det-engine is required");
        for (auto& [name, model] : models) model->start(std::chrono::milliseconds(120000));
        const double startup_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start)
                .count();
        std::cerr << "engines ready in " << startup_s << "s\n";

        // The stage names, for the log's metadata and the collector's expectations.
        std::vector<std::string> stage_names{"detect", "crop"};
        if (models.count("ship_segmenter")) stage_names.push_back("ship_segmenter");
        if (models.count("person_embedder")) stage_names.push_back("person_embedder");
        if (models.count("ship_embedder")) stage_names.push_back("ship_embedder");

        std::atomic<uint64_t> emitted{0};
        std::atomic<uint64_t> complete{0};
        FrameCollector collector(
            [&emitted, &complete](FrameResult&& result) {
                // The null sink: the event is built (the records are the expensive part and
                // they are built here, outside the collector's lock, deliberately) and
                // discarded. Same choice the Python driver makes, so neither side is being
                // measured with a sink the other does not have.
                // Counted apart, because "emitted" and "emitted complete" are different
                // numbers and quoting the first as throughput while most events are
                // Incomplete is not a like-for-like comparison.
                if (result.reason == FinishReason::Complete) complete.fetch_add(1);
                emitted.fetch_add(1);
            },
            static_cast<size_t>(options.reassembly_capacity), options.reassembly_timeout_ms);

        // Frames the queue still held when the run stopped are counted, not destroyed silently:
        // frames_read - frames_accepted is otherwise a number the reader has to explain by
        // hand.
        std::atomic<uint64_t> unread_at_stop{0};
        FairPriorityQueue<FrameWork> queue(
            "pipeline", static_cast<size_t>(options.queue_capacity), Overflow::Reject, 50, true,
            [&unread_at_stop](FrameWork&&, DropReason why) {
                if (why == DropReason::Closed) unread_at_stop.fetch_add(1);
            });

        // -- the sampler: the same log shape as the other two systems ---------------------
        OccupancySampler sampler(
            options.log_path,
            [&]() {
                // `_buffer_size`, exactly: `analysis.BUFFER_SUFFIX` is what the reader keys
                // on, and a log with the wrong suffix is refused outright rather than
                // silently read as empty — which is the right refusal and cost me one run.
                std::map<std::string, long long> row;
                row["pipeline_buffer_size"] = static_cast<long long>(queue.depth());
                for (const auto& [name, model] : models) {
                    row[name + "_buffer_size"] = static_cast<long long>(model->total_depth());
                }
                return row;
            },
            options.sample_interval_s, meta_json(options, stage_names));

        // -- workers ----------------------------------------------------------------------
        std::atomic<bool> stopping{false};
        std::atomic<uint64_t> accepted{0};
        std::atomic<uint64_t> failed{0};
        std::mutex refused_mutex;
        std::map<std::string, uint64_t> open_refused_by_camera;
        std::vector<std::thread> workers;
        const std::vector<std::string> unconditional{"detect", "crop"};
        // The pipeline queue hands frames to workers one at a time, as the Python runner does;
        // the batching happens in each model's own instance queue under its window, across
        // every frame in flight.
        const BatchWindow frame_window(1, 0);
        std::atomic<int> failures_shouted{0};
        for (int w = 0; w < options.workers; ++w) {
            const int device = options.devices[static_cast<size_t>(w) % options.devices.size()];
            workers.emplace_back([&, device]() {
                try {
                    GPU_CHECK(gpuSetDevice(device));
                    WorkerScratch scratch(Device::cuda(device));
                    // The frame's pixels live in this worker's own device buffer; one frame at
                    // a time, and every stage's future is awaited before the next frame
                    // overwrites it.
                    auto pixels = std::make_shared<DeviceBuffer>();

                    Dag dag;
                    dag.add(std::make_unique<DetectStage>(
                        "detect", *models.at("ship_detector"), DetectConfig{}, scratch,
                        std::chrono::milliseconds(options.stage_timeout_ms)));
                    std::vector<CropSpec> crops;
                    if (models.count("person_embedder"))
                        crops.push_back({"person_crops", "person", 0, 256, 128});
                    if (models.count("ship_segmenter"))
                        crops.push_back({"ship_crops_640", "ship", 8, 640, 640});
                    if (models.count("ship_embedder"))
                        crops.push_back({"ship_crops", "ship", 8, 256, 128});
                    if (crops.empty()) crops.push_back({"person_crops", "person", 0, 256, 128});
                    dag.add(std::make_unique<CropStage>("crop", crops,
                                                        DetectConfig{}.max_objects, scratch));
                    if (models.count("ship_segmenter")) {
                        dag.add(std::make_unique<ObjectStage>(
                            "ship_segmenter", *models.at("ship_segmenter"), "ship_crops_640",
                            "ship_segmenter_out",
                            std::chrono::milliseconds(options.stage_timeout_ms)));
                    }
                    if (models.count("person_embedder")) {
                        dag.add(std::make_unique<ObjectStage>(
                            "person_embedder", *models.at("person_embedder"), "person_crops",
                            "person_embedder_out",
                            std::chrono::milliseconds(options.stage_timeout_ms)));
                    }
                    if (models.count("ship_embedder")) {
                        dag.add(std::make_unique<ObjectStage>(
                            "ship_embedder", *models.at("ship_embedder"), "ship_crops",
                            "ship_embedder_out",
                            std::chrono::milliseconds(options.stage_timeout_ms)));
                    }

                    while (!stopping.load()) {
                        auto batch = queue.get_batch(frame_window);
                        if (batch.empty()) {
                            if (queue.is_closed()) break;
                            continue;
                        }
                        for (FrameWork& item : batch) {
                            accepted.fetch_add(1);
                            if (!collector.open(item.state, unconditional)) {
                                failed.fetch_add(1);
                                std::lock_guard<std::mutex> lock(refused_mutex);
                                ++open_refused_by_camera[item.tag.camera_id];
                                continue;
                            }
                            try {
                                const size_t bytes = static_cast<size_t>(item.frame.height) *
                                                     item.frame.width * 3;
                                if (pixels->bytes() < bytes) *pixels = DeviceBuffer(bytes);
                                GPU_CHECK(gpuMemcpyAsync(pixels->get(), item.frame.pixels,
                                                         bytes, gpuMemcpyHostToDevice,
                                                         scratch.stream()));
                                scratch.synchronise();
                                item.state->set_image(pixels, device);
                                CollectorObserver observer(collector, item.tag);
                                for (const StageOutcome& outcome :
                                     dag.execute(*item.state, observer)) {
                                    if (outcome.status == StageStatus::Failed &&
                                        failures_shouted.fetch_add(1) < 5) {
                                        std::cerr << "frame " << item.tag.key() << " stage "
                                                  << outcome.stage
                                                  << " failed: " << outcome.error << "\n";
                                    }
                                }
                            } catch (const std::exception& error) {
                                failed.fetch_add(1);
                                static std::atomic<int> shouted{0};
                                if (shouted.fetch_add(1) < 5) {
                                    std::cerr << "worker on gpu" << device << " failed frame "
                                              << item.tag.key() << ": " << error.what() << "\n";
                                }
                            }
                            // Sealed on every path, so "every opened frame is reported exactly
                            // once" holds even when a stage threw.
                            collector.seal(item.tag);
                        }
                    }
                } catch (const std::exception& error) {
                    std::cerr << "worker on gpu" << device << " exited: " << error.what()
                              << "\n";
                }
            });
        }

        // -- the sweeper ------------------------------------------------------------------
        std::thread sweeper([&]() {
            while (!stopping.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                collector.sweep();
            }
        });

        // -- cameras ----------------------------------------------------------------------
        auto person_library = std::make_shared<ReplaySource>(options.person_frames);
        if (!person_library->pinned()) {
            std::cerr << "warning: the frame library is not page-locked; host->device copies "
                         "will take the slow path\n";
        }
        auto ship_library = options.ship_frames.empty()
                                ? person_library
                                : std::make_shared<ReplaySource>(options.ship_frames);

        std::vector<std::unique_ptr<CameraActor>> cameras;
        for (int c = 0; c < options.cameras; ++c) {
            char name[32];
            std::snprintf(name, sizeof(name), "cam%02d", c);
            // Half the fleet on person frames and half on ship frames, exactly as the baseline
            // splits its source workers — the mix of content decides how many crops the
            // detector produces, so it has to be the same or it is not the same experiment.
            auto library = (c % 2 == 0) ? person_library : ship_library;
            cameras.push_back(std::make_unique<CameraActor>(
                name, library, options.fps,
                [&queue](const FrameTag& tag, HostFrame frame) -> bool {
                    FrameWork work;
                    work.tag = tag;
                    work.frame = frame;
                    work.state =
                        std::make_shared<FrameState>(tag, frame.height, frame.width, 0.0f);
                    return queue.put(std::move(work)) == PutStatus::Accepted;
                }));
        }

        sampler.start();
        for (auto& camera : cameras) camera->start();

        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<long long>(options.seconds * 1000)));

        // -- teardown, in dependency order ------------------------------------------------
        for (auto& camera : cameras) camera->stop();
        stopping.store(true);
        queue.close();
        for (auto& worker : workers) worker.join();
        // After the workers: a model stopped while a worker still had a frame in hand failed
        // that frame's embedder request as 'instance stopped' and sealed it Incomplete at
        // shutdown.
        for (auto& [name, model] : models) model->stop();
        sweeper.join();
        collector.drain();
        sampler.stop();

        uint64_t read = 0, dropped = 0;
        for (const auto& camera : cameras) {
            read += camera->read();
            dropped += camera->dropped();
        }
        const auto stats = queue.stats();

        // Printed in the same shape the Python driver prints, so a human comparing two runs
        // is comparing two identical reports.
        std::cout << "startup_s " << startup_s << "\n";
        std::cout << "frames_read " << read << "\n";
        std::cout << "frames_dropped " << dropped << "\n";
        // Per camera, because 5 000 drops from one starved camera and 100 from each of fifty
        // are the same total — and telling them apart is what ADR-005 exists for.
        for (const auto& camera : cameras) {
            if (camera->dropped() > 0) {
                std::cout << "frames_dropped_by_camera " << camera->id() << " "
                          << camera->dropped() << "\n";
            }
        }
        std::cout << "frames_accepted " << accepted.load() << "\n";
        std::cout << "frames_failed " << failed.load() << "\n";
        for (const auto& [camera, count] : open_refused_by_camera) {
            std::cout << "open_refused_by_camera " << camera << " " << count << "\n";
        }
        std::cout << "events_emitted " << emitted.load() << "\n";
        std::cout << "queue_rejected " << stats.rejected << "\n";
        std::cout << "queue_unread_at_stop " << unread_at_stop.load() << "\n";
        for (const auto& [camera, count] : stats.rejected_by_camera) {
            std::cout << "queue_rejected_by_camera " << camera << " " << count << "\n";
        }
        std::cout << "queue_evicted " << stats.evicted << "\n";
        std::cout << "collector_reported " << collector.reported() << "\n";
        std::cout << "collector_timeouts " << collector.timed_out() << "\n";
        std::cout << "collector_evicted " << collector.evicted() << "\n";
        for (const auto& [camera, count] : collector.evicted_by_camera()) {
            std::cout << "collector_evicted_camera " << camera << " " << count << "\n";
        }
        std::cout << "events_complete " << complete.load() << "\n";
        std::cout << "events_incomplete " << (emitted.load() - complete.load()) << "\n";
        // Requests executed per model per device — the per-device breakdown a PR needs
        // (ADR-006), now read from the instances themselves.
        for (const auto& [name, model] : models) {
            std::map<int, uint64_t> by_device;
            for (const auto& instance : model->instances()) {
                by_device[instance->device().index] += instance->stats().requests;
            }
            std::cout << "per_device " << name;
            for (const auto& [device, count] : by_device)
                std::cout << " " << device << ":" << count;
            std::cout << "\n";
        }
        std::cout << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "shipinfer_pipeline: " << error.what() << "\n";
        return 1;
    }
}
