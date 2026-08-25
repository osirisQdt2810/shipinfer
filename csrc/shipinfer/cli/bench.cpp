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
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/core/platform.h"
#include "shipinfer/ingest/sources/replay.h"
#include "shipinfer/obs/sampler.h"
#include "shipinfer/pipeline/graph/graph.h"
#include "shipinfer/pipeline/reassembly/collector.h"
#include "shipinfer/scheduling/queues/fair.h"

using namespace shipinfer;

namespace {

    // One frame's worth of work as it sits in the fair queue.
    //
    // The pixels stay on the **host** here, and the worker copies them to its own device. The first
    // version had the camera thread copy to a device chosen by camera index, and then any worker
    // could pick up any frame — a frame on device 1 executed by an instance on device 0, which is
    // a cross-device access. It does not fail at the call that caused it; it surfaces as `an
    // illegal memory access was encountered` inside an unrelated `gpuDeviceSynchronize` several
    // frames later, which is why ADR-002 makes the rule structural rather than advisory.
    //
    // Keeping one fleet-wide fair queue matters more than saving the copy: fairness across cameras
    // is the thing this project exists to get right, and a queue per device would make it fair
    // only within a device.
    struct FrameWork {
        FrameTag tag;
        std::shared_ptr<FrameState> state;
        HostFrame frame;  // the library owns the pixels and outlives every frame in flight

        size_t rows() const { return 1; }
        std::string camera() const { return tag.camera_id; }
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

    std::string meta_json(const Options& options, const PipelineGraph& graph) {
        std::ostringstream out;
        out << "{\"meta\": {\"system\": \"cpp\", \"config\": {";
        out << "\"cameras\": " << options.cameras;
        out << ", \"fps\": " << options.fps;
        out << ", \"seconds\": " << options.seconds;
        out << ", \"workers\": " << options.workers;
        out << ", \"buffer_capacity\": " << options.queue_capacity;
        out << ", \"gpus\": [";
        for (size_t i = 0; i < options.devices.size(); ++i) {
            out << (i ? ", " : "") << options.devices[i];
        }
        out << "]}, \"stages\": [";
        const auto stages = graph.stage_names();
        for (size_t i = 0; i < stages.size(); ++i)
            out << (i ? ", " : "") << "\"" << stages[i] << "\"";
        out << "], \"note\": \"C++ data plane; tracking and fused kernels are NOT in this "
               "measurement\"}}";
        return out.str();
    }

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);

        GraphConfig graph_config;
        graph_config.detector_plan = options.det_plan;
        graph_config.segmenter_plan = options.seg_plan;
        graph_config.embedder_plan = options.emb_plan;
        graph_config.devices = options.devices;
        graph_config.detector_instances = options.det_instances;
        graph_config.segmenter_instances = options.seg_instances;
        graph_config.embedder_instances = options.emb_instances;
        graph_config.ship_embedder_plan = options.ship_emb_plan;
        graph_config.ship_embedder_instances = options.ship_emb_instances;

        std::cerr << "loading engines...\n";
        const auto load_start = std::chrono::steady_clock::now();
        PipelineGraph graph(graph_config);
        // A frame whose object stages threw is counted by `execute`; the reason is printed for
        // the first few, because a count with no cause is a diagnosis nobody can start.
        graph.on_frame_error([](const FrameTag& tag, const char* what) {
            static std::atomic<int> shouted{0};
            if (shouted.fetch_add(1) < 5)
                std::cerr << "frame " << tag.key() << " failed: " << what << "\n";
        });
        const double startup_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();
        std::cerr << "engines ready in " << startup_s << "s\n";

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

        FairQueue<FrameWork> queue(static_cast<size_t>(options.queue_capacity), Overflow::Reject);

        // -- the sampler: the same log shape as the other two systems ---------------------
        OccupancySampler sampler(
            options.log_path,
            [&]() {
                // `_buffer_size`, exactly: `analysis.BUFFER_SUFFIX` is what the reader keys
                // on, and a log with the wrong suffix is refused outright rather than
                // silently read as empty — which is the right refusal and cost me one run.
                std::map<std::string, long long> row;
                row["pipeline_buffer_size"] = static_cast<long long>(queue.depth());
                row["ship_detector_buffer_size"] = graph.detector().waiting();
                if (graph.segmenter() != nullptr) {
                    row["ship_segmenter_buffer_size"] = graph.segmenter()->waiting();
                }
                if (graph.embedder() != nullptr) {
                    row["person_embedder_buffer_size"] = graph.embedder()->waiting();
                }
                if (graph.ship_embedder() != nullptr) {
                    row["ship_embedder_buffer_size"] = graph.ship_embedder()->waiting();
                }
                return row;
            },
            options.sample_interval_s, meta_json(options, graph));

        // -- workers ----------------------------------------------------------------------
        std::atomic<bool> stopping{false};
        std::atomic<uint64_t> accepted{0};
        std::atomic<uint64_t> failed{0};
        std::vector<std::thread> workers;
        // Only what runs for *every* frame. The conditional per-object stages are added by
        // the graph once the detections are known — see `PipelineGraph::execute`. Expecting
        // all of them up front sealed every ship-only frame as Incomplete with
        // `missing=["person_embedder"]`, which at a 50/50 library split is most of the fleet,
        // and made a real embedder outage emit a byte-identical event.
        const std::vector<std::string> unconditional{"detect", "crop"};

        const int detector_batch = graph.detector().max_batch();
        for (int w = 0; w < options.workers; ++w) {
            // Pinned to one device for its whole life (ADR-002). The worker owns the copy of
            // every frame it takes onto *its* device, so the lease it then takes is on the
            // same device by construction.
            const int device = options.devices[static_cast<size_t>(w) % options.devices.size()];
            workers.emplace_back([&, device]() {
                GPU_CHECK(gpuSetDevice(device));
                // One staging buffer per worker, sized once for a whole detector batch and
                // reused for the run.
                //
                // The first version declared this, wrote a comment saying `gpuMalloc` must not
                // be on the hot path, then voided it with `(void)` and allocated a fresh
                // `DeviceBuffer` per frame anyway. At ~1000 img/s that is a thousand
                // `gpuMalloc` **and a thousand `gpuFree` per second across 48 threads** — and
                // `gpuFree` is device-blocking, it synchronises every stream on the device. So
                // it reintroduced, on every single frame, precisely the device-wide stall that
                // moving preprocessing onto the instance's stream had just removed. Review
                // caught it, and the comment describing the opposite of the code below it was
                // worse than no comment at all.
                DeviceBuffer staging;
                gpuStream_t copy_stream = nullptr;
                GPU_CHECK(gpuStreamCreate(&copy_stream));

                while (!stopping.load()) {
                    // A detector-sized batch, because the plan is static at that batch and
                    // `setInputShape` refuses anything else. Rows, not items: the queue counts
                    // rows and one frame is one detector row.
                    auto batch = queue.drain(static_cast<size_t>(detector_batch), 50);
                    if (batch.empty()) continue;

                    // Grown once, never shrunk, and never freed inside the loop.
                    size_t frame_bytes = 0;
                    for (const auto& item : batch) {
                        frame_bytes = std::max(frame_bytes, static_cast<size_t>(item.frame.height) *
                                                                item.frame.width * 3);
                    }
                    const size_t needed = frame_bytes * static_cast<size_t>(detector_batch);
                    if (staging.bytes() < needed) staging = DeviceBuffer(needed);

                    std::vector<PipelineGraph::Work> work;
                    size_t slot = 0;
                    for (auto& item : batch) {
                        accepted.fetch_add(1);
                        if (!collector.open(item.state, unconditional)) {
                            // Refused, and *counted*. A frame that vanishes here is exactly
                            // the failure the collector exists to prevent.
                            failed.fetch_add(1);
                            continue;
                        }
                        const size_t bytes =
                            static_cast<size_t>(item.frame.height) * item.frame.width * 3;
                        uint8_t* target = staging.as<uint8_t>() + slot * frame_bytes;
                        ++slot;
                        // Async, on this worker's own stream, out of page-locked source pages
                        // (`ReplaySource` registers the library at load). The DMA was already
                        // available and the synchronous call was not using it.
                        GPU_CHECK(gpuMemcpyAsync(target, item.frame.pixels, bytes,
                                                 gpuMemcpyHostToDevice, copy_stream));
                        work.push_back({item.state.get(), target});
                    }
                    GPU_CHECK(gpuStreamSynchronize(copy_stream));

                    if (!work.empty()) {
                        try {
                            // Per-frame failures come back as a count; only a detector failure,
                            // which is a whole-batch fact, still arrives as a throw.
                            failed.fetch_add(graph.execute(work, device, collector));
                        } catch (const std::exception& error) {
                            failed.fetch_add(static_cast<uint64_t>(work.size()));
                            static std::atomic<int> shouted{0};
                            if (shouted.fetch_add(1) < 5) {
                                std::cerr << "worker on gpu" << device
                                          << " failed: " << error.what() << "\n";
                            }
                        }
                    }
                    // Sealed on every path, so "every opened frame is reported exactly once"
                    // holds even when the graph threw halfway through the batch.
                    for (auto& item : batch) collector.seal(item.tag);
                }
                gpuStreamDestroy(copy_stream);
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
                name, library, options.fps, [&queue](const FrameTag& tag, HostFrame frame) -> bool {
                    FrameWork work;
                    work.tag = tag;
                    work.frame = frame;
                    work.state = std::make_shared<FrameState>(tag, frame.height, frame.width, 0.0f);
                    return queue.put(std::move(work));
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
        std::cout << "frames_accepted " << accepted.load() << "\n";
        std::cout << "frames_failed " << failed.load() << "\n";
        std::cout << "events_emitted " << emitted.load() << "\n";
        std::cout << "queue_rejected " << stats.rejected << "\n";
        std::cout << "queue_evicted " << stats.evicted << "\n";
        std::cout << "collector_reported " << collector.reported() << "\n";
        std::cout << "collector_timeouts " << collector.timed_out() << "\n";
        std::cout << "collector_evicted " << collector.evicted() << "\n";
        for (const auto& [camera, count] : collector.evicted_by_camera()) {
            std::cout << "collector_evicted_camera " << camera << " " << count << "\n";
        }
        std::cout << "events_complete " << complete.load() << "\n";
        std::cout << "events_incomplete " << (emitted.load() - complete.load()) << "\n";
        std::cout << "per_device";
        for (const auto& [device, count] : graph.detector().per_device()) {
            std::cout << " ship_detector:gpu" << device << "=" << count;
        }
        if (graph.embedder() != nullptr) {
            for (const auto& [device, count] : graph.embedder()->per_device()) {
                std::cout << " person_embedder:gpu" << device << "=" << count;
            }
        }
        if (graph.segmenter() != nullptr) {
            for (const auto& [device, count] : graph.segmenter()->per_device()) {
                std::cout << " ship_segmenter:gpu" << device << "=" << count;
            }
        }
        if (graph.ship_embedder() != nullptr) {
            for (const auto& [device, count] : graph.ship_embedder()->per_device()) {
                std::cout << " ship_embedder:gpu" << device << "=" << count;
            }
        }
        std::cout << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "shipinfer_pipeline: " << error.what() << "\n";
        return 1;
    }
}
