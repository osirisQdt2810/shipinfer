// The measurable binary: ingest -> fair queue -> perception graph -> reassembly -> sink.
//
// It writes the same buffer-occupancy JSONL the Python driver and the baseline binary write, so
// `benchmarks/harness/analysis.py` scores all three with one implementation. That is the whole
// point of the file format being boring.
#include <atomic>
#include <chrono>
#include <cstdlib>
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
#include "shipinfer/engine/model.h"
#include "shipinfer/ingest/manager.h"
#include "shipinfer/ingest/sink.h"
#include "shipinfer/ingest/sources/replay.h"
#include "shipinfer/obs/sampler.h"
#include "shipinfer/pipeline/events/records.h"
#include "shipinfer/pipeline/graph/bench_models.h"
#include "shipinfer/pipeline/graph/dag.h"
#include "shipinfer/pipeline/graph/from_plan.h"
#include "shipinfer/pipeline/graph/plan.h"
#include "shipinfer/pipeline/graph/plan_stages.h"
#include "shipinfer/pipeline/graph/shapes.h"
#include "shipinfer/pipeline/graph/stages.h"
#include "shipinfer/pipeline/reassembly/collector.h"
#include "shipinfer/runtime/containment.h"
#include "shipinfer/scheduling/policies/registry.h"
#include "shipinfer/scheduling/queues/fair.h"

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

    // The bridge from the ingest plane to the fair queue: a `FrameSink` that turns a tagged
    // frame into one queue entry.
    //
    // It is *here*, in the application, rather than in `ingest/` — mapping a frame onto a unit
    // of scheduled work is dispatch policy, and the same code has to undo the mapping when the
    // results are reassembled. Refusal is thrown rather than returned because the actor is the
    // only component that knows whose frame it is and can charge the drop to that camera
    // (ADR-005).
    class QueueSink : public FrameSink {
      public:
        explicit QueueSink(FairPriorityQueue<FrameWork>& queue) : queue_(queue) {}

        void put(Frame&& frame) override {
            FrameWork work;
            work.tag = frame.tag;
            work.frame = std::move(frame.image);
            work.state = std::make_shared<FrameState>(work.tag, work.frame.height,
                                                      work.frame.width, 0.0f);
            const PutStatus status = queue_.put(std::move(work));
            if (status == PutStatus::Rejected) {
                const QueueStats stats = queue_.stats();
                throw QueueFullError("pipeline queue is full", stats.depth, stats.capacity);
            }
            if (status == PutStatus::Closed) {
                throw RequestCancelledError("pipeline queue is closed");
            }
        }

      private:
        FairPriorityQueue<FrameWork>& queue_;
    };

    struct Options {
        std::string person_frames;
        std::string ship_frames;
        std::string det_plan;
        std::string seg_plan;
        std::string emb_plan;
        std::string ship_emb_plan;
        // The resolved chain to run — `shipinfer plan -t <chain.yaml> -o <file>`. Absent,
        // the defaults below are assembled into the SAME plan struct, so there is one
        // construction path and no second spelling of the graph.
        std::string plan_path;
        // The model repository root the plan's `artefact` paths are relative to. Given with
        // `--plan`, it replaces the four `--*-engine` flags: the plan already says which
        // artefact each slot runs, and two spellings of one path is how they drift.
        std::string repository;
        std::string log_path = "buffers.jsonl";
        std::vector<int> devices{0};
        int cameras = 12;
        double fps = 10.0;
        double seconds = 40.0;
        // The detector's batch window, `dynamic_batching.max_queue_delay_us` in the demo
        // repository's config: once one frame is in, the queue waits this long for a batch to
        // fill.
        int batch_delay_us = 2000;
        // The placement policy for every model, by name — the Python plane's default.
        std::string policy = "locality_spillover";
        // The video source every camera uses, by name in `SOURCES()`. `replay` is the only one
        // this binary links today; naming it rather than hard-coding it is what makes the
        // GStreamer source a new file and nothing else.
        std::string source = "replay";
        int det_instances = 2;
        int seg_instances = 1;
        int emb_instances = 2;
        int ship_emb_instances = 1;
        double sample_interval_s = 1.0;
    };

    // How this binary's flags fill `BenchEngines`, which is the only place they are read.
    BenchEngines engines_of(const Options& options) {
        BenchEngines engines;
        engines.repository = options.repository;
        engines.detector = options.det_plan;
        engines.segmenter = options.seg_plan;
        engines.person_embedder = options.emb_plan;
        engines.ship_embedder = options.ship_emb_plan;
        engines.batch_delay_us = options.batch_delay_us;
        engines.detector_instances = options.det_instances;
        engines.segmenter_instances = options.seg_instances;
        engines.person_embedder_instances = options.emb_instances;
        engines.ship_embedder_instances = options.ship_emb_instances;
        return engines;
    }

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
            else if (flag == "--plan")
                options.plan_path = next();
            else if (flag == "--repository")
                options.repository = next();
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
            else if (flag == "--batch-delay-us")
                options.batch_delay_us = std::stoi(next());
            else if (flag == "--policy")
                options.policy = next();
            else if (flag == "--source")
                options.source = next();
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
        if (options.person_frames.empty()) throw ConfigError("--person-frames is required");
        // An engine comes from ONE of the two doors: `--plan` plus `--repository`, which is
        // what the resolved chain carries (P5-B), or the per-model flags. Both spellings
        // present is refused rather than ranked -- two sources for one path is exactly what
        // the plan replaced, and silently preferring either would hide the disagreement the
        // repository's own numbers already had with these flags.
        const bool from_plan = !options.repository.empty();
        const bool from_flags = !options.det_plan.empty() || !options.seg_plan.empty() ||
                                !options.emb_plan.empty() || !options.ship_emb_plan.empty();
        if (from_plan && from_flags) {
            throw ConfigError(
                "--repository and the --*-engine flags are two spellings of one "
                "path; the plan already names each slot's artefact, so pass one "
                "or the other");
        }
        if (!from_plan && !from_flags) {
            throw ConfigError(
                "no engine to run: pass --plan <file> --repository <root>, or "
                "the --*-engine flags");
        }
        if (from_plan && options.plan_path.empty()) {
            throw ConfigError(
                "--repository names the root a plan's `artefact` paths hang "
                "off, so it needs --plan <file>");
        }
        return options;
    }

    // The run record's first line. `benchmarks/harness/sampler.py` states its contract: it
    // records "the configuration, the resolved artefact paths ... and -- the reason it exists
    // -- which pipeline stages were actually wired", so "the omission travels with the data".
    // Which means the numbers the PLAN owns have to be in it: this printed one global
    // `batch_delay_us` that no instance used once the windows came per model, so two runs a
    // month apart could differ in every window and say the same thing.
    std::string meta_json(const Options& options, const PlanSettings& tuning,
                          const std::vector<std::string>& stages,
                          const std::vector<BenchModel>& models) {
        std::ostringstream out;
        out << "{\"meta\": {\"system\": \"cpp\", \"config\": {";
        out << "\"cameras\": " << options.cameras;
        out << ", \"fps\": " << options.fps;
        out << ", \"seconds\": " << options.seconds;
        // Every carried setting, not the two that used to be flags: the record's contract is
        // that "the omission travels with the data", and a run whose per-instance queue was 64
        // and one whose was 65536 are different measurements that used to print the same line.
        for (const SettingKey& key : setting_keys()) {
            out << ", \"" << key.name << "\": " << tuning.*(key.member);
        }
        if (options.repository.empty()) {
            // The real answer only on the flag path; under `--repository` every instance took
            // its window from the plan and this number is the binary's own default.
            out << ", \"batch_delay_us\": " << options.batch_delay_us;
        }
        out << ", \"policy\": \"" << options.policy << "\"";
        out << ", \"gpus\": [";
        for (size_t i = 0; i < options.devices.size(); ++i) {
            out << (i ? ", " : "") << options.devices[i];
        }
        out << "]}, \"models\": [";
        for (size_t i = 0; i < models.size(); ++i) {
            const BenchModel& model = models[i];
            out << (i ? ", " : "") << "{\"name\": \"" << model.name << "\"";
            out << ", \"engine\": \"" << model.engine << "\"";
            out << ", \"instances_per_device\": " << model.per_device;
            out << ", \"queue_delay_us\": " << model.queue_delay_us << "}";
        }
        out << "], \"stages\": [";
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

        // -- the models, FROM THE PLAN. Which slots run a model, the engine each one runs,
        // how many instances per device, the batch window and the row shape it is fed are all
        // resolved on the Python side and carried (ADR-020). What this used to be is the
        // measurement's real defect: four `--*-instances` flags and one global
        // `--batch-delay-us` restating `model_repository/*/config.yaml`, and disagreeing with
        // it on three of the four models -- so the two planes were benchmarked at different
        // instance counts and the head-to-head was not like for like.
        const ResolvedPlan plan =
            options.plan_path.empty() ? default_bench_plan() : read_plan(options.plan_path);
        // NOT defaulted here. A run this binary configured from its own numbers is exactly
        // what the head-to-head could not be: the worker count, both queue capacities and the
        // reassembly window are `core/settings/`'s, and a plan that states none of them was
        // written by a control plane too old to carry them.
        if (!plan.settings) {
            throw ConfigError(
                "the plan states no `setting` lines, and this binary will not "
                "supply its own: the worker count, the two queue capacities and "
                "the reassembly window are the deployment's (core/settings/). "
                "Rewrite it with `python -m shipinfer plan`");
        }
        const PlanSettings& tuning = *plan.settings;
        const std::vector<BenchModel> specs = bench_models(plan, engines_of(options));
        std::cerr << "loading engines...\n";
        const auto load_start = std::chrono::steady_clock::now();
        std::map<std::string, std::unique_ptr<Model>> models;
        for (const BenchModel& spec : specs) {
            if (spec.engine.empty()) continue;
            std::vector<std::unique_ptr<ModelInstance>> instances;
            for (int device : options.devices) {
                // One engine per device, shared by that device's instances: the weights are
                // paid for once per GPU rather than once per instance.
                auto engine = TrtEngine::load(spec.engine, device);
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
                                             spec.queue_delay_us);
                    instances.push_back(std::make_unique<ModelInstance>(
                        spec.name + ":" + std::to_string(device) + ":" + std::to_string(i),
                        std::move(adapter), window, static_cast<size_t>(tuning.instance_queue),
                        Overflow::Reject,
                        [](Device dev) { GPU_CHECK(gpuSetDevice(dev.index)); }));
                }
            }
            models[spec.name] = std::make_unique<Model>(spec.name, std::move(instances),
                                                        build_policy(options.policy));
        }
        // The chain's head. Named rather than "some model loaded", because a run whose
        // detector is missing produces zero detections and reads as a quiet fleet.
        if (models.count("ship_detector") == 0) {
            throw ConfigError(
                "no `ship_detector` was loaded; the plan must declare a slot "
                "running it and this process must be able to reach its engine");
        }
        for (auto& [name, model] : models) model->start(std::chrono::milliseconds(120000));
        const double startup_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start)
                .count();
        std::cerr << "engines ready in " << startup_s << "s\n";

        // WHAT THIS PROCESS RUNS, from one source. This used to be three hand-kept lists --
        // the stage names here, a label table below, a field map below that, and an
        // `if (models.count(...))` ladder inside each worker -- and they disagreed: the
        // labels said a ship was class 1 while the crop specs said 8, so every ship left the
        // event writer as `unknown` while the right rows were cropped.
        const PlanStages planned = plan_stages(plan, loaded_names(models));
        const std::vector<std::string>& stage_names = planned.stage_names;
        std::cerr << "chain '" << plan.name << "': " << stage_names.size() << " stage(s)";
        if (!planned.unsupported.empty()) {
            std::cerr << ", not run here:";
            for (const std::string& slot : planned.unsupported) std::cerr << " " << slot;
        }
        std::cerr << "\n";

        std::atomic<uint64_t> emitted{0};
        std::atomic<uint64_t> complete{0};
        std::atomic<uint64_t> event_bytes{0};
        //: Events the writer refused -- a non-finite score or mask area, which has no valid
        //: JSON spelling. Counted rather than thrown, and reported beside the totals: an
        //: event lost silently is the failure this whole pipeline was rebuilt to remove.
        std::atomic<uint64_t> unwritable{0};
        //: Per camera as well as in total, which is ADR-005's own argument: the total says
        //: events were lost and only the breakdown says WHOSE. The mutex costs nothing --
        //: this path runs only on a refusal, which on a healthy fleet is never.
        std::mutex unwritable_lock;
        std::map<std::string, uint64_t> unwritable_by_camera;
        // Both from the plan: the class ids are the CHECKPOINT's (this detector calls a ship
        // 8) and the batch names are a stage's OUTPUT name rather than its own
        // (`graph/stages.cpp`: `out.name = output_`), so `from_plan.cpp` derives them once
        // and nothing here can spell either differently.
        const pipeline::events::ClassLabels& labels = planned.labels;
        const pipeline::events::FieldMap& event_fields = planned.fields;
        FrameCollector collector(
            [&emitted, &complete, &event_bytes, &unwritable, &unwritable_lock,
             &unwritable_by_camera, &labels, &event_fields, &options](FrameResult&& result) {
                // The null sink: the event is built -- REALLY built since P5-A; this comment
                // used to claim it while the body only counted -- and then discarded. Same
                // choice the Python driver makes, so neither side is measured with a sink the
                // other does not have, and the records are the expensive part and are built
                // here, outside the collector's lock, deliberately.
                //
                // `event_bytes` is what makes the building unskippable: an optimiser may
                // delete work whose result nothing reads, and a benchmark measuring a deleted
                // event writer is the shape of fault `tests/test_support_models.py` exists for.
                //
                // CAUGHT HERE, and that is not defensive habit: `json_number` refuses a
                // non-finite double -- a NaN score off an fp16 engine is the input its own
                // comment names -- and neither path that reaches this lambda is
                // exception-safe. `collector.seal()` sits outside the worker's per-frame
                // catch, so a throw would end that worker for the rest of the run and the
                // per-device table would report a dead device as a slow one; and
                // `collector.sweep()` runs on a bare `std::thread`, where an escaping
                // exception is `std::terminate`. Refusing to write invalid JSON stays right;
                // what was missing was anything between that refusal and the thread.
                try {
                    const std::string line =
                        pipeline::events::event_of(result.inputs, result.reason, result.missing,
                                                   options.source, labels, event_fields)
                            .to_json();
                    event_bytes.fetch_add(line.size(), std::memory_order_relaxed);
                } catch (const std::exception& error) {
                    // Counted per camera, beside `evicted_by_camera`'s reasoning: the total
                    // says events were lost and only the breakdown says WHOSE, which is the
                    // difference between "the writer refused 4000 events" and "camera 17's
                    // engine is emitting NaNs".
                    unwritable.fetch_add(1, std::memory_order_relaxed);
                    {
                        std::lock_guard<std::mutex> lock(unwritable_lock);
                        ++unwritable_by_camera[result.inputs.tag.camera_id];
                    }
                    static std::atomic<int> shouted{0};
                    if (shouted.fetch_add(1) < 5) {
                        std::cerr << "camera " << result.inputs.tag.camera_id << " frame "
                                  << result.inputs.tag.frame_id
                                  << ": event not writable: " << error.what() << "\n";
                    }
                }
                // Counted apart, because "emitted" and "emitted complete" are different
                // numbers and quoting the first as throughput while most events are
                // Incomplete is not a like-for-like comparison.
                if (result.reason == FinishReason::Complete) complete.fetch_add(1);
                emitted.fetch_add(1);
            },
            static_cast<size_t>(tuning.reassembly_capacity), tuning.reassembly_timeout_ms);

        // Frames the queue still held when the run stopped are counted, not destroyed silently:
        // frames_read - frames_accepted is otherwise a number the reader has to explain by
        // hand.
        std::atomic<uint64_t> unread_at_stop{0};
        FairPriorityQueue<FrameWork> queue(
            "pipeline", static_cast<size_t>(tuning.pipeline_queue), Overflow::Reject,
            tuning.enqueue_block_timeout_ms, true,
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
            options.sample_interval_s, meta_json(options, tuning, stage_names, specs));

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
        for (int w = 0; w < tuning.workers; ++w) {
            const int device = options.devices[static_cast<size_t>(w) % options.devices.size()];
            workers.emplace_back([&, device]() {
                try {
                    GPU_CHECK(gpuSetDevice(device));
                    WorkerScratch scratch(Device::cuda(device));
                    // The frame's pixels live in this worker's own device buffer; one frame at
                    // a time, and every stage's future is awaited before the next frame
                    // overwrites it.
                    auto pixels = std::make_shared<DeviceBuffer>();

                    // The chain, from the plan. Per worker because a Dag holds this
                    // thread's `WorkerScratch`; from the same plan as the tables above, so
                    // the collector's expectations and the stages cannot disagree.
                    Dag dag = build_dag(planned, models, scratch,
                                        std::chrono::milliseconds(tuning.stage_timeout_ms));

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
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(tuning.reassembly_sweep_ms));
                collector.sweep();
            }
        });

        // -- cameras ----------------------------------------------------------------------
        const std::string ship_frames =
            options.ship_frames.empty() ? options.person_frames : options.ship_frames;
        // Held for the whole run, for two reasons: the folders are decoded once here instead of
        // fifty times as the actors start, and `pinned()` is an operator-visible property of
        // the library rather than of any one camera. Every source acquires the same object.
        std::vector<std::shared_ptr<const ReplayLibrary>> libraries;
        if (options.source == "replay") {
            libraries.push_back(ReplayLibrary::acquire(options.person_frames));
            if (ship_frames != options.person_frames) {
                libraries.push_back(ReplayLibrary::acquire(ship_frames));
            }
            for (const auto& library : libraries) {
                if (!library->pinned()) {
                    std::cerr << "warning: the frame library is not page-locked; host->device "
                                 "copies will take the slow path\n";
                }
                if (library->undecodable() > 0) {
                    std::cerr << "warning: " << library->undecodable()
                              << " file(s) did not decode and are not in the replay\n";
                }
            }
        }

        std::vector<IngestConfig> fleet;
        for (int c = 0; c < options.cameras; ++c) {
            char name[32];
            std::snprintf(name, sizeof(name), "cam%02d", c);
            IngestConfig camera;
            camera.camera_id = name;
            // Half the fleet on person frames and half on ship frames, exactly as the baseline
            // splits its source workers — the mix of content decides how many crops the
            // detector produces, so it has to be the same or it is not the same experiment.
            camera.uri = (c % 2 == 0) ? options.person_frames : ship_frames;
            camera.source = options.source;
            camera.fps = options.fps;
            fleet.push_back(std::move(camera));
        }
        QueueSink sink(queue);
        IngestManager manager(std::move(fleet), sink);

        sampler.start();
        manager.start();

        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<long long>(options.seconds * 1000)));

        // -- teardown, in dependency order ------------------------------------------------
        // Read before the fleet is torn down: `stop()` forgets its actors, as the Python
        // manager does, so a stopped manager has no per-camera numbers left to report.
        const std::map<std::string, CameraHealth> camera_health = manager.health();
        const size_t abandoned = manager.stop();
        if (abandoned != 0) {
            // An abandoned actor's detached thread still holds references into this frame —
            // the sink and the queue above all. Unwinding the stack now would free them under
            // a live thread, so take the manager's own containment to its conclusion: report,
            // and leave without running destructors. Unreachable with `replay` (a replay read
            // cannot block), so this is armour for the sources PR2 adds.
            std::cerr << "bench: " << abandoned
                      << " camera(s) abandoned past the stop deadline; exiting without "
                         "unwinding so their threads keep valid references\n";
            std::_Exit(1);
        }
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

        uint64_t read = 0, dropped = 0, published = 0;
        for (const auto& [id, health] : camera_health) {
            read += health.frames_read;
            dropped += health.frames_dropped;
            published += health.frames_published;
        }
        const auto stats = queue.stats();

        // Printed in the same shape the Python driver prints, so a human comparing two runs
        // is comparing two identical reports.
        std::cout << "startup_s " << startup_s << "\n";
        std::cout << "frames_read " << read << "\n";
        std::cout << "frames_published " << published << "\n";
        std::cout << "frames_dropped " << dropped << "\n";
        // Per camera, because 5 000 drops from one starved camera and 100 from each of fifty
        // are the same total — and telling them apart is what ADR-005 exists for.
        for (const auto& [id, health] : camera_health) {
            if (health.frames_dropped > 0) {
                std::cout << "frames_dropped_by_camera " << id << " " << health.frames_dropped
                          << "\n";
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
        // Reported unconditionally, zero included: a number that appears only when it is
        // non-zero is a number a reader does not know to look for.
        std::cout << "events_unwritable " << unwritable.load() << "\n";
        {
            std::lock_guard<std::mutex> lock(unwritable_lock);
            for (const auto& [camera, count] : unwritable_by_camera) {
                std::cout << "events_unwritable_camera " << camera << " " << count << "\n";
            }
        }
        std::cout << "event_bytes " << event_bytes.load() << "\n";
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
