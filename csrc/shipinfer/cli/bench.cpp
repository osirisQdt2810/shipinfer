// The measurable binary: ingest -> fair queue -> perception graph -> reassembly -> sink.
//
// It writes the same buffer-occupancy JSONL the Python driver and the baseline binary write, so
// `benchmarks/harness/analysis.py` scores all three with one implementation. That is the whole
// point of the file format being boring.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "shipinfer/backends/tensorrt/adapter.h"
#include "shipinfer/backends/tensorrt/engine.h"
#include "shipinfer/core/buffers.h"
#include "shipinfer/core/env.h"
#include "shipinfer/core/join_on_unwind.h"
#include "shipinfer/core/percentile.h"
#include "shipinfer/core/platform.h"
#include "shipinfer/core/thread_name.h"
#include "shipinfer/engine/model.h"
#include "shipinfer/ingest/camera_uris.h"
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
#include "shipinfer/pipeline/queue_sink.h"
#include "shipinfer/pipeline/reassembly/collector.h"
#include "shipinfer/pipeline/tracking/associator.h"
#include "shipinfer/runtime/containment.h"
#include "shipinfer/scheduling/policies/registry.h"
#include "shipinfer/scheduling/queues/fair.h"

using namespace shipinfer;

namespace {

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
        // The video source every camera uses, by name in `SOURCES()`. `replay` is the only one
        // this binary links today; naming it rather than hard-coding it is what makes the
        // GStreamer source a new file and nothing else.
        std::string source = "replay";
        // One URI per camera, written by the control plane. Empty means the `replay` shape,
        // where every even camera shares `--person-frames` because a folder of JPEGs IS
        // shared by design; a real source needs one URI each (`ingest/camera_uris.h`).
        std::string camera_uris;
        int det_instances = 2;
        int seg_instances = 1;
        int emb_instances = 2;
        int ship_emb_instances = 1;
        double sample_interval_s = 1.0;
        // doc: long why this is a knob at all, and why the DEFAULT stays what a server wants
        //: How long the fleet has to drain, charged ONCE to the fleet (`IngestManager::stop`).
        //: 5000 is the manager's own default and is what a server wants -- past it a camera's
        //: thread is detached rather than joined, deliberately, so a shutdown cannot hang.
        //: It is a flag because a SATURATED measurement needs longer: at 50x20 on the host
        //: path, 48 of 50 camera threads are inside a software decode of a 2K frame when the
        //: signal arrives, the budget is gone before they are joined, and the bench exits
        //: without a summary -- so the run that most needs reading produces no numbers.
        int stop_deadline_ms = 5000;
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
            else if (flag == "--source")
                options.source = next();
            else if (flag == "--camera-uris")
                options.camera_uris = next();
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
            else if (flag == "--stop-deadline-ms")
                options.stop_deadline_ms = std::stoi(next());
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
    std::string meta_json(const Options& options, const ResolvedPlan& plan,
                          const std::vector<std::string>& stages,
                          const std::vector<std::string>& unsupported,
                          const std::vector<BenchModel>& models, bool tracked) {
        const PlanSettings& tuning = *plan.settings;
        std::ostringstream out;
        out << "{\"meta\": {\"system\": \"cpp\", \"config\": {";
        out << "\"cameras\": " << options.cameras;
        out << ", \"fps\": " << options.fps;
        out << ", \"seconds\": " << options.seconds;
        // THE SOURCE, because a run whose artefact does not say where its frames came from
        // cannot be read afterwards. Two runs of this bench at the same shape differed by 16%
        // and the only way to tell whether one of them had taken the replay path was to trust
        // the shell history -- which is not evidence. The Python harness records it in
        // `summary.json` for the same reason.
        out << ", \"source\": \"" << options.source << "\"";
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
        out << ", \"policy\": \"" << plan.policy << "\"";
        for (const auto& [key, value] : plan.policy_options) {
            out << ", \"policy." << key << "\": \"" << value << "\"";
        }
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
        // AND THE SLOTS THIS PLANE DID NOT RUN, which is the half the contract above asks for
        // and the half that was missing: `stages` says what WAS wired, and a reader without
        // the chain file beside them cannot subtract. It is printed on stderr at start-up
        // ("not run here: decode track mtmc output") and that line does not survive into the
        // artefact, so a throughput number could be quoted from a run whose chain was missing
        // a third of its stages with nothing in the record to say so. It could, and I did.
        out << "], \"unsupported\": [";
        for (size_t i = 0; i < unsupported.size(); ++i)
            out << (i ? ", " : "") << "\"" << unsupported[i] << "\"";
        // DERIVED, and it used to be a constant. "tracking ... NOT in this measurement" was
        // true until this plane grew a tracking stage, and a static claim is the kind that
        // keeps being printed after it stops being true.
        //
        // FROM THE PLAN, passed in, and NOT from `stages` -- which holds SLOT names, so a chain
        // whose tracker is called `tap:` (as this tree's own fixtures do) would have run one
        // and stamped "neither is tracking", relocating the same lie onto a name coincidence.
        out << "], \"note\": \"C++ data plane; fused kernels are NOT in this measurement";
        if (!tracked) out << ", and neither is tracking";
        out << "\"}}";
        return out.str();
    }

    /// A microsecond span as `uint32_t`, saturating rather than wrapping.
    ///
    /// `int64_t` in, and 2^32 us is ~71 minutes: unreachable at any run length this binary
    /// takes, and a wrap would report the longest frame in the run as the shortest. Saturating
    /// costs one comparison and removes the question.
    uint32_t microseconds_clamped(int64_t value) {
        constexpr int64_t kMax = std::numeric_limits<uint32_t>::max();
        return static_cast<uint32_t>(std::clamp<int64_t>(value, 0, kMax));
    }

    /// One window's percentiles, under the caller's lock, with a `name_` prefix.
    ///
    /// PERCENTILES BECAUSE A MEAN HIDES THE TAIL, and the tail is what a 50-camera fleet is
    /// judged on. TAKES THE LOCK ITSELF -- #210's own review note: the call sites are on the
    /// drain path where nothing is still pushing, and that is exactly the safety that stops
    /// holding the day one moves.
    void report_window(const char* prefix, std::mutex& guard, std::vector<uint32_t>& samples) {
        const std::lock_guard<std::mutex> held(guard);
        std::cout << prefix << "_samples " << samples.size() << "\n";
        std::cout << prefix << "_p50 " << percentile(samples, 0.50) << "\n";
        std::cout << prefix << "_p95 " << percentile(samples, 0.95) << "\n";
        std::cout << prefix << "_p99 " << percentile(samples, 0.99) << "\n";
        std::cout << prefix << "_max "
                  << (samples.empty() ? 0 : *std::max_element(samples.begin(), samples.end()))
                  << "\n";
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
        // Not defaulted either, and for the sharper version of the same reason: the placement
        // policy is the seam this project exists to own, and it used to come from `--policy`
        // with this binary's own default sitting beside `scheduler.placement_policy`. They
        // agreed, which is why nothing noticed; an operator who changed the setting got it on
        // one plane. `build_policy` refuses an unknown name and lists the known ones.
        if (plan.policy.empty()) {
            throw ConfigError(
                "the plan states no `policy`, and this binary will not choose "
                "one: the placement policy is `scheduler.placement_policy`. "
                "Rewrite it with `python -m shipinfer plan`");
        }
        const std::vector<BenchModel> specs = bench_models(plan, engines_of(options));
        // doc: long the flag, why it is off by default, and what the measurement was
        // BEFORE any engine, because the driver refuses this once a device has a context.
        //
        // CUDA's default is `cudaDeviceScheduleAuto`, which spins when the active contexts do
        // not outnumber the logical processors -- this box, at five devices and 48 cores. With
        // it blocking instead, MEASURED over two interleaved pairs at the design load: the
        // model-instance threads' host CPU HALVES (632 -> 279 CPU-s), the pipeline workers get
        // 32% more because the spin was starving them, and events rise ~15%.
        //
        // OFF BY DEFAULT because it trades wake-up latency for host CPU, and the host is only
        // the wall at this load -- at a fifth of it the trade goes the other way.
        if (env_flag("SHIPINFER_CUDA_BLOCKING_SYNC")) {
            // The device goes back where it was: harmless HERE, since nothing below has
            // chosen one yet, and not harmless the day this moves. That is #203 round 3 on
            // the Python plane -- an unrestored device made an argument-less
            // `torch.cuda.synchronize()` wait on a different GPU in the two arms.
            int previous = 0;
            GPU_CHECK(gpuGetDevice(&previous));
            for (int device : options.devices) {
                GPU_CHECK(gpuSetDevice(device));
                GPU_CHECK(gpuSetDeviceFlags(gpuDeviceScheduleBlockingSync));
            }
            GPU_CHECK(gpuSetDevice(previous));
            std::printf("cuda: blocking sync on %zu device(s)\n", options.devices.size());
        }
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
            models[spec.name] =
                std::make_unique<Model>(spec.name, std::move(instances),
                                        build_policy(plan.policy, plan.policy_options));
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
        // ONCE FOR THE PROCESS, before any worker starts. A barrier per Dag is a barrier per
        // WORKER, and a cross-camera tracker that saw one worker's frames would be doing
        // within-camera deduplication -- which is the failure `mtmc/barrier.h` exists to
        // prevent, so `build_dag` refuses rather than building its own.
        const MtmcRuntime cross_camera = mtmc_runtime(planned, tuning);
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
        //: WHICH STAGE did not deliver, by name, for every result that carries a missing
        //: list -- Timeout, Evicted and Shutdown as well as Incomplete, because
        //: `finish_locked` fills it for all four and a stage lost to a timeout is worth the
        //: same line. So these lines need NOT sum to `events_incomplete`: a frame missing two
        //: stages counts twice, and an evicted frame that delivered everything counts zero.
        //: The total alone reads as a timeout and is not one: measured 11 Sep, 1 483 of 9 520
        //: events with `collector_timeouts 0`, and nothing said which stage. A tally under a
        //: mutex costs nothing on a healthy run, where the map stays empty.
        std::mutex missing_lock;
        std::map<std::string, uint64_t> missing_by_stage;
        //: Every finished frame's reassembly wait, so the run can report PERCENTILES. A vector
        //: and `nth_element` rather than a histogram: reserved once, exact, and a bucket
        //: boundary is an argument nobody then has to have. The lock is not the cost on this
        //: path -- the same lambda builds a JSON line per event.
        std::mutex latency_lock;
        std::vector<uint32_t> latency_us;
        latency_us.reserve(1 << 20);
        //: CAPTURE TO EMISSION -- what the Python plane calls "the number the deployment is
        //: judged on" (`pipeline/metrics.py`). `core/events/schema.cpp` already computes it on
        //: every event from `FrameTag::captured_ns`; nothing summarised it. Under the SAME
        //: lock, so a finished frame costs one acquisition and not two.
        std::vector<uint32_t> frame_us;
        frame_us.reserve(1 << 20);
        // Both from the plan: the class ids are the CHECKPOINT's (this detector calls a ship
        // 8) and the batch names are a stage's OUTPUT name rather than its own
        // (`graph/stages.cpp`: `out.name = output_`), so `from_plan.cpp` derives them once
        // and nothing here can spell either differently.
        const pipeline::events::ClassLabels& labels = planned.labels;
        const pipeline::events::FieldMap& event_fields = planned.fields;
        FrameCollector collector(
            [&emitted, &complete, &event_bytes, &unwritable, &unwritable_lock,
             &unwritable_by_camera, &missing_lock, &missing_by_stage, &latency_lock,
             &latency_us, &frame_us, &labels, &event_fields, &options](FrameResult&& result) {
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
                //: Zero means "not stamped", which the replay path can be; the sample count
                //: is reported so a reader can tell that from a fast run.
                int64_t captured_to_emitted_us = 0;
                try {
                    const auto event =
                        pipeline::events::event_of(result.inputs, result.reason, result.missing,
                                                   options.source, labels, event_fields);
                    // BEFORE `to_json`, which is the call that can refuse: a run whose events
                    // carry a NaN would otherwise report the latency of the ones that wrote.
                    captured_to_emitted_us = event.latency_us;
                    const std::string line = event.to_json();
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
                if (!result.missing.empty()) {
                    std::lock_guard<std::mutex> lock(missing_lock);
                    for (const std::string& stage : result.missing) ++missing_by_stage[stage];
                }
                emitted.fetch_add(1);
                // Every finished frame, complete or not: a run whose tail is timeouts is
                // exactly the run whose latency a reader needs, and dropping those samples
                // would report the p99 of the frames that went well.
                {
                    std::lock_guard<std::mutex> lock(latency_lock);
                    latency_us.push_back(microseconds_clamped(result.waited_us));
                    if (captured_to_emitted_us > 0) {
                        frame_us.push_back(microseconds_clamped(captured_to_emitted_us));
                    }
                }
            },
            static_cast<size_t>(tuning.reassembly_capacity), tuning.reassembly_timeout_ms);

        // Frames the queue still held when the run stopped are counted, not destroyed silently:
        // frames_read - frames_accepted is otherwise a number the reader has to explain by
        // hand.
        std::atomic<uint64_t> unread_at_stop{0};
        // ONE LANE, unless this run's source hands over device pixels -- in which case a frame
        // cannot move between GPUs (ADR-004) and each worker must see only its own GPU's
        // cameras. `pipeline/queue_sink.h` argues why one lane is the default and what the
        // per-GPU shape gives up. Asked of the REGISTRY rather than of a source, because the
        // lanes have to exist before any camera connects.
        const bool device_lanes = SOURCES().produces_device_frames(options.source);
        // doc: long a lane with no consumer, and why it cannot be found by running it
        // A LANE NEEDS A WORKER, refused here rather than discovered as a drop count. Workers
        // are bound `w % devices.size()`, so `workers` below the device count leaves the last
        // lanes with NO CONSUMER for the life of the run: their cameras' frames fill the lane
        // and are then rejected forever, with every camera reporting `Streaming` and no error
        // anywhere. `setting workers 4` is an ordinary value -- it is what the parity fixtures
        // use -- and `--devices 0,1,2,3,4` is an ordinary invocation.
        //
        // This is the one starvation shape that CANNOT be found by running it: it looks exactly
        // like ordinary backpressure. #160 refused it as a side effect of refusing every
        // multi-GPU device run; the lanes made that refusal unnecessary and this one necessary.
        if (device_lanes && tuning.workers < static_cast<int>(options.devices.size())) {
            throw ConfigError(
                "a device source needs at least one worker per GPU: " +
                std::to_string(tuning.workers) + " worker(s) for " +
                std::to_string(options.devices.size()) +
                " GPU(s) would leave the last lane(s) with no consumer, and their cameras' "
                "frames rejected for the life of the run. Raise `pipeline.workers`, or give "
                "this bench fewer `--devices`");
        }
        PipelineLanes lanes(
            options.devices, device_lanes, static_cast<size_t>(tuning.pipeline_queue),
            tuning.enqueue_block_timeout_ms, [&unread_at_stop](FrameWork&&, DropReason why) {
                if (why == DropReason::Closed) unread_at_stop.fetch_add(1);
            });

        // doc: long the declaration order is load-bearing here, and #160 got it backwards
        // DECLARED HERE, before the sampler that reports its pool and before the
        // `JoinOnUnwind` that stops the workers -- so it is destroyed AFTER them. The surfaces
        // hold their pool through a `shared_ptr` (`pipeline/surface_intake.h`), which makes the
        // order not matter; having it right as well is cheap, and #160 round 1 was a lifetime
        // argued from a declaration order that was the other way round.
        //
        // The pool's cap is `SurfaceIntake`'s own default, not the queue's capacity: it bounds
        // IDLE buffers, and the queue's number bounds in-flight ones (`queue_sink.h`).
        // The cap is DERIVED: the buffers that can be idle at once is bounded by how many can
        // be in flight from the worker side, which is this device's worker count. Measured at
        // 8 cameras on one GPU with 23 workers: the pool plateaus at 25 and a cap of 128 never
        // frees, while 8 sat at its cap and churned a `cudaMalloc`/`cudaFree` per frame. A
        // fixed number is wrong for a design-load run, so it scales with the pool that feeds
        // it.
        QueueSink sink(lanes,
                       static_cast<size_t>(std::max<int>(
                           1, tuning.workers / static_cast<int>(options.devices.size()))) +
                           8,
                       options.devices);

        // -- the sampler: the same log shape as the other two systems ---------------------
        OccupancySampler sampler(
            options.log_path,
            [&]() {
                // `_buffer_size`, exactly: `analysis.BUFFER_SUFFIX` is what the reader keys
                // on, and a log with the wrong suffix is refused outright rather than
                // silently read as empty — which is the right refusal and cost me one run.
                std::map<std::string, long long> row;
                row["pipeline_buffer_size"] = static_cast<long long>(lanes.depth());
                // The NV12 pool, in the same log: "is the cap buying anything" is a question a
                // run should answer. Zero on every host run, which is the right answer there.
                row["pipeline_pool_size"] = static_cast<long long>(sink.pooled_buffers());
                for (const auto& [name, model] : models) {
                    row[name + "_buffer_size"] = static_cast<long long>(model->total_depth());
                }
                return row;
            },
            options.sample_interval_s,
            meta_json(options, plan, stage_names, planned.unsupported, specs,
                      !planned.tracks.empty()));

        // -- workers ----------------------------------------------------------------------
        std::atomic<bool> stopping{false};
        std::atomic<uint64_t> accepted{0};
        std::atomic<uint64_t> failed{0};
        std::mutex refused_mutex;
        std::map<std::string, uint64_t> open_refused_by_camera;
        std::vector<std::thread> workers;
        // doc: long why `crop` came off this list, with the number that found it
        // DETECT ALONE. `crop` was here too, and it cannot run on a frame with no detections
        // (`Dag::runnable` requires every `needs()` input NON-EMPTY), so every such frame was
        // sealed Incomplete for a stage that had nothing to do -- while `collector_timeouts`
        // said 0 and nothing said which stage. Measured 11 Sep on the pan fixture, before and
        // after: `events_incomplete 1123` of 7 123, all of them `events_missing_stage crop`.
        // Nothing was lost in any of them.
        //
        // `detect` STAYS, and not for symmetry: `open` with an empty expected set makes
        // `complete()` trivially true (an empty set is a subset of anything), so a frame that
        // died before its first stage would be reported complete. One stage that always runs
        // -- it needs only the frame's pixels -- is what keeps that from being true.
        //
        // Everything else is added by `CollectorObserver::planned`, which `Dag::execute` calls
        // with the runnable set before EVERY stage, so a stage is expected exactly when it can
        // run. The Python plane has always done it this way: `pipeline/runner.py` calls
        // `collector.open(state)` with no `expected` at all.
        // THE DETECTOR'S SLOT, not the kind. `plan_stages` names the stage `detect->slot` and
        // `from_plan` builds `DetectStage(planned.detect_slot, ...)`, so a chain that spells it
        // anything -- `kind: detect` exists for exactly that, and
        // `scenarios/plans/defaults.yaml` ships `detect_small` -- would have every frame
        // expecting a stage nothing delivers. That is this PR's own artefact one level up:
        // `events_complete 0` and `events_missing_stage detect N` naming a stage not in the
        // chain, while `stage_names` above prints the real slot on the same stdout.
        const std::vector<std::string> unconditional{planned.detect_slot};
        // The pipeline queue hands frames to workers one at a time, as the Python runner does;
        // the batching happens in each model's own instance queue under its window, across
        // every frame in flight.
        const BatchWindow frame_window(1, 0);
        std::atomic<int> failures_shouted{0};
        for (int w = 0; w < tuning.workers; ++w) {
            const size_t slot = static_cast<size_t>(w) % options.devices.size();
            const int device = options.devices[slot];
            // The lane this worker pulls from: its own GPU's when there is one per GPU, and the
            // single fleet-wide one otherwise. `slot` indexes `options.devices` and
            // `PipelineLanes` was built from that same list in that order, which is what makes
            // the two agree without a lookup.
            const size_t lane_index = device_lanes ? slot : 0;
            workers.emplace_back([&, w, device, lane_index]() {
                name_this_thread(thread_name("pipe", std::to_string(w)));
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
                                        std::chrono::milliseconds(tuning.stage_timeout_ms),
                                        cross_camera);

                    PipelineLanes::Queue& queue = lanes.lane(lane_index);
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
                                if (item.device >= 0) {
                                    // ALREADY ON A DEVICE, and it cannot move: ADR-004 says a
                                    // frame stays where it was decoded. The state carries the
                                    // surface from the sink, so there is nothing to upload --
                                    // which is the whole of V156's route, and the check below
                                    // is what stops a worker reading another GPU's pointer.
                                    if (item.device != device) {
                                        throw ConfigError(
                                            "frame " + item.tag.key() + " was decoded on gpu" +
                                            std::to_string(item.device) +
                                            " and this worker is on gpu" +
                                            std::to_string(device) +
                                            ": a device frame cannot move (ADR-004), and a "
                                            "lane per GPU is supposed to make this "
                                            "unreachable -- so it is a lane/worker mapping "
                                            "fault, not a configuration one");
                                    }
                                } else {
                                    const size_t bytes =
                                        static_cast<size_t>(item.frame.height) *
                                        item.frame.width * 3;
                                    if (pixels->bytes() < bytes) *pixels = DeviceBuffer(bytes);
                                    GPU_CHECK(gpuMemcpyAsync(pixels->get(), item.frame.pixels,
                                                             bytes, gpuMemcpyHostToDevice,
                                                             scratch.stream()));
                                    scratch.synchronise();
                                    item.state->set_image(pixels, device);
                                }
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
            name_this_thread("sweeper");
            while (!stopping.load()) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(tuning.reassembly_sweep_ms));
                collector.sweep();
            }
        });

        // Declared AFTER the threads it watches, so it is destroyed BEFORE them: everything
        // from here to the shutdown block can throw, and until this existed such a throw
        // aborted the process instead of reporting.
        JoinOnUnwind join_on_unwind(stopping);
        for (std::thread& worker : workers) join_on_unwind.watch(worker);
        join_on_unwind.watch(sweeper);
        join_on_unwind.wake_with([&lanes]() { lanes.close(); });

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

        // Read BEFORE the loop so a short list fails at start-up rather than on camera 27.
        const std::vector<std::string> uris =
            options.camera_uris.empty()
                ? std::vector<std::string>{}
                : read_camera_uris(options.camera_uris,
                                   static_cast<size_t>(std::max(options.cameras, 0)));
        std::vector<IngestConfig> fleet;
        for (int c = 0; c < options.cameras; ++c) {
            char name[32];
            std::snprintf(name, sizeof(name), "cam%02d", c);
            IngestConfig camera;
            camera.camera_id = name;
            // Half the fleet on person frames and half on ship frames, exactly as the baseline
            // splits its source workers — the mix of content decides how many crops the
            // detector produces, so it has to be the same or it is not the same experiment.
            // A URI list carries that split itself: whoever writes it puts the person streams
            // first, exactly as `harness/shipinfer.py::_rtsp_cameras` does, so both planes get
            // the mix from one place instead of each deriving it.
            camera.uri = uris.empty() ? ((c % 2 == 0) ? options.person_frames : ship_frames)
                                      : uris[static_cast<size_t>(c)];
            camera.source = options.source;
            camera.fps = options.fps;
            // WHICH GPU DECODES, and it has to be said rather than defaulted: a device source
            // takes it from this option and its default is 0, so `--devices 3` had every camera
            // decoding on a GPU this process does not drive. Only for a source that produces
            // device frames, because the others refuse an option they do not know.
            if (SOURCES().produces_device_frames(options.source)) {
                camera.options["device"] = std::to_string(
                    options.devices[static_cast<size_t>(c) % options.devices.size()]);
            }
            fleet.push_back(std::move(camera));
        }
        IngestManager manager(std::move(fleet), sink);

        sampler.start();
        manager.start();

        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<long long>(options.seconds * 1000)));

        // -- teardown, in dependency order ------------------------------------------------
        // Read before the fleet is torn down: `stop()` forgets its actors, as the Python
        // manager does, so a stopped manager has no per-camera numbers left to report.
        const std::map<std::string, CameraHealth> camera_health = manager.health();

        // doc: long why one reporter serves both exits, and what the abandoned one cannot say
        // PRINTED ON BOTH EXITS. A run that abandons one camera of fifty used to print no
        // counters at all -- 70 s of work discarded by a shutdown detail. Measured: the
        // `gstreamer` arm at 50x20x70 s abandons and reports nothing, while `nvdec` at the
        // same load reports in full, so the arm V137/V156 mandate was the unmeasurable one.
        // This half needs no drain: `camera_health` is already copied above, the rest are
        // atomics and the queue's own counters, and a READ frees nothing. One lambda rather
        // than a second copy of the prints, because a report that exists twice drifts.
        const auto report_ingest_and_queue = [&] {
            uint64_t read = 0, dropped = 0, published = 0;
            for (const auto& [id, health] : camera_health) {
                read += health.frames_read;
                dropped += health.frames_dropped;
                published += health.frames_published;
            }
            const auto stats = lanes.stats();

            // Printed in the same shape the Python driver prints, so a human comparing two
            // runs is comparing two identical reports.
            std::cout << "startup_s " << startup_s << "\n";
            std::cout << "frames_read " << read << "\n";
            std::cout << "frames_published " << published << "\n";
            std::cout << "frames_dropped " << dropped << "\n";
            // Per camera, because 5 000 drops from one starved camera and 100 from each of
            // fifty are the same total -- and telling them apart is what ADR-005 exists for.
            for (const auto& [id, health] : camera_health) {
                if (health.frames_dropped > 0) {
                    std::cout << "frames_dropped_by_camera " << id << " "
                              << health.frames_dropped << "\n";
                }
            }
            std::cout << "frames_accepted " << accepted.load() << "\n";
            std::cout << "frames_failed " << failed.load() << "\n";
            // UNDER ITS MUTEX, and that is new here rather than pedantry: the old read stood
            // after `stopping.store(true)` and the worker joins, so no writer existed and no
            // lock was needed. The abandoned path calls this BEFORE either, with every worker
            // still in `while (!stopping.load())` and the detached actor still pushing frames,
            // so `collector.open` can refuse and insert a new key mid-iteration.
            {
                std::lock_guard<std::mutex> lock(refused_mutex);
                for (const auto& [camera, count] : open_refused_by_camera) {
                    std::cout << "open_refused_by_camera " << camera << " " << count << "\n";
                }
            }
            std::cout << "events_emitted " << emitted.load() << "\n";
            std::cout << "queue_rejected " << stats.rejected << "\n";
            std::cout << "queue_unread_at_stop " << unread_at_stop.load() << "\n";
            for (const auto& [camera, count] : stats.rejected_by_camera) {
                std::cout << "queue_rejected_by_camera " << camera << " " << count << "\n";
            }
            std::cout << "queue_evicted " << stats.evicted << "\n";
            // PER SLOT, from the associators rather than from a worker's Dag: one associator
            // serves every worker for one slot, so this is the run's answer. A reordered
            // frame is published with no ids and counted HERE -- not in `frames_failed` and
            // not as a reassembly timeout, which is what it used to become.
            for (const tracking::MadeAssociator& made : tracking::made_associators()) {
                std::cout << "track_frames_untracked " << made.slot << " "
                          << made.associator->untracked_frames() << "\n";
            }
            // The same shape for the cross-camera half: one tracker per (impl, slot), and an
            // instant it refused is published with null ids rather than failing a frame. A run
            // with a non-zero count here has real data the gate or the assigner would not take.
            // CLOSED FIRST, because `barrier.h` says that is what `close_all` is for: a
            // worker parked in `submit` would otherwise hold shutdown for the rest of its
            // window, and every instant still open at stop is counted as `window` or
            // `expired` rather than `shutdown` -- the run's own ledger mislabelling its last
            // instants (#222's review).
            for (const auto& [slot, barrier] : cross_camera.barriers) {
                const size_t closed = barrier->close_all();
                std::cout << "mtmc_instants " << slot << " closed_at_stop " << closed << "\n";
            }
            // THE BARRIER'S OWN LEDGER, per slot, and it is the line that says whether the
            // stage contributed at all: an instant that never closed on evidence answers no
            // ids, and without this the run reports `mtmc_identities 0 0` with no way to tell
            // "nothing to associate" from "every instant missed its window".
            for (const auto& [slot, barrier] : cross_camera.barriers) {
                for (const auto& [reason, count] : barrier->instant_stats()) {
                    std::cout << "mtmc_instants " << slot << " " << reason << " " << count
                              << "\n";
                }
                for (const auto& [reason, count] : barrier->frame_stats()) {
                    std::cout << "mtmc_frames " << slot << " " << reason << " " << count
                              << "\n";
                }
                // THE ROSTER'S OWN LINE, and it is why the reasons above read as they do: a
                // declared camera is waited for whether it exists or not, so a roster naming
                // cameras this fleet does not have makes `complete` unreachable and every
                // instant closes on its window. Measured 11 Sep -- `ship_person_cpu.yaml`
                // declares `cam-01 ... cam-04` and every bench fleet is `cam00 ... cam11`, so
                // not ONE instant closed complete in six runs and nothing said why. Printed
                // only when there is something to say, because an empty list every run is a
                // line readers learn to skip.
                const std::set<std::string> silent = barrier->silent_cameras();
                if (!silent.empty()) {
                    std::cout << "mtmc_cameras_silent " << slot << " ";
                    bool first = true;
                    for (const std::string& camera : silent) {
                        std::cout << (first ? "" : ",") << camera;
                        first = false;
                    }
                    std::cout << "\n";
                }
            }
            for (const mtmc::MadeClusterTracker& made : mtmc::made_cluster_trackers()) {
                std::cout << "mtmc_instants_refused " << made.slot << " "
                          << made.tracker->refused_instants() << "\n";
                std::cout << "mtmc_observations " << made.slot << " offered "
                          << made.tracker->observations_offered() << "\n";
                std::cout << "mtmc_observations " << made.slot << " admitted "
                          << made.tracker->observations_admitted() << "\n";
                const mtmc::IdentitySizes sizes = made.tracker->sizes();
                std::cout << "mtmc_identities " << made.slot << " " << sizes.identities << " "
                          << sizes.tracks << "\n";
            }
        };

        const size_t abandoned =
            manager.stop(std::chrono::milliseconds(options.stop_deadline_ms));
        if (abandoned != 0) {
            // An abandoned actor's detached thread still holds references into this frame —
            // the sink and the queue above all. Unwinding the stack now would free them under
            // a live thread, so take the manager's own containment to its conclusion: report,
            // and leave without running destructors. Unreachable with `replay` (a replay read
            // cannot block), so this is armour for the sources PR2 adds.
            std::cerr << "bench: " << abandoned
                      << " camera(s) abandoned past the stop deadline; reporting the ingest "
                         "and queue counters (the reassembly half needs a drain that would "
                         "block on those very threads) and exiting without unwinding so "
                         "their threads keep valid references\n";
            report_ingest_and_queue();
            // doc: long why this exit flushes by hand, and how the first draft failed
            // FLUSHED EXPLICITLY, because `_Exit` does not. It skips atexit and every stdio
            // buffer, and a run's stdout is redirected to a log -- so fully buffered. The
            // first version of this printed the whole report into a buffer that was then
            // discarded: the run still said nothing, and only `std::cerr` (unbuffered) came
            // through. Found by forcing the path with `--stop-deadline-ms 1`, not by reading.
            std::cout << "\n" << std::flush;
            std::_Exit(1);
        }
        stopping.store(true);
        lanes.close();
        for (auto& worker : workers) worker.join();
        // After the workers: a model stopped while a worker still had a frame in hand failed
        // that frame's embedder request as 'instance stopped' and sealed it Incomplete at
        // shutdown.
        for (auto& [name, model] : models) model->stop();
        sweeper.join();
        collector.drain();
        sampler.stop();

        report_ingest_and_queue();
        std::cout << "collector_reported " << collector.reported() << "\n";
        std::cout << "collector_timeouts " << collector.timed_out() << "\n";
        std::cout << "collector_evicted " << collector.evicted() << "\n";
        for (const auto& [camera, count] : collector.evicted_by_camera()) {
            std::cout << "collector_evicted_camera " << camera << " " << count << "\n";
        }
        std::cout << "events_complete " << complete.load() << "\n";
        std::cout << "events_incomplete " << (emitted.load() - complete.load()) << "\n";
        // ONE LINE PER STAGE that failed to answer, and none at all on a run where every
        // frame was complete. `events_incomplete` says how many and this says what: a stage
        // whose count is the incomplete total never ran for those frames, while several small
        // counts are frames that lost different stages.
        {
            // UNDER THE LOCK, for the same reason the refused-by-camera map is read under
            // its own: this read happens to sit after the workers are joined today, and
            // positional safety does not survive a hoist -- the abandoned exit reports with
            // every worker still sealing frames. (Naming that map here would also trip the
            // line-window test that pins its lock, which is fair: a mention is a use to a
            // reader too.)
            std::lock_guard<std::mutex> lock(missing_lock);
            for (const auto& [stage, count] : missing_by_stage) {
                std::cout << "events_missing_stage " << stage << " " << count << "\n";
            }
        }
        // TWO WINDOWS, and the names say which is which. `reassembly_us` starts when the
        // collector opens the frame, so it is the part this plane controls; `frame_us` starts
        // at capture, so it is the one a deployment is judged on.
        report_window("reassembly_us", latency_lock, latency_us);
        report_window("frame_us", latency_lock, frame_us);
        // IN THE OUTPUT, not only in a comment: `captured_ns` is stamped on every source path
        // there is, so a frame window with fewer samples than the reassembly one means some
        // source stopped stamping -- and a reader comparing the two figures has to know that
        // the second describes a subset.
        {
            const std::lock_guard<std::mutex> held(latency_lock);
            if (frame_us.size() < latency_us.size()) {
                std::cout << "frame_us_unstamped " << (latency_us.size() - frame_us.size())
                          << "\n";
            }
        }
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
        //
        // ROWS AS WELL AS REQUESTS, and the pair is the point: a request is one stage
        // INVOCATION and a row is one image into the model, so the detector's two are equal
        // while an embedder's differ by the crop fan-out. Reporting only requests understates
        // this plane's work against a one-model-per-image baseline by whatever that fan-out is
        // -- `C1-WHAT-IS-THE-5x-AGAINST?` could put no number on its own like-for-like
        // candidate because of it. `ModelInstance` has summed `rows` all along
        // (`engine/instance.cpp`); nothing printed it.
        for (const auto& [name, model] : models) {
            std::map<int, uint64_t> by_device;
            std::map<int, uint64_t> rows_by_device;
            std::map<int, double> compute_by_device;
            for (const auto& instance : model->instances()) {
                by_device[instance->device().index] += instance->stats().requests;
                rows_by_device[instance->device().index] += instance->stats().rows;
                compute_by_device[instance->device().index] += instance->stats().compute_us;
            }
            std::cout << "per_device " << name;
            for (const auto& [device, count] : by_device)
                std::cout << " " << device << ":" << count;
            std::cout << "\n";
            std::cout << "per_device_rows " << name;
            for (const auto& [device, count] : rows_by_device)
                std::cout << " " << device << ":" << count;
            std::cout << "\n";
            // doc: long why this is a percentage and not the microseconds it is made of
            // OCCUPANCY, not raw time. A microsecond total is unreadable without the wall
            // clock beside it, and a reader who has to divide will divide by the wrong thing
            // -- the run is `--seconds` long but an instance only exists for the serving
            // window. `compute_us / (seconds * 1e6)` per instance, summed per device and
            // printed as a percentage: 100% means that device's instances of this model were
            // executing the whole run, so the model is the bottleneck rather than merely the
            // busiest-looking. Above 100% is legitimate and means several instances per
            // device, which is what `instances_per_device` says it configured.
            std::cout << "per_device_busy_pct " << name;
            for (const auto& [device, micros] : compute_by_device) {
                const double pct =
                    options.seconds > 0.0 ? micros / (options.seconds * 1e6) * 100.0 : 0.0;
                // Formatted aside rather than on `std::cout`: `std::defaultfloat` restores
                // the float FORMAT and not the precision, so `setprecision(1)` would leak
                // onto the stream and cut the next double printed anywhere to one digit.
                std::ostringstream cell;
                cell << std::fixed << std::setprecision(1) << pct;
                std::cout << " " << device << ":" << cell.str();
            }
            std::cout << "\n";
        }
        std::cout << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "shipinfer_pipeline: " << error.what() << "\n";
        return 1;
    }
}
