// How fast this plane can write the events it emits, and how much it allocates doing it.
//
// P5-A made event building unskippable in `cli/bench.cpp`, and P5-A-ALLOC is the follow-up:
// every scalar on the emission path was a `std::string` returned by value, so at the design
// load -- 1000 frames/s, 10-20 objects each, a 2048-float embedding per object -- the
// serialiser did millions of small allocations a second. This is the measurement that says
// whether removing them helped, because "it must be faster" is not a number.
//
// Under `cli/` and not `tests/`: CI's offline job runs every `csrc/build/test_*` and a
// benchmark does not belong in a test loop. CUDA-free, so it builds in the offline lane and
// runs anywhere -- what is measured is `to_json`, which touches no device.
//
//   csrc/build/bench_events                    # the default shape, 2000 events
//   csrc/build/bench_events --events 5000 --objects 20 --embedding 2048

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "shipinfer/core/events/schema.h"

namespace {

    using namespace shipinfer;

    struct Options {
        int events = 2000;
        int objects = 15;      // CLAUDE.md's sizing: 10-20 per frame
        int embedding = 2048;  // `model_repository/ship_embedder/config.yaml`
    };

    Options parse(int argc, char** argv) {
        Options options;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (i + 1 >= argc) {
                std::printf("missing value for %s\n", flag.c_str());
                std::exit(2);
            }
            const int value = std::atoi(argv[++i]);
            if (flag == "--events")
                options.events = value;
            else if (flag == "--objects")
                options.objects = value;
            else if (flag == "--embedding")
                options.embedding = value;
            else {
                std::printf("unknown flag %s\n", flag.c_str());
                std::exit(2);
            }
        }
        return options;
    }

    // One frame's worth of records, half people and half ships, with the optional fields
    // filled the way a complete chain fills them -- an event where every `optional` is null
    // would measure the cheap path.
    events::PerceptionEvent one_event(const Options& options) {
        std::vector<events::ObjectRecord> objects;
        for (int i = 0; i < options.objects; ++i) {
            events::ObjectRecord record;
            record.det_id = "cam-01_" + std::to_string(i);
            record.class_name = (i % 2 == 0) ? "person" : "ship";
            record.score = 0.25 + 0.001 * i;
            const double base[4] = {12.5 + i, 20.25 + i, 300.75 + i, 480.0 + i};
            for (int k = 0; k < 4; ++k) record.bbox[k] = base[k];
            record.embedding.reserve(static_cast<size_t>(options.embedding));
            for (int k = 0; k < options.embedding; ++k) {
                // Spread across the exponent range Python's `repr` switches on, so the
                // measurement covers both branches rather than only the fixed one.
                record.embedding.push_back(0.1234567 * (k + 1) * (k % 7 == 0 ? 1e-6 : 1.0));
            }
            record.track_id = 1000 + i;
            record.track_state = "tracked";
            record.global_id = 7000 + i;
            if (record.class_name == "ship") {
                record.ship_id = 42 + i;
                record.similarity = 0.87654321;
                record.mask_area_px = 12345.678;
            }
            objects.push_back(std::move(record));
        }
        return events::build("cam-01", 4242, "quay-a", std::move(objects), 1920, 1080, 20.0, 1,
                             2, {}, "complete", 3, 4);
    }

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse(argc, argv);
    const events::PerceptionEvent event = one_event(options);

    // One serialisation first, so the reported size is the real one and the loop below is
    // measuring a warm allocator rather than the first-touch page faults.
    const size_t bytes_each = event.to_json().size();

    // The checksum makes a change that alters the BYTES visible here too, not only in the
    // parity gate -- a faster writer that emits different JSON is not an optimisation.
    uint64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < options.events; ++i) {
        const std::string line = event.to_json();
        for (const char byte : line)
            checksum = checksum * 131 + static_cast<unsigned char>(byte);
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    const double per_event_us = seconds * 1e6 / options.events;
    const double numbers = static_cast<double>(options.objects) * options.embedding;
    std::printf("events        %d x %d objects x %d floats\n", options.events, options.objects,
                options.embedding);
    std::printf("bytes/event   %zu\n", bytes_each);
    std::printf("per event     %.1f us\n", per_event_us);
    std::printf("throughput    %.0f events/s, %.1f MB/s, %.1f M numbers/s\n",
                options.events / seconds,
                options.events * static_cast<double>(bytes_each) / seconds / 1e6,
                options.events * numbers / seconds / 1e6);
    std::printf("checksum      %llu\n", static_cast<unsigned long long>(checksum));
    return 0;
}
