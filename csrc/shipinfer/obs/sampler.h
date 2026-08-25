// The buffer-occupancy log — deliberately byte-compatible with the other two systems.
//
// `benchmarks/harness/analysis.py` fits a line through this and decides SATURATED / SUSTAINED /
// DRAINING / UNMEASURED. Writing the same shape means the C++ path, the Python path and the
// baseline binary are all judged by **one** implementation with one set of guards — there is no
// way for this port to be scored by a friendlier judge than the thing it is being compared
// against, which given that this port exists to look good is the property that matters.
//
// Line 1 is a `{"meta": {...}}` object; every later line is `{"t": <seconds>,
// "<name>_buffer_size": <depth>, ...}` — `_buffer_size`, the key `analysis.read_log` strips.
#pragma once

#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <thread>

namespace shipinfer {

    class OccupancySampler {
      public:
        using Probe = std::function<std::map<std::string, long long>()>;

        OccupancySampler(const std::string& path, Probe probe, double interval_s,
                         const std::string& meta_json);
        ~OccupancySampler();

        void start();
        void stop();

      private:
        void run();
        void run_loop(std::chrono::duration<double> period,
                      std::chrono::steady_clock::time_point next);

        std::string path_;
        Probe probe_;
        double interval_s_;
        std::string meta_json_;
        std::ofstream out_;
        std::thread thread_;
        std::atomic<bool> stopping_{false};
        std::chrono::steady_clock::time_point started_;
    };

}  // namespace shipinfer
