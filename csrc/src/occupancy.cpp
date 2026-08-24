#include "shipinfer/obs/occupancy.hpp"

#include <iomanip>
#include <sstream>

#include "shipinfer/core/types.hpp"

namespace shipinfer {

OccupancySampler::OccupancySampler(const std::string& path, Probe probe, double interval_s,
                                   const std::string& meta_json)
    : path_(path), probe_(std::move(probe)), interval_s_(interval_s), meta_json_(meta_json) {}

OccupancySampler::~OccupancySampler() { stop(); }

void OccupancySampler::start() {
    if (thread_.joinable()) return;
    out_.open(path_, std::ios::out | std::ios::trunc);
    if (!out_) throw ConfigError("cannot write the occupancy log at " + path_);
    out_ << meta_json_ << "\n";
    out_.flush();
    started_ = std::chrono::steady_clock::now();
    thread_ = std::thread([this] { run(); });
}

void OccupancySampler::stop() {
    stopping_.store(true);
    if (thread_.joinable()) thread_.join();
    if (out_.is_open()) out_.close();
}

void OccupancySampler::run() {
    const auto period = std::chrono::duration<double>(interval_s_);
    auto next = started_;
    while (!stopping_.load()) {
        const auto now = std::chrono::steady_clock::now();
        const double t = std::chrono::duration<double>(now - started_).count();

        std::ostringstream line;
        line << "{\"t\": " << std::fixed << std::setprecision(3) << t;
        for (const auto& [name, depth] : probe_()) {
            line << ", \"" << name << "\": " << depth;
        }
        line << "}";
        out_ << line.str() << "\n";
        // Flushed every sample. A run that is killed — by a timeout, by an operator, by the
        // out-of-memory killer — still has to leave an analysable log behind, and a truncated
        // last line is how a buffered writer loses the interesting part.
        out_.flush();

        next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
        const auto after = std::chrono::steady_clock::now();
        if (next > after) {
            std::this_thread::sleep_for(next - after);
        } else {
            next = after;
        }
    }
}

}  // namespace shipinfer
