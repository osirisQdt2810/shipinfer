// Placement targets: `cpu` and `cuda:N` — `core/types/device.py`.
#pragma once

#include <stdexcept>
#include <string>

namespace shipinfer {

    struct Device {
        enum class Kind { Cpu, Cuda };
        Kind kind = Kind::Cpu;
        int index = 0;

        static Device cpu() { return Device{Kind::Cpu, 0}; }
        static Device cuda(int index) {
            if (index < 0) throw std::invalid_argument("cuda device index must be >= 0");
            return Device{Kind::Cuda, index};
        }
        bool is_cuda() const { return kind == Kind::Cuda; }
        std::string str() const { return is_cuda() ? "cuda:" + std::to_string(index) : "cpu"; }

        friend bool operator==(const Device& a, const Device& b) {
            return a.kind == b.kind && a.index == b.index;
        }
        friend bool operator!=(const Device& a, const Device& b) { return !(a == b); }
        friend bool operator<(const Device& a, const Device& b) {
            return a.kind != b.kind ? a.kind < b.kind : a.index < b.index;
        }
    };

}  // namespace shipinfer
