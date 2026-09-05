// The per-row shape rule, offline — g++ alone, no CUDA, no TensorRT.
//
// It lives in a CUDA-free header for exactly this reason. The rule used to be a CLAMP inside
// `TensorSpec` (`backends/tensorrt/engine.h`), which no gate here can include, so a defect
// that sized a device buffer from a lie had nothing checking it.

#include <cstdio>
#include <string>
#include <vector>

#include "shipinfer/backends/tensor_shape.h"

namespace {

    using namespace shipinfer;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    bool refused(const std::vector<int64_t>& dims, std::string& message) {
        try {
            require_static_row("out0", dims);
            return false;
        } catch (const std::exception& error) {
            message = error.what();
            return true;
        }
    }

    void a_fixed_row_is_its_product() {
        check(elements_per_row({3, 640, 640}) == 3 * 640 * 640, "a detector's letterbox row");
        check(elements_per_row({300, 38}) == 11400, "a YOLO-seg detection block");
        check(elements_per_row({32, 160, 160}) == 819200, "and its prototype bank");
        check(elements_per_row({}) == 1, "a scalar row is one element, not zero");
    }

    void a_row_this_plane_cannot_size_is_refused() {
        std::string message;
        check(refused({-1}, message), "a dynamic dimension");
        check(message.find("out0") != std::string::npos, "named");
        check(message.find("(-1)") != std::string::npos, "with the shape it was given");
        check(refused({32, -1, 160}, message), "and one in the middle of a real shape");
        check(message.find("(32, -1, 160)") != std::string::npos, "with the whole shape");
        check(refused({0}, message), "a zero, which allocates nothing");
        check(refused({3, 0, 640}, message), "and a zero among positives");
    }

    void the_clamp_this_replaced_would_have_sized_one_element() {
        // The defect, stated as arithmetic. `(32, -1, 160)` clamped to 1 is 32 * 1 * 160 =
        // 5120 floats a row; the engine writes 32 * h * 160 for whatever `h` the profile
        // fixes. At h = 160 that is 819 200 -- a buffer 160x too small, written by the
        // engine and read back by `gpuMemcpyAsync`.
        const std::vector<int64_t> clamped = {32, 1, 160};
        const std::vector<int64_t> real = {32, 160, 160};

        check(elements_per_row(clamped) * 160 == elements_per_row(real),
              "the clamp under-sized the buffer by the missing dimension");
        std::string message;
        check(refused({32, -1, 160}, message), "so the shape is refused instead of clamped");
    }

}  // namespace

int main() {
    try {
        a_fixed_row_is_its_product();
        a_row_this_plane_cannot_size_is_refused();
        the_clamp_this_replaced_would_have_sized_one_element();
    } catch (const std::exception& error) {
        std::printf("FAIL: uncaught: %s\n", error.what());
        ++failures;
    }
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
