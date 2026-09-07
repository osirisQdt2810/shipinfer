// `ingest/camera_uris.h`: one URI per camera, and a short list refused.
//
// `cli/bench.cpp` gave every even camera `--person-frames` verbatim. Right for `replay`, where
// a folder of JPEGs is shared by design; wrong for RTSP, where fifty cameras need fifty URIs
// and pointing twenty-five of them at one stream measures a fleet that does not exist.
//
// Offline: g++ alone, no CUDA, no TensorRT.

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "shipinfer/ingest/camera_uris.h"

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

    std::string a_file(const std::string& body) {
        // `tmpnam` is deprecated for a good reason but this is a test binary with no attacker;
        // what matters is that the path is distinct per call so two cases cannot collide.
        static int counter = 0;
        const std::string path = "/tmp/shipinfer-uris-" + std::to_string(++counter) + ".txt";
        std::ofstream out(path, std::ios::trunc);
        out << body;
        return path;
    }

    bool refused(const std::string& path, size_t cameras, std::string& message) {
        try {
            (void)read_camera_uris(path, cameras);
        } catch (const std::exception& error) {
            message = error.what();
            return true;
        }
        return false;
    }

    void every_line_is_one_camera_in_order() {
        const std::string path =
            a_file("rtsp://127.0.0.1:8554/cam0\nrtsp://127.0.0.1:8555/cam0\n");
        const std::vector<std::string> uris = read_camera_uris(path, 2);

        check(uris.size() == 2, "two lines, two cameras");
        check(uris[0] == "rtsp://127.0.0.1:8554/cam0",
              "and in file order, because camera N "
              "takes line N");
        check(uris[1] == "rtsp://127.0.0.1:8555/cam0",
              "the second server's stream, which is how the person/ship split crosses");
    }

    void comments_and_blank_lines_are_not_cameras() {
        // A generated file should be able to say what it is. `#` is already the comment
        // character in the plan format, so it is the same one here.
        const std::string path = a_file(
            "# person streams\nrtsp://a/cam0\n\n  \n# ship streams\nrtsp://b/cam0  # "
            "trailing\n");
        const std::vector<std::string> uris = read_camera_uris(path, 2);

        check(uris.size() == 2, "two URIs among five lines");
        check(uris[0] == "rtsp://a/cam0" && uris[1] == "rtsp://b/cam0",
              "and the trailing comment is not part of the URI");
    }

    void a_short_list_is_refused_rather_than_reused() {
        // The defect this exists to prevent: a run that quietly points several cameras at one
        // stream reports a fleet's throughput for a stream's.
        std::string message;
        const std::string path = a_file("rtsp://a/cam0\nrtsp://a/cam1\n");

        check(refused(path, 8, message), "two URIs cannot serve eight cameras");
        check(message.find("names 2 camera(s)") != std::string::npos &&
                  message.find("wants 8") != std::string::npos,
              "and the message states both numbers: " + message);
    }

    void a_longer_list_is_fine_so_one_file_serves_a_smoke_run() {
        const std::string path = a_file("rtsp://a/cam0\nrtsp://a/cam1\nrtsp://a/cam2\n");
        const std::vector<std::string> uris = read_camera_uris(path, 2);

        check(uris.size() == 3, "the extra line is kept; the caller indexes what it needs");
    }

    void a_missing_file_is_refused_by_name() {
        std::string message;

        check(refused("/tmp/shipinfer-uris-nonexistent", 1, message), "no file, no run");
        check(message.find("/tmp/shipinfer-uris-nonexistent") != std::string::npos,
              "and the path is in the message: " + message);
    }

    void an_empty_file_is_a_short_list_and_not_an_empty_fleet() {
        std::string message;
        const std::string path = a_file("# nothing but a comment\n");

        check(refused(path, 1, message), "a file of comments names no camera");
    }

    void zero_cameras_accepts_an_empty_list() {
        // Not a case anybody runs, but the boundary decides whether `read_camera_uris` can be
        // called unconditionally. It can.
        const std::string path = a_file("");

        check(read_camera_uris(path, 0).empty(), "zero wanted, zero required");
    }

}  // namespace

int main() {
    every_line_is_one_camera_in_order();
    comments_and_blank_lines_are_not_cameras();
    a_short_list_is_refused_rather_than_reused();
    a_longer_list_is_fine_so_one_file_serves_a_smoke_run();
    a_missing_file_is_refused_by_name();
    an_empty_file_is_a_short_list_and_not_an_empty_fleet();
    zero_cameras_accepts_an_empty_list();
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
