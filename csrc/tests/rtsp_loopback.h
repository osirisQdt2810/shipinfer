// A real RTSP server on 127.0.0.1, for the one check that has to see a pixel.
//
// WHY THIS EXISTS
// ---------------
// `build_pipeline` builds an `rtspsrc` pipeline by construction — that is the whole reason its
// output is a pure, assertable string — so nothing short of a real DESCRIBE/SETUP/PLAY session
// on a real socket makes `GStreamerSource` decode anything. Until this file, no check in the
// C++ plane had ever seen a frame come out of GStreamer, and #46's test section said so in
// place of pretending otherwise. This is the missing half: `test_ingest.cpp` section P opens
// the URI this serves, reads frames, and asserts on the bytes.
//
// WHY A CHILD PROCESS RUNNING `scripts/rtsp_serve.py`
// --------------------------------------------------
// **Because there is nothing to compile against.** `shipinfer-gst:jammy` has
// `libgstrtspserver-1.0.so.0` and the `gir1.2-gst-rtsp-server-1.0` binding, but no
// `libgstrtspserver-1.0-dev`: no headers, no `.pc` file, no `.so` symlink — measured, not
// assumed (`pkg-config --modversion gstreamer-rtsp-server-1.0` in the image: "not found"). So a
// C++ RTSP server in this tree could not be built in the one image that can run these tests,
// and extending the image is a different PR's decision.
//
// **And because the server already exists.** `scripts/rtsp_serve.py` is the RTSP fixture
// `benchmarks/harness/rtsp.py` stands up and `tests/test_rtsp_serve.py` pins (its two real
// callers — #48 round 1 trimmed a wider claim), with its pacing
// (`identity single-segment=true sync=true`) and its looping (`multifilesrc loop=true`) already
// argued out in its docstring — including the two bugs that produced a 170%-of-target
// measurement. A second server written here would be a second set of those bugs. `ffmpeg -f
// rtsp` is not an option and that is also settled: in 4.4 `rtsp_flags listen` is a *demuxer*
// option, so ffmpeg can ANNOUNCE to a server and never be one (re-confirmed in the image while
// writing this).
//
// The process boundary is not a compromise either — it is what `benchmarks/harness/rtsp.py`
// chose, for the reason that a GLib main loop owns the thread it runs on for the life of the
// process, and because frames crossing a real socket is the thing being tested.
//
// WHY THE JPEGs ARE MADE HERE
// ---------------------------
// `rtsp_serve.py` serves a directory of JPEGs (real 1920x1080 frames, in a benchmark run) and
// this repository ships none: `benchmarks/baseline/data/` is not in git. So the fixture is made
// on the spot from `ffmpeg`'s `testsrc` — ten *different* small frames, which is what lets a
// check assert that consecutive decoded frames differ instead of only that one frame is not a
// constant. Small on purpose: the point is a decoded pixel, not a throughput number, and a
// 320x240 fixture keeps the offline suite's runtime in the same order as it was.
//
// WHY THIS IS A HEADER UNDER `csrc/tests/`
// ---------------------------------------
// `scripts/build_csrc.py` globs `csrc/tests/*.cpp` as *entry points* — a `.cpp` here would be
// linked as its own binary and fail for want of a `main`. Header-only also means this needs no
// `EXTERNAL` lane, no link-line change, and no entry in the closure walker (which follows only
// `#include "shipinfer/..."`): it reaches nothing but POSIX and `std::filesystem`, so it
// compiles in the offline tier on a machine with nothing installed and decides at *runtime*
// whether this host can serve RTSP. A host that cannot — no `ffmpeg`, a `python3` without
// PyGObject — gets a counted skip carrying the server's own log, never a failure.
#pragma once

#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace shipinfer::testsupport {

    namespace detail {

        // The tail of a file, or "" — what a skip message quotes when the server would not
        // start. The reason a camera cannot connect is in the server's log, and a skip that
        // does not carry it is a skip somebody has to reproduce by hand.
        inline std::string tail_of(const std::string& path, size_t limit = 600) {
            std::ifstream file(path);
            if (!file) return "(no log)";
            std::ostringstream all;
            all << file.rdbuf();
            std::string text = all.str();
            if (text.size() > limit) text = "..." + text.substr(text.size() - limit);
            return text;
        }

        // Start `argv`, with both its streams appended to `log`. The pid, or -1.
        //
        // **A file, not a pipe.** Nobody drains a pipe for the length of a test, so at 64 KiB
        // the server blocks in `write()` and every camera stalls — which a test would report as
        // "no frames arrived", nowhere near the cause. `benchmarks/harness/rtsp.py` documents
        // the same decision for the same reason, one `GST_DEBUG` setting away from happening.
        inline pid_t spawn(const std::vector<std::string>& argv, const std::string& log) {
            // Built before the fork: after it, this process has one thread and may only call
            // async-signal-safe functions, which `new` is not.
            std::vector<char*> raw;
            raw.reserve(argv.size() + 1);
            for (const std::string& argument : argv) {
                raw.push_back(const_cast<char*>(argument.c_str()));
            }
            raw.push_back(nullptr);

            const pid_t pid = ::fork();
            if (pid != 0) return pid;  // the parent, or -1

            const int sink = ::open(log.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0600);
            if (sink >= 0) {
                ::dup2(sink, STDOUT_FILENO);
                ::dup2(sink, STDERR_FILENO);
                if (sink > STDERR_FILENO) ::close(sink);
            }
            const int empty = ::open("/dev/null", O_RDONLY);
            if (empty >= 0) {
                ::dup2(empty, STDIN_FILENO);
                if (empty > STDERR_FILENO) ::close(empty);
            }
            // A test that aborts, or a `check` that returns early past a `stop()`, must not
            // leave an RTSP server holding a port: the *kernel* kills this child when its
            // parent dies, whatever the parent's last instruction turned out to be.
            ::prctl(PR_SET_PDEATHSIG, SIGKILL);
            ::execvp(raw[0], raw.data());
            // 127 is the shell's "command not found", and it is the usual outcome here: no
            // ffmpeg, or no python3, on a host that was never going to serve RTSP.
            ::_exit(127);
        }

        // Run `argv` to completion; its exit status, or -1 if it could not be run at all.
        inline int run_to_completion(const std::vector<std::string>& argv,
                                     const std::string& log) {
            const pid_t pid = spawn(argv, log);
            if (pid < 0) return -1;
            int status = 0;
            if (::waitpid(pid, &status, 0) != pid) return -1;
            return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        }

        // Is somebody listening on 127.0.0.1:`port`?
        //
        // The readiness question asked directly. A fixed sleep is either too short — the source
        // fails its first connect, backs off, and the test measures the backoff — or dead time
        // in every run forever; `benchmarks/harness/rtsp.py` polls for the same reason. No
        // timeout is needed: a refused connection on the loopback returns immediately.
        inline bool accepting(int port) {
            const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) return false;
            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_port = htons(static_cast<uint16_t>(port));
            address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            const bool connected =
                ::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0;
            ::close(fd);
            return connected;
        }

        // A TCP port nothing is using, from the kernel rather than from a guess.
        //
        // Asking for port 0 and reading back what was bound is how a *concurrent* suite gets a
        // port at all: a hard-coded 8554 collides with a benchmark run, with a second test
        // binary, and with the last run's server if it outlived its test. The window between
        // this close and the server's bind is a race in theory; the alternative — a fixed port
        // with retries — races against other *processes*, which is the one that actually
        // happens on a shared box.
        inline int free_port() {
            const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) return -1;
            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_port = 0;
            address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            int port = -1;
            if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) {
                sockaddr_in bound{};
                socklen_t length = sizeof(bound);
                if (::getsockname(fd, reinterpret_cast<sockaddr*>(&bound), &length) == 0) {
                    port = ntohs(bound.sin_port);
                }
            }
            ::close(fd);
            return port;
        }

    }  // namespace detail

    // One RTSP camera on the loopback, for as long as this object lives.
    //
    // Not copyable and it kills its child in the destructor, so an early `return` out of a
    // failed check cannot leak a server process holding a port (`PR_SET_PDEATHSIG` covers the
    // harder case where the test does not get to run a destructor at all).
    class RtspLoopback {
      public:
        RtspLoopback() = default;
        RtspLoopback(const RtspLoopback&) = delete;
        RtspLoopback& operator=(const RtspLoopback&) = delete;
        ~RtspLoopback() { stop(); }

        // Serve one camera of `width` x `height` at `fps`, looping ten distinct frames.
        //
        // Returns "" once the port accepts connections; otherwise the reason this host cannot
        // serve, for a counted skip. **It does not throw and does not check**: "this host has
        // no PyGObject" is not a failure of the code under test, and a test binary that aborted
        // on it would stop being runnable on the driverless host the offline tier exists for.
        std::string start(int width, int height, int fps);

        // `rtsp://127.0.0.1:<port>/cam0`; empty until `start` has returned "".
        const std::string& uri() const { return uri_; }

        // Idempotent: the test calls it to assert a clean shutdown, the destructor calls it
        // again.
        void stop();

      private:
        pid_t server_ = -1;
        int port_ = 0;
        std::string uri_;
        std::string log_;
        std::filesystem::path dir_;
    };

    inline std::string RtspLoopback::start(int width, int height, int fps) {
        // 1. The server. Found from this binary's own path — `csrc/build/test_ingest` sits two
        //    directories under the repository root — so the test works from a copied tree (the
        //    container run copies `csrc/` and `scripts/` into /tmp) with nothing to configure.
        //    The env var is for whoever puts them somewhere else.
        const char* configured = std::getenv("SHIPINFER_RTSP_SERVE");
        std::string script = configured != nullptr ? configured : "";
        if (script.empty()) {
            char buffer[4096] = {};
            const ssize_t length = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
            if (length <= 0) {
                return "cannot read /proc/self/exe to find scripts/rtsp_serve.py (set "
                       "SHIPINFER_RTSP_SERVE)";
            }
            const std::string exe(buffer, static_cast<size_t>(length));
            const size_t build = exe.rfind("/csrc/build/");
            if (build == std::string::npos) {
                return "this binary is not under csrc/build/, so scripts/rtsp_serve.py cannot "
                       "be found relative to it (set SHIPINFER_RTSP_SERVE): " +
                       exe;
            }
            script = exe.substr(0, build) + "/scripts/rtsp_serve.py";
        }
        std::error_code ignored;
        if (!std::filesystem::is_regular_file(script, ignored)) {
            return "no RTSP server script at " + script + " (set SHIPINFER_RTSP_SERVE)";
        }

        // 2. A private directory for the fixture and the log, removed by `stop`.
        const std::filesystem::path root =
            std::filesystem::temp_directory_path(ignored) / "shipinfer-rtsp-XXXXXX";
        std::string pattern = root.string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        if (::mkdtemp(writable.data()) == nullptr) {
            return std::string("could not create a temporary directory for the fixture: ") +
                   std::strerror(errno);
        }
        dir_ = writable.data();
        log_ = (dir_ / "server.log").string();
        const std::filesystem::path frames = dir_ / "frames";
        std::filesystem::create_directories(frames, ignored);

        // 3. Ten distinct JPEGs. `testsrc` moves, so the frames differ from each other, which
        //    is what makes "consecutive decoded frames differ" an assertable property rather
        //    than a coincidence of a static test card.
        const std::string size = std::to_string(width) + "x" + std::to_string(height);
        const int drawn =
            detail::run_to_completion({"ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                                       "-f", "lavfi", "-i", "testsrc=size=" + size + ":rate=1",
                                       "-frames:v", "10", (frames / "frame%02d.jpg").string()},
                                      log_);
        if (drawn != 0) {
            return "ffmpeg could not write the JPEG fixture (exit " + std::to_string(drawn) +
                   "): " + detail::tail_of(log_);
        }

        // 4. The server itself, which encodes those JPEGs once and packetises them forever.
        port_ = detail::free_port();
        if (port_ <= 0) return "could not get a free TCP port on 127.0.0.1";
        server_ =
            detail::spawn({"python3", script, "--streams", "1", "--port", std::to_string(port_),
                           "--fps", std::to_string(fps), "--data", frames.string(), "--fixture",
                           (dir_ / "fixture.h264").string()},
                          log_);
        if (server_ < 0) return "could not fork the RTSP server";

        // 5. Ready when the port answers. A server that exits instead — `python3` without
        //    PyGObject is the common one — is reported with its own traceback rather than as a
        //    thirty-second wait.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (std::chrono::steady_clock::now() < deadline) {
            int status = 0;
            if (::waitpid(server_, &status, WNOHANG) == server_) {
                server_ = -1;  // reaped; nothing left to kill
                return "the RTSP server exited before it was ready (python3 without PyGObject, "
                       "or without gst-rtsp-server): " +
                       detail::tail_of(log_);
            }
            if (detail::accepting(port_)) {
                uri_ = "rtsp://127.0.0.1:" + std::to_string(port_) + "/cam0";
                return "";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return "the RTSP server did not accept a connection within 30s: " +
               detail::tail_of(log_);
    }

    inline void RtspLoopback::stop() {
        if (server_ > 0) {
            // TERM, then KILL — the shape `benchmarks/harness/rtsp.py` uses, for its reason: a
            // GLib main loop that ignored SIGTERM would hold the port and make the *next* run
            // fail with "address already in use", minutes away from the cause.
            ::kill(server_, SIGTERM);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            bool reaped = false;
            while (!reaped && std::chrono::steady_clock::now() < deadline) {
                int status = 0;
                const pid_t done = ::waitpid(server_, &status, WNOHANG);
                if (done == server_ || done < 0) {
                    reaped = true;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(25));
                }
            }
            if (!reaped) {
                ::kill(server_, SIGKILL);
                int status = 0;
                ::waitpid(server_, &status, 0);
            }
            server_ = -1;
        }
        if (!dir_.empty()) {
            std::error_code ignored;
            std::filesystem::remove_all(dir_, ignored);
            dir_.clear();
        }
        uri_.clear();
        port_ = 0;
    }

}  // namespace shipinfer::testsupport
