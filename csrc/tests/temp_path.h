// A scratch path unique to THIS PROCESS, because a fixed one is a collision waiting for a
// second run.
//
// Four test binaries wrote fixed names -- `/tmp/shipinfer_queue_parity_probe.scn` and
// friends -- and `test_camera_uris` used a per-call counter whose own comment said the point
// was "that the path is distinct per call so two cases cannot collide". The intent was right
// and one scope short: distinct per CALL, identical across PROCESSES. Two concurrent runs of
// the same binary on a shared box -- CI beside a local run, or another session's -- then
// truncate each other's fixture and fail in ways that read like product bugs.
//
// Found by a sweep that ran 24 copies of each binary at once looking for flakes: four of the
// five it reported were this, not flakes. CI runs the binaries sequentially so it never saw
// them, which is why a hazard this simple survived. The pid is what makes that sweep a valid
// tool rather than a generator of false alarms.
#pragma once

#include <unistd.h>

#include <string>

namespace shipinfer::tests {

    // `stem` names the fixture, not the file: the pid and the extension are added here so
    // every caller is unique by construction rather than by remembering to be.
    inline std::string temp_path(const std::string& stem) {
        return "/tmp/shipinfer-" + stem + "-" + std::to_string(static_cast<long>(::getpid())) +
               ".tmp";
    }

}  // namespace shipinfer::tests
