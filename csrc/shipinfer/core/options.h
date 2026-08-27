// Keyword options, as the settings tree carries them, and the two things every consumer of
// them does.
//
// A registry hands a factory a `map<string, string>` — `placement_policy_options` for a
// policy, a camera's `options` for a source — and the factory has to turn strings into
// arguments and refuse the ones it does not take. Both halves lived under
// `scheduling/policies/` while the policies were the only family with options; they are here
// so the second family does not grow its own subtly different copy of them.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace shipinfer {

    using KeywordOptions = std::map<std::string, std::string>;

    // An integer option, or `fallback` when the key is absent.
    //
    // `subject` is what the refusal names — "placement policy" for a policy, "camera 'cam0':
    // gstreamer" for a video source. It became a parameter when the second family arrived, as
    // the comment here asked it to: a camera told "placement policy option max_buffers must be
    // an integer" is a camera told to go and read the wrong file.
    int option_int(const std::string& subject, const KeywordOptions& options,
                   const std::string& key, int fallback);

    // A keyword the constructor does not take is a configuration error, as the Python
    // constructors' `TypeError` is one. Silently ignoring it is how a deployment runs for
    // months with a knob that was never connected to anything.
    void refuse_unknown_options(const std::string& subject, const KeywordOptions& options,
                                const std::vector<std::string>& accepted);

}  // namespace shipinfer
