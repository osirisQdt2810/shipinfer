#include "shipinfer/pipeline/graph/emission.h"

namespace shipinfer {

    const char* to_string(FinishReason reason) {
        switch (reason) {
            case FinishReason::Complete:
                return "complete";
            case FinishReason::Incomplete:
                return "incomplete";
            case FinishReason::Timeout:
                return "timeout";
            case FinishReason::Shutdown:
                return "shutdown";
            case FinishReason::Evicted:
                return "evicted";
        }
        return "unknown";
    }

}  // namespace shipinfer
