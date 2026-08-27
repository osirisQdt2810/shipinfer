#include "shipinfer/ingest/camera/health.h"

namespace shipinfer {

    const char* to_string(CameraState state) {
        switch (state) {
            case CameraState::Idle:
                return "idle";
            case CameraState::Connecting:
                return "connecting";
            case CameraState::Streaming:
                return "streaming";
            case CameraState::Degraded:
                return "degraded";
            case CameraState::Unhealthy:
                return "unhealthy";
            case CameraState::Exhausted:
                return "exhausted";
            case CameraState::Stopped:
                return "stopped";
        }
        return "unknown";
    }

    bool is_healthy(CameraState state) {
        // CONNECTING counts: a camera two seconds into start-up is not a fault, and a health
        // check that says otherwise pages somebody on every deploy.
        return state == CameraState::Streaming || state == CameraState::Connecting;
    }

}  // namespace shipinfer
