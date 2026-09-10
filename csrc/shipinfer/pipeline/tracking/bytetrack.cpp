// shipvision's ByteTrack behind the `Associator` seam — the LANE's only unit besides the shard.
//
// Nothing here reaches `core/platform.h`, which is what lets the lane's CI job build it with
// g++ alone (`--offline --with-external shipvision`). The stage that reads a `FrameState` is on
// the other side of that line, in `graph/stages.cpp`.
#include <memory>
#include <string>
#include <vector>

#include "shipinfer/pipeline/tracking/associator.h"
#include "shipinfer/pipeline/tracking/shard.h"

namespace shipinfer::tracking {

    namespace {

        // ONE SHARD PER PROCESS, held by the associator the registry hands out and shared by
        // every worker that asks for one: a shard per Dag would be a tracker per WORKER for one
        // camera, which is the identity split `shard.h` exists to prevent.
        class ShardAssociator : public Associator {
          public:
            std::vector<int> ids(const std::string& camera_id, int64_t frame_id,
                                 const std::vector<Detection>& detections) override {
                return shard_.update(camera_id, frame_id, detections).ids;
            }

          private:
            TrackerShard shard_;
        };

        // `impl: shipvision` in the chain, which is the name the plan carries. The algorithm
        // inside it (`params: algorithm: bytetrack`) is the only one this lane has, so the plan
        // writer does not emit it and this does not read it.
        const AssociatorRegistrar kShipvision("shipvision", [] {
            static const std::shared_ptr<Associator> one = std::make_shared<ShardAssociator>();
            return one;
        });

    }  // namespace

}  // namespace shipinfer::tracking
