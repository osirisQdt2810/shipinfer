#include "shipinfer/core/events/schema.h"

#include <chrono>
#include <cmath>

#include "shipinfer/core/events/json.h"

namespace shipinfer::events {

    namespace {

        // A JSON array, one element per object of one class. Every `*_vec` key in the payload
        // is this shape, which is why it is a template rather than eleven near-identical loops.
        template <typename Fn>
        std::string array_of(const std::vector<const ObjectRecord*>& objects, Fn&& element) {
            std::string out = "[";
            for (size_t i = 0; i < objects.size(); ++i) {
                if (i) out += ',';
                out += element(*objects[i]);
            }
            return out + "]";
        }

        std::string optional_int(const std::optional<int64_t>& value) {
            // `null`, not a sentinel: the Python field is `int | None` and 0 is a legitimate
            // gallery id, a legitimate track id and a legitimate global id.
            return value ? std::to_string(*value) : "null";
        }

        std::string optional_double(const std::optional<double>& value) {
            return value ? json_number(*value) : "null";
        }

        std::string optional_word(const std::optional<std::string>& value) {
            return value ? json_string(*value) : "null";
        }

        std::string bbox_of(const ObjectRecord& object) {
            std::string out = "[";
            for (int i = 0; i < 4; ++i) {
                if (i) out += ',';
                out += json_number(object.bbox[i]);
            }
            return out + "]";
        }

        std::string embedding_of(const ObjectRecord& object) {
            std::string out = "[";
            for (size_t i = 0; i < object.embedding.size(); ++i) {
                if (i) out += ',';
                out += json_number(object.embedding[i]);
            }
            return out + "]";
        }

        std::string words(const std::vector<std::string>& values) {
            std::string out = "[";
            for (size_t i = 0; i < values.size(); ++i) {
                if (i) out += ',';
                out += json_string(values[i]);
            }
            return out + "]";
        }

    }  // namespace

    std::vector<const ObjectRecord*> PerceptionEvent::objects_of(
        const std::string& class_name) const {
        std::vector<const ObjectRecord*> found;
        for (const ObjectRecord& object : objects) {
            if (object.class_name == class_name) found.push_back(&object);
        }
        return found;
    }

    // doc: long the key ORDER is the contract, so it is written out rather than assembled
    std::string PerceptionEvent::to_json() const {
        // In `as_det2mot()`'s order first and then `as_dict()`'s additions, because a Python
        // dict preserves insertion order and `update()` appends keys it does not already
        // hold -- so this sequence IS the byte sequence the other plane produces. Assembling
        // it from a map would sort the keys and the gate would fail on every event.
        const std::vector<const ObjectRecord*> people = objects_of("person");
        const std::vector<const ObjectRecord*> ships = objects_of("ship");
        const auto det_id = [](const ObjectRecord& o) { return json_string(o.det_id); };
        const auto score = [](const ObjectRecord& o) { return json_number(o.score); };

        std::string out = "{";
        out += "\"sub_id\":" + json_string(source_id);
        out += ",\"det_id_vec\":" + array_of(people, det_id);
        out += ",\"camera_id\":" + json_string(camera_id);
        out += ",\"image_id\":" + std::to_string(frame_id);
        out += ",\"det_body_score_vec\":" + array_of(people, score);
        out += ",\"body_bbox_vec\":" + array_of(people, bbox_of);
        out += ",\"body_feature_vec\":" + array_of(people, embedding_of);
        out += ",\"img_width\":" + std::to_string(img_width);
        out += ",\"img_height\":" + std::to_string(img_height);
        out += ",\"img_fps\":" + std::to_string(img_fps);
        out += ",\"type\":" + json_string(kMessageType);
        out += ",\"schema_version\":" + std::to_string(kSchemaVersion);
        out += ",\"ship_det_id_vec\":" + array_of(ships, det_id);
        out += ",\"det_ship_score_vec\":" + array_of(ships, score);
        out += ",\"ship_bbox_vec\":" + array_of(ships, bbox_of);
        out += ",\"ship_feature_vec\":" + array_of(ships, embedding_of);
        out += ",\"ship_id_vec\":" +
               array_of(ships, [](const ObjectRecord& o) { return optional_int(o.ship_id); });
        out += ",\"ship_similarity_vec\":" + array_of(ships, [](const ObjectRecord& o) {
                   return optional_double(o.similarity);
               });
        out += ",\"ship_mask_area_vec\":" + array_of(ships, [](const ObjectRecord& o) {
                   return optional_double(o.mask_area_px);
               });
        const auto track_id = [](const ObjectRecord& o) { return optional_int(o.track_id); };
        const auto track_state = [](const ObjectRecord& o) {
            return optional_word(o.track_state);
        };
        const auto global_id = [](const ObjectRecord& o) { return optional_int(o.global_id); };
        out += ",\"body_track_id_vec\":" + array_of(people, track_id);
        out += ",\"body_track_state_vec\":" + array_of(people, track_state);
        out += ",\"ship_track_id_vec\":" + array_of(ships, track_id);
        out += ",\"ship_track_state_vec\":" + array_of(ships, track_state);
        out += ",\"body_global_id_vec\":" + array_of(people, global_id);
        out += ",\"ship_global_id_vec\":" + array_of(ships, global_id);
        out += ",\"captured_unix_ns\":" + std::to_string(captured_unix_ns);
        out += ",\"emitted_unix_ns\":" + std::to_string(emitted_unix_ns);
        out += ",\"latency_us\":" + std::to_string(latency_us);
        out += ",\"partial\":" + std::string(is_partial() ? "true" : "false");
        out += ",\"missing_stages\":" + words(missing_stages);
        out += ",\"reason\":" + json_string(reason);
        if (!extra.empty()) {
            out += ",\"extra\":{";
            bool first = true;
            for (const auto& [key, value] : extra) {
                if (!first) out += ',';
                first = false;
                out += json_string(key) + ":" + json_string(value);
            }
            out += "}";
        }
        return out + "}";
    }

    PerceptionEvent build(const std::string& camera_id, int64_t frame_id,
                          const std::string& source_id, std::vector<ObjectRecord> objects,
                          int64_t width, int64_t height, double fps, int64_t captured_ns,
                          int64_t captured_unix_ns,
                          const std::vector<std::string>& missing_stages,
                          const std::string& reason, int64_t now_ns, int64_t now_unix_ns) {
        using namespace std::chrono;
        PerceptionEvent event;
        event.camera_id = camera_id;
        event.frame_id = frame_id;
        event.source_id = source_id;
        event.objects = std::move(objects);
        event.img_width = width;
        event.img_height = height;
        // `llrint` and NOT `llround`, matching `img_fps=round(fps)`: `llround` is half AWAY
        // FROM ZERO while Python's `round` is half TO EVEN, so they part company at exactly
        // the rates a deployment configures -- 12.5 is 13 there and 12 here. A fleet split
        // across Python and C++ shards would publish two `img_fps` for one camera, in the
        // one field this seam added a byte gate to protect. `llrint` uses the current
        // rounding mode, which is `FE_TONEAREST` (half to even) and never changed here.
        event.img_fps = static_cast<int64_t>(std::llrint(fps));
        event.captured_unix_ns = captured_unix_ns;
        // Injectable, and defaulted rather than required, because the parity gate has to
        // produce the same bytes twice and a wall clock cannot. Production passes nothing.
        event.emitted_unix_ns =
            now_unix_ns
                ? now_unix_ns
                : duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
        const int64_t monotonic =
            now_ns ? now_ns
                   : duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
        event.latency_us =
            captured_ns ? std::max<int64_t>(0, (monotonic - captured_ns) / 1000) : 0;
        event.missing_stages = missing_stages;
        event.reason = reason;
        return event;
    }

}  // namespace shipinfer::events
