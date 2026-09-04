#include "shipinfer/core/events/schema.h"

#include <charconv>
#include <chrono>
#include <cmath>

#include "shipinfer/core/events/json.h"

namespace shipinfer::events {

    namespace {

        // A JSON array, one element per object of one class. Every `*_vec` key in the payload
        // is this shape, which is why it is a template rather than eleven near-identical loops.
        //
        // Appends, like everything else here: a returned `std::string` per element meant one
        // allocation per number, and an embedding is 2048 of them (P5-A-ALLOC, measured by
        // `cli/bench_events.cpp`).
        template <typename Fn>
        void append_array(std::string& out, const std::vector<const ObjectRecord*>& objects,
                          Fn&& element) {
            out += '[';
            for (size_t i = 0; i < objects.size(); ++i) {
                if (i) out += ',';
                element(out, *objects[i]);
            }
            out += ']';
        }

        void append_optional_int(std::string& out, const std::optional<int64_t>& value) {
            // `null`, not a sentinel: the Python field is `int | None` and 0 is a legitimate
            // gallery id, a legitimate track id and a legitimate global id.
            if (!value) {
                out += "null";
                return;
            }
            char buffer[24];
            const std::to_chars_result done =
                std::to_chars(buffer, buffer + sizeof(buffer), *value);
            out.append(buffer, done.ptr);
        }

        void append_int(std::string& out, int64_t value) {
            char buffer[24];
            const std::to_chars_result done =
                std::to_chars(buffer, buffer + sizeof(buffer), value);
            out.append(buffer, done.ptr);
        }

        void append_optional_double(std::string& out, const std::optional<double>& value) {
            if (value) {
                append_number(out, *value);
            } else {
                out += "null";
            }
        }

        void append_optional_word(std::string& out, const std::optional<std::string>& value) {
            if (value) {
                append_string(out, *value);
            } else {
                out += "null";
            }
        }

        void append_bbox(std::string& out, const ObjectRecord& object) {
            out += '[';
            for (int i = 0; i < 4; ++i) {
                if (i) out += ',';
                append_number(out, object.bbox[i]);
            }
            out += ']';
        }

        void append_embedding(std::string& out, const ObjectRecord& object) {
            out += '[';
            for (size_t i = 0; i < object.embedding.size(); ++i) {
                if (i) out += ',';
                append_number(out, object.embedding[i]);
            }
            out += ']';
        }

        void append_words(std::string& out, const std::vector<std::string>& values) {
            out += '[';
            for (size_t i = 0; i < values.size(); ++i) {
                if (i) out += ',';
                append_string(out, values[i]);
            }
            out += ']';
        }

        // What to ask for up front. Deliberately generous rather than exact: one over-large
        // `reserve` costs a page or two of untouched address space, while an under-estimate
        // costs a reallocation and a 400 KB copy on the emission path. 14 bytes is the
        // measured mean for a `repr`-spelled double plus its comma.
        size_t estimated_bytes(const std::vector<ObjectRecord>& objects) {
            size_t total = 1024;
            for (const ObjectRecord& object : objects) {
                total += 320 + 14 * (object.embedding.size() + 4);
            }
            return total;
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
        const auto det_id = [](std::string& to, const ObjectRecord& o) {
            append_string(to, o.det_id);
        };
        const auto score = [](std::string& to, const ObjectRecord& o) {
            append_number(to, o.score);
        };
        const auto ship_id = [](std::string& to, const ObjectRecord& o) {
            append_optional_int(to, o.ship_id);
        };
        const auto similarity = [](std::string& to, const ObjectRecord& o) {
            append_optional_double(to, o.similarity);
        };
        const auto mask_area = [](std::string& to, const ObjectRecord& o) {
            append_optional_double(to, o.mask_area_px);
        };
        const auto track_id = [](std::string& to, const ObjectRecord& o) {
            append_optional_int(to, o.track_id);
        };
        const auto track_state = [](std::string& to, const ObjectRecord& o) {
            append_optional_word(to, o.track_state);
        };
        const auto global_id = [](std::string& to, const ObjectRecord& o) {
            append_optional_int(to, o.global_id);
        };

        // ONE buffer for the whole event, reserved once. Every helper above appends into it,
        // so a 2048-float embedding costs 2048 `to_chars` calls and no allocation at all --
        // where it used to cost about three per number.
        std::string out;
        out.reserve(estimated_bytes(objects));
        out += '{';
        out += "\"sub_id\":";
        append_string(out, source_id);
        out += ",\"det_id_vec\":";
        append_array(out, people, det_id);
        out += ",\"camera_id\":";
        append_string(out, camera_id);
        out += ",\"image_id\":";
        append_int(out, frame_id);
        out += ",\"det_body_score_vec\":";
        append_array(out, people, score);
        out += ",\"body_bbox_vec\":";
        append_array(out, people, append_bbox);
        out += ",\"body_feature_vec\":";
        append_array(out, people, append_embedding);
        out += ",\"img_width\":";
        append_int(out, img_width);
        out += ",\"img_height\":";
        append_int(out, img_height);
        out += ",\"img_fps\":";
        append_int(out, img_fps);
        out += ",\"type\":";
        append_string(out, kMessageType);
        out += ",\"schema_version\":";
        append_int(out, kSchemaVersion);
        out += ",\"ship_det_id_vec\":";
        append_array(out, ships, det_id);
        out += ",\"det_ship_score_vec\":";
        append_array(out, ships, score);
        out += ",\"ship_bbox_vec\":";
        append_array(out, ships, append_bbox);
        out += ",\"ship_feature_vec\":";
        append_array(out, ships, append_embedding);
        out += ",\"ship_id_vec\":";
        append_array(out, ships, ship_id);
        out += ",\"ship_similarity_vec\":";
        append_array(out, ships, similarity);
        out += ",\"ship_mask_area_vec\":";
        append_array(out, ships, mask_area);
        out += ",\"body_track_id_vec\":";
        append_array(out, people, track_id);
        out += ",\"body_track_state_vec\":";
        append_array(out, people, track_state);
        out += ",\"ship_track_id_vec\":";
        append_array(out, ships, track_id);
        out += ",\"ship_track_state_vec\":";
        append_array(out, ships, track_state);
        out += ",\"body_global_id_vec\":";
        append_array(out, people, global_id);
        out += ",\"ship_global_id_vec\":";
        append_array(out, ships, global_id);
        out += ",\"captured_unix_ns\":";
        append_int(out, captured_unix_ns);
        out += ",\"emitted_unix_ns\":";
        append_int(out, emitted_unix_ns);
        out += ",\"latency_us\":";
        append_int(out, latency_us);
        out += ",\"partial\":";
        out += is_partial() ? "true" : "false";
        out += ",\"missing_stages\":";
        append_words(out, missing_stages);
        out += ",\"reason\":";
        append_string(out, reason);
        if (!extra.empty()) {
            out += ",\"extra\":{";
            bool first = true;
            for (const auto& [key, value] : extra) {
                if (!first) out += ',';
                first = false;
                append_string(out, key);
                out += ':';
                append_string(out, value);
            }
            out += '}';
        }
        out += '}';
        return out;
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
