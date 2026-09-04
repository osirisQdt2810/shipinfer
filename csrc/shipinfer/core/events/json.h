// Writing what `json.dumps` writes — the same bytes, for the values an event carries.
//
// A perception event is a wire format read by a deployed `motservice`, and the parity gate
// compares this plane's line against a golden the Python plane emitted. That comparison is a
// string compare (this plane writes JSON and never parses it -- vendoring a parser for one
// format is refused by the ponytail principle), so "the same JSON" means the same bytes.
#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <system_error>

#include "shipinfer/core/types.h"

namespace shipinfer::events {

    namespace detail {

        inline std::string to_chars_or_throw(double value, std::chars_format format) {
            char buffer[64];
            const std::to_chars_result done =
                std::to_chars(buffer, buffer + sizeof(buffer), value, format);
            if (done.ec != std::errc()) {
                throw ConfigError("a double this event carries cannot be written as JSON");
            }
            return std::string(buffer, done.ptr);
        }

    }  // namespace detail

    // doc: long the exponent rule is Python's, not `to_chars`'s, and the two differ
    inline std::string json_number(double value) {
        // NOT a bare `inf`/`nan` token: neither is valid JSON, so one NaN score out of an
        // fp16 engine would make a strict consumer reject the WHOLE line rather than one
        // field. Python writes `Infinity`/`NaN`, which is not valid JSON either -- so rather
        // than choose which invalid line to emit, refuse. The caller has a real bug.
        if (!std::isfinite(value)) {
            throw ConfigError(
                "an event carries a non-finite number, which has no valid JSON "
                "spelling; a NaN score or a NaN mask area is a bug upstream");
        }
        // `std::to_chars` picks scientific whenever it is SHORTER, so it writes `1e+05` for
        // 100000.0 where Python writes `100000.0`. Python's `repr` goes scientific only when
        // the decimal exponent is < -4 or >= 16. That rule is applied here rather than
        // trusted to the library, and the gate pins both sides of both boundaries.
        const std::string scientific =
            detail::to_chars_or_throw(value, std::chars_format::scientific);
        const int exponent = std::stoi(scientific.substr(scientific.find('e') + 1));
        if (exponent < -4 || exponent >= 16) {
            // `to_chars` already writes a two-digit exponent, as Python does.
            return scientific;
        }
        std::string fixed = detail::to_chars_or_throw(value, std::chars_format::fixed);
        // `repr(1.0)` is `1.0` and `to_chars` writes `1`. A float is a float on this wire.
        if (fixed.find('.') == std::string::npos) fixed += ".0";
        return fixed;
    }

    // doc: long why this escapes rather than refuses, and what it has to match exactly
    inline std::string json_string(std::string_view value) {
        // `json.dumps` has `ensure_ascii=True` by default, so it escapes every non-ASCII code
        // point as a `\uXXXX` -- and this must do the same rather than refuse, because a
        // camera id in Vietnamese is an ordinary thing for this deployment to configure. An
        // earlier version threw here, on a worker thread, from a sink that had never been
        // able to throw before: refusing at runtime to protect a gate is the wrong trade.
        std::string out = "\"";
        for (size_t i = 0; i < value.size();) {
            const unsigned char byte = static_cast<unsigned char>(value[i]);
            if (byte == '"' || byte == '\\') {
                out += '\\';
                out += static_cast<char>(byte);
                ++i;
                continue;
            }
            if (byte >= 0x20 && byte < 0x7F) {
                out += static_cast<char>(byte);
                ++i;
                continue;
            }
            // The five short escapes Python spells by name; everything else in the control
            // range is a `\u00XX`, because `json.dumps` writes one for it.
            static const char* const kShort[] = {"\\b", "\\t", "\\n", nullptr, "\\f", "\\r"};
            if (byte >= 0x08 && byte <= 0x0D && kShort[byte - 0x08] != nullptr) {
                out += kShort[byte - 0x08];
                ++i;
                continue;
            }
            uint32_t code = byte;
            size_t width = 1;
            if (byte >= 0xF0) {
                code = byte & 0x07u;
                width = 4;
            } else if (byte >= 0xE0) {
                code = byte & 0x0Fu;
                width = 3;
            } else if (byte >= 0xC0) {
                code = byte & 0x1Fu;
                width = 2;
            }
            if (i + width > value.size()) {
                throw ConfigError(
                    "event string is truncated utf-8, which has no JSON "
                    "spelling either plane agrees on");
            }
            for (size_t k = 1; k < width; ++k) {
                code = (code << 6) | (static_cast<unsigned char>(value[i + k]) & 0x3Fu);
            }
            char escape[8];
            if (code >= 0x10000u) {
                // An astral code point is a surrogate PAIR, which is how `json.dumps` writes
                // it: an emoji in a camera id has to survive this too.
                const uint32_t rest = code - 0x10000u;
                std::snprintf(escape, sizeof(escape), "\\u%04x", 0xD800u + (rest >> 10));
                out += escape;
                std::snprintf(escape, sizeof(escape), "\\u%04x", 0xDC00u + (rest & 0x3FFu));
                out += escape;
            } else {
                std::snprintf(escape, sizeof(escape), "\\u%04x", code);
                out += escape;
            }
            i += width;
        }
        return out + "\"";
    }

}  // namespace shipinfer::events
