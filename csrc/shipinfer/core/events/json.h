// Writing what `json.dumps` writes — the same bytes, for the values an event carries.
//
// A perception event is a wire format read by a deployed `motservice`, and the parity gate
// compares this plane's line against a golden the Python plane emitted. That comparison is a
// string compare (this plane writes JSON and never parses it -- vendoring a parser for one
// format is refused by the ponytail principle), so "the same JSON" means the same bytes.
#pragma once

#include <algorithm>
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

        //: Every double a `repr` can produce fits: 24 characters for the shortest round trip,
        //: and the FIXED form is only ever requested for exponents in [-4, 16).
        constexpr size_t kNumberBuffer = 64;

        inline char* to_chars_or_throw(char* buffer, double value, std::chars_format format) {
            const std::to_chars_result done =
                std::to_chars(buffer, buffer + kNumberBuffer, value, format);
            if (done.ec != std::errc()) {
                throw ConfigError("a double this event carries cannot be written as JSON");
            }
            return done.ptr;
        }

        //: `\\uXXXX`, written by hand. `snprintf` was here and it is the wrong tool twice
        //: over: a format-string call per escaped code point on a path a Vietnamese camera id
        //: takes for every character, and it drew a truncation warning for a buffer that
        //: cannot overflow (a code point is at most four hex digits by construction).
        inline void append_escape(std::string& out, uint32_t code) {
            static const char kHex[] = "0123456789abcdef";
            out += "\\u";
            out += kHex[(code >> 12) & 0xF];
            out += kHex[(code >> 8) & 0xF];
            out += kHex[(code >> 4) & 0xF];
            out += kHex[code & 0xF];
        }

    }  // namespace detail

    // doc: long the exponent rule is Python's, not `to_chars`'s, and the two differ
    //
    // APPENDS rather than returning, and that is the whole of P5-A-ALLOC's first half: a
    // returned `std::string` per scalar meant three allocations per number, and at the design
    // load -- 1000 frames/s x 15 objects x a 2048-float embedding -- that is 30 M numbers a
    // second. Measured by `cli/bench_events.cpp`, before and after, in the PR that did it.
    inline void append_number(std::string& out, double value) {
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
        // the decimal exponent is < -4 or >= 16, and that rule is applied here rather than
        // trusted to the library. The gate pins both sides of both boundaries, and
        // `test_event_parity` fuzzes this against Python's own `repr`.
        //
        // FIXED is tried first, and the 64-byte buffer is the exponent test: the fixed form
        // of anything outside [-4, 16) needs more room (1e300 is 301 digits, 1e-300 is 300
        // zeros), so `value_too_large` IS "Python would write this in scientific" -- one
        // `to_chars` on the common path instead of two, which was the other half of the cost.
        char buffer[detail::kNumberBuffer];
        std::to_chars_result done = std::to_chars(buffer, buffer + detail::kNumberBuffer, value,
                                                  std::chars_format::fixed);
        if (done.ec == std::errc()) {
            char* const digits = (buffer[0] == '-') ? buffer + 1 : buffer;
            const char* const point = std::find(digits, done.ptr, '.');
            // >= 1e16: seventeen integer digits or more, which is `exponent >= 16`.
            const bool too_big = (point - digits) >= 17;
            // < 1e-4: `0.` then four zeros or more, which is `exponent < -4`.
            bool too_small = false;
            if (point != done.ptr && point - digits == 1 && digits[0] == '0') {
                size_t zeros = 0;
                for (const char* p = point + 1; p != done.ptr && *p == '0'; ++p) ++zeros;
                too_small = zeros >= 4;
            }
            if (!too_big && !too_small) {
                out.append(buffer, done.ptr);
                // `repr(1.0)` is `1.0` and `to_chars` writes `1`. A float is a float here.
                if (point == done.ptr) out += ".0";
                return;
            }
        }
        // `to_chars` already writes a two-digit exponent, as Python does.
        done = std::to_chars(buffer, buffer + detail::kNumberBuffer, value,
                             std::chars_format::scientific);
        if (done.ec != std::errc()) {
            throw ConfigError("a double this event carries cannot be written as JSON");
        }
        out.append(buffer, done.ptr);
    }

    //: The convenience form, for a caller that wants one number and not a stream of them --
    //: `pipeline/graph/plan.cpp` and the parity gate's spelling table.
    inline std::string json_number(double value) {
        std::string out;
        append_number(out, value);
        return out;
    }

    // doc: long why this escapes rather than refuses, and what it has to match exactly
    inline void append_string(std::string& out, std::string_view value) {
        // `json.dumps` has `ensure_ascii=True` by default, so it escapes every non-ASCII code
        // point as a `\uXXXX` -- and this must do the same rather than refuse, because a
        // camera id in Vietnamese is an ordinary thing for this deployment to configure. An
        // earlier version threw here, on a worker thread, from a sink that had never been
        // able to throw before: refusing at runtime to protect a gate is the wrong trade.
        out += '"';
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
            if (code >= 0x10000u) {
                // An astral code point is a surrogate PAIR, which is how `json.dumps` writes
                // it: an emoji in a camera id has to survive this too.
                const uint32_t rest = code - 0x10000u;
                detail::append_escape(out, 0xD800u + (rest >> 10));
                detail::append_escape(out, 0xDC00u + (rest & 0x3FFu));
            } else {
                detail::append_escape(out, code);
            }
            i += width;
        }
        out += '"';
    }

    inline std::string json_string(std::string_view value) {
        std::string out;
        append_string(out, value);
        return out;
    }

}  // namespace shipinfer::events
