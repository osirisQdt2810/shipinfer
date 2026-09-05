#include "shipinfer/pipeline/graph/plan.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "shipinfer/core/events/json.h"

namespace shipinfer {

    namespace {

        // Everything after the first `skip` whitespace-separated tokens, verbatim. The two
        // verbs whose last field is free text read their value this way.
        std::string rest_of(const std::string& line, size_t skip) {
            // `\r` too, not only spaces and tabs: `str.splitlines` drops it on the Python
            // side, so a CRLF plan would give `label 8 "ship\r"` here and `"ship"` there --
            // one label, two spellings, which is the drift this seam exists to prevent.
            static const char* const kSpace = " \t\r";
            size_t at = line.find_first_not_of(kSpace);
            for (size_t i = 0; i < skip; ++i) {
                at = line.find_first_of(kSpace, at);
                at = line.find_first_not_of(kSpace, at);
            }
            const std::string tail = at == std::string::npos ? "" : line.substr(at);
            const size_t last = tail.find_last_not_of(kSpace);
            return last == std::string::npos ? "" : tail.substr(0, last + 1);
        }

        std::vector<std::string> words(const std::string& line) {
            std::istringstream stream(line);
            std::vector<std::string> out;
            for (std::string word; stream >> word;) out.push_back(word);
            return out;
        }

        void want(const std::vector<std::string>& args, size_t count, const std::string& where,
                  const std::string& form) {
            if (args.size() != count) {
                throw ConfigError(where + ": expected `" + form + "`, got " +
                                  std::to_string(args.size()) + " argument(s)");
            }
        }

        // Digits and an optional sign, and nothing else -- then `stoi`. Python's `int()`
        // accepts a value too large for `int`, so an out-of-range `label` would load there
        // and refuse here; the refusal is now the same shape on both planes.
        int as_int(const std::string& text, const std::string& where) {
            const bool signed_digits =
                !text.empty() &&
                text.find_first_not_of("0123456789",
                                       text[0] == '-' || text[0] == '+' ? 1 : 0) ==
                    std::string::npos &&
                text.size() > (text[0] == '-' || text[0] == '+' ? 1u : 0u);
            if (signed_digits) {
                try {
                    return std::stoi(text);
                } catch (const std::out_of_range&) {
                    throw ConfigError(where + ": '" + text +
                                      "' does not fit in an int, and a plan carries class ids "
                                      "and pixel extents rather than arbitrary integers");
                } catch (const std::exception&) {
                }
            }
            throw ConfigError(where + ": '" + text + "' is not an integer");
        }

        // Finite, decimal, and the whole token consumed. `stod` alone is three kinds of
        // lenient where Python's `_NUMBER` regex is not: it reads `nan` and `inf` (which
        // `json_number` then refuses to write), it reads a HEX float (`0x10`), and it is
        // locale-sensitive about the decimal point. The character check comes first so all
        // three refuse identically on the two planes.
        double as_double(const std::string& text, const std::string& where) {
            const bool decimal =
                !text.empty() &&
                text.find_first_not_of("0123456789+-.eE") == std::string::npos &&
                text.find_first_of("0123456789") != std::string::npos;
            if (decimal) {
                // `strtod` and not `stod`: libstdc++ throws `out_of_range` whenever `errno`
                // is ERANGE, which `strtod` sets on UNDERFLOW as well as overflow -- so
                // `score 5e-324` was refused here while Python's `float()` returns the
                // subnormal and `_finite` passes it. A plan this plane's own emitter can
                // write must not be one it refuses to read.
                errno = 0;
                char* end = nullptr;
                const double value = std::strtod(text.c_str(), &end);
                if (end == text.c_str() + text.size() && std::isfinite(value)) return value;
            }
            throw ConfigError(where + ": '" + text + "' is not a finite decimal number");
        }

        Extent extent_of(const std::vector<std::string>& args, const std::string& where,
                         const std::string& verb) {
            want(args, 2, where, verb + " <height> <width>");
            const Extent extent{as_int(args[0], where), as_int(args[1], where)};
            if (extent.height <= 0 || extent.width <= 0) {
                throw ConfigError(where + ": " + verb + " must be two positive integers");
            }
            return extent;
        }

        // Attributes attach to the `node` block above them, the way `capacity` attaches to
        // `queue` in a queue scenario. A stray one before the first `node` is a refusal and
        // not a default, because guessing which slot it meant is how a plan runs a chain
        // nobody declared.
        void apply(PlanNode& node, const std::string& verb,
                   const std::vector<std::string>& args, const std::string& where) {
            if (verb == "model") {
                want(args, 1, where, "model <name>");
                node.model = args[0];
            } else if (verb == "classes") {
                // Comma-delimited, and a label may hold spaces: `classes cargo ship,fishing
                // vessel` is TWO labels. Space-delimited was the first spelling and it read
                // `[cargo ship]` back as two labels no detector emits -- selecting no rows,
                // running no model, reporting nothing wrong.
                if (args.empty()) {
                    throw ConfigError(where + ": expected `classes <label>[,<label>...]`");
                }
                std::string joined = args[0];
                for (size_t i = 1; i < args.size(); ++i) joined += " " + args[i];
                // `-` is a DECLARED empty selection: this slot selects no rows, which is a
                // different statement from carrying no `classes` line at all.
                if (joined == "-") {
                    node.classes = std::vector<std::string>{};
                    return;
                }
                node.classes = std::vector<std::string>{};
                size_t start = 0;
                while (start <= joined.size()) {
                    const size_t comma = joined.find(',', start);
                    const size_t stop = comma == std::string::npos ? joined.size() : comma;
                    std::string label = joined.substr(start, stop - start);
                    const size_t first = label.find_first_not_of(" \t");
                    const size_t last = label.find_last_not_of(" \t");
                    label =
                        first == std::string::npos ? "" : label.substr(first, last - first + 1);
                    if (label.empty())
                        throw ConfigError(where + ": an empty label in `classes`");
                    node.classes->push_back(label);
                    if (comma == std::string::npos) break;
                    start = comma + 1;
                }
            } else if (verb == "crop") {
                node.crop = extent_of(args, where, "crop");
            } else if (verb == "letterbox") {
                node.letterbox = extent_of(args, where, "letterbox");
            } else if (verb == "score") {
                want(args, 1, where, "score <threshold>");
                node.score_threshold = as_double(args[0], where);
            } else if (verb == "max_detections") {
                want(args, 1, where, "max_detections <count>");
                // A positive count, refused HERE as well as in `detect_config`, because the
                // Python reader refuses it and the shared table is the contract. `-1` for
                // "no limit" is no bound at all once it reaches `static_cast<size_t>`.
                const int count = as_int(args[0], where);
                if (count <= 0) {
                    throw ConfigError(where + ": max_detections is " + std::to_string(count) +
                                      "; a cap is a positive count on both planes");
                }
                node.max_detections = count;
            } else if (verb == "fold_score") {
                want(args, 1, where, "fold_score <threshold>");
                node.fold_score = as_double(args[0], where);
            } else if (verb == "fold_mask") {
                want(args, 1, where, "fold_mask <probability>");
                const double value = as_double(args[0], where);
                if (!(value > 0.0 && value < 1.0)) {
                    throw ConfigError(where + ": fold_mask is " + events::json_number(value) +
                                      "; a mask probability is strictly inside (0, 1), "
                                      "because the cut is log(m / (1 - m))");
                }
                node.fold_mask = value;
            } else if (verb == "instances") {
                want(args, 1, where, "instances <count>");
                // Positive, like `max_detections` and for the same reason: zero would load
                // the engine, run nothing, and report every stage ready.
                const int count = as_int(args[0], where);
                if (count <= 0) {
                    throw ConfigError(where + ": instances is " + std::to_string(count) +
                                      "; a slot runs at least one");
                }
                node.instances = count;
            } else if (verb == "queue_delay_us") {
                want(args, 1, where, "queue_delay_us <microseconds>");
                const int delay = as_int(args[0], where);
                if (delay < 0) {
                    throw ConfigError(where + ": queue_delay_us is " + std::to_string(delay) +
                                      "; a window is not negative");
                }
                node.queue_delay_us = delay;
            } else if (verb == "artefact") {
                want(args, 1, where, "artefact <path>");
                node.artefact = args[0];
            } else if (verb == "when") {
                if (args.empty()) throw ConfigError(where + ": expected `when <expression>`");
                std::string joined = args[0];
                for (size_t i = 1; i < args.size(); ++i) joined += " " + args[i];
                node.when = joined;
            } else if (verb == "per") {
                want(args, 1, where, "per <value>");
                node.per = args[0];
            } else if (verb == "scope") {
                want(args, 1, where, "scope <value>");
                node.scope = args[0];
            } else {
                throw ConfigError(where + ": unknown verb '" + verb +
                                  "'; expected one of artefact, classes, crop, edge, field, "
                                  "fold_mask, fold_score, instances, label, letterbox, "
                                  "max_detections, model, node, per, plan, queue_delay_us, "
                                  "score, scope, when");
            }
        }

    }  // namespace

    const PlanNode* ResolvedPlan::node(const std::string& slot) const {
        for (const PlanNode& candidate : nodes) {
            if (candidate.slot == slot) return &candidate;
        }
        return nullptr;
    }

    std::optional<int> ResolvedPlan::class_id(const std::string& label) const {
        for (const auto& [index, name] : labels) {
            if (name == label) return index;
        }
        return std::nullopt;
    }

    ResolvedPlan parse_plan(const std::string& text, const std::string& source) {
        ResolvedPlan plan;
        bool header_seen = false;
        std::istringstream stream(text);
        std::string raw;
        for (int number = 1; std::getline(stream, raw); ++number) {
            const std::string line = raw.substr(0, raw.find('#'));
            const std::vector<std::string> all = words(line);
            if (all.empty()) continue;
            const std::string verb = all[0];
            const std::vector<std::string> args(all.begin() + 1, all.end());
            const std::string where = source + ":" + std::to_string(number);
            if (verb == "plan") {
                if (header_seen) throw ConfigError(where + ": a second `plan` header");
                // The name is the REST of the line: a chain may be called `ship person
                // cpu`, and a fixed arity refused a plan `shipinfer plan` had just written.
                if (args.size() < 2) {
                    throw ConfigError(where + ": expected `plan <version> <name>`");
                }
                plan.version = as_int(args[0], where);
                if (plan.version != kPlanVersion) {
                    throw ConfigError(where + ": plan version " + args[0] +
                                      ", and this reader knows " +
                                      std::to_string(kPlanVersion) +
                                      ". A plan half understood is a chain running something "
                                      "other than what was declared");
                }
                const std::string name = rest_of(line, 2);
                plan.name = name == "-" ? "" : name;
                header_seen = true;
            } else if (!header_seen) {
                throw ConfigError(where + ": `" + verb + "` before the `plan` header");
            } else if (verb == "label") {
                // Also the rest of the line: `label 8 cargo ship` is one label, and a
                // multi-word label is the normal case in this domain (COCO's `traffic
                // light`, a ship taxonomy's `cargo ship`).
                if (args.size() < 2) {
                    throw ConfigError(where + ": expected `label <id> <name>`");
                }
                const int index = as_int(args[0], where);
                if (plan.labels.count(index) != 0) {
                    throw ConfigError(where + ": a second `label " + std::to_string(index) +
                                      "`");
                }
                plan.labels[index] = rest_of(line, 2);
            } else if (verb == "node") {
                want(args, 3, where, "node <slot> <kind> <impl>");
                if (plan.node(args[0]) != nullptr) {
                    throw ConfigError(where + ": a second `node " + args[0] + "`");
                }
                plan.nodes.push_back(PlanNode{args[0], args[1], args[2]});
            } else if (verb == "edge") {
                want(args, 3, where, "edge <producer> <consumer> <format@location>");
                plan.edges.push_back(PlanEdge{args[0], args[1], args[2]});
            } else if (verb == "field") {
                if (args.size() < 2) {
                    throw ConfigError(where + ": expected `field <name> <slot>...`");
                }
                // Refused rather than last-wins, as `node` and `label` are: this reader is
                // half of a shared contract, and the two halves agreeing on malformed input
                // is what the table in the gate exists to keep.
                if (plan.fields.count(args[0]) != 0) {
                    throw ConfigError(where + ": a second `field " + args[0] + "`");
                }
                plan.fields[args[0]] = std::vector<std::string>(args.begin() + 1, args.end());
            } else {
                if (plan.nodes.empty()) {
                    throw ConfigError(where + ": unknown verb '" + verb +
                                      "', or an attribute before any `node`");
                }
                apply(plan.nodes.back(), verb, args, where);
            }
        }
        if (!header_seen) {
            throw ConfigError(source + ": no `plan <version> <name>` header");
        }
        // A `field` may only name a slot some `node` declared -- a READING question, so both
        // readers answer it and the shared table can carry the row. `plan_stages` keeps its
        // own refusal for a plan built in code (`bench.cpp`'s `default_plan`), which never
        // passes through here.
        for (const auto& [field, slots] : plan.fields) {
            for (const std::string& slot : slots) {
                if (plan.node(slot) == nullptr) {
                    throw ConfigError(source + ": field '" + field + "' names slot '" + slot +
                                      "', which no `node` declares");
                }
            }
        }
        return plan;
    }

    ResolvedPlan read_plan(const std::string& path) {
        std::ifstream file(path);
        if (!file) throw ConfigError("cannot read plan " + path);
        std::ostringstream buffer;
        buffer << file.rdbuf();
        return parse_plan(buffer.str(), path);
    }

    std::string plan_text(const ResolvedPlan& plan) {
        std::string out =
            "# A RESOLVED chain, written by `shipinfer plan` -- do not hand-edit.\n";
        out += "plan " + std::to_string(plan.version) + " " +
               (plan.name.empty() ? "-" : plan.name) + "\n";
        for (const auto& [index, name] : plan.labels) {
            out += "label " + std::to_string(index) + " " + name + "\n";
        }
        for (const PlanNode& node : plan.nodes) {
            out += "\nnode " + node.slot + " " + node.kind + " " + node.impl + "\n";
            if (!node.model.empty()) out += "model " + node.model + "\n";
            if (node.classes) {
                out += "classes";
                if (node.classes->empty()) {
                    out += " -";
                } else {
                    for (size_t i = 0; i < node.classes->size(); ++i) {
                        out += (i ? "," : " ") + (*node.classes)[i];
                    }
                }
                out += "\n";
            }
            if (node.crop) {
                out += "crop " + std::to_string(node.crop->height) + " " +
                       std::to_string(node.crop->width) + "\n";
            }
            if (node.letterbox) {
                out += "letterbox " + std::to_string(node.letterbox->height) + " " +
                       std::to_string(node.letterbox->width) + "\n";
            }
            // `json_number` and not `to_string`: this has to be Python's `repr`, which
            // `std::to_string(0.25)` spells `0.250000` -- one byte's difference is a gate
            // that fails on a plan both planes understood perfectly.
            if (node.score_threshold) {
                out += "score " + events::json_number(*node.score_threshold) + "\n";
            }
            if (node.max_detections) {
                out += "max_detections " + std::to_string(*node.max_detections) + "\n";
            }
            if (node.instances) out += "instances " + std::to_string(*node.instances) + "\n";
            if (node.queue_delay_us) {
                out += "queue_delay_us " + std::to_string(*node.queue_delay_us) + "\n";
            }
            if (!node.artefact.empty()) out += "artefact " + node.artefact + "\n";
            // AFTER `artefact`, because `plan_text` writes them there and the gate is a byte
            // compare: an emission order that differs is a round trip that does not.
            if (node.fold_score) {
                out += "fold_score " + events::json_number(*node.fold_score) + "\n";
            }
            if (node.fold_mask) {
                out += "fold_mask " + events::json_number(*node.fold_mask) + "\n";
            }
            if (!node.when.empty()) out += "when " + node.when + "\n";
            if (!node.per.empty()) out += "per " + node.per + "\n";
            if (!node.scope.empty()) out += "scope " + node.scope + "\n";
        }
        out += "\n";
        for (const PlanEdge& edge : plan.edges) {
            out += "edge " + edge.producer + " " + edge.consumer + " " + edge.caps + "\n";
        }
        for (const auto& [name, slots] : plan.fields) {
            out += "field " + name;
            for (const std::string& slot : slots) out += " " + slot;
            out += "\n";
        }
        return out;
    }

}  // namespace shipinfer
