// The plan gate: every committed plan, read by this plane and written back byte-identically.
//
// The artefact that crosses the plane boundary IS the golden here (`benchmarks/parity/golden/
// plans/*.plan`, written by `topology/plan.py`), so the comparison is direct: parse,
// re-serialise, compare the bytes. A parse that silently discarded a verb it did not know
// would pass a "did it load?" check and lose the chain — the round trip is what makes that
// impossible.
//
// Refusals are compared too, because a plan one plane reads and the other rejects is the
// worst outcome this seam has, and it is the half a golden cannot express.
//
// Offline: g++ alone, no CUDA, no GStreamer.

#include <cstdio>
#include <string>
#include <vector>

#include "shipinfer/core/types.h"
#include "shipinfer/pipeline/graph/plan.h"
#include "tests/parity_files.h"

namespace {

    using namespace shipinfer;
    using namespace shipinfer::parity;

    int failures = 0;
    int checks = 0;

    void check(bool condition, const std::string& what) {
        ++checks;
        if (!condition) {
            ++failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    // Every plan the Python plane committed, by name. Listed rather than globbed: a golden
    // that disappears has to fail this gate, and a directory walk would call that "nothing
    // to check" and pass.
    const std::vector<std::string> kGoldens = {"minimal", "branching", "defaults",
                                               "ship_person_cpu"};

    std::string read_text(const std::string& path) {
        std::string text;
        for (const std::string& line : read_lines_keeping_blanks(path)) text += line + "\n";
        return text;
    }

    void round_trips(const std::string& name) {
        const std::string path = resolve("golden/plans/" + name + ".plan");
        const std::string expected = read_text(path);
        const ResolvedPlan plan = read_plan(path);
        const std::string written = plan_text(plan);
        check(written == expected, name + ": re-serialised plan differs from the golden");
        if (written != expected) {
            std::printf("  golden:\n%s\n  written:\n%s\n", expected.c_str(), written.c_str());
        }
        check(plan.version == kPlanVersion, name + ": version");
    }

    // What the reader is FOR: the values a builder then acts on. Asserted on the branching
    // plan, which carries every verb the format has.
    void reads_the_branching_plan() {
        const ResolvedPlan plan = read_plan(resolve("golden/plans/branching.plan"));

        check(plan.name == "branching", "chain name");
        check(plan.nodes.size() == 8, "eight nodes");
        // A ship is 8 in this checkout's detector, and that is the whole reason the plan
        // carries a label table: `bench.cpp` assumed 1 and labelled every ship `unknown`.
        check(plan.class_id("ship") == 8, "ship is class 8");
        check(plan.class_id("person") == 0, "person is class 0");
        check(!plan.class_id("vessel").has_value(), "a label nobody declared has no id");

        const PlanNode* detect = plan.node("detect");
        check(detect != nullptr && detect->model == "ship_detector", "detect model");
        check(detect != nullptr && detect->letterbox && detect->letterbox->height == 640,
              "detect letterbox from the model's config.yaml");
        check(detect != nullptr && detect->score_threshold && *detect->score_threshold == 0.3,
              "detect score threshold");
        check(detect != nullptr && detect->max_detections && *detect->max_detections == 120,
              "detect max_detections");

        const PlanNode* segment = plan.node("segment");
        check(segment != nullptr && segment->crop && segment->crop->height == 512,
              "a declared crop overrides the model's");
        check(segment != nullptr && segment->when == "fps == 20", "the frame-level guard");

        const PlanNode* ship = plan.node("embed_ship");
        check(ship != nullptr && ship->classes &&
                  *ship->classes == std::vector<std::string>{"ship"},
              "row selection");
        check(ship != nullptr && ship->crop && ship->crop->height == 256 &&
                  ship->crop->width == 128,
              "crop resolved from the model, tall not square");

        const PlanNode* track = plan.node("track");
        check(track != nullptr && track->per == "camera", "per: camera");
        check(plan.node("mtmc") != nullptr && plan.node("mtmc")->scope == "global", "scope");
        check(plan.node("nothing") == nullptr, "an undeclared slot is absent, not empty");

        check(plan.edges.size() == 8, "eight edges");
        check(plan.edges.front().producer == "decode" && plan.edges.front().caps == "bgr@cpu",
              "the head edge and its negotiated cap");
        check(plan.fields.at("embedding").size() == 2, "both embedders fill `embedding`");
        check(plan.fields.at("mask_area_px").size() == 1, "the segmenter fills mask_area_px");
    }

    // The minimal plan is here because a reader is most likely to assume the sections a
    // chain without models simply does not have.
    void reads_a_chain_with_no_model() {
        const ResolvedPlan plan = read_plan(resolve("golden/plans/minimal.plan"));

        check(plan.nodes.size() == 2, "decode and output");
        check(plan.labels.empty(), "no detector, no label table");
        check(plan.fields.empty(), "no embedder, no field lines");
        check(plan.node("decode")->model.empty(), "no model on a decode");
    }

    bool refused(const std::string& text) {
        try {
            parse_plan(text, "probe");
            return false;
        } catch (const ConfigError&) {
            return true;
        }
    }

    // The refusals `topology/plan.py` raises, in the same order, so a malformed plan cannot
    // load on one plane and fail on the other. `tests/topology/test_plan.py` holds the
    // Python half to this same table.
    void refuses_what_python_refuses() {
        check(refused(""), "an empty plan has no header");
        check(refused("node a decode replay\n"), "a verb before the header");
        check(refused("plan 1 x\nplan 1 y\n"), "a second header");
        check(refused("plan 2 x\n"), "an unknown version");
        check(refused("plan 1 x\nmodel m\n"), "an attribute before any node");
        check(refused("plan 1 x\nnode a b\n"), "node with two arguments");
        check(refused("plan 1 x\nnode a b c\ncrop 256\n"), "crop with one extent");
        check(refused("plan 1 x\nnode a b c\ncrop 0 128\n"), "a crop that is not positive");
        check(refused("plan 1 x\nnode a b c\ninstances 0\n"), "zero instances runs nothing");
        check(refused("plan 1 x\nnode a b c\ninstances -1\n"), "and a negative count");
        check(refused("plan 1 x\nnode a b c\nqueue_delay_us -1\n"), "a negative window");
        check(refused("plan 1 x\nnode a b c\ninstances two\n"), "a count that is not one");
        check(refused("plan 1 x\nnode a b c\nartefact a b\n"), "an artefact with a space");
        check(refused("plan 1 x\nnode a b c\nfold_mask 0\n"), "a mask cut of 0 is -inf");
        check(refused("plan 1 x\nnode a b c\nfold_mask 1\n"), "and 1 divides by zero");
        check(refused("plan 1 x\nnode a b c\nfold_mask 1.5\n"), "and past it");
        check(refused("plan 1 x\nnode a b c\nfold_score nan\n"), "a non-finite score floor");
        check(refused("plan 1 x\nnode a b c\nfold_detections a b\n"),
              "an output name holding a space");
        check(!refused("plan 1 x\nnode a b c\nfold_detections det\nfold_prototypes proto\n"),
              "an export that names its outputs something other than output0/output1");
        check(!refused("plan 1 x\nnode a b c\nfold_mask 0.5\nfold_score 0.25\n"),
              "the two fold cuts");
        check(!refused("plan 1 x\nnode a b c\nfold_score 0.0\n"),
              "a floor of zero: every crop is an area");
        check(!refused("plan 1 x\nnode a b c\nqueue_delay_us 0\n"), "0 is batching off");
        check(!refused("plan 1 x\nnode a b c\ninstances 1\n"), "the smallest count");
        check(!refused("plan 1 x\nnode a b c\nartefact m/1/model.plan\n"),
              "a repository-relative artefact");
        check(refused("plan 1 x\nnode a b c\nscore nan\n"), "a non-finite score");
        check(refused("plan 1 x\nnode a b c\nnonsense 1\n"), "an unknown verb");
        check(refused("plan 1 x\nlabel eight ship\n"), "a label id that is not an integer");
        check(refused("plan 1 x\nedge a b\n"), "an edge with no cap");
        check(refused("plan 1 x\nfield embedding\n"), "a field with no slot");
        check(refused("plan 1_0 x\n"), "an integer Python would accept and this would not");
        check(refused("plan 1 x\nnode a b c\nnode a b c\n"), "a second block for one slot");
        check(refused("plan 1 x\nlabel 8 ship\nlabel 8 vessel\n"), "a second row for one id");
        check(refused("plan 1 x\nnode a b c\nclasses ,ship\n"), "an empty label in `classes`");
        check(refused("plan 1 x\nnode a b c\nclasses ship,\n"), "a trailing comma");
        check(refused("plan 1 x\nfield embedding a\nfield embedding b\n"),
              "a second `field` for one name");
        check(refused("plan 1 x\nfield embedding nosuch\n"),
              "a `field` naming a slot no `node` declares");
        check(refused("plan 1 x\nnode a b c\nscore 0x10\n"),
              "a hex float, which bare `stod` would have read");
        check(refused("plan 1 x\nlabel 99999999999999999999 ship\n"),
              "an id too large for an int, which Python's unbounded `int` used to accept");
        check(refused("plan 1 x\nnode a b c\nmax_detections -1\n"),
              "`-1` for `no limit`: no bound here, one row fewer there");
        check(refused("plan 1 x\nnode a b c\nmax_detections 0\n"), "and zero");
        check(refused("plan 1 x\nnode a b c\nscore 1e400\n"),
              "an exponent that overflows to `inf`, which Python's regex alone allowed");
        // The emitter refuses a value holding a tab, a newline or a repeated space, because
        // `split()` collapses them -- and a newline emits an extra LINE, which injected a
        // `node` the chain never declared (#131 round 3). This reader is the other half: it
        // must read what the emitter can legally write, and these are not that.
        const ResolvedPlan collapsed =
            parse_plan("plan 1 x\nnode a b c\nclasses cargo  ship\n", "probe");
        check(
            collapsed.node("a")->classes && (*collapsed.node("a")->classes)[0] == "cargo ship",
            "a repeated space collapses here too, which is why the emitter refuses it");

        check(!refused("plan 1 -\n"), "`-` is the empty chain name, and is legal");
        check(!refused("plan 1 x  # trailing comment\n"), "a comment after a directive");
        check(!refused("plan 1 x\nnode a b c\nscore 5e-324\n"),
              "a subnormal threshold: `stod` threw on ERANGE-underflow where Python's "
              "`float()` returns it, so this plane refused a plan the other one writes");
    }

    // The multi-word half of the same table. `plan` and `label` take the REST of the line and
    // `classes` splits on commas, because a whitespace-delimited fixed arity refused a plan
    // `shipinfer plan` had just written -- and silently re-read `[cargo ship]` as two labels
    // that no detector emits. Found by #131's review, before either reader shipped.
    void carries_what_this_domain_names_things() {
        const ResolvedPlan named = parse_plan("plan 1 ship person cpu\n", "probe");
        check(named.name == "ship person cpu", "a multi-word chain name");

        const ResolvedPlan labelled =
            parse_plan("plan 1 x\nlabel 0 person\nlabel 8 cargo ship\n", "probe");
        check(labelled.labels.at(8) == "cargo ship", "a multi-word label");
        check(labelled.class_id("cargo ship") == 8, "and it is findable by name");

        const ResolvedPlan selected = parse_plan(
            "plan 1 x\nnode e embed pool\nclasses cargo ship,fishing vessel\n", "probe");
        const std::vector<std::string> want = {"cargo ship", "fishing vessel"};
        check(selected.node("e")->classes == want, "two labels, not four");

        // And the writer spells them back the same way, which is what the golden compare
        // would catch only if a golden happened to carry one.
        check(
            plan_text(selected).find("classes cargo ship,fishing vessel") != std::string::npos,
            "the writer joins classes with commas");
        check(plan_text(named).find("plan 1 ship person cpu") != std::string::npos,
              "and writes a multi-word name unquoted");
    }

    // The distinction #131's round 2 added, and the one a golden alone cannot make: an
    // ABSENT `classes` line is every row, a `classes -` line is NO rows. Conflating them
    // selects everything where the chain said nothing -- at an embedder, a doubled GPU bill.
    void tells_select_nothing_from_select_everything() {
        const ResolvedPlan absent = parse_plan("plan 1 x\nnode e embed pool\n", "probe");
        check(!absent.node("e")->classes.has_value(), "no line: no selection declared");

        const ResolvedPlan empty =
            parse_plan("plan 1 x\nnode e embed pool\nclasses -\n", "probe");
        check(empty.node("e")->classes.has_value() && empty.node("e")->classes->empty(),
              "`classes -`: a declared empty selection");

        // And both write back the way they came, which is what makes them distinguishable
        // across the boundary at all.
        check(plan_text(absent).find("classes") == std::string::npos, "absent stays absent");
        check(plan_text(empty).find("classes -") != std::string::npos, "and `-` stays `-`");

        // The committed `defaults` golden carries the real thing, resolved from `classes: []`.
        const ResolvedPlan golden = read_plan(resolve("golden/plans/defaults.plan"));
        const PlanNode* none = golden.node("embed_none");
        check(none != nullptr && none->classes && none->classes->empty(),
              "the golden's declared-empty slot");
        // The same golden pins the two values a silent detect slot must still carry, and the
        // exponent spelling `repr` produces -- which this writer has to match byte for byte.
        const PlanNode* silent = golden.node("detect");
        check(silent != nullptr && silent->score_threshold && *silent->score_threshold == 0.25,
              "a slot that declares nothing still carries its threshold");
        check(silent != nullptr && silent->max_detections && *silent->max_detections == 100,
              "and its cap");
        const PlanNode* small = golden.node("detect_small");
        check(small != nullptr && small->score_threshold && *small->score_threshold == 1e-05,
              "and an exponent-form threshold round-trips");
    }

}  // namespace

int main() {
    try {
        for (const std::string& name : kGoldens) round_trips(name);
        reads_the_branching_plan();
        reads_a_chain_with_no_model();
        refuses_what_python_refuses();
        carries_what_this_domain_names_things();
        tells_select_nothing_from_select_everything();
    } catch (const std::exception& error) {
        std::printf("FAIL: unexpected exception: %s\n", error.what());
        ++failures;
    }
    std::printf("%d checks, %d failure(s)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
