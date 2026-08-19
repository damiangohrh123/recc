// Checks ONE rule step against ONE freshly captured screen and prints the
// result as a single line of JSON. Meant to be invoked once per live
// cycle by a driver that re-reads the screen between steps (see
// recc_gen5_test_kit/automation_driver/README.md) -- a real machine's
// screen changes after every action, so live execution has to re-check
// one step at a time against a fresh read, rather than matching a whole
// rule against one saved screen in a single shot.
//
// This deliberately reuses find_by_keyword/box_center/find_field_near_label
// from rule_matcher.cpp unchanged -- the fuzzy matching itself is exactly
// what was already demonstrated against real board data; only *when* it's
// called changes here, not *how* it matches.
//
// Usage: step_matcher <screen.json> <keyword> <action> [value]
//   action is "click" or "type" (value is the text to type, required for "type")
//
// Prints one line of JSON, e.g.:
//   {"matched":true,"confidence":0.84,"best_text":"0089 Kerf check: off center","action":"click","x":296,"y":69}
//   {"matched":false,"confidence":0.67,"best_text":"sTOAT","action":"click"}
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "api/json_write.h"
#include "json_value.h"
#include "rule_matcher.h"

namespace {

// Turns a real POST /ocr response into DetectedBoxes. Same parsing for a
// freshly captured screen and a replayed saved one.
std::vector<DetectedBox> load_screen(const std::string& path) {
    std::vector<DetectedBox> boxes;
    const JsonValue doc = json_parse_file(path);
    const JsonValue* boxes_val = doc.find("boxes");
    if (!boxes_val) return boxes;
    for (const auto& item : boxes_val->arr) {
        DetectedBox b;
        if (auto* t = item.find("text")) b.text = t->str;
        if (auto* sc = item.find("score")) b.score = sc->num;
        if (auto* bx = item.find("box")) {
            for (size_t i = 0; i < 4 && i < bx->arr.size(); ++i) {
                const auto& pt = bx->arr[i];
                if (pt.arr.size() >= 2) {
                    b.box[i] = {static_cast<int>(pt.arr[0].num), static_cast<int>(pt.arr[1].num)};
                }
            }
        }
        boxes.push_back(std::move(b));
    }
    return boxes;
}

// The accept/reject decision for one step lives here: rule_matcher.cpp's
// find_by_keyword returns a raw score and leaves the threshold to its caller.
// automation_driver.py mirrors this value for log messages only.
constexpr double kConfidenceThreshold = 0.75;

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <screen.json> <keyword> <action> [value]\n", argv[0]);
        return 2;
    }
    const std::string screen_path = argv[1];
    const std::string keyword = argv[2];
    const std::string action = argv[3];
    const std::string value = argc > 4 ? argv[4] : "";

    if (action != "click" && action != "type") {
        fprintf(stderr, "unrecognized action %s (expected \"click\" or \"type\")\n", action.c_str());
        return 2;
    }

    std::vector<DetectedBox> boxes;
    try {
        boxes = load_screen(screen_path);
    } catch (const std::exception& e) {
        fprintf(stderr, "failed to load %s: %s\n", screen_path.c_str(), e.what());
        return 1;
    }

    auto [match, score] = find_by_keyword(keyword, boxes);
    const std::string best_text = match ? match->text : "";
    const bool matched = (match != nullptr && score >= kConfidenceThreshold);

    printf("{\"matched\":%s,\"confidence\":%.4f,\"best_text\":\"%s\",\"action\":\"%s\"",
           matched ? "true" : "false", score, json_escape(best_text).c_str(), action.c_str());

    if (matched) {
        if (action == "click") {
            auto [cx, cy] = box_center(match->box);
            printf(",\"x\":%d,\"y\":%d", cx, cy);
        } else {  // "type"
            auto [x, y] = find_field_near_label(*match, boxes);
            printf(",\"x\":%d,\"y\":%d,\"text\":\"%s\"", x, y, json_escape(value).c_str());
        }
    }
    printf("}\n");
    return 0;
}
