#pragma once
#include <array>
#include <string>
#include <vector>

// Matches an offline-drafted, pre-tested rule against real OCR output,
// and turns a match into a KVM action sequence. Deliberately has zero
// OpenCV/RKNN dependency -- this logic never touches an image, only the
// text and coordinates OCR already produced, so it builds and runs
// anywhere, unlike the OCR pipeline itself.

// One detected text box, same shape as a real POST /ocr response's
// "boxes" entries: text, confidence score, and 4 corner points
// (top-left, top-right, bottom-right, bottom-left).
struct DetectedBox {
    std::string text;
    double score = 0.0;
    std::array<std::array<int, 2>, 4> box{};
};

// Returns the box whose text best matches the keyword, and its match score.
// Scoring is the Ratcliff/Obershelp similarity ratio (the same algorithm
// Python's difflib.SequenceMatcher.ratio() uses), plus a 0.3 bonus when the
// keyword appears as a substring -- so the score ranges over [0, 1.3], not
// [0, 1]. Returns {nullptr, 0.0} if boxes is empty.
std::pair<const DetectedBox*, double> find_by_keyword(const std::string& keyword,
                                                       const std::vector<DetectedBox>& boxes);

// Center point of a 4-corner box.
std::pair<int, int> box_center(const std::array<std::array<int, 2>, 4>& box);

// Guesses an empty input field's position from its label, by checking
// for open space immediately to the right of the label, space no other
// detected box already occupies.
std::pair<int, int> find_field_near_label(const DetectedBox& label_box,
                                           const std::vector<DetectedBox>& boxes);
