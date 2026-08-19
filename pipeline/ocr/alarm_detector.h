#pragma once
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <opencv2/core.hpp>
#include "ppocr_system.h"

// C++ port of the Python reference implementation's alarm_detector.py.

// Whether an alarm banner was found, its box, and any text found inside it.
struct AlarmResult {
    bool alarm = false;
    cv::Rect bbox;                     // valid only when alarm is true
    std::optional<std::string> text;   // set only if OCR results were supplied and text was found inside bbox
};

// Finds a red alarm banner in an image and reads any text inside it.
class AlarmDetector {
public:
    // min_area: minimum pixel area to count as a banner. min_aspect_ratio:
    // minimum width/height ratio. max_y_fraction: how close to the top it must be.
    explicit AlarmDetector(int min_area = 10000, double min_aspect_ratio = 4.0, double max_y_fraction = 0.35);

    // Finds the banner, then extracts its text from ocr_results if supplied.
    AlarmResult detect(const cv::Mat& img, const std::vector<OcrResult>& ocr_results = {}) const;

private:
    // Masks for red, then finds the largest contour passing all 3 filters.
    std::pair<bool, cv::Rect> find_alarm_banner(const cv::Mat& img) const;

    // Joins the text of every OCR box whose center falls inside bbox.
    std::optional<std::string> extract_alarm_text(const std::vector<OcrResult>& ocr_results,
                                                    const cv::Rect& bbox) const;

    int min_area_;             // minimum red-region pixel area to count as a banner
    double min_aspect_ratio_;  // minimum width/height ratio to count as a banner
    double max_y_fraction_;    // maximum top-edge position, as a fraction of image height
};
