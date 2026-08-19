#include "alarm_detector.h"
#include <opencv2/imgproc.hpp>

AlarmDetector::AlarmDetector(int min_area, double min_aspect_ratio, double max_y_fraction)
    : min_area_(min_area), min_aspect_ratio_(min_aspect_ratio), max_y_fraction_(max_y_fraction) {}

// Finds the banner, then extracts its text from ocr_results if supplied.
AlarmResult AlarmDetector::detect(const cv::Mat& img, const std::vector<OcrResult>& ocr_results) const {
    AlarmResult result;
    auto [alarm, bbox] = find_alarm_banner(img);
    result.alarm = alarm;
    if (!alarm) return result;

    result.bbox = bbox;
    if (!ocr_results.empty()) {
        result.text = extract_alarm_text(ocr_results, bbox);
    }
    return result;
}

// Masks for red, then finds the largest contour passing all 3 filters.
std::pair<bool, cv::Rect> AlarmDetector::find_alarm_banner(const cv::Mat& img) const {
    int h_img = img.rows;

    // Red wraps around the hue wheel: masks both the 0-10 and 170-180 ranges.
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, cv::Scalar(0, 120, 70), cv::Scalar(10, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(170, 120, 70), cv::Scalar(180, 255, 255), mask2);
    cv::bitwise_or(mask1, mask2, mask);

    // Closes small gaps inside the banner's red region.
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    bool found = false;
    cv::Rect best_bbox;
    int best_area = 0;

    for (const auto& cnt : contours) {
        cv::Rect r = cv::boundingRect(cnt);
        int area = r.width * r.height;

        // Drops regions smaller than min_area_.
        if (area < min_area_) continue;

        // Drops regions narrower than min_aspect_ratio_.
        double aspect = r.height > 0 ? static_cast<double>(r.width) / r.height : 0.0;
        if (aspect < min_aspect_ratio_) continue;

        // Drops regions further down the image than max_y_fraction_.
        if (static_cast<double>(r.y) / h_img > max_y_fraction_) continue;

        if (area > best_area) {
            best_area = area;
            best_bbox = r;
            found = true;
        }
    }

    // best_bbox is still its default-constructed value here if nothing was found.
    return {found, best_bbox};
}

// Joins the text of every OCR box whose center falls inside bbox.
std::optional<std::string> AlarmDetector::extract_alarm_text(const std::vector<OcrResult>& ocr_results,
                                                               const cv::Rect& bbox) const {
    std::string joined;
    bool any_found = false;

    for (const auto& r : ocr_results) {
        // Averages the box's 4 corners to get its center point.
        float cx = 0.0f, cy = 0.0f;
        for (const auto& p : r.box) {
            cx += p.x;
            cy += p.y;
        }
        cx /= static_cast<float>(r.box.size());
        cy /= static_cast<float>(r.box.size());

        if (cx >= bbox.x && cx <= bbox.x + bbox.width && cy >= bbox.y && cy <= bbox.y + bbox.height) {
            if (any_found) joined += " ";
            joined += r.rec.text;
            any_found = true;
        }
    }

    if (!any_found) return std::nullopt;
    return joined;
}
