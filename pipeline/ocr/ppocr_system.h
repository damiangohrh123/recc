#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include "ppocr_det.h"
#include "ppocr_rec.h"

// C++ port of the Python reference implementation's ppocr_system.py.

// One detected box with its recognized text/score.
struct OcrResult {
	Quad box;
	RecResult rec;
};

// Optional per-call stage timing for TextSystem::run(), in milliseconds.
// Pass a pointer to get a breakdown (used by benchmark.cpp); existing
// callers that don't need it just omit the argument and pay no cost beyond
// one null check.
struct RunTiming {
	double det_ms = 0.0;  // 0 if the small-image fallback skipped detection
	double rec_ms = 0.0;
	int n_crops = 0;       // boxes recognized (1 on the small-image fallback path)
};

// Suppresses duplicate/nested boxes via Intersection-over-Min-Area (also
// catches a small box mostly contained in a larger one).
std::vector<Quad> nms_boxes(const std::vector<Quad>& boxes, double iom_threshold = 0.6);

// Checks whether a box's height and width both meet the given minimums.
bool box_size_ok(const Quad& b, double min_h, double min_w);

// Perspective-corrects a quad region into an upright crop, rotating 90
// degrees if the result comes out tall.
cv::Mat get_rotate_crop_image(const cv::Mat& img, const Quad& box);

// Top-to-bottom, then left-to-right within a row (~10px row tolerance).
std::vector<Quad> sorted_boxes(std::vector<Quad> dt_boxes);

// Ties detection and recognition together into the full per-image pipeline.
class TextSystem {
public:
	TextSystem(TextDetector detector, TextRecognizer recognizer,
		double drop_score, double min_height, double min_width);

	// Detects, crops, and recognizes all text in one image. If timing is
	// non-null, fills it in with a det/rec stage breakdown for this call.
	std::vector<OcrResult> run(const cv::Mat& img, RunTiming* timing = nullptr) const;

private:
	TextDetector detector_;      // finds text-box quads in an image
	TextRecognizer recognizer_;  // reads text out of a cropped box
	double drop_score_;          // minimum recognition confidence kept in results
	double min_height_;          // minimum detected box height kept
	double min_width_;           // minimum detected box width kept
};
