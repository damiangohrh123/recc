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
