#include "ppocr_system.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>
#include <opencv2/imgproc.hpp>
#include "text_correction.h"

// Filters overlapping boxes using Intersection-over-Min-Area (IoM),
// keeping the larger box in each overlapping pair.
std::vector<Quad> nms_boxes(const std::vector<Quad>& boxes, double iom_threshold) {
	if (boxes.size() <= 1) return boxes;

	// Converts each box to a point vector once (reused below), and computes its area from that.
	std::vector<std::vector<cv::Point2f>> points(boxes.size());
	std::vector<double> areas(boxes.size());
	for (size_t i = 0; i < boxes.size(); ++i) {
		points[i].assign(boxes[i].begin(), boxes[i].end());
		areas[i] = cv::contourArea(points[i]);
	}

	// Sort box indices by area in descending order.
	std::vector<int> order(boxes.size());
	for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
	std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return areas[a] > areas[b]; });

	std::vector<bool> suppressed(boxes.size(), false);
	std::vector<int> keep;
	for (int i : order) {
		if (suppressed[i]) continue;
		keep.push_back(i);

		for (int j : order) {
			if (j == i || suppressed[j]) continue;
			cv::Mat intersection_poly;  // unused, but required as an out-param
			float inter_area = cv::intersectConvexConvex(points[i], points[j], intersection_poly);

			// Calculate overlap ratio relative to the smaller box area.
			double iom = inter_area / (std::min(areas[i], areas[j]) + 1e-6);
			if (iom > iom_threshold) suppressed[j] = true;
		}
	}

	// Restore original detection index order.
	std::sort(keep.begin(), keep.end());
	std::vector<Quad> result;
	result.reserve(keep.size());
	for (int i : keep) result.push_back(boxes[i]);
	return result;
}

// Checks if a box meets minimum height and width thresholds.
bool box_size_ok(const Quad& b, double min_h, double min_w) {
	double h = cv::norm(b[0] - b[3]);
	double w = cv::norm(b[0] - b[1]);
	return h >= min_h && w >= min_w;
}

// Extracts and perspective-corrects a quad region into an upright crop.
cv::Mat get_rotate_crop_image(const cv::Mat& img, const Quad& box) {
	// Crop size is the longer of each axis's two parallel edges.
	int crop_width = static_cast<int>(
		std::max(cv::norm(box[0] - box[1]), cv::norm(box[2] - box[3])));
	int crop_height = static_cast<int>(
		std::max(cv::norm(box[0] - box[3]), cv::norm(box[1] - box[2])));

	// Map the 4 (possibly tilted) box corners onto a flat rectangle.
	std::vector<cv::Point2f> src(box.begin(), box.end());
	std::vector<cv::Point2f> dst_pts = {
		{0.0f, 0.0f},
		{static_cast<float>(crop_width), 0.0f},
		{static_cast<float>(crop_width), static_cast<float>(crop_height)},
		{0.0f, static_cast<float>(crop_height)},
	};

	cv::Mat m = cv::getPerspectiveTransform(src, dst_pts);
	cv::Mat dst_img;
	cv::warpPerspective(img, dst_img, m, cv::Size(crop_width, crop_height),
		cv::INTER_CUBIC, cv::BORDER_REPLICATE);

	// Rotates the crop 90 degrees if its height noticeably exceeds its width.
	if (dst_img.cols > 0 && static_cast<double>(dst_img.rows) / dst_img.cols >= 1.5) {
		cv::Mat rotated;
		cv::rotate(dst_img, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
		dst_img = rotated;
	}
	return dst_img;
}

// Sorts bounding boxes in reading order (top-to-bottom, then left-to-right).
std::vector<Quad> sorted_boxes(std::vector<Quad> dt_boxes) {
	// Sort by top-left corner (Y first, then X).
	std::sort(dt_boxes.begin(), dt_boxes.end(), [](const Quad& a, const Quad& b) {
		if (a[0].y != b[0].y) return a[0].y < b[0].y;
		return a[0].x < b[0].x;
	});

	// Group horizontally adjacent boxes within a 10px row tolerance.
	int num_boxes = static_cast<int>(dt_boxes.size());
	for (int i = 0; i < num_boxes - 1; ++i) {
		for (int j = i; j >= 0; --j) {
			bool same_row = std::abs(dt_boxes[j + 1][0].y - dt_boxes[j][0].y) < 10;
			bool out_of_order = dt_boxes[j + 1][0].x < dt_boxes[j][0].x;
			if (same_row && out_of_order) {
				std::swap(dt_boxes[j + 1], dt_boxes[j]);
			}
			else {
				break;
			}
		}
	}
	return dt_boxes;
}

namespace {

	// Appends `boxes` to `out`, shifting each point by (dx, dy).
	void append_shifted(std::vector<Quad>& out, const std::vector<Quad>& boxes, float dx, float dy) {
		for (Quad b : boxes) {
			for (auto& p : b) {
				p.x += dx;
				p.y += dy;
			}
			out.push_back(b);
		}
	}

	// Number of tiles needed to cover total pixels without exceeding
	// native_size per tile, given overlap between neighbors.
	int grid_tile_count(int total, int native_size, int overlap) {
		if (total <= native_size) return 1;
		int step = std::max(1, native_size - overlap);
		return static_cast<int>(std::ceil(static_cast<double>(total - overlap) / step));
	}

	// Start position and size of each of n_tiles roughly-equal tiles
	// covering total pixels, overlapping neighbors by `overlap` pixels.
	std::vector<std::pair<int, int>> make_tile_spans(int total, int n_tiles, int overlap) {
		int tile_size = (total + (n_tiles - 1) * overlap) / n_tiles;
		std::vector<std::pair<int, int>> spans;
		spans.reserve(n_tiles);
		for (int i = 0; i < n_tiles; ++i) {
			int start = (i == n_tiles - 1) ? (total - tile_size) : (i * (tile_size - overlap));
			spans.push_back({ start, tile_size });
		}
		return spans;
	}

}

// Runs the detector on the full image, then on a grid of overlapping
// tiles sized to its native 480x480 input, merging all detected boxes.
std::vector<Quad> run_det_tiled(const cv::Mat& img_det, const TextDetector& detector, int overlap) {
	int h = img_det.rows, w = img_det.cols;

	int n_x = grid_tile_count(w, TextDetector::kDetW, overlap);
	int n_y = grid_tile_count(h, TextDetector::kDetH, overlap);

	std::vector<Quad> all_boxes = detector.run(img_det);  // full-image "squash" pass

	if (n_x <= 1 && n_y <= 1) {
		return all_boxes;
	}

	// Tile x-spans, computed once and reused across all y tiles.
	std::vector<std::pair<int, int>> x_spans = make_tile_spans(w, n_x, overlap);
	for (const auto& [y, tile_h] : make_tile_spans(h, n_y, overlap)) {
		for (const auto& [x, tile_w] : x_spans) {
			cv::Mat tile = img_det(cv::Rect(x, y, tile_w, tile_h));
			append_shifted(all_boxes, detector.run(tile), static_cast<float>(x), static_cast<float>(y));
		}
	}
	return all_boxes;
}

TextSystem::TextSystem(TextDetector detector, TextRecognizer recognizer,
	double drop_score, double min_height, double min_width)
	: detector_(std::move(detector)),
	recognizer_(std::move(recognizer)),
	drop_score_(drop_score),
	min_height_(min_height),
	min_width_(min_width) {}

namespace {
	double ms_since(std::chrono::steady_clock::time_point start) {
		return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	}
}

// Full pipeline for one image: find text boxes, crop each one out straight,
// read the text in each crop, and return only the results we're confident in.
std::vector<OcrResult> TextSystem::run(const cv::Mat& img, RunTiming* timing, bool use_tiling) const {
	std::vector<Quad> dt_boxes;

	// Skips detection on images smaller than 80x400; falls through to the
	// whole-image fallback below. det_ms stays 0 on that path -- no
	// detection model ran.
	if (img.rows >= 80 && img.cols >= 400) {
		auto det_start = std::chrono::steady_clock::now();
		std::vector<Quad> raw_boxes = use_tiling ? run_det_tiled(img, detector_) : detector_.run(img);
		if (timing) timing->det_ms = ms_since(det_start);

		if (!raw_boxes.empty()) {
			dt_boxes = sorted_boxes(std::move(raw_boxes));

			// Drop boxes too small to contain readable text.
			std::vector<Quad> size_filtered;
			for (auto& b : dt_boxes) {
				if (box_size_ok(b, min_height_, min_width_)) size_filtered.push_back(b);
			}
			dt_boxes = nms_boxes(size_filtered);
		}
	}

	std::vector<OcrResult> results;

	if (dt_boxes.empty()) {
		// No boxes found -- falls back to recognizing the whole image as one block of text.
		int h = img.rows, w = img.cols;
		auto rec_start = std::chrono::steady_clock::now();
		std::vector<RecResult> rec_res = recognizer_.run({ img });
		if (timing) {
			timing->rec_ms = ms_since(rec_start);
			timing->n_crops = 1;
		}
		if (!rec_res.empty() && rec_res[0].score >= drop_score_) {
			Quad box = { cv::Point2f(0, 0), cv::Point2f(static_cast<float>(w), 0),
						cv::Point2f(static_cast<float>(w), static_cast<float>(h)),
						cv::Point2f(0, static_cast<float>(h)) };
			rec_res[0].text = correct_known_ocr_errors(rec_res[0].text);
			results.push_back({ box, rec_res[0] });
		}
		return results;
	}

	// Normal path: crop each detected box out straight, then recognize all
	// the crops in one batch.
	std::vector<cv::Mat> crops;
	crops.reserve(dt_boxes.size());
	for (const auto& box : dt_boxes) crops.push_back(get_rotate_crop_image(img, box));

	auto rec_start = std::chrono::steady_clock::now();
	std::vector<RecResult> rec_res = recognizer_.run(crops);
	if (timing) {
		timing->rec_ms = ms_since(rec_start);
		timing->n_crops = static_cast<int>(crops.size());
	}
	for (size_t i = 0; i < dt_boxes.size() && i < rec_res.size(); ++i) {
		if (rec_res[i].score >= drop_score_) {
			rec_res[i].text = correct_known_ocr_errors(rec_res[i].text);
			results.push_back({ dt_boxes[i], rec_res[i] });
		}
	}
	return results;
}
