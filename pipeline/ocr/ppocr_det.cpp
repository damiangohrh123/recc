#include "ppocr_det.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <opencv2/imgproc.hpp>
#include <polyclipping/clipper.hpp>

namespace {

// Resizes an image to (target_w, target_h) and records how much each
// dimension was scaled.
struct ResizeResult {
	cv::Mat image;
	float ratio_h;  // target_h / original image height
	float ratio_w;  // target_w / original image width
};

double polygon_area(const std::vector<cv::Point2f>& points) {
	double area = 0.0;
	size_t n = points.size();
	for (size_t i = 0; i < n; ++i) {
		const auto& p1 = points[i];
		const auto& p2 = points[(i + 1) % n];
		area += static_cast<double>(p1.x) * p2.y - static_cast<double>(p2.x) * p1.y;
	}
	return std::abs(area) / 2.0;  // abs() makes the result winding-order independent
}

double polygon_perimeter(const std::vector<cv::Point2f>& points) {
	double perimeter = 0.0;
	size_t n = points.size();
	for (size_t i = 0; i < n; ++i) {
		perimeter += cv::norm(points[(i + 1) % n] - points[i]);
	}
	return perimeter;
}

ResizeResult det_resize_for_test(const cv::Mat& img, int target_h, int target_w) {
	ResizeResult result;
	// Records how much each dimension is being scaled, to map coordinates back to the original image size later.
	result.ratio_h = static_cast<float>(target_h) / img.rows;
	result.ratio_w = static_cast<float>(target_w) / img.cols;
	cv::resize(img, result.image, cv::Size(target_w, target_h));
	return result;
}

	// Splits 4 x-sorted points into left/right pairs and picks the smaller-y
	// point in each pair as the top corner, returning tl, tr, br, bl.
	Quad order_quad(const std::vector<cv::Point2f>& sorted_by_x) {
		int i1, i4;
		if (sorted_by_x[1].y > sorted_by_x[0].y) { i1 = 0; i4 = 1; }
		else { i1 = 1; i4 = 0; }
		int i2, i3;
		if (sorted_by_x[3].y > sorted_by_x[2].y) { i2 = 2; i3 = 3; }
		else { i2 = 3; i3 = 2; }
		return { sorted_by_x[i1], sorted_by_x[i2], sorted_by_x[i3], sorted_by_x[i4] };
	}

}

// --- DBPostProcess -----------------------------------------------------

// Stores the detection thresholds and config used by the methods below.
DBPostProcess::DBPostProcess(float thresh, float box_thresh, int max_candidates, float unclip_ratio)
	: thresh_(thresh), box_thresh_(box_thresh), max_candidates_(max_candidates), unclip_ratio_(unclip_ratio) {}

std::pair<Quad, float> DBPostProcess::get_mini_boxes(const std::vector<cv::Point2f>& points_in) {
	// Computes the minimum-area rotated rectangle enclosing all the points.
	cv::RotatedRect bounding_box = cv::minAreaRect(points_in);

	cv::Point2f corners[4];
	bounding_box.points(corners);
	std::vector<cv::Point2f> points(corners, corners + 4);
	std::sort(points.begin(), points.end(),
		[](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });  // for order_quad

	Quad box = order_quad(points);
	float sside = std::min(bounding_box.size.width, bounding_box.size.height);
	return { box, sside };
}

std::vector<cv::Point2f> DBPostProcess::unclip(const std::vector<cv::Point2f>& box) const {
	double area = polygon_area(box);
	double perimeter = polygon_perimeter(box);
	double distance = area * unclip_ratio_ / perimeter;

	// Converts the box's corners to the integer coordinates Clipper expects.
	ClipperLib::Path path;
	for (const auto& p : box) {
		path << ClipperLib::IntPoint(static_cast<ClipperLib::cInt>(p.x),
			static_cast<ClipperLib::cInt>(p.y));
	}

	// Expands the polygon outward by `distance`, rounding its corners.
	ClipperLib::ClipperOffset offset;
	offset.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
	ClipperLib::Paths solution;
	offset.Execute(solution, distance);

	// Converts the expanded polygon's points back to floats.
	std::vector<cv::Point2f> result;
	if (!solution.empty()) {
		result.reserve(solution[0].size());
		for (const auto& ip : solution[0]) {
			result.emplace_back(static_cast<float>(ip.X), static_cast<float>(ip.Y));
		}
	}
	return result;
}

float DBPostProcess::box_score_fast(const cv::Mat& pred_map, const Quad& box) {
	int h = pred_map.rows, w = pred_map.cols;

	// Finds the bounding box of the 4 corners, clamped to the map's bounds.
	float xmin_f = box[0].x, xmax_f = box[0].x, ymin_f = box[0].y, ymax_f = box[0].y;
	for (const auto& p : box) {
		xmin_f = std::min(xmin_f, p.x);
		xmax_f = std::max(xmax_f, p.x);
		ymin_f = std::min(ymin_f, p.y);
		ymax_f = std::max(ymax_f, p.y);
	}
	int xmin = std::clamp(static_cast<int>(std::floor(xmin_f)), 0, w - 1);
	int xmax = std::clamp(static_cast<int>(std::ceil(xmax_f)), 0, w - 1);
	int ymin = std::clamp(static_cast<int>(std::floor(ymin_f)), 0, h - 1);
	int ymax = std::clamp(static_cast<int>(std::ceil(ymax_f)), 0, h - 1);

	// Draws the box's shape into a mask sized to just that bounding box.
	cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
	std::vector<cv::Point> shifted;
	shifted.reserve(box.size());
	for (const auto& p : box) {
		// Shifts each corner relative to the mask's own origin (xmin, ymin).
		shifted.emplace_back(static_cast<int>(p.x) - xmin, static_cast<int>(p.y) - ymin);
	}
	std::vector<std::vector<cv::Point>> fill_contours{ shifted };
	cv::fillPoly(mask, fill_contours, cv::Scalar(1));

	// Averages the probability map's values under the box's shape.
	cv::Rect roi(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
	cv::Scalar mean = cv::mean(pred_map(roi), mask);
	return static_cast<float>(mean[0]);
}

std::vector<Quad> DBPostProcess::boxes_from_bitmap(const cv::Mat& pred_map, const cv::Mat& bitmap_u8,
	int dest_width, int dest_height,
	float ratio_w, float ratio_h) const {
	int height = bitmap_u8.rows, width = bitmap_u8.cols;

	// Finds every connected white region in the binary mask.
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(bitmap_u8, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	int num_contours = std::min(static_cast<int>(contours.size()), max_candidates_);

	std::vector<Quad> boxes;
	for (int idx = 0; idx < num_contours; ++idx) {
		std::vector<cv::Point2f> contour_f(contours[idx].begin(), contours[idx].end());
		if (contour_f.empty()) continue;

		// Reduces the contour to a 4-corner box, skipping ones too thin to be text.
		auto [points, sside] = get_mini_boxes(contour_f);
		if (sside < kMinSize) continue;

		// Skips boxes whose average probability is below the threshold.
		float score = box_score_fast(pred_map, points);
		if (score < box_thresh_) continue;

		// Expands the box outward, then re-derives its 4 corners from the
		// expanded shape, skipping degenerate results.
		std::vector<cv::Point2f> points_vec(points.begin(), points.end());
		std::vector<cv::Point2f> unclipped = unclip(points_vec);
		if (unclipped.size() < 3) continue;

		auto [box2, sside2] = get_mini_boxes(unclipped);
		if (sside2 < kMinSize + 2) continue;

		// Falls back to computing the ratio from raw dimensions if none was given.
		float rw = (ratio_w > 0.0f) ? ratio_w : (static_cast<float>(width) / dest_width);
		float rh = (ratio_h > 0.0f) ? ratio_h : (static_cast<float>(height) / dest_height);

		// Scales each corner back to the original image size and clamps it in bounds.
		Quad final_box;
		for (int i = 0; i < 4; ++i) {
			float x = std::round(box2[i].x / rw);
			float y = std::round(box2[i].y / rh);
			x = std::clamp(x, 0.0f, static_cast<float>(dest_width));
			y = std::clamp(y, 0.0f, static_cast<float>(dest_height));
			final_box[i] = cv::Point2f(x, y);
		}
		boxes.push_back(final_box);
	}
	return boxes;
}

std::vector<Quad> DBPostProcess::run(const cv::Mat& pred_map, int dest_width, int dest_height,
	float ratio_w, float ratio_h) const {
	// Converts the raw probability map into a 0/255 binary mask.
	cv::Mat bitmap = pred_map > thresh_;
	return boxes_from_bitmap(pred_map, bitmap, dest_width, dest_height, ratio_w, ratio_h);
}

// --- detected-quad cleanup (file-local: only TextDetector::run uses it) ----

namespace {

Quad order_points_clockwise(const Quad& pts) {
	// Sorts the 4 points by x, then reorders them into tl, tr, br, bl.
	std::vector<cv::Point2f> sorted_by_x(pts.begin(), pts.end());
	std::sort(sorted_by_x.begin(), sorted_by_x.end(),
		[](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
	return order_quad(sorted_by_x);
}

Quad clip_det_res(Quad points, int img_height, int img_width) {
	for (auto& p : points) {
		// Clamps the point into the image bounds, then truncates it to an integer pixel.
		float x = std::clamp(p.x, 0.0f, static_cast<float>(img_width - 1));
		float y = std::clamp(p.y, 0.0f, static_cast<float>(img_height - 1));
		p.x = static_cast<float>(static_cast<int>(x));
		p.y = static_cast<float>(static_cast<int>(y));
	}
	return points;
}

std::vector<Quad> filter_tag_det_res(const std::vector<Quad>& dt_boxes,
	int image_height, int image_width) {
	std::vector<Quad> result;
	for (const auto& raw_box : dt_boxes) {
		// Reorders and clips each box, then drops it if it's too small to hold text.
		Quad box = order_points_clockwise(raw_box);
		box = clip_det_res(box, image_height, image_width);
		int rect_width = static_cast<int>(cv::norm(box[0] - box[1]));
		int rect_height = static_cast<int>(cv::norm(box[0] - box[3]));
		if (rect_width <= 3 || rect_height <= 3) continue;
		result.push_back(box);
	}
	return result;
}

}  // namespace

// --- TextDetector ----------------------------------------------------------

TextDetector::TextDetector(const std::string& det_model_path,
	float det_thresh, float box_thresh, float unclip_ratio, int max_candidates)
	: model_(load_model(det_model_path)),
	// Configures normalization as scale=1, mean=0, std=1 -- a no-op, since
	// this model already has normalization baked into its weights.
	normalize_(1.0, { 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 }),
	db_postprocess_(det_thresh, box_thresh, max_candidates, unclip_ratio) {
	if (!model_) {
		fprintf(stderr, "TextDetector: failed to load model %s\n", det_model_path.c_str());
	}
}

std::vector<Quad> TextDetector::run(const cv::Mat& img) const {
	if (!is_loaded()) return {};

	int src_h = img.rows, src_w = img.cols;
	// Resizes the image to the model's fixed input size, recording the scale ratios.
	ResizeResult resized = det_resize_for_test(img, kDetH, kDetW);
	cv::Mat normalized = normalize_(resized.image);
	std::vector<float> batch = to_nhwc_batch(normalized);

	// Runs the detection model on the prepared image.
	auto outputs = model_->run(batch);
	if (outputs.empty()) return {};

	// Checks that the output is a single-channel probability map.
	const RknnOutput& maps_out = outputs[0];
	if (maps_out.shape.size() != 4 || maps_out.shape[0] != 1 || maps_out.shape[1] != 1) {
		fprintf(stderr, "TextDetector: unexpected det output shape (expected NCHW with N=1,C=1)\n");
		return {};
	}
	int map_h = maps_out.shape[2], map_w = maps_out.shape[3];
	// Wraps the raw output buffer as an image, without copying it.
	cv::Mat pred_map(map_h, map_w, CV_32FC1, const_cast<float*>(maps_out.data.data()));

	// Converts the probability map into boxes, then cleans up their order and bounds.
	std::vector<Quad> raw_boxes = db_postprocess_.run(pred_map, src_w, src_h, resized.ratio_w, resized.ratio_h);
	return filter_tag_det_res(raw_boxes, src_h, src_w);
}
