#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "rknn_executor.h"
#include "preprocess.h"

// C++ port of the Python reference implementation's ppocr_det.py.

// A detected quadrilateral: 4 corners ordered top-left, top-right,
// bottom-right, bottom-left.
using Quad = std::array<cv::Point2f, 4>;

// Turns the detector's raw probability map into candidate text-box quads.
class DBPostProcess {
public:
    DBPostProcess(float thresh, float box_thresh, int max_candidates, float unclip_ratio);

    // Thresholds the probability map into a binary mask, then extracts
    // boxes from it. The only entry point; everything below is internal.
    std::vector<Quad> run(const cv::Mat& pred_map, int dest_width, int dest_height,
                           float ratio_w, float ratio_h) const;

private:
    // Computes the smallest rotated rectangle enclosing a set of points,
    // returned as 4 ordered corners plus its short side length ("sside").
    static std::pair<Quad, float> get_mini_boxes(const std::vector<cv::Point2f>& points);

    // Expands a quad outward by a distance based on its own area and
    // perimeter; may return more than 4 points.
    std::vector<cv::Point2f> unclip(const std::vector<cv::Point2f>& box) const;

    // Computes the average probability-map value inside a box's shape.
    static float box_score_fast(const cv::Mat& pred_map, const Quad& box);

    // Finds contours in the binary mask and turns each into a scored,
    // expanded, resized quadrilateral.
    std::vector<Quad> boxes_from_bitmap(const cv::Mat& pred_map, const cv::Mat& bitmap_u8,
                                         int dest_width, int dest_height,
                                         float ratio_w, float ratio_h) const;

    float thresh_;          // probability threshold used to binarize pred_map
    float box_thresh_;      // minimum average score for a candidate box to survive
    int max_candidates_;    // cap on contours examined per image
    float unclip_ratio_;    // how far unclip() expands each box outward
    static constexpr float kMinSize = 3.0f;
};

// Runs the full detection pipeline: resize, normalize, run the model, and
// turn its output into text-box quadrilaterals.
class TextDetector {
public:
    TextDetector(const std::string& det_model_path,
                 float det_thresh, float box_thresh, float unclip_ratio, int max_candidates);

    bool is_loaded() const { return model_ && model_->is_loaded(); }

    // Resizes and normalizes the image, runs the model, and converts its
    // output into quads in the original image's coordinates.
    std::vector<Quad> run(const cv::Mat& img) const;

    static constexpr int kDetH = 480;  // fixed input height the model expects
    static constexpr int kDetW = 480;  // fixed input width the model expects

private:
    std::unique_ptr<RknnExecutor> model_;  // the loaded detection model
    NormalizeImage normalize_;              // normalizes pixel values before inference
    DBPostProcess db_postprocess_;          // converts the model's raw output into boxes
};
