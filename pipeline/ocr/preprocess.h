#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>

// C++ port of the Python reference implementation's utils/preprocess.py.

// Applies (img * scale - mean) / std per channel. Shared by ppocr_det.cpp
// and ppocr_rec.cpp, each with its own scale/mean/std values.
// Only the HWC path is ported; Python's CHW branch was unused here too.
class NormalizeImage {
public:
    NormalizeImage(double scale, const std::array<double, 3>& mean, const std::array<double, 3>& std);

    // Converts img to CV_32FC3 and applies (img * scale - mean) / std per channel.
    cv::Mat operator()(const cv::Mat& img) const;

private:
    double scale_;      // multiplier applied first
    cv::Scalar mean_;   // per-channel value subtracted after scaling
    cv::Scalar std_;    // per-channel value divided by last
};

// Flattens an HWC CV_32FC3 image into a flat float32 buffer (batch size 1),
// the layout RknnExecutor::run() expects.
std::vector<float> to_nhwc_batch(const cv::Mat& img);
