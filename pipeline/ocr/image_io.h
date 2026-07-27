#pragma once
#include <cstddef>
#include <string>
#include <opencv2/core.hpp>

// Raw BGR888 only -- no JPEG/PNG decode anywhere in this pipeline (JPEG
// artifacts can distort small text, and production only ever captures raw
// to begin with, so there's nothing to decode).

// Loads a file via load_raw_bgr888() below. Kept as a thin wrapper so
// benchmark.cpp's calling convention doesn't need to change; returns an
// empty Mat (instead of exiting the process) for any path that isn't a
// recognized ".bgr888" file.
cv::Mat load_image(const std::string& path);

// Loads a raw, uncompressed BGR888 image: 3 bytes per pixel, blue-green-red
// order, no header, no compression -- the same bytes a CV_8UC3 cv::Mat holds
// in memory, so there's nothing to decode. Raw pixel data carries no size
// information of its own, so the file name must encode it:
// ..._<width>x<height>.bgr888, e.g. alarm_1024x768.bgr888. Returns an empty
// Mat on failure.
cv::Mat load_raw_bgr888(const std::string& path);
