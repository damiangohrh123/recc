#pragma once
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "rknn_executor.h"
#include "preprocess.h"

// C++ port of the Python reference implementation's ppocr_rec.py.

// Holds one crop's recognized text and its confidence score.
struct RecResult {
    std::string text;  // UTF-8 encoded recognized text
    float score;       // average confidence over the kept characters
};

// Decodes a recognition model's raw per-timestep class scores into text.
class CtcLabelDecode {
public:
    // Loads the character dictionary, one UTF-8 character per line, with
    // "blank" prepended and an optional trailing space character.
    explicit CtcLabelDecode(const std::string& character_dict_path, bool use_space_char = true);

    // Picks the highest-scoring class at each timestep, then collapses
    // repeats and blanks into the final decoded text and its mean score.
    RecResult decode(const float* preds, int seq_len, int num_classes) const;

private:
    std::vector<std::string> character_;  // index 0 is always "blank"
};

// Runs recognition on cropped text-box images.
class TextRecognizer {
public:
    TextRecognizer(const std::string& rec_model_path,
                   const std::string& character_dict_path);

    bool is_loaded() const { return model_ && model_->is_loaded(); }

    // Resizes each crop to 320x48, normalizes it, and runs the model.
    // Crops are split across kNumCores NPU cores and run concurrently;
    // results are returned in the same order as imgs.
    std::vector<RecResult> run(const std::vector<cv::Mat>& imgs) const;

    static constexpr int kRecH = 48;   // fixed crop height the model expects
    static constexpr int kRecW = 320;  // fixed crop width the model expects
    static constexpr int kNumCores = 3;  // NPU cores to dispatch recognition across

private:
    std::unique_ptr<RknnExecutor> model_;  // the loaded recognition model
    NormalizeImage normalize_;              // scales pixel values before inference
    CtcLabelDecode ctc_decode_;             // turns the model's output into text
};
