#include "ppocr_rec.h"
#include <cstdio>
#include <fstream>
#include <thread>
#include <opencv2/imgproc.hpp>

CtcLabelDecode::CtcLabelDecode(const std::string& character_dict_path, bool use_space_char) {
	// The blank token always comes first, before any dictionary entries.
	character_.push_back("blank");

	std::ifstream fin(character_dict_path, std::ios::binary);
	if (!fin) {
		fprintf(stderr, "CtcLabelDecode: failed to open character dict: %s\n", character_dict_path.c_str());
	}
	else {
		std::string line;
		while (std::getline(fin, line)) {
			// Strips a trailing \r left behind on CRLF-terminated files.
			if (!line.empty() && line.back() == '\r') line.pop_back();
			character_.push_back(std::move(line));
		}
	}
	if (use_space_char) character_.push_back(" ");  // adds a space character last
}

RecResult CtcLabelDecode::decode(const float* preds, int seq_len, int num_classes) const {
	std::string text;
	float score_sum = 0.0f;
	int kept_count = 0;
	int prev_idx = -1;  // no previous timestep yet

	for (int t = 0; t < seq_len; ++t) {
		// Finds the highest-scoring class for this timestep.
		const float* row = preds + static_cast<size_t>(t) * num_classes;
		int best_idx = 0;
		float best_val = row[0];
		for (int c = 1; c < num_classes; ++c) {
			if (row[c] > best_val) {
				best_val = row[c];
				best_idx = c;
			}
		}

		// Skips repeats of the previous timestep's class, and blank tokens.
		bool is_duplicate = (t > 0 && best_idx == prev_idx);
		prev_idx = best_idx;
		if (is_duplicate) continue;
		if (best_idx == 0) continue;
		if (best_idx < 0 || best_idx >= static_cast<int>(character_.size())) continue;

		// Appends the surviving character and tracks its score.
		text += character_[best_idx];
		score_sum += best_val;
		++kept_count;
	}

	float mean_score = kept_count > 0 ? score_sum / static_cast<float>(kept_count) : 0.0f;
	return { text, mean_score };
}

TextRecognizer::TextRecognizer(const std::string& rec_model_path,
	const std::string& character_dict_path)
	: model_(load_model(rec_model_path, kNumCores)),
	// Scales pixel values into [0,1]; no mean/std shift is applied.
	normalize_(1.0 / 255.0, { 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 }),
	ctc_decode_(character_dict_path, /*use_space_char=*/true) {
	if (!model_) {
		fprintf(stderr, "TextRecognizer: failed to load model %s\n", rec_model_path.c_str());
	}
}

std::vector<RecResult> TextRecognizer::run(const std::vector<cv::Mat>& imgs) const {
	if (!is_loaded()) return {};
	std::vector<RecResult> results(imgs.size());

	// Resizes, normalizes, and recognizes one crop, writing into its slot.
	auto process_one = [&](size_t idx, int core_slot) {
		cv::Mat resized;
		cv::resize(imgs[idx], resized, cv::Size(kRecW, kRecH));
		cv::Mat normalized = normalize_(resized);
		std::vector<float> batch = to_nhwc_batch(normalized);

		auto outputs = model_->run(batch, core_slot);
		if (outputs.empty()) {
			results[idx] = { "", 0.0f };
			return;
		}

		// Checks that the output is one row of per-timestep class scores.
		const RknnOutput& logits = outputs[0];
		if (logits.shape.size() != 3 || logits.shape[0] != 1) {
			fprintf(stderr, "TextRecognizer: unexpected rec output shape\n");
			results[idx] = { "", 0.0f };
			return;
		}
		int seq_len = logits.shape[1];
		int num_classes = logits.shape[2];
		results[idx] = ctc_decode_.decode(logits.data.data(), seq_len, num_classes);
	};

	// Below 2 crops, or if only one core loaded, dispatch stays sequential.
	int n_cores = model_->num_cores();
	if (n_cores <= 1 || imgs.size() <= 1) {
		for (size_t i = 0; i < imgs.size(); ++i) process_one(i, 0);
		return results;
	}

	// Splits crops round-robin across core-pinned contexts, each on its own thread.
	std::vector<std::thread> workers;
	workers.reserve(n_cores);
	for (int core = 0; core < n_cores; ++core) {
		workers.emplace_back([&, core]() {
			for (size_t i = core; i < imgs.size(); i += n_cores) process_one(i, core);
		});
	}
	for (auto& worker : workers) worker.join();
	return results;
}
