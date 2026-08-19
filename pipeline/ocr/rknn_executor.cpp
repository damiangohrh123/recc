#include "rknn_executor.h"

#include "third_party/rknn_api.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

// Destructor: release() is idempotent, so this is safe even if never loaded.
RknnExecutor::~RknnExecutor() {
	release();
}

namespace {
	// Logs and returns false on failure; keeps each run() error check to one line.
	bool check_ret(int ret, const char* what) {
		if (ret != RKNN_SUCC) {
			fprintf(stderr, "%s failed (ret=%d)\n", what, ret);
			return false;
		}
		return true;
	}

	// One core mask per NPU core slot on RK3588; index i pins slot i to core i.
	constexpr rknn_core_mask kCoreMasks[3] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2};
}

bool RknnExecutor::load(const std::string& model_path, int num_cores) {
	rknn_context ctx = 0;
	// size=0 means model_path is a file path, not an in-memory buffer.
	int ret = rknn_init(&ctx, const_cast<char*>(model_path.c_str()), 0, 0, nullptr);
	if (ret != RKNN_SUCC) {
		fprintf(stderr, "Load model failed: %s (ret=%d)\n", model_path.c_str(), ret);
		return false;
	}
	ctxs_.assign(1, static_cast<uint64_t>(ctx));
	loaded_ = true;

	// Query and cache the output shapes once (see output_shapes_ in the header).
	rknn_input_output_num io_num;
	memset(&io_num, 0, sizeof(io_num));
	if (!check_ret(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)), "rknn_query(IN_OUT_NUM)")) {
		release();
		return false;
	}

	// Look up each output's shape by index and store it for later use in run().
	output_shapes_.assign(io_num.n_output, {});
	for (uint32_t i = 0; i < io_num.n_output; ++i) {
		rknn_tensor_attr attr;
		memset(&attr, 0, sizeof(attr));
		attr.index = i;
		int qret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
		if (qret == RKNN_SUCC) {
			output_shapes_[i].assign(attr.dims, attr.dims + attr.n_dims);
		}
		else {
			fprintf(stderr, "rknn_query(OUTPUT_ATTR, index=%u) failed (ret=%d) -- shape left empty\n", i, qret);
		}
	}

	// Duplicates the context once per extra core requested, pinning each
	// duplicate (and the original) to its own core so run() calls made with
	// different core_slot values can execute concurrently.
	int n = std::max(1, std::min(num_cores, 3));
	if (n > 1) {
		check_ret(rknn_set_core_mask(ctx, kCoreMasks[0]), "rknn_set_core_mask(slot 0)");
		for (int slot = 1; slot < n; ++slot) {
			rknn_context src = ctx;
			rknn_context dup_ctx = 0;
			if (!check_ret(rknn_dup_context(&src, &dup_ctx), "rknn_dup_context")) continue;
			if (!check_ret(rknn_set_core_mask(dup_ctx, kCoreMasks[slot]), "rknn_set_core_mask")) {
				rknn_destroy(dup_ctx);
				continue;
			}
			ctxs_.push_back(static_cast<uint64_t>(dup_ctx));
		}
	}
	return true;
}

std::vector<RknnOutput> RknnExecutor::run(const std::vector<float>& input_nhwc, int core_slot) {
	if (!loaded_) {
		fprintf(stderr, "ERROR: rknn has been released\n");
		return {};
	}
	if (core_slot < 0 || core_slot >= static_cast<int>(ctxs_.size())) core_slot = 0;
	rknn_context ctx = static_cast<rknn_context>(ctxs_[core_slot]);

	// Describe the single input tensor and hand it to the runtime.
	rknn_input in;
	memset(&in, 0, sizeof(in));
	in.index = 0;
	in.buf = const_cast<float*>(input_nhwc.data());
	in.size = static_cast<uint32_t>(input_nhwc.size() * sizeof(float));
	in.type = RKNN_TENSOR_FLOAT32;
	in.fmt = RKNN_TENSOR_NHWC;
	in.pass_through = 0;

	if (!check_ret(rknn_inputs_set(ctx, 1, &in), "rknn_inputs_set")) return {};
	if (!check_ret(rknn_run(ctx, nullptr), "rknn_run")) return {};

	// Output count == output_shapes_.size(), cached from load().
	const uint32_t n_output = static_cast<uint32_t>(output_shapes_.size());
	std::vector<rknn_output> outputs(n_output);
	memset(outputs.data(), 0, sizeof(rknn_output) * n_output);
	for (uint32_t i = 0; i < n_output; ++i) {
		outputs[i].index = i;
		outputs[i].want_float = 1;  // dequantize to float32
		outputs[i].is_prealloc = 0;
	}

	if (!check_ret(rknn_outputs_get(ctx, n_output, outputs.data(), nullptr), "rknn_outputs_get")) return {};

	// Copy each output's data out of the RKNN-owned buffer before releasing it.
	std::vector<RknnOutput> results;
	results.reserve(n_output);
	for (uint32_t i = 0; i < n_output; ++i) {
		RknnOutput out;
		const float* data = reinterpret_cast<const float*>(outputs[i].buf);
		size_t n_floats = outputs[i].size / sizeof(float);
		out.data.assign(data, data + n_floats);
		out.shape = output_shapes_[i];  // cached from load()
		results.push_back(std::move(out));
	}

	rknn_outputs_release(ctx, n_output, outputs.data());
	return results;
}

// Destroys every RKNN context and resets state; no-op if not loaded.
void RknnExecutor::release() {
	if (loaded_) {
		for (uint64_t c : ctxs_) rknn_destroy(static_cast<rknn_context>(c));
		ctxs_.clear();
		loaded_ = false;
		output_shapes_.clear();
	}
}

// Combines construction + load() so callers get a ready-to-use object or nullptr.
std::unique_ptr<RknnExecutor> load_model(const std::string& model_path, int num_cores) {
	auto exec = std::make_unique<RknnExecutor>();
	if (!exec->load(model_path, num_cores)) {
		return nullptr;
	}
	return exec;
}
