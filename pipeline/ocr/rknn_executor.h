#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Thin C++ wrapper around librknnrt's C API (third_party/rknn_api.h).
// C++ port of the Python reference implementation's rknn_executor.py.

// One model output: dequantized float data plus its tensor shape (the C API
// doesn't report shape the way a numpy array would).
struct RknnOutput {
    std::vector<float> data;
    std::vector<int> shape;  // e.g. {1, 1, 480, 480} for det's output map
};

class RknnExecutor {
public:
    RknnExecutor() = default;  // Starts unloaded; call load() before use.
    ~RknnExecutor();           // Releases the model if still loaded.

    // Non-copyable and non-movable: this class owns raw rknn contexts, and is
    // only ever held through std::unique_ptr (see ppocr_det.h / ppocr_rec.h),
    // so the pointer moves, never the executor itself.
    RknnExecutor(const RknnExecutor&) = delete;
    RknnExecutor& operator=(const RknnExecutor&) = delete;

    // Loads the model and initializes the runtime. Returns false on failure
    // instead of exiting the process. Inference always runs on this board's
    // own NPU.
    //
    // num_cores duplicates the loaded context that many times (capped at 3,
    // the number of NPU cores on RK3588) and pins each duplicate to its own
    // core, so concurrent run() calls on different core_slot values execute
    // on different physical cores. num_cores=1 (the default) keeps the old
    // single-context, single-core behavior.
    bool load(const std::string& model_path, int num_cores = 1);

    // Runs inference on one input tensor, using the context pinned to
    // core_slot (0-based; see load()'s num_cores). Returns one RknnOutput
    // per model output, always dequantized to float32.
    std::vector<RknnOutput> run(const std::vector<float>& input_nhwc, int core_slot = 0);

    bool is_loaded() const { return loaded_; }  // True once load() has succeeded.

    // Number of core-pinned contexts available (see load()'s num_cores).
    int num_cores() const { return static_cast<int>(ctxs_.size()); }

private:
    void release();  // Frees all contexts; safe to call even if not loaded.

    std::vector<uint64_t> ctxs_;  // One RKNN context handle per core slot; empty when unloaded.
    bool loaded_ = false;         // Whether ctxs_ currently owns loaded model(s).

    // Per-output shape, cached once in load() since it's fixed for the life
    // of the model -- avoids re-querying it on every run() call.
    std::vector<std::vector<int>> output_shapes_;
};

// Constructs and loads an RknnExecutor in one step. Returns nullptr on failure.
std::unique_ptr<RknnExecutor> load_model(const std::string& model_path, int num_cores = 1);
